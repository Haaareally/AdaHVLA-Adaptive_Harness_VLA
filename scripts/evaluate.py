#!/usr/bin/env python3
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Execute one sealed harness candidate with Go2 and an external NaVILA server.

CommandEvaluator supplies --request and --output. No simulator, model, or API is
initialized when this module is imported or when --help is requested.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("__init__.py", "harness.py", "vla.py")
DEFAULTS = {
    "vla_host": "127.0.0.1", "vla_port": 54321, "vla_frames": 8,
    "device": "cuda:0", "max_episode_seconds": 180.0, "max_decisions": 128,
    "frame_interval": 0.5, "evidence_interval": 2.0,
    "reasoner_timeout": 180.0, "vla_timeout": 300.0,
    "headless": True,
}


def encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode()


def verify_candidate(candidate: dict) -> Path:
    """Verify exactly the same three-file source bundle sealed by Workspace."""
    directory = Path(candidate["path"])
    package = directory / "adahvla"
    if directory.is_symlink() or package.is_symlink() or not package.is_dir():
        raise ValueError("Candidate must be a regular directory")
    if {path.name for path in package.iterdir()} != set(SOURCE_FILES):
        raise ValueError("Candidate must contain exactly the sealed runtime source files")
    files = {}
    for name in SOURCE_FILES:
        path = package / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Invalid candidate source: {name}")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if files != candidate["files"] or hashlib.sha256(encoded(files)).hexdigest() != candidate["digest"]:
        raise ValueError("Candidate source digest does not match its manifest")
    manifest = directory / "candidate.json"
    if manifest.is_symlink() or not manifest.is_file() or json.loads(manifest.read_text()) != candidate:
        raise ValueError("Candidate identity does not match candidate.json")
    return directory.resolve()


@contextmanager
def candidate_modules(candidate: dict):
    """Import the candidate rather than an installed or previously cached package."""
    directory = verify_candidate(candidate)
    saved = {name: module for name, module in sys.modules.items() if name == "adahvla" or name.startswith("adahvla.")}
    old_path, old_bytecode = sys.path[:], sys.dont_write_bytecode
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, str(directory))
    sys.dont_write_bytecode = True
    try:
        harness = importlib.import_module("adahvla.harness")
        vla = importlib.import_module("adahvla.vla")
        for module, name in ((sys.modules["adahvla"], "__init__.py"), (harness, "harness.py"), (vla, "vla.py")):
            if Path(module.__file__).resolve() != directory / "adahvla" / name:
                raise ValueError("Evaluator imported a module outside the candidate bundle")
        yield harness, vla
    finally:
        try:
            verify_candidate(candidate)
        finally:
            for name in list(sys.modules):
                if name == "adahvla" or name.startswith("adahvla."):
                    del sys.modules[name]
            sys.modules.update(saved)
            sys.path[:] = old_path
            sys.dont_write_bytecode = old_bytecode


def fixed_locomotion():
    """Expose the fixed simulator adapter after loading the sealed candidate."""
    os.environ["ADAHVLA_ROOT"] = str(PROJECT_ROOT)
    path = PROJECT_ROOT / "src/adahvla/locomotion.py"
    spec = importlib.util.spec_from_file_location("_adahvla_evaluator_locomotion", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load fixed locomotion adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # The candidate directory stays sealed; the fixed scene config imports this alias.
    sys.modules["adahvla.locomotion"] = module
    return module


_DISTANCE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(cm|centimeter|centimeters|m|meter|meters|foot|feet|ft)\b", re.I)
_DEGREES = re.compile(r"\b(\d+(?:\.\d+)?)\s*(degree|degrees|deg)\b", re.I)
_LEFT = re.compile(r"\b(?:turn|rotate|pivot|veer|bear|face|swing)\b(?:\s+\w+){0,3}\s+\b(?:left|counterclockwise|anti[-\s]?clockwise|anticlockwise)\b", re.I)
_RIGHT = re.compile(r"\b(?:turn|rotate|pivot|veer|bear|face|swing)\b(?:\s+\w+){0,3}\s+\b(?:right|clockwise)\b", re.I)
_FORWARD = re.compile(r"\b(?:move|go|walk|head|proceed|continue|advance|step)\b(?:\s+\w+){0,3}\s+\b(?:forward|straight|ahead)\b|\bstraight ahead\b", re.I)
_STOP = re.compile(
    r"\b(?:stop|halt|wait here|stay here|hold position|hold here|"
    r"finished the instruction|finished this instruction|i have finished|"
    r"task is complete|we are done|destination reached)\b", re.I,
)
_NEGATED_STOP_PREFIX = re.compile(r"\b(?:not|never|without|don['’]t)\s+$", re.I)


def parse_native_action(text: str) -> dict:
    """Map NaVILA output to the original NaVILA-Bench velocity/time interface."""
    lowered = str(text).strip().lower()
    command = {"kind": "hold", "velocity": [0.0, 0.0, 0.0], "seconds": 0.5, "recognized": False}
    if any(not _NEGATED_STOP_PREFIX.search(lowered[:match.start()]) for match in _STOP.finditer(lowered)):
        return dict(command, kind="stop", seconds=0.0, recognized=True)
    left = bool(_LEFT.search(lowered)) or any(term in lowered for term in ("turn left", "rotate left", "face left", "veer left", "bear left", "pivot left"))
    right = bool(_RIGHT.search(lowered)) or any(term in lowered for term in ("turn right", "rotate right", "face right", "veer right", "bear right", "pivot right"))
    if left or right:
        match = _DEGREES.search(lowered)
        degrees = float(match.group(1)) if match else (
            15.0 if any(term in lowered for term in ("slight", "slightly", "a little", "small turn"))
            else 45.0 if any(term in lowered for term in ("sharp", "hard turn")) else 15.0
        )
        degrees = max(5.0, degrees or 15.0)
        if not math.isfinite(degrees):
            raise ValueError("NaVILA returned a non-finite turn amount")
        return dict(command, kind="turn", velocity=[0.0, 0.0, (1.0 if left else -1.0) * math.pi / 6],
                    seconds=max(0.35, degrees / 30.0), recognized=True)
    forward_terms = ("move forward", "go forward", "walk forward", "go straight", "walk straight", "move straight",
                     "advance", "move ahead", "go ahead", "head forward", "proceed forward", "continue forward",
                     "continue straight", "step forward")
    if _FORWARD.search(lowered) or any(term in lowered for term in forward_terms):
        match = _DISTANCE.search(lowered)
        distance_cm = 25.0
        if match:
            amount, unit = float(match.group(1)), match.group(2).lower()
            distance_cm = amount * (100.0 if unit in {"m", "meter", "meters"} else 30.48 if unit in {"foot", "feet", "ft"} else 1.0)
        distance_cm = max(10.0, distance_cm or 25.0)
        if not math.isfinite(distance_cm):
            raise ValueError("NaVILA returned a non-finite distance")
        return dict(command, kind="move", velocity=[0.5, 0.0, 0.0], seconds=max(0.25, distance_cm / 50.0), recognized=True)
    command["recognized"] = any(term in lowered for term in ("hold", "pause", "wait", "stay put"))
    return command


def success_at_stop(position: list[float], goal: list[float], radius: float, stopped: bool) -> tuple[bool, float]:
    """Euclidean 3D proxy, not the benchmark's route-distance or geodesic metric."""
    if len(position) != 3 or len(goal) != 3 or not all(math.isfinite(float(x)) for x in [*position, *goal, radius]) or radius <= 0:
        raise ValueError("Success measurement requires finite 3D positions and a positive radius")
    distance = math.dist(position, goal)
    return bool(stopped and distance < radius), distance


class RecordedVLA:
    def __init__(self, executor):
        self.executor, self.calls = executor, 0
        self.action, self.query, self.frame_times = None, "", []

    def predict(self, query, frames):
        self.action = self.executor.predict(query, frames)
        self.query, self.frame_times = query, [frame.timestamp for frame in frames]
        self.calls += 1
        return self.action

    def verify_handoff(self, execution, previous_calls):
        if execution.handoff.completed:
            if self.calls != previous_calls or execution.action is not None:
                raise ValueError("Completed harness handoff must not invoke the VLA or contain a physical action")
        elif self.calls != previous_calls + 1 or execution.action != self.action:
            raise ValueError("Harness must return exactly the native action from one VLA call")


def runtime_options(args: argparse.Namespace, case: dict) -> dict:
    runtime = case.get("metadata", {}).get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("case.metadata.runtime must be an object")
    options = dict(DEFAULTS)
    for key in (*DEFAULTS, "reasoner_base_url", "reasoner_model"):
        if key in runtime:
            options[key] = runtime[key]
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    if not options.get("reasoner_base_url") or not options.get("reasoner_model"):
        raise ValueError("--reasoner-base-url and --reasoner-model are required")
    for key in ("max_episode_seconds", "frame_interval", "evidence_interval", "reasoner_timeout", "vla_timeout"):
        if type(options[key]) not in {int, float} or not math.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key, minimum in (("vla_port", 1), ("vla_frames", 2), ("max_decisions", 1)):
        if type(options[key]) is not int or options[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if type(options["headless"]) is not bool:
        raise ValueError("headless must be a boolean")
    if not isinstance(options["device"], str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", options["device"]):
        raise ValueError("device must be cpu, cuda, or cuda:N")
    return options


def validate_episode(case: dict) -> dict:
    episode = case.get("metadata", {}).get("episode")
    if not isinstance(episode, dict):
        raise ValueError("case.metadata.episode must contain the complete navigation episode")
    if case["task"].strip() != episode["instruction"]["instruction_text"].strip():
        raise ValueError("Case task differs from the episode instruction")
    if case["environment"] != Path(episode["scene_id"]).stem:
        raise ValueError("Case environment differs from the episode scene")
    if type(case["seed"]) is not int:
        raise ValueError("Case seed must be an integer")
    start, quat = episode["start_position"], episode["start_rotation"]
    if len(start) != 3 or len(quat) != 4 or not all(math.isfinite(float(x)) for x in [*start, *quat]):
        raise ValueError("Episode requires a finite 3D start position and wxyz quaternion")
    if abs(sum(float(x) ** 2 for x in quat) - 1.0) > 0.01:
        raise ValueError("Episode start rotation must be a normalized wxyz quaternion")
    goal = episode["goals"][0]
    success_at_stop(start, goal["position"], float(goal["radius"]), False)
    return episode


def run_episode(request: dict, output: Path, options: dict) -> dict:
    candidate, case = request["candidate"], request["case"]
    episode = validate_episode(case)
    api_key = os.environ.get("ADAHVLA_API_KEY", "")
    if not api_key:
        raise ValueError("Set ADAHVLA_API_KEY for the online harness reasoner")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output.parent / "frames"
    frame_dir.mkdir(exist_ok=True)
    events, images = [], []
    app, env = None, None
    start_wall = time.monotonic()
    with (output.parent / "events.jsonl").open("w", encoding="utf-8") as journal, candidate_modules(candidate) as (harness_module, vla_module):
        def record(kind: str, **details: Any) -> None:
            event = {"event": kind, **details}
            events.append(event)
            journal.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            journal.flush()

        locomotion = fixed_locomotion()
        try:
            # AppLauncher must precede simulator-dependent imports.
            if options["headless"]:
                # An inherited remote X display can fail with GLXBadFBConfig.
                os.environ.pop("DISPLAY", None)
            from omni.isaac.lab.app import AppLauncher

            device_id = int(options["device"].split(":")[1]) if ":" in options["device"] else 0
            launcher = AppLauncher(headless=options["headless"], enable_cameras=True, device_id=device_id, multi_gpu=False)
            app = launcher.app
            import numpy as np
            import torch
            from PIL import Image
            from omni.isaac.lab.envs import ManagerBasedRLEnv

            class SingleEpisodeEnv(ManagerBasedRLEnv):
                _suppress_autoreset = False

                def reset(self, **kwargs):
                    self._suppress_autoreset = False
                    try:
                        return super().reset(**kwargs)
                    finally:
                        self._suppress_autoreset = True

                def _reset_idx(self, env_ids):
                    if not self._suppress_autoreset:
                        return super()._reset_idx(env_ids)

            random.seed(case["seed"])
            np.random.seed(case["seed"])
            torch.manual_seed(case["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(case["seed"])
            config_path = case.get("metadata", {}).get("locomotion_config")
            settings = locomotion.load_config(config_path)
            cfg = locomotion.load_env_cfg(case["environment"], config_path)
            cfg.scene.num_envs = 1
            cfg.sim.device = options["device"]
            x, y, z = episode["start_position"]
            cfg.scene.robot.init_state.pos = (x, y, z + 0.4)
            cfg.scene.robot.init_state.rot = tuple(episode["start_rotation"])
            cfg.scene.disk_1.init_state.pos = (x, y, z + 2.5)
            gx, gy, gz = episode["goals"][0]["position"]
            cfg.scene.disk_2.init_state.pos = (gx, gy, gz + 2.5)
            raw_env = SingleEpisodeEnv(cfg=cfg)
            env = locomotion.Go2HistoryWrapper(raw_env, settings["history_length"])
            policy = locomotion.load_policy(settings["paths"]["policy"], options["device"])
            low_obs, extras = env.reset(seed=case["seed"])
            with torch.inference_mode():
                for _ in range(100):
                    env.update_command([0.0, 0.0, 0.0])
                    low_obs, _, done, extras = env.step(policy(low_obs))
                    if bool(done[0]):
                        raise RuntimeError("Go2 terminated during its 100-step startup warmup")
            dt = cfg.sim.dt * cfg.decimation
            max_steps = max(1, int(options["max_episode_seconds"] / dt))
            sample_steps = max(1, round(options["frame_interval"] / dt))
            reasoner = harness_module.JSONReasoner(
                base_url=options["reasoner_base_url"], model=options["reasoner_model"], api_key=api_key,
                timeout=options["reasoner_timeout"],
            )
            vla = RecordedVLA(vla_module.NaVILAClient(
                host=options["vla_host"], port=options["vla_port"], num_frames=options["vla_frames"],
                timeout=options["vla_timeout"],
            ))
            harness = harness_module.Harness(reasoner, capabilities="Navigate with forward motion and left/right turns. The VLA supplies all motion commands; no backward action is supported.")
            runtime = harness_module.HarnessVLA(harness, vla)
            runtime.reset(case["task"])
            step_count, decision_count, native_stops = 0, 0, 0
            feedback, stop_reason = "Episode initialized after locomotion warmup.", "decision_limit"
            stopped = False
            last_saved_time = -math.inf

            def robot_position():
                return raw_env.scene["robot"].data.root_pos_w[0].detach().cpu().tolist()

            position = robot_position()
            goal, radius = episode["goals"][0]["position"], float(episode["goals"][0]["radius"])
            path_length = 0.0
            oracle_distance = math.dist(position, goal)

            def capture(timestamp: float, force_save: bool = False):
                nonlocal last_saved_time
                save = force_save or timestamp - last_saved_time >= options["evidence_interval"]
                frame = None
                for view, camera in (("ego", "rgbd_camera"), ("viz", "viz_rgb_camera")):
                    if view == "viz" and not save:
                        continue
                    rgb = raw_env.scene.sensors[camera].data.output["rgb"][0, :, :, :3].detach().cpu().numpy()
                    buffer = BytesIO()
                    Image.fromarray(rgb.astype(np.uint8)).save(buffer, format="JPEG", quality=90)
                    data = buffer.getvalue()
                    if view == "ego":
                        frame = harness_module.Frame("data:image/jpeg;base64," + base64.b64encode(data).decode(), timestamp, "ego")
                    if save:
                        relative = f"frames/{len(images):06d}_{view}.jpg"
                        (output.parent / relative).write_bytes(data)
                        images.append({"path": relative, "timestamp": timestamp, "view": view})
                if save:
                    last_saved_time = timestamp
                return frame

            record("protocol", case_id=case["id"], candidate_id=candidate["id"], source_digest=candidate["digest"],
                   metric_definition="Stopped by harness and Euclidean 3D base-to-goal distance < radius; not official navigation SR/geodesic/SPL",
                   control_dt=dt, frame_interval=sample_steps * dt, success_radius_m=radius,
                   max_episode_seconds=options["max_episode_seconds"], max_decisions=options["max_decisions"],
                   reasoner_model=options["reasoner_model"], online_views=["ego"], evidence_views=["ego", "viz"])
            pending = [capture(0.0, force_save=True)]
            for decision_count in range(1, options["max_decisions"] + 1):
                if not app.is_running():
                    stop_reason = "simulator_closed"
                    break
                boundary_time = round(step_count * dt, 6)
                record("boundary", request_id=decision_count, sim_time_sec=boundary_time,
                       frame_times=[frame.timestamp for frame in pending], executor_feedback=feedback)
                previous_vla_calls = vla.calls
                execution = runtime.step(harness_module.Observation(tuple(pending), executor_feedback=feedback))
                vla.verify_handoff(execution, previous_vla_calls)
                pending = []
                handoff = asdict(execution.handoff)
                state = asdict(harness.state)
                if execution.handoff.completed:
                    stopped, stop_reason = True, "harness_completed"
                    record("completion", request_id=decision_count, sim_time_sec=boundary_time, handoff=handoff, harness_state=state)
                    break
                if not isinstance(execution.action, str) or not execution.action.strip():
                    raise ValueError("NaVILA must return a nonempty native action string")
                command = parse_native_action(execution.action)
                planned_steps = int(command["seconds"] / dt)
                record("decision", request_id=decision_count, sim_time_sec=boundary_time, handoff=handoff,
                       harness_state=state, native_action=execution.action, parsed_command=command,
                       vla_query=vla.query, vla_frame_times=vla.frame_times,
                       planned_control_steps=planned_steps)
                if command["kind"] == "stop":
                    native_stops += 1
                steps_before = step_count
                done_now = False
                with torch.inference_mode():
                    for _ in range(min(planned_steps, max_steps - step_count)):
                        env.update_command(command["velocity"])
                        low_obs, _, done, extras = env.step(policy(low_obs))
                        step_count += 1
                        next_position = robot_position()
                        path_length += math.dist(position, next_position)
                        position = next_position
                        oracle_distance = min(oracle_distance, math.dist(position, goal))
                        done_now = bool(done[0])
                        if step_count % sample_steps == 0 or done_now:
                            pending.append(capture(round(step_count * dt, 6)))
                        if done_now or not app.is_running():
                            break
                now = round(step_count * dt, 6)
                if not pending or pending[-1].timestamp != now:
                    pending.append(capture(now))
                executed_seconds = round((step_count - steps_before) * dt, 6)
                feedback = json.dumps({
                    "native_action": execution.action, "action_type": command["kind"],
                    "executed_seconds": executed_seconds,
                    "local_stop_requested": command["kind"] == "stop",
                }, ensure_ascii=False)
                record("execution", request_id=decision_count, sim_time_sec=now,
                       executed_seconds=executed_seconds, control_steps=step_count - steps_before,
                       robot_position=position, environment_done=done_now,
                       native_stop_feedback_only=command["kind"] == "stop")
                if done_now:
                    stop_reason = "environment_done"
                    break
                if step_count >= max_steps:
                    stop_reason = "simulation_time_limit"
                    break
            capture(round(step_count * dt, 6), force_save=True)
            success, distance = success_at_stop(position, goal, radius, stopped)
            metrics = {
                "euclidean_goal_distance_m": distance, "euclidean_oracle_distance_m": oracle_distance,
                "euclidean_path_length_m": path_length, "success_radius_m": radius,
                "sim_seconds": step_count * dt, "control_steps": step_count,
                "decision_count": decision_count, "stop_declared": int(stopped), "native_stop_count": native_stops,
                "wall_seconds": time.monotonic() - start_wall,
            }
            record("termination", sim_time_sec=round(step_count * dt, 6), reason=stop_reason, success=success,
                   metric_definition="Euclidean 3D stop proxy", robot_position=position, metrics=metrics)
            return {"source_digest": candidate["digest"], "success": success, "metrics": metrics, "events": events, "images": images}
        except BaseException as exc:
            record("error", error_type=type(exc).__name__, message=str(exc).replace(api_key, "[redacted]"))
            raise
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                if app is not None:
                    app.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("reasoner-base-url", "reasoner-model", "vla-host", "device"):
        parser.add_argument("--" + name)
    for name in ("vla-port", "vla-frames", "max-decisions"):
        parser.add_argument("--" + name, type=int)
    for name in ("max-episode-seconds", "frame-interval", "evidence-interval", "reasoner-timeout", "vla-timeout"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    options = runtime_options(args, request["case"])
    def terminate(_signum, _frame):
        raise KeyboardInterrupt("Evaluator terminated")

    signal.signal(signal.SIGTERM, terminate)
    result = run_episode(request, args.output.resolve(), options)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(encoded(result))
    temporary.replace(args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        secret = os.environ.get("ADAHVLA_API_KEY", "")
        message = str(exc).replace(secret, "[redacted]") if secret else str(exc)
        print(f"Evaluator failed: {type(exc).__name__}: {message}", file=sys.stderr)
        raise SystemExit(1)
