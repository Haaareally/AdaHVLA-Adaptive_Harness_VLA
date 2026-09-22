#!/usr/bin/env python3
"""Evaluate the prototype, adapt harness source, then test the frozen selection."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from adahvla.adaptation import Adaptation, AdaptationPolicy
from adahvla.harness import JSONReasoner
from adahvla.locomotion import load_config
from adahvla.workspace import Case, CommandEvaluator, Workspace, atomic_json


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be positive and finite")
    return number


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Offline asset, split and prototype checks; no API or simulator")
    parser.add_argument("--prototype-only", action="store_true", help="Evaluate the prototype on the first training episode, then stop")
    parser.add_argument("--output-dir", type=Path, default=PROJECT.parent / "AdaHVLA-runs/navila-LH")
    parser.add_argument("--train-ids", default="27", help="Comma-separated episode_id values, not dataset row indices")
    parser.add_argument("--validation-ids", default="124")
    parser.add_argument("--test-ids", default="167", help="Use an empty string to omit held-out testing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=os.environ.get("ADAHVLA_MODEL", ""))
    parser.add_argument("--base-url", default=os.environ.get("ADAHVLA_BASE_URL", ""))
    parser.add_argument("--vla-host", default=os.environ.get("ADAHVLA_VLA_HOST", "127.0.0.1"))
    parser.add_argument("--vla-port", type=positive_int, default=int(os.environ.get("ADAHVLA_VLA_PORT", "54321")))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=positive_int, default=16)
    parser.add_argument("--max-candidates", type=positive_int, default=4)
    parser.add_argument("--max-rollouts", type=positive_int, default=12)
    parser.add_argument("--width", type=positive_int, default=2)
    parser.add_argument("--depth", type=positive_int, default=2)
    parser.add_argument("--max-decisions", type=positive_int, default=128)
    parser.add_argument("--max-episode-seconds", type=positive_float, default=180.0)
    parser.add_argument("--rollout-timeout", type=positive_float, default=7200.0, help="Wall-clock seconds per simulator process")
    parser.add_argument("--reasoner-timeout", type=positive_float, default=180.0)
    parser.add_argument("--vla-timeout", type=positive_float, default=300.0)
    return parser.parse_args(argv)


def build_cases(args, settings):
    with gzip.open(settings["paths"]["dataset"], "rt") as stream:
        episodes = json.load(stream)["episodes"]
    by_id = {str(episode["episode_id"]): episode for episode in episodes}
    if len(by_id) != len(episodes):
        raise ValueError("Dataset episode_id values must be unique")
    cases, seen = [], set()
    for split, supplied in (("train", args.train_ids), ("validation", args.validation_ids), ("test", args.test_ids)):
        identifiers = [item.strip() for item in supplied.split(",") if item.strip()]
        if split == "train" and not identifiers:
            raise ValueError("At least one training episode is required")
        for identifier in identifiers:
            if identifier in seen or identifier not in by_id:
                raise ValueError(f"Duplicate, overlapping or unknown episode_id: {identifier}")
            seen.add(identifier)
            episode = by_id[identifier]
            # Route labels stay out of adaptation context and online control.
            selected = {key: episode[key] for key in (
                "episode_id", "scene_id", "start_position", "start_rotation", "goals",
            )}
            selected["instruction"] = {"instruction_text": episode["instruction"]["instruction_text"]}
            scene = Path(episode["scene_id"]).stem
            if not (settings["paths"]["scenes"] / scene / f"{scene}.usd").is_file():
                raise ValueError(f"Missing scene: {scene}")
            cases.append(Case(
                id=f"episode-{identifier}", task=selected["instruction"]["instruction_text"],
                environment=scene, split=split, seed=args.seed,
                metadata={"episode": selected, "locomotion_config": str(PROJECT / "configs/locomotion.json")},
            ))
    return cases


class Go2Evaluator(CommandEvaluator):
    """Freeze the host environment implementation as well as the candidate source."""

    def __init__(self, command, timeout, dependencies):
        self.dependencies = tuple(Path(path).resolve() for path in dependencies)
        super().__init__(command, timeout)

    def _current_fingerprint(self):
        fingerprint = super()._current_fingerprint()
        fingerprint["environment_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in self.dependencies
        }
        return fingerprint


def make_evaluator(args, settings):
    command = [sys.executable, str(PROJECT / "scripts/evaluate.py")]
    for name, value in (
        ("reasoner-base-url", args.base_url), ("reasoner-model", args.model),
        ("reasoner-timeout", args.reasoner_timeout), ("vla-host", args.vla_host),
        ("vla-port", args.vla_port), ("vla-timeout", args.vla_timeout),
        ("device", args.device), ("max-episode-seconds", args.max_episode_seconds),
        ("max-decisions", args.max_decisions),
    ):
        command.extend(("--" + name, str(value)))
    dependencies = [
        PROJECT / "src/adahvla/locomotion.py", PROJECT / "configs/go2_vision.py",
        PROJECT / "configs/locomotion.json", PROJECT / "assets/manifest.json",
        PROJECT / "scripts/run.py", settings["paths"]["policy"], settings["paths"]["dataset"],
    ]
    return Go2Evaluator(command, args.rollout_timeout, dependencies)


def run_session(session, prototype_only=False):
    # Guarantee a measured prototype before the manager can propose a revision.
    first_train = next(case for case in session.cases.values() if case.split == "train")
    measured = any(row["candidate_id"] == "C0000" and row["case_id"] == first_train.id
                   for row in session.state["rollouts"].values())
    if session.state["phase"] == "adapt" and not measured:
        if not session.workspace.run_checks(session.candidate("C0000"))["passed"]:
            raise RuntimeError("Prototype source checks failed; see the session checks directory")
        print(f"Evaluating prototype C0000 on {first_train.id}", flush=True)
        session.evaluate("C0000", first_train.id)
    if prototype_only:
        return session.selected

    while session.state["phase"] != "complete":
        result = session.step()
        print(json.dumps({"phase": session.state["phase"], "step": session.state["steps"],
                          "action": result.get("tool", result.get("reason", "complete")),
                          "selected": session.selected.id}), flush=True)
    if session.state["frozen"] is None and any(case.split == "test" for case in session.cases.values()):
        session.start_test()
        while session.state["phase"] != "complete":
            result = session.step()
            print(json.dumps({"phase": session.state["phase"], "action": result.get("tool", "complete"),
                              "selected": session.selected.id}), flush=True)
    return session.selected


def main(argv=None):
    args = arguments(argv)
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in 0..2**32-1")
    # The checked-in entry point always resolves host resources from its own checkout.
    os.environ["ADAHVLA_ROOT"] = str(PROJECT)
    settings = load_config()
    cases = build_cases(args, settings)
    subprocess.run([sys.executable, str(PROJECT / "scripts/check_assets.py"), "--root", str(PROJECT)], check=True)
    policy = AdaptationPolicy(max_steps=args.max_steps, max_candidates=args.max_candidates,
                              max_rollouts=args.max_rollouts, width=args.width, depth=args.depth)
    if args.check:
        with tempfile.TemporaryDirectory(prefix="adahvla-check-") as temporary:
            workspace = Workspace(Path(temporary), PROJECT / "src/adahvla")
            report = workspace.run_checks(workspace.baseline())
            if not report["passed"]:
                raise RuntimeError("Prototype source checks failed")
        print(json.dumps({"status": "offline checks passed", "cases": {split: [case.id for case in cases if case.split == split]
                         for split in ("train", "validation", "test")}, "policy": asdict(policy)}, indent=2))
        return

    api_key = os.environ.get("ADAHVLA_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_API_KEY":
        raise ValueError("Fill ADAHVLA_API_KEY in scripts/run.sh or export it in this terminal")
    url = urlsplit(args.base_url)
    if (url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password
            or url.query or url.fragment or "YOUR_" in args.base_url):
        raise ValueError("Set a valid reasoning API base URL without credentials or query parameters")
    if not args.model or args.model == "YOUR_VISION_MODEL":
        raise ValueError("Set ADAHVLA_MODEL to a vision model supporting the JSON chat interface")
    if args.vla_port > 65535:
        raise ValueError("VLA port must be in 1..65535")
    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(PROJECT):
        raise ValueError("Use an output directory outside AdaHVLA to keep the source bundle clean")
    evaluator = make_evaluator(args, settings)
    workspace = Workspace(output, PROJECT / "src/adahvla")
    reasoners = {role: JSONReasoner(base_url=args.base_url, model=args.model, api_key=api_key,
                                  timeout=args.reasoner_timeout)
                 for role in ("manager", "analyst", "engineer", "reviewer")}
    session = Adaptation(workspace, cases, evaluator, policy=policy, **reasoners)
    selected = run_session(session, args.prototype_only)
    atomic_json(output / "selected.json", asdict(selected))
    print(f"Selected {selected.id}: {selected.source_path}", flush=True)
    print(f"Session: {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; rerun with the same arguments to resume the session.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        secret = os.environ.get("ADAHVLA_API_KEY", "")
        message = str(exc).replace(secret, "[redacted]") if secret else str(exc)
        print(f"Run failed: {type(exc).__name__}: {message}", file=sys.stderr)
        raise SystemExit(1)
