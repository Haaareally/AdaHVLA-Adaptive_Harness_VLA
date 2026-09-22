# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab).
# SPDX-License-Identifier: BSD-3-Clause
"""Go2 checkpoint inference and observations, extracted from NaVILA-Bench.

Importing this module does not import PyTorch or start Isaac Sim. The simulator
configuration is loaded explicitly after the caller creates its AppLauncher.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


PROPRIO_DIM = 45
HEIGHT_MAP_DIM = 459
POLICY_OBSERVATION_DIM = PROPRIO_DIM + HEIGHT_MAP_DIM
HISTORY_LENGTH = 9
ACTOR_INPUT_DIM = POLICY_OBSERVATION_DIM + HISTORY_LENGTH * PROPRIO_DIM
ACTION_DIM = 12


def project_root() -> Path:
    return Path(os.environ.get("ADAHVLA_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read locomotion settings; asset paths are relative to the project root."""
    root = project_root()
    config_path = Path(path).expanduser() if path is not None else root / "configs/locomotion.json"
    if not config_path.is_absolute():
        config_path = root / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    config["paths"] = {
        key: (Path(value).expanduser() if Path(value).expanduser().is_absolute() else root / value).resolve()
        for key, value in paths.items()
    }
    for key in ("dataset", "policy", "robot_usd", "scenes"):
        if key not in config["paths"]:
            raise ValueError(f"Missing locomotion asset path: {key}")
    if config["history_length"] != HISTORY_LENGTH:
        raise ValueError("The bundled Go2 policy requires nine proprioceptive history frames")
    if config["physics_dt"] <= 0 or config["decimation"] < 1 or config["action_scale"] <= 0:
        raise ValueError("physics_dt, decimation, and action_scale must be positive")
    return config


def load_policy(path: str | Path | None = None, device: str = "cpu") -> Any:
    """Load the frozen 909-input, 12-output locomotion actor without RSL-RL."""
    import torch

    policy_path = Path(path).expanduser() if path is not None else load_config()["paths"]["policy"]
    if not policy_path.is_absolute():
        policy_path = project_root() / policy_path
    policy = torch.jit.load(str(policy_path), map_location=device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def load_env_cfg(scene_id: str, config_path: str | Path | None = None) -> Any:
    """Build the Go2 scene configuration after Isaac Sim has been launched.

    The caller sets the episode's robot start pose and the two light positions
    before constructing ManagerBasedRLEnv. This helper does not reset an episode.
    """
    settings = load_config(config_path)
    source = project_root() / "configs/go2_vision.py"
    module_name = "_adahvla_go2_vision_config"
    module = sys.modules.get(module_name)
    if module is None or Path(module.__file__).resolve() != source.resolve():
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Go2 environment configuration: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    scene_name = Path(scene_id).stem
    if not scene_name or scene_name in {".", ".."}:
        raise ValueError("A Matterport scene id is required")
    cfg = module.Go2MatterportVisionCfg()
    cfg.scene.robot.spawn.usd_path = str(settings["paths"]["robot_usd"])
    cfg.scene.terrain.obj_filepath = str(settings["paths"]["scenes"] / scene_name / f"{scene_name}.usd")
    cfg.sim.dt = float(settings["physics_dt"])
    cfg.decimation = int(settings["decimation"])
    cfg.sim.render_interval = cfg.decimation
    cfg.actions.joint_pos.scale = float(settings["action_scale"])
    cfg.scene.height_scanner.update_period = cfg.decimation * cfg.sim.dt
    cfg.scene.lidar_sensor.update_period = cfg.decimation * cfg.sim.dt
    cfg.scene.contact_forces.update_period = cfg.sim.dt
    return cfg


def base_rpy(env: Any, asset_cfg: Any = None) -> Any:
    import torch

    quat = env.scene[asset_cfg.name if asset_cfg is not None else "robot"].data.root_quat_w
    qw, qx, qy, qz = quat.unbind(dim=1)
    roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = 2 * (qw * qy - qz * qx)
    pitch = torch.where(torch.abs(sinp) < 1, torch.asin(sinp), torch.sign(sinp) * (torch.pi / 2))
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return torch.stack((roll, pitch, yaw), dim=1)


def height_map_lidar(env: Any, sensor_cfg: Any, offset: float = 0.5) -> Any:
    """The checkpoint's 17 x 27 minimum-height map followed by 3 x 3 max pooling."""
    import torch
    import torch.nn.functional as functional
    import omni.isaac.lab.utils.math as math_utils

    sensor = env.scene.sensors[sensor_cfg.name]
    hit_vec = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    hit_vec[torch.isinf(hit_vec) | torch.isnan(hit_vec)] = 0.0
    shape = hit_vec.shape
    robot_quat = env.scene["robot"].data.root_quat_w
    sensor_quat = torch.tensor([-0.131, 0.0, -0.991, 0.0], device=robot_quat.device).unsqueeze(0).repeat(shape[0], 1)
    sensor_quat = math_utils.quat_mul(robot_quat, sensor_quat)
    quaternions = sensor_quat.unsqueeze(1).repeat(1, shape[1], 1).view(-1, 4)
    points = math_utils.quat_rotate_inverse(quaternions, hit_vec.view(-1, 3)).view(shape)
    x_bins = torch.arange(-0.8, 0.2 + 1e-9, 0.06, device=points.device)
    y_bins = torch.arange(-0.8, 0.8 + 1e-9, 0.06, device=points.device)
    x, y, z = points.unbind(dim=-1)
    valid = (x > -0.8) & (x <= 0.2 + 1e-9) & (y > -0.8) & (y <= 0.8 + 1e-9) & (z >= 0.0) & (z <= 5.0)
    x_indices = torch.bucketize(x[valid], x_bins) - 1
    y_indices = torch.bucketize(y[valid], y_bins) - 1
    env_indices = torch.arange(shape[0], device=points.device).unsqueeze(1).expand_as(valid)[valid]
    indices = env_indices * len(x_bins) * len(y_bins) + x_indices * len(y_bins) + y_indices
    height_map = torch.full((shape[0], len(x_bins), len(y_bins)), float("inf"), device=points.device)
    height_map = height_map.view(-1).scatter_reduce_(0, indices, z[valid], reduce="amin") - offset
    height_map = torch.where(height_map < 0.05, 0.0, height_map)
    height_map = torch.where(torch.isinf(height_map), 0.0, height_map)
    height_map = height_map.view(shape[0], len(x_bins), len(y_bins))
    return functional.max_pool2d(height_map, kernel_size=3, stride=1, padding=1).view(shape[0], -1)


def isaac_camera_data(env: Any, sensor_cfg: Any, data_type: str) -> Any:
    import torch

    output = env.scene.sensors[sensor_cfg.name].data.output[data_type].clone()
    if data_type == "distance_to_image_plane":
        output = output.unsqueeze(1)
        output[torch.isnan(output) | torch.isinf(output)] = 0.0
    return output


def process_depth_image(env: Any, sensor_cfg: Any, data_type: str, far_clip: float = 5.0) -> Any:
    import torch

    output = env.scene.sensors[sensor_cfg.name].data.output[data_type].clone().unsqueeze(1)
    output[torch.isnan(output) | torch.isinf(output)] = far_clip
    return output


class Go2HistoryWrapper:
    """Pack current proprioception + lidar + nine proprioceptive history frames.

    Wrap a raw ManagerBasedRLEnv. Step accepts joint actions and returns the
    legacy four-item inference tuple (observations, reward, done, extras).
    """

    def __init__(self, env: Any, history_length: int = HISTORY_LENGTH):
        import torch

        if history_length != HISTORY_LENGTH:
            raise ValueError("The bundled Go2 policy requires history_length=9")
        self.env = env
        self.history_length = history_length
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.num_obs = ACTOR_INPUT_DIM
        self.num_actions = ACTION_DIM
        self.proprio_obs_dim = PROPRIO_DIM
        self.clip_actions = 20.0
        self.proprio_obs_buf = torch.zeros(self.num_envs, history_length, PROPRIO_DIM, device=self.device)
        self._observations = None

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _pack(self, observations: dict[str, Any]) -> Any:
        import torch

        policy, proprio = observations["policy"], observations["proprio"]
        if policy.shape != (self.num_envs, POLICY_OBSERVATION_DIM) or proprio.shape != (self.num_envs, PROPRIO_DIM):
            raise ValueError("Go2 observations must contain 504 policy and 45 proprioceptive values per environment")
        self._observations = torch.cat((policy, self.proprio_obs_buf.reshape(self.num_envs, -1)), dim=1)
        return self._observations

    def get_observations(self) -> tuple[Any, dict[str, Any]]:
        observations = self.unwrapped.observation_manager.compute()
        self.proprio_obs_buf = observations["proprio"].unsqueeze(1).repeat(1, self.history_length, 1)
        packed = self._pack(observations)
        observations["policy"] = packed
        return packed, {"observations": observations}

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observations, extras = self.env.reset(**kwargs)
        self.proprio_obs_buf.zero_()
        packed = self._pack(observations)
        extras["observations"] = observations
        return packed, extras

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        import torch

        observations, reward, terminated, truncated, extras = self.env.step(actions.clamp(-self.clip_actions, self.clip_actions))
        proprio = observations["proprio"]
        history = torch.cat((self.proprio_obs_buf[:, 1:], proprio.unsqueeze(1)), dim=1)
        self.proprio_obs_buf = torch.where(
            (self.unwrapped.episode_length_buf < 1)[:, None, None], torch.zeros_like(history), history
        )
        packed = self._pack(observations)
        observations["policy"] = packed
        extras["observations"] = observations
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return packed, reward, (terminated | truncated).long(), extras

    def update_command(self, command: Any) -> None:
        """Set vx, vy, yaw rate before the next actor call, matching the original wrapper."""
        import torch

        if self._observations is None:
            raise RuntimeError("Call reset() or get_observations() before update_command()")
        command = torch.as_tensor(command, dtype=self._observations.dtype, device=self.device)
        self._observations[:, 6:9] = command
        self.proprio_obs_buf[:, -1, 6:9] = command

    def close(self) -> None:
        self.env.close()
