import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adahvla.locomotion import Go2HistoryWrapper, load_config, load_policy, project_root

try:
    import torch
except (ImportError, OSError):
    torch = None


class LocomotionPathTests(unittest.TestCase):
    def test_assets_resolve_from_project_root_independently_of_working_directory(self):
        with patch.dict(os.environ):
            os.environ.pop("ADAHVLA_ROOT", None)
            expected = load_config()
            root = project_root()
            previous = Path.cwd()
            with tempfile.TemporaryDirectory() as temporary:
                try:
                    os.chdir(temporary)
                    actual = load_config("configs/locomotion.json")
                finally:
                    os.chdir(previous)
            self.assertEqual(actual, expected)
            self.assertEqual(actual["paths"]["policy"], root / "assets/locomotion/go2/policy.jit")
            self.assertTrue(actual["paths"]["policy"].is_file())

    def test_explicit_asset_root_supports_a_relocated_project(self):
        config = {
            "paths": {"dataset": "data/episodes.json.gz", "policy": "assets/policy.jit",
                      "robot_usd": "assets/go2.usd", "scenes": "scenes"},
            "history_length": 9, "physics_dt": 0.005, "decimation": 4, "action_scale": 0.25,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs/locomotion.json").write_text(json.dumps(config))
            with patch.dict(os.environ, {"ADAHVLA_ROOT": str(root)}):
                relocated = load_config()
            self.assertEqual(relocated["paths"]["policy"], root / "assets/policy.jit")
            self.assertEqual(relocated["paths"]["dataset"], root / "data/episodes.json.gz")


class FakeGo2Environment:
    def __init__(self):
        self.unwrapped = self
        self.num_envs, self.device = 2, "cpu"
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.cfg = SimpleNamespace(is_finite_horizon=False)
        self.observation_manager = SimpleNamespace(compute=self.observations)
        self.tick = 0
        self.next_lengths = [1, 1]
        self.terminated, self.truncated = [False, False], [False, False]
        self.last_actions = None

    def observations(self):
        proprio = torch.arange(90, dtype=torch.float32).reshape(2, 45) / 100 + self.tick
        lidar = torch.full((2, 459), float(self.tick + 10))
        return {"proprio": proprio, "policy": torch.cat((proprio, lidar), dim=1)}

    def reset(self):
        self.tick = 0
        self.episode_length_buf.zero_()
        return self.observations(), {}

    def step(self, actions):
        self.last_actions = actions.clone()
        self.tick += 1
        self.episode_length_buf = torch.tensor(self.next_lengths)
        return (self.observations(), torch.tensor([1.0, 2.0]),
                torch.tensor(self.terminated), torch.tensor(self.truncated), {})


@unittest.skipIf(torch is None, "PyTorch is optional; install requirements-locomotion.txt to test the actor")
class Go2InferenceTests(unittest.TestCase):
    def test_bundled_actor_runs_frozen_finite_909_to_12_on_cpu(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ):
            os.environ.pop("ADAHVLA_ROOT", None)
            try:
                os.chdir(temporary)
                actor = load_policy(device="cpu")
            finally:
                os.chdir(previous)
        inputs = torch.zeros(2, 909)
        inputs[1] = torch.linspace(-0.1, 0.1, 909)
        with torch.inference_mode():
            actions = actor(inputs)
            repeated = actor(inputs)
        self.assertEqual(tuple(actions.shape), (2, 12))
        self.assertEqual(actions.device.type, "cpu")
        self.assertTrue(torch.isfinite(actions).all().item())
        torch.testing.assert_close(actions, repeated, rtol=0, atol=0)
        self.assertTrue(all(not parameter.requires_grad for parameter in actor.parameters()))

    def test_reset_is_empty_history_but_initial_observation_repeats_current_proprioception(self):
        env = FakeGo2Environment()
        wrapper = Go2HistoryWrapper(env)
        reset_input, reset_info = wrapper.reset()
        self.assertEqual(tuple(reset_input.shape), (2, 909))
        self.assertEqual(torch.count_nonzero(reset_input[:, 504:]).item(), 0)
        self.assertEqual(tuple(reset_info["observations"]["policy"].shape), (2, 504))
        env.tick = 3
        current = env.observations()
        initial_input, info = wrapper.get_observations()
        history = initial_input[:, 504:].reshape(2, 9, 45)
        for sample in history.unbind(dim=1):
            torch.testing.assert_close(sample, current["proprio"])
        self.assertIs(info["observations"]["policy"], initial_input)
        again, _ = wrapper.reset()
        self.assertEqual(torch.count_nonzero(again[:, 504:]).item(), 0)

    def test_commands_and_history_follow_original_vln_decision_timing(self):
        env = FakeGo2Environment()
        wrapper = Go2HistoryWrapper(env)
        packed, _ = wrapper.reset()
        commands = torch.tensor([[0.4, 0.0, 0.2], [-0.1, 0.0, -0.3]])
        wrapper.update_command(commands)
        torch.testing.assert_close(packed[:, 6:9], commands)
        self.assertEqual(torch.count_nonzero(packed[:, 504:]).item(), 0)
        joint_actions = torch.tensor([[30.0, -30.0] * 6] * 2)
        next_input, reward, done, info = wrapper.step(joint_actions)
        history = next_input[:, 504:].reshape(2, 9, 45)
        torch.testing.assert_close(history[:, -2, 6:9], commands)
        torch.testing.assert_close(history[:, -1], env.observations()["proprio"])
        self.assertEqual(torch.count_nonzero(history[:, :-2]).item(), 0)
        torch.testing.assert_close(next_input[:, :504], env.observations()["policy"])
        torch.testing.assert_close(env.last_actions, joint_actions.clamp(-20, 20))
        torch.testing.assert_close(reward, torch.tensor([1.0, 2.0]))
        self.assertEqual(done.tolist(), [0, 0])
        self.assertIs(info["observations"]["policy"], next_input)

    def test_done_union_and_history_reset_are_independent_for_batched_environments(self):
        env = FakeGo2Environment()
        wrapper = Go2HistoryWrapper(env)
        wrapper.get_observations()
        before = env.observations()["proprio"]
        env.next_lengths = [0, 2]
        env.terminated, env.truncated = [True, False], [False, True]
        packed, _, done, info = wrapper.step(torch.zeros(2, 12))
        history = packed[:, 504:].reshape(2, 9, 45)
        self.assertEqual(done.tolist(), [1, 1])
        self.assertEqual(done.dtype, torch.long)
        self.assertEqual(torch.count_nonzero(history[0]).item(), 0)
        torch.testing.assert_close(history[1, -2], before[1])
        torch.testing.assert_close(history[1, -1], env.observations()["proprio"][1])
        self.assertEqual(info["time_outs"].tolist(), [False, True])
        env.cfg.is_finite_horizon = True
        _, _, _, finite_info = wrapper.step(torch.zeros(2, 12))
        self.assertNotIn("time_outs", finite_info)


if __name__ == "__main__":
    unittest.main()
