import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from adahvla.adaptation import Adaptation
from adahvla.workspace import Case, Workspace


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("adahvla_runner_under_test", PROJECT / "scripts/run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class NeverCalledRole:
    def complete(self, messages):
        raise AssertionError("The stopped prototype run must not call revision roles")


class StopManager:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def complete(self, messages):
        self.events.append("manager")
        self.calls.append(messages)
        return {"tool": "stop", "arguments": {}, "rationale": "Retain the measured prototype"}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_known_episode_ids_are_disjoint_and_route_labels_are_excluded(self):
        with patch.dict(os.environ, {"ADAHVLA_ROOT": str(PROJECT)}):
            settings = runner.load_config()
        args = runner.arguments([])
        cases = runner.build_cases(args, settings)
        self.assertEqual([(case.id, case.split) for case in cases], [
            ("episode-27", "train"), ("episode-124", "validation"), ("episode-167", "test"),
        ])
        for case in cases:
            episode = case.metadata["episode"]
            self.assertEqual(str(episode["episode_id"]), case.id.removeprefix("episode-"))
            self.assertEqual(set(episode), {
                "episode_id", "scene_id", "start_position", "start_rotation", "goals", "instruction",
            })
            self.assertEqual(episode["instruction"], {"instruction_text": case.task})
        for flags in (["--validation-ids", "27"], ["--train-ids", "999999999"], ["--train-ids", ""]):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                runner.build_cases(runner.arguments(flags), settings)

    def test_shell_offline_check_works_from_another_directory_without_credentials(self):
        env = {key: value for key, value in os.environ.items() if not key.startswith("ADAHVLA_")}
        env.update(ADAHVLA_PYTHON=sys.executable, ADAHVLA_ROOT=str(self.root / "wrong-root"),
                   PYTHONDONTWRITEBYTECODE="1")
        output = self.root / "unused-output"
        result = subprocess.run(
            ["bash", str(PROJECT / "scripts/run.sh"), "--check", "--output-dir", str(output)],
            cwd=self.root, env=env, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"status": "offline checks passed"', result.stdout)
        self.assertIn('"episode-27"', result.stdout)
        self.assertFalse(output.exists())

    def session(self):
        events = []
        manager = StopManager(events)
        cases = [Case("train", "Hold the object", "table"),
                 Case("PRIVATE_TEST", "Place the object", "cabinet", split="test")]

        def evaluator(candidate, case, output):
            events.append(f"evaluate:{case.id}")
            return {"success": False, "metrics": {"observed_steps": 1},
                    "events": [{"timestamp": 0.0, "observation": "The object is stationary"}]}

        roles = {"manager": manager, "analyst": NeverCalledRole(),
                 "engineer": NeverCalledRole(), "reviewer": NeverCalledRole()}
        session = Adaptation(Workspace(self.root / "session"), cases, evaluator, **roles)
        return session, roles, events

    def assert_measured_before_manager_and_frozen_test(self, session, roles, events):
        self.assertEqual(events, ["evaluate:train", "manager", "evaluate:PRIVATE_TEST"])
        packet = json.loads(roles["manager"].calls[0][1]["content"][0]["text"])
        self.assertEqual(len(packet["rollouts"]), 1)
        self.assertEqual(packet["rollouts"][0]["candidate_id"], "C0000")
        self.assertEqual(packet["rollouts"][0]["metrics"], {"observed_steps": 1})
        self.assertNotIn("PRIVATE_TEST", json.dumps(roles["manager"].calls))
        self.assertEqual(session.state["phase"], "complete")
        self.assertEqual(session.state["frozen"]["candidate_id"], "C0000")
        self.assertEqual(len(session.state["rollouts"]), 1)
        self.assertEqual(set(session.state["test_rollouts"]), {"PRIVATE_TEST"})

    def test_regular_session_measures_prototype_before_manager_and_tests_after_stop(self):
        session, roles, events = self.session()
        with contextlib.redirect_stdout(io.StringIO()):
            selected = runner.run_session(session)
        self.assertEqual(selected.id, "C0000")
        self.assert_measured_before_manager_and_frozen_test(session, roles, events)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run_session(session)
        self.assertEqual(events, ["evaluate:train", "manager", "evaluate:PRIVATE_TEST"])

    def test_prototype_only_then_resume_reuses_measurement(self):
        session, roles, events = self.session()
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run_session(session, prototype_only=True)
            runner.run_session(session, prototype_only=True)
        self.assertEqual(events, ["evaluate:train"])
        self.assertEqual(session.state["phase"], "adapt")
        self.assertEqual(session.state["steps"], 0)
        resumed = Adaptation(Workspace(self.root / "session"), list(session.cases.values()),
                             session.evaluator, **roles)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run_session(resumed)
        self.assert_measured_before_manager_and_frozen_test(resumed, roles, events)

    def test_host_dependency_change_is_rejected_before_evaluator_launch(self):
        script = self.root / "evaluate.py"
        script.write_text("raise AssertionError('This evaluator must not launch')\n")
        dependency = self.root / "environment.py"
        dependency.write_text("control_period = 0.02\n")
        evaluator = runner.Go2Evaluator([sys.executable, str(script)], 10, [dependency])
        candidate = Workspace(self.root / "session").baseline()
        dependency.write_text("control_period = 0.10\n")
        output = self.root / "output"
        output.mkdir()
        with patch("adahvla.workspace.subprocess.run") as launch:
            with self.assertRaisesRegex(ValueError, "changed"):
                evaluator(candidate, Case("train", "Hold the object", "table"), output)
        launch.assert_not_called()
        self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
