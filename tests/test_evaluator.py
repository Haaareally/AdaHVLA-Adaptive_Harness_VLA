import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

import adahvla
import adahvla.harness as host_harness
import adahvla.vla as host_vla


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("adahvla_evaluator_under_test", PROJECT / "scripts/evaluate.py")
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode()


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def candidate(self, name="candidate"):
        directory = self.root / name
        package = directory / "adahvla"
        package.mkdir(parents=True)
        for filename in ("__init__.py", "harness.py", "vla.py"):
            shutil.copyfile(PROJECT / "src/adahvla" / filename, package / filename)
        with (package / "harness.py").open("a", encoding="utf-8") as stream:
            stream.write("\nCANDIDATE_TEST_MARKER = 'sealed revision'\n")
        files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in package.iterdir()}
        manifest = {
            "id": name, "parent_id": None, "path": str(directory), "files": files,
            "digest": hashlib.sha256(canonical_bytes(files)).hexdigest(),
        }
        (directory / "candidate.json").write_bytes(canonical_bytes(manifest))
        return manifest

    def test_native_turns_preserve_original_yaw_rate_and_duration(self):
        # Expected constants come from NaVILA-Bench's get_vel_command contract.
        cases = [
            ("Turn left 90 degrees", 1, 3.0),
            ("Rotate clockwise 45 deg", -1, 1.5),
            ("Turn sharply right", -1, 1.5),
            ("Turn slightly left", 1, 0.5),
            ("Turn left 0 degrees", 1, 0.5),
            ("Turn right 2 degrees", -1, 0.35),
        ]
        for text, direction, seconds in cases:
            with self.subTest(text=text):
                result = evaluator.parse_native_action(text)
                self.assertEqual(result["kind"], "turn")
                self.assertTrue(result["recognized"])
                self.assertEqual(result["velocity"][:2], [0.0, 0.0])
                self.assertAlmostEqual(result["velocity"][2], direction * math.pi / 6)
                self.assertAlmostEqual(result["seconds"], seconds)

    def test_native_distance_units_stop_priority_and_unknown_hold(self):
        for text, seconds in [
            ("Move forward 1.5 meters", 3.0),
            ("Go straight 75 cm", 1.5),
            ("Advance 2 feet", 1.2192),
            ("Walk forward 0 cm", 0.5),
            ("Step forward 1 cm", 0.25),
            ("Continue straight", 0.5),
        ]:
            with self.subTest(text=text):
                result = evaluator.parse_native_action(text)
                self.assertEqual(result["kind"], "move")
                self.assertEqual(result["velocity"], [0.5, 0.0, 0.0])
                self.assertAlmostEqual(result["seconds"], seconds)
        for text in ("Move forward, then stop", "Turn left; destination reached", "Hold position"):
            with self.subTest(text=text):
                self.assertEqual(evaluator.parse_native_action(text), {
                    "kind": "stop", "velocity": [0.0, 0.0, 0.0], "seconds": 0.0, "recognized": True,
                })
        for text, recognized in (("Inspect the blue doorway", False), ("", False), ("Pause", True)):
            with self.subTest(text=text):
                self.assertEqual(evaluator.parse_native_action(text), {
                    "kind": "hold", "velocity": [0.0, 0.0, 0.0], "seconds": 0.5, "recognized": recognized,
                })

    def test_stop_words_require_boundaries_and_unnegated_requests(self):
        for text in (
            "Move forward nonstop for 25 cm",
            "Continue straight without stopping for 25 cm",
            "Do not stop. Move forward 25 cm",
            "Don't stop; move forward 25 cm",
            "Never halt, continue forward 25 cm",
        ):
            with self.subTest(text=text):
                result = evaluator.parse_native_action(text)
                self.assertEqual(result["kind"], "move")
                self.assertEqual(result["seconds"], 0.5)
        for text in ("Stop.", "Move forward nonstop, then stop", "Do not stop yet; then stop."):
            with self.subTest(text=text):
                self.assertEqual(evaluator.parse_native_action(text)["kind"], "stop")

    def test_success_requires_stop_and_strict_three_dimensional_radius(self):
        self.assertEqual(evaluator.success_at_stop([0, 0, 0.5], [0, 0, 0], 1, True), (True, 0.5))
        self.assertEqual(evaluator.success_at_stop([0, 0, 0.5], [0, 0, 0], 1, False), (False, 0.5))
        self.assertEqual(evaluator.success_at_stop([0, 0, 1], [0, 0, 0], 1, True), (False, 1.0))
        self.assertEqual(evaluator.success_at_stop([0, 0, 2], [0, 0, 0], 1, True), (False, 2.0))
        for position, goal, radius in [
            ([0, 0], [0, 0, 0], 1), ([0, 0, 0], [0, 0], 1),
            ([math.nan, 0, 0], [0, 0, 0], 1), ([0, 0, 0], [math.inf, 0, 0], 1),
            ([0, 0, 0], [0, 0, 0], 0), ([0, 0, 0], [0, 0, 0], math.inf),
        ]:
            with self.subTest(position=position, goal=goal, radius=radius), self.assertRaises(ValueError):
                evaluator.success_at_stop(position, goal, radius, True)

    def test_candidate_imports_sealed_sources_and_restores_host_after_failure(self):
        manifest = self.candidate()
        old_path, old_bytecode = sys.path[:], sys.dont_write_bytecode
        old_modules = {name: module for name, module in sys.modules.items()
                       if name == "adahvla" or name.startswith("adahvla.")}
        for fail in (False, True):
            with self.subTest(fail=fail):
                try:
                    with evaluator.candidate_modules(manifest) as (harness, vla):
                        self.assertIsNot(harness, host_harness)
                        self.assertIsNot(vla, host_vla)
                        self.assertIsNot(sys.modules["adahvla"], adahvla)
                        self.assertEqual(harness.CANDIDATE_TEST_MARKER, "sealed revision")
                        self.assertEqual(Path(harness.__file__), self.root / "candidate/adahvla/harness.py")
                        self.assertEqual(Path(vla.__file__), self.root / "candidate/adahvla/vla.py")
                        self.assertIs(sys.modules["adahvla"].Harness, harness.Harness)
                        if fail:
                            raise RuntimeError("execution failed")
                except RuntimeError as exc:
                    self.assertTrue(fail)
                    self.assertEqual(str(exc), "execution failed")
                self.assertEqual(sys.path, old_path)
                self.assertEqual(sys.dont_write_bytecode, old_bytecode)
                current = {name: module for name, module in sys.modules.items()
                           if name == "adahvla" or name.startswith("adahvla.")}
                self.assertEqual(current, old_modules)
        self.assertEqual({path.name for path in (self.root / "candidate/adahvla").iterdir()},
                         {"__init__.py", "harness.py", "vla.py"})

    def test_candidate_verification_rejects_source_tampering_extra_files_and_symlinks(self):
        for damage in ("changed_source", "extra_file", "source_symlink", "changed_manifest"):
            with self.subTest(damage=damage):
                manifest = self.candidate(damage)
                directory = Path(manifest["path"])
                self.assertEqual(evaluator.verify_candidate(manifest), directory.resolve())
                source = directory / "adahvla/harness.py"
                if damage == "changed_source":
                    source.write_bytes(source.read_bytes() + b"\n# unsealed edit\n")
                elif damage == "extra_file":
                    (directory / "adahvla/extra.py").write_text("pass\n")
                elif damage == "source_symlink":
                    target = directory / "external.py"
                    source.replace(target)
                    source.symlink_to(target)
                else:
                    (directory / "candidate.json").write_bytes(canonical_bytes(dict(manifest, id="different")))
                with self.assertRaises(ValueError):
                    evaluator.verify_candidate(manifest)

    def test_recorded_vla_enforces_one_native_action_and_no_call_on_completion(self):
        native_action = "Turn left 90 degrees"
        executor = SimpleNamespace(predict=lambda query, frames: native_action)
        recorded = evaluator.RecordedVLA(executor)
        frames = [SimpleNamespace(timestamp=1.0), SimpleNamespace(timestamp=2.0)]
        action = recorded.predict("Face the exit", frames)
        execution = SimpleNamespace(handoff=SimpleNamespace(completed=False), action=action)
        recorded.verify_handoff(execution, 0)
        self.assertEqual(recorded.query, "Face the exit")
        self.assertEqual(recorded.frame_times, [1.0, 2.0])
        for previous_calls, changed_action in ((1, native_action), (-1, native_action), (0, "Move forward")):
            with self.subTest(previous_calls=previous_calls, action=changed_action), self.assertRaises(ValueError):
                recorded.verify_handoff(SimpleNamespace(handoff=execution.handoff, action=changed_action), previous_calls)
        completed = SimpleNamespace(handoff=SimpleNamespace(completed=True), action=None)
        recorded.verify_handoff(completed, 1)
        with self.assertRaises(ValueError):
            recorded.verify_handoff(completed, 0)
        completed.action = native_action
        with self.assertRaises(ValueError):
            recorded.verify_handoff(completed, 1)

    def test_runtime_options_use_defaults_then_metadata_then_explicit_cli(self):
        runtime = {"reasoner_base_url": "https://reasoner.invalid/v1", "reasoner_model": "metadata-model",
                   "vla_port": 12345, "max_decisions": 12, "headless": True}
        case = {"metadata": {"runtime": runtime}}
        args = evaluator.build_parser().parse_args([
            "--request", "request.json", "--output", "result.json",
            "--reasoner-model", "cli-model", "--max-decisions", "3", "--no-headless",
        ])
        options = evaluator.runtime_options(args, case)
        self.assertEqual(options["reasoner_base_url"], runtime["reasoner_base_url"])
        self.assertEqual(options["reasoner_model"], "cli-model")
        self.assertEqual(options["vla_port"], 12345)
        self.assertEqual(options["max_decisions"], 3)
        self.assertFalse(options["headless"])
        self.assertEqual(options["frame_interval"], 0.5)
        self.assertEqual(runtime["max_decisions"], 12)
        self.assertTrue(runtime["headless"])

    def test_runtime_options_reject_invalid_limits_and_missing_reasoner(self):
        for key, value in [
            ("max_episode_seconds", 0), ("frame_interval", math.nan), ("evidence_interval", True),
            ("vla_timeout", math.inf), ("vla_frames", 1), ("max_decisions", 2.5),
            ("vla_port", True), ("headless", "false"), ("device", "cuda:abc"),
        ]:
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                evaluator.runtime_options(SimpleNamespace(), {"metadata": {"runtime": {
                    "reasoner_base_url": "https://reasoner.invalid/v1", "reasoner_model": "test-model",
                    key: value,
                }}})
        with self.assertRaisesRegex(ValueError, "required"):
            evaluator.runtime_options(SimpleNamespace(), {})
        with self.assertRaisesRegex(ValueError, "object"):
            evaluator.runtime_options(SimpleNamespace(), {"metadata": {"runtime": []}})


if __name__ == "__main__":
    unittest.main()
