import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from copy import deepcopy
from pathlib import Path

from adahvla.adaptation import Adaptation, AdaptationPolicy
from adahvla.workspace import Case, CommandEvaluator, Workspace, _copy_redacted


EVALUATOR_SOURCE = '''
import argparse
import json
from pathlib import Path
from adahvla.harness import Frame, Harness, Observation
import adahvla.harness as implementation

parser = argparse.ArgumentParser()
parser.add_argument("--request")
parser.add_argument("--output")
args = parser.parse_args()
request = json.loads(Path(args.request).read_text())
class Reasoner:
    def complete(self, messages):
        return dict(plan=[dict(instruction="Hold the object", completion="Object held")],
                    progress="hold", mode="execute", query="Hold the object",
                    evidence="Object observed", memory="observed-memory-long")
harness = Harness(Reasoner())
harness.reset(request["case"]["task"])
harness.step(Observation((Frame("aW1hZ2U=", 0.0),)))
retained = len(harness.state.memory)
result = dict(source_digest=request["candidate"]["digest"], success=retained <= 8,
              metrics=dict(retained_chars=retained),
              events=[dict(timestamp=0.0, memory=harness.state.memory,
                           implementation=implementation.__file__)])
Path(args.output).write_text(json.dumps(result))
'''


def evidence_refs(messages):
    return [item["text"].removeprefix("Observed rollout ")
            for item in messages[1]["content"]
            if item["type"] == "text" and item["text"].startswith("Observed rollout ")]


class Role:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        return deepcopy(self.responder(messages))


class Roles:
    def __init__(self, actions=(), assessment=None):
        actions = iter(actions)
        self.manager = Role(lambda _: next(actions))
        self.analyst = Role(lambda messages: {
            "facts": ["Event 0 records the retained memory"],
            "failure_signature": "The retained memory exceeds the intended bound",
            "questions": [], "evidence_refs": evidence_refs(messages),
        })
        self.engineer = Role(self.propose)
        self.assessment = assessment or {}
        self.reviewer = Role(self.review)

    def propose(self, messages):
        if isinstance(messages[-1]["content"], list):
            return {"tool": "read_source", "arguments": {"path": "harness.py"}}
        source = json.loads(messages[-1]["content"])["result"]
        match = re.search(r"memory_chars: int = (\d+)", source)
        old = match.group(0)
        limit = int(match.group(1))
        packet = json.loads(messages[1]["content"][0]["text"])
        return {"tool": "propose", "arguments": {
            "hypothesis": "A bounded summary reduces retained context",
            "mechanism": "Reduce the progress policy memory budget",
            "expected_effect": "The observed memory length decreases",
            "observable_signal": "The next event records a shorter memory",
            "falsifier": "Memory length does not decrease",
            "control_surface": "ProgressPolicy.memory_chars",
            "protected_behavior": "The VLA continues to own actions",
            "state_lifecycle": "The bound is applied at every committed transition",
            "source_refs": ["harness.py:ProgressPolicy"],
            "evidence_refs": packet["analysis"]["evidence_refs"],
            "edits": [{"path": "harness.py", "old": old,
                       "new": f"memory_chars: int = {8 if limit > 8 else max(1, limit // 2)}"}],
        }}

    def review(self, messages):
        if not evidence_refs(messages):
            return {"eligible": True, "reason": "The bounded memory change preserves interfaces"}
        report = {
            "mechanism_triggered": "yes", "pre_trigger_divergence": "no",
            "target_effect": "improved", "overall_effect": "improved",
            "observations": "The child event records less retained memory",
            "hard_regressions": [], "new_failure": "", "evidence_refs": evidence_refs(messages),
        }
        report.update(self.assessment)
        return report

    def kwargs(self):
        return {name: getattr(self, name) for name in ("manager", "analyst", "engineer", "reviewer")}

    def call_counts(self):
        return [len(role.calls) for role in self.kwargs().values()]


def action(tool, **arguments):
    return {"tool": tool, "arguments": arguments, "rationale": "Exercise the recorded revision lifecycle"}


def synthetic_evaluator(candidate, case, output):
    return {"success": candidate.parent_id is not None, "metrics": {"steps": 1},
            "events": [{"timestamp": 0, "observed": "The object is held"}]}


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cases = [Case("train_a", "Hold an object", "table"),
                      Case("train_b", "Move an object", "shelf"),
                      Case("validation", "Put an object away", "cabinet", split="validation"),
                      Case("SECRET_HELDOUT", "Hidden test task", "unseen", split="test")]

    def controller(self, *, roles=None, policy=None, evaluator=synthetic_evaluator, directory="session"):
        roles = roles or Roles()
        controller = Adaptation(Workspace(self.root / directory), self.cases, evaluator,
                                policy=policy, **roles.kwargs())
        return controller, roles

    def child(self, controller):
        parent = controller.evaluate("C0000", "train_a")
        return controller.revise("C0000", parent.id), parent

    def test_full_loop_executes_edited_source_then_freezes_without_heldout_feedback(self):
        evaluator_path = self.root / "evaluate.py"
        evaluator_path.write_text(EVALUATOR_SOURCE)
        evaluator = CommandEvaluator([sys.executable, str(evaluator_path)])
        roles = Roles([
            action("evaluate", candidate_id="C0000", case_id="train_a"),
            action("revise", parent_id="C0000", rollout_id="R0001"),
            action("evaluate", candidate_id="C0001", case_id="train_a"),
            action("evaluate", candidate_id="C0000", case_id="validation"),
            action("evaluate", candidate_id="C0001", case_id="validation"),
            action("select", candidate_id="C0001"), action("start_test"),
        ])
        controller, _ = self.controller(roles=roles, evaluator=evaluator)
        controller.run(steps=7)
        self.assertEqual(controller.state["phase"], "test")
        self.assertEqual(controller.selected.id, "C0001")
        self.assertEqual(controller.rollout("R0001").metrics["retained_chars"], 20)
        self.assertEqual(controller.rollout("R0002").metrics["retained_chars"], 8)
        evidence = json.loads(Path(controller.rollout("R0002").evidence_path).read_text())
        self.assertEqual(Path(evidence["events"][0]["implementation"]), controller.selected.source_path)
        self.assertEqual(controller.state["hypotheses"]["H0001"]["support"], 2)
        self.assertTrue(controller.state["revisions"]["C0001"]["checks"]["passed"])
        calls = roles.call_counts()
        controller.run()
        self.assertEqual(controller.state["phase"], "complete")
        self.assertEqual(roles.call_counts(), calls)
        self.assertEqual(len(controller.state["rollouts"]), 4)
        self.assertEqual(controller.state["test_rollouts"]["SECRET_HELDOUT"]["candidate_id"], "C0001")
        for role in roles.kwargs().values():
            self.assertNotIn("SECRET_HELDOUT", json.dumps(role.calls))
        with self.assertRaisesRegex(ValueError, "closed"):
            controller.evaluate("C0001", "train_a")

    def test_attribution_requires_trigger_and_no_prior_divergence(self):
        combinations = [("yes", "no", True), ("no", "no", False),
                        ("yes", "yes", False), ("uncertain", "no", False),
                        ("yes", "uncertain", False)]
        for index, (trigger, divergence, expected) in enumerate(combinations):
            with self.subTest(trigger=trigger, divergence=divergence):
                roles = Roles(assessment={"mechanism_triggered": trigger,
                                          "pre_trigger_divergence": divergence})
                controller, _ = self.controller(roles=roles, directory=f"gate-{index}")
                child, _ = self.child(controller)
                run = controller.evaluate(child.id, "train_a")
                effect = next(iter(controller.state["effects"].values()))
                self.assertEqual(effect["chi"], expected)
                self.assertEqual(controller.state["hypotheses"]["H0001"]["support"], int(expected))
                self.assertTrue(run.success)

    def test_comparison_requires_same_case_and_direct_parent(self):
        controller, roles = self.controller()
        child, baseline_run = self.child(controller)
        wrong_case = controller.evaluate("C0000", "train_b")
        child_run = controller.evaluate(child.id, "train_a")
        grandchild = controller.revise(child.id, child_run.id)
        grandchild_run = controller.evaluate(grandchild.id, "train_a")
        before = len(roles.reviewer.calls)
        with self.assertRaisesRegex(ValueError, "direct lineage"):
            controller.assess(wrong_case.id, child_run.id)
        with self.assertRaisesRegex(ValueError, "direct lineage"):
            controller.assess(baseline_run.id, grandchild_run.id)
        self.assertEqual(len(roles.reviewer.calls), before)

    def test_review_does_not_substitute_for_behavioral_assessment(self):
        controller, roles = self.controller()
        child, baseline = self.child(controller)
        with self.assertRaisesRegex(ValueError, "behavioral evidence"):
            controller.select(child.id)
        with self.assertRaisesRegex(ValueError, "this parent"):
            controller.revise(child.id, baseline.id)
        controller.evaluate(child.id, "train_a")

        def unavailable(messages):
            raise TimeoutError("Assessment service unavailable")

        roles.reviewer.responder = unavailable
        with self.assertRaises(TimeoutError):
            controller.evaluate(child.id, "train_a")
        latest = controller.rollout("R0003")
        with self.assertRaises(ValueError):
            controller.revise(child.id, latest.id)
        attempts = controller.state["rollout_attempts"]
        roles.reviewer.responder = roles.review
        controller.step()
        self.assertEqual(controller.state["rollout_attempts"], attempts)
        self.assertEqual(len(roles.manager.calls), 0)
        self.assertTrue(any(effect["child_rollout_id"] == latest.id
                            for effect in controller.state["effects"].values()))

    def test_extension_needs_improvement_or_a_new_observed_failure(self):
        for index, new_failure in enumerate(("", "The child exposes a later grasp failure")):
            with self.subTest(new_failure=new_failure):
                roles = Roles(assessment={"target_effect": "unchanged", "overall_effect": "noninferior",
                                          "new_failure": new_failure})
                controller, _ = self.controller(roles=roles, directory=f"extension-{index}")
                child, _ = self.child(controller)
                run = controller.evaluate(child.id, "train_a")
                if new_failure:
                    self.assertEqual(controller.revise(child.id, run.id).parent_id, child.id)
                else:
                    with self.assertRaises(ValueError):
                        controller.revise(child.id, run.id)

    def test_discarded_ancestor_frees_width_taken_by_its_descendants(self):
        controller, _ = self.controller(policy=AdaptationPolicy(width=1))
        child, baseline = self.child(controller)
        run = controller.evaluate(child.id, "train_a")
        grandchild = controller.revise(child.id, run.id)
        controller.evaluate(grandchild.id, "train_a")
        controller.discard(child.id)
        with self.assertRaisesRegex(ValueError, "eligible"):
            controller.evaluate(grandchild.id, "train_a")
        sibling = controller.revise("C0000", baseline.id)
        self.assertEqual(sibling.parent_id, "C0000")

    def test_clean_repeat_resolves_uncertain_assessment_before_selection(self):
        roles = Roles(assessment={"target_effect": "uncertain", "overall_effect": "uncertain"})
        controller, _ = self.controller(roles=roles)
        child, _ = self.child(controller)
        controller.evaluate(child.id, "train_a")
        with self.assertRaises(ValueError):
            controller.select(child.id)
        roles.assessment.clear()
        controller.evaluate(child.id, "train_a")
        controller.evaluate("C0000", "validation")
        controller.evaluate(child.id, "validation")
        self.assertEqual(controller.select(child.id).id, child.id)

    def test_further_local_improvement_can_replace_already_selected_parent(self):
        controller, _ = self.controller()
        child, _ = self.child(controller)
        run = controller.evaluate(child.id, "train_a")
        controller.evaluate("C0000", "validation")
        controller.evaluate(child.id, "validation")
        controller.select(child.id)
        grandchild = controller.revise(child.id, run.id)
        controller.evaluate(grandchild.id, "train_a")
        controller.evaluate(grandchild.id, "validation")
        self.assertEqual(controller.select(grandchild.id).id, grandchild.id)

    def test_heldout_failure_can_resume_without_model_feedback(self):
        failed = False

        def evaluator(candidate, case, output):
            nonlocal failed
            if case.split == "test" and not failed:
                failed = True
                raise RuntimeError("Transient heldout simulator failure")
            return synthetic_evaluator(candidate, case, output)

        controller, _ = self.controller(evaluator=evaluator)
        controller.start_test()
        frozen = deepcopy(controller.state["frozen"])
        with self.assertRaises(RuntimeError):
            controller.step()
        resumed, roles = self.controller(evaluator=evaluator)
        resumed.run()
        self.assertEqual(resumed.state["phase"], "complete")
        self.assertEqual(resumed.state["frozen"], frozen)
        self.assertEqual(roles.call_counts(), [0, 0, 0, 0])
        self.assertEqual(set(resumed.state["test_rollouts"]), {"SECRET_HELDOUT"})

    def test_stopped_adaptation_can_test_once_without_reopening_model_decisions(self):
        endings = [action("stop"), action("evaluate", candidate_id="C0000", case_id="train_a")]
        for index, ending in enumerate(endings):
            with self.subTest(ending=ending["tool"]):
                roles = Roles([ending])
                policy = AdaptationPolicy(max_steps=1)
                directory = f"stopped-{index}"
                controller, _ = self.controller(roles=roles, policy=policy, directory=directory)
                controller.run()
                self.assertEqual(controller.state["phase"], "complete")
                self.assertIsNone(controller.state["frozen"])
                resumed, _ = self.controller(roles=roles, policy=policy, directory=directory)
                calls = roles.call_counts()
                resumed.start_test()
                frozen = deepcopy(resumed.state["frozen"])
                self.assertEqual(frozen["candidate_id"], resumed.selected.id)
                self.assertEqual(resumed.available_tools(), {})
                with self.assertRaises(ValueError):
                    resumed.start_test()
                with self.assertRaisesRegex(ValueError, "closed"):
                    resumed.evaluate("C0000", "train_a")
                resumed.run()
                self.assertEqual(resumed.state["phase"], "complete")
                self.assertEqual(resumed.state["frozen"], frozen)
                self.assertEqual(roles.call_counts(), calls)
                self.assertEqual(set(resumed.state["test_rollouts"]), {"SECRET_HELDOUT"})
                state = deepcopy(resumed.state)
                with self.assertRaises(ValueError):
                    resumed.start_test()
                with self.assertRaisesRegex(ValueError, "closed"):
                    resumed.switch_task("train_b")
                self.assertEqual(resumed.state, state)

    def test_width_depth_and_total_candidate_budgets_are_enforced(self):
        for name, policy in [
            ("width", AdaptationPolicy(width=1)),
            ("depth", AdaptationPolicy(depth=1)),
            ("Candidate budget", AdaptationPolicy(max_candidates=1)),
        ]:
            with self.subTest(limit=name):
                controller, _ = self.controller(policy=policy, directory=name.replace(" ", "-"))
                child, baseline = self.child(controller)
                if name == "depth":
                    run = controller.evaluate(child.id, "train_a")
                    target, evidence = child.id, run.id
                else:
                    target, evidence = "C0000", baseline.id
                attempts = controller.state["candidate_attempts"]
                with self.assertRaisesRegex(ValueError, name):
                    controller.revise(target, evidence)
                self.assertEqual(controller.state["candidate_attempts"], attempts)

    def test_failed_evaluations_consume_budget_and_heldout_cannot_be_queried(self):
        def failing(candidate, case, output):
            raise RuntimeError("Synthetic simulator failure")

        controller, roles = self.controller(evaluator=failing, policy=AdaptationPolicy(max_rollouts=1))
        with self.assertRaisesRegex(ValueError, "Held-out"):
            controller.evaluate("C0000", "SECRET_HELDOUT")
        self.assertEqual(controller.state["rollout_attempts"], 0)
        with self.assertRaisesRegex(RuntimeError, "Synthetic simulator"):
            controller.evaluate("C0000", "train_a")
        with self.assertRaisesRegex(ValueError, "budget"):
            controller.evaluate("C0000", "train_a")
        self.assertEqual(controller.state["rollout_attempts"], 1)
        self.assertEqual(roles.call_counts(), [0, 0, 0, 0])

    def test_resume_preserves_graph_and_rejects_changed_protocol(self):
        controller, _ = self.controller()
        child, _ = self.child(controller)
        controller.evaluate(child.id, "train_a")
        saved = deepcopy(controller.state)
        resumed, roles = self.controller()
        self.assertEqual(resumed.state, saved)
        self.assertEqual(roles.call_counts(), [0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "preserve cases"):
            self.controller(policy=AdaptationPolicy(depth=7))

    def test_failure_events_redact_credentials_before_persisting(self):
        canary = "synthetic-adaptation-secret"

        def failing(candidate, case, output):
            raise RuntimeError("Provider failure: " + canary)

        roles = Roles([action("evaluate", candidate_id="C0000", case_id="train_a")])
        controller, _ = self.controller(roles=roles, evaluator=failing)
        with patch.dict(os.environ, {"ADAHVLA_API_KEY": canary}):
            with self.assertRaises(RuntimeError):
                controller.step()
        saved = controller.path.read_text()
        self.assertNotIn(canary, saved)
        self.assertIn("Provider failure: [redacted]", saved)

    def test_task_switch_keeps_revision_history_but_resets_local_depth(self):
        controller, _ = self.controller(policy=AdaptationPolicy(depth=1))
        child, _ = self.child(controller)
        controller.evaluate(child.id, "train_a")
        controller.evaluate("C0000", "validation")
        controller.evaluate(child.id, "validation")
        controller.select(child.id)
        controller.switch_task("train_b")
        controller.evaluate("C0000", "train_b")
        run = controller.evaluate(child.id, "train_b")
        next_child = controller.revise(child.id, run.id)
        self.assertEqual(next_child.parent_id, child.id)
        self.assertEqual(controller.state["focus_root"], child.id)
        self.assertEqual(len(controller.state["hypotheses"]), 2)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Workspace(Path(self.temporary.name))
        self.baseline = self.workspace.baseline()

    def test_smoke_checks_do_not_inherit_credentials_and_redact_output(self):
        canary = "synthetic-check-secret"
        injection = (
            "import json\nimport os\n"
            "assert 'ADAHVLA_API_KEY' not in os.environ\n"
            "assert 'OPENAI_API_KEY' not in os.environ\n"
            f"print({canary!r})"
        )
        child = self.workspace.apply_patch(self.baseline, "C0001", [{
            "path": "harness.py", "old": "import json", "new": injection,
        }])
        with patch.dict(os.environ, {"ADAHVLA_API_KEY": canary, "OPENAI_API_KEY": "synthetic-other-secret"}):
            report = self.workspace.run_checks(child)
        self.assertTrue(report["passed"], report["output"])
        self.assertNotIn(canary, report["output"])
        self.assertNotIn(canary, (self.workspace.root / "checks/C0001.json").read_text())
        self.assertIn("[redacted]", report["output"])

    def test_log_redaction_handles_every_secret_chunk_boundary(self):
        canary = "synthetic-stream-secret"

        class Chunks:
            def __init__(self, values):
                self.values = iter(values)

            def read1(self, _size):
                return next(self.values, b"")

        payload = ("prefix " + canary + " suffix " + canary).encode()
        for boundary in range(1, len(payload)):
            with self.subTest(boundary=boundary):
                target = io.BytesIO()
                _copy_redacted(Chunks([payload[:boundary], payload[boundary:]]), target, [canary])
                self.assertEqual(target.getvalue(), b"prefix [redacted] suffix [redacted]")

    def test_evaluator_logs_redact_credentials_on_success_failure_and_timeout(self):
        canary = "synthetic-evaluator-secret"
        source = '''
import argparse, json, os, sys, time
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--request')
parser.add_argument('--output')
args = parser.parse_args()
assert 'OPENAI_API_KEY' not in os.environ
secret = os.environ['ADAHVLA_API_KEY']
sys.stdout.write(secret[:8])
sys.stdout.flush()
time.sleep(0.03)
sys.stdout.write(secret[8:] + '\\n')
sys.stdout.flush()
MODE
request = json.loads(Path(args.request).read_text())
Path(args.output).write_text(json.dumps(dict(source_digest=request['candidate']['digest'],
    success=True, metrics={}, events=[dict(event='done')])))
'''
        cases = [("success", "", None), ("failure", "raise RuntimeError(secret)", RuntimeError),
                 ("timeout", "time.sleep(20)", subprocess.TimeoutExpired)]
        for index, (name, mode, error) in enumerate(cases):
            with self.subTest(name=name):
                script = self.workspace.root / f"evaluator-{name}.py"
                script.write_text(source.replace("MODE", mode))
                evaluator = CommandEvaluator([sys.executable, str(script)], timeout=0.4)
                with patch.dict(os.environ, {"ADAHVLA_API_KEY": canary, "OPENAI_API_KEY": "synthetic-other-secret"}):
                    if error:
                        with self.assertRaises(error):
                            self.workspace.evaluate(self.baseline, Case("case", "Hold", "table"), evaluator, f"R{index}")
                    else:
                        self.workspace.evaluate(self.baseline, Case("case", "Hold", "table"), evaluator, f"R{index}")
                log = (self.workspace.rollouts / f"R{index}" / "output/process.log").read_text()
                self.assertNotIn(canary, log)
                self.assertIn("[redacted]", log)

    @unittest.skipUnless(os.name == "posix", "Process groups are a POSIX feature")
    def test_timeout_stops_descendants_that_inherit_the_log_pipe(self):
        script = self.workspace.root / "spawn-child.py"
        script.write_text(
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "print('descendant started', flush=True)\n"
            "time.sleep(30)\n"
        )
        evaluator = CommandEvaluator([sys.executable, str(script)], timeout=0.4)
        started = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            self.workspace.evaluate(self.baseline, Case("case", "Hold", "table"), evaluator, "R0001")
        self.assertLess(time.monotonic() - started, 4.0)
        log = (self.workspace.rollouts / "R0001/output/process.log").read_text()
        self.assertIn("descendant started", log)

    def test_patch_changes_only_harness_and_preserves_parent_source(self):
        original = self.workspace.read_source(self.baseline)
        child = self.workspace.apply_patch(self.baseline, "C0001", [{
            "path": "harness.py", "old": "memory_chars: int = 1200", "new": "memory_chars: int = 8",
        }])
        self.assertNotEqual(child.digest, self.baseline.digest)
        self.assertEqual(child.parent_id, self.baseline.id)
        self.assertEqual(self.workspace.read_source(self.baseline), original)
        self.assertEqual(self.workspace.read_source(child, "vla.py"),
                         self.workspace.read_source(self.baseline, "vla.py"))
        self.assertTrue(self.workspace.run_checks(child)["passed"])
        for path in ("vla.py", "../harness.py", "__init__.py"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "Only harness.py"):
                self.workspace.apply_patch(child, "C0002", [{"path": path, "old": "x", "new": "y"}])

    def test_source_and_evidence_tampering_are_detected(self):
        case = Case("case", "Hold the object", "table")
        rollout = self.workspace.evaluate(self.baseline, case, synthetic_evaluator, "R0001")
        Path(rollout.evidence_path).write_text("{}")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            self.workspace.rollout_content(rollout)
        self.baseline.source_path.write_text(self.baseline.source_path.read_text() + "\n# modified\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.workspace.verify(self.baseline)

    def test_evaluator_cannot_change_frozen_source_even_when_it_fails(self):
        def mutate(candidate, case, output):
            source = Path(candidate.path) / "adahvla" / "vla.py"
            source.write_text(source.read_text() + "\n# modified\n")
            raise RuntimeError("Evaluation also failed")

        with self.assertRaisesRegex(ValueError, "source changed"):
            self.workspace.evaluate(self.baseline, Case("case", "Hold the object", "table"), mutate, "R0001")

    def test_evaluator_cannot_mutate_case_metadata_or_redefine_its_protocol(self):
        case = Case("case", "Hold the object", "table", metadata={"camera": {"fps": 4}})

        def mutate(candidate, run_case, output):
            run_case.metadata["camera"]["fps"] = 30
            return synthetic_evaluator(candidate, run_case, output)

        with self.assertRaisesRegex(ValueError, "case definition"):
            self.workspace.evaluate(self.baseline, case, mutate, "R0001")
        self.assertEqual(case.metadata, {"camera": {"fps": 4}})
        script = self.workspace.root / "evaluate.py"
        script.write_text(EVALUATOR_SOURCE)
        evaluator = CommandEvaluator([sys.executable, str(script)])
        script.write_text(EVALUATOR_SOURCE + "\n# protocol changed\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            evaluator.fingerprint()


if __name__ == "__main__":
    unittest.main()
