"""Adapt harness source between rollouts using separate evidence and revision contexts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .harness import Reasoner
from .workspace import Candidate, Case, Rollout, Workspace, atomic_json, redact_secrets


MANAGER_PROMPT = """You schedule embodied harness adaptation across tasks and environments.
Choose exactly one available tool using the supplied records and remaining budgets.
Evaluate before revising, and compare a child against its actual parent on the same case.
Investigate one general mechanism at a time; keep observed successes under regression checks.
Code review only permits a rollout. Behavior assessments and environment measurements guide
selection. Prefer task success, then targeted effects, then overall behavior. Repeat uncertain
experiments; do not treat one success as proof of reliability. Never use held-out evidence to adapt.
Return {"tool": "...", "arguments": {...}, "rationale": "..."}.
"""

ANALYSIS_PROMPT = """You analyze one embodied rollout, independently of implementation ideas.
Read the supplied ordered events, measurements and visual evidence. Distinguish observed facts
from uncertain explanations. Cite the supplied rollout ID; use event/time references in facts.
Images only cover the supplied frames, not an unseen complete video. Missing observations are
uncertainty, not evidence of success or failure. Do not prescribe code or benchmark-specific fixes.
Return {"facts": ["..."], "failure_signature": "...", "questions": ["..."],
"evidence_refs": ["rollout ID"]}.
"""

ENGINEERING_PROMPT = """You revise reusable embodied coordination code from factual evidence.
Inspect source before proposing one mechanism. Preserve VLA ownership of all physical actions.
Only harness.py is editable; the VLA, evaluator, measurements and adaptation controller are fixed.
Do not add scene names, task IDs, exact instruction strings or other benchmark-specific triggers.
Keep investigation and implementation in this context. Read-only tools are read_source(path)
and search_source(pattern). Return {"tool": "...", "arguments": {...}} for each call.
When ready use tool="propose" with arguments containing nonempty strings hypothesis, mechanism,
expected_effect, observable_signal, falsifier, control_surface, protected_behavior, state_lifecycle;
source_refs (source locations), evidence_refs (supplied rollout IDs), and edits (an array of
{path: "harness.py", old: "unique exact source text", new: "replacement"}). Each edit must match
exactly once. The workflow applies the patch, runs checks, and obtains independent source review.
Use tool="abstain", arguments={"reason": "..."} when evidence cannot support a useful revision.
"""

SOURCE_REVIEW_PROMPT = """Independently review a proposed harness revision before execution.
Compare the stated mechanism, actual diff, source and checks. Reject confirmed interface/runtime
breakage, VLA action ownership violations, unclosed state lifecycle, benchmark-specific shortcuts,
or a material mismatch between the proposed mechanism and implementation. Do not predict rollout
quality or redesign the patch. Eligibility is not behavioral improvement.
Return {"eligible": true, "reason": "..."}, or eligible=false with a concrete blocker.
"""

ASSESSMENT_PROMPT = """Independently compare parent and child embodied rollouts on the same case.
The proposed mechanism is a hypothesis, not evidence. Read both ordered event logs, measurements
and supplied images. Report whether the mechanism actually ran, whether trajectories diverged
before its trigger, its targeted effect and the overall effect. Distinguish a patch-caused serious
regression from a downstream failure exposed by farther progress. Cite both rollout IDs.
Do not invent observations outside the supplied evidence or overwrite environment success metrics.
Return {"mechanism_triggered": "yes|no|uncertain", "pre_trigger_divergence": "yes|no|uncertain",
"target_effect": "improved|unchanged|worse|uncertain",
"overall_effect": "improved|noninferior|worse|uncertain", "observations": "...",
"hard_regressions": ["..."], "new_failure": "...", "evidence_refs": ["parent ID", "child ID"]}.
"""


@dataclass(frozen=True)
class AdaptationPolicy:
    max_steps: int = 40
    max_candidates: int = 12
    max_rollouts: int = 32
    width: int = 2
    depth: int = 3
    engineer_steps: int = 8
    memory_items: int = 12

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("Adaptation budgets must be positive integers")


TOOLS = {
    "evaluate": "(candidate_id, case_id): run/repeat a reviewed source revision; evaluate its parent on this case first.",
    "revise": "(parent_id, rollout_id): analyze a focus-case rollout, investigate source, patch, check and review.",
    "select": "(candidate_id): select using comparable focus, validation and previously successful cases.",
    "discard": "(candidate_id): discontinue a branch and record attributable negative evidence.",
    "switch_task": "(case_id): move to a training case; retain source and experience, reset local search limits.",
    "start_test": "(): freeze the selected revision; run held-out cases without further model decisions.",
    "stop": "(): finish adaptation and keep the selected revision.",
}


def _text(record: dict, key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be nonempty text")
    return value.strip()


def _strings(record: dict, key: str, *, nonempty: bool = False) -> list[str]:
    value = record.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{key} must be a list of nonempty strings")
    if nonempty and not value:
        raise ValueError(f"{key} cannot be empty")
    return value


class Adaptation:
    """Single-writer experiment controller; all source and evidence I/O lives in Workspace."""

    def __init__(
        self, workspace: Workspace, cases: list[Case], evaluator: Any, *,
        manager: Reasoner, analyst: Reasoner, engineer: Reasoner, reviewer: Reasoner,
        policy: AdaptationPolicy | None = None,
    ):
        self.workspace, self.evaluator = workspace, evaluator
        self.manager, self.analyst, self.engineer, self.reviewer = manager, analyst, engineer, reviewer
        self.policy = policy or AdaptationPolicy()
        self.cases = {case.id: Case(**json.loads(json.dumps(asdict(case)))) for case in cases}
        if len(self.cases) != len(cases) or not any(case.split == "train" for case in cases):
            raise ValueError("Cases need unique IDs and at least one training case")
        self.path = workspace.root / "adaptation.json"
        fingerprint = getattr(evaluator, "fingerprint", lambda: {"adapter": type(evaluator).__qualname__})
        self.contract = {"version": 1, "cases": [asdict(case) for case in cases],
                         "policy": asdict(self.policy), "evaluator": fingerprint()}
        baseline = workspace.baseline()
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            if saved["contract"] != self.contract:
                raise ValueError("Resume must preserve cases and adaptation policy")
            self.state = saved["state"]
            if self.candidate("C0000").digest != baseline.digest:
                raise ValueError("Resume baseline changed")
            for value in self.state["candidates"].values():
                workspace.verify(Candidate(**value))
        else:
            focus = next(case.id for case in cases if case.split == "train")
            self.state = {
                "phase": "adapt", "selected": baseline.id, "focus": focus, "focus_round": 0,
                "focus_root": baseline.id, "steps": 0, "candidate_attempts": 0, "rollout_attempts": 0,
                "candidates": {baseline.id: asdict(baseline)}, "rollouts": {}, "comparison_parents": {}, "analyses": {},
                "hypotheses": {}, "revisions": {}, "effects": {}, "events": [],
                "test_rollouts": {}, "test_attempts": 0, "frozen": None,
            }
            self._save()

    @property
    def selected(self) -> Candidate:
        return self.candidate(self.state["selected"])

    def candidate(self, identifier: str) -> Candidate:
        return Candidate(**self.state["candidates"][identifier])

    def rollout(self, identifier: str) -> Rollout:
        return Rollout(**self.state["rollouts"][identifier])

    def _save(self) -> None:
        atomic_json(self.path, {"contract": self.contract, "state": self.state})

    def _event(self, action: str, **details: Any) -> None:
        self.state["events"].append({"action": action, **details})
        self._save()

    def _adapting(self) -> None:
        if self.state["phase"] != "adapt":
            raise ValueError("Source adaptation is closed once testing or completion begins")

    def _latest(self, candidate_id: str, case_id: str) -> Rollout | None:
        values = [value for value in self.state["rollouts"].values()
                  if value["candidate_id"] == candidate_id and value["case_id"] == case_id]
        return Rollout(**values[-1]) if values else None

    def _check_protocol(self) -> None:
        fingerprint = getattr(self.evaluator, "fingerprint", lambda: {"adapter": type(self.evaluator).__qualname__})
        if fingerprint() != self.contract["evaluator"]:
            raise ValueError("The evaluation protocol changed during adaptation")

    def _check_evidence(self, rollout: Rollout) -> None:
        if self.workspace.read_evidence(rollout)["case"] != asdict(self.cases[rollout.case_id]):
            raise ValueError("Recorded case, environment, seed or protocol changed")

    def _blocked(self, candidate_id: str) -> bool:
        while candidate_id != "C0000":
            revision = self.state["revisions"][candidate_id]
            if revision["status"] in {"rejected", "discarded"}:
                return True
            candidate_id = self.candidate(candidate_id).parent_id
        return False

    def _memory(self) -> list[dict]:
        # Keep provenance in persistent records; only a bounded slice enters role context.
        records = []
        for identifier, revision in self.state["revisions"].items():
            hypothesis = self.state["hypotheses"][revision["hypothesis_id"]]
            effects = [effect for effect in self.state["effects"].values() if effect["candidate_id"] == identifier]
            records.append({"candidate_id": identifier, "parent_id": self.candidate(identifier).parent_id,
                            "case_id": revision["case_id"], "status": revision["status"],
                            "hypothesis": hypothesis, "review": revision.get("review"), "effects": effects[-3:]})
        return records[-self.policy.memory_items:]

    def _messages(self, prompt: str, packet: dict, rollouts: tuple[Rollout, ...] = ()) -> list[dict]:
        content = [{"type": "text", "text": json.dumps(packet, ensure_ascii=False)}]
        for rollout in rollouts:
            content.append({"type": "text", "text": f"Observed rollout {rollout.id}"})
            content.extend(self.workspace.rollout_content(rollout))
        return [{"role": "system", "content": prompt}, {"role": "user", "content": content}]

    @staticmethod
    def _refs(record: dict, allowed: set[str]) -> None:
        if not set(_strings(record, "evidence_refs", nonempty=True)) <= allowed:
            raise ValueError("Evidence references must identify supplied rollouts")

    def _ask(self, client: Reasoner, prompt: str, packet: dict, rollouts: tuple[Rollout, ...] = ()) -> dict:
        reply = client.complete(self._messages(prompt, packet, rollouts))
        if not isinstance(reply, dict):
            raise ValueError("Adaptation roles must return a JSON object")
        return reply

    def revise(self, parent_id: str, rollout_id: str) -> Candidate | None:
        """Evidence analysis → source investigation/edit → checks → independent review."""
        self._adapting()
        parent, evidence = self.candidate(parent_id), self.rollout(rollout_id)
        if evidence.candidate_id != parent_id or evidence.case_id != self.state["focus"]:
            raise ValueError("Revision needs evidence from this parent on the current training case")
        if self._blocked(parent_id):
            raise ValueError("Cannot extend a rejected or discontinued branch")
        self._check_evidence(evidence)
        if parent_id != "C0000":
            effect = next((item for item in self.state["effects"].values()
                           if item["child_rollout_id"] == rollout_id), None)
            if effect is None:
                raise ValueError("Assess this parent rollout before adding another patch")
            if effect["target_effect"] != "improved" and not effect["new_failure"].strip():
                raise ValueError("Extension needs targeted improvement or a newly exposed failure")
        current = [revision for revision in self.state["revisions"].values()
                   if revision["focus_round"] == self.state["focus_round"]
                   and not self._blocked(revision["candidate_id"])]
        leaves = [revision for revision in current if revision["status"] not in {"discarded", "rejected"}
                  and not any(other["parent_id"] == revision["candidate_id"] and other["status"] not in {"discarded", "rejected"}
                              for other in current)]
        extending_leaf = any(revision["candidate_id"] == parent_id for revision in leaves)
        if len(leaves) >= self.policy.width and not extending_leaf:
            raise ValueError("Current task search width is exhausted")
        local_depth, cursor = 1, parent_id
        while cursor in self.state["revisions"]:
            revision = self.state["revisions"][cursor]
            if revision["focus_round"] != self.state["focus_round"]:
                break
            local_depth += 1
            cursor = revision["parent_id"]
        if local_depth > self.policy.depth:
            raise ValueError("Current task search depth is exhausted")
        if self.state["candidate_attempts"] >= self.policy.max_candidates:
            raise ValueError("Candidate budget is exhausted")
        self.state["candidate_attempts"] += 1
        number = self.state["candidate_attempts"]
        identifier, analysis_id, hypothesis_id = f"C{number:04}", f"A{number:04}", f"H{number:04}"
        self._event("revision_started", candidate_id=identifier, parent_id=parent_id, rollout_id=rollout_id)

        analysis = self._ask(self.analyst, ANALYSIS_PROMPT, {"case": asdict(self.cases[evidence.case_id])}, (evidence,))
        _strings(analysis, "facts", nonempty=True)
        _strings(analysis, "questions")
        _text(analysis, "failure_signature")
        self._refs(analysis, {rollout_id})
        self.state["analyses"][analysis_id] = {"rollout_id": rollout_id, **analysis}
        self._save()

        messages = self._messages(ENGINEERING_PROMPT, {
            "analysis": analysis, "parent_id": parent_id, "source_files": ["harness.py", "vla.py"],
            "experience": self._memory(),
        })
        inspected = False
        plan = None
        for _ in range(self.policy.engineer_steps):
            reply = self.engineer.complete(messages)
            if not isinstance(reply, dict):
                raise ValueError("Engineering response must be a JSON object")
            tool, arguments = reply.get("tool"), reply.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError("Engineering tool arguments must be an object")
            messages.append({"role": "assistant", "content": json.dumps(reply, ensure_ascii=False)})
            if tool == "abstain":
                self._event("revision_abstained", analysis_id=analysis_id, reason=_text(arguments, "reason"))
                return None
            if tool == "propose":
                if not inspected:
                    raise ValueError("Read harness source before proposing a revision")
                plan = arguments
                break
            if tool == "read_source":
                path = arguments.get("path", "harness.py")
                result = self.workspace.read_source(parent, path)
                inspected = inspected or path == "harness.py"
            elif tool == "search_source":
                result = self.workspace.search_source(parent, _text(arguments, "pattern"))
            else:
                raise ValueError(f"Unknown engineering tool: {tool}")
            messages.append({"role": "user", "content": json.dumps({"tool": tool, "result": result}, ensure_ascii=False)})
        if plan is None:
            raise ValueError("Engineering call budget exhausted without a proposal")
        for key in ("hypothesis", "mechanism", "expected_effect", "observable_signal", "falsifier",
                    "control_surface", "protected_behavior", "state_lifecycle"):
            _text(plan, key)
        _strings(plan, "source_refs", nonempty=True)
        self._refs(plan, {rollout_id})
        hypothesis = {key: value for key, value in plan.items() if key != "edits"}
        hypothesis.update(analysis_id=analysis_id, support=0)
        self.state["hypotheses"][hypothesis_id] = hypothesis
        self._save()
        child = self.workspace.apply_patch(parent, identifier, plan.get("edits"))
        self.state["candidates"][child.id] = asdict(child)
        revision = {"candidate_id": child.id, "parent_id": parent_id, "analysis_id": analysis_id,
                    "hypothesis_id": hypothesis_id, "case_id": evidence.case_id,
                    "focus_round": self.state["focus_round"], "status": "rejected", "review": None}
        self.state["revisions"][child.id] = revision
        self._save()
        checks = self.workspace.run_checks(child)
        revision["checks"] = checks
        if checks["passed"]:
            review = self._ask(self.reviewer, SOURCE_REVIEW_PROMPT, {
                "plan": hypothesis, "diff": self.workspace.diff(parent, child),
                "source": self.workspace.read_source(child), "checks": checks,
            })
            if type(review.get("eligible")) is not bool:
                raise ValueError("Source review requires a boolean eligible field")
            _text(review, "reason")
            revision["review"] = review
            revision["status"] = "eligible" if review["eligible"] else "rejected"
        self.workspace.verify(child)
        self._event("revision_reviewed", candidate_id=child.id, status=revision["status"])
        return child

    def evaluate(self, candidate_id: str, case_id: str) -> Rollout:
        """Execute a sealed revision, then assess it against the preselected parent run."""
        self._adapting()
        candidate, case = self.candidate(candidate_id), self.cases[case_id]
        self._check_protocol()
        if case.split == "test":
            raise ValueError("Held-out cases cannot provide adaptation evidence")
        if self._blocked(candidate_id):
            raise ValueError("Candidate is not eligible for evaluation")
        parent_run = self._latest(candidate.parent_id, case_id) if candidate.parent_id else None
        if candidate.parent_id and parent_run is None:
            raise ValueError("Evaluate the actual parent on this case before its child")
        if self.state["rollout_attempts"] >= self.policy.max_rollouts:
            raise ValueError("Adaptation rollout budget is exhausted")
        self.state["rollout_attempts"] += 1
        identifier = f"R{self.state['rollout_attempts']:04}"
        # Reserve the budget before starting an external process, including failed attempts.
        self._event("evaluation_started", rollout_id=identifier, candidate_id=candidate_id, case_id=case_id)
        result = self.workspace.evaluate(candidate, case, self.evaluator, identifier)
        self.state["rollouts"][result.id] = asdict(result)
        if parent_run is not None:
            self.state["comparison_parents"][result.id] = parent_run.id
        self._event("evaluation_completed", rollout_id=result.id)
        if parent_run is not None:
            self.assess(parent_run.id, result.id)
        return result

    def assess(self, parent_rollout_id: str, child_rollout_id: str) -> dict:
        """Record behavior separately from source review and measured task success."""
        self._adapting()
        parent, child = self.rollout(parent_rollout_id), self.rollout(child_rollout_id)
        candidate = self.candidate(child.candidate_id)
        if candidate.parent_id != parent.candidate_id or parent.case_id != child.case_id:
            raise ValueError("Comparison requires direct lineage and the same case/seed/protocol")
        if self.state["comparison_parents"].get(child.id) != parent.id:
            raise ValueError("Comparison must use the parent rollout selected before child evaluation")
        if parent.source_digest != self.candidate(parent.candidate_id).digest or child.source_digest != candidate.digest:
            raise ValueError("Comparison source fingerprints do not match")
        self._check_evidence(parent)
        self._check_evidence(child)
        previous = next((effect for effect in self.state["effects"].values()
                         if effect["child_rollout_id"] == child_rollout_id), None)
        if previous:
            return previous
        revision = self.state["revisions"][candidate.id]
        hypothesis = self.state["hypotheses"][revision["hypothesis_id"]]
        report = self._ask(self.reviewer, ASSESSMENT_PROMPT, {
            "case": asdict(self.cases[child.case_id]), "plan": hypothesis,
        }, (parent, child))
        for key in ("mechanism_triggered", "pre_trigger_divergence"):
            if report.get(key) not in {"yes", "no", "uncertain"}:
                raise ValueError(f"Invalid {key}")
        if report.get("target_effect") not in {"improved", "unchanged", "worse", "uncertain"}:
            raise ValueError("Invalid targeted effect")
        if report.get("overall_effect") not in {"improved", "noninferior", "worse", "uncertain"}:
            raise ValueError("Invalid overall effect")
        _text(report, "observations")
        _strings(report, "hard_regressions")
        if not isinstance(report.get("new_failure"), str):
            raise ValueError("new_failure must be text")
        self._refs(report, {parent.id, child.id})
        if set(report["evidence_refs"]) != {parent.id, child.id}:
            raise ValueError("Assessment must cite both rollouts")
        # Attribution gates hypothesis support, not the environment's measured success.
        chi = report["mechanism_triggered"] == "yes" and report["pre_trigger_divergence"] == "no"
        effect = {**report, "id": f"E{len(self.state['effects']) + 1:04}", "candidate_id": candidate.id,
                  "case_id": child.case_id, "parent_rollout_id": parent.id, "child_rollout_id": child.id,
                  "chi": chi, "weakened": False}
        self.state["effects"][effect["id"]] = effect
        revision["status"] = "assessed" if revision["status"] != "discarded" else "discarded"
        if chi and effect["target_effect"] == "improved":
            hypothesis["support"] += 1
        if effect["hard_regressions"] or revision["status"] == "discarded":
            self.discard(candidate.id)
        elif self.state["selected"] == candidate.id and effect["overall_effect"] == "worse":
            self.state["selected"] = "C0000"
        self._event("effect_recorded", effect_id=effect["id"], chi=chi)
        return effect

    def discard(self, candidate_id: str) -> None:
        self._adapting()
        if candidate_id == "C0000":
            raise ValueError("The baseline cannot be discarded")
        revision = self.state["revisions"][candidate_id]
        revision["status"] = "discarded"
        for effect in self.state["effects"].values():
            if effect["candidate_id"] == candidate_id and effect["chi"] and not effect["weakened"]:
                if effect["target_effect"] in {"unchanged", "worse"}:
                    self.state["hypotheses"][revision["hypothesis_id"]]["support"] -= 1
                    effect["weakened"] = True
        if self._blocked(self.state["selected"]):
            self.state["selected"] = "C0000"
        self._event("candidate_discarded", candidate_id=candidate_id)

    def _score(self, candidate_id: str, cases: set[str]) -> tuple:
        rollouts = [self._latest(candidate_id, case_id) for case_id in cases]
        if any(rollout is None for rollout in rollouts):
            raise ValueError("Selection needs matched evaluation cases for both candidates")
        ids = {rollout.id for rollout in rollouts}
        effects = [effect for effect in self.state["effects"].values() if effect["child_rollout_id"] in ids]
        target = {"improved": 1, "unchanged": 0, "uncertain": 0, "worse": -1}
        overall = {"improved": 1, "noninferior": 0, "uncertain": 0, "worse": -1}
        return (sum(rollout.success for rollout in rollouts),
                sum(target[effect["target_effect"]] for effect in effects),
                sum(overall[effect["overall_effect"]] for effect in effects))

    def select(self, candidate_id: str) -> Candidate:
        """Compare focus, validation and preservation cases before changing the selected source."""
        self._adapting()
        candidate = self.candidate(candidate_id)
        if candidate_id == self.state["selected"]:
            return candidate
        if self._blocked(candidate_id):
            raise ValueError("Cannot select a blocked branch")
        effects = [effect for effect in self.state["effects"].values() if effect["candidate_id"] == candidate_id]
        latest_effects = {effect["case_id"]: effect for effect in effects}
        if candidate_id != "C0000" and (not effects or any(
            effect["overall_effect"] in {"worse", "uncertain"} for effect in latest_effects.values()
        ) or any(effect["hard_regressions"] for effect in effects)):
            raise ValueError("Selection requires independent non-adverse behavioral evidence")
        required = {self.state["focus"]} | {case.id for case in self.cases.values() if case.split == "validation"}
        required |= {value["case_id"] for value in self.state["rollouts"].values()
                     if value["candidate_id"] == self.state["selected"] and value["success"]}
        for case_id in required:
            child, selected = self._latest(candidate_id, case_id), self._latest(self.state["selected"], case_id)
            if child is None or selected is None:
                raise ValueError("Evaluate both candidates on focus, validation and preservation cases")
            if selected.success and not child.success:
                raise ValueError("Candidate regressed on a demonstrated successful case")
            if candidate_id != "C0000" and not any(effect["child_rollout_id"] == child.id for effect in effects):
                raise ValueError("Assess the latest rollout before selection")
        score, reference = self._score(candidate_id, required), self._score(self.state["selected"], required)
        if candidate.parent_id == self.state["selected"]:
            # These effects already compare the child to the selected source, whose delta is zero.
            reference = (reference[0], 0, 0)
        elif candidate.parent_id != self.selected.parent_id and score[0] == reference[0]:
            raise ValueError("Equal-success branches need a common comparison parent or intermediate selection")
        if score <= reference:
            raise ValueError("Candidate does not improve the matched selection ordering")
        self.workspace.verify(candidate)
        self.state["selected"] = candidate_id
        self._event("candidate_selected", candidate_id=candidate_id, cases=sorted(required))
        return candidate

    def switch_task(self, case_id: str) -> None:
        self._adapting()
        if self.cases[case_id].split != "train":
            raise ValueError("Only training cases can become adaptation focus")
        if case_id == self.state["focus"]:
            raise ValueError("Already focused on this case")
        self.state.update(focus=case_id, focus_round=self.state["focus_round"] + 1, focus_root=self.state["selected"])
        self._event("task_switched", case_id=case_id, selected=self.state["selected"])

    def start_test(self) -> None:
        """Freeze the selected source, including after adaptation has already stopped."""
        if self.state["phase"] not in {"adapt", "complete"} or self.state["frozen"] is not None:
            raise ValueError("Held-out evaluation can only start once")
        if not any(case.split == "test" for case in self.cases.values()):
            raise ValueError("No held-out cases were supplied")
        self.workspace.verify(self.selected)
        self._check_protocol()
        self.state.update(phase="test", frozen={"candidate_id": self.selected.id, "digest": self.selected.digest})
        self._event("test_started", **self.state["frozen"])

    def stop(self) -> None:
        self._adapting()
        self.state["phase"] = "complete"
        self._event("adaptation_stopped", selected=self.state["selected"])

    def available_tools(self) -> dict[str, str]:
        if self.state["phase"] != "adapt":
            return {}
        names = {"select", "discard", "switch_task", "stop"}
        if any(case.split == "test" for case in self.cases.values()):
            names.add("start_test")
        if self.state["rollout_attempts"] < self.policy.max_rollouts:
            names.add("evaluate")
        if self.state["candidate_attempts"] < self.policy.max_candidates:
            names.add("revise")
        return {name: description for name, description in TOOLS.items() if name in names}

    def step(self) -> dict:
        if self.state["phase"] == "complete":
            raise ValueError("Adaptation is complete")
        if self.state["phase"] == "test":
            # Held-out results are archived separately and never enter any role context.
            frozen = self.state["frozen"]
            if self.selected.id != frozen["candidate_id"] or self.selected.digest != frozen["digest"]:
                raise ValueError("Held-out source selection changed")
            pending = [case for case in self.cases.values() if case.split == "test" and case.id not in self.state["test_rollouts"]]
            if not pending:
                self.state["phase"] = "complete"
                self._save()
                return {"phase": "complete", "selected": self.selected.id}
            case = pending[0]
            self._check_protocol()
            self.state["test_attempts"] += 1
            identifier = f"T{self.state['test_attempts']:04}"
            self._save()
            result = self.workspace.evaluate(self.selected, case, self.evaluator, identifier)
            self.state["test_rollouts"][case.id] = asdict(result)
            self._save()
            return {"tool": "test", "rollout": asdict(result)}
        # Finish an interrupted assessment without spending another environment rollout.
        for value in self.state["rollouts"].values():
            candidate = self.candidate(value["candidate_id"])
            assessed = any(effect["child_rollout_id"] == value["id"] for effect in self.state["effects"].values())
            if candidate.parent_id and not assessed:
                parent_id = self.state["comparison_parents"][value["id"]]
                effect = self.assess(parent_id, value["id"])
                return {"tool": "assess", "result": effect}
        if self.state["steps"] >= self.policy.max_steps:
            self.stop()
            return {"phase": "complete", "reason": "manager budget exhausted"}
        tools = self.available_tools()
        packet = {"focus": self.state["focus"], "selected": self.state["selected"],
                  "cases": [asdict(case) for case in self.cases.values() if case.split != "test"],
                  "tools": tools, "experience": self._memory(),
                  "search_limits": {"width": self.policy.width, "depth": self.policy.depth},
                  "rollouts": list(self.state["rollouts"].values()),
                  "recent_events": self.state["events"][-self.policy.memory_items:],
                  "remaining": {"candidates": self.policy.max_candidates - self.state["candidate_attempts"],
                                "rollouts": self.policy.max_rollouts - self.state["rollout_attempts"]}}
        self.state["steps"] += 1
        self._save()
        action = self._ask(self.manager, MANAGER_PROMPT, packet)
        name, arguments = action.get("tool"), action.get("arguments")
        if name not in tools or not isinstance(arguments, dict):
            raise ValueError("Manager selected an unavailable tool or invalid arguments")
        _text(action, "rationale")
        try:
            value = getattr(self, name)(**arguments)
            result = asdict(value) if isinstance(value, (Candidate, Rollout)) else value
            self._event("tool_completed", tool=name, arguments=arguments, rationale=action["rationale"])
            return {"tool": name, "result": result}
        except Exception as error:
            self._event("tool_failed", tool=name, arguments=arguments, error=redact_secrets(str(error)))
            raise

    def run(self, steps: int | None = None) -> Candidate:
        count = 0
        while self.state["phase"] != "complete" and (steps is None or count < steps):
            self.step()
            count += 1
        return self.selected
