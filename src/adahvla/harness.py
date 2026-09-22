"""Embodied coordination: state, context, editable policies, and the harness loop."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from typing import TYPE_CHECKING, Any, Mapping, Protocol
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from .vla import VLA


# State and decision contract.

@dataclass(frozen=True)
class Frame:
    image: str
    timestamp: float
    view: str = "ego"

    def __post_init__(self) -> None:
        if not self.image or not self.view or not isfinite(self.timestamp):
            raise ValueError("A frame requires image data, a view, and a finite timestamp")
        if self.timestamp < 0:
            raise ValueError("Frame timestamps must be nonnegative episode seconds")


@dataclass(frozen=True)
class Observation:
    """New evidence at an execution boundary; feedback describes actual outcomes."""

    frames: tuple[Frame, ...]
    executor_feedback: str = ""
    proprioception: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Goal:
    instruction: str
    completion: str


@dataclass(frozen=True)
class Decision:
    """The model proposes transitions; policies decide whether they can commit."""

    progress: str
    mode: str
    query: str
    evidence: str
    memory: str
    plan: tuple[Goal, ...] = ()
    rollback_to: int | None = None
    refresh_context: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Decision":
        if not isinstance(value, dict):
            raise ValueError("The reasoning response must be a JSON object")
        required = {"progress", "mode", "query", "evidence", "memory"}
        allowed = required | {"plan", "rollback_to", "refresh_context"}
        if not required <= value.keys() or value.keys() - allowed:
            raise ValueError("Decision fields do not match the response contract")
        if any(not isinstance(value[key], str) for key in required):
            raise ValueError("Decision text fields must be strings")
        if value["progress"] not in {"hold", "advance", "rollback", "complete"}:
            raise ValueError("Unknown progress transition")
        if value["mode"] not in {"execute", "recover"}:
            raise ValueError("Unknown execution mode")
        target = value.get("rollback_to")
        if target is not None and type(target) is not int:
            raise ValueError("rollback_to must be an integer or null")
        refresh = value.get("refresh_context", False)
        if type(refresh) is not bool:
            raise ValueError("refresh_context must be a boolean")
        raw_plan = value.get("plan", [])
        if not isinstance(raw_plan, list):
            raise ValueError("plan must be an array")
        goals = []
        for goal in raw_plan:
            if not isinstance(goal, dict) or set(goal) != {"instruction", "completion"}:
                raise ValueError("Each goal requires instruction and completion")
            if any(not isinstance(text, str) or not text.strip() for text in goal.values()):
                raise ValueError("Goal descriptions must be nonempty strings")
            goals.append(Goal(goal["instruction"].strip(), goal["completion"].strip()))
        return cls(
            **{key: value[key].strip() for key in required},
            plan=tuple(goals), rollback_to=target, refresh_context=refresh,
        )


@dataclass(frozen=True)
class HarnessState:
    task: str
    plan: tuple[Goal, ...] = ()
    goal_index: int = 0
    query: str = ""
    mode: str = "execute"
    memory: str = ""
    step: int = 0
    completed: bool = False

    @property
    def current_goal(self) -> Goal | None:
        return self.plan[self.goal_index] if self.goal_index < len(self.plan) else None


@dataclass(frozen=True)
class Handoff:
    query: str
    reset_context: bool
    completed: bool
    goal_index: int
    mode: str
    progress: str
    evidence: str


# Editable embodied reasoning prompt.

DECISION_PROMPT = """\
You coordinate long-horizon embodied execution. A frozen VLA executes your short task
instruction and owns all robot actions. Use observations to assess outcomes; requested
actions and previous beliefs are not proof of physical progress.

On the first call, decompose the task into a short ordered plan of observable goals.
Give each goal an instruction and a visible completion condition. Start at goal 0
with progress="hold". On later calls, keep the plan unchanged and return plan=[].

At each decision boundary, assess the current goal and produce one coherent decision:
- hold: continue this goal, including when evidence is uncertain.
- advance: the current goal is visibly complete; address the next goal.
- rollback: observations contradict an earlier progress commitment; give its goal index.
- complete: the final goal is visibly complete and no earlier obligation remains.
Advance only one goal at a time. Use complete rather than advance at the final goal.
Give concrete observed evidence for every progress change. A related object or state
being visible alone does not prove the required interaction or transition occurred.

Use mode="recover" while correcting a deviation or blockage; otherwise use "execute".
Do not advance or complete while recovery is still needed. Keep a useful local query
stable until its objective is met or no longer appropriate. Following a progress change,
the query must address the resulting goal. Describe an embodied objective, not raw robot
actions. Set refresh_context=true only when old executor observations have become stale;
the harness applies its configured refresh rules at goal and recovery-mode changes.

Earlier checkpoint images provide history, not proof of current completion. Recent
images are chronological and show the latest state. Compare views and time labels.
Memory replaces the previous summary: preserve verified progress, unfinished obligations,
and the latest observed effect within the supplied memory character limit. Do not invent
hidden objects, geometry, actions, or success signals.

Return only a JSON object with these fields:
{
  "plan": [{"instruction": "...", "completion": "..."}],
  "progress": "hold|advance|rollback|complete",
  "rollback_to": null,
  "mode": "execute|recover",
  "query": "one concise local objective, empty only when complete",
  "evidence": "what the observations establish or leave uncertain",
  "memory": "compact updated task memory",
  "refresh_context": false
}
"""


# Visual context and retention policy.

def sample_frames(frames: list[Frame], count: int) -> list[Frame]:
    if count < 1:
        raise ValueError("Frame budget must be positive")
    latest = {frame.view: index for index, frame in enumerate(frames)}
    if len(latest) > count:
        raise ValueError("Frame budget must cover every camera view")
    if len(frames) <= count:
        return list(frames)
    # Preserve the current observation from every camera before sampling history.
    selected = set(latest.values())
    remaining = [index for index in range(len(frames)) if index not in selected]
    slots = count - len(selected)
    if slots == 1:
        selected.add(remaining[0])
    elif slots > 1:
        selected.update(remaining[round(i * (len(remaining) - 1) / (slots - 1))] for i in range(slots))
    return [frames[index] for index in sorted(selected)]


@dataclass(frozen=True)
class ContextPolicy:
    recent_seconds: float = 24.0
    recent_frames: int = 32
    checkpoint_count: int = 4
    executor_frames: int = 16
    max_buffer_frames: int = 512

    def __post_init__(self) -> None:
        if not isfinite(self.recent_seconds) or self.recent_seconds <= 0:
            raise ValueError("recent_seconds must be positive and finite")
        if min(self.recent_frames, self.executor_frames, self.max_buffer_frames) < 1:
            raise ValueError("Frame budgets must be positive")
        if self.checkpoint_count < 0:
            raise ValueError("checkpoint_count cannot be negative")


@dataclass(frozen=True)
class Checkpoint:
    goal_index: int
    frames: tuple[Frame, ...]


@dataclass
class VisualContext:
    """Task-level evidence survives a refresh of the executor's local history."""

    policy: ContextPolicy = field(default_factory=ContextPolicy)
    recent: list[Frame] = field(default_factory=list)
    executor: list[Frame] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)

    def observe(self, observation: Observation) -> None:
        if not observation.frames:
            raise ValueError("A decision boundary requires visual observations")
        frames = list(observation.frames)
        times = [frame.timestamp for frame in frames]
        if times != sorted(times) or (self.recent and times[0] < self.recent[-1].timestamp):
            raise ValueError("Observations must follow episode time order")
        cutoff = times[-1] - self.policy.recent_seconds
        self.recent = [frame for frame in self.recent + frames if frame.timestamp >= cutoff]
        self.recent = sample_frames(self.recent, self.policy.max_buffer_frames)
        self.executor = sample_frames(self.executor + frames, self.policy.max_buffer_frames)

    def checkpoint(self, goal_index: int, frames: tuple[Frame, ...]) -> None:
        self.checkpoints = [item for item in self.checkpoints if item.goal_index < goal_index]
        if self.policy.checkpoint_count:
            current = sample_frames(list(frames), len({frame.view for frame in frames}))
            self.checkpoints.append(Checkpoint(goal_index, tuple(current)))
            self.checkpoints = self.checkpoints[-self.policy.checkpoint_count:]

    def refresh_executor(self, frames: tuple[Frame, ...]) -> None:
        # A local visual refresh must not erase task-level reasoning history.
        self.executor = sample_frames(list(frames), len({frame.view for frame in frames}))

    def executor_context(self) -> list[Frame]:
        return sample_frames(self.executor, self.policy.executor_frames)

    def messages(
        self, state: HarnessState, observation: Observation, *, prompt: str,
        capabilities: str, memory_chars: int,
    ) -> list[dict]:
        task_context = {
            "task": state.task,
            "plan": [asdict(goal) for goal in state.plan],
            "current_goal_index": state.goal_index,
            "current_query": state.query,
            "mode": state.mode,
            "memory": state.memory,
            "memory_character_limit": memory_chars,
            "executor_capabilities": capabilities,
            "executor_feedback": observation.executor_feedback,
            "proprioception": dict(observation.proprioception),
        }
        content = [{"type": "text", "text": json.dumps(task_context, ensure_ascii=False)}]

        def add_frames(frames: list[Frame] | tuple[Frame, ...], label: str) -> None:
            for frame in frames:
                url = frame.image if frame.image.startswith("data:") else "data:image/jpeg;base64," + frame.image
                content.extend([
                    {"type": "text", "text": f"{label}; t={frame.timestamp:g}s; view={frame.view}"},
                    {"type": "image_url", "image_url": {"url": url}},
                ])

        for checkpoint in self.checkpoints:
            add_frames(checkpoint.frames, f"Historical checkpoint, entering goal {checkpoint.goal_index}")
        add_frames(sample_frames(self.recent, self.policy.recent_frames), "Recent observation")
        return [{"role": "system", "content": prompt}, {"role": "user", "content": content}]


# Progress and handoff policies.

@dataclass(frozen=True)
class ProgressPolicy:
    """Editable transition rules, separate from the model's visual judgment."""

    memory_chars: int = 1200

    def __post_init__(self) -> None:
        if self.memory_chars < 1:
            raise ValueError("memory_chars must be positive")

    def apply(self, state: HarnessState, decision: Decision) -> HarnessState:
        plan = state.plan or decision.plan
        if not plan:
            raise ValueError("The first decision must create a task plan")
        if state.plan and decision.plan and decision.plan != state.plan:
            raise ValueError("A decision cannot silently rewrite the episode plan")
        if not state.plan and decision.progress != "hold":
            raise ValueError("Initialize the plan before committing task progress")
        if decision.progress != "rollback" and decision.rollback_to is not None:
            raise ValueError("rollback_to is only valid for a rollback")
        if decision.progress != "hold" and not decision.evidence:
            raise ValueError("Progress changes require observational evidence")
        if decision.mode == "recover" and decision.progress in {"advance", "complete"}:
            raise ValueError("Recovery cannot simultaneously commit forward progress")

        index = state.goal_index
        if decision.progress == "advance":
            if index >= len(plan) - 1:
                raise ValueError("Final-goal completion requires an explicit complete decision")
            index += 1
        elif decision.progress == "rollback":
            if decision.rollback_to is None or not 0 <= decision.rollback_to < index:
                raise ValueError("Rollback must identify an earlier goal")
            index = decision.rollback_to
        elif decision.progress == "complete":
            if index != len(plan) - 1:
                raise ValueError("Cannot complete while later goals remain")
            index = len(plan)
        completed = decision.progress == "complete"
        if not completed and not decision.query:
            raise ValueError("An unfinished task requires an executor query")
        return replace(
            state, plan=plan, goal_index=index, completed=completed,
            query="" if completed else decision.query, mode=decision.mode,
            memory=decision.memory[:self.memory_chars], step=state.step + 1,
        )


@dataclass(frozen=True)
class HandoffPolicy:
    """Choose when a changed objective needs a fresh executor context."""

    refresh_on_goal_change: bool = True
    refresh_on_mode_change: bool = True

    def apply(self, before: HarnessState, after: HarnessState, decision: Decision) -> Handoff:
        reset = (
            decision.refresh_context
            or (self.refresh_on_goal_change and before.goal_index != after.goal_index)
            or (self.refresh_on_mode_change and before.mode != after.mode)
        )
        return Handoff(
            query=after.query, reset_context=reset and not after.completed,
            completed=after.completed, goal_index=after.goal_index, mode=after.mode,
            progress=decision.progress, evidence=decision.evidence,
        )


# Reasoning interface and execution loop.

class JSONReasoner:
    def __init__(
        self, *, base_url: str, model: str, api_key: str,
        timeout: float = 120.0, temperature: float = 0.0,
    ):
        if not api_key or not model:
            raise ValueError("A reasoning model and API key are required")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key = model, api_key
        self.timeout, self.temperature = timeout, temperature

    def complete(self, messages: list[dict]) -> dict:
        payload = json.dumps({
            "model": self.model, "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }).encode()
        request = Request(self.url, data=payload, headers={
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        })
        with urlopen(request, timeout=self.timeout) as response:
            body = json.load(response)
        text = body["choices"][0]["message"]["content"]
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("The reasoning model did not return a JSON object")
        return result


class Reasoner(Protocol):
    def complete(self, messages: list[dict]) -> dict: ...


class Harness:
    """One episode's coordination state; reset before each independent rollout."""

    def __init__(
        self, reasoner: Reasoner, *, capabilities: str = "Execute local embodied task instructions.",
        context_policy: ContextPolicy | None = None,
        progress_policy: ProgressPolicy | None = None,
        handoff_policy: HandoffPolicy | None = None,
        prompt: str = DECISION_PROMPT,
    ):
        self.reasoner = reasoner
        self.capabilities = capabilities
        self.context_policy = context_policy or ContextPolicy()
        self.progress_policy = progress_policy or ProgressPolicy()
        self.handoff_policy = handoff_policy or HandoffPolicy()
        self.prompt = prompt
        self.state: HarnessState | None = None
        self.context = VisualContext(self.context_policy)

    def reset(self, task: str) -> None:
        if not task.strip():
            raise ValueError("An episode requires a task instruction")
        self.state = HarnessState(task=task.strip())
        self.context = VisualContext(self.context_policy)

    def step(self, observation: Observation) -> Handoff:
        if self.state is None:
            raise RuntimeError("Call reset(task) before the first observation")
        if self.state.completed:
            raise RuntimeError("The episode is complete; reset before a new task")
        before = self.state
        context = deepcopy(self.context)
        context.observe(observation)
        decision = Decision.from_dict(self.reasoner.complete(context.messages(
            before, observation, prompt=self.prompt, capabilities=self.capabilities,
            memory_chars=self.progress_policy.memory_chars,
        )))
        after = self.progress_policy.apply(before, decision)
        handoff = self.handoff_policy.apply(before, after, decision)
        if not after.completed and (not before.plan or before.goal_index != after.goal_index):
            context.checkpoint(after.goal_index, observation.frames)
        if handoff.reset_context:
            context.refresh_executor(observation.frames)
        # Invalid model responses leave both state and visual memory untouched.
        self.state, self.context = after, context
        return handoff


@dataclass(frozen=True)
class ExecutionStep:
    handoff: Handoff
    action: Any = None


class HarnessVLA:
    """Predict actions under the harness; the environment owns physical execution."""

    def __init__(self, harness: Harness, vla: VLA):
        self.harness, self.vla = harness, vla

    def reset(self, task: str) -> None:
        self.harness.reset(task)

    def step(self, observation: Observation) -> ExecutionStep:
        before = self.harness.state, self.harness.context
        try:
            handoff = self.harness.step(observation)
            if handoff.completed:
                return ExecutionStep(handoff)
            action = self.vla.predict(handoff.query, self.harness.context.executor_context())
            return ExecutionStep(handoff, action)
        except Exception:
            # Prediction has no environment side effects; failed handoffs do not commit.
            self.harness.state, self.harness.context = before
            raise
