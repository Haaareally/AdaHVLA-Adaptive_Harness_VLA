import json
import unittest
from copy import deepcopy

from adahvla.harness import ContextPolicy, Frame, Harness, HarnessVLA, Observation


PLAN = [
    {"instruction": "Pick up the object", "completion": "The object is held"},
    {"instruction": "Place it in the container", "completion": "The object rests inside"},
    {"instruction": "Close the container", "completion": "The lid is closed"},
]


def decision(**changes):
    result = {
        "plan": [], "progress": "hold", "mode": "execute",
        "query": "Pick up the object", "evidence": "The object remains on the table",
        "memory": "The container is open",
    }
    result.update(changes)
    return result


def observation(time, *views):
    return Observation(tuple(
        Frame(f"image-{time}-{view}", float(time), view)
        for view in (views or ("ego",))
    ))


class FakeReasoner:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return deepcopy(response)


class FakeVLA:
    def __init__(self, action=None):
        self.action = object() if action is None else action
        self.calls = []

    def predict(self, query, frames):
        self.calls.append((query, list(frames)))
        if isinstance(self.action, Exception):
            raise self.action
        return self.action


class HarnessTests(unittest.TestCase):
    def harness(self, *responses, **kwargs):
        model = FakeReasoner(*responses)
        harness = Harness(model, **kwargs)
        harness.reset("Put the object away and close the container")
        return harness, model

    def test_plan_is_created_once_and_progress_advances_one_goal_per_boundary(self):
        harness, model = self.harness(
            decision(plan=PLAN),
            decision(progress="advance", query="Place the object in the container"),
            decision(progress="advance", query="Close the container"),
            decision(progress="complete", query="", evidence="The lid is closed"),
        )
        results = [harness.step(observation(time)) for time in range(4)]
        self.assertEqual([result.goal_index for result in results], [0, 1, 2, 3])
        self.assertEqual([result.completed for result in results], [False, False, False, True])
        self.assertEqual(len(model.calls), 4)
        inputs = [json.loads(call[1]["content"][0]["text"]) for call in model.calls]
        self.assertEqual(inputs[0]["plan"], [])
        self.assertEqual([value["plan"] for value in inputs[1:]], [PLAN] * 3)
        self.assertIsNone(harness.state.current_goal)
        with self.assertRaisesRegex(RuntimeError, "complete"):
            harness.step(observation(4))
        self.assertEqual(len(model.calls), 4)

    def test_invalid_decisions_cannot_commit_progress_or_observations(self):
        invalid = [
            [],
            {"progress": "hold"},
            decision(progress="skip"),
            decision(progress="advance", mode="recover"),
            decision(progress="advance", evidence=""),
            decision(progress="complete"),
            decision(progress="rollback", rollback_to=0),
            decision(rollback_to=0),
            decision(plan=PLAN[:1]),
            decision(query=""),
            decision(refresh_context="yes"),
        ]
        for response in invalid:
            with self.subTest(response=response):
                harness, _ = self.harness(decision(plan=PLAN), response)
                harness.step(observation(0))
                state, context = harness.state, deepcopy(harness.context)
                with self.assertRaises(ValueError):
                    harness.step(observation(1))
                self.assertIs(harness.state, state)
                self.assertEqual(harness.context, context)

    def test_final_goal_needs_explicit_completion(self):
        harness, _ = self.harness(
            decision(plan=PLAN[:1]), decision(progress="advance"),
            decision(progress="complete", query="", evidence="The object is held"),
        )
        harness.step(observation(0))
        with self.assertRaisesRegex(ValueError, "explicit complete"):
            harness.step(observation(1))
        self.assertTrue(harness.step(observation(1)).completed)

    def test_rollback_discards_later_checkpoints_and_refreshes_executor(self):
        harness, _ = self.harness(
            decision(plan=PLAN), decision(progress="advance"),
            decision(progress="advance"),
            decision(progress="rollback", rollback_to=0, mode="recover",
                     evidence="The object is back on the table"),
        )
        for time in range(3):
            harness.step(observation(time))
        self.assertEqual([item.goal_index for item in harness.context.checkpoints], [0, 1, 2])
        handoff = harness.step(observation(3))
        self.assertEqual(handoff.goal_index, 0)
        self.assertTrue(handoff.reset_context)
        self.assertEqual([item.goal_index for item in harness.context.checkpoints], [0])
        self.assertEqual(harness.context.executor_context(), list(observation(3).frames))

    def test_recovery_refresh_preserves_reasoning_history_then_accumulates_frames(self):
        recovery = decision(mode="recover", query="Regrasp the object")
        harness, model = self.harness(decision(plan=PLAN), recovery, recovery)
        harness.step(observation(0))
        first_recovery = harness.step(observation(1))
        self.assertTrue(first_recovery.reset_context)
        self.assertEqual([frame.timestamp for frame in harness.context.executor_context()], [1])
        self.assertEqual([frame.timestamp for frame in harness.context.recent], [0, 1])
        self.assertEqual(harness.context.checkpoints[0].frames, observation(0).frames)
        continued = harness.step(observation(2))
        self.assertFalse(continued.reset_context)
        self.assertEqual(continued.query, first_recovery.query)
        self.assertEqual([frame.timestamp for frame in harness.context.executor_context()], [1, 2])
        labels = [item["text"] for item in model.calls[-1][1]["content"] if item["type"] == "text"]
        self.assertTrue(any("Historical checkpoint" in label and "t=0s" in label for label in labels))

    def test_reasoning_retains_checkpoints_after_recent_window_expires(self):
        harness, model = self.harness(
            decision(plan=PLAN), decision(),
            context_policy=ContextPolicy(recent_seconds=2),
        )
        harness.step(observation(0))
        harness.step(observation(5))
        self.assertEqual([frame.timestamp for frame in harness.context.recent], [5])
        images = [item["image_url"]["url"] for item in model.calls[-1][1]["content"]
                  if item["type"] == "image_url"]
        self.assertTrue(any("image-0-ego" in image for image in images))
        self.assertTrue(any("image-5-ego" in image for image in images))

    def test_named_views_timestamps_and_physical_feedback_reach_reasoner(self):
        harness, model = self.harness(decision(plan=PLAN))
        obs = Observation(
            observation(1.5, "overhead", "wrist").frames,
            executor_feedback="Gripper closed", proprioception={"gripper_width": 0.02},
        )
        harness.step(obs)
        content = model.calls[0][1]["content"]
        task = json.loads(content[0]["text"])
        self.assertEqual(task["executor_feedback"], "Gripper closed")
        self.assertEqual(task["proprioception"], {"gripper_width": 0.02})
        labels = [item["text"] for item in content[1:] if item["type"] == "text"]
        self.assertEqual(labels, [
            "Recent observation; t=1.5s; view=overhead",
            "Recent observation; t=1.5s; view=wrist",
        ])
        self.assertEqual(harness.context.executor_context(), list(obs.frames))

    def test_sampling_and_refresh_preserve_latest_observation_of_each_view(self):
        harness, model = self.harness(
            decision(plan=PLAN), decision(mode="recover", query="Regrasp the object"),
            context_policy=ContextPolicy(recent_frames=2, executor_frames=4),
        )
        frames = tuple(frame for time in range(3)
                       for frame in observation(time, "ego", "wrist").frames)
        harness.step(Observation(frames))
        latest = observation(2, "ego", "wrist").frames
        self.assertEqual(harness.context.executor_context()[-2:], list(latest))
        self.assertEqual(harness.context.checkpoints[0].frames, latest)
        labels = [item["text"] for item in model.calls[0][1]["content"][1:]
                  if item["type"] == "text"]
        self.assertEqual(labels, [
            "Recent observation; t=2s; view=ego",
            "Recent observation; t=2s; view=wrist",
        ])
        recovery_frames = tuple(frame for time in (3, 4)
                                for frame in observation(time, "ego", "wrist").frames)
        handoff = harness.step(Observation(recovery_frames))
        self.assertTrue(handoff.reset_context)
        self.assertEqual(harness.context.executor_context(), list(observation(4, "ego", "wrist").frames))

    def test_insufficient_multiview_budget_fails_without_committing(self):
        harness, model = self.harness(
            decision(plan=PLAN), context_policy=ContextPolicy(recent_frames=1),
        )
        state = harness.state
        with self.assertRaises(ValueError):
            harness.step(observation(0, "ego", "wrist"))
        self.assertIs(harness.state, state)
        self.assertEqual(harness.context.recent, [])
        self.assertEqual(model.calls, [])

    def test_episode_reset_clears_plan_memory_and_all_visual_history(self):
        harness, model = self.harness(decision(plan=PLAN), decision(plan=PLAN[:1]))
        harness.step(observation(10))
        self.assertTrue(harness.state.memory)
        harness.reset("Pick up the new object")
        self.assertEqual(harness.state.memory, "")
        self.assertEqual(harness.state.plan, ())
        self.assertEqual(harness.context.recent, [])
        self.assertEqual(harness.context.executor, [])
        self.assertEqual(harness.context.checkpoints, [])
        harness.step(observation(0))
        inputs = json.loads(model.calls[-1][1]["content"][0]["text"])
        self.assertEqual(inputs["task"], "Pick up the new object")
        self.assertEqual(inputs["memory"], "")

    def test_reasoner_failure_can_retry_same_observation_without_partial_commit(self):
        failures = [TimeoutError("Reasoner unavailable"), json.JSONDecodeError("Invalid JSON", "{", 1)]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                harness, model = self.harness(failure, decision(plan=PLAN))
                state, context = harness.state, deepcopy(harness.context)
                with self.assertRaises(type(failure)):
                    harness.step(observation(1))
                self.assertIs(harness.state, state)
                self.assertEqual(harness.context, context)
                harness.step(observation(1))
                self.assertEqual(model.calls[0], model.calls[1])
                self.assertEqual(harness.state.step, 1)

    def test_out_of_order_observations_fail_before_reasoning(self):
        harness, model = self.harness(decision(plan=PLAN))
        harness.step(observation(2))
        with self.assertRaisesRegex(ValueError, "time order"):
            harness.step(observation(1))
        self.assertEqual(len(model.calls), 1)


class RuntimeTests(unittest.TestCase):
    def test_vla_owns_actions_and_is_not_called_after_completion(self):
        reasoner = FakeReasoner(
            decision(plan=PLAN[:1]),
            decision(progress="complete", query="", evidence="The object is held"),
        )
        action = {"joint_targets": [0.1, 0.2], "gripper": 1}
        vla = FakeVLA(action)
        runtime = HarnessVLA(Harness(reasoner), vla)
        runtime.reset("Pick up the object")
        result = runtime.step(observation(0))
        self.assertIs(result.action, action)
        self.assertEqual(vla.calls[0], (result.handoff.query, list(observation(0).frames)))
        completed = runtime.step(observation(1))
        self.assertTrue(completed.handoff.completed)
        self.assertIsNone(completed.action)
        self.assertEqual(len(vla.calls), 1)

    def test_vla_failure_does_not_commit_an_unexecuted_handoff(self):
        reasoner = FakeReasoner(decision(plan=PLAN), decision(plan=PLAN))
        vla = FakeVLA(ConnectionError("Executor unavailable"))
        harness = Harness(reasoner)
        runtime = HarnessVLA(harness, vla)
        runtime.reset("Put the object away")
        state, context = harness.state, deepcopy(harness.context)
        with self.assertRaises(ConnectionError):
            runtime.step(observation(0))
        self.assertIs(harness.state, state)
        self.assertEqual(harness.context, context)
        vla.action = object()
        self.assertIs(runtime.step(observation(0)).action, vla.action)
        self.assertEqual(harness.state.step, 1)


if __name__ == "__main__":
    unittest.main()
