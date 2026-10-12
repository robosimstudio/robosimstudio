# SPDX-License-Identifier: Apache-2.0
"""Raw point/span capture is typed, exact, lazy, and contains no prose.

The tape is what replaced the description layer: ``--capture-state`` writes the
numbers a simulator can produce -- indices, arrays, contact pairs, tool spans --
and nothing that interprets them. These six pin the parts that were wrong at
least once: the two views of one tape, the refusal that keeps a span honest on a
decimated capture, what opens a span and what does not, the boundary close, the
absence of the tools this layer replaced, and the step stream capture rides on.
"""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from robosimstudio.core.env import EnvBase
from robosimstudio.record import EpisodeCapture, ToolSpanRecorder
from robosimstudio.specs import CaptureSpec
from robosimstudio_bench.envs.env import Env
from robosimstudio_policy_agent.sim_tools import SimTools

TIER = "fast"


def _episode(path: Path) -> None:
    (path / "capture_manifest.json").write_text(
        json.dumps({"schema": "robosimstudio.capture.v1", "entities": {}, "arms": {}})
    )
    np.savez_compressed(
        path / "capture.npz",
        step=np.array([0, 1, 2], dtype=np.int64),
        sim_time=np.array([0.0, 0.02, 0.04]),
        wall_time=np.array([10.0, 10.02, 10.04]),
        body_q=np.zeros((3, 2, 7), dtype=np.float32),
    )
    (path / "events.jsonl").write_text(
        json.dumps({"event": "episode_start", "step": 0})
        + "\n"
        + json.dumps({"event": "episode_end", "step": 2})
        + "\n"
    )
    (path / "spans.jsonl").write_text(
        json.dumps({"span_id": 4, "start_step": 0, "end_step": 2, "tool": "robot.go_to_pose"}) + "\n"
    )
    (path / "contacts.jsonl").write_text(json.dumps({"step": 2, "pairs": [{"shape_a": 1, "shape_b": 2}]}) + "\n")


class TestRawCapture(unittest.TestCase):
    def test_point_and_span_are_two_views_of_one_tape(self) -> None:
        """A span's endpoints and the steps between them come off one tape.

        The whole point of the span interface: a blocking tool runs many control
        steps inside one policy call, so a runner-loop recorder sees only
        endpoints. A span carries its two exact endpoints AND a lazy iterator
        over the motion between them, from the same arrays, with no second file.
        """
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            _episode(path)
            capture = EpisodeCapture(path)
            self.assertEqual(capture.point(event="episode_start").step, 0)
            self.assertEqual(
                capture.point(event="episode_end").contacts,
                [{"shape_a": 1, "shape_b": 2}],
            )
            span = capture.span(4)
            self.assertEqual((span.before.step, span.after.step), (0, 2))
            self.assertEqual([point.step for point in span.points()], [0, 1, 2])
            self.assertEqual([item.id for item in capture.iter_spans()], [4])

    def test_span_markers_require_exact_per_step_capture(self) -> None:
        """A span on a decimated tape would name steps the tape does not hold.

        ``every=2`` and ``tool_spans=True`` is refused at construction rather
        than writing spans whose endpoints land between recorded frames.
        """
        with self.assertRaisesRegex(ValueError, "exact"):
            CaptureSpec(every=2, tool_spans=True)
        self.assertEqual(CaptureSpec(every=2, tool_spans=False).every, 2)

    def test_tool_observer_records_only_state_mutating_calls(self) -> None:
        """Only ``sim_step`` tools open a span; a privileged read is not motion.

        And the span's ``end_step`` is read when the call returns, not when it
        began -- which is the difference between a span and a timestamp.
        """
        step = [3]
        recorder = ToolSpanRecorder(lambda: step[0])
        recorder.reset(episode_id=9)
        self.assertIsNone(recorder.begin("sim.get_object_pose", ("privileged",), object(), {}))
        token = recorder.begin(
            "robot.go_to_pose",
            ("sim_step",),
            types.SimpleNamespace(_node_id="move"),
            {"arm_id": 1},
        )
        step[0] = 17
        recorder.end(token, result={"ok": True})
        row = recorder.rows()[0]
        self.assertEqual(
            (row["start_step"], row["end_step"], row["arm_id"], row["node"]),
            (3, 17, 1, "move"),
        )

    def test_inflight_tool_is_closed_at_the_episode_boundary(self) -> None:
        """An episode that ends mid-tool leaves a closed row, not a dangling one.

        Marked ``interrupted`` and given the boundary as its end step, so a
        reader cannot mistake a truncated move for one that completed.
        """
        step = [5]
        recorder = ToolSpanRecorder(lambda: step[0])
        recorder.reset(episode_id=2)
        recorder.begin("robot.go_to_pose", ("sim_step",), object(), {"arm_id": 0})
        step[0] = 11
        row = recorder.rows()[0]
        self.assertEqual((row["start_step"], row["end_step"]), (5, 11))
        self.assertEqual(row["status"], "interrupted")
        self.assertIn("episode ended", row["error"])

    def test_runtime_has_no_scene_description_tools(self) -> None:
        """``sim.describe``/``sim.diff`` stay off the surface, by design.

        Turning simulator state into English is a research variable this
        checkout took out of the benchmark: the tape keeps the numbers and the
        caller owns the language. A re-added describe tool fails here.
        """
        names = {name for name, _fn, _summary in SimTools.__new__(SimTools).entries()}
        self.assertNotIn("sim.describe", names)
        self.assertNotIn("sim.diff", names)

    def test_environment_observer_sees_every_completed_step(self) -> None:
        """Capture attaches to completed steps, and detaches cleanly.

        The observer is handed the env, the action, the reward and the info of
        every step that finished -- and stops being called the moment it is
        removed, so a stale observer cannot keep writing into a finished tape.
        """

        def completed_step(_self, action, mode):
            return {"step": 1}, 0.25, False, False, {"reward": 0.25, "mode": mode}

        env = Env.__new__(Env)
        env._step_observers = []
        seen = []

        def observer(*args):
            seen.append(args)

        with patch.object(EnvBase, "step", completed_step):
            env.add_step_observer(observer)
            result = env.step([1.0], mode="osc_pose")
            self.assertEqual(result[1], 0.25)
            self.assertEqual(len(seen), 1)
            self.assertIs(seen[0][0], env)
            self.assertEqual(seen[0][1], [1.0])
            self.assertEqual(seen[0][3], 0.25)
            self.assertEqual(seen[0][4]["mode"], "osc_pose")

            env.remove_step_observer(observer)
            env.step([2.0])
            self.assertEqual(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
