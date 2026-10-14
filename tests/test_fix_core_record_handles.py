"""Regression tests: ragged step keys, restricted contact views in capture,
trace summaries over metrics absent from early rows, bodiless static poses."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.probes.contact import ContactView
from robosimstudio.record import recorder as recorder_module
from robosimstudio.record.recorder import EpisodeRecorder, _stack_steps
from robosimstudio.record.trace import summarise
from robosimstudio.scene.handles import EntityHandle
from robosimstudio.specs import CaptureSpec, RecordSpec

TIER = "fast"


class StackStepsTest(unittest.TestCase):
    def test_held_steps_pad_the_action(self) -> None:
        steps = {
            "action": [np.zeros(0, np.float32), np.ones(7, np.float32), np.ones(3, np.float32)],
            "reward": [np.float32(0.0)] * 3,
        }
        arrays = _stack_steps(steps)
        self.assertEqual(arrays["action"].shape, (3, 7))
        np.testing.assert_array_equal(arrays["action_len"], [0, 7, 3])
        self.assertTrue(np.isnan(arrays["action"][0]).all())
        self.assertTrue(np.isnan(arrays["action"][2, 3:]).all())
        np.testing.assert_array_equal(arrays["action"][1], np.ones(7))
        self.assertEqual(arrays["reward"].shape, (3,))
        self.assertNotIn("reward_len", arrays)

    def test_episode_with_held_steps_is_written(self) -> None:
        root = tempfile.mkdtemp()
        rec = EpisodeRecorder(
            root, RecordSpec(record_video=False, privileged=False), capture=CaptureSpec(enabled=False)
        )
        rec._choose_particles = lambda env: None
        rec._record_anchors = lambda env: None
        original = recorder_module.episode_row
        recorder_module.episode_row = lambda *a, **k: {"recorded_steps": k.get("recorded_steps")}
        try:
            env = SimpleNamespace(step_count=0, sim_time=0.0, episode=SimpleNamespace(episode_id=0, seed=0), task=None)
            rec.begin(SimpleNamespace(episode_id=0, seed=0), env)
            for action in (None, np.ones(7), np.ones(7)):
                env.step_count += 1
                env.sim_time += 0.01
                rec.step(env, action, {}, 0.0, {})
            rec.end(env, {})
        finally:
            recorder_module.episode_row = original
        self.assertTrue(rec.enabled, rec.failures)
        with np.load(Path(root) / "0000" / "steps.npz") as data:
            self.assertEqual(data["action"].shape, (3, 7))


class CaptureRestrictedContactsTest(unittest.TestCase):
    def test_restricted_view_is_read_without_update(self) -> None:
        view = ContactView(
            geoms=np.array([[0, 1]], dtype=np.int32),
            impulses=np.array([[0.0, 0.0, 1.0]]),
            depths=np.array([0.001]),
            geom_to_shape=np.array([4, 5], dtype=np.int32),
            dt=0.01,
        )
        context = SimpleNamespace(
            contacts=view,
            body_q=np.zeros((1, 7)),
            body_qd=np.zeros((1, 6)),
            settled_body_q=np.zeros((1, 7)),
            joint_target_q=np.zeros(2),
        )
        snap = SimpleNamespace(joint_q=np.zeros(2), joint_qd=np.zeros(2))
        env = SimpleNamespace(
            step_count=3,
            sim_time=0.03,
            last_context=context,
            pipeline=SimpleNamespace(snapshot=snap),
            model=SimpleNamespace(particle_count=0),
        )
        rec = EpisodeRecorder(tempfile.mkdtemp(), RecordSpec(enabled=False), capture=CaptureSpec(enabled=True))
        rec._record_capture(env)
        self.assertEqual(len(rec._capture_contacts), 1)
        pair = rec._capture_contacts[0]["pairs"][0]
        self.assertEqual((pair["shape_a"], pair["shape_b"]), (4, 5))


class SummariseGatedMetricTest(unittest.TestCase):
    def test_steps_come_from_the_rows_that_carry_the_metric(self) -> None:
        rows = [
            {"step": 1, "success": False, "reward": 0.0, "metrics": {"awaiting": "lid"}},
            {"step": 2, "success": False, "reward": 0.0, "metrics": {"awaiting": "lid"}},
            {"step": 3, "success": False, "reward": 0.0, "metrics": {"fired": False, "d": 0.1}},
            {"step": 4, "success": False, "reward": 0.0, "metrics": {"fired": True, "d": 0.9}},
            {"step": 5, "success": True, "reward": 1.0, "metrics": {"fired": True, "d": 0.2}},
        ]
        summary = summarise(rows)
        self.assertEqual(summary.boolean["fired"]["first_true_step"], 4.0)
        self.assertEqual(summary.numeric["d"]["argmax_step"], 4.0)
        self.assertNotIn("awaiting", summary.numeric)


class BodilessStaticPoseTest(unittest.TestCase):
    def test_static_without_a_body_reads_its_xform(self) -> None:
        handle = EntityHandle(name="pegboard", body=-1, static=True, xform=(0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0))
        body_q = np.array([[9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 1.0]])
        np.testing.assert_allclose(handle.pose(body_q), [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(handle.position(body_q), [0.1, 0.2, 0.3])

    def test_body_handle_reads_body_q(self) -> None:
        handle = EntityHandle(name="cube", body=0)
        body_q = np.array([[9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 1.0]])
        np.testing.assert_allclose(handle.position(body_q), [9.0, 8.0, 7.0])


if __name__ == "__main__":
    unittest.main()
