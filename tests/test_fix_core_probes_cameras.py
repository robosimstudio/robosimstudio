# SPDX-License-Identifier: Apache-2.0
"""Regressions for the core probes and the camera spec, all host-side."""

import dataclasses
import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.core.probes.clearance import _segment_segment
from robosimstudio.core.probes.contact import FleetContactProbe
from robosimstudio.core.probes.particle import ParticleProbe
from robosimstudio.core.streams import stream_lock

TIER = "fast"


class TestCameraCopiesKeepEveryField(unittest.TestCase):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class _Tagged(CameraSpec):
        tag: int = 0

    def test_scaled_keeps_fields_and_type(self):
        cam = self._Tagged(width=64, height=48, fx=50.0, cx=30.0, tag=7)
        small = cam.scaled(32, 24)
        self.assertIs(type(small), self._Tagged)
        self.assertEqual(small.tag, 7)
        np.testing.assert_allclose(small.intrinsics, [[25.0, 0, 15.0], [0, 25.0, 12.0], [0, 0, 1]])

    def test_with_pose_keeps_fields_and_drops_xform(self):
        cam = self._Tagged(xform=(0, 0, 1, 0, 0, 0, 1), tag=3)
        moved = cam.with_pose((1, 2, 3), (0, 0, 0))
        self.assertIs(type(moved), self._Tagged)
        self.assertEqual(moved.tag, 3)
        self.assertIsNone(moved.xform)
        self.assertEqual(moved.eye, (1.0, 2.0, 3.0))


class TestSegmentAgainstPoint(unittest.TestCase):
    def _d(self, p1, q1, p2, q2):
        return float(
            _segment_segment(
                np.array([p1], float), np.array([q1], float), np.array([p2], float), np.array([q2], float)
            )[0, 0]
        )

    def test_point_second_measures_to_nearest_point_on_segment(self):
        self.assertAlmostEqual(self._d((0, 0, 0), (1, 0, 0), (0.5, 1, 0), (0.5, 1, 0)), 1.0)

    def test_both_orders_agree(self):
        self.assertAlmostEqual(self._d((0.5, 1, 0), (0.5, 1, 0), (0, 0, 0), (1, 0, 0)), 1.0)

    def test_point_past_the_end_clamps(self):
        self.assertAlmostEqual(self._d((0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 0, 0)), 1.0)


class _Arr:
    def __init__(self, value, seen):
        self.value, self.seen = np.asarray(value), seen

    def numpy(self):
        self.seen.append(stream_lock._is_owned())
        return self.value


class TestFleetContactProbe(unittest.TestCase):
    def _probe(self, nacon):
        probe = FleetContactProbe.__new__(FleetContactProbe)
        seen = []
        probe.solver = SimpleNamespace(mjw_data=SimpleNamespace(nacon=_Arr([nacon], seen)), mjw_model=None)
        probe._ids = np.zeros(4, dtype=np.int32)
        probe.geoms = np.zeros((0, 2), dtype=np.int32)
        probe.impulses = np.zeros((0, 3))
        probe.depths = np.zeros(0)
        probe.worldids = np.zeros(0, dtype=np.int32)
        probe.geom_to_shape = np.zeros((1, 1), dtype=np.int32)
        probe.dt = 0.01
        probe.step = None
        probe._views = {}
        return probe, seen

    def test_a_view_before_the_first_read_is_empty(self):
        k = 3
        solver = SimpleNamespace(
            mjc_geom_to_newton_shape=SimpleNamespace(numpy=lambda: np.arange(k, dtype=np.int32).reshape(1, k)),
            mjw_data=SimpleNamespace(contact=SimpleNamespace(geom=np.zeros((4, 2)))),
        )
        probe = FleetContactProbe(
            solver,
            SimpleNamespace(device="cpu"),
            0.01,
            worlds=SimpleNamespace(shapes_per_world=k, num_worlds=1),
        )
        view = probe.view(0)
        self.assertEqual(len(view.depths), 0)
        self.assertEqual(len(view.penetrations()), 0)

    def test_nacon_read_holds_the_stream_lock(self):
        probe, seen = self._probe(0)
        probe.update_all(1)
        self.assertEqual(seen, [True])
        self.assertEqual(len(probe.view(0).depths), 0)


class TestCoverageNaNBaseline(unittest.TestCase):
    def test_nan_baseline_reads_nan(self):
        probe = ParticleProbe.__new__(ParticleProbe)
        probe.positions = np.array([[0.0, 0, 0], [0.05, 0, 0]])
        probe.baseline = np.array([[np.nan, np.nan, 0.0], [np.nan, 0, 0]])
        probe.indices = lambda name=None: np.arange(2)
        probe.points = lambda name=None: probe.positions
        self.assertTrue(np.isnan(probe.coverage(cell=0.01)))

    def test_nan_rows_in_baseline_are_not_a_cell(self):
        probe = ParticleProbe.__new__(ParticleProbe)
        probe.positions = np.array([[0.0, 0, 0], [0.05, 0, 0], [0.05, 0, 0]])
        probe.baseline = np.array([[0.0, 0, 0], [0.05, 0, 0], [np.nan, 0, 0]])
        probe.indices = lambda name=None: np.arange(3)
        probe.points = lambda name=None: probe.positions
        self.assertAlmostEqual(probe.coverage(cell=0.01), 1.0)


if __name__ == "__main__":
    unittest.main()
