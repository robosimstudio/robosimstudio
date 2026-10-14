# SPDX-License-Identifier: Apache-2.0
"""The rod held to what it says: the four spawn ranges are independent, a rod
whose centreline cannot be read does not publish 0 m, the grip shapes are
found off one copy of ``shape_body``, and winding is measured about the axis
as declared, not about a vertical line through its origin.

No GPU: the spawn is numpy and Warp's host quaternions, the entity is a
stand-in built without a scene, and the measures are pure numpy.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

TIER = "fast"
"""numpy, Warp's host maths and stand-ins: no GPU."""


def _straight(n: int = 8):
    import warp as wp

    points = [wp.vec3(0.0, 0.1 * i, 0.01) for i in range(n)]
    quats = [wp.quat_identity() for _ in range(n)]
    return points, quats


def _spec(**ranges):
    return SimpleNamespace(direction=(0.0, 1.0, 0.0), **ranges)


class TestTheSpawnRangesAreIndependent(unittest.TestCase):
    """``rod/entity.py::_bow_cable``: a shift or a yaw needs no bow to happen."""

    def test_a_shift_without_a_bow_moves_the_rod(self) -> None:
        from robosimstudio.domains.rod.entity import _bow_cable

        points, quats = _straight()
        spec = _spec(shift_range=(-0.02, 0.02))
        got, _ = _bow_cable(points, quats, spec, SimpleNamespace(overrides={"cable_shift": 1.0}))
        moved = np.asarray([list(p) for p in got]) - np.asarray([list(p) for p in points])
        # Lateral to a rod along +y is -x: up x along.
        np.testing.assert_allclose(moved, np.tile([-0.02, 0.0, 0.0], (len(points), 1)), atol=1e-9)

    def test_a_yaw_without_a_bow_turns_the_rod(self) -> None:
        from robosimstudio.domains.rod.entity import _bow_cable

        points, quats = _straight()
        spec = _spec(yaw_range=(0.0, 20.0))
        got, _ = _bow_cable(points, quats, spec, SimpleNamespace(overrides={"cable_yaw": 1.0}))
        pts = np.asarray([list(p) for p in got])
        chord = pts[-1] - pts[0]
        self.assertAlmostEqual(math.degrees(math.atan2(-chord[0], chord[1])), 20.0, places=4)

    def test_a_bow_drawn_at_zero_keeps_the_shift(self) -> None:
        from robosimstudio.domains.rod.entity import _bow_cable

        points, quats = _straight()
        spec = _spec(bow_range=(-0.02, 0.02), shift_range=(0.0, 0.03))
        episode = SimpleNamespace(overrides={"cable_bow": 0.0, "cable_shift": 1.0})
        got, _ = _bow_cable(points, quats, spec, episode)
        self.assertAlmostEqual(float(got[3][0]) - float(points[3][0]), -0.03, places=6)

    def test_nothing_drawn_is_the_same_objects(self) -> None:
        from robosimstudio.domains.rod.entity import _bow_cable

        points, quats = _straight()
        spec = _spec(shift_range=(-0.02, 0.02), yaw_range=(-5.0, 5.0))
        got_p, got_q = _bow_cable(points, quats, spec, SimpleNamespace(overrides={}))
        self.assertIs(got_p, points)
        self.assertIs(got_q, quats)


class _Rod:
    """An ``EntityRod`` with only what ``published_states`` and ``centerline`` read."""

    @staticmethod
    def make(n_bodies: int, n_half: int):
        from robosimstudio.domains.rod.entity import EntityRod

        class Rod(EntityRod):
            name = "cable"

        rod = object.__new__(Rod)
        rod.bodies = np.arange(n_bodies)
        rod._half_heights = np.full(n_half, 0.006)
        rod._half_heights_per_world = {}
        rod._hold_flag = None
        return rod


class TestTheLengthIsNotInvented(unittest.TestCase):
    """``rod/entity.py::published_states``: a rod it cannot read is an error, not a 0 m rod."""

    def test_a_mismatched_rod_raises(self) -> None:
        rod = _Rod.make(4, 3)
        snap = SimpleNamespace(body_q=np.tile([0, 0, 0, 0, 0, 0, 1.0], (4, 1)), world=0)
        with self.assertRaises(RuntimeError):
            rod.published_states(snap)

    def test_a_built_rod_reports_its_length(self) -> None:
        rod = _Rod.make(4, 4)
        body_q = np.tile([0, 0, 0, 0, 0, 0, 1.0], (4, 1))
        body_q[:, 2] = 0.012 * np.arange(4) + 0.006  # capsules end to end up +z
        got = rod.published_states(SimpleNamespace(body_q=body_q, world=0))
        self.assertAlmostEqual(got["length_m"], 0.048, places=9)
        self.assertEqual(got["n_segments"], 4.0)

    def test_no_poses_leaves_the_length_out(self) -> None:
        got = _Rod.make(4, 4).published_states(None)
        self.assertNotIn("length_m", got)
        self.assertEqual(got["held"], 0.0)


class TestTheBladesAreReadOnce(unittest.TestCase):
    """``rod/entity.py::_blade_shapes``: one host copy of ``shape_body``, not one per shape."""

    def test_shape_body_is_copied_once(self) -> None:
        from robosimstudio.domains.rod.entity import EntityRod

        calls = []

        class Body:
            def numpy(self):
                calls.append(1)
                return np.array([0, 1, 1, 2, -1])

        model = SimpleNamespace(
            body_label=["a/base", "a/left_tip", "a/palm"], body_count=3, shape_count=5, shape_body=Body()
        )
        fake = SimpleNamespace(_hand_body_labels=lambda: ({"left_tip"}, set()))
        self.assertEqual(EntityRod._blade_shapes(fake, model), {1, 2})
        self.assertEqual(len(calls), 1)


def _helix(origin, u, v, d, *, turns: float = 2.0, radius: float = 0.03, n: int = 80) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi * turns, n)
    return (
        np.asarray(origin)
        + np.outer(radius * np.cos(t), u)
        + np.outer(radius * np.sin(t), v)
        + np.outer(np.linspace(-0.02, 0.02, n), d)
    )


class TestWindingIsAboutTheDeclaredAxis(unittest.TestCase):
    """``rod/measures.py``: a horizontal axis (a hook's) counts its turns."""

    def test_a_horizontal_axis(self) -> None:
        from robosimstudio.domains.rod import measures

        origin = np.array([0.3, -0.1, 0.2])
        wrap = _helix(origin, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
        got = measures._winding_about_axis(wrap, {"origin": origin, "direction": (1.0, 0.0, 0.0)})
        self.assertAlmostEqual(got["turns"], 2.0, delta=0.05)
        along = np.array([0.05, 0.0, 0.0])
        ends = np.stack([origin - along, origin + along])
        self.assertAlmostEqual(measures._winding_about_rod(wrap, {"axis": ends})["turns"], 2.0, delta=0.05)

    def test_a_vertical_axis_reads_as_before(self) -> None:
        from robosimstudio.domains.rod import measures

        origin = np.array([0.1, 0.2, 0.05])
        wrap = _helix(origin, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        before = measures.winding_turns(wrap, origin[:2])
        self.assertEqual(
            measures._winding_about_axis(wrap, {"origin": origin, "direction": (0, 0, 1)})["turns"], before[0]
        )


if __name__ == "__main__":
    unittest.main()
