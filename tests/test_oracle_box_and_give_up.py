# SPDX-License-Identifier: Apache-2.0
"""Two connector-side defects the level-1 sorting tier exposed, pinned.

Both were found by attributing the tier's Oracle losses object by object
(2026-09-08): 27 losses across 16 scenes, 14 of them objects abandoned without
an attempt on the pass of a *different* object, and the tape measure missed in
every scene it appears in because the oracle boxed it 107 x 109 x 46 mm.
"""

from __future__ import annotations

import unittest

import newton
import numpy as np
import warp as wp

from robosimstudio.solvers.grasp.obb import compute_obb
from robosimstudio_policy_agent.sim_tools import GoalCursor

TIER = "fast"
"""No GPU: a cursor over a faked goal tree, and a builder finalized on the CPU."""


class _Cursor(GoalCursor):
    """A cursor whose goal tree and evaluator verdict are plain data."""

    def __init__(self, goals, satisfied):
        super().__init__(env=None, bus=None)
        self._goals = tuple(goals)
        self.satisfied_now = dict(satisfied)

    @property
    def goals(self):
        return self._goals

    def _satisfied(self):
        return dict(self.satisfied_now)


class TestGiveUpAbandonsThePassThatFailed(unittest.TestCase):
    GOALS = (("a", "ra"), ("b", "rb"), ("c", "rc"))

    def test_a_failed_carry_that_lands_its_object_does_not_abandon_the_next_one(self):
        """The shipped failure branch opens the hand before it gives up. When
        the drop lands the object in its own compartment, the cursor's first
        outstanding goal is the NEXT object -- which is what an empty-name
        skip used to abandon, unattempted."""
        cursor = _Cursor(self.GOALS, dict.fromkeys(self.GOALS, False))
        self.assertEqual(cursor.next_goal(count_attempt=True)["object_name"], "a")
        cursor.satisfied_now[("a", "ra")] = True  # the release landed it
        skipped = cursor.skip_goal("")
        self.assertEqual(skipped["object_name"], "a")
        self.assertEqual(cursor.given_up, {"a"})
        self.assertEqual(cursor.next_goal(count_attempt=True)["object_name"], "b")

    def test_a_failed_grasp_still_abandons_its_own_object(self):
        cursor = _Cursor(self.GOALS, dict.fromkeys(self.GOALS, False))
        cursor.next_goal(count_attempt=True)
        cursor.skip_goal("")
        self.assertEqual(cursor.given_up, {"a"})
        self.assertEqual(cursor.next_goal(count_attempt=True)["object_name"], "b")

    def test_without_a_gate_call_the_empty_name_falls_back_to_the_first_outstanding(self):
        cursor = _Cursor(self.GOALS, dict.fromkeys(self.GOALS, False))
        self.assertEqual(cursor.skip_goal("")["object_name"], "a")

    def test_a_named_skip_is_unchanged(self):
        cursor = _Cursor(self.GOALS, dict.fromkeys(self.GOALS, False))
        cursor.next_goal(count_attempt=True)
        self.assertEqual(cursor.skip_goal("c")["object_name"], "c")
        self.assertEqual(cursor.given_up, {"c"})


class TestTheGraspBoxIsTheCollidersBox(unittest.TestCase):
    """A drawn-hull tier collides through a hull at identity pose and keeps the
    export's fitted box beside it, rotated and non-colliding. The grasp box is
    the collider's; the fitted box's corners must not inflate it."""

    HALF = np.array([0.039, 0.036, 0.018])  # a tape measure, roughly

    def _body(self, builder):
        return builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), mass=0.15)

    def test_a_non_colliding_rotated_box_does_not_inflate_the_collider(self):
        builder = newton.ModelBuilder()
        body = self._body(builder)
        # The collider: an upright box standing in for the hull, at identity.
        builder.add_shape_box(body=body, hx=float(self.HALF[0]), hy=float(self.HALF[1]), hz=float(self.HALF[2]))
        # The export's fitted box: the same extents, rotated, drawn only.
        q = wp.quat_from_axis_angle(wp.vec3(0.3, 0.8, 0.5), 1.1)
        cfg = builder.default_shape_cfg.copy()
        cfg.has_shape_collision = False
        builder.add_shape_box(
            body=body,
            xform=wp.transform((0.0, 0.0, 0.0), q),
            hx=float(self.HALF[0]),
            hy=float(self.HALF[1]),
            hz=float(self.HALF[2]),
            cfg=cfg,
        )
        box = compute_obb(builder, body, builder.body_com[body])
        np.testing.assert_allclose(box.half, self.HALF, atol=1e-7)
        self.assertEqual(box.long_axis, 0)

    def test_a_body_that_collides_with_nothing_is_boxed_by_what_it_draws(self):
        builder = newton.ModelBuilder()
        body = self._body(builder)
        cfg = builder.default_shape_cfg.copy()
        cfg.has_shape_collision = False
        builder.add_shape_box(body=body, hx=0.05, hy=0.01, hz=0.01, cfg=cfg)
        box = compute_obb(builder, body, builder.body_com[body])
        np.testing.assert_allclose(box.half, [0.05, 0.01, 0.01], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
