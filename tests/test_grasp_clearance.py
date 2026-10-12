# SPDX-License-Identifier: Apache-2.0
"""Where the jaws land: :meth:`AntipodalSampler._pads_clear` and its geometry.

The sampler had one obstacle test and it asked the wrong question. ``_clearance``
casts a ray *backwards from the tool centre* along the approach, which for a
top-down grasp on a resting object runs up into free air -- while the two plates
that have to straddle the object are centimetres to either side of that ray,
outside the object's silhouette, and therefore the parts of the hand nearest to
whatever is next to it.

Measured on ``astro/wiring_sort``, where it cost the whole suite: the
screwdriver's collision box sits flush against the combination pliers' over
160 mm of shared length, every grasp the graph could fly put a plate 8 to 11 mm
inside it, and every one of those approaches reported a full 100 mm of
clearance. The object was never picked up and its goal was never attempted.

Three things have to hold for the fix to be worth having, and they are the three
classes below: the plate volume has to be a *sweep* rather than a pose, the
overlap test has to be exact rather than an axis-aligned box around a rotated
one, and the geometry it reads has to be the geometry that stops a finger.
"""

from __future__ import annotations

import unittest

import newton
import numpy as np
import warp as wp

from robosimstudio.solvers import AntipodalSampler
from robosimstudio.solvers.grasp.antipodal import PadBox, _obb_hits_boxes

TIER = "fast"
"""No GPU: a two-box model on the host, and arithmetic."""

TARGET_HALF = (0.030, 0.015, 0.010)
"""The body being grasped: 60 x 30 x 20 mm, so pairs across all three axes fit."""


def _scene(*, neighbour: bool, gap: float = 0.0005, hidden: bool = False) -> tuple[newton.Model, int]:
    """A box to grasp, optionally with a second box flush against its ``+y`` face.

    Args:
        neighbour: Whether to add the second box at all.
        gap: How much daylight to leave between the two [m]. Half a millimetre
            is a sorting heap's own spacing between a pair of pliers and a
            screwdriver.
        hidden: Give the neighbour an *invisible collider* and a visible shape
            too small to matter, which is how a bare-OBJ import is built.
    """
    builder = newton.ModelBuilder()
    target = builder.add_body(xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(target, hx=TARGET_HALF[0], hy=TARGET_HALF[1], hz=TARGET_HALF[2])
    if neighbour:
        half = 0.025
        centre = TARGET_HALF[1] + gap + half
        other = builder.add_body(xform=wp.transform((0.0, centre, 0.0), wp.quat_identity()))
        shape = builder.add_shape_box(other, hx=TARGET_HALF[0], hy=half, hz=0.011)
        if hidden:
            builder.shape_flags[shape] = int(builder.shape_flags[shape]) & ~int(newton.ShapeFlags.VISIBLE)
            drawn = builder.add_shape_box(other, hx=0.002, hy=0.002, hz=0.002)
            builder.shape_flags[drawn] = int(builder.shape_flags[drawn]) & ~int(newton.ShapeFlags.COLLIDE_SHAPES)
    return builder.finalize(device="cpu"), target


def _headings(model: newton.Model, target: int, **kwargs) -> list[tuple[float, float]]:
    """``(|close . y|, width)`` per proposed grasp, so a test can name a heading."""
    sampler = AntipodalSampler(model, span=0.085, **kwargs)
    found = sampler.propose(target, model.body_q.numpy(), near=None, k=12)
    return [(abs(float(c.close_axis[1])), float(c.width)) for c in found]


class TestAJawLandingOnANeighbour(unittest.TestCase):
    """The defect itself, on the two boxes that reproduce it."""

    def test_the_blocked_heading_goes_and_the_clear_one_stays(self):
        """Closing across the neighbour is refused; closing along the free axis is not.

        The box is 60 x 30 x 20 mm with a second box half a millimetre off its
        ``+y`` face, so the 30 mm pair across ``y`` is the one whose ``+`` plate
        has to be where the neighbour is, and the 60 mm pair across ``x`` is
        clear. Both are perfectly good antipodal pairs and the old test kept
        both.
        """
        alone, target = _scene(neighbour=False)
        crowded, target_two = _scene(neighbour=True)

        free = _headings(alone, target)
        blocked = _headings(crowded, target_two)

        self.assertTrue(
            any(along > 0.9 for along, _ in free), "the 30 mm pair across y is a grasp when nothing is there"
        )
        self.assertFalse(
            any(along > 0.9 for along, _ in blocked),
            "a jaw cannot straddle the target across y without going through the neighbour",
        )
        self.assertTrue(
            any(along < 0.1 and width > 0.05 for along, width in blocked),
            "the 60 mm pair across x has room either side and must survive",
        )

    def test_room_for_the_plate_is_what_decides_it(self):
        """Move the neighbour a plate's width away and the heading comes back.

        The distance that matters is not the jaw gap -- the pads clear the
        neighbour's face at any of these spacings -- but the gap plus the
        plate's own thickness plus the margin a preshape leaves. Below that the
        plate is inside the neighbour with the jaws nominally around the object.
        """
        near, target = _scene(neighbour=True, gap=0.002)
        far, target_two = _scene(neighbour=True, gap=0.030)
        self.assertFalse(any(along > 0.9 for along, _ in _headings(near, target)))
        self.assertTrue(any(along > 0.9 for along, _ in _headings(far, target_two)))

    def test_the_robots_own_shapes_can_be_excluded(self):
        """``ignore_shapes`` takes the neighbour back out, which is how a hand is excluded.

        The hand that is going to fly the approach is the thing that moves, so
        it is not an obstacle to itself; ``GraspTools`` passes ``ArmHandle.shapes``
        for exactly this. Asserted on the neighbour because it is the shape this
        model has, and the mechanism is the same one.
        """
        crowded, target = _scene(neighbour=True)
        shapes = [s for s in range(crowded.shape_count) if int(crowded.shape_body.numpy()[s]) != target]
        self.assertTrue(any(along > 0.9 for along, _ in _headings(crowded, target, ignore_shapes=shapes)))


class TestThePlateIsASweep(unittest.TestCase):
    """:class:`PadBox`, whose every number is an interval rather than a pose.

    Two measurements on this hand have already been wrong by being snapshots: a
    band taken at the home posture, when the distal links sag 4 mm further back
    once the wrist points down, and a fingertip plane, when the plate reaches
    37 mm the other way. So the class states both ends of the stroke.
    """

    def test_the_interval_is_the_open_ends_back_and_the_shut_ends_front(self):
        """The plate only travels one way, so the union is one end of each."""
        pad = PadBox()
        low, high = pad.along_approach(0.0, 1.0)
        self.assertAlmostEqual(low, -pad.back[0])
        self.assertAlmostEqual(high, pad.front[1])

    def test_it_never_shrinks_as_the_stroke_grows(self):
        """A wider range of commands can only put the plate in more places."""
        pad = PadBox()
        inner = pad.along_approach(0.4, 0.6)
        outer = pad.along_approach(0.2, 0.8)
        self.assertLessEqual(outer[0], inner[0])
        self.assertGreaterEqual(outer[1], inner[1])

    def test_a_snapshot_band_would_miss_the_swing(self):
        """Open and shut disagree by the 13 mm the four-bar swings, which is the point."""
        pad = PadBox()
        at_open = pad.along_approach(0.0, 0.0)
        at_shut = pad.along_approach(1.0, 1.0)
        self.assertGreaterEqual(at_shut[0] - at_open[0], 0.010, "the plate's back edge comes forward as it closes")
        self.assertGreaterEqual(at_shut[1] - at_open[1], 0.005, "and so does its front edge")
        # A band read at one command is short of the swept one by that much,
        # which on a 20 mm-tall object is the whole object.
        snapshot = pad.along_approach(0.5, 0.5)
        swept = pad.along_approach(0.0, 1.0)
        self.assertGreater((swept[1] - swept[0]) - (snapshot[1] - snapshot[0]), 0.010)


class TestTheOverlapTestIsExact(unittest.TestCase):
    """:func:`_obb_hits_boxes`, and the grasp a looser test would refuse.

    A plate is 8 mm through and tens of millimetres long. Boxing it axis-aligned
    once it is turned 45 degrees claims a footprint most of its length across,
    so the loose version refuses grasps with a centimetre of room.
    """

    DIAGONAL = np.array(
        [[0.7071067811865476, 0.7071067811865476, 0.0], [-0.7071067811865476, 0.7071067811865476, 0.0], [0.0, 0.0, 1.0]]
    )
    HALF = np.array([0.004, 0.011, 0.020])

    def test_a_box_inside_the_plate_is_a_hit(self):
        centre = np.array([0.05, 0.0, 0.0])
        lo = np.array([[0.048, -0.002, -0.005]])
        hi = np.array([[0.052, 0.002, 0.005]])
        self.assertTrue(_obb_hits_boxes(centre, self.DIAGONAL, self.HALF, lo, hi))

    def test_a_box_in_the_corner_of_the_plates_own_bounding_box_is_not(self):
        """It overlaps the plate's axis-aligned bounds and misses the plate."""
        centre = np.array([0.05, 0.0, 0.0])
        lo = np.array([[0.055, 0.008, -0.010]])
        hi = np.array([[0.065, 0.014, 0.010]])
        corners = (
            centre
            + (np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * self.HALF)
            @ self.DIAGONAL
        )
        self.assertTrue(
            np.all(corners.min(axis=0) <= hi[0]) and np.all(corners.max(axis=0) >= lo[0]),
            "the case is only interesting while the two bounding boxes do overlap",
        )
        self.assertFalse(_obb_hits_boxes(centre, self.DIAGONAL, self.HALF, lo, hi))

    def test_no_obstacles_is_no_hit(self):
        self.assertFalse(_obb_hits_boxes(np.zeros(3), np.eye(3), self.HALF, np.zeros((0, 3)), np.zeros((0, 3))))


class TestItReadsTheColliderNotTheDrawing(unittest.TestCase):
    """The set of shapes an obstacle test is about.

    An object imported from a bare OBJ is drawn as a mesh carrying no collider
    and collides as an invisible fitted box, and the box stands up to 28.6 mm proud of the
    drawing. A test run over what is drawn would find nothing in the way and
    the hand would still hit the box.
    """

    def test_an_invisible_collider_still_blocks_a_jaw(self):
        crowded, target = _scene(neighbour=True, hidden=True)
        self.assertFalse(
            any(along > 0.9 for along, _ in _headings(crowded, target)),
            "the neighbour's collider is invisible, and it is still what a plate would hit",
        )


class TestBothRollsAreOffered(unittest.TestCase):
    """:meth:`GraspCandidate.flipped` is how an IK is given the other wrist roll."""

    def test_flipping_a_candidate_keeps_every_field(self):
        model, target = _scene(neighbour=False)
        sampler = AntipodalSampler(model, span=0.085)
        found = sampler.propose(target, model.body_q.numpy(), near=None, k=1)
        self.assertTrue(found, "need a candidate to flip")
        flipped = found[0].flipped()
        self.assertAlmostEqual(flipped.width, found[0].width)
        # The field that was missing, and whose absence made every call raise.
        self.assertAlmostEqual(flipped.local_width, found[0].local_width)
        np.testing.assert_allclose(flipped.close_axis, -np.asarray(found[0].close_axis))


if __name__ == "__main__":
    unittest.main()
