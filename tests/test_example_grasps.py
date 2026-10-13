# SPDX-License-Identifier: Apache-2.0
"""The grasps the examples promise, made the way their pages make them.

An example whose docstring says "drag the hand down onto it, press Close and
lift" is a claim about physics, and the claim used to be unchecked: under the
default arrangement a hand pushes a rigid body and cannot pinch one, so the
gesture the page invited lifted nothing. Each case here runs the example's own
``main()``, drives its arm panel exactly as the gizmo and the Grip button do
(:mod:`tests.fixtures.grasp`), and holds the result to numbers: the thing rose
with the hand and did not slide in the fingers. Nothing in these scenes is
pinned to a hand, so a pass is friction's.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import drag_to_grasp, open_session, tick

TIER = "medium"
"""Builds each example and steps it through a page with no browser on it."""


class TestTheRigidExampleLiftsTheCube(unittest.TestCase):
    """``01_rigid/00_drag_things``: down, Close, up, and the cube comes too."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, kwargs = _run("01_rigid/00_drag_things.py")
        cls.session = open_session(cls.sim, 8961, kwargs.get("page"))
        tick(cls.session, 50)  # the ball and the pin land; the cube is already down
        cube = cls.sim["cube"]
        cls.lying = np.asarray(cube.position, dtype=np.float64)
        cls.grasp = drag_to_grasp(cls.session, "right", lambda: cube.position, lift=0.15)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_hand_starts_over_the_cube(self) -> None:
        """The gesture is one drag straight down, so the hand must start above it."""
        hand = self.session.arms["right"].ik.home_pose[0]
        self.assertLess(float(np.linalg.norm(hand[:2] - self.lying[:2])), 0.005)
        self.assertGreater(float(hand[2] - self.lying[2]), 0.10)

    def test_the_cube_comes_up_with_the_hand(self) -> None:
        self.assertTrue(self.grasp.finite)
        self.assertLess(self.grasp.reached, 0.010, "the hand did not get to the top of the lift")
        self.assertGreater(self.grasp.rise, 0.13, "the cube stayed behind")

    def test_it_does_not_slide_in_the_fingers(self) -> None:
        self.assertLess(self.grasp.drift, 0.005)
        self.assertAlmostEqual(0.05, self.grasp.width, delta=0.004, msg="the fingers are not on its faces")

    def test_the_hand_holds_its_height_with_the_cube_in_it(self) -> None:
        """A loaded hand sits a little under its target; it must not sink on.

        The page used to re-seat its cursor on the hand every idle step, so each
        step's sag became the next target: a lifted cube sank 41 mm in 2 s."""
        z0 = float(self.session.arms["right"].ik.ee_pose(self.sim.state)[0][2])
        tick(self.session, 100)
        z1 = float(self.session.arms["right"].ik.ee_pose(self.sim.state)[0][2])
        self.assertLess(abs(z1 - z0), 0.002)


if __name__ == "__main__":
    unittest.main()
