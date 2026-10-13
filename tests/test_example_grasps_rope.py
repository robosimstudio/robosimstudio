# SPDX-License-Identifier: Apache-2.0
"""The rope example's grasp, made the way its page makes it: down, Close, up, and the cable comes too.

``examples/10_rope/00_cable.py`` says a person can drag the right hand down onto
the cable, press **Close** and lift it. This runs the example's own ``main()``,
drives its arm panel as the gizmo and the Grip button do
(:mod:`tests.fixtures.grasp`), and holds the lift to numbers. The cable has no
latch (``allow_grasp_latch=False``) and nothing else pins it to the hand, so what
comes up came up on the pads' friction.

The drag is a drag, not a jump: this scene's arm has no device IK (the rod's cable
joints rule it out), so the page commands each cursor update straight into the
joints with no rate limit, and a 120 mm jump of the cursor flicks the hand up at
about 0.4 m/s with nothing to ease it -- the capsule falls out of that, as it would
out of a real hand flicked as hard. A 0.2 m/s drag is a person moving the gizmo.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import drag_to_grasp, open_session, tick

TIER = "medium"
"""Builds the example and steps it through a page with no browser on it."""

ABOVE_AXIS = 0.010
"""Where the tool centre closes, over the cable's axis [m]. The YAM's fingertips reach
16.8 mm past it, so they stop 4 mm over the bench under an 11 mm-radius cable; at the
axis itself they press into the bench and the hand is shoved 50 mm sideways."""


class TestTheRopeExampleLiftsTheCable(unittest.TestCase):
    """``10_rope/00_cable``: the hand starts over the cable, closes on it and lifts its middle."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, kwargs = _run("10_rope/00_cable.py")
        cls.session = open_session(cls.sim, 9400, kwargs.get("page"))
        tick(cls.session, 30)  # the cable lands on the bench
        bodies = np.asarray(cls.sim["cable"].bodies)
        middle = slice(len(bodies) // 2 - 1, len(bodies) // 2 + 1)  # the two capsules either side of it

        def capsules() -> np.ndarray:
            return cls.sim.state.body_q.numpy()[bodies, :3].astype(np.float64)

        cls.lying = capsules()
        cls.home = cls.session.arms["right"].ik.home_pose[0]
        cls.under = int(np.argmin(np.linalg.norm(cls.lying[:, :2] - cls.home[:2], axis=1)))
        cls.grasp = drag_to_grasp(
            cls.session,
            "right",
            lambda: capsules()[cls.under],
            at=cls.lying[cls.under] + np.array([0.0, 0.0, ABOVE_AXIS]),
            hover=0.03,
            lift=0.12,
            speed=0.2,
        )
        held = capsules()
        hand = cls.session.arms["right"].ik.ee_pose(cls.sim.state)[0]
        cls.rise = float(held[middle].mean(axis=0)[2] - cls.lying[middle].mean(axis=0)[2])
        cls.gap = float(np.linalg.norm(held[cls.under] - hand))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_hand_starts_over_the_cable(self) -> None:
        """The gesture is one drag straight down, so the hand must start above a capsule near the middle."""
        self.assertLess(float(np.linalg.norm(self.home[:2] - self.lying[self.under, :2])), 0.005)
        self.assertIn(self.under, (13, 14), "the hand is not over the cable's middle")
        self.assertGreater(float(self.home[2] - self.lying[self.under, 2]), 0.07)

    def test_nothing_but_the_pads_holds_it(self) -> None:
        spec = self.sim["cable"].entity.spec
        self.assertFalse(spec.allow_grasp_latch)
        self.assertTrue(spec.scene_friction, "the pads would keep the bench's mu of 2, not the rod's 60")
        self.assertEqual("sticky", self.sim.request.physics.collision.contact_matching)

    def test_the_cable_comes_up_with_the_hand(self) -> None:
        self.assertTrue(self.grasp.finite)
        self.assertGreater(self.rise, 0.07, f"the cable's middle rose {self.rise * 1e3:.1f} mm")

    def test_it_does_not_slide_in_the_fingers(self) -> None:
        """Held where it was pinched, the capsule's offset from the tool centre changes by under 5 mm."""
        self.assertLess(self.grasp.drift, 0.005)
        self.assertLess(self.gap, 0.015, f"the pinched capsule ended {self.gap * 1e3:.1f} mm from the tool centre")
        self.assertAlmostEqual(0.022, self.grasp.width, delta=0.003, msg="the fingers are not on the cable")


if __name__ == "__main__":
    unittest.main()
