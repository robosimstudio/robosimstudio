# SPDX-License-Identifier: Apache-2.0
"""The soft grasp the sponge example promises, made the way its page makes it.

``03_soft/00_sponge`` says: drag the hand down onto the sponge, press **Close**
and lift. That is a claim about three things the default scene got wrong, each
measured in the example: the foam's friction had no static part, so a held
sponge slid out of the pads; the foam's push on the 10 g pads was capped at
0.3 N, so the jaws closed through it; and the grip had no ceiling. The case
here runs the example's own ``main()``, drives its arm panel exactly as the
gizmo and the Grip button do (:mod:`tests.fixtures.grasp`), and holds the result
to numbers. Nothing is pinned, so what comes up came up by friction.

Skipped by name without the ``[superdex]`` extra, which is what steps the foam.
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

TIER = "medium"
"""Builds the example on a GPU and steps it through a page with no browser on it."""

HAS_SUPERDEX = importlib.util.find_spec("superdex") is not None
CLOSE_HEIGHT = 0.020
"""Where the tool centre closes, over the bench [m]: the pads then span 11-27 mm of the 35 mm foam."""


@unittest.skipUnless(HAS_SUPERDEX, "[superdex] is not installed; nothing steps the sponge")
class TestTheSpongeExampleLiftsTheSponge(unittest.TestCase):
    """``03_soft/00_sponge``: down, Close, up, and the sponge comes too."""

    @classmethod
    def setUpClass(cls) -> None:
        from tests.fixtures.examples import _run
        from tests.fixtures.grasp import drag_to_grasp, open_session, tick

        cls.sim, kwargs = _run("03_soft/00_sponge.py")
        cls.session = open_session(cls.sim, 9311, kwargs.get("page"))
        tick(cls.session, 30)  # the foam settles on the bench
        sponge = cls.sim["sponge"]
        cls.lying = np.asarray(sponge.particle_q, dtype=np.float64)
        home = cls.session.arms["right"].ik.home_pose[0]
        bench = float(cls.lying[:, 2].min())
        at = np.array([home[0], home[1], bench + CLOSE_HEIGHT])
        cls.grasp = drag_to_grasp(cls.session, "right", lambda: sponge.particle_q, at=at, lift=0.10, hold=3.0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_hand_starts_over_the_sponge(self) -> None:
        """The gesture is one drag straight down, so the hand must start above it."""
        hand = self.session.arms["right"].ik.home_pose[0]
        centre = self.lying.mean(axis=0)
        self.assertLess(float(np.linalg.norm(hand[:2] - centre[:2])), 0.005)
        self.assertGreater(float(hand[2] - self.lying[:, 2].max()), 0.15)

    def test_the_sponge_comes_up_with_the_hand(self) -> None:
        self.assertTrue(self.grasp.finite)
        self.assertLess(self.grasp.reached, 0.010, "the hand did not get to the top of the lift")
        self.assertGreater(self.grasp.rise, 0.06, f"the sponge rose {self.grasp.rise * 1e3:.0f} mm")

    def test_it_does_not_slide_in_the_fingers(self) -> None:
        """From the close to three seconds after the lift arrived: the foam stays where it was taken."""
        self.assertLess(self.grasp.drift, 0.003, f"it slid {self.grasp.drift * 1e3:.1f} mm")

    def test_the_jaws_squeeze_the_foam_and_do_not_cut_it(self) -> None:
        """The 50 mm sponge between the pads, squeezed a few millimetres, not closed through."""
        self.assertGreater(self.grasp.width, 0.035, f"the jaws are {self.grasp.width * 1e3:.0f} mm apart")
        self.assertLess(self.grasp.width, 0.050, "the jaws are not on the foam")

    def test_the_hand_holds_its_height_with_the_sponge_in_it(self) -> None:
        from tests.fixtures.grasp import tick

        ik = self.session.arms["right"].ik
        z0 = float(ik.ee_pose(self.sim.state)[0][2])
        tick(self.session, 120)
        z1 = float(ik.ee_pose(self.sim.state)[0][2])
        self.assertLess(abs(z1 - z0), 0.002)


if __name__ == "__main__":
    unittest.main()
