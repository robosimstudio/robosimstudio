# SPDX-License-Identifier: Apache-2.0
"""The cloth grasp the towel example promises, made the way its page makes it.

``02_cloth/00_towel`` says: drag the hand down onto the towel's edge, press Close and
lift. Like :mod:`tests.test_example_grasps`, this runs the example's own ``main()``, drives
its arm panel exactly as the gizmo and the Grip button do (:mod:`tests.fixtures.grasp`) and
holds the result to numbers: the strip of edge under the jaws comes up with the hand and
stays in it. Nothing in the scene is pinned to a hand, so a pass is friction's -- the pads'
against the cloth, under one-way coupling with full-surface contact.

It also checks the coupler that makes that true: VBD's friction is the particle's slip
against how far the pad under it moved, so the one-way coupler must hand it the arm's
previous poses as well as its new ones. Handed the new pose twice, every finger stood still
to friction, and the same gesture kept 0 of the 15 particles it pinched.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import drag_to_grasp, open_session, tick

TIER = "medium"
"""Builds the two cloth examples and steps each through a page with no browser on it: about 80 s."""

FINGERTIPS = 0.0175
"""Where the gesture puts the tool centre over the bench [m]: the pads end 16.7 mm under it."""


class TestTheTowelExamplePinchesItsEdge(unittest.TestCase):
    """``02_cloth/00_towel``: down onto the near edge, Close, up 100 mm, and held for 2 s."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        cls.sim, kwargs = _run("02_cloth/00_towel.py")
        cls.session = open_session(cls.sim, 9500, kwargs.get("page"))
        tick(cls.session, 30)  # the towel lands from 20 mm
        cls.surface = rss.get_cell("bench_0728").surface_z
        ids = np.asarray(cls.sim["towel"].entity.particles, dtype=np.int64)
        cls.lying = cls.sim.state.particle_q.numpy().astype(np.float64)[ids]
        cls.home = np.asarray(cls.session.arms["right"].ik.home_pose[0], dtype=np.float64)
        # What the jaws close on: the cloth within 20 mm of where the tool centre closes.
        at = np.array([cls.home[0], cls.home[1], cls.surface + FINGERTIPS])
        cls.strip = ids[np.linalg.norm(cls.lying - at, axis=1) < 0.020]

        def strip():
            return cls.sim.state.particle_q.numpy().astype(np.float64)[cls.strip]

        cls.grasp = drag_to_grasp(cls.session, "right", strip, at=at, hover=0.03, lift=0.10)
        hand = cls.session.arms["right"].ik.ee_pose(cls.sim.state)[0]
        cls.held = float((np.linalg.norm(strip() - hand, axis=1) < 0.025).mean())
        print(f"\n[towel] {cls.grasp} held {cls.held:.3f} of {len(cls.strip)}", flush=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_hand_starts_over_the_edge(self) -> None:
        """The gesture is one drag straight down, so the hand must start above the edge."""
        edge = self.lying[:, 1].min()
        self.assertLess(abs(self.home[1] - edge), 0.005)
        self.assertLess(abs(self.home[0] - float(np.median(self.lying[:, 0]))), 0.01, "over the middle of it")
        self.assertGreater(float(self.home[2] - self.surface), 0.10)

    def test_the_edge_comes_up_with_the_hand(self) -> None:
        self.assertTrue(self.grasp.finite)
        self.assertLess(self.grasp.reached, 0.010, "the hand did not get to the top of the lift")
        self.assertGreater(self.grasp.rise, 0.06, "the edge stayed on the bench")

    def test_it_stays_in_the_fingers(self) -> None:
        self.assertGreaterEqual(len(self.strip), 5, "nothing under the jaws")
        self.assertGreaterEqual(self.held, 0.5, "the edge slipped out of the jaws")
        self.assertLess(self.grasp.drift, 0.02, "the edge slid in the jaws during the hold")

    def test_the_coupler_hands_vbd_the_arms_motion(self) -> None:
        """``state_in`` keeps the previous body poses through the deformable solve."""
        import warp as wp

        from robosimstudio.scene.physics.couplers import SolverOneWay

        solver = self.sim.pipeline.program.solver
        self.assertIsInstance(solver, SolverOneWay)
        pipeline = self.sim.pipeline
        before = wp.clone(pipeline.state_0.body_q)
        # Move the arm, so its new poses differ from the ones it starts the substep at.
        self.session.arms["right"].cursor_pos = self.session.arms["right"].cursor_pos + np.array([0.0, 0.0, 0.05])
        tick(self.session, 3)
        state_in, state_out = pipeline.state_0, pipeline.state_1
        wp.copy(before, state_in.body_q)
        solver.step(state_in, state_out, pipeline.control, None, pipeline.sim_dt)
        np.testing.assert_array_equal(before.numpy(), state_in.body_q.numpy())
        self.assertGreater(float(np.abs(state_out.body_q.numpy() - before.numpy()).max()), 0.0, "the arm did not move")


class TestYourOwnClothExamplePinchesItsHem(unittest.TestCase):
    """``02_cloth/04_your_own_cloth``: the Panda down onto the draped hem, Grip amount 0.95, up 100 mm."""

    HEM = (0.0, -0.095, 0.05)
    """Where the tool centre closes, in the bench's work-surface frame [m]: on the free hem of the -y side."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        cls.sim, kwargs = _run("02_cloth/04_your_own_cloth.py")
        cls.session = open_session(cls.sim, 9501, kwargs.get("page"))
        # The page's reset puts the sheet back flat over the ball, which is where it was built.
        tick(cls.session, 180)
        origin = np.array(rss.get_cell("bench_075").surface_frame[:3])
        ids = np.asarray(cls.sim["sheet"].entity.particles, dtype=np.int64)
        cls.lying = cls.sim.state.particle_q.numpy().astype(np.float64)[ids]
        (cls.arm,) = cls.session.arms
        cls.home = np.asarray(cls.session.arms[cls.arm].ik.home_pose[0], dtype=np.float64)
        at = origin + np.array(cls.HEM)
        cls.hem = ids[np.linalg.norm(cls.lying - at, axis=1) < 0.020]
        cls.at = at

        def hem():
            return cls.sim.state.particle_q.numpy().astype(np.float64)[cls.hem]

        # 0.95 of the slider is 5 mm between the pads: the sheet's 4 mm contact thickness. Shut
        # (Close, 0 mm) they squeezed the hem out and it slid 37 mm down the hand; at 0.90 (9 mm) it fell.
        cls.grasp = drag_to_grasp(cls.session, cls.arm, hem, at=at, hover=0.06, lift=0.10, grip=0.95)
        hand = cls.session.arms[cls.arm].ik.ee_pose(cls.sim.state)[0]
        cls.held = float((np.linalg.norm(hem() - hand, axis=1) < 0.025).mean())
        print(f"\n[hem] {cls.grasp} held {cls.held:.3f} of {len(cls.hem)}", flush=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_hand_starts_over_the_hem(self) -> None:
        self.assertLess(float(np.linalg.norm(self.home[:2] - self.at[:2])), 0.005)
        self.assertGreater(float(self.home[2] - self.at[2]), 0.10)

    def test_the_hem_comes_up_with_the_hand(self) -> None:
        self.assertTrue(self.grasp.finite)
        self.assertLess(self.grasp.reached, 0.010, "the hand did not get to the top of the lift")
        self.assertGreater(self.grasp.rise, 0.06, "the hem stayed down")

    def test_it_stays_in_the_fingers(self) -> None:
        self.assertGreaterEqual(len(self.hem), 3, "nothing under the jaws")
        self.assertGreaterEqual(self.held, 0.5, "the hem slipped out of the jaws")
        self.assertLess(self.grasp.drift, 0.02, "the hem slid in the jaws during the hold")


if __name__ == "__main__":
    unittest.main()
