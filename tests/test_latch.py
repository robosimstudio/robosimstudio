# SPDX-License-Identifier: Apache-2.0
"""The grasp latch holds a cable segment where it was taken, and lets it go.

Deterministic on purpose. The rollouts that exercise the latch as part of a
task -- ``TestCableScene``, ``test_repro_cable`` -- are contact-sensitive and
red or flaky at HEAD, so they cannot tell a latch regression from the weather.
This file asks the one question that has a yes/no answer under gravity alone:
pin a segment to a hand, step, and the pinned end stays where it was while the
free end sags; drop the pin, step, and the whole rod falls.

It is also the test for the unweld. ``EntityRod`` used to allocate the latch's
five device arrays and write the ``wp.launch`` over them itself -- a second copy
of what the rod pack's ``latch.build_hooks`` builds for a repro bundle, kept in
step by hand. The entity now *consumes* that module, and the identity checks
below are what say so.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "medium"

LIFT = 0.05
"""How far the pinned end is asked to rise [m]. Lifting one end of a resting rod
loads the pin with about half the rod's weight -- a quarter of a newton, against
a force ceiling of two -- so the pin *drives*, and does not merely slip.

Not a drop test, and that was measured rather than assumed: a 0.4 m rod
released from 0.30 m and caught by one end overpowers the ceiling by an order of
magnitude, the latch slips exactly as its docstring says it must, and the
segment comes to rest 0.2 m below its target with the flag still set. The
ceiling is sized for dragging a rod that is already supported, which is the
latch's job; catching a fall is not."""


def _scene():
    import robosimstudio as rss

    scene = rss.Scene(cell="bench_075", name="latch")
    scene.add(rss.robot("yam_bimanual"))
    scene.add(rss.rod(name="rope", length=0.4, radius=0.011, at=(0.0, -0.20, 0.011), segments=20))
    sim = scene.build()
    sim.run(30).sync()  # let it settle onto the bench first
    return sim, sim.scene.by_name["rope"]


def _engage(sim, rod, *, slot: int = 0, arm: str = "right") -> int:
    """Pin segment 0 to ``arm``'s hand at its *current* relative pose.

    The picker on the entity refuses a segment more than 50 mm from the jaws;
    this writes the same five arrays it would, without that gate, because what
    is under test is the hold and not the reach.
    """
    import warp as wp

    from robosimstudio.core.math import quat_to_mat

    del quat_to_mat
    body_q = sim.pipeline.state_0.body_q.numpy()
    seg = int(rod.bodies[0])
    grip = int(rod._grip_bodies[arm])
    g, s = body_q[grip], body_q[seg]
    g_tf = wp.transform(wp.vec3(*[float(v) for v in g[:3]]), wp.quat(*[float(v) for v in g[3:7]]))
    s_tf = wp.transform(wp.vec3(*[float(v) for v in s[:3]]), wp.quat(*[float(v) for v in s[3:7]]))
    rel = wp.transform_multiply(wp.transform_inverse(g_tf), s_tf)
    rels = [rod._hold_rel.numpy()[i] for i in range(2)]
    rels[slot] = rel
    rod._hold_rel.assign(rels)
    for arr, value in ((rod._hold_seg, seg), (rod._hold_grip, grip)):
        v = arr.numpy()
        v[slot] = value
        arr.assign(v)
    flags = rod._hold_flag.numpy()
    flags[slot] = 1
    rod._hold_flag.assign(flags)
    return seg


class TestTheLatchIsTheSharedModule(unittest.TestCase):
    """The unweld: the entity's arrays and stage are ``latch.build_hooks``'s."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, cls.rod = _scene()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def test_the_arrays_are_the_modules_own(self) -> None:
        sinks = self.rod._latch.aux_sinks
        for name in ("hold_flag", "hold_seg", "hold_grip", "hold_rel", "hold_gains"):
            with self.subTest(array=name):
                self.assertIs(getattr(self.rod, f"_{name}"), sinks[name], "a second copy, not the module's")

    def test_the_stage_is_the_modules_own(self) -> None:
        stage = self.rod._latch.stages["rigid_force"][0]
        hooked = [getattr(h, "fn", h) for h in self.sim.pipeline.hooks.rigid_force]
        self.assertIn(stage, hooked, "the entity launched its own copy of the kernel")

    def test_the_gains_were_seeded_from_the_spec(self) -> None:
        kp, kd = self.rod.hold_gains()
        got = self.rod._hold_gains.numpy()
        np.testing.assert_allclose(got, [kp, kd, self.rod.hold_force_max()], rtol=1e-6)


def _retarget(sim, rod, seg: int, *, slot: int = 0, arm: str = "right", dz: float) -> None:
    """Ask the pin to hold ``seg`` ``dz`` higher than it is now, in world terms."""
    import warp as wp

    body_q = sim.pipeline.state_0.body_q.numpy()
    g = body_q[int(rod._grip_bodies[arm])]
    s = body_q[seg].copy()
    s[2] += dz
    g_tf = wp.transform(wp.vec3(*[float(v) for v in g[:3]]), wp.quat(*[float(v) for v in g[3:7]]))
    s_tf = wp.transform(wp.vec3(*[float(v) for v in s[:3]]), wp.quat(*[float(v) for v in s[3:7]]))
    rels = [rod._hold_rel.numpy()[i] for i in range(2)]
    rels[slot] = wp.transform_multiply(wp.transform_inverse(g_tf), s_tf)
    rod._hold_rel.assign(rels)


class TestTheLatchHolds(unittest.TestCase):
    """The rod rides the hand: a pinned end goes where the pin says, and lets go."""

    def test_the_pinned_end_follows_its_target_and_the_free_end_does_not(self) -> None:
        sim, rod = _scene()
        try:
            seg = _engage(sim, rod)
            free = int(rod.bodies[-1])
            q0 = sim.pipeline.state_0.body_q.numpy()
            held_z0, free_z0 = float(q0[seg, 2]), float(q0[free, 2])

            sim.run(20).sync()  # pinned where it lies: nothing should move
            q1 = sim.pipeline.state_0.body_q.numpy()
            self.assertLess(abs(float(q1[seg, 2]) - held_z0), 0.01, "pinned in place, it drifted")

            _retarget(sim, rod, seg, dz=LIFT)
            sim.run(60).sync()
            q2 = sim.pipeline.state_0.body_q.numpy()
            self.assertGreater(float(q2[seg, 2]) - held_z0, 0.6 * LIFT, "the pinned end did not follow its target")
            self.assertLess(float(q2[free, 2]) - free_z0, 0.02, "the far end came up too -- the whole rod moved")
            self.assertEqual(rod.published_states(None)["held"], 1.0)
        finally:
            sim.close()

    def test_released_it_comes_back_down(self) -> None:
        sim, rod = _scene()
        try:
            seg = _engage(sim, rod)
            z0 = float(sim.pipeline.state_0.body_q.numpy()[seg, 2])
            _retarget(sim, rod, seg, dz=LIFT)
            sim.run(60).sync()
            lifted = float(sim.pipeline.state_0.body_q.numpy()[seg, 2])
            self.assertGreater(lifted - z0, 0.6 * LIFT, "precondition: it was lifted")

            rod.hold_segment(False, slot=0)
            sim.run(60).sync()
            z2 = float(sim.pipeline.state_0.body_q.numpy()[seg, 2])
            self.assertLess(z2 - z0, 0.015, "released, it should have settled back onto the bench")
            self.assertEqual(rod.published_states(None)["held"], 0.0)
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()
