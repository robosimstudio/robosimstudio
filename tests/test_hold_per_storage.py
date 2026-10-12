# SPDX-License-Identifier: Apache-2.0
"""Every rod implementation can be held, through whatever holds its kind of state.

``Centerline.hold`` is the affordance; the storage decides the mechanism. The
capsule chain pins a *body* to a gripper frame in a ``rigid_force`` kernel; a
Cosserat rod on SuperDex asks the engine for a compliant node constraint and
has the engine carry it on the hand, every substep. To the caller -- a routine,
a graph tool, this test -- the two are one method with one signature, and the
same ``held`` flag comes back in ``published_states``.

Until this existed the SuperDex rod could not be held at all: the latch is a
body kernel and a particle rod has no bodies, so "two implementations coexist"
was true for building and measuring and false for manipulating. Deterministic
under gravity, the way the latch test is: pin a node, retarget the hand-frame
offset 5 cm up, the node follows; release, it settles back.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "medium"

LIFT = 0.05


def _superdex_rod_scene():
    import robosimstudio as rss
    from robosimstudio.specs.physics import GENERIC, backend

    physics = rss.replace(
        GENERIC,
        backends=(("articulated", backend("mujoco")), ("rod", backend("superdex", substeps=2)), ("*", backend("vbd"))),
    )
    scene = rss.Scene(cell="bench_075", name="hold", physics=physics)
    scene.add(rss.robot("yam_bimanual"))
    scene.add(rss.rod(name="rope", length=0.4, radius=0.011, at=(0.0, -0.20, 0.011), segments=20))
    sim = scene.build()
    sim.run(30).sync()
    return sim, sim.scene.by_name["rope"]


def _aim(sim, rod, target, slot: int = 0) -> None:
    """Move a held node's target to ``target`` (world), as an offset in the hand's body frame."""
    from robosimstudio.core.math import quat_to_mat

    grip, _, _ = rod.hand_tcp("right", sim.pipeline.state_0.body_q.numpy())
    body = sim.pipeline.state_0.body_q.numpy()[grip].astype(np.float64)
    rod.bridge.ride(rod._pins[slot][0], grip, (quat_to_mat(body[3:7]).T @ (target - body[:3]))[None, :])


class TestTheContractIsOneMethod(unittest.TestCase):
    def test_both_implementations_answer_hold(self) -> None:
        import inspect

        from robosimstudio.domains.rod import Centerline, EntityRod, EntityRodSuperdex

        for cls in (EntityRod, EntityRodSuperdex):
            with self.subTest(cls=cls.__name__):
                self.assertIn("hold", vars(cls), "each storage answers for itself")
                self.assertEqual(
                    list(inspect.signature(cls.hold).parameters),
                    list(inspect.signature(Centerline.hold).parameters),
                    "one signature, so a caller need not know which it holds",
                )
                self.assertIn("published_states", dir(cls))


class TestTheParticleRodIsHeldThroughTheEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import superdex  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("needs the [superdex] extra") from None

    def test_the_bridge_is_found_and_a_node_is_taken(self) -> None:
        sim, rod = _superdex_rod_scene()
        try:
            self.assertIsNotNone(rod.bridge, "on a host engine the rod must find its bridge")
            self.assertEqual(rod.published_states(None)["held"], 0.0)
            node = rod.hold(True, sim.pipeline.state_0, arm="right", at=0)
            self.assertEqual(node, int(rod.particles[0]), "a named node is taken as that node, past the reach gate")
            self.assertEqual(rod.published_states(None)["held"], 1.0)
            self.assertEqual(
                rod.hold(True, sim.pipeline.state_0, arm="right"), -1, "nothing within reach of a home hand"
            )
        finally:
            sim.close()

    def test_the_pinned_node_follows_a_retarget_and_the_far_end_does_not(self) -> None:
        sim, rod = _superdex_rod_scene()
        try:
            rod.hold(True, sim.pipeline.state_0, arm="right", at=0)
            q0 = rod.positions(sim.pipeline.state_0)
            held_z0, free_z0 = float(q0[0, 2]), float(q0[-1, 2])

            sim.run(20).sync()
            self.assertLess(
                abs(float(rod.positions(sim.pipeline.state_0)[0, 2]) - held_z0), 0.01, "pinned in place, it drifted"
            )

            # A world-space lift: ask for the node LIFT higher than it is.
            _aim(sim, rod, q0[0] + np.array([0.0, 0.0, LIFT]))
            sim.run(60).sync()
            q2 = rod.positions(sim.pipeline.state_0)
            self.assertGreater(float(q2[0, 2]) - held_z0, 0.6 * LIFT, "the pinned node did not follow its target")
            self.assertLess(float(q2[-1, 2]) - free_z0, 0.02, "the far end came up too -- the whole rod moved")
        finally:
            sim.close()

    def test_released_it_settles_back(self) -> None:
        sim, rod = _superdex_rod_scene()
        try:
            rod.hold(True, sim.pipeline.state_0, arm="right", at=0)
            z0 = float(rod.positions(sim.pipeline.state_0)[0, 2])
            _aim(sim, rod, rod.positions(sim.pipeline.state_0)[0] + np.array([0.0, 0.0, LIFT]))
            sim.run(60).sync()
            self.assertGreater(
                float(rod.positions(sim.pipeline.state_0)[0, 2]) - z0, 0.6 * LIFT, "precondition: lifted"
            )

            self.assertEqual(rod.hold(False), -1)
            self.assertEqual(rod.published_states(None)["held"], 0.0)
            sim.run(60).sync()
            self.assertLess(
                float(rod.positions(sim.pipeline.state_0)[0, 2]) - z0, 0.015, "released, it should settle back"
            )
        finally:
            sim.close()

    def test_a_reset_lets_go_and_a_hold_after_it_holds(self) -> None:
        """As the capsule chain's reset does, and as the engine's own already had.

        The engine's reset removed its pins while the rod kept its record of
        them, so ``held`` read 1.0 over nothing and the re-aim skipped the
        missing pin without a word. A hold taken after the reset has to be a
        live one: its node follows its target.
        """
        sim, rod = _superdex_rod_scene()
        try:
            rod.hold(True, sim.pipeline.state_0, arm="right", at=0)
            rod.hold(True, sim.pipeline.state_0, arm="right", at=1, slot=1)
            self.assertEqual({"held": 1.0, "anchor_held": 1.0}, rod.published_states(None))
            sim.reset()
            self.assertEqual({"held": 0.0, "anchor_held": 0.0}, rod.published_states(None), "the reset left a hold")

            rod.hold(True, sim.pipeline.state_0, arm="right", at=0)
            z0 = float(rod.positions(sim.pipeline.state_0)[0, 2])
            _aim(sim, rod, rod.positions(sim.pipeline.state_0)[0] + np.array([0.0, 0.0, LIFT]))
            sim.run(60).sync()
            lift = float(rod.positions(sim.pipeline.state_0)[0, 2]) - z0
            self.assertGreater(lift, 0.6 * LIFT, "a hold taken after a reset did not hold")

            # An engine that lets go by any other road: the hold says so.
            rod.bridge.unpin_all()
            self.assertEqual(0.0, rod.published_states(None)["held"])
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()
