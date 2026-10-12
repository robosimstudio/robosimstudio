# SPDX-License-Identifier: Apache-2.0
"""Clicking a robot link takes hold of it, and pulling actually moves the arm.

Half of "click and drag the object or the robot" did not work. A click on a
link resolved to the arm and then stopped: ``PanelObject`` built ``by_body``
from the objects table and the suite's rigid chains, so a robot body named
nothing and ``_attach`` returned early. The gizmo the page did offer sat where
``build`` first put it, at the home pose, so the obvious follow-up -- drag what
you clicked -- meant hunting for a control somewhere else.

Two claims here, and the second is the one that is easy to get wrong.

**A wrench is not enough.** The arm is position-servoed: MuJoCo drives
``joint_q`` towards ``joint_target_q`` every substep. Hang a PD wrench off a
link and leave the target where it was, and the two fight -- the arm deflects a
little, and snaps back the moment the handle is let go. ``suspend()`` does not
help, because it stops *updating* the target rather than the servoing.
:meth:`PanelArm._take_hand` is the mechanism: called every step while a handle
is live, the target *is* the achieved pose, so there is no restoring error left
and the wrench has only inertia and the joint limits to work against.

**Each arm's own links.** ``ArmHandle.shapes`` cannot say which arm a link
belongs to -- it is filled for contact exclusion and on ``yam_bimanual`` both
arms list the same 112 shapes. Slackening by that would let go of the arm
nobody is touching, and stop its gizmo working mid-drag.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "medium"
"""Builds a bimanual scene on a GPU and steps it."""

CELL = "void"
EMBODIMENT = "yam_bimanual"


def _scene():
    import robosimstudio as rss

    scene = rss.Scene(cell=CELL, name="drag")
    scene.add(rss.robot(EMBODIMENT))
    scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0.0, 0.0, 0.10)))
    return scene.build(control_mode="joint_position")


def _page(sim):
    """The two panels that matter, over one stub page, wired as the session wires them."""
    from robosimstudio_studio.panels import PanelArm, PanelObject
    from tests.fixtures.viser_stubs import _StubGui, _StubSession

    gui = _StubGui()
    session = _StubSession(sim, gui)
    objects = PanelObject(session, handles=2)
    objects.build(gui)
    session.objects = objects
    arms = {name: PanelArm(session, name) for name in sim.handles.arms}
    for panel in arms.values():
        panel.build(gui)
    session.arms = arms
    return session, objects, arms


class TestARobotLinkIsGrabbable(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _scene()
        cls.session, cls.objects, cls.arms = _page(cls.sim)

    def test_each_arm_owns_its_own_links(self) -> None:
        """Disjoint chains, where ``shapes`` would have said both arms own everything."""
        chains = self.objects.arm_bodies
        self.assertEqual(sorted(chains), sorted(self.sim.handles.arms))
        for name, bodies in chains.items():
            with self.subTest(arm=name):
                self.assertTrue(bodies, f"{name} got no links")
        left, right = (set(chains[n]) for n in sorted(chains))
        self.assertEqual(set(), left & right, "the two arms share a link")

    def test_the_links_are_in_the_body_table(self) -> None:
        for name, bodies in self.objects.arm_bodies.items():
            for body in bodies:
                with self.subTest(arm=name, body=body):
                    self.assertEqual(name, self.objects.by_body.get(int(body)))

    def test_a_click_on_a_link_puts_a_handle_on_it(self) -> None:
        arm = sorted(self.objects.arm_bodies)[0]
        body = self.objects.arm_bodies[arm][0]
        report = self.objects.click_body(int(body))
        self.assertIn(arm, report)
        self.objects.drain()
        self.assertIsNotNone(self.objects.handles.find(int(body)), "no handle was placed")
        self.assertTrue(self.objects.holding_arm(arm))
        self.objects.handles.release_all()
        for dragger in self.objects.draggers:
            dragger.detach()

    def test_holding_one_arm_does_not_hold_the_other(self) -> None:
        held, other = sorted(self.objects.arm_bodies)
        self.objects.click_body(int(self.objects.arm_bodies[held][0]))
        self.objects.drain()
        try:
            self.assertTrue(self.objects.holding_arm(held))
            self.assertFalse(self.objects.holding_arm(other), "the untouched arm was reported held")
        finally:
            self.objects.handles.release_all()
            for dragger in self.objects.draggers:
                dragger.detach()

    def test_the_wrench_is_sized_by_the_whole_arm(self) -> None:
        """A wrist link weighs grams; a handle clamped to its own mass stalls."""
        mass = self.sim.model.body_mass.numpy()
        for name, bodies in self.objects.arm_bodies.items():
            with self.subTest(arm=name):
                whole = float(mass[np.asarray(bodies, dtype=np.int64)].sum())
                self.assertAlmostEqual(whole, self.objects.chain_mass[name], places=6)
                self.assertGreater(whole, float(mass[bodies[0]]))


class TestTheArmStopsResistingWhileHeld(unittest.TestCase):
    """The claim that makes the tug work rather than merely register."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _scene()
        cls.session, cls.objects, cls.arms = _page(cls.sim)
        cls.arm = sorted(cls.objects.arm_bodies)[0]
        cls.coords = np.asarray(cls.sim.handles.arms[cls.arm].arm_coords, dtype=np.int64)

    def _drain_and_step(self, count: int = 1) -> None:
        for _ in range(count):
            self.objects.drain()
            for panel in self.arms.values():
                panel.drain()
            self.sim.step_teleop()

    def _target_error(self) -> float:
        """How far the arm's target is from where the arm actually is [rad]."""
        self.sim.pipeline.sync()
        q = np.asarray(self.sim.state.joint_q.numpy(), dtype=np.float64)[self.coords]
        target = np.asarray(self.sim.control.joint_target_q.numpy(), dtype=np.float64)[self.coords]
        return float(np.abs(target - q).max())

    def _arm_q(self) -> np.ndarray:
        self.sim.pipeline.sync()
        return np.asarray(self.sim.state.joint_q.numpy(), dtype=np.float64)[self.coords].copy()

    def test_the_target_tracks_the_arm_while_a_link_is_held(self) -> None:
        """Two halves, and neither is enough alone.

        The arm has to actually *move* -- a wrench that does nothing would leave
        the target trivially correct -- and the target has to stay with it. What
        is left over is one control step of lag, because the target is written
        before the step that then advances the arm; with the target frozen at
        its pre-grab value the same drag leaves ~0.1 rad of error instead.
        """
        body = int(self.objects.arm_bodies[self.arm][0])
        self.objects.click_body(body)
        self._drain_and_step(3)
        try:
            self.assertTrue(self.objects.holding_arm(self.arm))
            before = self._arm_q()
            slot = self.objects.handles.find(body)
            slot.position = np.asarray(slot.position, dtype=np.float64) + np.array([0.0, 0.0, 0.12])
            self.objects.draggers[slot.index].set_target(slot.position, slot.quat)
            self._drain_and_step(40)

            moved = float(np.abs(self._arm_q() - before).max())
            self.assertGreater(moved, 0.05, "the wrench did not move the arm, so tracking is untested")
            self.assertLess(self._target_error(), 0.02, "the arm is still servoing to a stale target")
            self.assertLess(self._target_error(), moved / 4.0, "the target is lagging the arm, not tracking it")
            self.assertTrue(self.sim.state_is_finite())
        finally:
            self.objects.handles.release_all()
            for dragger in self.objects.draggers:
                dragger.detach()

    def test_the_arm_stays_where_it_was_left(self) -> None:
        """Letting go hands driving back mid-air rather than snapping to a stale pose."""
        body = int(self.objects.arm_bodies[self.arm][0])
        self.objects.click_body(body)
        self._drain_and_step(2)
        slot = self.objects.handles.find(body)
        slot.position = np.asarray(slot.position, dtype=np.float64) + np.array([0.0, 0.0, 0.10])
        self.objects.draggers[slot.index].set_target(slot.position, slot.quat)
        self._drain_and_step(40)
        self.sim.pipeline.sync()
        held_q = np.asarray(self.sim.state.joint_q.numpy(), dtype=np.float64)[self.coords].copy()

        self.objects.handles.release_all()
        for dragger in self.objects.draggers:
            dragger.detach()
        self._drain_and_step(30)
        self.sim.pipeline.sync()
        after = np.asarray(self.sim.state.joint_q.numpy(), dtype=np.float64)[self.coords]
        self.assertLess(float(np.abs(after - held_q).max()), 0.20, "the arm sprang back to a pose from before the drag")


class TestClickingAnArmBringsItsGizmo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _scene()
        cls.session, cls.objects, cls.arms = _page(cls.sim)

    def test_selecting_an_arm_reseats_its_cursor_on_the_hand(self) -> None:
        from robosimstudio_studio.selection import Subject

        name = sorted(self.arms)[0]
        panel = self.arms[name]
        panel.cursor_pos = np.array([9.0, 9.0, 9.0])
        panel.on_select(Subject(kind="arm", name=name, handle=self.sim.handles.arms[name]))
        expected, _ = panel.ik.ee_pose(self.sim.state)
        np.testing.assert_allclose(panel.cursor_pos, expected, atol=1e-6)

    def test_selecting_another_arm_leaves_this_one_alone(self) -> None:
        from robosimstudio_studio.selection import Subject

        mine, theirs = sorted(self.arms)
        panel = self.arms[mine]
        panel.cursor_pos = np.array([9.0, 9.0, 9.0])
        panel.on_select(Subject(kind="arm", name=theirs, handle=self.sim.handles.arms[theirs]))
        np.testing.assert_allclose(panel.cursor_pos, np.array([9.0, 9.0, 9.0]))


class TestADragOnARealPageActuallyMoves(unittest.TestCase):
    """The claim every other test here assumed, on a session that has reset.

    ``StudioSession`` builds its panels and *then* resets, and a reset clears
    every hook installed after the scene was built -- deliberately, so a policy
    that pins a tab cannot still be holding it next episode
    (``tests/test_hook_lifetime.py``). The panels' draggers were on that side of
    the line and were never put back, so on a real page the handles appeared,
    the gizmo moved, and the thing it was holding sat still. Every test that
    drove a panel directly missed it, because none of them reset.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss
        from robosimstudio_studio.session import StudioConfig, StudioSession

        scene = rss.Scene(cell=CELL, name="realdrag")
        scene.add(rss.robot(EMBODIMENT))
        scene.add(rss.box(name="cube", half=(0.03,) * 3, at=(0.0, 0.0, 0.10)))
        scene.add(rss.sheet(name="towel", size=(0.25, 0.25), cells=(12, 12), at=(-0.125, -0.125, 0.45)))
        cls.sim = scene.build(control_mode="joint_position")
        cls.session = StudioSession(cls.sim, config=StudioConfig(port=8937, panels="drive"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_hooks_survive_the_sessions_own_reset(self) -> None:
        pipeline = self.sim.pipeline
        self.assertTrue(list(pipeline.hooks.rigid_force), "the wrenches were dropped by the reset")
        self.assertTrue(list(pipeline.hooks.post_solve), "the pins were dropped by the reset")

    def test_dragging_a_box_moves_the_box(self) -> None:
        objects = self.session.objects
        body = int(next(b for b, name in objects.by_body.items() if name == "cube"))
        objects.click_body(body)
        objects.drain()
        slot = objects.handles.find(body)
        self.assertIsNotNone(slot, "no handle landed on the cube")
        before = np.asarray(self.sim["cube"].position, dtype=np.float64).copy()
        for _ in range(90):
            slot.position = np.asarray(slot.position, dtype=np.float64) + np.array([0.0, 0.0, 0.002])
            objects.drain()
            self.session.env.step_teleop()
        self.sim.pipeline.sync()
        lifted = float(self.sim["cube"].position[2] - before[2])
        self.assertGreater(lifted, 0.05, "the wrench was in the schedule but the box did not move")

    def test_dragging_a_towel_moves_the_towel(self) -> None:
        deformables = self.session.deformables
        self.assertIsNotNone(deformables)
        q0 = np.asarray(self.sim["towel"].particle_q, dtype=np.float64)
        deformables.click_at("towel", q0[int(np.argmax(q0[:, 2]))])
        deformables.drain()
        self.assertTrue(deformables.handles.live, f"the grab missed by {deformables.miss}")
        slot = deformables.handles.live[0]
        for _ in range(90):
            slot.position = np.asarray(slot.position, dtype=np.float64) + np.array([0.0, 0.0, 0.002])
            deformables.drain()
            self.session.env.step_teleop()
        self.sim.pipeline.sync()
        held = np.asarray(deformables.draggers[slot.index].grabbed_indices)
        q1 = np.asarray(self.sim["towel"].particle_q, dtype=np.float64)
        self.assertGreater(float((q1[held, 2] - q0[held, 2]).max()), 0.05, "the pinned patch did not follow")


if __name__ == "__main__":
    unittest.main()
