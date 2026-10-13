# SPDX-License-Identifier: Apache-2.0
"""Closing a hand, and letting go of a handle, without going looking for a control.

Both halves of the interaction were mouse gestures that finished somewhere else
on the page. Closing a gripper meant unrolling a fold, finding a slider and
dragging it -- three actions for one bit of information, on a folder that
started shut. Letting go of a handle meant finding ``Release last`` two folds
deep, while *taking* hold of something was a single click: the page's own
gesture was asymmetric.

Two changes, checked here.

**A grip is a button**, beside the slider that stays for the grip which is
neither open nor shut, in a folder that now opens by default.

**A click on what you are already holding puts it down.** The toggle every
direct manipulation interface uses, and it costs no new control. Rigid handles
key on the body; a cloth handle keys on its sheet and matches by distance,
because a patch of cloth is not one index.

The threading contract is the one every panel obeys: the button's callback fires
on viser's threadpool and may only assign, so the grip it asks for is applied in
``drain`` -- which is also where the slider it has to move can be written.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "medium"
"""Compiles scenes and starts a viser server on a spare port."""


def _sim(cloth: bool = False):
    import robosimstudio as rss

    scene = rss.Scene(cell="void", name="shortcuts")
    scene.add(rss.robot("yam_bimanual"))
    scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0.0, 0.0, 0.10)))
    if cloth:
        scene.add(rss.sheet(name="towel", size=(0.2, 0.2), cells=(8, 8), at=(-0.1, -0.1, 0.4)))
    return scene.build(control_mode="joint_position")


def _session(sim, port: int):
    from robosimstudio_studio.session import StudioConfig, StudioSession

    return StudioSession(sim, config=StudioConfig(port=port, panels="drive"))


class TestTheGrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _sim()
        cls.session = _session(cls.sim, 8931)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_one_click_shuts_the_hand_and_one_opens_it(self) -> None:
        """What the slider took a drag to say."""
        panel = self.session.arms[sorted(self.session.arms)[0]]
        for asked, expected in ((1.0, 1.0), (0.0, 0.0)):
            panel.grip_request = asked
            panel.drain()
            self.assertEqual(expected, panel.gripper_command)
            self.assertEqual(expected, panel._grip.value, "the slider did not follow the button")

    def test_the_request_is_applied_on_the_main_thread_not_in_the_callback(self) -> None:
        """The panel contract: a callback assigns, ``drain`` acts. Asserted because
        the slider has to move too, and writing a widget from viser's threadpool
        is the one thing the contract forbids."""
        panel = self.session.arms[sorted(self.session.arms)[0]]
        panel.gripper_command = 0.0
        panel.grip_request = 1.0
        self.assertEqual(0.0, panel.gripper_command, "the request was acted on before drain")
        panel.drain()
        self.assertIsNone(panel.grip_request)
        self.assertEqual(1.0, panel.gripper_command)

    def test_the_grip_button_is_there_and_writes_only_a_request(self) -> None:
        import inspect

        from robosimstudio_studio import panels

        source = inspect.getsource(panels.PanelArm.build)
        self.assertIn('add_button_group("Grip"', source)
        self.assertIn('setattr(self, "grip_request"', source)

    def test_the_arm_folder_opens_by_default(self) -> None:
        """Half of what a person came here to do should not start behind a fold."""
        import inspect

        from robosimstudio_studio import panels

        source = inspect.getsource(panels.PanelArm.build)
        self.assertIn("expand_by_default=True", source)


class TestStoppingAPolicyHoldsWhatItHeld(unittest.TestCase):
    """``Stop`` gives the arm back. It must not open the hand on the way.

    The arm panels are not drained while a policy drives -- they write the
    ``joint_target_q`` that is the policy's to write -- so their
    ``gripper_command`` sits at whatever the slider last said, which on a page
    nobody dragged it on is 0.0: wide open. The first drain after the hand-off
    used to write that back, so stopping a run to look at what it had grasped
    dropped the object in the act of stopping, and the grip looked like it had
    failed when what failed was the hand-off.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _sim()
        cls.session = _session(cls.sim, 8935)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def _arm(self):
        return self.session.arms[sorted(self.session.arms)[0]]

    def test_the_hand_keeps_the_grip_the_policy_left(self) -> None:
        panel = self._arm()
        panel.grip_request = 0.0  # the page's own idea of the hand: open
        panel.drain()

        panel.suspend()  # a policy takes the arm
        panel.gripper.command = 0.8  # and closes on something
        panel.drain()
        self.assertAlmostEqual(0.8, panel.gripper.command, msg="a suspended panel wrote the fingers anyway")

        panel.resume()  # Stop
        panel.drain()
        self.assertAlmostEqual(0.8, panel.gripper.command, places=5, msg="stopping flung the hand open")
        self.assertAlmostEqual(0.8, panel.gripper_command, places=5)
        self.assertAlmostEqual(0.8, float(panel._grip.value), places=5, msg="the slider does not say what is held")

    def test_a_reset_still_opens_the_hand(self) -> None:
        """The other half: a fresh episode starts from an open hand, and the
        grip adopted a moment earlier must not survive it -- a policy run resets
        *after* stopping the previous one."""
        panel = self._arm()
        panel.suspend()
        panel.gripper.command = 0.8
        panel.resume()
        panel.on_reset({}, {})
        panel.drain()
        self.assertEqual(0.0, panel.gripper_command, "a reset did not open the hand")
        self.assertAlmostEqual(0.0, panel.gripper.command, msg="the reset's open hand was overwritten")


class TestClickingAgainLetsGo(unittest.TestCase):
    def test_a_second_click_on_a_body_releases_it(self) -> None:
        sim = _sim()
        session = _session(sim, 8933)
        try:
            objects = session.objects
            body = int(sorted(objects.by_body)[0])
            objects.click_body(body)
            objects.drain()
            held = objects.handles.find(body)
            self.assertIsNotNone(held)
            self.assertTrue(held.live)

            objects.click_body(body)
            objects.drain()
            self.assertFalse(objects.handles.find(body) is not None and objects.handles.find(body).live)
            self.assertEqual([], objects.handles.live)
        finally:
            session.teardown()

    def test_a_second_click_on_a_cloth_grip_releases_it(self) -> None:
        sim = _sim(cloth=True)
        session = _session(sim, 8934)
        try:
            deformables = session.deformables
            self.assertIsNotNone(deformables)
            point = np.asarray(sim["towel"].particle_q[0], dtype=np.float64)
            deformables.click_at("towel", point)
            deformables.drain()
            self.assertTrue(deformables.handles.live, f"the grab missed by {deformables.miss}")

            deformables.click_at("towel", point)
            deformables.drain()
            self.assertEqual([], deformables.handles.live, "clicking the same grip did not let go")
        finally:
            session.teardown()

    def test_clicking_elsewhere_on_the_same_sheet_adds_a_second_grip(self) -> None:
        """The toggle must not turn a two-handed hold into a one-handed one."""
        sim = _sim(cloth=True)
        session = _session(sim, 8935)
        try:
            deformables = session.deformables
            q = np.asarray(sim["towel"].particle_q, dtype=np.float64)
            near, far = q[int(np.argmin(q[:, 0]))], q[int(np.argmax(q[:, 0]))]
            for point in (near, far):
                deformables.click_at("towel", point)
                deformables.drain()
            self.assertEqual(2, len(deformables.handles.live), "the far corner released the near grip")
        finally:
            session.teardown()


class TestTheClothSlots(unittest.TestCase):
    """What a cloth handle's slot holds when it is seized again or handed to another grip."""

    def test_regrip_seizes_again_rather_than_letting_go(self) -> None:
        """The newest handle sits at distance 0 from itself, which the click toggle reads as "let go"."""
        sim = _sim(cloth=True)
        session = _session(sim, 9321)
        try:
            deformables = session.deformables
            point = np.asarray(sim["towel"].particle_q[0], dtype=np.float64)
            deformables.click_at("towel", point)
            deformables.drain()
            self.assertEqual(1, len(deformables.handles.live), f"the grab missed by {deformables.miss}")
            slot = deformables.handles.newest()
            deformables.regrip_requested = True
            deformables.drain()
            self.assertEqual([slot], deformables.handles.live, "Re-grip here let go of the grip")
            self.assertTrue(deformables.draggers[slot.index].active, "and it holds the cloth again")
        finally:
            session.close()

    def test_a_recycled_slot_lets_go_of_its_engine_pin(self) -> None:
        """A full set recycles the oldest handle; an engine-side pin left on it would be aimed at the new grip."""
        import types

        sim = _sim(cloth=True)
        session = _session(sim, 9322)
        try:
            deformables = session.deformables
            q = np.asarray(sim["towel"].particle_q, dtype=np.float64)
            u, v = q[:, 0] + q[:, 1], q[:, 0] - q[:, 1]
            corners = [q[int(np.argmin(u))], q[int(np.argmax(u))], q[int(np.argmin(v))]]  # three distinct corners
            deformables.click_at("towel", corners[0])
            deformables.drain()
            oldest = deformables.handles.newest()
            released = []
            deformables.pins[oldest.index] = types.SimpleNamespace(
                nodes=np.arange(3), aim=lambda position, quat: True, release=lambda: released.append(True)
            )
            for point in corners[1:]:  # the second fills the set, the third recycles the oldest slot
                deformables.click_at("towel", point)
                deformables.drain()
            self.assertEqual([True], released, "the recycled slot's engine pin was not let go")
            self.assertNotIn(oldest.index, deformables.pins)
            self.assertTrue(deformables.draggers[oldest.index].active, "the slot holds its new grip through Newton")
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
