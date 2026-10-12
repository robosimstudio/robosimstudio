# SPDX-License-Identifier: Apache-2.0
"""Every segment of a cable answers a click, and the wrench is sized for the rope.

A rod is one entity and many bodies, and the two facts are carried by one
handle: ``body`` is the first segment, for the pose readout and the ``Attach
handle`` dropdown, and ``bodies`` is the whole run. ``examples/10_rope/rod/entity.py``'s
``handle()`` says so in as many words -- *"``bodies`` is the run, which is what
lets a click land on any segment"*.

:class:`~robosimstudio_studio.panels.PanelObject` builds the reverse lookup a
click is resolved through, and it is the half that has to agree. Two ways it can
silently stop agreeing, both regressions this file exists to catch:

**The lookup maps ``body`` alone.** Then one capsule out of twenty-eight answers
a click and the rest of the rope is dead surface -- and it looks like nothing at
all from the page, because a click that resolves to no name is answered by
returning early rather than by any complaint.

**``chain_mass`` loses the chain.** The dragger's clamp is sized per attach from
the mass it is told about. A segment of this rope weighs about a third of a
gram, and a handle clamped to *its* 20 g of acceleration cannot lift the rope
hanging off it: it stalls a hand's width off the floor and reads as a solver
bug. The number that has to reach the dragger is the whole chain's.

Both went unnoticed once because every test of the object panel used a box,
which is a chain of one and cannot tell the two lookups apart.
"""

from __future__ import annotations

import unittest

TIER = "medium"
"""Compiles a scene and starts a viser server on a spare port."""

LENGTH = 0.6
RADIUS = 0.011
SEGMENTS = 28


def _sim():
    import robosimstudio as rss

    scene = rss.Scene(cell="void", name="cable_grab")
    scene.add(rss.robot("yam_bimanual"))
    scene.add(
        rss.rod(
            name="rope",
            length=LENGTH,
            radius=RADIUS,
            at=(0.0, -LENGTH / 2, RADIUS + 0.06),
            along=(0, 1, 0),
            segments=SEGMENTS,
        )
    )
    return scene.build(control_mode="joint_position")


def _session(sim, port: int):
    from robosimstudio_studio.session import StudioConfig, StudioSession

    return StudioSession(sim, config=StudioConfig(port=port, panels="drive"))


class TestEverySegmentIsGrabbable(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _sim()
        cls.session = _session(cls.sim, 8941)
        cls.objects = cls.session.objects
        cls.bodies = tuple(int(b) for b in cls.sim.handles.objects["rope"].bodies)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_handle_owns_the_whole_run(self) -> None:
        """The premise. If this fails the panel is not the thing at fault."""
        handle = self.sim.handles.objects["rope"]
        self.assertEqual(SEGMENTS, len(self.bodies))
        self.assertEqual(int(handle.body), self.bodies[0], "`body` should be the first segment")

    def test_every_segment_resolves_to_the_rope(self) -> None:
        """The regression: mapping `body` alone leaves 27 of 28 capsules dead."""
        missing = [b for b in self.bodies if self.objects.by_body.get(b) != "rope"]
        self.assertEqual([], missing, f"{len(missing)} of {SEGMENTS} segments answer a click with nothing")

    def test_the_rope_is_named_once(self) -> None:
        """A chain is an ordinary object, so it is already in `names`; adding it
        again for being a chain puts it in the dropdown twice."""
        self.assertEqual(1, self.objects.names.count("rope"))

    def test_the_wrench_is_sized_for_the_rope_not_the_capsule(self) -> None:
        import numpy as np

        mass = self.sim.model.body_mass.numpy()
        whole = float(mass[list(self.bodies)].sum())
        segment = float(mass[self.bodies[len(self.bodies) // 2]])
        self.assertIn("rope", self.objects.chain_mass, "the dragger would be sized by one capsule")
        self.assertTrue(np.isclose(whole, self.objects.chain_mass["rope"], rtol=1e-6))
        self.assertGreater(whole, 5.0 * segment, "the fixture is too short to tell the two masses apart")

    def test_a_click_on_the_middle_takes_hold_there(self) -> None:
        """Not on a nominal first body: dragging a rope by its end and by its
        middle are different manipulations."""
        middle = self.bodies[len(self.bodies) // 2]
        self.objects.click_body(middle)
        self.objects.drain()
        try:
            held = self.objects.handles.find(middle)
            self.assertIsNotNone(held, "a click on the middle segment took hold of nothing")
            self.assertTrue(held.live)
        finally:
            for slot in list(self.objects.handles.live):
                self.objects._release(slot)

    def test_attach_handle_takes_the_middle_rather_than_the_far_end(self) -> None:
        """The dropdown's own button. A chain is in `objects` too, so a branch
        that checks that first hands back segment 0 -- the far end of the rope,
        which is the one place a hand would not choose."""
        self.objects.selected = "rope"
        self.objects.attach_requested = True
        self.objects.drain()
        try:
            live = list(self.objects.handles.live)
            self.assertEqual(1, len(live))
            self.assertEqual(self.bodies[len(self.bodies) // 2], int(live[0].key))
        finally:
            for slot in list(self.objects.handles.live):
                self.objects._release(slot)


if __name__ == "__main__":
    unittest.main()
