# SPDX-License-Identifier: Apache-2.0
"""A routine plays from its Run button, through the arm panels, and holds only what the fingers hold.

The demos finish their jobs from the robot's home pose with a routine, and the
claim the page makes about it is the same one it makes about a person: the
gizmo moves, the arm follows it, and a thing comes up because the fingers
closed on it. So the routine here is a pick written against the page's own
helpers, run on the rigid example by pressing its button, and held to the
numbers the hand-driven grasp test holds.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import open_session, tick

TIER = "medium"
"""Builds the rigid example and steps it through a page with no browser on it."""


def _pick(sim, session):
    from robosimstudio_studio.routine import grip, reach

    cube = np.asarray(sim["cube"].position, dtype=np.float64)

    def over(dz: float) -> np.ndarray:
        return cube + np.array([0.0, 0.0, dz])

    yield from reach(session, "right", over(0.06))
    yield from reach(session, "right", over(0.005))  # fingertips clear of the bench
    yield from grip(session, "right", 1.0)
    yield from reach(session, "right", over(0.15))


def _lost(sim, session):
    from robosimstudio_studio.routine import reach

    yield from reach(session, "right", (5.0, 5.0, 5.0), speed=10.0, settle=0.2)


def _hands_off(sim):
    for _ in range(25):
        yield


class TestARoutinePlaysFromItsButton(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, kwargs = _run("01_rigid/00_drag_things.py")
        cls.session = open_session(cls.sim, 8965, kwargs.get("page"))
        cls.session.add_routines({"Pick": _pick, "Lost": _lost, "Hands off": _hands_off})
        cls.panel = cls.session.panels[-1]
        tick(cls.session, 50)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def _play(self, name: str, limit: int = 3000) -> int:
        self.panel.requested = name
        tick(self.session, 1)
        steps = 0
        while self.panel.gen is not None and steps < limit:
            tick(self.session, 1)
            steps += 1
        return steps

    def test_1_the_pick_lifts_the_cube_by_its_fingers(self) -> None:
        lying = np.asarray(self.sim["cube"].position, dtype=np.float64)
        self._play("Pick")
        self.assertIn("done", self._said())
        rose = float(self.sim["cube"].position[2] - lying[2])
        self.assertGreater(rose, 0.13, "the cube stayed on the bench")
        self.assertAlmostEqual(0.05, self.sim.gripper("right").width(self.sim.state), delta=0.004)

    def test_2_a_hand_that_cannot_arrive_stops_the_routine_and_says_so(self) -> None:
        self._play("Lost", limit=600)
        self.assertIsNone(self.panel.gen)
        self.assertIn("short of where it was sent", self._said())

    def test_3_a_routine_that_drives_the_scene_itself_has_the_panels_held_off(self) -> None:
        self.panel.requested = "Hands off"
        tick(self.session, 1)
        self.assertTrue(all(p.suspended for p in self.session.arms.values()))
        while self.panel.gen is not None:
            tick(self.session, 1)
        self.assertFalse(any(p.suspended for p in self.session.arms.values()), "the hands were not given back")

    def test_4_stop_and_reset_end_it_where_it_is(self) -> None:
        self.panel.requested = "Pick"
        tick(self.session, 5)
        self.panel.stop_requested = True
        tick(self.session, 1)
        self.assertIsNone(self.panel.gen)
        self.panel.requested = "Pick"
        tick(self.session, 5)
        self.session.reset_requested = True
        tick(self.session, 1)
        self.assertIsNone(self.panel.gen)

    def _said(self) -> str:
        return str(self.panel._status.value)


if __name__ == "__main__":
    unittest.main()
