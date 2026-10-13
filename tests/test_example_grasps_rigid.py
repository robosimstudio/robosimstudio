# SPDX-License-Identifier: Apache-2.0
"""The grasps the first examples invite, made the way their pages make them.

The getting-started scripts, the robot added from a file, the page with a panel
of its own and the leader arm all put a cube in front of a hand, and each now
stages ``rss.MUJOCO_500HZ`` so that closing the hand on it is a grasp. This holds
that to numbers, as :mod:`tests.test_example_grasps` does for ``01_rigid``: each
case runs the example's own ``main()``, drives its arm panel as the gizmo and
the Grip button do (:mod:`tests.fixtures.grasp`), and asks that the cube rose
with the hand and did not slide in the fingers. Nothing is pinned to a hand, so
a pass is friction's.

The panel example also tosses its cube, and that is checked here because it
broke with the move: MuJoCo steps joint coordinates and ignores ``body_qd``,
which is what the toss used to write. And the first task's page is carried
through by hand, gizmo and Grip only, until its goal turns true.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.fixtures.examples import _load, _run, _stub_studio
from tests.fixtures.grasp import drag_to_grasp, go, open_session, tick

TIER = "medium"
"""Builds each example and steps it through a page with no browser on it."""

LIFT = 0.15
"""How far each gesture raises the hand after the close [m]."""


class _Lift:
    """Down onto the cube, **Close**, up 150 mm, hold 2 s; the subclass names the example."""

    EXAMPLE = ""
    PORT = 0
    ARM = "right"
    HOVER = 0.0
    """0 for a hand that starts over the cube; else go over it this high first, then down."""
    TURN: tuple[float, ...] | None = None
    """A tool orientation to turn the cursor to before the gesture [xyzw], or ``None`` to keep home's."""
    STARTS_OVER = True

    @classmethod
    def open(cls):
        """``(sim, the page callback to install or None)``."""
        sim, kwargs = _run(cls.EXAMPLE)
        return sim, kwargs.get("page")

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim, page = cls.open()
        cls.session = open_session(cls.sim, cls.PORT, page)
        tick(cls.session, 50)  # a released cube lands in under a second
        cube = cls.sim["cube"]
        cls.lying = np.asarray(cube.position, dtype=np.float64)
        if cls.TURN is not None:
            cls.session.arms[cls.ARM].cursor_quat = np.asarray(cls.TURN, dtype=np.float64)
            go(cls.session, cls.ARM, cls.lying + np.array([0.0, 0.0, 2 * cls.HOVER]))
        cls.grasp = drag_to_grasp(cls.session, cls.ARM, lambda: cube.position, lift=LIFT, hover=cls.HOVER)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_cube_comes_up_with_the_hand(self) -> None:
        self.assertTrue(self.grasp.finite)
        self.assertLess(self.grasp.reached, 0.010, "the hand did not get to the top of the lift")
        self.assertGreater(self.grasp.rise, 0.08, "the cube stayed behind")

    def test_it_does_not_slide_in_the_fingers(self) -> None:
        self.assertLess(self.grasp.drift, 0.010)
        self.assertAlmostEqual(0.05, self.grasp.width, delta=0.004, msg="the fingers are not on its faces")

    def test_the_hand_starts_where_the_page_says(self) -> None:
        """Straight down for a hand that starts over the cube; else the cube is in reach."""
        hand = self.session.arms[self.ARM].ik.home_pose[0]
        if self.STARTS_OVER:
            self.assertLess(float(np.linalg.norm(hand[:2] - self.lying[:2])), 0.005)
            self.assertGreater(float(hand[2] - self.lying[2]), 0.10)
        else:
            self.assertGreater(float(np.linalg.norm(hand[:2] - self.lying[:2])), 0.05, "it is a drag over, then down")


class TestHelloWorldLiftsTheCube(_Lift, unittest.TestCase):
    """``00_getting_started/00_hello_world``: the cube lands under the Panda's hand."""

    EXAMPLE = "00_getting_started/00_hello_world.py"
    PORT = 9210


class TestCoreConceptsLiftsTheCube(_Lift, unittest.TestCase):
    """``00_getting_started/01_core_concepts``: the page is the bench of the two places."""

    EXAMPLE = "00_getting_started/01_core_concepts.py"
    PORT = 9211


class TestTheAddedRobotLiftsTheCube(_Lift, unittest.TestCase):
    """``04_extending/00_add_a_robot``: a robot declared inline grips like a registered one."""

    EXAMPLE = "04_extending/00_add_a_robot.py"
    PORT = 9212


class TestThePanelPageLiftsTheCube(_Lift, unittest.TestCase):
    """``05_studio/00_your_own_panel``: the cube in the middle, a drag over it and down."""

    EXAMPLE = "05_studio/00_your_own_panel.py"
    PORT = 9213
    HOVER = 0.10
    STARTS_OVER = False


class TestTheLeaderArmFollowerLiftsTheCube(_Lift, unittest.TestCase):
    """``06_teleop/00_leader_arm --simulate``: the follower, with its teleoperation gains, holds the cube.

    The leader's page takes the arms off their gizmos, so it is not installed:
    the arm panel stands in for the operator's hand, the tool turned to point
    down, and drives the same follower in the same scene."""

    EXAMPLE = "06_teleop/00_leader_arm.py"
    PORT = 9214
    HOVER = 0.06
    TURN = (0.0, 1.0, 0.0, 0.0)  # the YAM's tool pointing down, jaws closing along world y
    STARTS_OVER = False

    @classmethod
    def open(cls):
        with _stub_studio() as opened:
            _load(cls.EXAMPLE).main(["--simulate"])
        ((sim, _),) = opened
        return sim, None


class TestTheFirstTaskByHand(unittest.TestCase):
    """``07_tasks/00_your_first_task``: the gizmo carries the cube into the square, and the goal turns true.

    The task's page, as :func:`robosimstudio.view` opens it: its goal panel is
    there to turn, and the hand is the only thing that moves the cube."""

    def test_pick_carry_and_let_go_in_the_square(self) -> None:
        env, _ = _run("07_tasks/00_your_first_task.py")
        session = open_session(env, 9215)
        try:
            self.assertIsNotNone(session.goal_panel, "the goal beside the scene")
            tick(session, 25)
            cube = env["cube"]
            lift = 0.10
            grasp = drag_to_grasp(session, "right", lambda: cube.position, lift=lift, hover=0.10, hold=0.5)
            self.assertGreater(grasp.rise, 0.08, "the cube stayed behind")
            self.assertLess(grasp.drift, 0.010)
            self.assertFalse(env.evaluate()[-1]["success"], "carried, not yet placed")
            # Over the square's middle at the carry height, then down by the lift. The
            # square is x 0.05..0.15, y -0.20..-0.10 on a bench whose middle is world (0.6, 0).
            over = np.array([0.6 + 0.10, -0.15, cube.position[2]])
            self.assertLess(go(session, "right", over), 0.010)
            self.assertLess(go(session, "right", over - np.array([0.0, 0.0, lift])), 0.010)
            session.arms["right"].grip_request = 0.0  # Open
            tick(session, 25)
            go(session, "right", over)
            tick(session, 60)
            self.assertTrue(env.evaluate()[-1]["success"], f"the cube is at {cube.position.round(3)}")
        finally:
            session.teardown()


class TestThePanelTossesTheCube(unittest.TestCase):
    """``05_studio/00_your_own_panel``: the button throws the cube up, and the slider says how hard."""

    def test_a_toss_rises_as_high_as_its_speed_says(self) -> None:
        from tests.fixtures.viser_stubs import _StubGui

        sim, kwargs = _run("05_studio/00_your_own_panel.py")

        class _Studio:
            def __init__(self) -> None:
                self.gui = _StubGui()
                self.steps: list = []

            def each_step(self, fn) -> None:
                self.steps.append(fn)

        studio = _Studio()
        kwargs["page"](studio)
        readout = studio.gui.find("height above the bench (mm)")

        def step(n: int) -> list[float]:
            seen = []
            for _ in range(n):
                sim.run(1).sync()
                for fn in studio.steps:
                    fn()
                seen.append(readout.value)
            return seen

        rest = step(100)[-1]  # released 200 mm up; on the bench well inside two seconds
        self.assertAlmostEqual(25.0, rest, delta=2.0, msg="a 50 mm cube resting on the bench")
        # A free flight's peak is v^2 / 2g; measured 112.7 mm at 1.5 m/s and 453.2 at 3.0.
        for speed in (1.5, 3.0):
            with self.subTest(speed=speed):
                studio.gui.find("toss speed (m/s)").value = speed
                studio.gui.find("Toss the cube").fire()
                peak = max(step(60)) - rest
                self.assertAlmostEqual(1000.0 * speed**2 / (2 * 9.81), peak, delta=0.05 * peak)
                step(100)  # down again before the next one
        self.assertTrue(sim.state_is_finite())
        sim.close()


if __name__ == "__main__":
    unittest.main()
