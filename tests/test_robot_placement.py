# SPDX-License-Identifier: Apache-2.0
"""``rss.robot(..., at=...)`` puts the base where it says, in the work-surface frame.

Every constructor's ``at`` is in the cell's work-surface frame, and a robot's is
no exception: it is where the base goes. Left out, the robot stands at its own
mount, which is its spec's ``base_offset`` in that same frame. Three claims
hold that together, each measured on the wrist of a built scene rather than on
a field of the request:

* ``at`` equal to the spec's own ``base_offset`` builds the scene the default builds;
* moving ``at`` moves the wrist by the same vector, for a registered robot;
* a position is a pose with no rotation, and a yaw turns the arm about its base.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

TIER = "medium"
"""Builds scenes on a GPU."""

ROBOTS = ("panda", "yam_bimanual")


def _wrist(robot_name: str, at=None, cell: str = "bench_075") -> np.ndarray:
    """World position of the first arm's wrist body in a freshly built scene."""
    import robosimstudio as rss

    scene = rss.Scene(cell=cell, name="placement")
    scene.add(rss.robot(robot_name) if at is None else rss.robot(robot_name, at=at))
    sim = scene.build()
    position, _ = sim.arm().ee_pose(sim.state)
    return np.asarray(position, dtype=np.float64)


class TestARobotStandsWhereAtSays(unittest.TestCase):
    def test_at_its_own_mount_is_the_default_scene(self) -> None:
        from robosimstudio.robots.registry import get_embodiment

        for name in ROBOTS:
            with self.subTest(robot=name):
                own = get_embodiment(name).base_offset
                np.testing.assert_allclose(_wrist(name, at=own), _wrist(name), atol=1e-6)

    def test_moving_at_moves_the_robot_by_the_same_vector(self) -> None:
        from robosimstudio.robots.registry import get_embodiment

        step = np.array([0.12, -0.07, 0.05])
        for name in ROBOTS:
            with self.subTest(robot=name):
                own = np.asarray(get_embodiment(name).base_offset, dtype=np.float64)
                moved = (*(own[:3] + step), *own[3:])
                np.testing.assert_allclose(_wrist(name, at=moved) - _wrist(name), step, atol=1e-6)

    def test_a_position_is_a_pose_with_no_rotation(self) -> None:
        np.testing.assert_allclose(
            _wrist("panda", at=(-0.5, 0.1, 0.0)),
            _wrist("panda", at=(-0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0)),
            atol=1e-9,
        )

    def test_at_is_in_the_work_surface_frame(self) -> None:
        """The same ``at`` on a bench and on the bare floor differs by the bench's
        own frame: 750 mm up, and centred at world x = 0.6."""
        on_bench = _wrist("panda", at=(-0.5, 0.0, 0.0), cell="bench_075")
        on_floor = _wrist("panda", at=(-0.5, 0.0, 0.0), cell="void")
        np.testing.assert_allclose(on_bench - on_floor, (0.6, 0.0, 0.75), atol=1e-6)

    def test_a_yaw_turns_the_arm_about_its_base(self) -> None:
        half = math.sin(math.pi / 4)
        straight = _wrist("panda", at=(0.0, 0.0, 0.0), cell="void")
        turned = _wrist("panda", at=(0.0, 0.0, 0.0, 0.0, 0.0, half, half), cell="void")
        # A quarter turn about z carries (x, y) to (-y, x) and leaves the height alone.
        # To 10 um: the chain is single precision, and two solver versions land 1.1 um apart.
        np.testing.assert_allclose(turned, (-straight[1], straight[0], straight[2]), atol=1e-5)


class TestTwoWaysToSayWhere(unittest.TestCase):
    def test_at_and_a_binding_mount_together_are_refused(self) -> None:
        import robosimstudio as rss
        from robosimstudio.specs import Binding

        binding = Binding(scene="bench_075", embodiment="panda", mount=(0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0))
        scene = rss.Scene(cell="bench_075", binding=binding)
        scene.add(rss.robot("panda", at=(-0.5, 0.0, 0.0)))
        with self.assertRaisesRegex(ValueError, "both say where the robot stands"):
            _ = scene.request


if __name__ == "__main__":
    unittest.main()
