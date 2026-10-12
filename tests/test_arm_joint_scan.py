# SPDX-License-Identifier: Apache-2.0
"""An arm's joints are the arm's, not everything sharing its prefix.

A robot whose :class:`~robosimstudio.specs.ArmSpec` names no joints explicitly
finds them by **label prefix** -- ``left_`` claims ``left_joint1`` through
``left_joint6``. That is how four of the shipped robots are indexed and it works,
right up until something else in the scene is called ``left_something``.

A free-jointed object is exactly that something. ``rss.box(name="left_cube")``
adds ``left_cube_free_joint``, the scan claims it as a seventh arm joint, and
what comes out is ``joint_dof_mask partially masks joint 16 (FREE)`` from inside
Newton's IK -- a true statement about a mask that says nothing about the name
that caused it, three layers away from the line that chose it.

**No shipped task can reach this**, because a benchmark's objects are named for
what they are and its pairings are declared in advance. It is reachable the
moment a scene can be assembled by hand, which is what this refactor is for -- so
the fix is at the source (an arm has no free joints, ever) and this is what holds
it there.
"""

from __future__ import annotations

import unittest

TIER = "medium"
"""Compiles scenes with a robot in them."""

PREFIXED = ("left_cube", "right_block", "left_thing")
"""Names that begin with an arm's prefix on the shipped bimanual YAMs."""


class TestAFreeJointIsNeverAnArmJoint(unittest.TestCase):
    def _arms(self, *names: str):
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075")
        scene.add(rss.robot("yam_bimanual"))
        for i, name in enumerate(names):
            scene.add(rss.box(name=name, half=(0.02,) * 3, at=(0.0, 0.15 - 0.15 * i, 0.10)))
        sim = scene.build()
        return sim.scene.embodiment.arms

    def test_a_neutrally_named_object_is_not_claimed(self) -> None:
        """The control: six joints an arm, whatever else is in the scene."""
        arms = self._arms("cube_a", "cube_b")
        for name, index in arms.items():
            with self.subTest(arm=name):
                self.assertEqual(6, len(index.arm_coords), f"{name} claimed {len(index.arm_coords)} joints")

    def test_an_object_named_for_an_arm_is_not_claimed_either(self) -> None:
        """The bug. Before the fix this raised out of Newton's IK during the
        build, with a message about a DOF mask."""
        arms = self._arms(*PREFIXED)
        for name, index in arms.items():
            with self.subTest(arm=name):
                self.assertEqual(
                    6,
                    len(index.arm_coords),
                    f"{name} claimed {len(index.arm_coords)} joints; a free-jointed object was swept in",
                )

    def test_the_arms_still_solve(self) -> None:
        """Not just indexed correctly -- usable. An arm indexed with a phantom
        seventh joint fails at the solver, so the index alone is not the claim."""
        import numpy as np

        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075")
        scene.add(rss.robot("yam_bimanual"))
        scene.add(rss.box(name="left_cube", half=(0.02,) * 3, at=(0.0, 0.15, 0.10)))
        sim = scene.build()
        sim.run(30).sync()
        seed = np.asarray(sim.pipeline.snapshot.joint_q, dtype=np.float64)
        solved = sim.arm("left").solve_q(seed, [0.6, 0.15, 0.90], (0.0, 0.0, -1.0))
        self.assertEqual(len(seed), len(solved))


if __name__ == "__main__":
    unittest.main()
