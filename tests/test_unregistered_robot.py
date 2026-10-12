# SPDX-License-Identifier: Apache-2.0
"""A robot loaded from a file, registered nowhere, builds and solves.

pyroki's ``Robot.from_urdf(...)`` has a counterpart now. ``rss.robot("panda")``
needs a registered :class:`~robosimstudio.specs.EmbodimentSpec`; this hands the
compiler the spec itself -- the same file the registered Panda reads, as an
inline spec under a name the registry has never heard of -- and checks that the
arm is indexed, IK solves, and a control mode exists. **A bare arm needs no
registry at all**: no gripper is named, so nothing is looked up.

The one thing the registry still owns is a *hand*. A gripper's finger bodies,
travel and tool-centre point are measured facts, and ``GRIPPERS`` is where they
live; a spec that names one resolves it there exactly as a registered robot
does.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "medium"
"""Compiles a scene with a robot in it."""

NAME = "inline_panda"
"""Deliberately not a registry key."""


def _inline_panda():
    """The registered Panda's file and arm, restated without the registration."""
    from robosimstudio.robots.embodiments.panda import PANDA_HOME_Q
    from robosimstudio.specs import ActuationSpec, ArmSpec, AssetSpec, EmbodimentSpec

    return EmbodimentSpec(
        name=NAME,
        asset=AssetSpec(path="embodiments/panda/panda_standalone.xml"),
        base_offset=(-0.6, -0.115, 0.132, 0.0, 0.0, 0.0, 1.0),
        collapse_fixed_joints=False,
        hull_colliders=True,
        arms=(
            ArmSpec(
                name="right",
                joint_names=tuple(f"robot0_joint{i}" for i in range(1, 8)),
                ee_body_name="robot0_link7",
                home_q=PANDA_HOME_Q,
            ),
        ),
        arm_actuation=ActuationSpec(target_ke=650.0, target_kd=100.0),
    )


class TestAnInlineRobot(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        cls.spec = _inline_panda()
        scene = rss.Scene(cell="bench_075", name="inline")
        scene.add(rss.box(name="cube", half=(0.02,) * 3, at=(0.05, 0.10, 0.10)))
        cls.placed = scene.add(rss.robot(cls.spec))
        cls.sim = scene.build()

    def test_it_was_never_registered(self) -> None:
        from robosimstudio.robots.registry import EMBODIMENTS

        self.assertNotIn(NAME, EMBODIMENTS)
        self.assertEqual(NAME, self.placed.name)

    def test_the_inline_spec_is_the_one_that_built(self) -> None:
        """Not a lookup that happened to find something similar."""
        self.assertIs(self.spec, self.sim.scene.embodiment.spec)

    def test_the_arm_is_indexed_and_driven(self) -> None:
        self.assertIn("right", self.sim.handles.arms)
        self.assertEqual(7, len(self.sim.handles.arms["right"].arm_coords))
        self.assertIn("joint_position", self.sim.controllers)

    def test_ik_solves(self) -> None:
        self.sim.run(30).sync()
        seed = np.asarray(self.sim.pipeline.snapshot.joint_q, dtype=np.float64)
        solved = self.sim.arm().solve_q(seed, [0.65, 0.10, 0.90], (0.0, 0.0, -1.0))
        self.assertEqual(int(self.sim.model.joint_coord_count), len(solved))
        landed, _ = self.sim.arm().fk_ee(np.asarray(solved, dtype=np.float32))
        self.assertLess(float(np.linalg.norm(landed - np.array([0.65, 0.10, 0.90]))), 0.02)


if __name__ == "__main__":
    unittest.main()
