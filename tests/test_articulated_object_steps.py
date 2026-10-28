# SPDX-License-Identifier: Apache-2.0
"""A drawer driven open on MuJoCo opens, its cabinet stays bolted, and a reset closes it.

A SimReady-shaped cabinet on a fixed base, robot-less, on ``MUJOCO_500HZ``: one
drawer gets a position drive and a target 250 mm out. After two seconds it is
past 200 mm, read off where its bodies are; the other drawer is where it
started; the cabinet's root part has not moved a tenth of a millimetre; and a
reset brings the driven drawer back to its start. :mod:`tests.test_articulated_object`
checks the same entity on a CPU builder without stepping.

Two shapes Newton builds only through RSS's session-layer split, stepped the
same way: a linkage whose joints close a loop holds the loop (the closing joint
is no articulation's, and MuJoCo keeps it as an equality), and a jar in two
loose parts on a fixed base stays bolted while its lid rests on it.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.fixtures.simready import Joint, Part, write_stage

TIER = "medium"

TARGET = 0.25
"""Where the driven drawer is sent [m]."""


class TestADrivenDrawerOpens(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        cls._dir = tempfile.TemporaryDirectory()
        path = write_stage(
            Path(cls._dir.name) / "cabinet.usda",
            [
                Part("body_01", (0.0, 0.0, 0.3), (0.2, 0.2, 0.3)),
                Part("drawer_01", (0.21, 0.0, 0.45), (0.05, 0.15, 0.04)),
                Part("drawer_02", (0.21, 0.0, 0.25), (0.05, 0.15, 0.04)),
            ],
            [
                Joint("drawer_01", "prismatic", "body_01", "drawer_01", (0.21, 0.0, 0.45), "X", (0.0, 0.3)),
                Joint("drawer_02", "prismatic", "body_01", "drawer_02", (0.21, 0.0, 0.25), "X", (0.0, 0.3)),
            ],
        )
        scene = rss.Scene(cell="void", name="drawer", physics=rss.MUJOCO_500HZ)
        scene.add(
            rss.articulated(
                name="cabinet",
                asset=str(path),
                at=(0.5, 0.0, 0.0),
                joints=(rss.JointSpec(name="drawer_01", q=0.0, target_ke=200.0, target_kd=20.0),),
            )
        )
        cls.sim = scene.build()
        cls.handle = cls.sim.handles.objects["cabinet"]
        cls.root = cls.sim.pipeline.state_0.body_q.numpy()[cls.handle.body].copy()
        drawer = cls.handle.joints["drawer_01"]
        targets = cls.sim.pipeline.control.joint_target_q.numpy()
        targets[drawer.q_start] = TARGET
        cls.sim.pipeline.control.joint_target_q.assign(targets)
        cls.sim.run(100).sync()
        cls.after = cls.sim.pipeline.state_0.body_q.numpy().copy()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()

    def test_the_driven_drawer_opens(self) -> None:
        self.assertGreater(self.handle.joints["drawer_01"].coordinate_from_bodies(self.after), 0.2)

    def test_the_other_drawer_stays_shut(self) -> None:
        self.assertLess(abs(self.handle.joints["drawer_02"].coordinate_from_bodies(self.after)), 0.005)

    def test_the_cabinet_is_bolted(self) -> None:
        np.testing.assert_allclose(self.after[self.handle.body], self.root, atol=1.0e-4)
        self.assertTrue(np.isfinite(self.after).all())

    def test_a_reset_closes_it(self) -> None:
        self.sim.reset()
        body_q = self.sim.pipeline.state_0.body_q.numpy()
        self.assertLess(abs(self.handle.joints["drawer_01"].coordinate_from_bodies(body_q)), 1.0e-4)


class TestALoopAndALooseLidHold(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        cls._dir = tempfile.TemporaryDirectory()
        root = Path(cls._dir.name)
        pliers = write_stage(
            root / "pliers.usda",
            [
                Part("handleLeft_01", (0.0, 0.0, 0.1), (0.1, 0.01, 0.01)),
                Part("handleRight_01", (0.0, 0.03, 0.1), (0.1, 0.01, 0.01)),
                Part("jawLeft_01", (0.15, 0.0, 0.1), (0.03, 0.01, 0.01)),
                Part("jawRight_01", (0.15, 0.03, 0.1), (0.03, 0.01, 0.01)),
            ],
            [
                Joint(
                    "jawRight_01", "revolute", "handleRight_01", "jawRight_01", (0.12, 0.03, 0.1), "Z", (-15.0, 15.0)
                ),
                Joint(
                    "handleRight_01",
                    "revolute",
                    "handleLeft_01",
                    "handleRight_01",
                    (0.1, 0.015, 0.1),
                    "Z",
                    (-30.0, 30.0),
                ),
                Joint("jawLeft_01", "revolute", "handleLeft_01", "jawLeft_01", (0.12, 0.0, 0.1), "Z", (-15.0, 15.0)),
                Joint("jawLeft_01", "revolute", "jawRight_01", "jawLeft_01", (0.15, 0.015, 0.1), "Z", (-15.0, 15.0)),
            ],
        )
        jar = write_stage(
            root / "jar.usda",
            [
                Part("jar_01", (0.0, 0.0, 0.05), (0.04, 0.04, 0.05), mass=4.0),
                Part("lid_01", (0.0, 0.0, 0.111), (0.04, 0.04, 0.01)),
            ],
        )
        scene = rss.Scene(cell="void", name="loops", physics=rss.MUJOCO_500HZ)
        scene.add(rss.articulated(name="pliers", asset=str(pliers), at=(0.0, 0.0, 0.0)))
        scene.add(rss.articulated(name="jar", asset=str(jar), at=(0.6, 0.0, 0.0)))
        cls.sim = scene.build()
        model = cls.sim.pipeline.model
        cls.loop = next(j for j in range(model.joint_count) if model.joint_articulation.numpy()[j] < 0)
        cls.before = cls.sim.pipeline.state_0.body_q.numpy().copy()
        cls.sim.run(100).sync()
        cls.after = cls.sim.pipeline.state_0.body_q.numpy().copy()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dir.cleanup()

    def test_the_loop_joint_keeps_its_two_anchors_together(self) -> None:
        from robosimstudio.core.math import transform_matrix

        model = self.sim.pipeline.model
        parent, child = int(model.joint_parent.numpy()[self.loop]), int(model.joint_child.numpy()[self.loop])
        X_p, X_c = model.joint_X_p.numpy()[self.loop], model.joint_X_c.numpy()[self.loop]
        a = transform_matrix(self.after[parent]) @ transform_matrix(X_p)
        b = transform_matrix(self.after[child]) @ transform_matrix(X_c)
        self.assertLess(float(np.linalg.norm(a[:3, 3] - b[:3, 3])), 1.0e-3)
        self.assertTrue(np.isfinite(self.after).all())

    def test_the_jar_is_bolted_and_its_lid_stays_on_it(self) -> None:
        handle = self.sim.handles.objects["jar"]
        jar, lid = handle.parts["jar_01"], handle.parts["lid_01"]
        np.testing.assert_allclose(self.after[jar], self.before[jar], atol=1.0e-4)
        np.testing.assert_allclose(self.after[lid][:2], self.before[lid][:2], atol=5.0e-3)
        self.assertLess(abs(self.after[lid][2] - self.before[lid][2]), 5.0e-3)


if __name__ == "__main__":
    unittest.main()
