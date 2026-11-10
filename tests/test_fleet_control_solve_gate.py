# SPDX-License-Identifier: Apache-2.0
"""The fleet's control stage solves IK only on a step some world is servoing.

The solve's result reaches ``joint_target_q`` through a per-world gate, its seeds
are re-gathered from ``joint_q`` every solve and the LM damping restarts on every
``step``, so a solve with every gate shut writes nothing and leaves nothing behind
for the next one. It cost 7.9 ms of a 60 ms control step, on steps where no world was servoing. These pin which
graph runs when, on stubs: no device, no simulator.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robosimstudio.solvers.ik.batched import RobotIKRig
from robosimstudio_bench.envs import fleet as fleet_module
from robosimstudio_bench.envs.fleet import FleetEnv

TIER = "fast"
"""No GPU. Gates every commit."""


class _Arm:
    def __init__(self, active: bool = False) -> None:
        self.active = active
        self.flushed = 0

    def flush(self) -> None:
        self.flushed += 1


def _rig(**arms: _Arm) -> RobotIKRig:
    rig = RobotIKRig.__new__(RobotIKRig)
    rig.arms = dict(arms)
    rig.driving = False
    return rig


class TestDriving(unittest.TestCase):
    """What the gates on the device say, which is what the stage has to agree with."""

    def test_it_follows_the_flush_not_the_host_mirror(self):
        arm = _Arm()
        rig = _rig(right=arm)
        arm.active = True
        self.assertFalse(rig.driving, "an enable not yet pushed has not reached the gates")
        rig.flush()
        self.assertTrue(rig.driving)
        arm.active = False
        self.assertTrue(rig.driving)
        rig.flush()
        self.assertFalse(rig.driving)

    def test_any_arm_is_enough(self):
        rig = _rig(right=_Arm(False), left=_Arm(True))
        rig.flush()
        self.assertTrue(rig.driving)


class _Stage(SimpleNamespace):
    def solve(self, joint_q, targets) -> None:
        self.calls.append("solve")

    def drive(self, targets) -> None:
        self.calls.append("drive")

    def apply(self, targets) -> None:
        self.calls.append("apply")


def _fleet(*, cuda: bool, driving: bool = False, ik: bool = True) -> tuple[FleetEnv, _Stage]:
    env = FleetEnv.__new__(FleetEnv)
    stage = _Stage(calls=[], driving=driving)
    env.ik = stage if ik else None
    env.segments = {"right": stage}
    env.grippers = {"right": stage}
    env.model = SimpleNamespace(device=SimpleNamespace(is_cuda=cuda))
    env.pipeline = SimpleNamespace(
        control=SimpleNamespace(joint_target_q=SimpleNamespace(ptr=1)),
        state_0=SimpleNamespace(joint_q=SimpleNamespace(ptr=2)),
    )
    env._control_key = None
    env._control_graph = None
    env._control_graph_unsolved = None
    return env, stage


class TestEager(unittest.TestCase):
    def test_a_step_with_nobody_servoing_skips_the_solve(self):
        env, stage = _fleet(cuda=False)
        env._control_stage()
        self.assertEqual(stage.calls, ["drive", "apply"])

    def test_a_step_with_a_servo_solves_first(self):
        env, stage = _fleet(cuda=False, driving=True)
        env._control_stage()
        self.assertEqual(stage.calls, ["solve", "drive", "apply"])

    def test_no_rig_is_no_solve(self):
        env, stage = _fleet(cuda=False, ik=False)
        env._control_stage()
        self.assertEqual(stage.calls, ["drive", "apply"])


class TestCaptured(unittest.TestCase):
    """Two graphs, both captured at the one moment the stage already was."""

    def setUp(self) -> None:
        self.launched: list[str] = []
        self.captures = 0

        @contextmanager
        def capturing(_device):
            self.captures += 1
            yield SimpleNamespace(graph=f"graph{self.captures}")

        patches = [
            mock.patch.object(fleet_module, "capturing", capturing),
            mock.patch.object(fleet_module.wp, "capture_launch", self.launched.append),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_the_solving_graph_is_the_first_and_runs_only_when_driving(self):
        env, stage = _fleet(cuda=True, driving=True)
        env._control_stage()
        self.assertEqual(self.launched, ["graph1"])
        stage.driving = False
        env._control_stage()
        stage.driving = True
        env._control_stage()
        self.assertEqual(self.launched, ["graph1", "graph2", "graph1"])
        self.assertEqual(self.captures, 2, "one capture of each per array pair")

    def test_the_stage_runs_eagerly_once_however_it_starts(self):
        """A drive advances a segment on the device, so the capture's eager pass must not be run twice."""
        env, stage = _fleet(cuda=True, driving=False)
        env._control_stage()
        # The eager pass, the solving graph's capture, the unsolved graph's capture.
        self.assertEqual(stage.calls, ["solve", "drive", "apply", "solve", "drive", "apply", "drive", "apply"])
        self.assertEqual(self.launched, ["graph2"])

    def test_new_arrays_recapture_both(self):
        env, _stage = _fleet(cuda=True)
        env._control_stage()
        env.pipeline.state_0.joint_q = SimpleNamespace(ptr=3)
        env._control_stage()
        self.assertEqual(self.captures, 4)
        self.assertEqual(self.launched, ["graph2", "graph4"])

    def test_a_failed_capture_runs_eagerly(self):
        @contextmanager
        def broken(_device):
            raise RuntimeError("no stream")
            yield

        with mock.patch.object(fleet_module, "capturing", broken), mock.patch("builtins.print"):
            env, stage = _fleet(cuda=True, driving=False)
            env._control_stage()
        self.assertEqual(self.launched, [])
        self.assertIsNone(env._control_graph)
        self.assertEqual(stage.calls[-2:], ["drive", "apply"])
        self.assertEqual(np.sum(np.array(stage.calls) == "solve"), 1, "only the capture's eager pass solved")


if __name__ == "__main__":
    unittest.main()
