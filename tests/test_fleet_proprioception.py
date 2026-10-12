# SPDX-License-Identifier: Apache-2.0
"""A world's proprioception is the embodiment's own reading of that world's rows.

The single-world observation is ``embodiment.observe(state)``: the tool
centre through the arm's IK offset, and the MEASURED gripper closure. The
fleet formatter used to hand back the end-effector body and the bank's
command -- so every height a graph built on the tool centre was a hand's length
off, and a close on a gear read exactly like a close on air. No GPU: a fake
embodiment with the two readers.
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from robosimstudio_policy_agent.fleet.obs_format import format_runtime_obs, world_observation

TIER = "fast"

EE = 2
TCP = np.array([0.0, 0.0, 0.165])


class _IK:
    """The template solver, which is where the fleet reads an arm's tool offsets from."""

    ee_offset = np.asarray(TCP, dtype=np.float64)
    ee_offset_rot = np.array([0.0, 0.0, 0.0, 1.0])

    def ee_pose(self, state):
        tf = state.body_q.numpy()[EE]
        return tf[:3] + TCP, tf[3:7]


class _Gripper:
    def __init__(self, open_q=0.0, closed_q=0.8):
        self.open_q, self.closed_q = open_q, closed_q

    def closed_fraction(self, state):
        q = state.joint_q.numpy()
        return float(np.clip((q[7] - self.open_q) / (self.closed_q - self.open_q), 0.0, 1.0))


class _Snapshot:
    def __init__(self, coords, body_q):
        self._coords, self._body = coords, body_q

    def coords(self, world):
        return self._coords

    def body(self, world):
        return self._body


def _fleet(command=1.0):
    embodiment = types.SimpleNamespace(
        arms={"right": types.SimpleNamespace(arm_coords=list(range(7)), ee_body=EE)},
        iks={"right": _IK()},
        grippers={"right": _Gripper()},
        hands={"right": types.SimpleNamespace(tcp_offset=tuple(TCP))},
    )
    bank = types.SimpleNamespace(commands=lambda: np.array([command]))
    return types.SimpleNamespace(
        scene=types.SimpleNamespace(embodiment=embodiment), grippers={"right": bank}, episodes=[None]
    )


class TestAWorldReportsItsToolCentreAndItsMeasuredClosure(unittest.TestCase):
    def test_tool_centre_not_flange(self):
        body = np.zeros((EE + 1, 7))
        body[EE] = [0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]
        obs = world_observation(_fleet(), 0, _Snapshot(np.zeros(8), body))
        np.testing.assert_allclose(obs["proprio"]["right/ee_pose"][:3], [0.5, 0.0, 0.465])

    def test_measured_closure_not_command(self):
        coords = np.zeros(8)
        coords[7] = 0.4  # the fingers stalled half-way: something is in the hand
        obs = world_observation(_fleet(command=1.0), 0, _Snapshot(coords, np.zeros((EE + 1, 7))))
        self.assertAlmostEqual(obs["proprio"]["right/gripper"], 0.5)
        self.assertAlmostEqual(obs["proprio"]["right/gripper_command"], 1.0)
        # And the runtime's vector carries the OPEN fraction of the measurement.
        snapshot = types.SimpleNamespace(obs=obs, frames={})
        out = format_runtime_obs(snapshot, arm_count=1)
        self.assertAlmostEqual(out["robot_joint_pos_0"][-1], 0.5)


if __name__ == "__main__":
    unittest.main()
