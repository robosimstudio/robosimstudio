# SPDX-License-Identifier: Apache-2.0
"""The ``gripped`` stall test, on hands whose fingers run either way.

A mirrored prismatic pair closes towards zero; a Robotiq 2F-85's driver opens at
zero and closes upward. The stall test read "measured trails command" as
``measured - commanded > 0``, which is that only on the first kind: on a hand
closing upward every stall read as a negative lag, and the linear gap derived
from the raw coordinate was a different number again, so ``gripped`` could not
come true.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.domains.rigid.measures.jaws import _jaw_stall
from robosimstudio.scene.handles import ArmHandle

TIER = "fast"
"""No GPU: the handler over a handle and plain arrays."""

BOX = {"half": [0.0325, 0.01, 0.05]}
"""A part 20 mm across its narrowest side."""


def _snapshot(arm: ArmHandle, measured, commanded) -> SimpleNamespace:
    joint_q = np.zeros(4)
    target = np.zeros(4)
    for coord, m, c in zip(arm.gripper_coords, measured, commanded, strict=True):
        joint_q[coord], target[coord] = m, c
    ctx = SimpleNamespace(state=SimpleNamespace(joint_q=SimpleNamespace(numpy=lambda: joint_q)), joint_target_q=target)
    handles = SimpleNamespace(work_arm=arm.name, arms={arm.name: arm})
    return SimpleNamespace(ctx=ctx, handles=handles)


class TestTheStallReadsTheHandsOwnDirection(unittest.TestCase):
    def test_a_pair_closing_towards_zero(self):
        """Mirrored convention: open at +/-0.04, shut at 0, span 0.08. Holding 20 mm leaves 0.01 per finger."""
        arm = ArmHandle(name="a", gripper_coords=(1, 2), gripper_open=0.04, gripper_span=0.08)
        out = _jaw_stall(_snapshot(arm, measured=(0.01, -0.01), commanded=(0.0, 0.0)), BOX)
        self.assertAlmostEqual(0.02, out["gap"])
        self.assertGreater(out["lag"], 0.0)
        self.assertTrue(out["gripped"])

    def test_a_linear_hand_closing_upward(self):
        """Open at 0, shut at +0.04: the same hold, in the opposite sign."""
        arm = ArmHandle(
            name="a",
            gripper_coords=(1, 2),
            gripper_open=0.04,
            gripper_span=0.08,
            gripper_open_q=(0.0, 0.0),
            gripper_closed_q=(0.04, 0.04),
        )
        out = _jaw_stall(_snapshot(arm, measured=(0.03, 0.03), commanded=(0.04, 0.04)), BOX)
        self.assertAlmostEqual(0.02, out["gap"])
        self.assertAlmostEqual(0.01, out["lag"])
        self.assertTrue(out["gripped"])

    def test_empty_jaws_that_reached_their_command_are_not_holding(self):
        arm = ArmHandle(
            name="a",
            gripper_coords=(1, 2),
            gripper_open=0.04,
            gripper_span=0.08,
            gripper_open_q=(0.0, 0.0),
            gripper_closed_q=(0.04, 0.04),
        )
        self.assertFalse(_jaw_stall(_snapshot(arm, measured=(0.04, 0.04), commanded=(0.04, 0.04)), BOX)["gripped"])

    def test_a_geared_hand_fails_closed_rather_than_reading_a_wrong_curve(self):
        """A 2F-85's gap is not linear in its driver angle, as ``GripperSpec.jaw_gap`` refuses too."""
        arm = ArmHandle(
            name="a",
            gripper_coords=(1, 2),
            gripper_open=0.8,
            gripper_span=0.085,
            gripper_open_q=(0.0, 0.0),
            gripper_closed_q=(0.8, 0.8),
            gripper_geared=True,
        )
        out = _jaw_stall(_snapshot(arm, measured=(0.5, 0.5), commanded=(0.8, 0.8)), BOX)
        self.assertFalse(out["gripped"])
        self.assertTrue(math.isnan(out["gap"]))
        self.assertAlmostEqual(0.3, out["lag"], msg="the stall itself still reads positive")
        with self.assertRaises(ValueError):
            arm.jaw_gap((0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
