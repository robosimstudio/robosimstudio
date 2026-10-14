# SPDX-License-Identifier: Apache-2.0
"""Regressions for the action-space factory, OSC measurement, and task pairing.

Over stub arms and hand-built specs: what is tested is which errors surface,
how many host copies a control step makes, and which axes a pairing keeps.
"""

import typing
import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.solvers import OscController
from robosimstudio.solvers.control.action import ActionEEDelta, ActionOsc, make_action_spaces
from robosimstudio.specs import TaskSpec

TIER = "fast"
"""No GPU. Gates every commit."""


class _StubArm:
    """The slice of :class:`ArmIK` the action spaces and the OSC law read."""

    def __init__(self, width=7):
        self.model = SimpleNamespace(body_com=SimpleNamespace(numpy=lambda: np.zeros((1, 3))))
        self.ee_body = 0
        self.ee_offset = np.zeros(3)
        self.ee_offset_rot = np.array([0.0, 0.0, 0.0, 1.0])
        self.home_pose = (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
        self.arm_coords = list(range(width))


def _counting_state(position=(0.0, 0.0, 0.0)):
    """A ``State`` stand-in that counts host copies of each array."""
    body_q = np.array([[*position, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    body_qd = np.zeros((1, 6), dtype=np.float64)
    counts = {"body_q": 0, "body_qd": 0}

    def copy(name, array):
        def numpy():
            counts[name] += 1
            return array

        return numpy

    state = SimpleNamespace(
        body_q=SimpleNamespace(numpy=copy("body_q", body_q)),
        body_qd=SimpleNamespace(numpy=copy("body_qd", body_qd)),
    )
    return state, counts


class TestMakeActionSpacesSkipsOnlyUnsupportedModes(unittest.TestCase):
    def test_a_bad_option_raises_instead_of_dropping_the_mode(self):
        arms = [(_StubArm(), None)]
        with self.assertRaisesRegex(ValueError, "convention"):
            make_action_spaces(arms, control_hz=20.0, options={"ee_delta": {"convention": "algin"}})
        with self.assertRaisesRegex(ValueError, "gripper_range"):
            make_action_spaces(arms, control_hz=20.0, options={"osc_pose": {"gripper_range": (1.0, 0.0)}})

    def test_arms_of_unequal_width_still_skip_the_joint_mode(self):
        spaces = make_action_spaces([(_StubArm(7), None), (_StubArm(6), None)], control_hz=20.0)
        self.assertNotIn("joint_position", spaces)
        self.assertIn("ee_delta", spaces)
        self.assertIn("osc_pose", spaces)

    def test_a_shared_argument_reaches_only_the_modes_that_take_it(self):
        spaces = make_action_spaces([(_StubArm(), None)], control_hz=20.0, gripper_range=(-1.0, 1.0))
        self.assertIn("joint_position", spaces)
        self.assertEqual(spaces["ee_delta"].gripper_range, (-1.0, 1.0))
        self.assertEqual(spaces["osc_pose"].gripper_range, (-1.0, 1.0))

    def test_a_shared_argument_no_mode_takes_raises(self):
        with self.assertRaisesRegex(TypeError, "gripper_rnage"):
            make_action_spaces([(_StubArm(), None)], control_hz=20.0, gripper_rnage=(-1.0, 1.0))

    def test_unsupported_mode_is_on_the_facade(self):
        from robosimstudio.solvers import UnsupportedMode
        from robosimstudio.solvers.control import action

        self.assertIs(UnsupportedMode, action.UnsupportedMode)

    def test_no_control_rate_skips_osc(self):
        spaces = make_action_spaces([(_StubArm(), None)], control_hz=0.0)
        self.assertNotIn("osc_pose", spaces)
        self.assertIn("joint_position", spaces)


class TestGripperRangeIsShared(unittest.TestCase):
    def test_one_convention_for_both_cartesian_spaces(self):
        self.assertIs(ActionOsc.grip_command, ActionEEDelta.grip_command)
        self.assertIs(ActionOsc.grip_action, ActionEEDelta.grip_action)

    def test_both_validate_and_map_the_range(self):
        arms = [(_StubArm(), None)]
        for space in (
            ActionEEDelta(arms, gripper_range=(-1.0, 1.0)),
            ActionOsc(arms, control_hz=20.0, gripper_range=(-1.0, 1.0)),
        ):
            self.assertEqual(space.gripper_range, (-1.0, 1.0))
            self.assertAlmostEqual(space.grip_command(0.0), 0.5)
            self.assertAlmostEqual(space.grip_action(1.0), 1.0)
        with self.assertRaises(ValueError):
            ActionEEDelta(arms, gripper_range=(1.0, 1.0))


class TestOscAdvanceReadsTheStateOnce(unittest.TestCase):
    def test_advance_makes_one_body_q_copy_and_no_body_qd_copy(self):
        controller = OscController(_StubArm(), control_dt=1.0 / 20.0)
        state, counts = _counting_state()
        controller.reset(state)
        counts.update(body_q=0, body_qd=0)
        controller.advance([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], state)
        self.assertEqual(counts, {"body_q": 1, "body_qd": 0})

    def test_goal_is_unchanged(self):
        controller = OscController(_StubArm(), control_dt=1.0 / 20.0)
        state, _ = _counting_state((0.4, 0.1, 0.3))
        pos, quat, _, _ = controller.measure(state)
        goal_pos, goal_quat = controller.goal([0.5, 0.0, 0.0, 0.0, 0.0, 0.2], state)
        expected_pos, expected_quat = controller._goal_from(pos, quat, [0.5, 0.0, 0.0, 0.0, 0.0, 0.2])
        np.testing.assert_allclose(goal_pos, expected_pos)
        np.testing.assert_allclose(goal_quat, expected_quat)


class TestCreateRendererHints(unittest.TestCase):
    def test_type_hints_resolve(self):
        from robosimstudio.solvers.render import create_renderer

        self.assertIn("cameras", typing.get_type_hints(create_renderer))


class TestWithPairingKeepsAPinnedGripper(unittest.TestCase):
    def test_an_empty_gripper_means_the_tasks_own(self):
        pinned = TaskSpec(name="t", scene="cellA", embodiment="robotX").with_pairing(gripper="hand_g")
        moved = pinned.with_pairing(scene="cellB")
        self.assertEqual((moved.scene, moved.embodiment, moved.gripper), ("cellB", "robotX", "hand_g"))
        self.assertEqual(pinned.with_pairing().gripper, "hand_g")
        self.assertIn(("cellB", "robotX", "hand_g"), moved.binding_keys)


if __name__ == "__main__":
    unittest.main()
