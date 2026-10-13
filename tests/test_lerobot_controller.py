# SPDX-License-Identifier: Apache-2.0
"""Feetech leader-arm calibration: raw servo ticks in, a joint_position action out.

``build_action`` is the pure half -- ticks to radians and to a normalised grip
command, with no bus and no env. :class:`LerobotFeetechController` is the thin
integration half: it reads a bus, calls ``build_action``, and applies the
result through whichever ``ActionSpace`` ``env.use_control("joint_position")``
returns, exactly as the studio's gizmo-drag teleop already does.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from robosimstudio.robots.teleop import lerobot_controller
from robosimstudio.robots.teleop.lerobot_controller import (
    ArmCalibration,
    GripperCalibration,
    JointCalibration,
    LeaderUnavailable,
    LerobotFeetechController,
    apply_arm_actions,
    build_action,
    load_calibration,
    parse_leader_ports,
    resolve_leaders,
    serial_from_port,
)

TIER = "fast"
"""No GPU, no hardware."""


class TestBuildAction(unittest.TestCase):
    def test_a_single_joint_converts_ticks_to_radians(self):
        cal = JointCalibration(name="shoulder", ticks_per_radian=651.9)
        arm = ArmCalibration(joints=(cal,))
        action = build_action({"shoulder": 651.9}, [arm])
        np.testing.assert_allclose(action, [1.0, 0.0])

    def test_sign_and_offset_apply_before_scaling(self):
        cal = JointCalibration(name="elbow", ticks_per_radian=100.0, sign=-1.0, offset_ticks=50.0)
        arm = ArmCalibration(joints=(cal,))
        action = build_action({"elbow": 150.0}, [arm])
        np.testing.assert_allclose(action, [-1.0, 0.0])

    def test_a_missing_gripper_reports_a_zero_command(self):
        cal = JointCalibration(name="j0", ticks_per_radian=1.0)
        arm = ArmCalibration(joints=(cal,))
        action = build_action({"j0": 0.0}, [arm])
        self.assertEqual(action[-1], 0.0)

    def test_the_gripper_command_is_normalised_between_open_and_closed(self):
        joint = JointCalibration(name="j0", ticks_per_radian=1.0)
        gripper = GripperCalibration(name="grip", open_ticks=0.0, closed_ticks=100.0)
        arm = ArmCalibration(joints=(joint,), gripper=gripper)
        action = build_action({"j0": 0.0, "grip": 25.0}, [arm])
        self.assertAlmostEqual(action[-1], 0.25)

    def test_the_gripper_command_clips_outside_the_calibrated_range(self):
        joint = JointCalibration(name="j0", ticks_per_radian=1.0)
        gripper = GripperCalibration(name="grip", open_ticks=0.0, closed_ticks=100.0)
        arm = ArmCalibration(joints=(joint,), gripper=gripper)
        action = build_action({"j0": 0.0, "grip": 500.0}, [arm])
        self.assertEqual(action[-1], 1.0)

    def test_two_arms_concatenate_joints_then_grip_per_arm(self):
        left = ArmCalibration(joints=(JointCalibration(name="l0", ticks_per_radian=1.0),))
        right = ArmCalibration(joints=(JointCalibration(name="r0", ticks_per_radian=1.0),))
        action = build_action({"l0": 2.0, "r0": 3.0}, [left, right])
        np.testing.assert_allclose(action, [2.0, 0.0, 3.0, 0.0])


class _FakeActionSpace:
    def __init__(self) -> None:
        self.applied = None

    def apply(self, action, control, state) -> None:
        self.applied = (list(action), control, state)


class _FakeEnv:
    def __init__(self, space) -> None:
        self._space = space
        self.control = object()
        self.state = object()

    def use_control(self, mode):
        assert mode == "joint_position"
        return self._space


class _FakeBus:
    def __init__(self, readings) -> None:
        self.readings = readings
        self.disconnected = False

    def sync_read(self, data_name, *, normalize=True):
        assert data_name == "Present_Position"
        # The real feetech bus normalises Present_Position to the servo's
        # configured range by default; the calibration here expects raw
        # ticks, so poll() must ask for them plain.
        assert normalize is False, "poll() must read raw ticks, not normalized units"
        return dict(self.readings)

    def disconnect(self) -> None:
        self.disconnected = True


class TestLerobotFeetechController(unittest.TestCase):
    def test_read_action_reads_raw_ticks_without_touching_the_env(self):
        """The half a caller driving several controllers into one shared,
        multi-arm action space needs: the reading, with nothing applied."""
        cal = JointCalibration(name="j0", ticks_per_radian=2.0)
        arm = ArmCalibration(joints=(cal,))
        bus = _FakeBus({"j0": 6.0})
        controller = LerobotFeetechController(bus, [arm])

        action = controller.read_action()

        np.testing.assert_allclose(action, [3.0, 0.0])

    def test_polling_writes_a_joint_position_action_through_the_active_control_mode(self):
        cal = JointCalibration(name="j0", ticks_per_radian=1.0)
        arm = ArmCalibration(joints=(cal,))
        bus = _FakeBus({"j0": 4.0})
        space = _FakeActionSpace()
        env = _FakeEnv(space)
        controller = LerobotFeetechController(bus, [arm])

        action = controller.poll(env)

        np.testing.assert_allclose(action, [4.0, 0.0])
        applied_action, control, state = space.applied
        np.testing.assert_allclose(applied_action, [4.0, 0.0])
        self.assertIs(control, env.control)
        self.assertIs(state, env.state)

    def test_close_disconnects_the_bus(self):
        bus = _FakeBus({})
        controller = LerobotFeetechController(bus, [])
        controller.close()
        self.assertTrue(bus.disconnected)


class TestLoadCalibration(unittest.TestCase):
    """A measured rig's ticks-to-radians facts, read from a JSON file.

    The calibration itself is a fact about one physical leader arm -- centres,
    directions, counts per turn -- that nothing in this repository can know, so
    it lives in a file a session points at rather than in code.
    """

    def _write(self, data) -> str:
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        with handle:
            json.dump(data, handle)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_one_arm_with_a_gripper(self) -> None:
        path = self._write(
            {
                "left": {
                    "joints": [{"name": "l_shoulder", "ticks_per_radian": 651.9, "sign": -1.0, "offset_ticks": 2048.0}],
                    "gripper": {"name": "l_grip", "open_ticks": 2048.0, "closed_ticks": 3000.0},
                }
            }
        )
        arms = load_calibration(path)
        self.assertEqual(set(arms), {"left"})
        joint = arms["left"].joints[0]
        self.assertEqual(joint.name, "l_shoulder")
        self.assertEqual(joint.sign, -1.0)
        self.assertEqual(joint.offset_ticks, 2048.0)
        self.assertEqual(arms["left"].gripper.name, "l_grip")

    def test_a_joint_may_omit_sign_and_offset(self) -> None:
        path = self._write({"right": {"joints": [{"name": "r0", "ticks_per_radian": 100.0}]}})
        joint = load_calibration(path)["right"].joints[0]
        self.assertEqual(joint.sign, 1.0)
        self.assertEqual(joint.offset_ticks, 0.0)

    def test_an_arm_with_no_gripper_is_none(self) -> None:
        path = self._write({"right": {"joints": [{"name": "r0", "ticks_per_radian": 1.0}]}})
        self.assertIsNone(load_calibration(path)["right"].gripper)

    def test_two_arms_are_both_read(self) -> None:
        path = self._write(
            {
                "left": {"joints": [{"name": "l0", "ticks_per_radian": 1.0}]},
                "right": {"joints": [{"name": "r0", "ticks_per_radian": 1.0}]},
            }
        )
        self.assertEqual(set(load_calibration(path)), {"left", "right"})


class TestNamingALeader(unittest.TestCase):
    """``name:port`` pairs, and the serial a calibration is looked up by."""

    def test_pairs_are_split_on_the_first_colon(self) -> None:
        ports = parse_leader_ports("right:/dev/ch340-5AE6080553, left:/dev/ch340-5AE6080669")
        self.assertEqual(ports, {"right": "/dev/ch340-5AE6080553", "left": "/dev/ch340-5AE6080669"})

    def test_nothing_named_is_an_empty_table_not_an_error(self) -> None:
        self.assertEqual(parse_leader_ports(""), {})

    def test_an_entry_with_no_port_is_refused_in_a_sentence(self) -> None:
        with self.assertRaises(LeaderUnavailable) as caught:
            parse_leader_ports("right")
        self.assertIn("name:port", str(caught.exception))

    def test_the_serial_is_read_out_of_the_udev_name(self) -> None:
        self.assertEqual(serial_from_port("/dev/ch340-5AE6080553"), "5AE6080553")

    def test_a_port_that_does_not_follow_the_convention_is_its_own_key(self) -> None:
        self.assertEqual(serial_from_port("/dev/ttyACM0"), "ttyACM0")


class TestResolvingLeaders(unittest.TestCase):
    """Everything that can be known before a bus is opened, each as one sentence."""

    def setUp(self) -> None:
        self.ENTRY = {"joints": [{"name": "r0", "ticks_per_radian": 651.9}]}

    def _file(self, data) -> str:
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        with handle:
            json.dump(data, handle)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def _sentence(self, *args, **kwargs) -> str:
        with self.assertRaises(LeaderUnavailable) as caught:
            resolve_leaders(*args, **kwargs)
        message = str(caught.exception)
        self.assertNotIn("\n", message, "one sentence, not a report")
        return message

    def test_no_leader_named_says_how_to_name_one(self) -> None:
        self.assertIn("<arm>:<port>", self._sentence({}, self._file({})))

    def test_a_missing_calibration_file_names_the_tool_that_writes_one(self) -> None:
        message = self._sentence({"right": "/dev/ch340-ABC"}, "/nonexistent/leader.json")
        self.assertIn("/nonexistent/leader.json", message)
        self.assertIn("scripts/calibrate_leader_arms.py", message)

    def test_a_serial_the_file_does_not_hold_says_which_it_does(self) -> None:
        """The sample file describes somebody else's arm, and this is how that reads."""
        message = self._sentence({"right": "/dev/ch340-MINE"}, self._file({"THEIRS": self.ENTRY}))
        self.assertIn("'MINE'", message)
        self.assertIn("THEIRS", message)

    def test_a_port_with_nothing_on_it_says_to_plug_the_arm_in(self) -> None:
        message = self._sentence({"right": "/dev/ch340-ABC"}, self._file({"ABC": self.ENTRY}))
        self.assertIn("/dev/ch340-ABC", message)
        self.assertIn("plug the leader arm in", message)

    def test_a_missing_driver_names_the_extra(self) -> None:
        with mock.patch.object(lerobot_controller.importlib.util, "find_spec", return_value=None):
            message = self._sentence({"right": "ABC"}, self._file({"ABC": self.ENTRY}), check_ports=False)
        self.assertIn("uv sync --extra lerobot-feetech", message)

    def test_a_leader_that_checks_out_carries_its_own_calibration(self) -> None:
        with mock.patch.object(lerobot_controller.importlib.util, "find_spec", return_value=object()):
            leaders = resolve_leaders({"right": "/x/ch340-ABC"}, self._file({"ABC": self.ENTRY}), check_ports=False)
        leader = leaders["right"]
        self.assertEqual((leader.name, leader.port, leader.serial), ("right", "/x/ch340-ABC", "ABC"))
        self.assertEqual(leader.calibration.joints[0].name, "r0")

    def test_the_shipped_sample_is_a_bimanual_six_joint_pair(self) -> None:
        """What ``scripts/calibration/yam_leader.json`` describes is what the
        follower takes: six joints and a grip per arm."""
        sample = Path(__file__).resolve().parents[1] / "scripts" / "calibration" / "yam_leader.json"
        arms = load_calibration(sample)
        self.assertEqual(len(arms), 2)
        for arm in arms.values():
            self.assertEqual(len(arm.joints), 6)
            self.assertIsNotNone(arm.gripper)


class _TwoArmSpace:
    per_arm = 3

    def __init__(self) -> None:
        self.applied = None

    def target_action(self, control):
        return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)

    def apply(self, action, control, state) -> None:
        self.applied = np.asarray(action).copy()


class _TwoArmEnv:
    def __init__(self) -> None:
        self.space = _TwoArmSpace()
        self.control = object()
        self.state = object()
        self.scene = type("Scene", (), {"embodiment": type("Emb", (), {"iks": {"left": None, "right": None}})()})()

    def use_control(self, mode):
        assert mode == "joint_position"
        return self.space


class TestApplyingArmActions(unittest.TestCase):
    """One combined action, in the control's own arm order."""

    def test_a_named_arm_gets_its_pose_and_the_other_keeps_its_target(self) -> None:
        env = _TwoArmEnv()
        action = apply_arm_actions(env, {"right": [1.0, 2.0, 0.5]})
        np.testing.assert_allclose(env.space.applied, [0.1, 0.2, 0.3, 1.0, 2.0, 0.5])
        np.testing.assert_allclose(action, env.space.applied)

    def test_no_pose_at_all_is_a_no_op_for_every_arm(self) -> None:
        env = _TwoArmEnv()
        apply_arm_actions(env, {})
        np.testing.assert_allclose(env.space.applied, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_an_arm_the_robot_does_not_have_is_ignored(self) -> None:
        env = _TwoArmEnv()
        apply_arm_actions(env, {"middle": [9.0, 9.0, 9.0], "left": [1.0, 1.0, 1.0]})
        np.testing.assert_allclose(env.space.applied, [1.0, 1.0, 1.0, 0.4, 0.5, 0.6])


if __name__ == "__main__":
    unittest.main()
