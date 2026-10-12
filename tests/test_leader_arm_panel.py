# SPDX-License-Identifier: Apache-2.0
"""The leader-arm panel: one bus per arm, applied as one combined action.

Structural, in the same sense as ``test_studio_semantics.py``: a panel needs a
viser server and a compiled scene to build, so these construct it with
``__new__`` and drive ``drain()`` directly against fakes, which is enough to
prove the panel's own logic -- who it hands the arm to, and when it polls --
without any of that.

Two independent Feetech buses, one per physical leader arm (confirmed on the
rig: each arm is its own USB-serial device, not a shared daisy chain), but
``ActionJointPosition.apply`` zips against *every* arm the scene's
``joint_position`` control was built with -- a single-arm controller cannot
apply just its own slice. So the panel reads every controller's action first,
concatenates them in the scene's own arm order, and applies once.
"""

from __future__ import annotations

import unittest

from robosimstudio_studio.leader_arm import PanelLeaderArm

TIER = "fast"
"""No GPU, no scene, no serial port."""


class _FakeWidget:
    def __init__(self, value="") -> None:
        self.value = value


class _FakeArmPanel:
    def __init__(self) -> None:
        self.suspended = False
        self.suspend_calls = 0
        self.resume_calls = 0

    def suspend(self) -> None:
        self.suspended = True
        self.suspend_calls += 1

    def resume(self) -> None:
        self.suspended = False
        self.resume_calls += 1


class _FakeController:
    def __init__(self, action) -> None:
        self.action = action
        self.reads = 0
        self.closed = False

    def read_action(self):
        self.reads += 1
        return self.action

    def close(self) -> None:
        self.closed = True


class _FakeActionSpace:
    def __init__(self, per_arm: int = 2, holds: dict | None = None) -> None:
        self.applied = None
        self.per_arm = per_arm
        self._holds = holds or {}

    def target_action(self, control) -> list:
        return list(self._holds.get(control, []))

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


class _FakeSession:
    def __init__(self, arms, env) -> None:
        self.arms = arms
        self.env = env


def _panel(arms=None, *, env=None, connect_fn=None, calibration_fn=None) -> PanelLeaderArm:
    panel = PanelLeaderArm.__new__(PanelLeaderArm)
    panel.session = _FakeSession(arms or {}, env)
    panel.env = env
    panel.ports = {"left": "/dev/ch340-LEFTSN", "right": "/dev/ch340-RIGHTSN"}
    panel.calibration_path = "cal.json"
    panel.controllers = {}
    panel.enabled = False
    panel.enabled_requested = False
    panel.connect_requested = False
    panel.disconnect_requested = False
    panel._connect_fn = connect_fn
    panel._calibration_fn = calibration_fn
    panel._enabled = _FakeWidget(False)
    panel._state = _FakeWidget("not connected")
    return panel


class TestConnecting(unittest.TestCase):
    def test_one_controller_is_opened_per_arm_with_its_own_port_and_calibration(self) -> None:
        seen = []

        def connect_fn(port, arms):
            seen.append((port, arms))
            return _FakeController(action=None)

        panel = _panel(
            {"left": _FakeArmPanel(), "right": _FakeArmPanel()},
            connect_fn=connect_fn,
            calibration_fn=lambda path: {"LEFTSN": "left-cal", "RIGHTSN": "right-cal"},
        )
        panel.connect_requested = True

        panel.drain()

        self.assertEqual(set(panel.controllers), {"left", "right"})
        self.assertIn(("/dev/ch340-LEFTSN", ["left-cal"]), seen)
        self.assertIn(("/dev/ch340-RIGHTSN", ["right-cal"]), seen)

    def test_an_arm_missing_from_the_calibration_file_is_reported_and_left_unconnected(self) -> None:
        panel = _panel(
            {"left": _FakeArmPanel(), "right": _FakeArmPanel()},
            connect_fn=lambda port, arms: _FakeController(action=None),
            calibration_fn=lambda path: {"LEFTSN": "left-cal"},
        )
        panel.connect_requested = True

        panel.drain()

        self.assertEqual(set(panel.controllers), {"left"})
        self.assertIn("right", panel._state.value)

    def test_an_arm_with_no_port_typed_in_is_reported_and_left_unconnected(self) -> None:
        panel = _panel(
            {"left": _FakeArmPanel(), "right": _FakeArmPanel()},
            connect_fn=lambda port, arms: _FakeController(action=None),
            calibration_fn=lambda path: {"LEFTSN": "left-cal", "RIGHTSN": "right-cal"},
        )
        panel.ports["right"] = ""
        panel.connect_requested = True

        panel.drain()

        self.assertEqual(set(panel.controllers), {"left"})
        self.assertIn("right", panel._state.value)

    def test_one_arms_connect_failure_does_not_stop_the_other_from_opening(self) -> None:
        def connect_fn(port, arms):
            if port == "/dev/ch340-RIGHTSN":
                raise RuntimeError("no such device")
            return _FakeController(action=None)

        panel = _panel(
            {"left": _FakeArmPanel(), "right": _FakeArmPanel()},
            connect_fn=connect_fn,
            calibration_fn=lambda path: {"LEFTSN": "left-cal", "RIGHTSN": "right-cal"},
        )
        panel.connect_requested = True

        panel.drain()  # must not raise

        self.assertEqual(set(panel.controllers), {"left"})
        self.assertIn("no such device", panel._state.value)

    def test_an_unreadable_calibration_file_leaves_every_arm_unconnected(self) -> None:
        def calibration_fn(path):
            raise FileNotFoundError(path)

        panel = _panel(
            {"left": _FakeArmPanel()},
            connect_fn=lambda port, arms: _FakeController(action=None),
            calibration_fn=calibration_fn,
        )
        panel.connect_requested = True

        panel.drain()  # must not raise

        self.assertEqual(panel.controllers, {})


class TestTogglingHandsTheArmBetweenGizmoAndLeaderArm(unittest.TestCase):
    def test_enabling_requires_at_least_one_arm_to_already_be_connected(self) -> None:
        left, right = _FakeArmPanel(), _FakeArmPanel()
        panel = _panel({"left": left, "right": right})
        panel.controllers = {}  # nothing connected
        panel.enabled_requested = True
        panel._enabled.value = True

        panel.drain()

        self.assertFalse(panel.enabled)
        self.assertFalse(left.suspended)
        self.assertEqual(panel._enabled.value, False)

    def test_enabling_with_every_arm_connected_suspends_every_arm_panel(self) -> None:
        left, right = _FakeArmPanel(), _FakeArmPanel()
        env = _FakeEnv(_FakeActionSpace())
        panel = _panel({"left": left, "right": right}, env=env)
        panel.controllers = {"left": _FakeController(action=[0.0]), "right": _FakeController(action=[0.0])}
        panel.enabled_requested = True
        panel._enabled.value = True

        panel.drain()

        self.assertTrue(left.suspended)
        self.assertTrue(right.suspended)
        self.assertTrue(panel.enabled)

    def test_enabling_with_only_one_arm_connected_suspends_only_that_arm(self) -> None:
        left, right = _FakeArmPanel(), _FakeArmPanel()
        env = _FakeEnv(_FakeActionSpace())
        panel = _panel({"left": left, "right": right}, env=env)
        panel.controllers = {"left": _FakeController(action=[0.0])}  # right stays on the gizmo
        panel.enabled_requested = True
        panel._enabled.value = True

        panel.drain()

        self.assertTrue(panel.enabled)
        self.assertTrue(left.suspended)
        self.assertFalse(right.suspended)

    def test_disabling_resumes_every_arm_panel(self) -> None:
        left = _FakeArmPanel()
        panel = _panel({"left": left})
        panel.controllers = {"left": _FakeController(action=[0.0])}
        panel.enabled = True
        left.suspended = True
        panel.enabled_requested = True
        panel._enabled.value = False

        panel.drain()

        self.assertFalse(left.suspended)
        self.assertEqual(left.resume_calls, 1)
        self.assertFalse(panel.enabled)


class TestDrainAppliesTheCombinedAction(unittest.TestCase):
    def test_every_controller_is_read_and_concatenated_in_the_sessions_own_arm_order(self) -> None:
        space = _FakeActionSpace(holds={None: [0.0, 0.0, 0.0, 0.0]})
        env = _FakeEnv(space)
        left, right = _FakeArmPanel(), _FakeArmPanel()
        panel = _panel({"left": left, "right": right}, env=env)
        panel.controllers = {
            "left": _FakeController(action=[1.0, 0.0]),
            "right": _FakeController(action=[2.0, 0.5]),
        }
        panel.enabled = True

        panel.drain()

        self.assertEqual(space.applied[0], [1.0, 0.0, 2.0, 0.5])
        self.assertIs(space.applied[1], env.control)
        self.assertIs(space.applied[2], env.state)

    def test_an_arm_with_no_controller_holds_its_current_target_instead(self) -> None:
        # env.control is a plain object(); target_action's fake keyed by it.
        env = _FakeEnv(None)
        space = _FakeActionSpace(holds={env.control: [9.0, 1.0, 8.0, 0.5]})
        env._space = space
        left, right = _FakeArmPanel(), _FakeArmPanel()
        panel = _panel({"left": left, "right": right}, env=env)
        panel.controllers = {"left": _FakeController(action=[1.0, 0.0])}  # right never connected
        panel.enabled = True

        panel.drain()

        self.assertEqual(space.applied[0], [1.0, 0.0, 8.0, 0.5])

    def test_a_connected_but_disabled_panel_polls_nothing(self) -> None:
        space = _FakeActionSpace()
        env = _FakeEnv(space)
        panel = _panel({"left": _FakeArmPanel()}, env=env)
        controller = _FakeController(action=[0.0, 0.0])
        panel.controllers = {"left": controller}

        panel.drain()

        self.assertEqual(controller.reads, 0)
        self.assertIsNone(space.applied)


class TestDisconnecting(unittest.TestCase):
    def test_disconnect_closes_every_controller_and_resumes_the_arms(self) -> None:
        left = _FakeArmPanel()
        panel = _panel({"left": left})
        controller = _FakeController(action=[0.0])
        panel.controllers = {"left": controller}
        panel.enabled = True
        left.suspended = True
        panel.disconnect_requested = True

        panel.drain()

        self.assertTrue(controller.closed)
        self.assertEqual(panel.controllers, {})
        self.assertFalse(panel.enabled)
        self.assertFalse(left.suspended)

    def test_close_disconnects(self) -> None:
        panel = _panel({"left": _FakeArmPanel()})
        controller = _FakeController(action=[0.0])
        panel.controllers = {"left": controller}
        panel.gizmo = None

        panel.close()

        self.assertTrue(controller.closed)
        self.assertEqual(panel.controllers, {})


if __name__ == "__main__":
    unittest.main()
