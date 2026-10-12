# SPDX-License-Identifier: Apache-2.0
"""A Feetech leader arm as a teleop source, toggled beside the gizmo drag.

The gizmo panels (:class:`~robosimstudio_studio.panels.PanelArm`) are
the studio's own input; this is another one, offered rather than substituted --
``Drive from leader arm`` hands every arm to a
:class:`~robosimstudio.robots.teleop.lerobot_controller.LerobotFeetechController`
the same way a task page's policy runner (``PanelSolution``) hands
them to a running policy: :meth:`PanelArm.suspend` stops the gizmo driving and
hides it, :meth:`PanelArm.resume` gives the hand back. Turning the toggle off
leaves the arm exactly where the leader left it, for the gizmo to pick up from.

**One bus per arm.** A bimanual leader is two independent Feetech chains, each
its own USB-serial device -- not one bus with two arms' servos daisy-chained
together -- so this panel opens one controller per arm name, each with its own
port. But :meth:`~robosimstudio.solvers.control.action.ActionJointPosition.apply`
zips against *every* arm the scene's ``joint_position`` control was built
with, so a single-arm controller cannot apply just its own slice: every
controller's :meth:`~.LerobotFeetechController.read_action` is read first,
concatenated in the scene's own arm order, and applied once, in :meth:`drain`.

The connection itself is deferred to a ``Connect`` button rather than made at
build time: a calibration file and the serial ports are per-session facts
about whatever leader is plugged in, worth changing without reloading the
task, and a page that dialled a missing port on every load would fail on
every load whether or not anyone meant to teleoperate today.

**Read-only.** This panel reads joint positions and writes nothing to a servo.
Rendering the simulated arm's resistance back onto the leader is
:mod:`robosimstudio.robots.teleop.force_feedback`, which owns each bus from a
thread of its own and stands this panel down while it runs
(:meth:`PanelLeaderArm.stand_down`).
"""

from __future__ import annotations

from typing import Any

from robosimstudio.robots.teleop.lerobot_controller import (
    LerobotFeetechController,
    load_calibration,
    serial_from_port,
)

from .panels import Panel

__all__ = ["PanelLeaderArm"]

_serial_from_port = serial_from_port
"""The calibration key for a port; see
:func:`~robosimstudio.robots.teleop.lerobot_controller.serial_from_port`."""


class PanelLeaderArm(Panel):
    """Connect a Feetech leader arm per session arm, and toggle who is driving.

    Args:
        session: The page.
        ports: Starting serial device per arm name, e.g.
            ``{"left": "/dev/yam_leader_left", "right": "/dev/yam_leader_right"}``.
            Only a starting value for the panel's own per-arm fields.
        calibration_path: A JSON file :func:`~robosimstudio.robots.teleop.lerobot_controller.load_calibration`
            reads, one entry per arm name. Re-read on every ``Connect``, so
            editing it and reconnecting is how a calibration pass is tried.
    """

    key = "leader_arm"
    """Two words, where the derived spelling would be ``leaderarm``."""

    def __init__(self, session, *, ports: dict[str, str] | None = None, calibration_path: str = "") -> None:
        super().__init__(session)
        self.ports: dict[str, str] = dict(ports or {})
        self.calibration_path = str(calibration_path)
        self.controllers: dict[str, Any] = {}
        self.enabled = False
        self.enabled_requested = False
        self.connect_requested = False
        self.disconnect_requested = False
        # Seams for a test to inject a fake bus and a fake calibration file
        # without touching disk or a serial port.
        self._connect_fn = LerobotFeetechController.connect
        self._calibration_fn = load_calibration

    def build(self, gui) -> None:
        with gui.add_folder("Leader arm", expand_by_default=False):
            self._port_widgets = {}
            for name in self.session.arms:
                widget = gui.add_text(f"Port ({name})", initial_value=self.ports.get(name, ""))
                widget.on_update(lambda _, name=name, w=widget: self.ports.__setitem__(name, str(w.value)))
                self._port_widgets[name] = widget
            self._calibration = gui.add_text("Calibration file", initial_value=self.calibration_path)
            self._calibration.on_update(lambda _: setattr(self, "calibration_path", str(self._calibration.value)))
            connect = gui.add_button("Connect")
            connect.on_click(lambda _: setattr(self, "connect_requested", True))
            disconnect = gui.add_button("Disconnect")
            disconnect.on_click(lambda _: setattr(self, "disconnect_requested", True))
            self._enabled = gui.add_checkbox("Drive from leader arm", initial_value=False)
            self._enabled.on_update(lambda _: setattr(self, "enabled_requested", True))
            self._state = gui.add_text("Leader arm", initial_value="not connected", disabled=True)

    def drain(self) -> None:
        if self.connect_requested:
            self.connect_requested = False
            self._connect()
        if self.disconnect_requested:
            self.disconnect_requested = False
            self._disconnect()
        if self.enabled_requested:
            self.enabled_requested = False
            self._set_enabled(bool(self._enabled.value))
        if self.enabled:
            self._poll_and_apply()

    # ------------------------------------------------------------- connection
    def _connect(self) -> None:
        """Open one bus per arm. Failures land in the readout, never raised.

        A missing calibration entry or an unplugged port is the common case
        here, not a bug -- reported the same way :class:`PanelTask` reports a
        pairing that will not compile. One arm's failure does not stop the
        other from opening: a session with only one leader plugged in can
        still use it, once :meth:`_set_enabled` refuses to go live short of
        every arm.
        """
        self._disconnect()
        try:
            calibration = self._calibration_fn(self.calibration_path)
        except Exception as exc:
            self._state.value = f"{type(exc).__name__}: {exc}"[:200]
            return
        opened: dict[str, Any] = {}
        failures: list[str] = []
        for name in self.session.arms:
            port = self.ports.get(name, "")
            if not port:
                failures.append(f"{name}: no port")
                continue
            serial = _serial_from_port(port)
            if serial not in calibration:
                failures.append(f"{name}: no calibration entry for serial {serial!r}")
                continue
            try:
                opened[name] = self._connect_fn(port, [calibration[serial]])
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        self.controllers = opened
        if failures:
            self._state.value = "; ".join(failures)[:200]
        else:
            self._state.value = f"connected: {', '.join(self.controllers)}"

    def _disconnect(self) -> None:
        self._set_enabled(False)
        for controller in self.controllers.values():
            controller.close()
        self.controllers = {}
        self._state.value = "not connected"

    # ------------------------------------------------------------ hand-off
    def _set_enabled(self, want: bool) -> None:
        """Give the connected arms to the leader, or give them all back to the gizmos.

        Not all-or-nothing: an arm with no controller simply holds its own
        current target every tick (see :meth:`_poll_and_apply`), so only the
        arms actually connected are handed over -- an unconnected arm's gizmo
        keeps working exactly as before.
        """
        if want and not self.controllers:
            want = False
            self._state.value = "connect at least one arm first"
        self._enabled.value = want
        if want == self.enabled:
            return
        self.enabled = want
        for name, arm in self.session.arms.items():
            if name in self.controllers:
                (arm.suspend if want else arm.resume)()

    def _poll_and_apply(self) -> None:
        """Read every connected controller and apply one combined action.

        ``ActionJointPosition.apply`` zips against *every* arm the scene's
        control was built with, so an arm with no leader connected still
        needs a slice of the vector -- filled in from
        :meth:`~robosimstudio.solvers.control.action.ActionJointPosition.target_action`,
        which reads back whatever that arm's target already is (gizmo-driven
        or simply parked), making this a no-op for that arm rather than a
        fight over its target.
        """
        env = self.env
        action_space = env.use_control("joint_position")
        action = list(action_space.target_action(env.control))
        per_arm = action_space.per_arm
        for i, name in enumerate(self.session.arms):
            if name in self.controllers:
                action[i * per_arm : (i + 1) * per_arm] = self.controllers[name].read_action()
        action_space.apply(action, env.control, env.state)

    def stand_down(self, reason: str) -> None:
        """Release every bus and say why this folder is no longer driving.

        For something else that takes the same ports for itself --
        :class:`~robosimstudio.robots.teleop.force_feedback.StudioBridge` reads
        each leader from its own thread, and two readers on one serial port
        corrupt each other's packets. The arms go back to their gizmos, the
        buses are closed, and the readout carries the reason rather than "not
        connected", which would read as a fault.
        """
        self._disconnect()
        self._state.value = str(reason)[:200]

    def close(self) -> None:
        super().close()
        self._disconnect()
