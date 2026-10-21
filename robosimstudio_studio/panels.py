# SPDX-License-Identifier: Apache-2.0
"""Scene panels, composed from a scene's handles.

Which panels exist is derived from the scene rather than written per task: a
task with two arms gets two arm panels, one with no cloth gets no cloth panel.
That is what lets one page serve every suite, and it is why these are library
code rather than part of the application --
:mod:`~robosimstudio_studio.session` composes them, but so may
anything else holding a built scene -- a ``Sim``, or the benchmark's ``Env`` --
and a viser server. The panels that drive the *page* rather than the scene --
recording, video export -- live in :mod:`~robosimstudio_studio.controls`, and
the two of those that need a task, its menu and its policy runner, in the
page's benchmark half.

**The interaction these panels add up to** is: click a thing and drag it.

* A click on a **rigid body** puts a live handle on it -- a PD wrench, aimed by
  a 6-DOF gizmo you can drag the instant it appears.
* A click on a **deformable** seizes the patch under the click and puts a handle
  on that. Up to two at a time, so a sheet can be held by two corners; with
  ``Move together`` a drag on either carries the pair as one rigid group. A
  deformable a *foreign* engine owns -- a SuperDex sponge -- is held through
  that engine's own node constraints instead; see :class:`PanelCloth`.
* An **arm** carries a gizmo of its own on its tool centre and tracks it every
  step. The fingers are the ``Close`` slider and nothing else; a vacuum cup is
  the ``Vacuum`` box, which seals onto whatever is under the face.

Placing a handle and taking hold are the same act, deliberately. A visible
handle is a handle that is holding something; the buttons that remain are for
what a click cannot reach.

Nothing here is a grasp. What a hand ends up holding is whatever its friction
and the contact settings let it hold, and what a handle holds it holds by a
wrench or a pin -- which is why the readouts say how many particles are held and
whether a hold is *kinematic* rather than implying a robot did it.

Every panel obeys the same contract:

* :meth:`Panel.build` runs on the main thread and registers viser callbacks.
* Those callbacks **only assign plain attributes**. They never call into Warp,
  because they fire on viser's server threads.
* :meth:`Panel.drain` runs on the main thread, once per control step, and is the
  only place those attributes are acted on.
* :meth:`Panel.refresh` updates read-only text, also on the main thread.
* :meth:`Panel.close` gives back the scene nodes the panel opened, so a session
  may outlive the environment it was built for.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from robosimstudio.core.math import quat_to_mat, quat_wxyz_to_xyzw, quat_xyzw_to_wxyz
from robosimstudio.solvers.ik.solver import POS_TOLERANCE

from .handles import HandleSet

__all__ = ["Panel", "PanelArm", "PanelCloth", "PanelObject"]

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])
"""xyzw identity. A cloth grab has no orientation of its own to start from."""


@dataclass
class _HostPin:
    """One handle's grip on a deformable a foreign engine owns.

    The engine-side counterpart of a :class:`~robosimstudio.core.ParticleDragger`:
    the same nodes, the same grip-frame offsets, held by the engine's own
    compliant node constraints instead of by a kernel in the substep schedule.

    Attributes:
        bridge: The host engine's bridge, which owns the constraint.
        pin: The pin id it handed back, for re-aiming and for letting go.
        name: Which entity, for the readouts and for a re-grab.
        nodes: Global model particle indices held.
        offsets: Those nodes in the grip frame at the moment of the grab [m],
            so the patch keeps its shape as the handle is carried and turned.
    """

    bridge: Any
    pin: int
    name: str
    nodes: np.ndarray
    offsets: np.ndarray

    def aim(self, position: np.ndarray, quat: np.ndarray) -> bool:
        """Re-aim at a handle pose. ``False`` once the pin is gone."""
        targets = np.asarray(position, dtype=np.float64) + self.offsets @ quat_to_mat(quat).T
        try:
            self.bridge.set_pin_targets(self.pin, targets)
        except (KeyError, IndexError):
            # A reset destroys every pin the engine holds (``restore_rest``
            # unpins before it restores), and a page whose reset came from
            # somewhere else finds out here. Saying so is the next grab's job.
            return False
        return True

    def release(self) -> None:
        """Let the engine go. Idempotent -- ``unpin`` pops rather than indexes."""
        self.bridge.unpin(self.pin)


def _role_suffix(panel) -> str:
    """`` (work)`` or `` (support)`` for an arm the binding gave a role to.

    Empty for a one-armed cell, where the role is the only one there is and
    saying so would be noise.
    """
    handles = getattr(panel.env, "handles", None)
    if handles is None or len(getattr(handles, "arms", ())) < 2:
        return ""
    for role in ("work", "support"):
        if panel.name == getattr(handles, f"{role}_arm", ""):
            return f" ({role})"
    return ""


def seal_text(suction, looked=None) -> str:
    """The **Seal** readout: sealed or not, the face's gap to the part, and the mass held.

    Holding, the gap is the held part's; otherwise it is whatever is under the
    cup (:meth:`~robosimstudio.core.suction.Suction.under`), so the number counts
    down as the hand comes down onto a plate. A look, not a test the cup made:
    it leaves :attr:`~robosimstudio.core.suction.Suction.last` to the holds.
    ``looked`` is a refusal the cup returned this step, which is that same look
    already taken: an armed box re-tests every step, and a refresh shows its
    answer rather than testing again.
    """
    held = suction.held
    if held >= 0:
        report = suction.seal(held, record=False)
    else:
        report = looked if looked is not None else suction.under(record=False)
    gap = f"gap {report.gap_m * 1000:.1f} mm" if np.isfinite(report.gap_m) else "nothing under the cup"
    mass = suction.held_mass if held >= 0 else 0.0
    return f"{'sealed' if held >= 0 else 'not sealed'}, {gap}, holding {mass:.3f} kg"


class Panel:
    """Base panel. Subclasses build widgets and act on their state in ``drain``."""

    def __init__(self, session) -> None:
        self.session = session
        self.env = session.env
        self.server = session.server

    @property
    def profile_name(self) -> str:
        """What this panel is called in a profile of the loop.

        The class name minus its ``Panel`` prefix, lowercased, so the phase
        table reads ``drain.arm`` rather than ``drain.PanelArm``. A panel that
        can exist more than once per page overrides this to say *which* one it
        is -- otherwise two arms would share a row and the table would show
        their sum while looking like it showed one arm's cost.
        """
        return type(self).__name__.removeprefix("Panel").lower()

    def build(self, gui) -> None:
        """Register widgets. Main thread."""

    def drain(self) -> None:
        """Act on whatever the callbacks wrote. Main thread, once per step."""

    def refresh(self) -> None:
        """Update read-only text. Main thread."""

    def on_reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        """Re-sync after the environment reset."""

    def on_select(self, subject) -> None:
        """The page is now about something else.

        Called on the main thread, from the session's own drain, so a panel may
        act here -- unlike a viser callback, which may only assign. Most panels
        want nothing: they are about the scene rather than about one thing in it.
        """

    def close(self) -> None:
        """Release what the panel added to the scene. Main thread, idempotent.

        A session that outlives one environment -- one that loads another task
        into the same browser page -- has to take the old scene's gizmos down
        before the new one goes up, or the page keeps showing controls bound to a
        model that no longer exists. GUI widgets are not removed here: a caller
        that rebuilds a page clears the panel wholesale.
        """
        gizmo = getattr(self, "gizmo", None)
        if gizmo is not None:
            try:
                gizmo.remove()
            except Exception:
                pass
            self.gizmo = None


class PanelArm(Panel):
    """One arm, driven directly: drag the gizmo, squeeze the slider.

    A 6-DOF transform gizmo sits on the tool centre and the arm tracks it every
    step through :class:`~robosimstudio.solvers.OscSetpoint` -- the same
    rate limit and lead clamp the graph runtime's servo and the ``osc_pose``
    action space use, so the hand crosses the bench at one speed in this package
    rather than at whatever a constant written here implied.

    The fingers are the slider and nothing else. There is no propose-a-grasp
    step and no scripted approach: what the hand ends up holding is whatever its
    friction and the contact settings let it hold, which is the thing this stack
    exists to show, and why the button says "Close" and not "Grasp". Whether it
    holds is the scene's physics: a rigid body under ``MUJOCO_500HZ``, a sheet
    under one-way coupling with full-surface contact, a rope with sticky
    contacts and a sponge with its own friction falloff all lift by friction;
    under ``GENERIC`` a hand
    pushes a free body and does not pinch it.

    **Not collision aware.** The arm goes where the gizmo is put: asked to
    descend through a bin rim, the hand descends through the bin rim.
    """

    def __init__(self, session, name: str) -> None:
        super().__init__(session)
        from robosimstudio.solvers import OscSetpoint

        self.name = name
        self.ik = self.env.arm(name)
        self.gripper = self.env.scene.embodiment.grippers.get(name)
        self.suction = getattr(self.env.scene.embodiment, "suctions", {}).get(name)
        """The arm's vacuum cup (:class:`~robosimstudio.core.suction.Suction`), or ``None`` for a hand with fingers."""
        index = self.env.scene.embodiment.arms.get(name)
        self.index = index
        self.finger_bodies = tuple(b for side in (index.finger_bodies if index else ()) for b in side)

        # The *arrangement's*, off the request: a control rate is a fact about
        # how the scene is stepped, and a page over a scene with no task has one.
        self.control_hz = max(1.0, float(self.env.request.physics.control_hz))
        # Seeded at the home pose and re-seeded on every reset, because a
        # setpoint carried across one is a command aimed at the layout the
        # previous episode had.
        self.law = OscSetpoint(control_dt=1.0 / self.control_hz)
        self.law.reset(*self.ik.home_pose)

        self.gizmo = None
        self.gripper_command = 0.0
        self.grip_request: float | None = None
        """A grip a callback asked for, applied on the main thread by ``drain``.

        Assigned rather than acted on because it has to move the slider too, and
        writing a widget from viser's threadpool is the one thing the panel
        contract forbids."""
        self.cursor_pos, self.cursor_quat = self.ik.home_pose
        self._seated: np.ndarray | None = None
        """The cursor the gizmo was last put back on the hand at, so it is put back once per move."""
        self.dragging = False
        self.suspended = False
        self.status = ""
        self._last_commanded: tuple[np.ndarray, np.ndarray] | None = None
        self._ee_readout = None
        self._grip = None
        self._width_readout = None
        self.vacuum = False
        self.vacuum_request: bool | None = None
        """The **Vacuum** box as a callback last set it, applied on the main thread by ``drain`` like a grip."""
        self._vacuum = None
        self._seal_readout = None
        self._held_last = False
        self._looked = None
        """This step's refused hold while the box is armed, which the **Seal** readout shows instead of looking again."""

    @property
    def profile_name(self) -> str:
        """``arm:left``, so a bimanual page profiles its two arms separately."""
        return f"arm:{self.name}"

    # ------------------------------------------------------------------- build
    def build(self, gui) -> None:
        self.gizmo = self.server.scene.add_transform_controls(
            f"/studio/{self.name}/cursor",
            scale=0.14,
            line_width=3.0,
            visible=True,
            position=tuple(float(v) for v in self.cursor_pos),
            wxyz=tuple(quat_xyzw_to_wxyz(self.cursor_quat)),
        )

        # The callback is handed a TransformControlsEvent, not the handle -- the
        # pose lives on event.target. Reading it off the handle instead raises
        # inside viser's callback threadpool, where the traceback goes nowhere
        # and the only symptom is a gizmo that moves while the arm does not.
        @self.gizmo.on_update
        def _(event) -> None:
            self.cursor_pos = np.array(event.target.position, dtype=np.float64)
            self.cursor_quat = quat_wxyz_to_xyzw(event.target.wxyz)
            # phase is "start" / "update" / "end". Knowing the drag is over is
            # what lets drain() put the cursor back on the hand without fighting
            # the browser for it mid-gesture.
            self.dragging = str(getattr(event, "phase", "update")) != "end"

        # Labelled with the role the binding gave it, not just its name. "left"
        # and "right" are facts about the robot; "work" and "support" are facts
        # about the task, and which is which flips between the two YAM cells.
        #
        # Open by default, unlike every other folder on the page. Closing a hand
        # is half of what a person came here to do, and it used to cost three
        # actions -- unroll the fold, find the slider, drag it -- for one bit of
        # information. The button group is that bit; the slider stays beneath it
        # for the grip that is neither open nor shut.
        with gui.add_folder(f"Arm: {self.name}{_role_suffix(self)}", expand_by_default=True):
            self._ee_readout = gui.add_text("EE", initial_value="", disabled=True)
            if self.gripper is not None:
                grip = gui.add_button_group("Grip", ("Open", "Close"))
                grip.on_click(lambda _: setattr(self, "grip_request", 0.0 if grip.value == "Open" else 1.0))
                # Named for the whole travel rather than for one end of it: the
                # buttons above are already ``Open`` and ``Close``, and a slider
                # sharing a name with one of them read as a control drawn twice.
                self._grip = gui.add_slider(
                    "Grip amount", min=0.0, max=1.0, step=0.01, initial_value=self.gripper_command
                )
                self._grip.on_update(lambda _: setattr(self, "gripper_command", float(self._grip.value)))
                self._width_readout = gui.add_text("Opening", initial_value="", disabled=True)
            elif self.suction is not None:
                # A cup in place of jaws: one switch, and what the cup found.
                # Ticked, it seals the first step the face is within the seal's
                # GAP_TOLERANCE of whatever is under it, so it can be ticked
                # before the hand comes down; the hold then keeps the part where
                # it sealed, up to that far under the face, where a tick after
                # the face is down holds it flush. Unticked, it lets go.
                self._vacuum = gui.add_checkbox("Vacuum", initial_value=False)
                self._vacuum.on_update(lambda _: setattr(self, "vacuum_request", bool(self._vacuum.value)))
                self._seal_readout = gui.add_text("Seal", initial_value="", disabled=True)
        self._take_hand()

    # -------------------------------------------------------------------- loop
    def drain(self) -> None:
        env = self.session.env
        # Being tugged by a handle wins over driving, and it takes both calls.
        #
        # `suspend` is not the mechanism: it only stops the *update*, leaving
        # `joint_target_q` at its last value with MuJoCo still servoing to it,
        # so the wrench fights a spring and the arm snaps back the moment it is
        # let go. `hold_into` is -- it makes the target the achieved pose every
        # step, so there is no error left to restore and the wrench moves the
        # arm freely. What still stops it is what should: joint limits,
        # self-collision, and whatever it is holding.
        #
        # `_take_hand` then walks the *cursor* along behind the arm. Without it
        # the joint targets would be right and the gizmo would be back where the
        # drag started, so letting go would drive the hand across the bench to
        # meet it. With it, releasing hands driving back mid-air.
        objects = getattr(self.session, "objects", None)
        if objects is not None and objects.holding_arm(self.name):
            self.ik.hold_into(env.state.joint_q, env.control.joint_target_q)
            self._take_hand()
        # A move is held while paused rather than consumed: the law would walk
        # its setpoint away from an arm that is not moving, and the whole lead
        # would be spent the moment you unpaused -- and so is one on any pass the
        # page does not step (held for a routine stepping the scene itself);
        # one it steps though paused (a routine driving through this panel) drives.
        elif getattr(self.session, "stepping_this_pass", not self.session.paused) and not self.suspended:
            self._drive(env)
        # The fingers go with the arm: a suspended panel is not driving, and a
        # slider that kept writing the hand while a policy (or a leader arm)
        # drove it would fight whatever that was commanding -- most visibly at
        # the hand-off, where this panel's stale 0.0 would open a grip the
        # policy had just closed. Nothing is written here until
        # :meth:`resume` has read back what is actually holding.
        if self.gripper is not None and not self.suspended:
            if self.grip_request is not None:
                self.gripper_command, self.grip_request = float(self.grip_request), None
                if self._grip is not None:
                    self._grip.value = self.gripper_command
            self.gripper.command = self.gripper_command
            self.gripper.apply(env.control.joint_target_q)
        self._looked = None
        if self.suction is not None and not self.suspended:
            self._drain_vacuum()

    def _drain_vacuum(self) -> None:
        """The box arms the cup and then follows it.

        Ticked, the cup seals the first step its face is close enough to
        something to pass Suction.seal; unticked, it lets go. A cup held or let
        go from code, or whose seal broke, moves the box rather than being
        undone by it. Armed and not sealed, the hold re-tests every step, and
        its refusal is kept for the **Seal** readout (:attr:`_looked`) so a
        refresh does not run the same test again. The cup is asked whether it
        holds once: on the wrench tier that is a read off the device.
        """
        holding = self.suction.holding
        if self.vacuum_request is not None:
            self.vacuum, self.vacuum_request = bool(self.vacuum_request), None
            if not self.vacuum and holding:
                self.suction.release()
                holding = False
        elif holding != self._held_last:
            self.vacuum = holding
        if self.vacuum and not holding:
            report = self.suction.hold()
            holding = report.sealed
            self._looked = None if holding else report
        self._held_last = holding

    def _drive(self, env) -> None:
        """Track the gizmo with the hand, and whatever the hand is on.

        Nothing is welded and nothing is teleported: an object comes along
        because the fingers are on it, so a grip that slips is visibly a grip
        that slips.
        """
        prof = self.session.profile
        with prof.phase(f"iter.drain.{self.profile_name}.pose"):
            position, quat = self.ik.ee_pose(env.state)
        with prof.phase(f"iter.drain.{self.profile_name}.ik"):
            if self.ik.solver is None:
                # On an LM-hostile model (a coupled rope scene, or one over 64
                # joint coords) solve_into runs the host FD solver, a few hundred ms
                # a call. The law would ask for one every step even with the gizmo
                # standing still, so there the cursor is commanded directly and only
                # when it moved. LM stays per-frame: it is device-side and cheap.
                moved = self._last_commanded is None or (
                    float(np.linalg.norm(self.cursor_pos - self._last_commanded[0])) > 1.0e-4
                    or float(np.linalg.norm(self.cursor_quat - self._last_commanded[1])) > 1.0e-3
                )
                if moved:
                    self.ik.set_target(self.cursor_pos, self.cursor_quat)
                    self._last_commanded = (self.cursor_pos.copy(), self.cursor_quat.copy())
                    self.ik.solve_into(env.state.joint_q, env.control.joint_target_q)
            else:
                self.ik.set_target(*self.law.advance(position, quat, self.cursor_pos, self.cursor_quat))
                self.ik.solve_into(env.state.joint_q, env.control.joint_target_q)

        # Put the gizmo back on the hand once the arm has caught up, so the next
        # drag starts from where the hand is. Never mid-gesture: writing the
        # handle while the browser is dragging it fights the client. And never
        # when the arm has *not* caught up -- a gizmo quietly walked back off an
        # unreachable target would hide the one thing worth reporting.
        #
        # Once per move, not on every tick. A hand carrying something sits a few
        # millimetres under its target, and re-seating the cursor there each
        # step made that sag the next target: a lifted 50 mm cube sank 41 mm in
        # two seconds while the page said "on target". Seated once, the target
        # stays where the gizmo was put and the servo holds the load against it.
        with prof.phase(f"iter.drain.{self.profile_name}.gizmo"):
            gap = float(np.linalg.norm(position - self.cursor_pos))
            if not self.dragging and gap < POS_TOLERANCE:
                if self._seated is None or not np.array_equal(self._seated, self.cursor_pos):
                    self._sync_cursor(position, quat)
                    self.cursor_pos, self.cursor_quat = position, quat
                    self._seated = np.array(position, dtype=np.float64)
                self.status = "on target"
            elif not self.dragging:
                self.status = f"{gap * 1000:.0f} mm behind the gizmo"

    # ----------------------------------------------------------------- cursor
    def _take_hand(self) -> None:
        """Put the gizmo on the tool centre and start tracking from there."""
        position, quat = self.ik.ee_pose(self.env.state)
        self.cursor_pos, self.cursor_quat = position, quat
        self._seated = np.array(position, dtype=np.float64)
        self.law.reset(position, quat)
        self._last_commanded = None
        self.dragging = False
        self._sync_cursor(position, quat)
        if self.gizmo is not None:
            self.gizmo.visible = not self.suspended

    def aim(self, position, quat=None, *, moving: bool = True) -> None:
        """Put the cursor somewhere from code, the way a drag of the gizmo does: the gizmo moves with it.

        ``moving`` is the drag's phase. While it is true the cursor stays where
        it is put, as it does mid-gesture; false ends the drag, after which the
        cursor is put back on the hand once the hand arrives. Main thread only,
        like every other write to a widget.
        """
        self.cursor_pos = np.asarray(position, dtype=np.float64)
        if quat is not None:
            self.cursor_quat = np.asarray(quat, dtype=np.float64)
        self.dragging = bool(moving)
        self._sync_cursor(self.cursor_pos, self.cursor_quat)

    def _sync_cursor(self, position: np.ndarray, quat: np.ndarray) -> None:
        if self.gizmo is None:
            return
        self.gizmo.position = tuple(float(v) for v in position)
        self.gizmo.wxyz = tuple(quat_xyzw_to_wxyz(quat))

    # ------------------------------------------------------- policy hand-off
    def suspend(self) -> None:
        """Give the arm up to a policy: stop driving and hide the gizmo."""
        self.suspended = True
        if self.gizmo is not None:
            self.gizmo.visible = False

    def resume(self) -> None:
        """Take the arm back from a policy, from wherever it left the hand.

        Including its grip: see :meth:`_adopt_grip`.
        """
        self.suspended = False
        self._adopt_grip()
        self._take_hand()

    def _adopt_grip(self) -> None:
        """Take whatever is commanding the fingers now as this panel's own.

        This panel is not drained while a policy (or a leader arm) drives -- see
        :data:`~robosimstudio_studio.session.CONTROL_PANELS` -- so its
        ``gripper_command`` is still whatever the slider last said, which on a
        page nobody dragged it on is 0.0, wide open. Writing that back on the
        first drain after the hand-off would fling the fingers open on whatever
        the policy was holding, and a run you stopped to *look* at what it
        grasped would drop it in the act of stopping.

        So the live command is read back instead: stopping holds what was held,
        and the ``Open`` button is how it is let go. The *commanded* closure and
        not the measured one, deliberately -- fingers squeezing an object sit
        wider than they are commanded, and adopting the width they achieved
        would relax the squeeze by exactly the amount that is holding the thing
        up.
        """
        if self.suction is not None:
            # A cup a policy left holding stays holding: the box follows the cup.
            self.vacuum_request = bool(self.suction.holding)
        if self.gripper is None:
            return
        # Through ``grip_request`` rather than assigned: it moves the slider too,
        # and this may be reached from a viser callback thread.
        self.grip_request = float(self.gripper.command)

    def on_select(self, subject) -> None:
        """Bring the gizmo to the hand when the page becomes about this arm.

        Re-seating it on the tool centre makes a click on any part of the arm
        the start of a drag, which is what it looks like it is.
        """
        if subject is not None and getattr(subject, "kind", "") == "arm" and subject.name == self.name:
            self._take_hand()

    # ---------------------------------------------------------------- display
    def report_click(self, text: str) -> None:
        """Say what the last click found. Nothing acts on it -- there is no
        propose step any more -- but knowing what is under the cursor is still
        worth a line."""
        self.status = text

    def refresh(self) -> None:
        if self._ee_readout is not None:
            position, _ = self.ik.ee_pose(self.env.state)
            where = " ".join(f"{v:+.3f}" for v in position)
            self._ee_readout.value = f"{where}  {self.status}".strip()
        if self._width_readout is not None and self.gripper is not None:
            self._width_readout.value = f"{self.gripper.width(self.env.state) * 1000:.1f} mm"
        if self._seal_readout is not None and self.suction is not None:
            self._seal_readout.value = seal_text(self.suction, self._looked)
        if self._vacuum is not None and self.vacuum_request is None and bool(self._vacuum.value) != self.vacuum:
            self._vacuum.value = self.vacuum

    def on_reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        del obs, info
        self.gripper_command = 0.0
        # Cancelled, not left pending: a policy start resets *after* stopping
        # the previous one, so a grip adopted by :meth:`_adopt_grip` a moment
        # ago would otherwise be applied on top of the hand this reset opened.
        self.grip_request = None
        if self._grip is not None:
            self._grip.value = 0.0
        self.vacuum, self.vacuum_request, self._held_last, self._looked = False, None, False, None
        if self._vacuum is not None:
            self._vacuum.value = False
        self._take_hand()


class PanelObject(Panel):
    """The clicked object, on a handle you can drag the moment it appears.

    Clicking a body in the 3D view puts a live handle on it: the gizmo is
    visible *because* a wrench is already on the body, so the obvious gesture --
    click it, drag it, lift it -- is the whole interaction. The dropdown and
    ``Attach handle`` remain for an object the camera cannot see round to.

    This is deliberately *not* a grasp -- that is the arm panel's business, and
    it involves a robot. It is a PD wrench on the body with nothing holding it:
    the body stays fully dynamic underneath, so dragging a tool into the jaws is
    a legitimate way to set a grasp up, and tugging on one the robot is already
    holding is how you find out whether the grip is real.

    Args:
        session: The page.
        handles: How many objects may be held at once. Each slot is a dragger,
            and a dragger is a hook the captured graph has to already hold, so
            this is fixed for the life of the panel.
    """

    def __init__(self, session, *, handles: int = 2, chrome: bool = True) -> None:
        super().__init__(session)
        from robosimstudio.core.draggers import ObjectDragger

        from .selection import arm_link_bodies

        self.names = [n for n, h in self.env.handles.objects.items() if not h.static]
        self.by_body = {int(h.body): n for n, h in self.env.handles.objects.items() if not h.static}
        # Rigid chains -- a rope built as capsules -- are grabbable per segment:
        # the handle goes where the click landed, not on a nominal first body,
        # because dragging a chain by its far end and dragging it by the middle
        # are different manipulations and both are wanted.
        #
        # A chain is an ordinary object now, so it arrives in `objects` above
        # with one EntityHandle over the whole run. But that handle's `body` is
        # only its first segment and `bodies` is the run -- that is how a chain
        # entity's `handle()` reports itself -- so the comprehension above
        # maps segment 0 and nothing else, and a click on the other 41 capsules
        # of a 42-capsule rope finds no name and is answered with nothing.
        self.chains = {
            name: handle
            for name, handle in self.env.handles.objects.items()
            if not handle.static and len(getattr(handle, "bodies", ())) > 1
        }
        body_mass = self.env.model.body_mass.numpy()
        # What one dragged segment actually carries: the whole chain. The
        # dragger's clamp and damping are sized per attach from this, or a
        # third-of-a-gram segment's 20 g ceiling stalls the chain a hand's width
        # off the table.
        self.chain_mass = {
            name: float(body_mass[np.asarray(chain.bodies, dtype=np.int64)].sum())
            for name, chain in self.chains.items()
        }
        for name, chain in self.chains.items():
            # Already in `names` -- it is an ordinary object -- so only the
            # per-segment reverse lookup is added here.
            for body in chain.bodies:
                self.by_body.setdefault(int(body), name)
        # And the robot's own links. Clicking a forearm and pulling on it is the
        # same gesture as clicking a block, and it used to be the one thing on
        # the page that answered a click with nothing at all. The arm stays
        # actuated underneath -- what a tug does to it is its joint limits' and
        # its servo's answer, which is the honest one.
        #
        # Sized by the whole chain for the same reason a rope segment is: a
        # wrist link weighs grams, and a handle clamped to *its* 20 g of
        # acceleration cannot move the forearm behind it.
        self.arm_bodies: dict[str, tuple[int, ...]] = {}
        for name, arm in self.env.handles.arms.items():
            bodies = arm_link_bodies(self.env.model, arm.ee_body)
            if not bodies:
                continue
            self.arm_bodies[name] = bodies
            self.names.append(name)
            self.chain_mass[name] = float(body_mass[np.asarray(bodies, dtype=np.int64)].sum())
            for body in bodies:
                self.by_body.setdefault(int(body), name)
        # Every dragger a session can ever use, contributed in one go: the
        # pipeline drops and re-captures its graph per invalidation, and a page
        # that added a hook per grab would pay for one every time you clicked.
        self.draggers = [ObjectDragger(self.env.model) for _ in range(max(1, int(handles)))]
        self.contribute()
        self.chrome = bool(chrome)
        """Whether to build a folder. See :meth:`build`."""
        self._dropdown: Any = None
        self.handles: HandleSet | None = None
        self.grab_on_click = True
        self.attach_requested = False
        self.release_requested = False
        self.release_all_requested = False
        self.pending_body: int | None = None
        """A click waiting to be serviced on the main thread."""
        self.selected = self.names[0] if self.names else ""

    def contribute(self) -> None:
        """Put this panel's wrenches back into the substep schedule.

        Called at construction and again after every reset, because a reset
        **clears every hook installed after the scene was built** -- deliberately,
        so a policy that pins a tab cannot still be holding it next episode
        (``tests/test_hook_lifetime.py`` is that claim, and the benchmark bug
        behind it). A panel's draggers are on the same side of that line and were
        simply never put back: the handles kept appearing and the gizmo kept
        moving, while the thing it was supposedly holding sat still.
        """
        phase = self.env.pipeline.hooks.rigid_force
        for dragger in self.draggers:
            if dragger.apply not in list(phase):
                phase.append(dragger.apply)
        self.env.pipeline.invalidate_graph()

    def build(self, gui) -> None:
        """Handles always; a folder only on the whole studio.

        Most of what this panel used to put in an ``Objects`` fold is now either
        the gesture itself or the session's: clicking a body attaches, clicking
        it again releases, and **Holding** / **Move together** / **Release
        last** / **Release all** sit at the top of the sidebar, where they are
        one set of controls for every kind rather than a set per kind. What is
        left is the way to reach a body a click cannot -- one behind the bench,
        one the camera cannot see round to -- which is worth a fold on a page
        about a task and is noise on a page about one rope.

        The panel itself does not care either way: it owns the draggers, the
        body table and the click handling, so a page with no chrome still drags.
        """
        self.handles = HandleSet(
            self.server.scene, "/studio/object_handle", count=len(self.draggers), scale=0.13, line_width=2.5
        )
        if not self.chrome:
            return
        with gui.add_folder("Objects", expand_by_default=False):
            self._dropdown = gui.add_dropdown("Object", options=tuple(self.names), initial_value=self.selected)
            self._dropdown.on_update(lambda _: setattr(self, "selected", str(self._dropdown.value)))
            grab = gui.add_checkbox("Grab on click", initial_value=self.grab_on_click)
            grab.on_update(lambda _: setattr(self, "grab_on_click", bool(grab.value)))
            attach = gui.add_button("Attach handle")
            attach.on_click(lambda _: setattr(self, "attach_requested", True))

    def drain(self) -> None:
        body, self.pending_body = self.pending_body, None
        if body is not None:
            self._attach(int(body))
        if self.attach_requested:
            self.attach_requested = False
            selected = self.selected
            chosen = self.env.handles.objects.get(selected)
            if selected in self.chains:
                # The middle segment: the least surprising place to take hold
                # of a chain nobody pointed at. Checked before the plain object
                # branch, because a chain is also in `objects` and its `body`
                # is the far end rather than anywhere a hand would choose.
                segments = self.chains[selected].bodies
                self._attach(int(segments[len(segments) // 2]))
            elif selected in self.arm_bodies:
                # The link nearest the tool: an arm is in the dropdown so that
                # it can be taken by the hand, and is not in `objects` at all.
                self._attach(int(self.arm_bodies[selected][0]))
            elif chosen is not None:
                self._attach(int(chosen.body))
        if self.release_requested:
            self.release_requested = False
            self._release(self.handles.newest())
        if self.release_all_requested:
            self.release_all_requested = False
            for slot in self.handles.live:
                self._release(slot)
        # Before the targets are read: a linked drag moves the handles that are
        # *not* being dragged, and a wrench aimed at last step's pose would lag
        # the gizmo by a step for every handle but one.
        self.handles.drain()
        for slot, dragger in zip(self.handles, self.draggers, strict=True):
            if dragger.active:
                dragger.set_target(slot.position, slot.quat)

    # ------------------------------------------------------------------ clicks
    def select_body(self, body: int) -> str:
        """Adopt the object a click landed on. Returns its name, or ``""``."""
        name = self.by_body.get(int(body), "")
        if name:
            self.selected = name
            if self._dropdown is not None:
                self._dropdown.value = name
        return name

    def click_body(self, body: int) -> str:
        """Select what a click hit, and take hold of it. Returns the report.

        Grabbing is what makes the gizmo mean something: the handle appears
        because the body is already on it. ``Grab on click`` turns that off for
        a session that wants clicking to name things and nothing more.
        """
        name = self.select_body(body)
        if not name:
            return ""
        if not self.grab_on_click:
            return f"{name} -- selected"
        self.pending_body = int(body)
        return f"{name} -- on a handle, drag it"

    # ----------------------------------------------------------------- handles
    def _attach(self, body: int) -> None:
        """Put a handle on a body at its current pose. Main thread."""
        name = self.by_body.get(int(body), "")
        if not name:
            return  # a static, or a body no handle names: nothing to drag
        # Clicking what you are already holding puts it down. ``Re-grip`` is
        # what re-seats a handle whose body has slipped out from under it.
        held = self.handles.find(int(body))
        if held is not None and held.live:
            self._release(held)
            return
        # The clicked body's own pose, not the entity's nominal one: for a
        # plain object they are the same row, and for a chain the clicked
        # segment is the thing being taken hold of.
        pose = np.asarray(self.env.state.body_q.numpy()[int(body)], dtype=np.float64)
        # Keyed by body, so clicking one that is already held re-seats its own
        # handle rather than putting a second wrench on it and letting the two
        # fight over the same body.
        slot = self.handles.acquire(key=int(body))
        slot.place(pose[:3], pose[3:7], key=int(body), label=name)
        self.draggers[slot.index].attach(int(body), slot.position, slot.quat, effective_mass=self.chain_mass.get(name))

    def holding_arm(self, name: str) -> bool:
        """Whether a live handle is on one of that arm's own links.

        Read by :class:`PanelArm` every step: an arm being tugged has to stop
        driving towards a setpoint it is being pulled away from, or the wrench
        spends itself fighting the servo instead of moving the arm. Keyed by
        body rather than by name because a body is what a slot carries --
        ``place(key=int(body))``.
        """
        bodies = self.arm_bodies.get(name)
        if not bodies or self.handles is None:
            return False
        held = {int(slot.key) for slot in self.handles.live if slot.key is not None}
        return not held.isdisjoint(bodies)

    def _release(self, slot) -> None:
        """Let one handle go. ``None`` is a no-op, so Release on an empty page is."""
        if slot is None:
            return
        self.draggers[slot.index].detach()
        slot.clear()

    def refresh(self) -> None:
        """Nothing of its own to write: the session reports what is held."""

    def close(self) -> None:
        super().close()
        if self.handles is not None:
            self.handles.close()
            self.handles = None

    def on_reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        del obs, info
        self.pending_body = None
        for dragger in self.draggers:
            dragger.detach()
        self.handles.release_all()
        self.contribute()


class PanelCloth(Panel):
    """Grab a patch of a deformable where you clicked, and carry it -- twice over.

    Aiming a grab is the whole difficulty. The grab seizes particles within a
    few centimetres of a point, and a point picked by dragging a floating gizmo
    into a thin sheet is rarely within a few centimetres of it. A click resolves
    against the particle cloud itself
    (:meth:`~robosimstudio_studio.pick.ScenePicker.particle_at`),
    so the point is on the cloth by construction, the handle it puts there is
    holding the cloth before it is drawn, and a miss is reported as a miss
    without costing a grip that had already landed.

    **Two handles, because one is not a manipulation.** A single grip can only
    translate a sheet; two can fold it, stretch it between them, hold one corner
    against the table while the other is lifted. That is what PhysTwin's
    playground calls ``--n_ctrl_parts 2``, and with ``Move together`` set the
    pair is carried as one rigid group by a drag on either one -- which is one
    mouse doing what two hands do.

    ``Pin to hand`` is named for what it is. It pins one handle's particles to
    an end-effector frame kinematically -- the hand does not close, nothing is
    held by contact, and the cloth cannot slip. That is genuinely useful for
    staging a scene, and it is not the robot doing it. A gripper actually
    closing on cloth is the ``Close`` slider on the arm panel, and the
    difference between the two is what the ``Held`` readout is for.

    **A body a foreign engine owns is held through that engine.** A SuperDex
    sponge's particles live in the Newton model -- which is what makes it
    clickable, drawable and recordable like any other deformable -- but they are
    flagged inactive and the bridge writes them from the host once per control
    step, into *both* states of the double buffer. A pin in the substep schedule
    therefore holds them for the length of a substep and is overwritten before
    anybody sees it: the handle appeared, the gizmo dragged, and the sponge did
    not move. So when the entity's island is a host engine the grab is handed to
    that engine's own compliant node constraints (``bridge.pin_nodes``, see
    :class:`_HostPin`), which is the same channel the surface-prep routines hold
    a sponge through. The click, the radius, the miss and the handle are
    unchanged -- only what ends up holding the patch differs, and the readout
    says which it was.

    **What it serves is what the scene observes.** Owning particles is not the
    question: a cuttable block owns eight, the corners its cage solver writes
    back after every control step so the page can draw the cut, with no probe
    to summarise them and nothing a pin on them would move. Listed here, it took
    the page down on the first refresh (``'EntityCuttable' object has no
    attribute 'observe'``, with ``panels="all"``). So the panel takes
    :meth:`~robosimstudio.sim.Sim.deformables`, the things carrying a particle
    probe -- every kind whose state *is* its particles, from a towel to the
    water to a SuperDex rope -- which is what ``obs["deformables"]`` reports.

    Args:
        session: The page.
        handles: How many patches may be held at once, each on its own gizmo.
    """

    def __init__(self, session, *, handles: int = 2, chrome: bool = True) -> None:
        super().__init__(session)
        from robosimstudio.core.draggers import ParticleDragger

        deformables = self.env.deformables()
        self.names = list(deformables)
        self.radius = 0.03
        """How far from a click particles are seized [m]. Also what counts as a miss."""
        self.draggers = [
            ParticleDragger(self.env.model, mode="pin", radius=self.radius) for _ in range(max(1, int(handles)))
        ]
        self.contribute()
        self.chrome = bool(chrome)
        """Whether to build a folder. See :meth:`PanelObject.build`."""
        self._held: Any = None
        self._state: Any = None
        self._dropdown: Any = None
        self.handles: HandleSet | None = None
        self.grab_on_click = True
        self.selected_set = ""
        """Which declared set ``Grab set`` takes hold of."""
        self.grab_set_requested = False
        self.regrip_requested = False
        self.release_requested = False
        self.release_all_requested = False
        self.follow_arm = [""] * len(self.draggers)
        """Which arm each handle rides, by slot. ``""`` leaves it on the mouse."""
        self.set_requests: deque[tuple[str, str, str, bool]] = deque()
        """A page's own set grabs and releases, ``(cloth, set, arm, hold)``, waiting for :meth:`drain`. See :meth:`request_set`."""
        self._follow: list[Any] = []
        """The ``Pin N to hand`` dropdowns, by slot; none on a page without the fold."""
        self._grab_pins: dict[int, str] = {}
        """Slots a set grab pinned to a hand, each with the pin it had before: the grab's to put back when it lets go."""
        self._requested: dict[int, tuple[str, str, str]] = {}
        """Slots :meth:`request_set` took, by the ``(cloth, set, arm)`` it was asked for."""
        self._retake: dict[int, str] = {}
        """Slots pinned in the dropdown while holding a grip, each with the hand it was pinned to, for :meth:`drain`
        to take again on that hand once :attr:`follow_arm` reads it."""
        self.selected = self.names[0] if self.names else ""
        self.pending: tuple[str, np.ndarray] | None = None
        """A click waiting to be serviced on the main thread."""
        self.miss = 0.0
        """How far the last click was from the cloth [m]; 0 when it landed."""
        self.pins: dict[int, _HostPin] = {}
        """Engine-side grips, by handle slot. See :class:`_HostPin`."""
        # Particle -> cloth, built once: a click asks this of one index and a
        # linear scan of every cloth's index list per click is a scan per click.
        self._owner = {
            int(i): name
            for name, cloth in deformables.items()
            for i in np.asarray(cloth.particles, dtype=np.int64).tolist()
        }

    def contribute(self) -> None:
        """Put this panel's pins back into the substep schedule.

        ``post_solve`` rather than ``pre_solve``: a pin writes positions, so it
        has to land where the solve cannot undo it. See
        :meth:`PanelObject.contribute` for why this is called again on every
        reset rather than only at construction.
        """
        phase = self.env.pipeline.hooks.post_solve
        for dragger in self.draggers:
            if dragger.apply not in list(phase):
                phase.append(dragger.apply)
        self.env.pipeline.invalidate_graph()

    def build(self, gui) -> None:
        """Handles always; a folder only on the whole studio.

        Same split as :meth:`PanelObject.build`, and for the same reason -- the
        release controls and the ``Held`` line are the session's now, one set for
        every kind. What stays here is what a click cannot say: grabbing a
        *declared* strip by name, re-seizing after the patch has stretched away,
        and riding a handle on a hand. A page about one towel has no use for any
        of it, and the missed-grab feedback the fold used to carry goes to the
        session's ``Holding`` line so it is not lost with the fold.
        """
        self.handles = HandleSet(
            self.server.scene, "/studio/cloth_handle", count=len(self.draggers), scale=0.10, line_width=2.0
        )
        if not self.chrome:
            return
        with gui.add_folder("Deformable", expand_by_default=False):
            self._dropdown = gui.add_dropdown("Cloth", options=tuple(self.names), initial_value=self.selected)
            self._dropdown.on_update(lambda _: self._pick(str(self._dropdown.value)))
            grab = gui.add_checkbox("Grab on click", initial_value=self.grab_on_click)
            grab.on_update(lambda _: setattr(self, "grab_on_click", bool(grab.value)))
            # Grabbing a *declared* strip, beside grabbing where you clicked.
            # The names are the garment's own -- and on the shirt they are the
            # names its goal is written over -- so "hold the hem edge" stops
            # being a thing you hunt for with a 3 cm cursor and hope you found.
            sets = self.set_names()
            self._set_dropdown = gui.add_dropdown("Set", options=sets or ("",), initial_value=sets[0] if sets else "")
            self._set_dropdown.on_update(lambda _: setattr(self, "selected_set", str(self._set_dropdown.value)))
            self.selected_set = sets[0] if sets else ""
            grab_set = gui.add_button("Grab set")
            grab_set.on_click(lambda _: setattr(self, "grab_set_requested", True))
            regrip = gui.add_button("Re-grip here")
            regrip.on_click(lambda _: setattr(self, "regrip_requested", True))
            arms = ("", *self.session.arms.keys())
            self._follow = []
            for slot in range(len(self.draggers)):
                follow = gui.add_dropdown(f"Pin {slot + 1} to hand", options=arms, initial_value=self.follow_arm[slot])
                # slot is bound per iteration, not read from the loop variable at
                # callback time -- viser fires these long after the loop is over.
                follow.on_update(lambda _, s=slot, w=follow: self._pick_arm(s, str(w.value)))
                self._follow.append(follow)
            self._held = gui.add_text("Held", initial_value="nothing", disabled=True)
            self._state = gui.add_text("Cloth", initial_value="", disabled=True)

    def set_names(self) -> tuple[str, ...]:
        """The selected cloth's declared particle sets, by their short names.

        Already resolved: the compiler evaluates each set's recipe against the
        rest layout and hands the indices to ``DeformableHandle.sets``, so this is a
        lookup rather than a selector run. The prefix a handle keys them under is
        the cloth's own name and is dropped for display -- ``far_edge`` under the
        shirt dropdown is unambiguous, and ``shirt/far_edge`` is only longer.
        """
        handle = self.env.handles.deformables.get(self.selected)
        if handle is None:
            return ()
        return tuple(sorted(key.split("/", 1)[-1] for key in handle.sets))

    def grab_set(self, set_name: str, arm: str = "", cloth: str = "") -> int:
        """Take hold of one declared set. Main thread. Returns particles held.

        Aimed at the set's own centroid and restricted to its own indices, with
        the seize radius widened to reach the furthest of them -- so what is held
        is the strip the garment declares, not whatever happened to be within
        3 cm of a cursor. The dragger's ``max_particles`` still caps it, keeping
        the ones nearest the centroid, which is why the readout says how many:
        the shirt's ``far_side`` is three thousand particles and a handle holding
        sixty of them should not look like a handle holding all of it. On a slot
        pinned to a hand it is aimed at the hand instead (:meth:`_hand_frame`),
        and the cap keeps the ones nearest the hand.

        ``arm`` pins the slot the grab takes to that hand, whichever slot that
        is: a page that pinned a slot by number before asking pinned the one a
        click grab held whenever the click came first. That pin is the grab's,
        shown on the slot's dropdown and put back when the grab lets go
        (:meth:`_forget`); a pin chosen in the dropdown is the user's and stays.
        ``cloth`` names the deformable; ``""`` is the selected one.

        Restores the click radius afterwards (:meth:`_grab_all`), because that
        number is the aim tolerance for the *other* way of grabbing and a set
        grab has no business changing what counts as a miss.
        """
        name = cloth or self.selected
        handle = self.env.handles.deformables.get(name)
        body = self.env.scene.by_name.get(name)
        if handle is None or body is None or not set_name:
            return 0
        indices = handle.sets.get(set_name) or handle.sets.get(f"{name}/{set_name}")
        if indices is None or not len(indices):
            return 0
        # Before a slot is taken: an unknown hand raised out of drain with the slot reserved, never placed.
        arms = self.env.scene.embodiment.iks if self.env.scene.embodiment else {}
        if arm and arm not in arms:
            self.session.status = f"{name}: no arm {arm!r} to hold {set_name!r} on; the arms are {sorted(arms)}"
            return 0
        slot = self._acquire()
        arm = arm or self.follow_arm[slot.index]
        if arm:
            centre, quat = self._hand_frame(arm)
        else:  # aimed at the set's own centroid, so only then is it read
            points = self.env.state.particle_q.numpy()[np.asarray(indices, dtype=np.int64)]
            centre, quat = points.mean(axis=0), IDENTITY_QUAT
        dragger = self.draggers[slot.index]
        taken = self._grab_all(dragger, indices, centre, quat)
        if not taken:
            self._release(slot)
            return 0
        if arm != self.follow_arm[slot.index]:
            self._grab_pins[slot.index] = self.follow_arm[slot.index]
            self._pin(slot.index, arm)
        self.miss = 0.0
        self._hand_over(slot, name, dragger, centre, quat)
        slot.place(centre, quat, key=("cloth", name), label=f"{set_name} ({taken})")
        if hasattr(body, "release_handle"):
            body.release_handle()
        return int(taken)

    def request_set(self, cloth: str, set_name: str, arm: str = "", *, hold: bool = True) -> None:
        """Ask for a grab of one declared set on the next step, or let go of one asked for. Any thread.

        A page's own way to :meth:`grab_set`: the request carries its cloth, set
        and hand, so the panel's ``Cloth`` and ``Set`` selections -- the user's
        -- are left as they are. ``hold=False`` lets go of the handles a request
        for the same three took, and of nothing else: a grip made by hand stays,
        pinned to the same hand or not.
        """
        self.set_requests.append((str(cloth), str(set_name), str(arm), bool(hold)))

    def _serve(self, cloth: str, set_name: str, arm: str, hold: bool) -> None:
        """Carry out one :meth:`request_set`. Main thread."""
        asked = (cloth, set_name, arm)
        if hold:
            if self.grab_set(set_name, arm=arm, cloth=cloth):
                self._requested[self.handles.newest().index] = asked
            return
        for handle in self.handles.live:
            if self._requested.get(handle.index) == asked:
                self._release(handle)

    def _pin(self, index: int, arm: str) -> None:
        """Pin slot ``index`` to ``arm`` (``""`` unpins), and show it on the slot's dropdown."""
        self.follow_arm[index] = arm
        if index < len(self._follow):
            self._follow[index].value = arm  # fires _pick_arm, which finds nothing changed

    def _pick_arm(self, index: int, arm: str) -> None:
        """A ``Pin N to hand`` dropdown changed: the slot's pin is the user's now, and a release leaves it.

        A grip the slot already holds is taken again in the hand's frame by
        :meth:`drain`, on the main thread, before the slot is first aimed at the
        hand: kept in the frame it was taken in, the patch was carried onto the
        hand, which :meth:`_seize` already prevents for a grip taken after the pin.
        """
        if arm == self.follow_arm[index]:
            return  # :meth:`_pin` showing a pin it made, not the user choosing one
        # Marked with the hand, before the pin is written: a drain that reads the
        # old pin keeps the mark for the step that reads the new one.
        if arm:
            self._retake[index] = arm
        else:
            self._retake.pop(index, None)
        self.follow_arm[index] = arm
        self._grab_pins.pop(index, None)

    def _retake_on_hand(self, index: int, arm: str) -> None:
        """Take slot ``index``'s grip again where it lies, in ``arm``'s frame (:meth:`_hand_frame`). Main thread."""
        position, quat = self._hand_frame(arm)
        pin = self.pins.get(index)
        if pin is not None:
            points = self.env.state.particle_q.numpy()[pin.nodes].astype(np.float64)
            pin.offsets = (points - position) @ quat_to_mat(quat)
            return
        dragger = self.draggers[index]
        self._grab_all(dragger, dragger.grabbed_indices.copy(), position, quat)

    def _forget(self, index: int) -> None:
        """What a slot's grab brought with it goes with the grab: the pin a set grab made, and its request."""
        self._requested.pop(index, None)
        self._retake.pop(index, None)
        if index in self._grab_pins:
            self._pin(index, self._grab_pins.pop(index))

    def owner_of(self, particle: int) -> str:
        """Which cloth a particle belongs to, or ``""``.

        The picker searches every particle rather than one cloth's, so that a
        click selects whichever deformable it landed on instead of only finding
        the one already in the dropdown.
        """
        return self._owner.get(int(particle), "")

    def select(self, name: str) -> None:
        if name and name in self.names:
            self.selected = name
            if self._dropdown is None:
                return  # no fold on this page, so nothing to keep in step
            self._dropdown.value = name
            # The set list belongs to the cloth, so it moves with the selection:
            # a page offering the towel's three strips while the shirt is
            # selected offers eleven names that are not there.
            self._sync_set_options()

    def _pick(self, name: str) -> None:
        """The Cloth dropdown changed by hand: the set list follows it.

        Not :meth:`select`, which writes the dropdown's value back and would
        fire this callback a second time.
        """
        self.selected = name
        self._sync_set_options()

    def _sync_set_options(self) -> None:
        """Point the set dropdown at whatever the selected cloth declares."""
        dropdown = getattr(self, "_set_dropdown", None)
        if dropdown is None:
            return
        sets = self.set_names()
        dropdown.options = sets or ("",)
        self.selected_set = sets[0] if sets else ""
        dropdown.value = self.selected_set

    def click_at(self, name: str, position) -> str:
        """Take hold of the patch a click landed on. Returns the report.

        The grab itself waits for :meth:`drain`, which is where this panel's
        Warp calls live -- so a click is serviced on the same control step it
        arrived on, but on the main thread.
        """
        self.select(name)
        if not self.grab_on_click:
            return f"{name} -- selected; Grab on click is off"
        self.pending = (name, np.asarray(position, dtype=np.float64).copy())
        return f"{name} -- on a handle, drag it"

    def drain(self) -> None:
        pending, self.pending = self.pending, None
        if pending is not None:
            self._grab(*pending)
        if self.grab_set_requested:
            self.grab_set_requested = False
            self.grab_set(self.selected_set)
        while self.set_requests:
            self._serve(*self.set_requests.popleft())
        if self.regrip_requested:
            self.regrip_requested = False
            self._regrip()
        if self.release_requested:
            self.release_requested = False
            self._release(self.handles.newest())
        if self.release_all_requested:
            self.release_all_requested = False
            for handle in self.handles.live:
                self._release(handle)
        # Before the targets are read, so a linked drag's followers are pinned
        # to where they are *now* rather than to last step's pose.
        self.handles.drain()
        for handle, dragger in zip(self.handles, self.draggers, strict=True):
            pin = self.pins.get(handle.index)
            if not dragger.active and pin is None:
                continue
            arm = self.follow_arm[handle.index]
            if arm:
                marked = self._retake.pop(handle.index, None)
                if marked == arm:
                    self._retake_on_hand(handle.index, arm)
                elif marked is not None:
                    # The dropdown moved on between reading the pin and the mark: the
                    # step that reads the new pin takes the grip on that hand.
                    self._retake.setdefault(handle.index, marked)
                # The hand is driving, so the handle follows it rather than the
                # other way round: a gizmo left behind on the bench while the
                # cloth rode away on the gripper is a gizmo that lies.
                position, quat = self.env.arm(arm).ee_pose(self.env.state)
                handle.position, handle.quat = np.asarray(position, dtype=np.float64), np.asarray(quat, np.float64)
                if not handle.dragging:
                    handle.sync()
            else:
                position, quat = handle.position, handle.quat
            # Aimed here rather than from a hook: this runs before the step, and
            # a host engine ticks in the step's ``after`` phase, so a target set
            # now is the one that engine drives to this control period.
            if pin is not None:
                if not pin.aim(position, quat):
                    self._release(handle)
            else:
                dragger.set_target(position, quat)

    # ----------------------------------------------------------------- handles
    def _grab(self, name: str, position) -> None:
        """Put a handle on the patch at ``position``. Main thread."""
        cloth = self.env.scene.by_name.get(name)
        if cloth is None:
            return
        # Asked before a slot is taken, because a slot taken on a full set
        # recycles a handle that was holding something: a click three
        # centimetres wide of the sheet must not cost a grip that landed.
        distance = self._distance_to(cloth, position)
        if distance >= self.radius:
            self.miss = distance
            return
        self.miss = 0.0
        # Clicking a grip you already have lets it go, the way clicking a body
        # you are already holding does. A cloth handle is keyed by its sheet
        # rather than by a particle -- a patch is not one index -- so "the same
        # grip" is one whose bead is within a radius of where you just clicked,
        # which is the same tolerance that decided the grab landed at all.
        for live in self.handles.live:
            if live.key == ("cloth", name) and float(np.linalg.norm(live.position - position)) < self.radius:
                self._release(live)
                return
        self._seize(self._acquire(), name, cloth, position, IDENTITY_QUAT)

    def _acquire(self):
        """A slot for a new grip, let go of whatever it was holding. Main thread.

        :meth:`HandleSet.acquire` recycles the oldest handle on a full set and
        leaves letting go to the caller -- and an engine-side pin left on a
        recycled slot would go on being aimed at the new handle's pose instead
        of the Newton grip that slot now carries.
        """
        slot = self.handles.acquire()
        self._drop_pin(slot.index)
        self.draggers[slot.index].release()
        self._forget(slot.index)
        return slot

    def _seize(self, slot, name: str, cloth, position, quat) -> bool:
        """Take hold of the patch at ``position`` on ``slot``. Main thread.

        ``slot`` holds nothing on the way in; a miss clears it and records how
        far off the aim was.
        """
        dragger = self.draggers[slot.index]
        # Not fallback_nearest: the point came from a click on the cloud itself,
        # so nothing inside the radius means the aim missed, and saying so beats
        # seizing a corner nobody pointed at.
        taken = dragger.grab(self.env.state, position, quat, candidates=cloth.particles, fallback_nearest=False)
        if not taken:
            self._release(slot)
            self.miss = dragger.last_distance
            return False
        self.miss = 0.0
        arm = self.follow_arm[slot.index]
        if arm:
            # Found where the click aimed, held where the hand is: the same patch, seized again in the
            # frame grab_set takes a set in on a pinned slot.
            position, quat = self._hand_frame(arm)
            self._grab_all(dragger, dragger.grabbed_indices.copy(), position, quat)
        self._hand_over(slot, name, dragger, position, quat)
        slot.place(position, quat, key=("cloth", name), label=f"{name} ({taken})")
        # A task with a rigid pull-tab must let go of it once something else
        # has taken the film, or the two holds fight over the same particles.
        if hasattr(cloth, "release_handle"):
            cloth.release_handle()
        return True

    def _hand_frame(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        """The frame a grip on a slot pinned to ``arm`` keeps its offsets in: the hand's pose, now.

        :meth:`drain` aims a pinned slot at the hand's pose every step, so a grip
        taken in any other frame -- the identity at a click, or at a set's
        centroid -- is moved onto the hand on the next step and turned by its
        orientation: a YAM pointing down holds its strip mirrored end for end.
        Taken in the hand's frame, the patch keeps the place it has relative to
        the fingers. Both ways of grabbing, a click and a declared set, use it.
        """
        position, quat = self.env.arm(arm).ee_pose(self.env.state)
        return np.asarray(position, dtype=np.float64), np.asarray(quat, dtype=np.float64)

    def _grab_all(self, dragger, indices, position, quat) -> int:
        """Seize ``indices`` in the frame at ``position``, the radius widened to reach the furthest of them.

        Up to the dragger's ``max_particles``, the nearest kept. Restores the
        click radius afterwards: it is the aim tolerance of a click, and a grab
        of particles already chosen has no business changing what counts as a
        miss.
        """
        points = self.env.state.particle_q.numpy()[np.asarray(indices, dtype=np.int64)]
        was = dragger.radius
        dragger.radius = float(np.linalg.norm(points - position, axis=1).max()) + 1e-6
        try:
            return dragger.grab(self.env.state, position, quat, candidates=indices, fallback_nearest=False)
        finally:
            dragger.radius = was

    def _hand_over(self, slot, name: str, dragger, position, quat) -> bool:
        """Give a landed grab to the engine that owns the body. Main thread.

        The particles were selected by the dragger -- same radius, same ceiling,
        same miss -- and this only changes what holds them: the engine's node
        constraints instead of a kernel the engine's next tick would overwrite.
        The dragger is released rather than left running beside the pin, so one
        thing holds the patch and the substep schedule launches over nothing.

        Returns:
            Whether the engine took it. ``False`` leaves the Newton pin holding,
            which is right for every deformable Newton is actually solving.
        """
        from robosimstudio.domains.capabilities import host_bridge

        # ``getattr`` for the same reason ``_record`` and ``declared_states``
        # read their env that way: a scene with no pipeline to ask cannot say
        # which engine owns the body, and "nobody but Newton" is the right
        # answer to that question rather than an error to raise on a grab.
        bridge = host_bridge(getattr(self.env, "pipeline", None), name)
        if bridge is None or not hasattr(bridge, "pin_nodes"):
            return False
        nodes = np.asarray(dragger.grabbed_indices, dtype=np.int64)
        if not len(nodes):
            return False
        position = np.asarray(position, dtype=np.float64)
        points = self.env.state.particle_q.numpy()[nodes].astype(np.float64)
        try:
            pin = bridge.pin_nodes(name, nodes, points.copy())
        except Exception as exc:
            # A body on a host island that the bridge did not build an actor for
            # is the case this catches, and it is reported rather than raised:
            # the Newton pin still holds -- uselessly for this body, but the
            # page is not the place to find that out by traceback.
            self.session.status = f"{name}: the engine would not pin it -- {type(exc).__name__}: {exc}"[:200]
            return False
        dragger.release()
        self.pins[slot.index] = _HostPin(
            bridge=bridge,
            pin=int(pin),
            name=name,
            nodes=nodes,
            offsets=(points - position) @ quat_to_mat(quat),
        )
        return True

    def _drop_pin(self, index: int) -> None:
        """Let go of one slot's engine-side grip, if it has one."""
        pin = self.pins.pop(int(index), None)
        if pin is not None:
            pin.release()

    def _held_count(self, index: int) -> int:
        """How many nodes slot ``index`` holds, whichever half is holding them."""
        pin = self.pins.get(int(index))
        return len(pin.nodes) if pin is not None else self.draggers[int(index)].n_grabbed

    def _regrip(self) -> None:
        """Re-seize whatever is under the newest handle, where it is now.

        The recovery when a carried patch has stretched away from its handle, or
        when a handle has been dragged onto a different part of the sheet. With
        nothing held there is nothing to re-seize -- the way to take a first
        grip is to click the cloth.
        """
        slot = self.handles.newest()
        if slot is None:
            return
        # Seized again on the same slot, in the handle's own frame, rather than
        # through _grab: a click at a grip you already hold lets it go, and the
        # newest handle is always at distance 0 from its own position.
        name = str(slot.key[1])
        cloth = self.env.scene.by_name.get(name)
        if cloth is None:
            return
        position, quat = slot.position.copy(), slot.quat.copy()
        self._drop_pin(slot.index)
        self.draggers[slot.index].release()
        self._seize(slot, name, cloth, position, quat)

    def _distance_to(self, cloth, position) -> float:
        """How far ``position`` is from the nearest particle of ``cloth`` [m]."""
        indices = np.asarray(cloth.particles, dtype=np.int64)
        if not len(indices):
            return float("inf")
        points = self.env.state.particle_q.numpy()[indices]
        return float(np.linalg.norm(points - np.asarray(position, dtype=np.float64), axis=1).min())

    def _release(self, slot) -> None:
        """Let one handle go. ``None`` is a no-op, so Release on an empty page is."""
        if slot is None:
            return
        self._drop_pin(slot.index)
        self.draggers[slot.index].release()
        self._forget(slot.index)
        slot.clear()

    def declared_states(self) -> str:
        """What the selected cloth's asset says it publishes, as it reads now.

        This line used to be ``summary.get("bond_broken_fraction")``: one
        entity's ``observe()`` override, named by hand, in a panel that serves
        every deformable. So the tape showed one of the nine states it declares,
        under a spelling of its own, and the film -- which declares ten -- showed
        none, because its entity does not use that key.

        Read through the entity's own ``published_states``, which is the reader a
        goal clause resolves through, so the page and the score cannot come from
        different numbers. It is passed the context the step already gathered:
        that reader caches on the context object, so this costs one dict lookup
        rather than a second device read, which is what makes it safe to call at
        refresh rate.

        Ordered and tiered by the declaration rather than by the dict, so
        ``internal`` is visible as internal -- no camera reads a bond count, and
        a number that only the simulator can know is worth knowing that about.
        """
        # Both by ``getattr``: a hand-assembled scene has neither a document to
        # declare states nor the task runtime's cached observation context, and
        # the guard below already treats a missing one as nothing to report.
        record, context = self._record(), getattr(self.env, "last_context", None)
        entity = self.env.scene.by_name.get(self.selected)
        if record is None or context is None or entity is None:
            return ""
        reader = getattr(entity, "published_states", None)
        values = dict(reader(context)) if callable(reader) else {}
        parts = []
        for state in record.published:
            value = values.get(state.name.replace("-", "_"))
            if value is None:
                continue
            unit = f" {state.unit}" if state.unit else ""
            mark = "*" if state.tier == "internal" else ""
            parts.append(f"{state.name} {float(value):.3g}{unit}{mark}")
        return "  |  ".join(parts)

    def _record(self):
        """The selected cloth's declaration, or ``None`` if nothing declares it.

        ``getattr`` because a page over a hand-assembled scene has no document at
        all -- the same reason the session reads it that way to build its
        :class:`~robosimstudio_studio.selection.SceneIndex`. A towel somebody put
        in a ``Scene`` has no asset entry to publish states from, and that is not
        an error; it is the readout having nothing extra to say.
        """
        doc = getattr(self.env, "doc", None)
        if doc is None:
            return None
        from robosimstudio.assetlib.registry import ASSETS

        types = {o.name: o.type for o in doc.objects}
        try:
            return ASSETS.get(types[self.selected])
        except KeyError:
            return None

    def refresh(self) -> None:
        if self._state is None:
            return  # no fold on this page; the session reports what is held
        summary = self.env.scene.by_name[self.selected].observe(self.env.state, self.env.step_count)
        # The z band is the panel's own: where the sheet is, which is a thing to
        # look at rather than a thing the asset declares.
        text = f"z {summary['z_min']:.3f}..{summary['z_max']:.3f} m"
        declared = self.declared_states()
        if declared:
            text += "  |  " + declared
        self._state.value = text
        held = [
            f"{handle.index + 1}: {self._held_count(handle.index)} particles"
            + (" through the engine" if handle.index in self.pins else "")
            + (f" pinned to {self.follow_arm[handle.index]}" if self.follow_arm[handle.index] else "")
            for handle in self.handles.live
        ]
        if held:
            self._held.value = "  |  ".join(held)
        elif self.miss:
            self._held.value = f"missed by {self.miss * 1000:.0f} mm -- click on the cloth"
        else:
            self._held.value = "nothing"

    def close(self) -> None:
        super().close()
        # Before the handles go: an engine-side constraint outlives this page,
        # so a session closed while holding a sponge would leave it hanging in
        # mid-air for whatever steps the scene next.
        for index in list(self.pins):
            self._drop_pin(index)
        if self.handles is not None:
            self.handles.close()
            self.handles = None

    def on_reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        del obs, info
        self.pending = None
        self.set_requests.clear()  # asked of the episode the reset ended
        # The engine has already dropped these -- ``restore_rest`` unpins before
        # it restores, since a pin made since the capture is a constraint the
        # captured composition does not have -- so this is the page catching up
        # with what the reset did, and ``unpin`` on one already gone is a no-op.
        for index in list(self.pins):
            self._drop_pin(index)
        for dragger in self.draggers:
            dragger.release()
        self.handles.release_all()
        for index in range(len(self.draggers)):
            self._forget(index)
        self.miss = 0.0
        if self._held is not None:
            self._held.value = "nothing"
        self.contribute()
