# SPDX-License-Identifier: Apache-2.0
"""Running every world's motion, one control step at a time, for all of them.

This is the loop the whole design converges on. Once per control step it:

1. **Drains newly submitted motions** and installs them in the fleet's device
   arrays -- a joint segment, a servo goal, a gripper ramp. Host work
   proportional to the number of motions *started*, not to the number of worlds.
2. **Steps the fleet.** Two graph launches, no synchronize, no per-world work.
3. **On cadence ticks, reads back once** and decides which motions have
   finished, which have run out of budget, and which world's episode is over.
   Every one of those decisions is numpy over the whole fleet.

The asymmetry between steps 1-2 and step 3 is deliberate. Completion is the only
thing that needs the host to look at the state, so it is the only thing paced by
a cadence -- and a motion finishing up to ``cadence - 1`` steps late costs a few
extra control steps, never a wrong answer. Everything a graph *decides* still
happens at motion boundaries, where it always did.

A world with no active motion is not written at all, which is what makes a world
whose script is thinking free: it holds its last targets exactly, and the fleet
does not know or care that it is idle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.core.bus import BusTimeout
from robosimstudio.core.math import quat_to_mat
from robosimstudio.core.streams import reading
from robosimstudio.solvers.ik.tool_frame import tcp_pose, tool_offsets

from ..servo import rotation_error, solver_goal
from .intents import GripperMove, JointTrajectory, MoveToJoints, ServoToPose, slot_kind

__all__ = ["MotionNotInstallable", "StepService"]


class MotionNotInstallable(RuntimeError):
    """A motion this fleet cannot perform, raised on the node that asked for it.

    The alternative -- completing the motion with no error and no steps -- is
    indistinguishable downstream from a move that arrived, because the
    connector's pose verbs discard what the servo returns and the graph reads
    only "the tool call came back". A wave that cannot do something a serial run
    can must say so where the graph can route on it, not produce a lower number
    with nothing to explain it. See :meth:`StepService._install_one`.
    """


class _Program:
    """What the fleet installed for one running motion, and how to end it.

    Attributes:
        kind: Which intent this came from, for the completion test.
        goal: The configuration or command the motion is heading for.
        deadline: The control step past which it has run out of budget.
        tolerance: How close counts as arrived.
        settle_until: For a trajectory, the step its last segment lands on --
            before that, "close to the goal" is not evidence of anything.
    """

    __slots__ = (
        "best_dist",
        "deadline",
        "goal",
        "honours_roll",
        "kind",
        "lead",
        "reach_judged_at",
        "requested_quat",
        "rot_tolerance",
        "settle_until",
        "solver_quat",
        "stall_epsilon",
        "stall_steps",
        "stalled",
        "steps",
        "stop_on_stall",
        "tolerance",
    )

    def __init__(
        self,
        kind: str,
        *,
        goal: Any,
        deadline: int,
        tolerance: float,
        settle_until: int = 0,
        lead: Any = None,
        requested_quat: Any = None,
        rot_tolerance: float = float("inf"),
        honours_roll: bool = False,
        stop_on_stall: bool = False,
        stall_steps: int = 12,
        stall_epsilon: float = 0.0002,
        solver_quat: Any = None,
    ) -> None:
        self.kind = kind
        self.solver_quat = None if solver_quat is None else np.asarray(solver_quat, dtype=np.float64)
        """The one orientation the lead is driven toward for the whole motion:
        ``requested_quat`` restated by :func:`~robosimstudio_policy_agent.servo.solver_goal`
        from the measurement at install (or at the first lead tick, for a
        program built without one)."""
        self.goal = goal
        self.deadline = int(deadline)
        self.tolerance = float(tolerance)
        self.settle_until = int(settle_until)
        self.lead = lead
        self.requested_quat = None if requested_quat is None else np.asarray(requested_quat, dtype=np.float64)
        """The orientation the caller asked for, before roll was freed -- what
        arrival is measured against, the way the single-world servo measures it."""
        self.rot_tolerance = float(rot_tolerance)
        self.honours_roll = bool(honours_roll)
        self.stop_on_stall = bool(stop_on_stall)
        self.stall_steps = int(stall_steps)
        self.stall_epsilon = float(stall_epsilon)
        # The single-world servo's own progress bookkeeping, per motion: the
        # closest the tool has come, how many turns since it last came closer,
        # and how many turns it has had. One turn is one control step.
        self.best_dist = float("inf")
        self.stalled = 0
        self.steps = 0
        self.reach_judged_at = 0
        """The stall count at which the goal was last judged for reach, so a
        move that stays stalled is judged once per stall window, not per tick."""
        """The rate-capped setpoint this servo is being led by.

        Every servo has one, full speed included -- the law's own ceiling is
        what serial runs at -- and it is advanced once per control step by the
        host, see :meth:`StepService._advance_lead` and :meth:`StepService._lead`.
        ``None`` only when the arm has no solver."""


class StepService:
    """Turns submitted motions into device work, and finished motions into wake-ups.

    Args:
        fleet: The batched environment providing the primitives.
        intents: Where graph threads submit motions.
        snapshots: Where per-world views are published.
        service: The marshalled-call queue, pumped between steps.
        waiter: The step waiter graph threads park on.
        arm_names: Arms in runtime ``arm_id`` order, so an intent's ``arm``
            selects the same hand a graph meant.
        cadence: Control steps between readbacks. 1 checks completion every
            step; higher trades that many steps of latency for less host work.
    """

    def __init__(
        self,
        fleet,
        *,
        intents,
        snapshots,
        service,
        waiter,
        arm_names: list[str],
        cadence: int = 1,
    ) -> None:
        self.fleet = fleet
        self.intents = intents
        self.snapshots = snapshots
        self.service = service
        self.waiter = waiter
        self.arm_names = list(arm_names)
        self.cadence = max(1, int(cadence))
        self.step = 0
        self._retired: set[int] = set()
        self._tool_offsets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ install
    def install(self, live: np.ndarray) -> int:
        """Drain and install every newly submitted motion. Returns how many.

        Called before the fleet steps. A motion for a world that is no longer
        live is completed with an error rather than installed -- its episode is
        over, and running it would move a scene that has already been scored.
        """
        drained = self.intents.drain(step=self.step)
        installed = 0
        for world, arm, intent, _handle in drained:
            kind = slot_kind(intent)
            if world in self._retired or not bool(live[world]):
                self.intents.finish(
                    world, arm, kind=kind, error=BusTimeout(f"world {world}'s episode has ended"), step=self.step
                )
                continue
            program = self._install_one(world, arm, intent)
            if isinstance(program, str):
                # An error, not a quiet completion: see _install_one.
                self.intents.finish(
                    world,
                    arm,
                    kind=kind,
                    error=MotionNotInstallable(f"world {world}, arm {arm}: {program}"),
                    step=self.step,
                )
                continue
            self.intents.attach(world, arm, program, kind=kind)
            installed += 1
        return installed

    GRIPPER_TRAVEL_STEPS = 40
    """Steps a hand is given to reach a commanded width before it is called done.

    A ceiling rather than a wait: the fingers are checked against the command
    every cadence tick and a hand that arrives early stops there. It exists so a
    close that never fully arrives -- because it is holding something, which is
    the whole point -- still ends.
    """

    GRIPPER_TOLERANCE = 0.002
    """How close a finger coordinate counts as arrived [m or rad]."""

    UNREACHABLE_GATE_M = 0.03
    """How far a stalled servo must still be from its goal before the goal is
    judged for reach -- the serial IK backend's own reach gate
    (``IKBackend.tolerance``), so the verdict is the one that backend
    would give."""

    def _install_one(self, world: int, arm: int, intent) -> _Program | str:
        """Write one motion into the fleet's device arrays.

        Returns the running program, or a *sentence* saying why this fleet cannot
        perform the motion -- which :meth:`install` turns into an error on the
        waiting node.

        It used to return ``None`` for both "cannot" cases and let ``install``
        complete the handle with no error and no steps. That reads, to the
        caller, exactly like a motion that ran and arrived: ``WorldFacade``'s
        wait hands back its ``{"outcome": "superseded"}`` default, the
        connector's pose verbs discard it, and ``waypoint_move`` reports
        ``done: true, flown_z: 1.100`` for an arm that never moved. Measured on
        ``tool_sort/level1_1``, which is host-tier so ``fleet.ik`` is ``None``
        and **every** Cartesian servo leg took that path: the graph grasped the
        tool, was told the transport flew, opened the hand over the pick site and
        scored 0/1 where the serial run scores 1/1 -- with nothing in the row
        saying a motion had been skipped.
        """
        name = self.arm_names[arm] if arm < len(self.arm_names) else None
        if name is None:
            return f"arm {arm} is not one of this fleet's arms {list(self.arm_names)}"
        # Where the arm is *commanded* now, read before anything is released --
        # a new segment starts from the last command, not from wherever contact
        # has left the arm stuck.
        start = self._arm_targets(world, name)
        # One driver per arm at a time. A segment left enabled from a finished
        # move would keep stamping its endpoint over whatever comes next, and an
        # enabled solver would keep solving a goal nobody is waiting for. The
        # hand is not one of the arm's drivers: a grip must not stop a carry.
        if not isinstance(intent, GripperMove):
            self._release_arm(world, name)

        if isinstance(intent, MoveToJoints):
            driver = self.fleet.segments.get(name)
            if driver is None:
                return f"arm {name!r} has no joint-segment driver, so a joint move cannot be installed"
            driver.plan(world, start, intent.target, intent.duration_steps)
            budget = intent.max_steps or intent.duration_steps + 1
            return _Program(
                "joints",
                goal=np.asarray(intent.target, dtype=np.float64),
                deadline=self.step + int(budget),
                tolerance=intent.tolerance,
                settle_until=self.step + int(intent.duration_steps),
            )

        if isinstance(intent, JointTrajectory):
            driver = self.fleet.segments.get(name)
            if driver is None:
                return f"arm {name!r} has no joint-segment driver, so a trajectory cannot be installed"
            if len(intent.waypoints) == 0:
                return f"the trajectory for arm {name!r} carries no waypoints"
            # Every waypoint, one per ``steps_per_waypoint`` control steps --
            # two by default, the single-world facade's measured cadence (see
            # ``JointTrajectory``). This used to plan one smoothstep from the start
            # to the *last* waypoint and throw the rest away: a straight
            # Cartesian line from ``plan_linear`` became a joint-space chord
            # between its endpoints -- the elbow swing that solver's own notes
            # warn about -- and took eight steps per waypoint doing it.
            duration = driver.plan_path(world, intent.waypoints, intent.steps_per_waypoint)
            return _Program(
                "joints",
                goal=np.asarray(intent.waypoints[-1], dtype=np.float64),
                deadline=self.step + duration + int(intent.settle_max_steps),
                tolerance=intent.settle_tolerance,
                settle_until=self.step + duration,
            )

        if isinstance(intent, GripperMove):
            bank = self.fleet.grippers.get(name)
            if bank is None:
                return f"arm {name!r} has no gripper bank, so a hand command cannot be installed"
            # The ramp is the bank's, on device, from wherever this world's
            # command is at the flush -- not a loop of one-step commands here.
            bank.set(world, intent.command, ramp_steps=int(intent.ramp_steps))
            # A hand move finishes when the fingers have *travelled*, not when
            # the command has been written. Waiting a fixed number of steps is
            # what the caller asked for and is not what the caller meant: a close
            # with no ramp would complete on the very next step, the verify node
            # would look at fingers that have barely started moving, and the
            # grasp would be reported empty on an object the hand goes on to hold
            # perfectly well. A ramped close cannot be called arrived before the
            # ramp has ended either: the fingers track the blend the whole way.
            budget = max(int(intent.ramp_steps) + int(intent.settle_steps), self.GRIPPER_TRAVEL_STEPS)
            return _Program(
                "gripper",
                goal=(float(intent.command), name),
                deadline=self.step + budget,
                tolerance=self.GRIPPER_TOLERANCE,
                settle_until=self.step + int(intent.ramp_steps) + int(intent.settle_steps),
            )

        if isinstance(intent, ServoToPose):
            rig = self.fleet.ik
            if rig is None:
                # Host tier: `plan_ik_model` found the model too hostile for a
                # batched solve and said so, `build_ik_rig` returned None, and a
                # ServoToPose has nothing to drive. The serial connector answers
                # the same pose verbs with a host feedback servo, so this is a
                # real gap between the two paths and not a property of the task
                # -- named here rather than skipped, because a skipped Cartesian
                # leg is indistinguishable downstream from one that arrived.
                reason = getattr(getattr(self.fleet, "ik_plan", None), "reason", "") or "no batched IK tier"
                return (
                    f"this wave has no batched IK rig, so a Cartesian servo move cannot run "
                    f"(ik tier {getattr(getattr(self.fleet, 'ik_plan', None), 'tier', '?')!r}: {reason}). "
                    f"Plan the leg with motion.plan_linear and fly it with robot.execute_trajectory, "
                    f"which a wave does install, or run this task serially."
                )
            if name not in rig.arms:
                return f"arm {name!r} is not in this wave's IK rig {sorted(rig.arms)}"
            solver = rig.arms[name]
            # The single-world servo's own goal law (``CartesianServo._solver_goal``):
            # the goal unchanged for a roll-honouring solve, else the MEASURED
            # orientation turned by the shortest arc that lands its approach
            # axis on the goal's -- stated ONCE here and held for the motion
            # (``solver_quat``; ``_advance_lead`` drives the lead toward it), as
            # the serial loop holds it for its move. Restating it from every
            # tick's measurement ratcheted the roll on a blocked leg: the LM
            # solve rolled the wrist to gain position and the next tick asked
            # it to keep that roll. Two other laws were measured wrong here: the
            # wrist's current x projected off the goal axis, taken once at
            # install, left the hand 78 degrees from the pose tool_hanging's
            # carry asked for (the arc from flat to vertical carries roll
            # differently from a projection -- the shortest arc, held, carries
            # it the same way the tick-by-tick restatement does along an
            # unblocked geodesic); and the full requested quaternion stalled the
            # pre-grasp leg 16 mm short, the saddle the docstring below describes.
            measured_pos, measured_quat = self._tcp(world, name)
            solver_quat = solver_goal(measured_quat, intent.quat_xyzw, intent.honours_roll)
            lead = self._lead(world, name, intent)
            # The FIRST target is the law's first step, exactly as the serial
            # servo's first iteration is ``law.advance`` from the measurement:
            # position led by at most one step, orientation slerped by at most
            # one step. This used to hand the solver the law's *endpoint*
            # orientation beside a measured position: on a leg whose requested
            # orientation is far from the current one, the per-step solve landed
            # on a flipped elbow branch for exactly one control step (a 2.9 rad
            # target jump, in every world, at the first tick of every such
            # servo) and yanked the arm -- which on tool_hanging's carry is the
            # step the wrench left the jaws.
            if lead is None:
                target, quat = np.asarray(intent.position, dtype=np.float64), solver_quat
            else:
                target, quat = lead.advance(
                    measured_pos, measured_quat, np.asarray(intent.position, dtype=np.float64), solver_quat
                )
            solver.set_target(world, target, quat)
            solver.enable_world(world, True)
            requested = np.asarray(intent.quat_xyzw, dtype=np.float64)
            norm = float(np.linalg.norm(requested))
            return _Program(
                "servo",
                goal=(np.asarray(intent.position, dtype=np.float64), name, np.asarray(quat, dtype=np.float64)),
                deadline=self.step + int(intent.max_steps),
                tolerance=intent.pos_tolerance,
                lead=lead,
                requested_quat=requested / norm if norm > 0.0 else requested,
                rot_tolerance=intent.rot_tolerance,
                honours_roll=intent.honours_roll,
                stop_on_stall=intent.stop_on_stall,
                stall_steps=intent.stall_steps,
                stall_epsilon=intent.stall_epsilon,
                solver_quat=solver_quat,
            )
        return f"{type(intent).__name__} is not a motion this fleet knows how to install"

    def _offsets(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """One arm's ``(ee_offset, ee_offset_rot)``, read once and kept."""
        cached = self._tool_offsets.get(name)
        if cached is None:
            cached = self._tool_offsets[name] = tool_offsets(self.fleet, name)
        return cached

    def _lead(self, world: int, name: str, intent: ServoToPose) -> Any:
        """The rate-capped setpoint every servo is led by.

        **The law is the single-world servo's, not a copy of it.**
        ``OscSetpoint`` is constructed here with the same control period and the
        same ceiling scaled the same way, so a wave's move and a serial one's are
        the same arithmetic rather than two implementations that have to be kept
        equal by hand. ``speed_scale`` lowers the ceiling, and with it both the
        cruise rate and the command lead, by one factor -- which is what makes it
        a Cartesian speed limit rather than a gain change.

        **Full speed is the law's own ceiling, not no law.** The first cut wrote
        an uncapped move's goal to the device once and let the joint servo chase
        it, which is not what serial does at ``speed_scale=1.0``: the serial
        servo always leads the arm through ``OscSetpoint`` at ``OSC_OUTPUT_MAX``
        (0.05 m per step of lead, ~0.31 m/s cruise). Measured on the seed graph's
        first move after a grasp, a 20 cm hover: serial took the tool at most
        11 mm/step and kept the sponge; the fleet took it 48 mm/step -- a
        twenty-centimetre jump in five control steps -- and the sponge left the
        pads in every world, then the same 400-step recovery that serial never
        needed. The lead is one small host computation per control step per
        moving arm, which a wave can afford; an arm flung at 3 m/s is not a
        speed a graph author ever asked for.

        **And the ceiling starts where the task says it does.** A suite that
        states ``(:servo-speed-mps X)`` has its graphs written for a tool that
        cruises at X, and the serial servo turns that into a base factor on the
        ceiling (``CartesianServo.base_speed_scale``, ``min(1, X / cruise)``)
        that every move's own ``speed_scale`` multiplies. The same factor is
        applied here, from the same law arithmetic
        (:meth:`~robosimstudio.solvers.OscSetpoint.speed_scale_for`), so a
        graph's 0.5 means half the speed it was written for on both runners.
        """
        # ``is None``, not ``or 1.0``: a scale of exactly 0.0 is falsy, so the
        # short form turned "do not move" into "full speed" -- the silent wrong
        # answer this whole argument is about, reintroduced inside the fix for
        # it. The range check comes before the fast path for the same reason.
        stated = getattr(intent, "speed_scale", None)
        scale = 1.0 if stated is None else float(stated)
        if not 0.0 < scale <= 1.0:
            raise ValueError(f"speed_scale must be in (0, 1], got {scale}")
        from robosimstudio.solvers.motion.osc import OSC_OUTPUT_MAX, OscSetpoint

        control_dt = float(self.fleet.control_dt)
        # The task's declared servo speed, as the serial servo applies it. Read
        # through ``getattr`` because the stepper's own tests stand a fleet in
        # with no task behind it, and an undeclared task is the law's own cruise.
        declared = getattr(getattr(self.fleet, "task", None), "servo_speed_mps", None)
        scale *= OscSetpoint(control_dt=control_dt).speed_scale_for(declared)
        law = OscSetpoint(
            control_dt=control_dt,
            output_max=np.asarray(OSC_OUTPUT_MAX, dtype=np.float64) * scale,
        )
        law.reset(*self._tcp(world, name))
        return law

    def _tcp(self, world: int, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Where this world's tool centre is now, as the lead law measures it.

        The **tool-centre point in the grasp frame** -- ``ee_body`` with the
        hand's offset and rotation composed on -- which is the same quantity
        :meth:`_finished` tests arrival against, the batched solver drives, and
        every commanded pose is stated in. Using one thing for all of them is not
        tidiness: this used to return the raw wrist row, and on a hand whose
        tool centre sits 134.7 mm past its wrist the lead law then clamped every
        setpoint to a ball around a point the tool was never at, while the
        arrival test measured a distance that could never close. See
        :mod:`robosimstudio.solvers.ik.tool_frame` for the measurement that found it.
        """
        index = self.fleet.scene.embodiment.arms[name]
        with reading():
            body = np.asarray(self.fleet.pipeline.state_0.body_q.numpy(), dtype=np.float64)
        rows = self.fleet.worlds.body_view(body, world)
        return tcp_pose(rows, index.ee_body, *self._offsets(name))

    def _arm_targets(self, world: int, name: str) -> np.ndarray:
        """Where this arm is currently commanded, as the start of a new segment.

        The *commanded* position rather than the measured one, for the reason the
        single-world facade gives: starting a segment from where the encoder is
        makes an arm that contact is holding back re-plan from where it is stuck,
        and the next move then begins with a jump.

        Read straight off the control vector rather than reconstructed from
        whichever driver wrote it last -- that array is the answer by definition,
        and installs are sparse enough (one per motion started, not one per step)
        that the copy does not matter.
        """
        index = self.fleet.scene.embodiment.arms[name]
        targets = self.fleet.pipeline.control.joint_target_q
        if targets is None:
            return np.zeros(len(index.arm_coords), dtype=np.float32)
        with reading():
            flat = np.asarray(targets.numpy(), dtype=np.float32)
        return np.array(self.fleet.worlds.coord_view(flat, world)[index.arm_coords], dtype=np.float32)

    # -------------------------------------------------------------- completion
    def settle(self, snapshot, *, infos: list[dict] | None = None) -> None:
        """Publish the fleet's state and retire whatever has finished.

        Called on cadence ticks, after the fleet has stepped. Publishing first
        matters: a thread woken by a completing motion reads the snapshot
        immediately, and it should see the state its motion ended in rather than
        the one before it.
        """
        from .obs_format import world_observations

        obs = world_observations(self.fleet, snapshot)
        self.snapshots.publish(snapshot, infos=infos, obs=obs)
        self.waiter.notify(snapshot.step)
        self.step = int(snapshot.step)

        for world, arm, intent, program, _started in self.intents.active():
            if program is None:
                continue
            self._advance_lead(world, program, snapshot)
            done, error, result = self._finished(world, arm, program, snapshot)
            if done:
                self.intents.finish(world, arm, kind=slot_kind(intent), error=error, result=result, step=self.step)

    def _advance_lead(self, world: int, program: _Program, snapshot) -> None:
        """Move a capped servo's setpoint one tick closer, and re-aim at it.

        Once per *control step*, not once per readback: the law integrates at the
        control period, so advancing it once per cadence tick would let a wave at
        ``eval_every=4`` travel a quarter as fast as the same move run serially.
        The clamp is nonlinear, so the ticks are taken one at a time rather than
        by scaling ``dt`` -- which would also let a blocked arm creep.
        """
        if program.lead is None or program.kind != "servo":
            return
        goal, name, quat = program.goal
        rig = self.fleet.ik
        solver = rig.arms.get(name) if rig is not None else None
        if solver is None:
            return
        index = self.fleet.scene.embodiment.arms[name]
        measured_pos, measured_quat = tcp_pose(snapshot.body(world), index.ee_body, *self._offsets(name))
        # ONE goal orientation for the whole motion, restated at install (or
        # here, on the first tick of a program built without one) -- exactly as
        # the single-world servo restates it once before its first ``law.advance``.
        if program.solver_quat is None:
            program.solver_quat = solver_goal(measured_quat, program.requested_quat, program.honours_roll)
        quat = program.solver_quat
        for _ in range(self.cadence):
            set_pos, set_quat = program.lead.advance(measured_pos, measured_quat, goal[:3], quat)
        # Both halves of the law's setpoint, as the single-world servo solves
        # them: the rate-capped position AND the slerped orientation. Handing
        # the solver the full goal orientation beside a millimetre of position
        # lead made every per-step solve trade a ninety-degree rotation error
        # against that millimetre, and the wrist rolled 42 degrees along
        # tool_hanging's carry arc where the serial servo drifts 13.
        solver.set_target(world, set_pos, set_quat)

    def _finished(self, world: int, arm: int, program: _Program, snapshot) -> tuple[bool, BaseException | None, Any]:
        """Whether one motion is over, and what to wake its thread with.

        ``(done, error, result)``: a servo completes with the outcome dict the
        single-world servo returns, a joint move with whether it converged
        (the facade's ``move_to_joints_blocking`` answer), a hand move with
        ``None``, and
        nothing here raises for running out of budget -- see the servo branch.
        """
        if program.kind == "gripper":
            if self.step >= program.deadline:
                return True, None, None
            if self.step < program.settle_until:
                return False, None, None
            _command, hand = program.goal
            bank = self.fleet.grippers.get(hand)
            if bank is None:
                return True, None, None
            with reading():
                coords = np.asarray(bank.coords.numpy(), dtype=np.int64)
            measured = snapshot.coords(world)[coords]
            commanded = snapshot.targets(world)[coords]
            # Reaching the commanded width ends the move early -- an open hand,
            # or a close on nothing. A hand that has closed *on something* never
            # reaches it, and that is the case the deadline above is for: the
            # fingers stall against the object and the move ends on the budget
            # rather than on arrival.
            return bool(np.abs(measured - commanded).max() < program.tolerance), None, None

        name = self.arm_names[arm] if arm < len(self.arm_names) else None
        if name is None:
            return True, None, None
        index = self.fleet.scene.embodiment.arms[name]

        if program.kind == "joints":
            if self.step < program.settle_until:
                return False, None, None
            current = snapshot.coords(world)[index.arm_coords]
            goal = program.goal[: len(current)]
            arrived = float(np.linalg.norm(current[: len(goal)] - goal)) < program.tolerance
            # Out of budget is not an error: the single-world blocking move also
            # returns after its step budget without complaining, and a graph that
            # cares checks the resulting pose itself.
            if arrived or self.step >= program.deadline:
                self._release_arm(world, name)
                return True, None, bool(arrived)
            return False, None, None

        if program.kind == "servo":
            # The single-world servo's loop body, one turn per control step,
            # over the published snapshot. The same three exits, in the same
            # order, with the same words: ``reached`` when both the position
            # ball and the orientation tolerance close; ``contact`` when a
            # stall-stopping move has made no progress for ``stall_steps``
            # turns -- observable without ground truth, which is why the
            # contact-seeking descent transfers to a real arm; ``timeout`` at
            # the budget. None of them raises. Running out of budget used to
            # raise here, described as matching the serial servo -- it did the
            # opposite (that servo *returns* a timeout and raises only for a
            # pose it cannot solve), and the raise routed every batched
            # Cartesian move to the graph's abort recovery, whose first act is
            # to open the gripper.
            target, arm_name, _quat = program.goal
            tcp, tcp_quat = tcp_pose(snapshot.body(world), index.ee_body, *self._offsets(arm_name))
            dist = float(np.linalg.norm(tcp - target[:3]))
            ang = (
                rotation_error(tcp_quat, program.requested_quat, program.honours_roll)
                if program.requested_quat is not None
                else 0.0
            )
            # A readback every ``cadence`` steps stands for that many turns, so a
            # stall window and a step count mean the same thing at any cadence.
            program.steps += self.cadence
            outcome = ""
            if dist < program.tolerance and ang < program.rot_tolerance:
                outcome = "reached"
            else:
                if dist < program.best_dist - program.stall_epsilon:
                    program.best_dist = dist
                    program.stalled = 0
                    program.reach_judged_at = 0
                else:
                    program.stalled += self.cadence
                stalled = program.steps >= program.stall_steps and program.stalled >= program.stall_steps
                if (
                    stalled
                    and dist > self.UNREACHABLE_GATE_M
                    and program.stalled - program.reach_judged_at >= program.stall_steps
                ):
                    # Stalled, and still well short: is the goal beyond the arm?
                    # The serial servo learns this from its solver -- three
                    # consecutive solves outside the 30 mm gate raise
                    # ``unreachable`` -- and the graph's pose verb routes on the
                    # word. The batched rig drives from live state and reports
                    # no residual, so the verdict is asked of the same solver
                    # here, once per stall window. Without it an unreachable
                    # hover ran its whole 400-step budget with the arm straining
                    # at its limit -- and the sponge in its hand slipped out
                    # while it did; the serial run had given up after 88.
                    program.reach_judged_at = program.stalled
                    if self._beyond_reach(world, arm_name, program, target, snapshot):
                        self._release_arm(world, arm_name)
                        return (
                            True,
                            RuntimeError(
                                f"servo: pose unreachable after {program.stalled} steps without progress "
                                f"(remaining {dist * 1000:.0f} mm / {np.degrees(ang):.0f} deg)"
                            ),
                            None,
                        )
                if program.stop_on_stall and stalled:
                    outcome = "contact"
                elif self.step >= program.deadline:
                    outcome = "timeout"
            if outcome:
                self._release_arm(world, arm_name)
                return True, None, {"outcome": outcome, "remaining_distance": dist, "steps": int(program.steps)}
        return False, None, None

    def _beyond_reach(self, world: int, name: str, program: _Program, target: np.ndarray, snapshot) -> bool:
        """Whether the goal is outside what the arm can solve, by the serial backend's own test.

        The template solver against the same gate ``IKBackend._solve``
        applies: roll-honouring moves on ``solve_q_full`` gated on both position
        and rotation, roll-free ones on ``solve_best`` over the approach axis
        gated on position. Seeded three ways -- this world's live joints, its
        commanded targets, and the home posture -- and the goal is unreachable
        only if none of them solves. The serial servo re-seeds every turn from
        its own last solution, a well-conditioned chain; a single seed taken
        from an arm jammed against a fixture is not, and judged from it alone a
        wipe stroke the serial run completed was called unreachable 186 mm
        short. A goal any seed solves is not unreachable -- the arm is blocked
        or slow, and the budget decides, as it does serially.
        """
        from robosimstudio.solvers.ik.solver import ROT_TOLERANCE_DEG

        embodiment = self.fleet.scene.embodiment
        solver = getattr(embodiment, "iks", {}).get(name) if embodiment is not None else None
        if solver is None:
            return False
        seeds = [np.asarray(snapshot.coords(world), dtype=np.float64)]
        commanded = snapshot.targets(world)
        if commanded is not None and len(commanded) == len(seeds[0]):
            seeds.append(np.asarray(commanded, dtype=np.float64))
        home = getattr(self.fleet, "_home_q", None)
        if home is not None and len(home) == len(seeds[0]):
            seeds.append(np.asarray(home, dtype=np.float64))
        goal = np.asarray(target[:3], dtype=np.float64)
        quat = program.requested_quat if program.requested_quat is not None else np.asarray(program.goal[2])
        iterations = int(getattr(solver, "iterations", 80) or 80)
        try:
            with reading():
                if program.honours_roll:
                    for seed in seeds:
                        solution = solver.solve_q_full(seed, goal, quat, iterations=iterations)
                        error, rotation = solver.pose_errors(solution, goal, quat)
                        if error <= self.UNREACHABLE_GATE_M and rotation <= np.radians(ROT_TOLERANCE_DEG):
                            return False
                    return True
                approach = quat_to_mat(np.asarray(quat, dtype=np.float64))[:, 2]
                _solution, error = solver.solve_best(seeds, goal, approach, iterations=iterations)
                return bool(error > self.UNREACHABLE_GATE_M)
        except Exception:
            # A solver that cannot answer does not make a goal unreachable; the
            # budget still ends the move, as it always did.
            return False

    def _release_arm(self, world: int, name: str) -> None:
        """Stop everything driving one world's arm, so it simply holds.

        Both drivers, not just the one that finished, and this is the whole
        reason the method exists rather than each branch switching itself off.
        A segment that has run to its end still *writes* -- it clamps at ``q1``
        and keeps stamping it every step -- and the control stage runs the
        segment drivers after the solve. So a finished joint move that stayed
        enabled would overwrite the next Cartesian move's solution on every step,
        for as long as that move lasted: the arm holds perfectly still while the
        servo reports, correctly, that it never arrived.

        Releasing instead is what makes holding free. Nothing writes the arm's
        targets, so they keep the last values written, exactly.
        """
        driver = self.fleet.segments.get(name)
        if driver is not None:
            driver.stop(world)
        rig = self.fleet.ik
        if rig is not None and name in rig.arms:
            rig.arms[name].enable_world(world, False)

    # ------------------------------------------------------------------ hygiene
    def pump(self) -> int:
        """Service marshalled calls between steps."""
        return self.service.pump()

    def retire(self, world: int, reason: str = "the episode ended") -> None:
        """Take a world out of service, waking anything blocked on it.

        What scoring a world terminal does. A node blocked on a motion whose
        world has been retired must not wait out its deadline: the exception is
        what the runtime turns into a node error rather than a hang.
        """
        self._retired.add(int(world))
        self.intents.cancel_world(world, BusTimeout(f"world {world}: {reason}"))
        self.service.drop_world(world)

    def reset(self) -> None:
        """Forget the previous wave. Called after a fleet reset."""
        self.step = 0
        self._retired.clear()
        self.intents.cancel_all(BusTimeout("the wave was reset"))
