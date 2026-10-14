# SPDX-License-Identifier: Apache-2.0
"""What a graph asks the fleet to do, and how it waits for the answer.

The single-world bus marshals *one control step* per round trip: a node
streaming forty waypoints posts forty requests and blocks forty times, each
serviced between two steps. That is the right shape for one world -- the graph
and the simulation take turns -- and the wrong shape for five hundred, where the
step is shared and a world that is mid-decision must not be able to hold it up.

So the unit changes from a step to a **motion**. A node says "move this arm along
these waypoints" once; the fleet executes it on device over however many steps it
takes; the node's thread parks on an event until it completes. Between the two,
the world is being stepped along with everyone else, and the host is doing
nothing per world per step.

Four things follow from that, and they are what this module is:

* **An intent is a plan, not a command.** It carries everything the fleet needs
  to run the motion to completion without asking again, which is what lets the
  host write it once.
* **Slots are per (world, arm, kind).** A bimanual graph drives a support arm
  from one node while another moves the work arm, and the single-world bus
  serialised that only because there was one queue. Submitting to a busy slot
  *supersedes* what was there, which is the non-blocking contract the pinned
  runtime already has for pose moves. A hand is its own slot beside its arm's,
  because on the single-world facade the two are independent writes.
* **Holding is not an intent.** A world with no active intent is not written at
  all, so its targets stay exactly where the last motion left them. The
  single-world path has to re-issue the last commanded targets every step,
  because the rollout stamps an action whether or not the graph wanted one; here
  there is nothing to re-issue.
* **Errors travel on the handle.** A servo that cannot reach its goal completes
  the handle with an exception, which is re-raised in the node's own thread and
  becomes a node error the graph's own routing can catch -- the same path a
  failed tool call takes today.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "GripperMove",
    "Intent",
    "IntentHandle",
    "IntentTable",
    "JointTrajectory",
    "MoveToJoints",
    "ServoToPose",
    "Superseded",
    "slot_kind",
]


class Superseded(RuntimeError):
    """A motion was replaced by a newer one for the same arm."""


@dataclass(frozen=True)
class MoveToJoints:
    """Drive one arm to a joint configuration and settle there.

    Args:
        arm: Which arm, by index into the robot's arms.
        target: The configuration to reach, one value per arm joint.
        duration_steps: Control steps to take getting there.
        tolerance: How close counts as arrived [rad or m].
        max_steps: Give up after this many steps. Zero fires and forgets -- the
            motion is installed and the caller does not wait, which is what a
            non-blocking pose command wants.
    """

    arm: int
    target: np.ndarray
    duration_steps: int = 60
    tolerance: float = 0.01
    max_steps: int = 1000


@dataclass(frozen=True)
class JointTrajectory:
    """Walk one arm through a sequence of configurations.

    Args:
        arm: Which arm.
        waypoints: ``[k, dof]``, walked one at a time with linear interpolation
            between consecutive ones.
        steps_per_waypoint: Control steps to spend reaching each waypoint.
            **Two is what the single-world facade actually does**, whatever its
            docstring says about one per step: ``stream_joint_trajectory`` steps
            the environment once inside each waypoint's bus call, and the
            rollout loop steps it once more with ``hold_action`` before the next
            call is serviced -- 86 control steps for every 40-waypoint line in
            every serial tape (2.15 per waypoint, the rest is the settle). The
            fleet has no hold step, so at one waypoint per step it walked the
            same line in 43 steps: a planned 12 mm/waypoint carry became a
            1 m/s carry, and a sponge that serial carries at 0.5 m/s left the
            pads in every world. Two here is the serial cadence, interpolated
            rather than step-and-hold, so a graph tuned serially moves at the
            speed it was tuned at.
        settle_tolerance: How close to the last waypoint counts as settled.
        settle_max_steps: How long to wait for that after the last waypoint.
    """

    arm: int
    waypoints: np.ndarray
    steps_per_waypoint: int = 2
    settle_tolerance: float = 0.01
    settle_max_steps: int = 120


@dataclass(frozen=True)
class ServoToPose:
    """Servo one arm's tool-centre point to a world pose.

    The batched replacement for the host servo loop, which solves inverse
    kinematics on the host once per control step per world. Here the goal is
    written once and the fleet's batched solver drives every servoing world from
    live state each step.

    Args:
        arm: Which arm.
        position: Target tool-centre position [m].
        quat_xyzw: Target orientation.
        pos_tolerance: Arrival radius [m].
        rot_tolerance: Arrival angle [rad]. Ignored when ``honours_roll`` is off.
        honours_roll: Whether the orientation has to be reached, or only the
            approach direction.
        max_steps: Give up after this many steps.
        stop_on_stall: End the move as ``contact`` when the tool stops making
            progress towards the goal, exactly as the single-world servo's
            ``stop_on_stall`` does: no fixture identity, no force oracle, no
            simulator contact state -- the same signal a position-controlled
            real arm exposes. What ``robot.move_cartesian_until_contact`` asks
            for, and until this field existed it asked a fleet for it and got a
            ``TypeError``.
        stall_steps: Turns without progress that count as a stall, and the
            minimum number of turns before one may be declared.
        stall_epsilon: How much closer [m] counts as progress.
        speed_scale: Cartesian speed limit, in (0, 1]. 1.0 is the law's own
            ceiling.

            **A real bound, not a preference.** It scales the setpoint's cruise
            rate and its command lead by the same factor, which is what the
            single-world servo does with it -- and tool hanging's graph carries
            0.40 to 0.60 on the scissors and screwdriver carries, where a
            full-speed reorientation swings the tool out of the jaws. A wave that
            ignored it would run those at full speed and report the result as
            the graph's.
    """

    arm: int
    position: np.ndarray
    quat_xyzw: np.ndarray
    pos_tolerance: float = 0.004
    rot_tolerance: float = 0.05
    honours_roll: bool = False
    max_steps: int = 1000
    speed_scale: float = 1.0
    """Ceiling on the setpoint law's output, in (0, 1]: the single-world servo's Cartesian speed limit."""
    stop_on_stall: bool = False
    stall_steps: int = 12
    stall_epsilon: float = 0.0002


@dataclass(frozen=True)
class GripperMove:
    """Open or close one hand, optionally over a ramp.

    Args:
        arm: Which arm's hand.
        command: 0 open to 1 closed, in the environment's convention.
        ramp_steps: Blend over this many steps. Zero commands it outright.
        settle_steps: Steps to wait after arriving, for the grasp to take.
    """

    arm: int
    command: float
    ramp_steps: int = 0
    settle_steps: int = 0


Intent = MoveToJoints | JointTrajectory | ServoToPose | GripperMove


def slot_kind(intent: Intent) -> str:
    """Which of an arm's two slots a motion occupies: ``"arm"`` or ``"hand"``.

    A hand command and an arm motion are independent writes on the single-world
    facade, and they are independent here: keyed together, a gripper move
    *superseded* the arm motion in flight -- and the install released the arm's
    drivers -- so a graph re-gripping during a carry silently cancelled the
    carry.
    """
    return "hand" if isinstance(intent, GripperMove) else "arm"


class IntentHandle:
    """What a node's thread waits on while its motion runs.

    One handle per submitted intent. The fleet completes it -- with an error, or
    as superseded -- and the waiting thread wakes and either returns or raises.
    """

    __slots__ = ("_done", "error", "ident", "result", "steps", "superseded")

    def __init__(self) -> None:
        self._done = threading.Event()
        self.error: BaseException | None = None
        self.result: Any = None
        """What the motion ended with, for a caller that reads one.

        A servo completes with the outcome dict the single-world servo returns
        (``outcome`` / ``remaining_distance`` / ``steps``); a joint move with
        whether it converged; a hand move with ``None``. Carried on the handle rather than raised, because "did
        not arrive" is a *result* the connector routes on, not an error."""
        self.superseded = False
        self.steps = 0

        self.ident = threading.get_ident()
        """Whoever submitted this, which is whoever will wait on it.

        Recorded so the fleet unparks *that* thread. A world can have more than
        one thread in a fleet wait -- see :class:`~.barrier.ParkTable` -- and
        unparking the wrong one releases a park its owner is still inside."""

    def complete(
        self,
        *,
        error: BaseException | None = None,
        superseded: bool = False,
        steps: int = 0,
        result: Any = None,
    ) -> None:
        """Called by the fleet when the motion is over. Idempotent."""
        if self._done.is_set():
            return
        self.error = error
        self.superseded = bool(superseded)
        self.steps = int(steps)
        self.result = result
        self._done.set()

    def wait(self, timeout: float | None = None) -> Any:
        """Block until the motion finishes, re-raising whatever ended it.

        Returns :attr:`result`. A superseded motion returns rather than
        raising: being replaced is what the caller asked for when it issued the
        replacement, and the shipped non-blocking pose command relies on it not
        being an error.
        """
        if not self._done.wait(timeout):
            raise TimeoutError("the motion did not complete within the deadline")
        if self.error is not None:
            raise self.error
        return self.result

    @property
    def done(self) -> bool:
        return self._done.is_set()


@dataclass
class _Slot:
    """One (world, arm) pair's pending and running motion."""

    intent: Intent | None = None
    handle: IntentHandle | None = None
    started: int = 0
    program: Any = None


@dataclass
class IntentTable:
    """Every world's outstanding motions, by arm.

    Submitted from graph threads, drained by whatever is stepping the fleet.
    The lock is held only long enough to swap a slot -- never across anything
    that touches the device, which is the mistake the single-world bus's
    docstring warns about: a lock held across a launch stalls the simulation for
    as long as a node takes to think.

    Args:
        num_worlds: Worlds in the fleet.
        num_arms: Arms per robot.
    """

    num_worlds: int
    num_arms: int = 1
    parks: Any = None
    """The fleet's :class:`~robosimstudio_policy_agent.fleet.barrier.ParkTable`, or ``None``.

    Held so a completion can take its world out of the parked set *before* the
    handle's event is set. A thread whose event has been set is runnable, and a
    fleet that read it as parked would step a world whose script was about to
    run -- which is the whole thing :mod:`.barrier` exists to stop. ``None`` is
    the pre-barrier behaviour and is what the unit tests construct.
    """
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _pending: dict[tuple[int, int, str], _Slot] = field(default_factory=dict, repr=False)
    _active: dict[tuple[int, int, str], _Slot] = field(default_factory=dict, repr=False)

    def _wake(self, world: int, handle: IntentHandle | None, **completion: Any) -> None:
        """Complete one handle, unparking its world first.

        Every completion in this class goes through here, so the ordering is
        stated once rather than remembered five times.
        """
        if handle is None:
            return
        if self.parks is not None:
            self.parks.unpark(int(world), getattr(handle, "ident", None))
        handle.complete(**completion)

    def submit(self, world: int, arm: int, intent: Intent) -> IntentHandle:
        """Queue a motion for one arm of one world, superseding what it had.

        Returns immediately: the caller decides whether to wait on the handle,
        which is what separates a blocking tool call from a fire-and-forget one.
        """
        key = (int(world), int(arm), slot_kind(intent))
        if not 0 <= key[0] < self.num_worlds:
            raise IndexError(f"world {world} is outside a fleet of {self.num_worlds}")
        handle = IntentHandle()
        with self._lock:
            displaced = self._pending.pop(key, None)
            self._pending[key] = _Slot(intent=intent, handle=handle)
        # Outside the lock: completing a handle wakes another thread, and waking
        # it while holding the lock invites it straight back into submit().
        if displaced is not None:
            self._wake(key[0], displaced.handle, superseded=True)
        return handle

    def drain(self, step: int = 0) -> list[tuple[int, int, Intent, IntentHandle]]:
        """Start everything submitted since the last drain. Fleet thread.

        Draining *is* starting: a motion moves from pending to running under one
        lock, and whatever it displaces is woken as superseded. The alternative
        -- hand the caller a motion and trust it to register the motion
        afterwards -- leaves a window in which the table has forgotten a handle
        that nothing can complete, so a world cancelled during that window
        strands the thread waiting on it until its deadline.

        The caller still has to *install* each motion in the fleet's device
        arrays, and attaches whatever bookkeeping it needs through
        :meth:`attach`. Failing to do that leaves a motion that never finishes,
        which the fleet's own step budget ends -- a bounded failure rather than a
        lost wake-up.
        """
        with self._lock:
            taken, self._pending = self._pending, {}
            displaced = []
            for key, slot in taken.items():
                previous = self._active.get(key)
                if previous is not None and previous.handle is not None and previous.handle is not slot.handle:
                    displaced.append((key[0], previous.handle))
                self._active[key] = _Slot(intent=slot.intent, handle=slot.handle, started=int(step), program=None)
        # Outside the lock, for the reason submit() gives: waking a thread while
        # holding it invites that thread straight back in.
        for world, handle in displaced:
            self._wake(world, handle, superseded=True)
        return [(key[0], key[1], slot.intent, slot.handle) for key, slot in taken.items() if slot.intent is not None]

    def attach(self, world: int, arm: int, program: Any, *, kind: str = "arm") -> None:
        """Record what the fleet installed for a running motion."""
        with self._lock:
            slot = self._active.get((int(world), int(arm), kind))
            if slot is not None:
                slot.program = program

    def finish(
        self,
        world: int,
        arm: int,
        *,
        error: BaseException | None = None,
        step: int = 0,
        result: Any = None,
        kind: str = "arm",
    ) -> None:
        """Retire a running motion and wake whoever was waiting on it with ``result``."""
        key = (int(world), int(arm), kind)
        with self._lock:
            slot = self._active.pop(key, None)
        if slot is not None:
            self._wake(key[0], slot.handle, error=error, result=result, steps=max(0, int(step) - slot.started))

    def active(self) -> list[tuple[int, int, Intent, Any, int]]:
        """Every running motion, as ``(world, arm, intent, program, started)``."""
        with self._lock:
            return [
                (key[0], key[1], slot.intent, slot.program, slot.started)
                for key, slot in self._active.items()
                if slot.intent is not None
            ]

    def running(self, world: int, arm: int, kind: str = "arm") -> Intent | None:
        with self._lock:
            slot = self._active.get((int(world), int(arm), kind))
        return slot.intent if slot is not None else None

    def busy_worlds(self) -> np.ndarray:
        """Per world, whether any of its arms has a motion running."""
        mask = np.zeros(self.num_worlds, dtype=bool)
        with self._lock:
            for world, _arm, _kind in self._active:
                mask[world] = True
        return mask

    def cancel_world(self, world: int, error: BaseException) -> None:
        """End every motion of one world, waking its threads with ``error``.

        What an episode ending does: a node blocked on a motion whose world has
        been retired must not wait out its deadline, and the exception is what
        the graph runtime turns into a node error rather than a hang.
        """
        world = int(world)
        with self._lock:
            slots = [
                table.pop(key)
                for table in (self._active, self._pending)
                for key in [k for k in list(table) if k[0] == world]
            ]
        for slot in slots:
            self._wake(world, slot.handle, error=error)

    def cancel_all(self, error: BaseException) -> None:
        """End every motion in the fleet. What closing the run does."""
        with self._lock:
            slots = [(key[0], slot) for key, slot in self._active.items()]
            slots += [(key[0], slot) for key, slot in self._pending.items()]
            self._active.clear()
            self._pending.clear()
        for world, slot in slots:
            self._wake(world, slot.handle, error=error)
