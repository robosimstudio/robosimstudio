# SPDX-License-Identifier: Apache-2.0
"""One world, presented to the pinned runtime as if it were the whole simulation.

The runtime's ``Connector`` drives a duck-typed environment: it looks for
``get_observation``, ``_step_once``, ``move_to_joints_blocking``,
``stream_joint_trajectory`` and ``_set_gripper`` by name, and adapts what it does
to whichever it finds. :class:`~robosimstudio_policy_agent.facade.GapEnvFacade`
presents that surface over a single-world environment; this presents the *same*
surface over one world of a fleet.

That sameness is the point, and it is a constraint rather than a convenience: the
graph, its node scripts and its tool interface are the artifact that also runs on
a real robot, so a batched run must not add anything for them to know. The world
index is bound here, in the constructor, and nothing above this line mentions it.

What changes underneath is only *how a motion is executed*. The single-world
facade steps the environment itself, once per waypoint, through the bus -- it can,
because it owns the only world. Here a motion becomes an intent: submitted once,
run on device by the fleet over as many steps as it takes, and awaited on a
handle. The node's thread blocks in exactly the same place it did before; what it
no longer does is wake up once per control step to issue the next waypoint.

Three conversions still happen here and nowhere else, unchanged from the
single-world facade because they are properties of the runtime's conventions and
not of how many worlds there are:

* Quaternions become wxyz. The runtime is scalar-first; Newton is xyzw.
* The gripper becomes an *open* fraction. The runtime's 1.0 is fully open; this
  stack's gripper command is 1.0 closed. Getting it backwards produces a robot
  that opens to grasp, which reads as a planning bug for a long time.
* Joint vectors mean joint targets, so every command selects the joint
  controller rather than whatever the scene's default mode happens to be.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.core.bus import BusTimeout

from .barrier import park_if
from .intents import GripperMove, IntentHandle, IntentTable, JointTrajectory, MoveToJoints, ServoToPose
from .obs_format import RENDER, format_runtime_obs

__all__ = ["WorldFacade"]


class WorldFacade:
    """The runtime's env surface, bound to one world of a fleet.

    Args:
        world: Which world. The only place it appears in the whole stack above
            the fleet.
        fleet: The batched environment, for its handles, arms and scene facts.
        bus: This world's view of the simulation.
        intents: Where motions are submitted.
        arm_names: Arms in the order a runtime ``arm_id`` counts them -- work
            first. Passed in rather than derived, because role order is a fact
            about the *pairing* and the fleet resolves it once for every world.
        max_steps: Step budget the runtime reads. 0 uses the task's own.
    """

    LOST_WAKEUP_S = 3600.0
    """Ceiling on waiting for a motion [s]. Not a motion budget -- a bug detector.

    **A motion is bounded in steps, and the steps are already bounded.**
    :class:`~robosimstudio_policy_agent.fleet.stepper.StepService` completes a motion the
    moment the fleet's step count passes the budget the intent carried; the
    barrier's watchdog retires a world that stops the fleet; and the wave's own
    step limit ends the episode. Between them there is no way for a motion to
    run forever that a clock could catch and those three could not.

    What a *wall* clock would catch is something else entirely, and it catches it
    wrongly. Under the barrier a wave's seconds include every hold for every
    other world, so the same motion takes longer in wall time the wider the wave
    is and the further its worlds have diverged -- neither of which is a fact
    about the motion. Measured on ``cable_route/port_easy`` at two worlds: every
    world failed its node on the 120 s default, and the same two worlds ran to
    success in 3751 and 3747 steps once the clock stopped being the bound. A
    deadline that fires on a healthy run is worse than no deadline, because it
    produces a failure with a plausible-looking cause.

    So this is an hour: long enough that nothing healthy reaches it, short enough
    that a genuinely lost wake-up surfaces as a timeout rather than as a process
    that has to be killed. If it ever fires, the bug is that something completed
    a motion's slot without completing its handle.
    """

    JOINT_MODE = "joint_position"
    """The mode a joint vector means something in, named for the same reason the
    single-world facade names it: the graph's IK solutions *are* joint targets,
    so what the scene's default mode happens to be must stop mattering."""

    def __init__(
        self,
        world: int,
        *,
        fleet,
        bus,
        intents: IntentTable,
        arm_names: list[str],
        max_steps: int = 0,
    ) -> None:
        self.world = int(world)
        self.fleet = fleet
        self.bus = bus
        self.intents = intents
        self.arm_names = list(arm_names)
        self.max_steps = int(max_steps or getattr(fleet, "max_steps", 0))
        # The runtime's open fraction, per arm. Per arm because a hand is: one
        # scalar for the robot means a bimanual graph pinching with one hand
        # drives the other to the same width.
        self._gripper_fraction: dict[int, float] = {}
        self._closed = False

    # ---------------------------------------------------------------- helpers
    def _arm(self, arm_id: int) -> int:
        """Clamp a runtime ``arm_id`` to a real arm, as the runtime expects.

        Clamping rather than raising is the runtime's own convention: its
        single-arm tools pass 0 unconditionally, and a bimanual graph that
        over-counts should drive the last arm rather than end the episode.
        """
        if not self.arm_names:
            raise RuntimeError("this pairing has no arms")
        return max(0, min(int(arm_id), len(self.arm_names) - 1))

    def _submit(self, intent, *, wait: bool) -> IntentHandle:
        """Hand one motion to the fleet, and optionally block until it is done.

        The park brackets the *submit* as well as the wait. Submitting first and
        parking second leaves a window in which the fleet reads this world as
        running while its thread is in fact about to block -- and a motion that
        completes inside that window is a step the batched run takes and the
        serial run does not. See :mod:`.barrier`.

        A fire-and-forget submit parks nothing, which is right: the caller goes
        straight back to running host code, and that is exactly what holds the
        barrier.
        """
        if self._closed:
            # A world whose episode is over drives nothing. Without this a graph
            # thread that outlived its wave -- joined on a timeout while inside
            # an uninterruptible tool call -- could post a motion into the next
            # wave's table for its world index and move the next episode's arm.
            raise BusTimeout(f"world {self.world}: the episode ended")
        if not wait:
            return self.intents.submit(self.world, intent.arm, intent)
        with park_if(getattr(self.bus, "parks", None), self.world):
            handle = self.intents.submit(self.world, intent.arm, intent)
            # :attr:`LOST_WAKEUP_S`, not the bus timeout. A marshalled call that
            # is never pumped is a hang a clock is the only witness to; a motion
            # is not, because its own budget is in steps and the fleet enforces
            # it. See the attribute for what a wall clock got wrong here.
            handle.wait(self.LOST_WAKEUP_S)
        return handle

    # ------------------------------------------------------------ observation
    def get_observation(self) -> dict[str, Any]:
        """The runtime's observation dict: this world's state, and a fresh look.

        The state half is *read* rather than gathered -- the fleet publishes one
        batched snapshot per cadence tick and this slices it -- so a node reading
        three poses costs three dictionary lookups instead of three device
        synchronizes, which is what makes the runtime's own observation poller
        harmless at five hundred worlds.

        The picture half cannot be read, because there is nothing to read: a rig
        that photographed every world every step whether or not anybody looked
        would be the most expensive thing in a wave, by a wide margin. So a look
        is *asked for*, through the rendezvous, and every world looking at the
        same moment -- which is every world of a wave running the same graph --
        is photographed in one pass of the rig.

        A scene with no rig has no group to join and takes the published frames,
        which are empty. That is the privileged case and it should cost nothing.
        """
        frames = self.bus.batched(RENDER, None) if self.bus.batchable(RENDER) else None
        return format_runtime_obs(
            self.bus.snapshot,
            arm_count=len(self.arm_names),
            frames=frames,
            # The rig, so each frame arrives with the calibration that makes it
            # an observation rather than a picture. Without it a perception node
            # locates something in pixels and has nothing to unproject it with.
            cameras=self.fleet.cameras,
        )

    def get_proprioception(self) -> dict[str, Any]:
        """Encoders and tool pose, without a look.

        The connector's motion feedback -- ``get_ee_pose``,
        ``get_gripper_fraction``, the joint read behind every IK seed -- calls
        this by name when it exists and :meth:`get_observation` when it does
        not. It did not exist here, so under a rig every one of those reads
        joined the render rendezvous: a full pass of every camera over every
        world, plus a barrier round, to learn where a hand was. The seed graph
        for surface_prep reads its hand thirteen times in one pick.

        The published snapshot already holds everything this needs and is
        already on the host; no rendezvous, no render, no device touch.
        """
        return format_runtime_obs(self.bus.snapshot, arm_count=len(self.arm_names), frames={})

    # ------------------------------------------------------------------- step
    def _step_once(self) -> None:
        """Let one control period pass, changing nothing but time.

        The runtime's blocking loops call this between waypoints. Nothing needs
        to be written to hold position: a world with no active motion is not
        written at all, so its targets stay exactly where the last motion left
        them. The single-world facade has to re-issue the held command every step
        because the rollout stamps an action whether or not the graph wanted one.
        """
        self.bus.wait_for_step(after=self.bus.snapshot.step)

    # ---------------------------------------------------------------- motions
    def move_to_joints_blocking(
        self,
        target: np.ndarray,
        *,
        tolerance: float = 0.01,
        max_steps: int = 120,
        arm_id: int = 0,
    ) -> bool:
        """Drive one arm to a joint target and wait for it to get there.

        Returns whether it converged inside the budget. ``max_steps == 0`` is
        the runtime's fire-and-forget contract: submit the motion and return
        ``False`` without waiting -- not known to have arrived.
        """
        goal = np.asarray(target, dtype=np.float32).reshape(-1)
        handle = self._submit(
            MoveToJoints(
                arm=self._arm(arm_id),
                target=goal,
                tolerance=float(tolerance),
                max_steps=int(max_steps),
            ),
            wait=max_steps != 0,
        )
        return max_steps != 0 and bool(handle.result)

    def stream_joint_trajectory(
        self,
        waypoints: list[list[float]],
        *,
        settle_tolerance: float = 0.01,
        settle_max_steps: int = 60,
        arm_id: int = 0,
    ) -> bool:
        """Feed a dense joint trajectory as one motion; whether the arm settled on its last waypoint.

        The runtime looks for this method by name and, finding it, treats a
        trajectory as *the control sequence* rather than a list of poses to
        converge on one at a time -- which is what keeps a smooth path from
        becoming a series of accelerate-and-stop moves. That reasoning is
        unchanged here; what changes is that the whole path is submitted once
        instead of one waypoint per bus round trip, so a forty-waypoint plan
        costs one block rather than forty.
        """
        path = np.asarray(waypoints, dtype=np.float32)
        if path.size == 0:
            return True
        handle = self._submit(
            JointTrajectory(
                arm=self._arm(arm_id),
                waypoints=path.reshape(len(waypoints), -1),
                # One waypoint per control step: the runtime's contract, stated
                # here rather than left to a default.
                steps_per_waypoint=2,  # the serial facade's measured cadence -- see JointTrajectory
                settle_tolerance=float(settle_tolerance),
                settle_max_steps=int(settle_max_steps),
            ),
            wait=True,
        )
        return bool(handle.result)

    def apply_joint_action(self, joints: np.ndarray, arm_id: int = 0) -> None:
        """Command one arm's joint targets without waiting.

        Kept because suite tools call it directly. It is the fire-and-forget form
        of :meth:`move_to_joints_blocking`, which is what it always was -- the
        single-world version stepped the environment as a side effect only
        because the caller was on the owner thread.
        """
        self.move_to_joints_blocking(joints, max_steps=0, arm_id=arm_id)

    def servo_to_pose(
        self,
        position,
        quat_xyzw,
        *,
        arm_id: int = 0,
        max_steps: int = 0,
        pos_tolerance: float = 0.004,
        rot_tolerance: float = 0.05,
        honours_roll: bool = False,
        speed_scale: float = 1.0,
        stop_on_stall: bool = False,
        stall_steps: int = 12,
        stall_epsilon: float = 0.0002,
    ) -> dict[str, Any]:
        """Servo this world's tool-centre point to a world pose, and wait.

        The batched replacement for the per-step host servo loop: the goal is
        written once, and the fleet's own solver drives every servoing world from
        live state each step. Returns what the single-world servo returns --
        ``{"outcome": "reached" | "contact" | "timeout", "remaining_distance",
        "steps"}`` -- and, like it, does not raise for a move that ran out of
        budget: "did not arrive" is a result the connector's pose verbs route
        on, and raising it sent every batched Cartesian move that timed out to
        the graph's abort recovery.
        """
        handle = self._submit(
            ServoToPose(
                arm=self._arm(arm_id),
                position=np.asarray(position, dtype=np.float64),
                quat_xyzw=np.asarray(quat_xyzw, dtype=np.float64),
                pos_tolerance=float(pos_tolerance),
                rot_tolerance=float(rot_tolerance),
                honours_roll=bool(honours_roll),
                max_steps=int(max_steps) or 1000,
                stop_on_stall=bool(stop_on_stall),
                stall_steps=int(stall_steps),
                stall_epsilon=float(stall_epsilon),
                speed_scale=float(speed_scale),
            ),
            wait=True,
        )
        return dict(handle.result or {"outcome": "superseded", "remaining_distance": -1.0, "steps": handle.steps})

    # ---------------------------------------------------------------- gripper
    def _set_gripper(self, fraction: float, arm_id: int = 0) -> None:
        """Set one arm's open fraction. 1.0 is open, which is the runtime's convention.

        Waited, because on this fleet a hand move is ONE device ramp
        (:meth:`set_grip` blends on device; the runtime's own open/close is a
        single motion too), so the wait is one motion's worth of steps and the
        world parks for it like any other fleet wait. A non-waiting write here
        left the intent queued and the world unparked, and the wave stood still.
        """
        arm = self._arm(arm_id)
        self._gripper_fraction[arm] = float(np.clip(fraction, 0.0, 1.0))
        self._submit(GripperMove(arm=arm, command=1.0 - self._gripper_fraction[arm]), wait=True)

    def set_grip(self, fraction: float, *, ramp_steps: int = 24, settle_steps: int = 0, arm_id: int = 0) -> None:
        """Close a hand over a ramp rather than at once.

        One motion, not a loop of twenty-four commands each waiting a step. The
        single-world path spends a step per blend increment because that is the
        only way it can interpolate; here the ramp is a property of the motion
        and the device does the blending.
        """
        arm = self._arm(arm_id)
        self._gripper_fraction[arm] = float(np.clip(fraction, 0.0, 1.0))
        self._submit(
            GripperMove(
                arm=arm,
                command=1.0 - self._gripper_fraction[arm],
                ramp_steps=int(ramp_steps),
                settle_steps=int(settle_steps),
            ),
            wait=True,
        )

    # ---------------------------------------------------------------- scoring
    def compute_reward(self) -> float:
        return float(self.bus.snapshot.info.get("reward", 0.0))

    def task_completed(self) -> bool:
        """What the *evaluator* said. A graph may read it; it may not write it."""
        return bool(self.bus.snapshot.info.get("success", False))

    @property
    def _current_done(self) -> bool:
        return bool(self.bus.snapshot.info.get("done", False)) or self._closed

    @property
    def _sim_step_count(self) -> int:
        return int(self.bus.snapshot.step)

    # -------------------------------------------------------------- lifecycle
    # No ``reset``: the runtime probes for one with ``hasattr`` and would drive
    # it mid-wave, which would re-pose every world. The wave scheduler owns
    # resets, and its absence here is what says so.

    def close(self) -> None:
        """Stop serving this world. Whatever it had pending is released."""
        self._closed = True
        self.bus.close()
