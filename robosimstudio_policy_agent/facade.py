# SPDX-License-Identifier: Apache-2.0
"""Presenting a toolkit-layer :class:`~robosimstudio_bench.envs.env.Env` as the env surface
the pinned GaP runtime expects.

The runtime's ``Connector`` was written against a specific duck-typed
environment: ``get_observation()`` returning a dict of ``robot_joint_pos_<i>``
and ``robot_cartesian_pos_<i>`` plus per-camera blocks, ``_step_once()``,
``compute_reward()``, a step counter and a step budget. Rather than change the
runtime -- it is pinned, and the whole point of reusing it is that it is the
same code the real evaluations run -- this presents that surface.

Two conversions happen here and nowhere else:

* **Quaternions become wxyz.** GaP is scalar-first throughout; Newton is
  xyzw. Every pose crossing this boundary is converted, so nothing downstream
  has to remember which side it is on.
* **The gripper becomes an open fraction.** GaP's 1.0 is fully *open*;
  :class:`~robosimstudio.solvers.Gripper`'s 1.0 is fully *closed*. Getting
  this backwards produces a robot that opens to grasp, which looks like a
  planning bug for a long time before anyone checks the convention.

Every simulator touch goes through the :class:`~robosimstudio.core.bus.SimBus`,
because the runtime calls these from node worker threads.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from robosimstudio.arms import arm_name, arm_order, arm_slot
from robosimstudio.core.bus import SimBus, SimSnapshot
from robosimstudio.core.math import quat_xyzw_to_wxyz
from robosimstudio_bench.envs.env import Env

__all__ = ["EnvFacade", "GapEnvFacade"]


@runtime_checkable
class EnvFacade(Protocol):
    """The env surface the GaP runtime drives, whichever world count it is on.

    Two things implement this: :class:`GapEnvFacade` over a single ``Env``, and
    ``WorldFacade`` over one world of a fleet. Every method below has the *same
    signature* in both -- which is what makes this a contract rather than a
    resemblance, and it is why the runtime can be handed either without knowing
    which.

    **What is outside it is as load-bearing as what is in it.** ``reset`` and
    ``publish_step`` belong to the single-world facade because there the facade
    drives the episode; a fleet's worlds are reset in waves by the driver, so
    its facade has no business offering either. ``servo_to_pose`` and
    ``set_grip`` are the fleet's, reaching segment drivers a single ``Env``
    does not have. A runtime that reached for any of the four would be assuming
    a world count it is supposed not to know.
    """

    JOINT_MODE: str

    def get_observation(self) -> dict[str, Any]:
        """This world's observation, in :attr:`JOINT_MODE`'s layout."""
        ...

    def apply_joint_action(self, joints: np.ndarray, arm_id: int = 0) -> None:
        """Write one arm's joint target."""
        ...

    def stream_joint_trajectory(
        self,
        waypoints: list[list[float]],
        *,
        settle_tolerance: float = 0.01,
        settle_max_steps: int = 60,
        arm_id: int = 0,
    ) -> bool:
        """Feed waypoints one per control step. The rate is the cell's."""
        ...

    def move_to_joints_blocking(
        self,
        target: np.ndarray,
        *,
        tolerance: float = 0.01,
        max_steps: int = 120,
        arm_id: int = 0,
    ) -> bool:
        """Drive to a joint target and wait for it to settle."""
        ...

    def compute_reward(self) -> float:
        """The task's reward as of the last step."""
        ...

    def task_completed(self) -> bool:
        """Whether the goal is satisfied."""
        ...

    def close(self) -> None:
        """Release what the facade opened. Idempotent."""
        ...


class GapEnvFacade:
    """The env surface the GaP ``Connector`` drives, over a toolkit-layer ``Env``.

    Args:
        env: The environment.
        bus: Marshals access onto the owner thread.
        max_steps: Step budget the runtime reads. 0 uses the task's own.
    """

    def __init__(self, env: Env, bus: SimBus, *, max_steps: int = 0) -> None:
        self.env = env
        self.bus = bus
        self.max_steps = int(max_steps or env.max_steps)
        # Per arm, because a hand is. Keyed by runtime arm_id; GaP's 1.0 is open.
        self._gripper_fraction: dict[int, float] = {}
        self._current_done = False
        self._sim_step_count = 0
        self._last_reward = 0.0
        # The last *commanded* joint target per arm. Holding position means
        # re-issuing this, not re-reading the encoder -- see _step_holding.
        self._held: dict[int, np.ndarray] = {}
        # The last published step's proprioception, so a blocking move's error
        # check does not read the embodiment back a second time.
        self._last_proprio: dict[str, Any] = {}

    # ---------------------------------------------------------------- helpers
    @property
    def arm_names(self) -> list[str]:
        """Arms in the order a runtime ``arm_id`` counts them: **work first**.

        Not declaration order. A graph saying ``arm_id=0`` means the hand that
        does the task, and on a robot that declares its idle hand first those
        are different arms -- see :mod:`robosimstudio.arms`.

        This is emphatically *not* the action-vector order. Every write into an
        action vector goes through :meth:`_slot`.
        """
        return list(arm_order(self.env))

    def _arm(self, arm_id: int) -> str:
        if not self.arm_names:
            raise RuntimeError(f"task {self.env.task.full_name!r} has no arms")
        return arm_name(self.env, arm_id)

    def _slot(self, name: str) -> int:
        """Where this arm's slice starts in an action vector, in units of ``per_arm``."""
        return arm_slot(self.env, name)

    # ------------------------------------------------------------ observation
    def get_observation(self) -> dict[str, Any]:
        """The runtime's observation dict, gathered on the owner thread."""
        return self.bus.call(self._observation, refresh_cameras=True)

    def get_proprioception(self) -> dict[str, Any]:
        """Current encoders and robot FK, without triggering camera renders."""
        return self.bus.call(self._proprioception)

    def _proprioception(self, obs=None) -> dict[str, Any]:
        # ``env`` bound locally, as every other reader in this module does: the
        # fleet's world-view contract test scans for ``self.env.<name>`` across
        # the graph package, and ``observe`` is not one of the names it exempts.
        env = self.env
        if obs is None:
            obs = env.observe()
        out: dict[str, Any] = {}
        proprio = obs.get("proprio") or {}
        for arm_id, name in enumerate(self.arm_names):
            joints = np.asarray(proprio.get(f"{name}/joint_q", np.zeros(0)), dtype=np.float64)
            grip_closed = float(proprio.get(f"{name}/gripper", 0.0))
            # GaP's fraction is 1.0 open; the Gripper's command is 1.0 closed.
            out[f"robot_joint_pos_{arm_id}"] = np.append(joints, 1.0 - grip_closed)
            pose = np.asarray(proprio.get(f"{name}/ee_pose", np.zeros(7)), dtype=np.float64)
            out[f"robot_cartesian_pos_{arm_id}"] = np.concatenate([pose[:3], quat_xyzw_to_wxyz(pose[3:7])])

        return out

    def _observation(self, *, refresh_cameras: bool = False, obs=None, body_q=None) -> dict[str, Any]:
        env = self.env
        # A headless evaluation may suppress per-step rendering. Perception
        # still receives pixels and camera poses from the same current state.
        if refresh_cameras and not env.obs_render_enabled and env.renderer is not None:
            env.frames = env.renderer.render(env.state)
            obs = None
        # A stepper passes the obs ``env.step`` already gathered: observing is a
        # device readback, and twice a step is once too many.
        if obs is None:
            obs = env.observe()
        out = self._proprioception(obs)

        for camera_name, frame in (obs.get("cameras") or {}).items():
            spec = next((c for c in env.cameras if c.name == camera_name), None)
            block: dict[str, Any] = {"images": {"rgb": frame.get("rgb")}}
            if "depth" in frame:
                # Under ``images``, because that is the key the runtime's own
                # ``Connector._build_camera_frame`` reads. ``depth_data`` is the
                # *wire* convention -- what the msgpack bridge sends and
                # ``gap.envs.franka_real_env`` unpacks -- and putting it only
                # there is silent: the CameraFrame arrives with rgb, intrinsics
                # and a pose, and no depth. Every perception node then dies
                # *after* GroundingDINO, the VLM tournament and SAM3 have all
                # run and succeeded, which is a diagnosis several tool calls and
                # one VLM bill away from the cause. Both keys are set, so a block
                # is readable by either convention.
                block["images"]["depth"] = frame["depth"]
                block["depth_data"] = frame["depth"]
            if spec is not None:
                block["intrinsics"] = spec.intrinsics
                if body_q is None:
                    body_q = env.state.body_q.numpy()
                world = spec.pose_on_body(body_q)
                from robosimstudio.core.math import xform_from_matrix

                xform = xform_from_matrix(world)
                block["pose"] = np.concatenate([xform[:3], quat_xyzw_to_wxyz(xform[3:7])])
            out[camera_name] = block
        out["instruction"] = obs.get("instruction", "")
        return out

    # ------------------------------------------------------------------- step
    def _step_once(self) -> None:
        """Advance one control period holding position and the gripper.

        The runtime's blocking-move loops call this between waypoints, so it has
        to be a step that changes nothing but time.
        """
        self.bus.call(self._step_holding)

    def _step_holding(self) -> None:
        """Hold the last *commanded* target, not the measured configuration.

        Re-reading the arm's actual joints and commanding those is not "hold" --
        it is "stop pushing". A position servo told to go where the arm already
        is applies no correcting effort, so under load the arm creeps and each
        step re-targets the creep. Measured on a settle-while-closing: the hand
        drifted 15 mm up and 25 mm across over 60 steps and the gripper closed
        beside the object instead of on it. Re-issuing the commanded target is
        what a real controller does, and it holds.
        """
        obs, reward, terminated, truncated, info = self.env.step(self.hold_action(), mode=self.JOINT_MODE)
        self.publish_step(reward, terminated, truncated, info, obs=obs)

    def hold_action(self) -> np.ndarray | None:
        """The action that re-issues every arm's last *commanded* target.

        Shared with the graph policy, and that sharing is the point. A graph run
        interleaves two steppers: the facade's own, servicing a node's blocking
        move, and the rollout loop's, which steps once per iteration whatever
        the graph is doing. If the second one idles, it commands the *measured*
        configuration -- and per this class's own reasoning that is not "hold",
        it is "stop pushing", so the arm relaxes on every other step.

        For a rigid pick that reads as sloppy tracking. For a peel it is the
        difference between a task and no task: tension across the bond line
        drops every time the servo lets go, so the front never advances.
        Measured on the tape graph, same waypoints and same latch as the routine
        that peels 100%: 2-7% of bonds broken while the rollout idled between
        waypoints.

        Built in :attr:`JOINT_MODE`'s layout, not the environment's *active* one.
        Everything this facade commands is a joint vector, so reading the active
        space would make the hold whatever the last policy to touch the scene
        happened to speak: a session that ran a Cartesian routine and then a graph
        produced a 7-wide OSC hold for an 8-wide joint step, which is a shape
        error if you are lucky and radians in a delta slot if you are not.
        """
        env = self.env
        space = env.controllers.get(self.JOINT_MODE)
        action = env.idle_action(self.JOINT_MODE)
        if action is None or space is None:
            return action
        per_arm = space.per_arm
        # arm_id indexes _held; the *slot* indexes the action. They differ on a
        # robot whose work arm is not declared first, and using one for the
        # other writes each hand's joint targets into the other's slice.
        for arm_id, name in enumerate(self.arm_names):
            held = self._held.get(arm_id)
            if held is not None:
                base = self._slot(name) * per_arm
                action[base : base + len(held)] = held
        for name in self.arm_names:
            action[(self._slot(name) + 1) * per_arm - 1] = self._gripper_command(name)
        return action

    def _gripper_command(self, arm_name: str) -> float:
        """One arm's held gripper command, 0 open .. 1 closed.

        Read from the environment's own :class:`~robosimstudio.solvers.Gripper`
        rather than a facade-side copy, for two reasons that both bit. A copy
        is one scalar for every arm, so a bimanual graph closing one hand
        closed both. And it is a *second writer*: a suite tool that commands a
        calibrated partial width (``sim.set_gripper``) writes the environment's
        gripper, and the very next hold action stamped the copy back over it --
        measured as a close command that reported success while the measured
        jaws stayed at 94 mm.
        """
        grippers = self.env.scene.embodiment.grippers if self.env.scene.embodiment else {}
        gripper = grippers.get(arm_name)
        return float(gripper.command) if gripper is not None else 0.0

    def stream_joint_trajectory(
        self,
        waypoints: list[list[float]],
        *,
        settle_tolerance: float = 0.01,
        settle_max_steps: int = 60,
        arm_id: int = 0,
    ) -> bool:
        """Feed a dense joint trajectory one waypoint per control step.

        Returns whether the arm settled on the last waypoint (``True`` for an
        empty trajectory: there was nowhere to go).

        The runtime looks for this method by name and, finding it, treats a
        trajectory as *the control sequence* rather than as a list of poses to
        converge on one at a time. That difference is the whole reason this
        exists. The per-waypoint fallback drives each waypoint to a tolerance
        before starting the next, which turns a smooth path into a series of
        accelerate-and-stop moves -- the exact failure
        :class:`~robosimstudio_policy_agent.ik.IKBackend`
        documents for sparse plans ("shakes a pinched object out of the fingers
        mid-transport") and that the tape routine avoids by smoothstepping
        between its waypoints instead of chasing them.

        A deformable makes it unmissable: peeling an adhesive strip is a
        *rate*-dependent process, and a hand that stops at every waypoint gives
        the bonds time to sit inside their break threshold rather than being
        walked past it.

        The settle at the end is what makes the move blocking, so a graph's next
        node starts from a arm at rest rather than mid-swing.

        Every waypoint goes through the bus. This method runs on a graph's
        *worker* thread, and :meth:`apply_joint_action` steps the environment
        itself -- calling it directly would put a second writer on the
        simulation alongside the rollout loop, which is the one thing the bus
        exists to prevent. The symptom is not a crash: the arm simply does not
        track, because roughly every other step is the rollout's idle action
        overwriting the waypoint just commanded.
        """
        for positions in waypoints:
            self.bus.call(self.apply_joint_action, np.asarray(positions, dtype=np.float64), arm_id)
        if not waypoints:
            return True
        return self.move_to_joints_blocking(
            np.asarray(waypoints[-1], dtype=np.float64),
            tolerance=settle_tolerance,
            max_steps=settle_max_steps,
            arm_id=arm_id,
        )

    def stream_dual_trajectory(
        self,
        tracks: dict[int, list[list[float]]],
        *,
        settle_tolerance: float = 0.005,
        settle_max_steps: int = 60,
    ) -> None:
        """Advance several arms along their own dense trajectories TOGETHER.

        :meth:`stream_joint_trajectory` for more than one arm, and it exists
        because two hands carrying one rigid object cannot take turns. Moving
        them alternately -- a waypoint on this arm, then a waypoint on that one
        -- either strains the grasp or, done through separate tool calls,
        costs a graph round-trip per waypoint: measured at roughly seventy
        control steps each, which spent an entire 12000-step episode on one
        two-second carry.

        Every arm's next waypoint is written into ONE action vector and the
        environment is stepped once, so the pair advances in lockstep at the
        cell's control rate. Short tracks hold their last waypoint while the
        longest finishes.

        Same bus discipline as the single-arm form: this runs on a graph worker
        thread and :meth:`apply_joint_action` steps the environment, so every
        write is marshalled onto the owner thread.
        """
        tracks = {int(a): list(w) for a, w in tracks.items() if w}
        if not tracks:
            return
        length = max(len(w) for w in tracks.values())
        for i in range(length):
            frame = {arm_id: np.asarray(w[min(i, len(w) - 1)], dtype=np.float64) for arm_id, w in tracks.items()}
            self.bus.call(self.apply_multi_joint_action, frame)
        # Settle each arm on its own end point, which also leaves `_held` fresh
        # for both -- the next hold action then holds where the pair arrived.
        for arm_id, w in tracks.items():
            self.move_to_joints_blocking(
                np.asarray(w[-1], dtype=np.float64),
                tolerance=settle_tolerance,
                max_steps=settle_max_steps,
                arm_id=arm_id,
            )

    def apply_multi_joint_action(self, frame: dict[int, np.ndarray]) -> None:
        """Command several arms' joint targets in one step. Owner thread."""
        env = self.env
        space = env.use_control(self.JOINT_MODE)
        action = env.idle_action(self.JOINT_MODE)
        per_arm = space.per_arm
        for arm_id, raw in frame.items():
            values = np.asarray(raw, dtype=np.float32).reshape(-1)
            base = self._slot(self._arm(int(arm_id))) * per_arm
            action[base : base + len(values)] = values
            self._held[int(arm_id)] = values.copy()
        for i, name in enumerate(self.arm_names):
            other = self._held.get(i)
            if i not in frame and other is not None:
                other_base = self._slot(name) * per_arm
                action[other_base : other_base + len(other)] = other
        for name in self.arm_names:
            action[(self._slot(name) + 1) * per_arm - 1] = self._gripper_command(name)
        obs, reward, terminated, truncated, info = env.step(action, mode=self.JOINT_MODE)
        self.publish_step(reward, terminated, truncated, info, obs=obs)

    JOINT_MODE = "joint_position"
    """The mode a joint vector means something in.

    Named explicitly rather than assumed, and this is the whole reason the runtime
    can drive a scene a VLA is also evaluated on: the graph's IK solutions *are*
    joint targets, so every command from here selects the joint controller, and
    what the environment's default mode happens to be stops mattering. Before this,
    a session built for ``osc_pose`` fed those radians into a normalized delta and
    the arm walked off while the graph reported progress.
    """

    def apply_joint_action(self, joints: np.ndarray, arm_id: int = 0) -> None:
        """Command one arm's joint targets, holding the others. Owner thread."""
        env = self.env
        space = env.use_control(self.JOINT_MODE)
        action = env.idle_action(self.JOINT_MODE)
        per_arm = space.per_arm
        base = self._slot(self._arm(arm_id)) * per_arm
        values = np.asarray(joints, dtype=np.float32).reshape(-1)
        action[base : base + len(values)] = values
        self._held[int(arm_id)] = values.copy()
        # Every arm holds its own last command, not its measured position. Keyed
        # by arm_id, written by slot -- see hold_action.
        for i, name in enumerate(self.arm_names):
            other = self._held.get(i)
            if i != int(arm_id) and other is not None:
                other_base = self._slot(name) * per_arm
                action[other_base : other_base + len(other)] = other
        for name in self.arm_names:
            action[(self._slot(name) + 1) * per_arm - 1] = self._gripper_command(name)
        obs, reward, terminated, truncated, info = env.step(action, mode=self.JOINT_MODE)
        self.publish_step(reward, terminated, truncated, info, obs=obs)

    def move_to_joints_blocking(
        self,
        target: np.ndarray,
        *,
        tolerance: float = 0.01,
        max_steps: int = 120,
        arm_id: int = 0,
    ) -> bool:
        """Drive one arm to a joint target, stepping until it converges.

        Returns whether it converged inside ``max_steps``.

        The runtime's ``Connector`` prefers this over its own fallback loop, and
        that preference is worth honouring: the fallback drives a raw
        ``env.handle.step(action)`` in a robosuite-shaped action layout, with a
        gripper sign convention (``-1`` open, ``+1`` closed) opposite to this
        stack's. Implementing the blocking move here keeps the action space in
        the one place that owns it.

        ``max_steps == 0`` is the runtime's fire-and-forget contract: publish
        the target once and return ``False`` without polling -- not known to
        have arrived.
        """
        goal = np.asarray(target, dtype=np.float64).reshape(-1)
        if max_steps == 0:
            self.bus.call(self.apply_joint_action, goal, arm_id)
            return False
        name = self._arm(arm_id)
        for step in range(int(max_steps)):
            reached = self.bus.call(self._step_toward, goal, arm_id, name)
            if step > 0 and reached < tolerance:
                return True
        return False

    def _step_toward(self, goal: np.ndarray, arm_id: int, arm: str) -> float:
        """One step of a blocking move and its error, in one owner-thread call."""
        self.apply_joint_action(goal, arm_id)
        return self._joint_error(arm, goal)

    def _joint_error(self, arm: str, goal: np.ndarray) -> float:
        """How far this arm is from a joint target [rad], on the owner thread.

        Read off the proprioception the last step already observed, when it
        carries this arm; the embodiment is read again only when it does not.
        """
        proprio = getattr(self, "_last_proprio", None) or {}
        if f"{arm}/joint_q" not in proprio:
            proprio = self.env.scene.embodiment.observe(self.env.state)
        current = np.asarray(proprio[f"{arm}/joint_q"], dtype=np.float64)
        n = min(len(current), len(goal))
        return float(np.linalg.norm(current[:n] - goal[:n]))

    def publish_step(
        self,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        *,
        obs: dict[str, Any] | None = None,
    ) -> None:
        """Record a step's outcome and publish the snapshot workers read.

        Public because the graph policy calls it once at reset, so a node that
        reads before the first step sees the reset state rather than an empty
        snapshot. A stepper passes the ``obs`` its ``env.step`` returned; without
        one the env is observed here.
        """
        self._sim_step_count = self.env.step_count
        self._last_reward = float(reward)
        self._current_done = bool(terminated or truncated)
        env = self.env  # bound locally: see _proprioception
        if obs is None:
            obs = env.observe()
        else:
            # ``env.step`` observes before it evaluates, and the bench renders
            # in ``evaluate`` by rebinding ``env.frames`` -- so the camera dict
            # the step's obs carries is the *previous* render, under a pose
            # read from this step's ``body_q``. Re-read the frames off the env
            # (no device readback) so image and pose come from the same step.
            if "cameras" in obs:
                obs = {**obs, "cameras": env.frames}
            if "video_cameras" in obs:
                obs = {**obs, "video_cameras": env.video_frames}
        self._last_proprio = dict(obs.get("proprio") or {})
        # Read back once, for the camera poses and the snapshot both.
        body_q = self.env.state.body_q.numpy()
        self.bus.publish(
            SimSnapshot(
                step=self.env.step_count,
                sim_time=self.env.sim_time,
                obs=self._observation(obs=obs, body_q=body_q),
                body_q=np.asarray(body_q, dtype=np.float64),
                info=dict(info),
                frames=dict(self.env.frames),
            )
        )

    # ---------------------------------------------------------------- gripper
    def _set_gripper(self, fraction: float, arm_id: int = 0) -> None:
        """Set one arm's open fraction. 1.0 is open, which is GaP's convention.

        Writes the environment's own gripper for the named arm -- the single
        holder of the command; :meth:`_gripper_command` reads it back into
        every hold action.

        Recorded per arm.
        """
        self._gripper_fraction[int(arm_id)] = float(np.clip(fraction, 0.0, 1.0))
        self.bus.call(self._write_gripper, arm_id)

    def _write_gripper(self, arm_id: int) -> None:
        gripper = self.env.scene.embodiment.grippers.get(self._arm(arm_id))
        if gripper is not None:
            gripper.command = 1.0 - self._gripper_fraction.get(int(arm_id), 1.0)
            gripper.apply(self.env.control.joint_target_q)

    # ---------------------------------------------------------------- scoring
    def compute_reward(self) -> float:
        return self._last_reward

    def task_completed(self) -> bool:
        """What the *evaluator* said. A graph may read it; it may not write it."""
        return bool(self.bus.snapshot.info.get("success", False))

    def reset(self, episode_id: int | None = None) -> dict[str, Any]:
        obs, info = self.bus.call(self.env.reset, episode_id)
        self._sim_step_count = 0
        self._current_done = False
        self._last_reward = 0.0
        # The last episode's commanded targets are not this one's hold.
        self._held.clear()
        self._gripper_fraction.clear()
        self._last_proprio = {}
        del obs
        return info

    def close(self) -> None:
        self.bus.close()
