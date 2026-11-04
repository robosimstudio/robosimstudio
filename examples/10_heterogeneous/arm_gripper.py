# SPDX-License-Identifier: Apache-2.0
"""The AIRBOT Play arm and its own parallel hand, for the Clawhauser scenes.

The arm is ``articulated`` and resolves to MuJoCo while the toy stays on VBD, so
the two islands meet through a ``one_way`` coupler with ``slide_feedback``: a
finger stops where the toy pushes back with its actuator's force, which is what
makes ``effort_limit`` the grip. ``contact_forces`` reads that force back.
Sticky contact matching must survive the split, since it is what VBD's friction
is measured from; :func:`check_arm` asserts the live pipeline reports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import robosimstudio as rss
from robosimstudio.core.render.palette import BENCH_LIGHT, GROUND
from robosimstudio.engines.pose import quat_rotate
from robosimstudio.robots.registry import register_gripper
from robosimstudio.specs import ActuationSpec, ArmSpec, AssetSpec, CellSpec, EmbodimentSpec, GripperSpec, StaticSpec

ASSET = Path(__file__).resolve().parent / "assets" / "airbot_play" / "airbot_play.xml"
CELL = "bench_075_airbot"
"""``bench_075``'s top, extended behind the arm's mount."""

MOUNT = (-0.48, 0.0, 0.132, 0.0, 0.0, 0.0, 1.0)
"""The arm's base in the work-surface frame: on its pedestal, 0.35 m from the toy.

Lowering it to the bench top changes the IK's wrist roll and the pads miss the
loop, so the mount height every route was planned and measured at is kept, and
the pedestal below carries it.
"""

PEDESTAL_TOP = 0.136
"""The plinth's height above the bench [m].

The base mesh starts 3.9 mm above the body origin, so this is flush with it.
"""

rss.register_cell(
    CellSpec(
        name=CELL,
        ground_z=0.0,
        surface_z=0.75,
        # The frame stays bench_075's, so every scene coordinate is unchanged.
        center=(0.6, 0.0, 0.0),
        half=(0.62, 0.65, 0.375),
        # bench_075's top reaches x = 0.3025 and the arm's mount is at 0.12, so its
        # base stood 180 mm off the near edge in mid-air.
        statics=(
            StaticSpec(
                name="table",
                half_extents=(0.62, 0.65, 0.375),
                xform=(0.6, 0.0, 0.375, 0.0, 0.0, 0.0, 1.0),
                color=GROUND,
            ),
            StaticSpec(
                name="arm_pedestal",
                half_extents=(0.09, 0.09, PEDESTAL_TOP / 2.0),
                xform=(0.104, 0.0, 0.75 + PEDESTAL_TOP / 2.0, 0.0, 0.0, 0.0, 1.0),
                color=BENCH_LIGHT,
            ),
        ),
        provenance="bench_075's table extended 180 mm to the AIRBOT's mount",
    )
)
ARM = "right"
"""The arm's name in every scene: the AIRBOT carries one."""

HOME_Q = (0.0, -1.0, 1.2, 1.5708, -1.2, -1.5708)
"""The pose the source model's own demonstrations start from [rad]."""

GRASP_QUAT = (0.70710678, 0.70710678, 0.0, 0.0)
"""Tool frame facing down the bench with the jaw axis along world x, xyzw.

The loop's pads close along x, so this puts the jaws across the cord.
"""

FINGER_OPEN_Q = 0.04
"""One finger's slide coordinate at full open [m]; the MJCF's own range."""

PAD_INNER_OFFSET = 0.0
"""Each pad face outboard of its finger's origin [m].

The stock pad's face is the finger body's own origin plane.
"""

PAD_HALF_THICKNESS = 0.002
"""Half a pad's thickness along the closing axis [m], for a scene stating a gap."""

OPEN_HALF_GAP = FINGER_OPEN_Q + PAD_INNER_OFFSET
"""Half the gap between the pad faces at full open [m]: 40 mm."""

CLOSED_HALF_GAP = PAD_INNER_OFFSET
"""Half the gap at full closed [m]: the two faces meet."""

SPAN = 2.0 * OPEN_HALF_GAP
"""Pad-face separation at full open [m]; what ``GripperSpec.jaw_gap`` inverts."""

GRASP_LIMIT_N = 20.0
"""The jaws' closing budget [N]: one finger's actuator force, never more.

At 10 N the pads stall 5.6 mm into the loop's 8.3 mm cord and the ring creeps
out during the lift; at 20 N it holds.
"""

CLOSE_TRAVEL = 0.016
"""How far past a measured half-width the pads are commanded [m].

A command, not a distance: the cap stops the fingers wherever the toy's push
balances it.
"""

CLOSE_AXIS = (0.0, 1.0, 0.0)
"""The jaws' closing axis in the hand's frame: the spec's ``close_axis``."""

GRIP_TOLERANCE = 0.5
"""How far past ``GRASP_LIMIT_N`` a pad's contact readback may spike.

The actuator is clamped exactly; only the readback overshoots, and only while
the pads are still biting.
"""

PUSH_LIMIT_N = 60.0
"""The arm's press budget [N], the ceiling its own actuators reach.

The claw's 200 N came from a 50 kN/m carrier spring; six joints at 24 and 16 N.m
stall well below it, so the scene's press path is measured against this.
"""


PAD_RISE = 0.011
"""How far above the tool-centre point this hand's pads sit [m], grasp down.

Measured on the built rod scene: finger bodies land 6.2 mm above the commanded
point and each pad box is centred 5 mm up its finger's frame.
"""


def tcp_for_pads(point) -> np.ndarray:
    """The tool-centre point that puts this hand's pads on ``point`` [m]."""
    return np.asarray(point, dtype=np.float64) - np.array((0.0, 0.0, PAD_RISE), dtype=np.float64)


def pad_forces(sim, forces: np.ndarray, sides, *, name: str = ARM) -> tuple[list[float], list[float]]:
    """Each pad's contact force [N]: summed, and the part along the closing axis.

    ``shape_contact_forces`` sums every rigid-soft record against a shape, so the
    total carries friction as well as the squeeze. The cap bounds the component
    along the closing axis, which is the grip.
    """
    _, quat = sim.arm(name).ee_pose(sim.pipeline.state_0)
    axis = quat_rotate(quat, np.asarray(CLOSE_AXIS, dtype=np.float64))
    total, squeeze = [], []
    for side in sides:
        force = forces[side].sum(axis=0)
        total.append(float(np.linalg.norm(force)))
        squeeze.append(float(abs(np.dot(force, axis))))
    return total, squeeze


def finger_command(half_gap: float) -> float:
    """The 0 (open) .. 1 (closed) command that aims the pad faces at ``half_gap``."""
    travel = OPEN_HALF_GAP - CLOSED_HALF_GAP
    return float(np.clip((OPEN_HALF_GAP - half_gap) / travel, 0.0, 1.0))


def half_gap_of(command: float) -> float:
    """The pad-face half-gap a command aims at [m], the inverse of the above."""
    return OPEN_HALF_GAP - float(np.clip(command, 0.0, 1.0)) * (OPEN_HALF_GAP - CLOSED_HALF_GAP)


def grip_actuation(force_limit: float = GRASP_LIMIT_N) -> ActuationSpec:
    """The hand's servo, stiff enough that ``force_limit`` is what stops the fingers.

    At the MJCF's own ``kp=350`` a stalled pad stands 28 mm of error at 10 N and
    the servo, not the ceiling, would decide the grip; at 1e5 the same force is
    0.1 mm.
    """
    return ActuationSpec(target_ke=1.0e5, target_kd=20.0, effort_limit=force_limit, joint_damping=200.0)


register_gripper(
    GripperSpec(
        name="airbot_play_loop",
        finger_names=("endleft", "endright"),
        finger_open_q=(FINGER_OPEN_Q, FINGER_OPEN_Q),
        finger_closed_q=(0.0, 0.0),
        open_width=FINGER_OPEN_Q,
        # The stock pads' inner faces are the finger origins, 40 mm of slide each
        # side, so the mouth is exactly 80 mm.
        span=SPAN,
        tcp_body="link6",
        tcp_offset=(0.0, 0.0, 0.005),
        approach_axis=(0.0, 0.0, 1.0),
        close_axis=(0.0, 1.0, 0.0),
        finger_bodies=(("left",), ("right",)),
        open_axis_signs=(1.0, -1.0),
        pad_half_span=FINGER_OPEN_Q + 0.002,
        pad_thickness=0.040,
        tip_reach=0.020,
        housing_half=0.04,
        actuation=grip_actuation(),
    )
)


def arm_embodiment(
    *, base_offset, name: str = "airbot_play_loop", home_q=HOME_Q, force_limit: float = GRASP_LIMIT_N
) -> EmbodimentSpec:
    """The AIRBOT as an embodiment, mounted at ``base_offset`` (xyz + xyzw)."""
    return EmbodimentSpec(
        name=name,
        asset=AssetSpec(path=str(ASSET)),
        base_offset=base_offset,
        # A scene overrides the default grip through this one number.
        gripper_actuation=grip_actuation(force_limit),
        # The MJCF declares gravcomp="1" on every body; restating it names this
        # robot's bodies and nothing else.
        gravcomp=True,
        arms=(
            ArmSpec(
                name=ARM,
                joint_names=tuple(f"joint{index}" for index in range(1, 7)),
                ee_body_name="link6",
                gripper="airbot_play_loop",
                home_q=home_q,
            ),
        ),
    )


def add_arm(scene, *, base_offset, name: str = "airbot_play_loop", home_q=HOME_Q, force_limit: float = GRASP_LIMIT_N):
    """Add the arm to ``scene`` and return its placement."""
    return scene.add(
        rss.robot(arm_embodiment(base_offset=base_offset, name=name, home_q=home_q, force_limit=force_limit))
    )


def vbd_solver(solver):
    """The deformable solver inside ``solver``, or ``solver`` when it is bare.

    A two-island scene hands back the coupler, which owns VBD as ``soft``.
    """
    return getattr(solver, "soft", solver)


def check_arm(sim, *, limit: float = GRASP_LIMIT_N, name: str = ARM) -> dict:
    """Assert the built arm carries the grip this module promises, and report it."""
    arm = sim.handles.arms[name]
    effort = sim.model.joint_effort_limit.numpy()[list(arm.gripper_dofs)]
    if not np.allclose(effort, limit, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"the arm's finger effort limits are {effort.tolist()}, not {limit}")
    solver = sim.pipeline.program.solver
    # The collider the pads' contacts come from: the coupler's own pipeline once
    # there are two islands, else the scene's.
    pipeline = getattr(solver, "collision", None) or getattr(sim.pipeline.program, "collision", None)
    mode = getattr(pipeline, "contact_matching", None)
    if mode != "sticky":
        raise RuntimeError(f"contact matching is {mode!r}, not 'sticky': pad friction would be per-pass drift")
    if abs(float(arm.gripper_span) - SPAN) > 1e-9:
        raise RuntimeError(f"the arm's span is {arm.gripper_span}, not {SPAN}")
    return {
        "gripper": "airbot_play_loop",
        "finger_effort_limit_n": effort.tolist(),
        "close_travel_m": CLOSE_TRAVEL,
        "pad_faces_open_m": SPAN,
        "pad_faces_closed_m": 2.0 * CLOSED_HALF_GAP,
        "contact_matching": mode,
        "coupler": type(solver).__name__,
        "press_budget_n": PUSH_LIMIT_N,
    }


@dataclass(frozen=True)
class Route:
    """One arm move, resampled at the control rate.

    ``joint_q`` is this arm's coordinates per frame, in ``arm_coords`` order, and
    ``command`` the 0..1 closing command for the same frame.
    """

    joint_q: np.ndarray
    command: np.ndarray
    worst_position_error_m: float


class ArmControl:
    """One arm's planned route, and the joint targets that follow it.

    The IK runs here: a scene states a tool-centre path in metres and a pad-face
    half-gap against time, and gets back the joint targets the servo is given at
    each control frame. Seeding each solve from the last keeps it on one branch.
    """

    def __init__(self, sim, *, name: str = ARM) -> None:
        self.sim = sim
        self.name = name
        self.arm = sim.arm(name)
        self.hand = sim.gripper(name)
        self.targets = sim.pipeline.control.joint_target_q
        self.coords = np.asarray(self.arm.arm_coords, dtype=int)

    def plan(self, *, quat, capacity: int, dt: float, tcp_at, half_gap_at) -> Route:
        """Solve the arm's path: ``tcp_at(time)`` in metres, ``half_gap_at(time)`` in metres."""

        def sample(index: int) -> tuple[np.ndarray, float]:
            time = index * dt
            return np.asarray(tcp_at(time), dtype=np.float64), float(half_gap_at(time))

        return self._plan(quat=quat, capacity=capacity, sample=sample)

    def plan_poses(self, poses, *, quat, substeps: int, origin=0.0, pad_inset: float = 0.0) -> Route:
        """A scene's own claw path, retargeted onto the arm's tool centre.

        ``poses`` is the ``[N, 2, 7]`` per-substep array the claw scenes build for
        ``ParallelGripper``. Its midpoint becomes the tool-centre path, shifted by
        ``origin``; the separation, less ``pad_inset``, becomes the pad faces'
        half gap. The hand's face is 2 mm outboard of its finger's origin, where
        the claw's is 3 mm inboard, so the caller passes the difference.
        """
        poses = np.asarray(poses, dtype=np.float32)
        if poses.ndim != 3 or poses.shape[1:] != (2, 7):
            raise ValueError(f"poses must be [N, 2, 7], got {poses.shape}")
        frames = len(poses)
        capacity = -(-frames // substeps)
        shift = np.broadcast_to(np.asarray(origin, dtype=np.float64), (3,))

        def sample(index: int) -> tuple[np.ndarray, float]:
            frame = min(index * substeps, frames - 1)
            place = 0.5 * (poses[frame, 0, :3].astype(np.float64) + poses[frame, 1, :3].astype(np.float64))
            gap = 0.5 * float(np.linalg.norm(poses[frame, 1, :3] - poses[frame, 0, :3]))
            # The poses are the claw's pads, so the point is a pad midpoint.
            return tcp_for_pads(place + shift), gap - float(pad_inset)

        return self._plan(quat=quat, capacity=capacity, sample=sample)

    def _plan(self, *, quat, capacity: int, sample) -> Route:
        """Solve one tool-centre waypoint per control frame, seeded from the last."""
        seed = self.sim.pipeline.state_0.joint_q.numpy().astype(np.float32)
        joint_q = np.empty((capacity, len(self.coords)), dtype=np.float32)
        command = np.empty(capacity, dtype=np.float32)
        worst = 0.0
        solved = seed
        last: tuple[np.ndarray, float] | None = None
        with self.arm.host_fk():
            for index in range(capacity):
                target, half_gap = sample(index)
                # Most frames hold still; only solve when the pose or the
                # aperture actually moved.
                if last is None or np.linalg.norm(target - last[0]) > 1e-7 or abs(half_gap - last[1]) > 1e-9:
                    solved = self.arm.solve_q_full(seed, target, quat)
                    position_error, _ = self.arm.pose_errors(solved, target, quat)
                    worst = max(worst, float(position_error))
                    seed = solved
                    last = (target, half_gap)
                joint_q[index] = solved[self.coords]
                command[index] = finger_command(half_gap)
        return Route(joint_q=joint_q, command=command, worst_position_error_m=worst)

    def start(self, route: Route) -> None:
        """Place the arm and its fingers at the route's first frame.

        Without this the pads spend the close travelling from the arm's home
        pose and arrive on the loop already shut. MuJoCo syncs its ``qpos`` from
        Newton's state on the first step, so the assignment lands before anything
        is solved.
        """
        q = self.sim.pipeline.state_0.joint_q.numpy().copy()
        q[self.coords] = route.joint_q[0]
        # The fingers at this frame's aperture: the route's joint vector was
        # seeded from the built state, where this hand sits shut.
        self.hand.command = float(route.command[0])
        self.hand.apply(self.targets)
        targets = self.targets.numpy()
        q[list(self.hand.coords)] = targets[list(self.hand.coords)]
        # Both buffers: the substep loop swaps them, so a caller cannot know which
        # half the first step reads.
        for state in (self.sim.pipeline.state_0, self.sim.pipeline.state_1):
            state.joint_q.assign(q)
        self.targets.assign(q)

    def follow(self, route: Route, frame: int) -> None:
        """Write this frame's arm targets and closing command.

        Past the end of a route the arm holds its last frame, as the claw's own
        trajectory does.
        """
        frame = min(int(frame), len(route.joint_q) - 1)
        self.arm.set_joint_targets(route.joint_q[frame], self.targets)
        self.hand.command = float(route.command[frame])
        self.hand.apply(self.targets)

    def tcp(self) -> np.ndarray:
        """The measured tool-centre point [m]."""
        return self.arm.ee_pose(self.sim.pipeline.state_0)[0]

    def aperture(self) -> float:
        """The measured pad-face half-gap [m].

        Read off the finger coordinates, not ``Gripper.width``: this hand's two
        fingers slide the same way, so ``width`` falls back to a span-scaled
        approximation.
        """
        q = self.sim.pipeline.state_0.joint_q.numpy()[list(self.hand.coords)]
        return PAD_INNER_OFFSET + 0.5 * float(q.sum())
