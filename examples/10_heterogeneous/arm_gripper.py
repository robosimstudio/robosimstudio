# SPDX-License-Identifier: Apache-2.0
"""The AIRBOT Play arm and its own parallel hand, for the Clawhauser scenes.

The claw these scenes used to carry was a mechanism on a kinematic guide: a
level carrier and two prismatic jaws, prescribed along a path. The arm replaces
both halves of that -- it reaches, its own fingers close on the loop, and the toy
hangs from the pads -- while the grasp's numbers stay the ones the claw was
measured at: a 20 N closing clamp, a close commanded 16 mm past the loop's
measured half-width, and the pipeline's global soft-contact ``mu``.

**Two islands, joined one way.** The arm is an ``articulated`` robot and resolves
to MuJoCo; the toy stays on VBD. They therefore meet through a coupler, and the
coupler that carries a force-limited grip is ``one_way`` with
``slide_feedback``. Per substep it steps the rigid island, runs one Newton
collision pipeline over the whole model, and steps the deformable island with
``integrate_with_external_rigid_solver=True``: the arm is a boundary the toy
never pushes back on. ``slide_feedback`` then resolves each soft contact's push
along every prismatic joint's axis and applies it to the finger, opposite its
parent, so a finger stops where the toy pushes back with its actuator's force --
"a hand's ``effort_limit`` is its grip" (``couplers.py``). ``contact_forces``
reads that force back for ``peak_force_n`` without changing the solve.

``admm`` and ``proxy`` are the other two couplers and both are two-way. A 1.2 N
toy has no reaction worth solving, and the two-way split is what ``93133b4``
tried for the claw and dropped at the merge.

**The clamp is the model's.** The MJCF declares
``actuatorfrcrange="-5 5"`` on each finger joint. This module raises it to 20 and
restates the same number as an ``ActuationSpec``, which Newton's MuJoCo solver
writes into the actuator's ``forcerange``; :func:`check_arm` asserts the built
model carries it on both finger DOFs, so a model edited back to 5 fails here
instead of gripping at a quarter.

**The 16 mm close is a command, not a distance.** Each pad face is commanded
16 mm inside the loop's measured outer surface. The pads cannot travel that far
-- the faces meet at the finger origins -- and are not allowed to: the clamp
stalls them first. That is the same arithmetic the claw used, whose pads also
overlap at full close.

**Sticky matching is the scene's, and it has to survive the split.**
``contact_matching`` is a ``CollisionPipeline`` property, and ``build_one_way``
forwards the scene's ``CollisionSpec`` into the coupler's own pipeline. Sticky
matching keeps each pad contact's points across passes, and VBD's friction is the
drift measured from them; without it the pinch slides the loop out however high
``mu`` is. :func:`check_arm` asserts the live pipeline reports it.

**Reading the solver through the coupler.** Once there are two islands,
``sim.pipeline.program.solver`` is the coupler, not VBD. Anything that reads a
VBD attribute -- the self-contact pair buffers, the coarse solve's path -- has to
unwrap it first, or the read returns ``None`` and the diagnostic silently reports
zeros. :func:`vbd_solver` is that unwrap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import robosimstudio as rss
from robosimstudio.engines.pose import quat_rotate
from robosimstudio.robots.registry import register_gripper
from robosimstudio.specs import ActuationSpec, ArmSpec, AssetSpec, EmbodimentSpec, GripperSpec

ASSET = Path(__file__).resolve().parent / "assets" / "airbot_play" / "airbot_play.xml"
ARM = "right"
"""The arm's name in every scene: the AIRBOT carries one."""

HOME_Q = (0.0, -1.0, 1.2, 1.5708, -1.2, -1.5708)
"""The pose the source model's own demonstrations start from [rad]."""

GRASP_QUAT = (0.70710678, 0.70710678, 0.0, 0.0)
"""Tool frame facing down the bench, its jaw axis spanning world x, xyzw.

The loop's pads close along x (``ring_geometry``'s ``side_reach`` is an x
extent), so this is the orientation that puts the jaws across the cord.
"""

FINGER_OPEN_Q = 0.04
"""One finger's slide coordinate at full open [m]; the MJCF's own range."""

PAD_INNER_OFFSET = 0.0
"""Each pad face outboard of its finger's origin [m].

This is the stock DISCOVERSE hand: a 10.5 x 4 x 40 mm pad box centred 2 mm out
along the closing axis, so the face is the finger body's own origin plane.
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

Ten was the claw's clamp and the paper fold's, and on this loop it stalls the
pads 5.6 mm into an 8.3 mm cord, where the round cord creeps out (1.96x ring
stretch, lost at frame 61 of the lift). Twenty holds it: the pads read 16-21 N
through the lift and the toy leaves the bench.
"""

CLOSE_TRAVEL = 0.016
"""How far past a measured half-width the pads are commanded [m].

A command, not a distance: the cap stops the fingers wherever the toy's push
balances it, and on this loop the command saturates before the cap does.
"""

CLOSE_AXIS = (0.0, 1.0, 0.0)
"""The jaws' closing axis in the hand's frame: the spec's ``close_axis``."""

GRIP_TOLERANCE = 0.5
"""How far past ``GRASP_LIMIT_N`` a pad's measured squeeze may read.

Not a free parameter: it is the paper fold's own bound for the same stack -- a
10 N ``effort_limit``, 9.8 N holding, 13.5 N pressing, never over 15 N at any
step (``examples/03_materials/05_paper``). The actuator is clamped exactly; what
overshoots is the contact readback, and only while the pads are still biting.
Measured on the loop: the rod's close spiked to 14.6 N as the pads landed and
settled to 10.1 / 10.2 N, and the pick-and-drop's carry held 10.1 N with 11.1 N
of total load.
"""

PUSH_LIMIT_N = 60.0
"""The arm's press budget [N], the ceiling its own actuators reach.

The claw pressed through a 50 kN/m carrier spring, and its 200 N budget was a
number that spring could deliver. Six joints at 24 and 16 N.m cannot: the press
stops where the arm stalls, so the budget is stated at what the arm can do and
the scene's press path is measured against it rather than against 200.
"""


PAD_RISE = 0.011
"""How far above the tool-centre point this hand's pads sit [m], grasp down.

Measured on the built rod scene: the finger bodies land 6.2 mm above the point
the arm is commanded to and each pad box is centred 5 mm up its own finger's
frame. The claw states a grasp as its pads' midpoint, and a scene handing the arm
the same point has to put the tool centre this much below it -- 11 mm is a third
of the loop's 34 mm reach and most of its hole.
"""


def tcp_for_pads(point) -> np.ndarray:
    """The tool-centre point that puts this hand's pads on ``point`` [m]."""
    return np.asarray(point, dtype=np.float64) - np.array((0.0, 0.0, PAD_RISE), dtype=np.float64)


def pad_forces(sim, forces: np.ndarray, sides, *, name: str = ARM) -> tuple[list[float], list[float]]:
    """Each pad's contact force [N]: summed, and the part along the closing axis.

    ``shape_contact_forces`` sums every rigid-soft record against a shape, so the
    vector carries friction as well as the squeeze: on the loop at ``mu`` 4 a pad
    capped at 10 N read 11.1 N total, 4 N of it along the hand's z. What the cap
    bounds, and what the claw's own drive reports, is the component along the
    closing axis -- that is the grip, and the total is the pad's load.
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
    """The 0 (open) .. 1 (closed) command that aims the pad faces at ``half_gap``.

    Exact for this hand: the pad faces are the finger bodies' own origins, so the
    command is the inverse of the 40 mm slide and ``GripperSpec.jaw_gap``'s
    ``1 - gap / span`` gives the same number.
    """
    travel = OPEN_HALF_GAP - CLOSED_HALF_GAP
    return float(np.clip((OPEN_HALF_GAP - half_gap) / travel, 0.0, 1.0))


def half_gap_of(command: float) -> float:
    """The pad-face half-gap a command aims at [m], the inverse of the above."""
    return OPEN_HALF_GAP - float(np.clip(command, 0.0, 1.0)) * (OPEN_HALF_GAP - CLOSED_HALF_GAP)


def grip_actuation(force_limit: float = GRASP_LIMIT_N) -> ActuationSpec:
    """The hand's servo: stiff enough that ``force_limit`` is what stops the fingers.

    The paper fold's measured stack. At the MJCF's own ``kp=350`` a 12 mm command
    against a stalled pad stands 28 mm of error at 10 N, so the servo -- not the
    actuator's ceiling -- would decide the grip. At 1e5 the same force is 0.1 mm
    of error, and the ceiling binds where it is meant to.
    """
    return ActuationSpec(target_ke=1.0e5, target_kd=20.0, effort_limit=force_limit, joint_damping=200.0)


register_gripper(
    GripperSpec(
        name="airbot_play_loop",
        finger_names=("endleft", "endright"),
        finger_open_q=(FINGER_OPEN_Q, FINGER_OPEN_Q),
        finger_closed_q=(0.0, 0.0),
        open_width=FINGER_OPEN_Q,
        # The faces' own separation: the stock pads' inner faces are the finger
        # origins, 40 mm of slide each side, so the mouth is exactly 80 mm.
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
        # The scene's knob reaches the hand here rather than through the registry:
        # the entry states this hand's shape and its default grip, and a scene that
        # asks for a different clamp is a scene overriding one number.
        gripper_actuation=grip_actuation(force_limit),
        # The MJCF declares gravcomp="1" on every body and the importer harvests
        # it; restating it names this robot's bodies and nothing else, which is
        # what a scene holding loose objects needs.
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

    A one-island scene hands back VBD; a two-island one hands back the coupler,
    which owns VBD as ``soft``. Reading a VBD attribute off the coupler returns
    ``None`` and the diagnostic it feeds reports zero, which is why this exists.
    """
    return getattr(solver, "soft", solver)


def check_arm(sim, *, limit: float = GRASP_LIMIT_N, name: str = ARM) -> dict:
    """Assert the built arm carries the grip this module promises, and report it."""
    arm = sim.handles.arms[name]
    effort = sim.model.joint_effort_limit.numpy()[list(arm.gripper_dofs)]
    if not np.allclose(effort, limit, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"the arm's finger effort limits are {effort.tolist()}, not {limit}")
    solver = sim.pipeline.program.solver
    # The collider that matters is the one the pads' contacts come from: the
    # coupler's own pipeline once there are two islands, else the scene's.
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
    ``command`` the 0..1 closing command for the same frame. Built once from the
    scene's schedule; a scene that replans mid-run builds another.
    """

    joint_q: np.ndarray
    command: np.ndarray
    worst_position_error_m: float


class ArmControl:
    """One arm's planned route, and the joint targets that follow it.

    The IK runs here rather than in a scene: a scene states a tool-centre path in
    metres and a pad-face half-gap against time, and gets back the joint targets
    the servo is given at each control frame. Seeding each solve from the last
    keeps the solution on one branch, which matters on a path that lifts a loop
    off a bench and threads a rod through it.
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
        ``ParallelGripper``: two finger poses whose midpoint is the claw's centre
        and whose separation is its jaw gap. The arm takes both: the midpoint
        becomes its tool-centre path (shifted by ``origin``, for a path stated in
        the work-surface frame), and the separation, less ``pad_inset``, the half
        gap the pad *faces* are commanded to. The caller states a separation it
        measured between finger origins and this is where it becomes a face gap --
        so a scene keeps one motion and each mover reads it at its own faces. The
        hand's face is 2 mm outboard of its finger's origin, where the claw's is
        3 mm inboard, so the bottle scene adds that back before passing it here.

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
            # The poses are the claw's pads, so the point is a pad midpoint, not a
            # tool centre: the hand's pads are `PAD_RISE` above what it is given.
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
                # Most of a scene's frames hold still; only solve when the pose
                # or the aperture actually moved.
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
        """Place the arm at the route's first frame, before the scene's first step.

        The claw starts at its grasp pose; the arm starts at its home keyframe
        340 mm above it, so without this the pads spend the close travelling and
        arrive on the loop already shut -- measured, they hit the cord at 0.8 m/s
        and the squeeze spiked to 15 N before settling. A scene states its phases
        as motion from the grasp, and this is what makes the two agree. MuJoCo
        syncs its own ``qpos`` from Newton's state on the first step, so the
        assignment lands before anything is solved.
        """
        q = self.sim.pipeline.state_0.joint_q.numpy().copy()
        q[self.coords] = route.joint_q[0]
        # The fingers at the aperture this frame asks for, not at whatever the
        # model's rest pose is: the joint vector the route carries was seeded from
        # the built state, where this hand sits shut, so copying it whole started
        # every scene with the pads closed and drove them open through the loop.
        self.hand.command = float(route.command[0])
        self.hand.apply(self.targets)
        targets = self.targets.numpy()
        q[list(self.hand.coords)] = targets[list(self.hand.coords)]
        # Both buffers: the substep loop swaps them, and a caller cannot know
        # which half the first step reads. The entity's own reset writes both for
        # the same reason.
        for state in (self.sim.pipeline.state_0, self.sim.pipeline.state_1):
            state.joint_q.assign(q)
        self.targets.assign(q)

    def follow(self, route: Route, frame: int) -> None:
        """Write this frame's arm targets and closing command.

        Past the end of a route the arm holds its last frame, as the claw's own
        trajectory does: a guarded handover plans a couple of seconds of hold,
        open and withdraw inside a run that goes on, so most of its frames outlive
        the plan.
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
        fingers slide the same way, so their difference is no separation and
        ``width`` falls back to a span-scaled approximation. A finger's own slide
        is what the face is mounted on.
        """
        q = self.sim.pipeline.state_0.joint_q.numpy()[list(self.hand.coords)]
        return PAD_INNER_OFFSET + 0.5 * float(q.sum())
