# SPDX-License-Identifier: Apache-2.0
"""A level parallel claw with a body, two finger posts and prismatic pads.

The archive's mechanism (`yiduo/archive-clawhauser-backend-20261027`,
`parallel_gripper.py`) is the reference: a carrier body, two prismatic finger
children, and a jaw budget separate from the body's push. The archive's carrier
was a free rigid body; on this toy the 5 N jaws' reaction turned it, and the pads
turned with it. Here the carrier is held level by a kinematic guide and may move
only in z, where the 50 N push budget lives. The two jaws slide in x on the
carrier under 5 N. No joint can rotate a pad.

The carrier is the claw body (120 x 30 x 18 mm). Each finger is a 10 x 12 x 30 mm
post with a 6 x 12 x 12 mm pad on its inner lower face: a small patch that stays
within the post's own 12 mm width and 30 mm height, as in the sketch. There are
no ledges: the pad face and its friction are the only grip aid.
"""

from __future__ import annotations

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp
from gripper import trajectory_velocities

import robosimstudio as rss


@wp.func
def _slide_state(
    carrier: int,
    child: int,
    side: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
):
    """The prismatic coordinate, rate, lateral error and relative angle."""
    parent_pose, child_pose = body_q[carrier], body_q[child]
    parent_origin = wp.transform_get_translation(parent_pose)
    child_origin = wp.transform_get_translation(child_pose)
    axis = wp.quat_rotate(wp.transform_get_rotation(parent_pose), wp.vec3(side, 0.0, 0.0))
    displacement = child_origin - parent_origin
    coordinate = wp.dot(displacement, axis)
    parent_omega = wp.spatial_bottom(body_qd[carrier])
    parent_velocity = wp.spatial_top(body_qd[carrier]) + wp.cross(
        parent_omega, parent_origin - wp.transform_point(parent_pose, body_com[carrier])
    )
    child_velocity = wp.spatial_top(body_qd[child]) + wp.cross(
        wp.spatial_bottom(body_qd[child]), child_origin - wp.transform_point(child_pose, body_com[child])
    )
    rate = wp.dot(child_velocity - parent_velocity - wp.cross(parent_omega, displacement), axis)
    lateral = wp.length(displacement - coordinate * axis)
    cosine = wp.abs(wp.dot(wp.transform_get_rotation(parent_pose), wp.transform_get_rotation(child_pose)))
    angle = 2.0 * wp.acos(wp.clamp(cosine, 0.0, 1.0))
    return coordinate, rate, lateral, angle


@wp.kernel
def _set_guide(
    tick: wp.array[int],
    lengths: wp.array[int],
    poses: wp.array2d[wp.transform],
    rates: wp.array2d[wp.spatial_vector],
    guide: int,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Hold the guide at the finger midpoint, level with the target orientation."""
    frame = wp.min(tick[0], lengths[0] - 1)
    left, right = poses[frame, 0], poses[frame, 1]
    midpoint = (wp.transform_get_translation(left) + wp.transform_get_translation(right)) * 0.5
    linear = (wp.spatial_top(rates[frame, 0]) + wp.spatial_top(rates[frame, 1])) * 0.5
    angular = (wp.spatial_bottom(rates[frame, 0]) + wp.spatial_bottom(rates[frame, 1])) * 0.5
    body_q[guide] = wp.transform(midpoint, wp.transform_get_rotation(left))
    body_qd[guide] = wp.spatial_vector(linear, angular)


@wp.kernel
def _drive_carrier(
    dof: int,
    carrier: int,
    guide: int,
    fingers: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_world: wp.array[int],
    gravity: wp.array[wp.vec3],
    ke: float,
    kd: float,
    press_ke: float,
    limit: float,
    compensate_gravity: bool,
    pressing: wp.array[int],
    joint_f: wp.array[float],
    applied: wp.array[wp.spatial_vector],
    peak_force: wp.array[float],
    peak_torque: wp.array[float],
):
    """Carry the carrier on its guide, and press against the push budget.

    Two modes, chosen by the claw's own state, differing only in stiffness. The
    press needs a stiff one because the force is ``ke * lag``: 1500 N/m reaches
    only 127 N over the whole 85 mm stroke, so the budget was decoration, while
    50 kN/m reaches 200 N in 4 mm and the pads still follow the path instead of
    sinking into the body. A commanded force was tried and is worse -- with no
    position term the pads drove 20-40 mm past the path, through the toy.

    The coordinate is read from the bodies, not the joint q: the joint's q sign
    is the solver's, and a PD aimed at the wrong sign drove the carrier 40 mm
    into the bench on the first frame.
    """
    axis = wp.vec3(0.0, 0.0, 1.0)
    parent, child = body_q[guide], body_q[carrier]
    parent_origin = wp.transform_get_translation(parent)
    child_origin = wp.transform_get_translation(child)
    coordinate = wp.dot(child_origin - parent_origin, axis)
    parent_velocity = wp.spatial_top(body_qd[guide]) + wp.cross(
        wp.spatial_bottom(body_qd[guide]), parent_origin - wp.transform_point(parent, body_com[guide])
    )
    child_velocity = wp.spatial_top(body_qd[carrier]) + wp.cross(
        wp.spatial_bottom(body_qd[carrier]), child_origin - wp.transform_point(child, body_com[carrier])
    )
    rate = wp.dot(child_velocity - parent_velocity, axis)
    stiffness = press_ke if pressing[0] != 0 else ke
    effort = stiffness * (0.0 - coordinate) + kd * (0.0 - rate)
    if compensate_gravity:
        for link in range(3):
            body = carrier if link == 0 else fingers[link - 1]
            effort -= body_mass[body] * gravity[body_world[body]][2]
    effort = wp.clamp(effort, -limit, limit)
    joint_f[dof] = effort
    applied[0] = wp.spatial_vector(wp.vec3(0.0, 0.0, effort), wp.vec3())
    if pressing[0] != 0:
        peak_force[0] = wp.max(peak_force[0], wp.abs(effort))
        peak_torque[0] = 0.0


@wp.kernel
def _drive_slides(
    tick: wp.array[int],
    lengths: wp.array[int],
    poses: wp.array2d[wp.transform],
    rates: wp.array2d[wp.spatial_vector],
    carrier: int,
    bodies: wp.array[int],
    dofs: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    ke: float,
    kd: float,
    limit: float,
    joint_f: wp.array[float],
    applied: wp.array[wp.spatial_vector],
    peak_force: wp.array[float],
):
    """PD the two slides to the target half-gap, clamped to the jaw budget."""
    finger = wp.tid()
    frame = wp.min(tick[0], lengths[0] - 1)
    left, right = poses[frame, 0], poses[frame, 1]
    rotation = wp.transform_get_rotation(left)
    axis = wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0))
    separation = wp.transform_get_translation(right) - wp.transform_get_translation(left)
    half_gap = wp.dot(separation, axis) * 0.5
    gap_rate = wp.dot(
        wp.spatial_top(rates[frame, 1]) - wp.spatial_top(rates[frame, 0]), axis
    ) * 0.5
    coordinate, rate, _lateral, _angle = _slide_state(
        carrier,
        bodies[finger],
        float(2 * finger - 1),
        body_q,
        body_qd,
        body_com,
    )
    effort = wp.clamp(
        ke * (half_gap - coordinate) + kd * (gap_rate - rate), -limit, limit
    )
    joint_f[dofs[finger]] = effort
    applied[finger] = wp.spatial_vector(wp.vec3(effort, 0.0, 0.0), wp.vec3())
    peak_force[finger] = wp.max(peak_force[finger], wp.abs(effort))


@wp.kernel
def _measure_slides(
    carrier: int,
    bodies: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    coordinates: wp.array[float],
    rates: wp.array[float],
    lateral: wp.array[float],
    angle: wp.array[float],
):
    finger = wp.tid()
    coordinate, rate, lateral_error, angular_error = _slide_state(
        carrier,
        bodies[finger],
        float(2 * finger - 1),
        body_q,
        body_qd,
        body_com,
    )
    coordinates[finger] = coordinate
    rates[finger] = rate
    lateral[finger] = lateral_error
    angle[finger] = angular_error


@wp.kernel
def _advance_tick(tick: wp.array[int], lengths: wp.array[int]):
    tick[0] = wp.min(tick[0] + 1, lengths[0] - 1)


GRASP_LIMIT_N = 5.0
"""The jaws' closing budget: the slides never press the loop harder [N]."""
PUSH_LIMIT_N = 200.0
"""The carrier's push budget while pressing the toy [N]. Independent of the
jaws: a claw that can only pinch at 5 N can still push at 200 N, and neither
clamp appears in the other's drive."""


@dataclass(frozen=True, kw_only=True)
class ParallelGripperSpec:
    entity: str = "parallel_loop_gripper"
    name: str = "parallel_loop_gripper"
    mode: str = "dynamic"
    center: tuple[float, float, float] = (0.0, 0.0, 0.02)
    half_gap: float = 0.015
    closed_half_gap: float = 0.005
    minimum_half_gap: float = 0.0032
    maximum_half_gap: float = 0.060
    force_limit: float = GRASP_LIMIT_N
    """The jaws' closing budget. The slide drive never exceeds it [N]."""
    push_force_limit: float = PUSH_LIMIT_N
    """The carrier's push budget. Its own drive clamps here, so the jaws' clamp
    does not bound the press [N]."""
    jaw_ke: float = 1500.0
    jaw_kd: float = 15.0
    carrier_ke: float = 1500.0
    carrier_kd: float = 42.0
    carrier_press_ke: float = 50000.0
    """The carrier's stiffness while pressing [N/m]. The press force is this
    spring's lag, so it has to be stiff enough to reach the push budget inside a
    few millimetres; at the carrying 1500 N/m the budget was unreachable and the
    pads lagged 40 mm, which just buried them in the body.
    """
    gravity_compensation: bool = True
    pad_half_thickness: float = 0.003
    pad_half_width: float = 0.006
    pad_half_height: float = 0.006
    post_half_x: float = 0.005
    post_half_y: float = 0.006
    post_half_z: float = 0.021
    """Half the finger post's height. It runs from the pad's bottom face to the
    carrier's underside, so the pad rests flush against the post's inner face."""
    carrier_half_extents: tuple[float, float, float] = (0.060, 0.015, 0.009)
    carrier_offset_z: float = 0.045
    carrier_mass: float = 0.2
    finger_mass: float = 0.05
    pad_mu: float = 4.0
    ledge: bool = False
    carrier: bool = True
    control_hz: int = 30
    substeps: int = 20
    duration: float = 15.0

    def __post_init__(self):
        if self.mode != "dynamic" or self.ledge or not self.carrier:
            raise ValueError("the parallel claw is dynamic, has no ledges and always has a body")
        positive = (
            self.pad_half_thickness,
            self.pad_half_width,
            self.pad_half_height,
            self.post_half_x,
            self.post_half_y,
            self.post_half_z,
            self.carrier_mass,
            self.finger_mass,
            self.force_limit,
            self.push_force_limit,
            self.carrier_offset_z,
            self.control_hz,
            self.substeps,
            self.duration,
        )
        gains = (
            self.jaw_ke,
            self.jaw_kd,
            self.carrier_ke,
            self.carrier_kd,
            self.carrier_press_ke,
        )
        if not np.isfinite(positive).all() or min(positive) <= 0 or not np.isfinite(gains).all() or min(gains) < 0:
            raise ValueError("parallel claw dimensions, masses, limits and rates must be positive")
        if not 0.003 < self.minimum_half_gap <= self.half_gap <= self.maximum_half_gap:
            raise ValueError("initial half-gap must lie within the slide limits")
        if self.closed_half_gap < self.minimum_half_gap:
            raise ValueError("closed half-gap is below the mechanical slide limit")
        if self.push_force_limit < self.force_limit:
            raise ValueError("the push budget must not be below the jaw budget")
        if len(self.carrier_half_extents) != 3 or min(self.carrier_half_extents) <= 0:
            raise ValueError("carrier_half_extents must be three positive dimensions")


@rss.register_entity("parallel_loop_gripper")
class ParallelGripper(rss.Entity):
    """A level carrier with two prismatic jaws and separate grasp/push caps."""

    domain = "rigid"

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        self.center = np.asarray(self.placed((*spec.center, 0.0, 0.0, 0.0, 1.0))[:3], dtype=np.float64)
        half = np.asarray(spec.carrier_half_extents, dtype=float)
        carrier_inertia = np.diag(
            np.array((half[1] ** 2 + half[2] ** 2, half[0] ** 2 + half[2] ** 2, half[0] ** 2 + half[1] ** 2))
            * spec.carrier_mass
            / 3
        )
        cfg = newton.ModelBuilder.ShapeConfig(mu=spec.pad_mu, ke=1.0e5, kd=25.0, gap=0.001, density=0.0)
        self.guide_body = builder.add_link(
            xform=wp.transform(self.center, wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
            label=f"{spec.name}_guide",
        )
        self.carrier_body = builder.add_link(
            xform=wp.transform(self.center, wp.quat_identity()),
            com=wp.vec3(0.0, 0.0, spec.carrier_offset_z),
            mass=spec.carrier_mass,
            inertia=wp.mat33(carrier_inertia),
            lock_inertia=True,
            label=f"{spec.name}_carrier",
        )
        self.carrier_shape = builder.add_shape_box(
            self.carrier_body,
            xform=wp.transform(wp.vec3(0.0, 0.0, spec.carrier_offset_z), wp.quat_identity()),
            hx=float(half[0]),
            hy=float(half[1]),
            hz=float(half[2]),
            cfg=cfg,
            color=(0.16, 0.30, 0.39),
            label=f"{spec.name}_body",
        )
        # The finger's COM is the pad + post assembly, the post centred outboard
        # of the pad and from the pad's bottom face up to the carrier.
        pad_volume = 8.0 * spec.pad_half_thickness * spec.pad_half_width * spec.pad_half_height
        post_volume = 8.0 * spec.post_half_x * spec.post_half_y * spec.post_half_z
        post_center_z = spec.post_half_z - spec.pad_half_height
        post_center_x = spec.pad_half_thickness + spec.post_half_x
        com_z = post_volume * post_center_z / (pad_volume + post_volume)
        com_x = post_volume * post_center_x / (pad_volume + post_volume)
        finger_inertia = (
            spec.finger_mass
            / 3
            * np.array(
                (
                    spec.pad_half_width**2 + (spec.pad_half_height + spec.post_half_z) ** 2,
                    spec.pad_half_thickness**2 + (spec.pad_half_height + spec.post_half_z) ** 2,
                    spec.pad_half_thickness**2 + spec.pad_half_width**2,
                )
            )
        )
        self.joints = [
            builder.add_joint_free(self.guide_body, label=f"{spec.name}_guide_free"),
            builder.add_joint_prismatic(
                self.guide_body,
                self.carrier_body,
                axis=wp.vec3(0.0, 0.0, 1.0),
                target_pos=0.0,
                target_ke=0.0,
                target_kd=0.0,
                effort_limit=spec.push_force_limit,
                actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
                limit_lower=-0.20,
                limit_upper=0.20,
                damping=0.0,
                friction=0.0,
                label=f"{spec.name}_push",
            ),
        ]
        self.bodies, self.finger_shapes, self.shapes_by_side = [], [], []
        for sign in (-1, 1):
            body = builder.add_link(
                xform=wp.transform(
                    self.center + np.array((sign * spec.half_gap, 0.0, 0.0)), wp.quat_identity()
                ),
                com=wp.vec3(sign * com_x, 0.0, com_z),
                mass=spec.finger_mass,
                inertia=wp.mat33(np.diag(finger_inertia)),
                lock_inertia=True,
                label=f"{spec.name}_finger_{sign:+d}",
            )
            joint = builder.add_joint_prismatic(
                self.carrier_body,
                body,
                axis=wp.vec3(float(sign), 0.0, 0.0),
                target_pos=spec.half_gap,
                target_ke=0.0,
                target_kd=0.0,
                effort_limit=spec.force_limit,
                actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
                limit_lower=spec.minimum_half_gap,
                limit_upper=spec.maximum_half_gap,
                damping=0.0,
                friction=0.0,
                label=f"{spec.name}_slide_{sign:+d}",
            )
            builder.joint_q[builder.joint_q_start[joint]] = spec.half_gap
            self.joints.append(joint)
            pad = builder.add_shape_box(
                body,
                hx=spec.pad_half_thickness,
                hy=spec.pad_half_width,
                hz=spec.pad_half_height,
                cfg=cfg,
                color=(0.16, 0.30, 0.39),
                label=f"{spec.name}_pad_{sign:+d}",
            )
            post = builder.add_shape_box(
                body,
                xform=wp.transform(
                    wp.vec3(
                        sign * (spec.pad_half_thickness + spec.post_half_x),
                        0.0,
                        spec.post_half_z - spec.pad_half_height,
                    ),
                    wp.quat_identity(),
                ),
                hx=spec.post_half_x,
                hy=spec.post_half_y,
                hz=spec.post_half_z,
                cfg=cfg,
                color=(0.16, 0.30, 0.39),
                label=f"{spec.name}_post_{sign:+d}",
            )
            self.bodies.append(body)
            self.finger_shapes.extend((pad, post))
            self.shapes_by_side.append((pad, post))
        builder.add_articulation(self.joints, label=f"{spec.name}_mechanism")

    def on_scene_ready(self, pipeline) -> None:
        self.pipeline = pipeline
        self._model = self.stepping_model if self.stepping_model is not None else pipeline.model
        spec, model = self.spec, self._model
        self.dt = 1.0 / (spec.control_hz * spec.substeps)
        self.capacity = int(np.ceil(spec.duration / self.dt)) + 2
        device = model.device
        dofs = model.joint_qd_start.numpy()
        self._body_indices = wp.array(self.bodies, dtype=int, device=device)
        self._carrier_dof = int(dofs[self.joints[1]])
        self._finger_dofs = wp.array(dofs[self.joints[2:]], dtype=int, device=device)
        self._poses = wp.zeros((self.capacity, 2), dtype=wp.transform, device=device)
        self._velocities = wp.zeros((self.capacity, 2), dtype=wp.spatial_vector, device=device)
        self._lengths = wp.ones(1, dtype=int, device=device)
        self._tick = wp.zeros(1, dtype=int, device=device)
        self._pressing = wp.zeros(1, dtype=int, device=device)
        self.applied_wrenches = wp.zeros(2, dtype=wp.spatial_vector, device=device)
        self.applied_carrier = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        self.peak_force = wp.zeros(2, dtype=float, device=device)
        self.peak_torque = wp.zeros(2, dtype=float, device=device)
        self.peak_push_force = wp.zeros(1, dtype=float, device=device)
        self.peak_push_torque = wp.zeros(1, dtype=float, device=device)
        self.actual_half_gaps = wp.zeros(2, dtype=float, device=device)
        self.actual_gap_rates = wp.zeros(2, dtype=float, device=device)
        self.slide_lateral_error = wp.zeros(2, dtype=float, device=device)
        self.slide_angular_error = wp.zeros(2, dtype=float, device=device)

    def set_trajectory(self, poses: np.ndarray) -> None:
        """Replace the finger targets with [N, 2, 7] world poses at the physics cadence."""
        poses = np.asarray(poses, dtype=np.float32)
        if poses.ndim != 3 or poses.shape[1:] != (2, 7) or not 1 <= len(poses) <= self.capacity:
            raise ValueError(f"trajectory needs 1 to {self.capacity} poses of shape (2, 7)")
        norms = np.linalg.norm(poses[..., 3:], axis=-1, keepdims=True)
        if not np.isfinite(poses).all() or np.any(norms < 1.0e-8):
            raise ValueError("trajectory poses must be finite with nonzero quaternions")
        poses = poses.copy()
        poses[..., 3:] /= norms
        pairs = poses.reshape((-1, 2, 7))
        qa, qb = pairs[:, 0, 3:], pairs[:, 1, 3:]
        sign = np.where(np.sum(qa * qb, axis=1) < 0, -1.0, 1.0)
        if np.any(np.linalg.norm(qa - qb * sign[:, None], axis=1) > 1.0e-4):
            raise ValueError("parallel targets must share one orientation")
        x, y, z, w = qa.T
        axes = np.stack((1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)), axis=1)
        delta = pairs[:, 1, :3] - pairs[:, 0, :3]
        gap = np.sum(delta * axes, axis=1) / 2
        if np.any(np.linalg.norm(delta - 2 * gap[:, None] * axes, axis=1) > 1.0e-5):
            raise ValueError("finger separation must be parallel to the common local x")
        if np.any(gap < self.spec.minimum_half_gap - 1.0e-6) or np.any(gap > self.spec.maximum_half_gap + 1.0e-6):
            raise ValueError("target half-gap lies outside the mechanical slide limits")
        device = self._model.device
        source = wp.array(poses, dtype=wp.transform, device=device)
        wp.copy(self._poses, source, count=source.size)
        rates = wp.array(trajectory_velocities(poses, self.dt), dtype=wp.spatial_vector, device=device)
        wp.copy(self._velocities, rates, count=rates.size)
        self._lengths.assign(np.array([len(poses)], dtype=np.int32))
        self._tick.zero_()

    def set_press(self, pressing: bool) -> None:
        """Mark the push and reset its peaks; the carrier keeps the push budget."""
        self._pressing.assign(np.array([int(pressing)], dtype=np.int32))
        if pressing:
            self.peak_push_force.zero_()
            self.peak_push_torque.zero_()

    def pressing(self) -> bool:
        return bool(self._pressing.numpy()[0])

    def contribute_hooks(self, hooks) -> None:
        hooks.add("rigid_force", self._drive, name="drive-parallel-carrier-and-jaws")
        hooks.add("post_solve", self._measure, name="measure-and-advance-parallel-jaws")

    def _drive(self, state, dt) -> None:
        del dt
        spec, model = self.spec, self._model
        wp.launch(
            _set_guide,
            dim=1,
            inputs=[
                self._tick,
                self._lengths,
                self._poses,
                self._velocities,
                self.guide_body,
            ],
            outputs=[state.body_q, state.body_qd],
            device=model.device,
        )
        wp.launch(
            _drive_carrier,
            dim=1,
            inputs=[
                self._carrier_dof,
                self.carrier_body,
                self.guide_body,
                self._body_indices,
                state.body_q,
                state.body_qd,
                model.body_com,
                model.body_mass,
                model.body_world,
                model.gravity,
                spec.carrier_ke,
                spec.carrier_kd,
                spec.carrier_press_ke,
                spec.push_force_limit,
                spec.gravity_compensation,
                self._pressing,
            ],
            outputs=[
                self.pipeline.control.joint_f,
                self.applied_carrier,
                self.peak_push_force,
                self.peak_push_torque,
            ],
            device=model.device,
        )
        wp.launch(
            _drive_slides,
            dim=2,
            inputs=[
                self._tick,
                self._lengths,
                self._poses,
                self._velocities,
                self.carrier_body,
                self._body_indices,
                self._finger_dofs,
                state.body_q,
                state.body_qd,
                model.body_com,
                spec.jaw_ke,
                spec.jaw_kd,
                spec.force_limit,
            ],
            outputs=[
                self.pipeline.control.joint_f,
                self.applied_wrenches,
                self.peak_force,
            ],
            device=model.device,
        )

    def _measure(self, state, dt) -> None:
        del dt
        wp.launch(
            _measure_slides,
            dim=2,
            inputs=[
                self.carrier_body,
                self._body_indices,
                state.body_q,
                state.body_qd,
                self._model.body_com,
            ],
            outputs=[
                self.actual_half_gaps,
                self.actual_gap_rates,
                self.slide_lateral_error,
                self.slide_angular_error,
            ],
            device=self._model.device,
        )
        wp.launch(
            _advance_tick,
            dim=1,
            inputs=[self._tick, self._lengths],
            device=self._model.device,
        )

    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        del pipeline, episode, world
        if not hasattr(self, "_tick"):
            return
        self._tick.zero_()
        self._pressing.zero_()
        self.applied_wrenches.zero_()
        self.applied_carrier.zero_()
        self.peak_force.zero_()
        self.peak_torque.zero_()
        self.peak_push_force.zero_()
        self.peak_push_torque.zero_()
