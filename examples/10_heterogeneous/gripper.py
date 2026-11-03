# SPDX-License-Identifier: Apache-2.0
"""A two-finger gripper on RSS substep hooks.

``mode="kinematic"`` (the default) prescribes finger poses, as it always has:
the jaws have unlimited authority and the contact force is whatever the
collision needs. ``mode="dynamic"`` gives each finger its own mass and inertia
and drives it with a clamped PD wrench, so the jaw effort a scene reports is
bounded by ``force_limit`` and ``torque_limit``, with gravity compensation
inside the same budget. That is what turns overcompression into a measurable
choice instead of a consequence of prescribed motion. A claw that also presses
has a second budget: ``set_press(True)`` swaps the drive to ``push_force_limit``
and its torque limit, so closing stays a grasp and pressing can push harder.
The choice is the claw's own state; no scene passes a limit per call.
"""

from __future__ import annotations

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

import robosimstudio as rss


def smooth(time: float, start: float, end: float) -> float:
    t = float(np.clip((time - start) / (end - start), 0.0, 1.0))
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def trajectory_velocities(poses: np.ndarray, dt: float) -> np.ndarray:
    """World-frame linear/angular rates for a substep-sampled [N, 2, 7] path."""
    velocities = np.zeros((*poses.shape[:2], 6), dtype=np.float32)
    if len(poses) < 2:
        return velocities
    velocities[:-1, :, :3] = np.diff(poses[:, :, :3], axis=0) / dt
    a, b = poses[1:, :, 3:], poses[:-1, :, 3:]
    # q_next * inverse(q_current): rotation increments in world coordinates.
    xyz = -a[..., 3, None] * b[..., :3] + b[..., 3, None] * a[..., :3] - np.cross(a[..., :3], b[..., :3])
    scalar = np.sum(a * b, axis=-1)
    sign = np.where(scalar < 0, -1.0, 1.0)
    xyz *= sign[..., None]
    scalar *= sign
    magnitude = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(magnitude, np.clip(scalar, 0.0, 1.0))
    scale = np.divide(angle, magnitude * dt, out=np.zeros_like(angle), where=magnitude > 1.0e-10)
    velocities[:-1, :, 3:] = xyz * scale[..., None]
    return velocities


@wp.kernel
def _drive_fingers(
    tick: wp.array[int],
    lengths: wp.array[int],
    bodies: wp.array[int],
    poses: wp.array2d[wp.transform],
    velocities: wp.array2d[wp.spatial_vector],
    q0: wp.array[wp.transform],
    q1: wp.array[wp.transform],
    qd0: wp.array[wp.spatial_vector],
    qd1: wp.array[wp.spatial_vector],
):
    finger = wp.tid()
    body = bodies[finger]
    frame = wp.min(tick[0], lengths[0] - 1)
    q0[body] = poses[frame, finger]
    q1[body] = poses[frame, finger]
    qd0[body] = velocities[frame, finger]
    qd1[body] = velocities[frame, finger]


@wp.kernel
def _drive_dynamic(
    tick: wp.array[int],
    lengths: wp.array[int],
    bodies: wp.array[int],
    poses: wp.array2d[wp.transform],
    velocities: wp.array2d[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_world: wp.array[int],
    gravity: wp.array[wp.vec3],
    linear_ke: float,
    linear_kd: float,
    angular_ke: float,
    angular_kd: float,
    force_limit: float,
    torque_limit: float,
    compensate_gravity: bool,
    pressing: wp.array[int],
    push_force_limit: float,
    push_torque_limit: float,
    push_angular_ke: float,
    push_angular_kd: float,
    body_f: wp.array[wp.spatial_vector],
    applied: wp.array[wp.spatial_vector],
    peak_force: wp.array[float],
    peak_torque: wp.array[float],
    peak_push_force: wp.array[float],
    peak_push_torque: wp.array[float],
):
    finger = wp.tid()
    body = bodies[finger]
    limit = force_limit
    tlimit = torque_limit
    ake = angular_ke
    akd = angular_kd
    if pressing[0] != 0:
        limit = push_force_limit
        tlimit = push_torque_limit
        ake = push_angular_ke
        akd = push_angular_kd
    frame = wp.min(tick[0], lengths[0] - 1)
    target = poses[frame, finger]
    current = body_q[body]
    rotation = wp.transform_get_rotation(current)
    target_rotation = wp.transform_get_rotation(target)
    com = wp.transform_point(current, body_com[body])
    target_com = wp.transform_point(target, body_com[body])
    target_rate = velocities[frame, finger]
    velocity = body_qd[body]
    target_angular = wp.spatial_bottom(target_rate)
    target_linear = wp.spatial_top(target_rate) + wp.cross(
        target_angular, wp.quat_rotate(target_rotation, body_com[body])
    )
    force = linear_ke * (target_com - com) + linear_kd * (target_linear - wp.spatial_top(velocity))
    if compensate_gravity:
        force -= body_mass[body] * gravity[body_world[body]]
    # Clamp after all actuator contributions, including only this finger's own
    # gravity compensation. Payload weight consumes the same finite budget.
    force_norm = wp.length(force)
    if force_norm > limit:
        force *= limit / force_norm
    error = target_rotation * wp.quat_inverse(rotation)
    if error[3] < 0.0:
        error = wp.quat(-error[0], -error[1], -error[2], -error[3])
    axis = wp.vec3(error[0], error[1], error[2])
    sine = wp.length(axis)
    rotation_vector = wp.vec3(0.0)
    if sine > 1.0e-8:
        rotation_vector = axis * (2.0 * wp.atan2(sine, error[3]) / sine)
    torque = ake * rotation_vector + akd * (target_angular - wp.spatial_bottom(velocity))
    torque_norm = wp.length(torque)
    if torque_norm > tlimit:
        torque *= tlimit / torque_norm
    wrench = wp.spatial_vector(force, torque)
    wp.atomic_add(body_f, body, wrench)
    applied[finger] = wrench
    peak_force[finger] = wp.max(peak_force[finger], wp.length(force))
    peak_torque[finger] = wp.max(peak_torque[finger], wp.length(torque))
    if pressing[0] != 0:
        peak_push_force[finger] = wp.max(peak_push_force[finger], wp.length(force))
        peak_push_torque[finger] = wp.max(peak_push_torque[finger], wp.length(torque))


@wp.kernel
def _place_carrier(
    fingers: wp.array[int],
    carrier: int,
    offset: float,
    q0: wp.array[wp.transform],
    q1: wp.array[wp.transform],
):
    """Keep the carrier centred over the fingers, wherever the drive has them."""
    left = q0[fingers[0]]
    right = q0[fingers[1]]
    midpoint = (wp.transform_get_translation(left) + wp.transform_get_translation(right)) * 0.5
    pose = wp.transform(midpoint + wp.vec3(0.0, 0.0, offset), wp.quat_identity())
    q0[carrier] = pose
    q1[carrier] = pose


@wp.kernel
def _advance_tick(tick: wp.array[int], lengths: wp.array[int]):
    tick[0] = wp.min(tick[0] + 1, lengths[0] - 1)


@dataclass(frozen=True, kw_only=True)
class ScriptedGripperSpec:
    entity: str = "scripted_loop_gripper"
    name: str = "loop_gripper"
    mode: str = "kinematic"
    finger_mass: float = 0.05
    force_limit: float = 10.0
    """Closing clamp. A grasp squeezes with this and never more."""
    push_force_limit: float = 50.0
    """Press clamp. ``set_press`` swaps the drive onto it, so a press can push
    harder than the jaws may close. One owner: the claw's press state.
    """
    push_torque_limit: float = 0.5
    push_angular_ke: float = 0.5
    push_angular_kd: float = 0.005
    torque_limit: float = 0.03
    linear_ke: float = 1500.0
    linear_kd: float = 15.0
    angular_ke: float = 0.03
    angular_kd: float = 0.0005
    gravity_compensation: bool = True
    pad_half_width: float = 0.020
    pad_half_thickness: float = 0.003
    """Half-extent along the closing axis: the pad face sits this far inboard.

    A scene that states its closing gap from a measured width has to know it.
    """
    ledge: bool = True
    carrier: bool = False
    """A rectangular wrist block above the fingers, as the archived parallel
    gripper has: the claw's body, not a third drive axis. It is kinematic and
    follows the fingers' midpoint.
    """
    carrier_half_extents: tuple[float, float, float] = (0.060, 0.015, 0.009)
    carrier_offset_z: float = 0.045
    """The carrier's bottom sits ``offset - hz`` above the finger centres: 36 mm
    for the archived 120 x 30 x 18 mm block, clear of the loop's top at 31.5 mm.
    """
    center: tuple[float, float, float] = (0.0, 0.0, 0.02)
    half_gap: float = 0.015
    closed_half_gap: float = 0.005
    release_half_gap: float = 0.070
    approach_z: float = -0.030
    travel_x: float = 0.18
    lift_z: float = 0.40
    lower_offset_z: float = 0.295
    release_lower_z: float = 0.17
    control_hz: int = 30
    substeps: int = 20
    duration: float = 13.0

    def __post_init__(self):
        if self.mode not in ("dynamic", "kinematic"):
            raise ValueError("gripper mode must be dynamic or kinematic")
        positive = (
            self.finger_mass,
            self.force_limit,
            self.push_force_limit,
            self.push_torque_limit,
            self.torque_limit,
            self.control_hz,
            self.substeps,
            self.duration,
            self.pad_half_width,
            self.pad_half_thickness,
        )
        gains = (
            self.linear_ke,
            self.linear_kd,
            self.angular_ke,
            self.angular_kd,
            self.push_angular_ke,
            self.push_angular_kd,
        )
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("gripper mass, limits, rates and duration must be finite and positive")
        if not all(np.isfinite(v) and v >= 0 for v in gains):
            raise ValueError("gripper gains must be finite and nonnegative")
        if self.push_force_limit < self.force_limit:
            raise ValueError("push limit must not be below the closing limit")
        if self.carrier and (
            len(self.carrier_half_extents) != 3
            or not np.isfinite(self.carrier_half_extents).all()
            or min(self.carrier_half_extents) <= 0
            or not np.isfinite(self.carrier_offset_z)
            or self.carrier_offset_z <= 0
        ):
            raise ValueError("carrier needs three positive half extents and a positive offset")


@rss.register_entity("scripted_loop_gripper")
class ScriptedGripper(rss.Entity):
    domain = "rigid"

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        center = self.placed((*spec.center, 0.0, 0.0, 0.0, 1.0))[:3]
        self.center = np.asarray(center, dtype=np.float64)
        cfg = newton.ModelBuilder.ShapeConfig(mu=2.0, ke=1.0e5, kd=25.0, gap=0.001, density=0.0)
        self.bodies = []
        self.finger_shapes = []
        self.shapes_by_side = []
        if spec.carrier:
            half = np.asarray(spec.carrier_half_extents, dtype=float)
            self.carrier_mass = 0.2
            inertia = np.diag(
                np.array((half[1] ** 2 + half[2] ** 2, half[0] ** 2 + half[2] ** 2, half[0] ** 2 + half[1] ** 2))
                * self.carrier_mass
                / 3
            )
            self.carrier_body = builder.add_body(
                xform=wp.transform(
                    self.center + np.array([0.0, 0.0, spec.carrier_offset_z]),
                    wp.quat_identity(),
                ),
                mass=self.carrier_mass,
                inertia=wp.mat33(inertia),
                lock_inertia=True,
                is_kinematic=True,
                label=f"{spec.name}_carrier",
            )
            self.carrier_shape = builder.add_shape_box(
                self.carrier_body,
                hx=float(half[0]),
                hy=float(half[1]),
                hz=float(half[2]),
                cfg=cfg,
                color=(0.16, 0.30, 0.39),
                label=f"{spec.name}_carrier_bar",
            )
        # Explicit solid-finger envelope inertia: the pads are collision shapes,
        # not the body's mass, so a dynamic finger is exactly as heavy as the
        # mass the drive's gravity compensation and clamp talk about.
        hx, hy, hz = (
            (0.006 if spec.ledge else spec.pad_half_thickness),
            spec.pad_half_width,
            (0.008 if spec.ledge else 0.006),
        )
        inertia = np.diag(np.array((hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy)) * spec.finger_mass / 3)
        for side in (-1, 1):
            pos = self.center + np.array([side * spec.half_gap, 0.0, 0.0])
            body = builder.add_body(
                xform=wp.transform(pos, wp.quat_identity()),
                mass=spec.finger_mass,
                inertia=wp.mat33(inertia),
                lock_inertia=True,
                is_kinematic=spec.mode == "kinematic",
                label=f"{spec.name}_{side:+d}",
            )
            shape = builder.add_shape_box(
                body,
                hx=spec.pad_half_thickness,
                hy=spec.pad_half_width,
                hz=0.006,
                cfg=cfg,
                color=(0.16, 0.30, 0.39),
                label=f"{spec.name}_{side:+d}_pad",
            )
            self.bodies.append(body)
            self.finger_shapes.append(shape)
            side_shapes = [shape]
            if spec.ledge:
                # A shallow ledge slides under the flat loop as the jaws close.
                # It carries the loop during lift through ordinary rigid-soft
                # contact, without welding particles to either finger.
                ledge = builder.add_shape_box(
                    body,
                    xform=wp.transform(wp.vec3(0.0, 0.0, -0.0068), wp.quat_identity()),
                    hx=0.006,
                    hy=spec.pad_half_width,
                    hz=0.0012,
                    cfg=cfg,
                    color=(0.16, 0.30, 0.39),
                    label=f"{spec.name}_{side:+d}_ledge",
                )
                self.finger_shapes.append(ledge)
                side_shapes.append(ledge)
            self.shapes_by_side.append(tuple(side_shapes))

    def on_scene_ready(self, pipeline) -> None:
        self.pipeline = pipeline
        self._model = self.stepping_model if self.stepping_model is not None else pipeline.model
        spec = self.spec
        count = int(np.ceil(spec.duration * spec.control_hz * spec.substeps)) + 2
        dt = 1.0 / (spec.control_hz * spec.substeps)
        self.dt = dt
        self.capacity = count
        poses = np.zeros((count, 2, 7), dtype=np.float32)
        poses[:, :, 6] = 1.0
        for index in range(count):
            time = index * dt
            close = smooth(time, 0.40, 0.85)
            grasp_gap = spec.half_gap + (spec.closed_half_gap - spec.half_gap) * close
            left_gap = grasp_gap + (spec.release_half_gap - grasp_gap) * smooth(time, 8.0, 8.5)
            right_gap = grasp_gap + (spec.release_half_gap - grasp_gap) * smooth(time, 10.2, 10.7)
            approach = spec.approach_z * smooth(time, 0.05, 0.35)
            lower = smooth(time, 5.6, 7.2)
            lift = spec.lift_z * smooth(time, 1.2, 2.8) * (1.0 - lower)
            move = spec.travel_x * smooth(time, 3.0, 4.8)
            retreat = smooth(time, 11.3, 12.1)
            release_lower = spec.release_lower_z * smooth(time, 8.5, 9.4)
            center = self.center + np.array(
                [move, -0.12 * retreat, approach + lift + spec.lower_offset_z * lower - release_lower + 0.08 * retreat]
            )
            poses[index, 0, :3] = center + np.array([-left_gap, 0.0, 0.0])
            poses[index, 1, :3] = center + np.array([right_gap, 0.0, 0.0])
            # Once the left jaw is clear, tip the supporting right ledge
            # downward so the loop slides off before that jaw withdraws.
            tip = smooth(time, 9.4, 10.2) * (1.0 - smooth(time, 10.7, 11.2))
            angle = -0.5 * np.pi * tip
            poses[index, 1, 4] = np.sin(angle / 2.0)
            poses[index, 1, 6] = np.cos(angle / 2.0)
        device = self._model.device
        velocities = trajectory_velocities(poses, dt)
        self._poses = wp.array(poses, dtype=wp.transform, device=device)
        self._velocities = wp.array(velocities, dtype=wp.spatial_vector, device=device)
        self._body_indices = wp.array(self.bodies, dtype=int, device=device)
        self._lengths = wp.full(1, count, dtype=int, device=device)
        self._tick = wp.zeros(1, dtype=int, device=device)
        self.applied_wrenches = wp.zeros(2, dtype=wp.spatial_vector, device=device)
        self.peak_force = wp.zeros(2, dtype=float, device=device)
        self.peak_torque = wp.zeros(2, dtype=float, device=device)
        self.peak_push_force = wp.zeros(2, dtype=float, device=device)
        self.peak_push_torque = wp.zeros(2, dtype=float, device=device)
        self._pressing = wp.zeros(1, dtype=int, device=device)

    def set_press(self, pressing: bool) -> None:
        """Select the press clamp instead of the closing clamp, or back.

        A press is the claw driving the object down, not closing on it, so it
        uses the push budget; the closing clamp is untouched and still bounds
        every grasp.
        """
        self._pressing.assign(np.array([int(pressing)], dtype=np.int32))
        if pressing:
            self.peak_push_force.zero_()
            self.peak_push_torque.zero_()

    def pressing(self) -> bool:
        return bool(self._pressing.numpy()[0]) if hasattr(self, "_pressing") else False

    def set_trajectory(self, poses: np.ndarray) -> None:
        """Replace the finger targets with [N, 2, 7] poses at the physics cadence.

        The final sample is held. Velocities are world-frame rates of the same
        path, so a scene that plans its own waypoints still prescribes motion,
        never force: in dynamic mode the drive's clamp decides what is applied.
        """
        poses = np.asarray(poses, dtype=np.float32)
        if poses.ndim != 3 or poses.shape[1:] != (2, 7) or not 1 <= len(poses) <= self.capacity:
            raise ValueError(f"trajectory needs 1 to {self.capacity} poses of shape (2, 7)")
        norms = np.linalg.norm(poses[..., 3:], axis=-1, keepdims=True)
        if not np.isfinite(poses).all() or np.any(norms < 1.0e-8):
            raise ValueError("trajectory poses must be finite with nonzero quaternions")
        poses = poses.copy()
        poses[..., 3:] /= norms
        source = wp.array(poses, dtype=wp.transform, device=self._model.device)
        wp.copy(self._poses, source, count=source.size)
        rates = wp.array(trajectory_velocities(poses, self.dt), dtype=wp.spatial_vector, device=self._model.device)
        wp.copy(self._velocities, rates, count=rates.size)
        self._lengths.assign(np.array([len(poses)], dtype=np.int32))
        self._tick.zero_()

    def contribute_hooks(self, hooks) -> None:
        phase = "rigid_force" if self.spec.mode == "dynamic" else "pre_solve"
        hooks.add(phase, self._drive, name="drive-loop-fingers")
        hooks.add("post_solve", self._advance, name="advance-loop-fingers")

    def _drive(self, state, dt) -> None:
        del dt
        spec = self.spec
        if spec.mode == "dynamic":
            model = self._model
            wp.launch(
                _drive_dynamic,
                dim=2,
                inputs=[
                    self._tick,
                    self._lengths,
                    self._body_indices,
                    self._poses,
                    self._velocities,
                    state.body_q,
                    state.body_qd,
                    model.body_com,
                    model.body_mass,
                    model.body_world,
                    model.gravity,
                    spec.linear_ke,
                    spec.linear_kd,
                    spec.angular_ke,
                    spec.angular_kd,
                    spec.force_limit,
                    spec.torque_limit,
                    spec.gravity_compensation,
                    self._pressing,
                    spec.push_force_limit,
                    spec.push_torque_limit,
                    spec.push_angular_ke,
                    spec.push_angular_kd,
                ],
                outputs=[
                    state.body_f,
                    self.applied_wrenches,
                    self.peak_force,
                    self.peak_torque,
                    self.peak_push_force,
                    self.peak_push_torque,
                ],
                device=model.device,
            )
        else:
            wp.launch(
                _drive_fingers,
                dim=2,
                inputs=[
                    self._tick,
                    self._lengths,
                    self._body_indices,
                    self._poses,
                    self._velocities,
                    state.body_q,
                    self.pipeline.state_1.body_q,
                    state.body_qd,
                    self.pipeline.state_1.body_qd,
                ],
            )
        if spec.carrier:
            wp.launch(
                _place_carrier,
                dim=1,
                inputs=[
                    self._body_indices,
                    self.carrier_body,
                    spec.carrier_offset_z,
                    state.body_q,
                    self.pipeline.state_1.body_q,
                ],
                device=self._model.device,
            )

    def _advance(self, state, dt) -> None:
        del state, dt
        wp.launch(_advance_tick, dim=1, inputs=[self._tick, self._lengths])

    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        del pipeline, episode, world
        if hasattr(self, "_tick"):
            self._tick.zero_()
            self.applied_wrenches.zero_()
            self.peak_force.zero_()
            self.peak_torque.zero_()
            self.peak_push_force.zero_()
            self.peak_push_torque.zero_()
            self._pressing.zero_()
