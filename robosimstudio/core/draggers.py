# SPDX-License-Identifier: Apache-2.0
"""Dragging things around: one rigid body, or a patch of a deformable.

Both draggers exist because Newton's :class:`newton.Picking` only has mouse-ray
plumbing in the OpenGL viewer (``ViewerViser.apply_forces`` is a documented
no-op), so the target has to come from somewhere else -- a viser gizmo, a
scripted routine, or a graph node. The force laws are the same idea: stiffness
towards a target, damping against velocity, and an acceleration clamp so a light
object is not flung across the room.

Both are also written for capture: which body is dragged, how many particles are
held, and where they are going all live in *device* arrays, and the launch
happens unconditionally with the kernel returning early on an inactive handle.
Skipping the launch instead would leave it out of the captured graph, so
attaching mid-run would do nothing at all.
"""

from __future__ import annotations

from collections.abc import Sequence

import newton
import numpy as np
import warp as wp

from .math import quat_to_mat

__all__ = ["ObjectDragger", "ParticleDragger"]


@wp.kernel
def _drag_wrench_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33],
    drag_body: wp.array[wp.int32],
    drag_target: wp.array[wp.transform],
    drag_mass: wp.array[wp.float32],
    lin_stiffness: float,
    ang_stiffness: float,
    damping_ratio: float,
    max_accel: float,
    body_f: wp.array[wp.spatial_vector],
):
    """Add a clamped PD wrench pulling one body towards a target pose.

    Follows Newton's conventions: ``body_qd`` is (linear, angular) with the
    linear part at the centre of mass, and ``body_f`` is (force, torque) in
    world frame about that same point.

    ``drag_mass`` is the load the clamp and the damping are sized against:
    zero means the body's own mass, and a caller dragging one link of a chain
    passes the chain's total instead -- a cable segment weighs a third of a
    gram, so 20 g of *its* acceleration cannot even hold the rod it is chained
    to, and a handle sized to the segment stalls a hand's width off the table.
    """
    body_index = drag_body[0]
    if body_index < 0:
        return

    target_tf = drag_target[0]
    target_pos = wp.transform_get_translation(target_tf)
    target_rot = wp.transform_get_rotation(target_tf)

    tf = body_q[body_index]
    rot = wp.transform_get_rotation(tf)
    com_local = body_com[body_index]
    com_world = wp.transform_point(tf, com_local)

    qd = body_qd[body_index]
    lin_vel = wp.spatial_top(qd)
    ang_vel = wp.spatial_bottom(qd)

    # Where the COM would sit with the body already at the target. Driving the
    # COM there rather than the body origin keeps the linear spring from
    # inducing a torque on an off-centre body, leaving orientation entirely to
    # the angular term below.
    com_target = target_pos + wp.quat_rotate(target_rot, com_local)

    # Critical damping c = 2*zeta*sqrt(k*m), computed here rather than passed
    # in: the picked body's mass is only known once the gizmo is attached,
    # which is after the graph has been captured.
    mass = wp.max(body_mass[body_index], drag_mass[0])
    lin_damping = 2.0 * damping_ratio * wp.sqrt(wp.max(lin_stiffness * mass, 1.0e-9))
    ang_damping = 0.2 * damping_ratio * wp.sqrt(wp.max(ang_stiffness * mass, 1.0e-9))

    force = (com_target - com_world) * lin_stiffness - lin_vel * lin_damping

    if mass > 0.0:
        max_force = mass * max_accel
        mag = wp.length(force)
        if mag > max_force and mag > 0.0:
            force = force * (max_force / mag)

    # Shortest-arc orientation error, as a rotation vector.
    q_err = target_rot * wp.quat_inverse(rot)
    if q_err[3] < 0.0:
        q_err = wp.quat(-q_err[0], -q_err[1], -q_err[2], -q_err[3])
    axis = wp.vec3(q_err[0], q_err[1], q_err[2])
    sin_half = wp.length(axis)
    rot_vec = wp.vec3(0.0, 0.0, 0.0)
    if sin_half > 1.0e-8:
        rot_vec = (axis / sin_half) * (2.0 * wp.atan2(sin_half, q_err[3]))

    torque = rot_vec * ang_stiffness - ang_vel * ang_damping
    # Bound angular acceleration as the linear term is bounded, using the
    # smallest principal inertia -- the easiest axis to spin.
    inertia = body_inertia[body_index]
    min_inertia = wp.min(inertia[0, 0], wp.min(inertia[1, 1], inertia[2, 2]))
    max_torque = min_inertia * max_accel
    tmag = wp.length(torque)
    if max_torque > 0.0 and tmag > max_torque and tmag > 0.0:
        torque = torque * (max_torque / tmag)

    wp.atomic_add(body_f, body_index, wp.spatial_vector(force, torque))


@wp.kernel
def _particle_drag_kernel(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    grabbed: wp.array[wp.int32],
    grab_offset: wp.array[wp.vec3],
    n_grabbed: wp.array[wp.int32],
    target: wp.array[wp.transform],
    stiffness: float,
    damping_ratio: float,
    max_accel: float,
    particle_f: wp.array[wp.vec3],
):
    """Spring a handful of particles towards a moving grip frame.

    Each grabbed particle keeps the offset it had when it was seized, so the
    patch is carried with its shape intact; pulling every particle to one point
    balls the cloth up instead.
    """
    i = wp.tid()
    if i >= n_grabbed[0]:
        return
    p = grabbed[i]

    goal = wp.transform_point(target[0], grab_offset[i])

    mass = particle_mass[p]
    if mass <= 0.0:
        return
    damping = 2.0 * damping_ratio * wp.sqrt(wp.max(stiffness * mass, 1.0e-9))
    force = (goal - particle_q[p]) * stiffness - particle_qd[p] * damping

    max_force = mass * max_accel
    mag = wp.length(force)
    if mag > max_force and mag > 0.0:
        force = force * (max_force / mag)

    wp.atomic_add(particle_f, p, force)


@wp.kernel
def _particle_pin_kernel(
    grabbed: wp.array[wp.int32],
    grab_offset: wp.array[wp.vec3],
    n_grabbed: wp.array[wp.int32],
    target: wp.array[wp.transform],
    max_step: float,
    inv_dt: float,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    """Place held particles where the grip frame says, with a matching velocity.

    A *kinematic* hold, not a spring, and that distinction is the whole reason
    an adhesive peels: a spring saturates at whatever its stiffness and
    acceleration clamp allow, so bonds stretch to a fixed equilibrium and never
    reach their break threshold. A pin applies whatever force it takes, so the
    bonds stretch until they let go.

    The velocity write matters as much as the position write. The pin lands
    after the solve, so the next substep's inertia target is built from it: set
    to ``(new - old)/dt`` the solve is told the particle is *moving* and carries
    it on; zeroed, the solve reads a particle at rest, the membrane's pull-back
    wins, and the carry stalls under load.
    """
    i = wp.tid()
    if i >= n_grabbed[0]:
        return
    p = grabbed[i]
    goal = wp.transform_point(target[0], grab_offset[i])

    # Rate-limit the pin. Teleported straight to the goal, every bond in a
    # strip exceeds its break threshold on the same substep and the film rips
    # off whole; advancing at most max_step lets the bonds nearest the grip
    # load first, which is what makes a peel front travel.
    if max_step > 0.0:
        delta = goal - particle_q[p]
        dist = wp.length(delta)
        if dist > max_step:
            goal = particle_q[p] + delta * (max_step / dist)

    particle_qd[p] = (goal - particle_q[p]) * inv_dt
    particle_q[p] = goal


class ObjectDragger:
    """Pull one rigid body towards a target pose with a clamped PD wrench.

    The dragged body stays fully dynamic -- it collides with bins, tools and
    the gripper the whole time -- which is the point: dragging a tool into the
    jaws is a legitimate way to set up a grasp, and dragging one the robot is
    already holding tests the grip.

    Args:
        model: The finalized model.
        lin_stiffness: Translational spring constant [N/m].
        ang_stiffness: Rotational spring constant [N·m/rad].
        damping_ratio: 1.0 is critically damped; below that the object
            overshoots and oscillates around the handle.
        max_accel: Acceleration ceiling in multiples of g, mirroring
            :class:`newton.Picking`'s ``pick_max_acceleration``.
    """

    def __init__(
        self,
        model: newton.Model,
        *,
        lin_stiffness: float = 400.0,
        ang_stiffness: float = 8.0,
        damping_ratio: float = 1.0,
        max_accel: float = 20.0,
    ) -> None:
        self.model = model
        self.lin_stiffness = float(lin_stiffness)
        self.ang_stiffness = float(ang_stiffness)
        self.damping_ratio = float(damping_ratio)
        self.max_accel = float(max_accel) * 9.81

        self._body = wp.array([-1], dtype=wp.int32, device=model.device)
        self._target = wp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=wp.transform, device=model.device)
        self._mass = wp.zeros(1, dtype=wp.float32, device=model.device)
        self._body_host = np.array([-1], dtype=np.int32)
        self._target_host = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self._mass_host = np.zeros(1, dtype=np.float32)

    @property
    def active(self) -> bool:
        return int(self._body_host[0]) >= 0

    @property
    def body_index(self) -> int:
        """The body being dragged, or -1."""
        return int(self._body_host[0])

    def attach(
        self,
        body_index: int,
        position: Sequence[float],
        quat_xyzw: Sequence[float],
        *,
        effective_mass: float | None = None,
    ) -> None:
        """Take hold of one body.

        Args:
            body_index: The model body to pull.
            position: Where the handle starts [m].
            quat_xyzw: Its orientation.
            effective_mass: The load the wrench's clamp and damping are sized
                against [kg], for a body that carries more than itself -- one
                segment of a cable drags the whole chain. ``None`` sizes them
                to the body's own mass, which is right for everything else.
        """
        self._body_host[0] = int(body_index)
        self._body.assign(self._body_host)
        self._mass_host[0] = float(effective_mass) if effective_mass is not None else 0.0
        self._mass.assign(self._mass_host)
        self.set_target(position, quat_xyzw)

    def detach(self) -> None:
        self._body_host[0] = -1
        self._body.assign(self._body_host)
        self._mass_host[0] = 0.0
        self._mass.assign(self._mass_host)

    def set_target(self, position: Sequence[float], quat_xyzw: Sequence[float]) -> None:
        self._target_host[0, :3] = position
        self._target_host[0, 3:] = quat_xyzw
        self._target.assign(self._target_host)

    def apply(self, state: newton.State, dt: float | None = None) -> None:
        """Add the drag wrench to ``state.body_f``. A :class:`SubstepHooks` hook."""
        del dt  # a wrench does not need the timestep
        wp.launch(
            _drag_wrench_kernel,
            dim=1,
            inputs=[
                state.body_q,
                state.body_qd,
                self.model.body_com,
                self.model.body_mass,
                self.model.body_inertia,
                self._body,
                self._target,
                self._mass,
                self.lin_stiffness,
                self.ang_stiffness,
                self.damping_ratio,
                self.max_accel,
            ],
            outputs=[state.body_f],
            device=self.model.device,
        )


class ParticleDragger:
    """Grab a patch of a deformable and carry it.

    :class:`ObjectDragger` has nothing to attach to on a cloth -- there is no
    body, only particles -- so this seizes the particles within a radius of a
    grip point and either springs each towards where the grip frame says it
    should be, or pins it there kinematically.

    Args:
        model: The finalized model.
        radius: How far from the grip point to seize particles [m].
        max_particles: Ceiling on the patch size, so a grab in a dense region
            cannot cost an unbounded amount of work per substep. Also the
            launch dimension, so it is fixed for the life of a captured graph.
        stiffness: Spring constant per particle [N/m], ``mode="spring"`` only.
        damping_ratio: 1.0 is critically damped.
        max_accel: Acceleration ceiling in multiples of g.
        mode: ``"pin"`` for a kinematic hold (post-solve), ``"spring"`` for a
            force (pre-solve). See :func:`_particle_pin_kernel` for why an
            adhesive peel needs the pin.
        pin_max_step: Ceiling on how far a pinned particle moves per substep
            [m]. Only used in pin mode.
    """

    def __init__(
        self,
        model: newton.Model,
        *,
        radius: float = 0.03,
        max_particles: int = 64,
        stiffness: float = 120.0,
        damping_ratio: float = 1.0,
        max_accel: float = 40.0,
        mode: str = "pin",
        pin_max_step: float = 0.0005,
    ) -> None:
        if mode not in ("pin", "spring"):
            raise ValueError(f"mode must be 'pin' or 'spring', got {mode!r}")
        self.model = model
        self.mode = mode
        self.pin_max_step = float(pin_max_step)
        self.radius = float(radius)
        self.max_particles = int(max_particles)
        self.stiffness = float(stiffness)
        self.damping_ratio = float(damping_ratio)
        self.max_accel = float(max_accel) * 9.81

        device = model.device
        self._grabbed = wp.zeros(self.max_particles, dtype=wp.int32, device=device)
        self._offsets = wp.zeros(self.max_particles, dtype=wp.vec3, device=device)
        self._count = wp.zeros(1, dtype=wp.int32, device=device)
        self._target = wp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=wp.transform, device=device)
        self._count_host = np.zeros(1, dtype=np.int32)
        self._target_host = np.zeros((1, 7), dtype=np.float32)
        self._target_host[0, 6] = 1.0
        self.last_distance = float("inf")
        """Distance from the last grab point to the nearest candidate particle [m].

        Recorded on a miss as well as a hit, which is what makes a failed
        scripted grasp say how far off it was rather than merely that it failed.
        """
        self.grabbed_indices = np.zeros(0, dtype=np.int32)
        """Which particles are held, for evaluators asking whether a grasp latched."""

    @property
    def is_pin(self) -> bool:
        return self.mode == "pin"

    @property
    def active(self) -> bool:
        return int(self._count_host[0]) > 0

    @property
    def n_grabbed(self) -> int:
        return int(self._count_host[0])

    def grab(
        self,
        state: newton.State,
        position: Sequence[float],
        quat_xyzw: Sequence[float],
        *,
        candidates: Sequence[int] | None = None,
        fallback_nearest: bool = True,
    ) -> int:
        """Seize the particles near ``position`` and hold them in that frame.

        Args:
            state: The state to read particle positions from.
            position: World grip point [m].
            quat_xyzw: Orientation of the grip frame the offsets are kept in.
            candidates: Restrict the grab to these particle indices, e.g. one
                entity's range in a scene holding several deformables. ``None``
                considers every particle in the model.
            fallback_nearest: With nothing inside the radius, take the nearest
                few anyway. A hand-dragged handle rarely lands within 2 cm of a
                thin strip, and :attr:`last_distance` reports how far off the
                aim was. A scripted routine should pass ``False`` and treat a
                miss as a failed grasp rather than dragging a strip off from
                centimetres away.

        Returns:
            How many particles were taken; 0 is a miss, not an error.
        """
        q = state.particle_q.numpy()
        pool = np.arange(len(q), dtype=np.int32) if candidates is None else np.asarray(candidates, dtype=np.int32)
        if len(pool) == 0:
            self.release()
            return 0
        pos = np.asarray(position, dtype=np.float64)
        d = np.linalg.norm(q[pool] - pos, axis=1)
        # How far off the aim was, recorded whether or not anything was taken:
        # a miss whose distance is unknown is a miss nobody can diagnose.
        self.last_distance = float(d.min())
        local = np.flatnonzero(d < self.radius)
        if len(local) > self.max_particles:  # keep the closest
            local = local[np.argsort(d[local])[: self.max_particles]]
        if len(local) == 0 and fallback_nearest:
            local = np.argsort(d)[: min(8, self.max_particles)]
        if len(local) == 0:
            self.release()
            return 0
        idx = pool[local]

        # Offsets in the grip frame, so the patch keeps its shape as it moves.
        offsets = (q[idx] - pos) @ quat_to_mat(quat_xyzw)

        padded_idx = np.zeros(self.max_particles, dtype=np.int32)
        padded_off = np.zeros((self.max_particles, 3), dtype=np.float32)
        padded_idx[: len(idx)] = idx
        padded_off[: len(idx)] = offsets
        self._grabbed.assign(padded_idx)
        self._offsets.assign(padded_off)
        self._count_host[0] = len(idx)
        self._count.assign(self._count_host)
        self.grabbed_indices = np.asarray(idx, dtype=np.int32)
        self.set_target(position, quat_xyzw)
        return len(idx)

    def release(self) -> None:
        self._count_host[0] = 0
        self._count.assign(self._count_host)
        self.grabbed_indices = np.zeros(0, dtype=np.int32)

    def set_target(self, position: Sequence[float], quat_xyzw: Sequence[float]) -> None:
        self._target_host[0, :3] = position
        self._target_host[0, 3:] = quat_xyzw
        self._target.assign(self._target_host)

    def apply(self, state: newton.State, dt: float | None = None) -> None:
        """Hold the grabbed patch for one substep. A :class:`SubstepHooks` hook.

        In ``pin`` mode this writes positions and must therefore be contributed
        to ``post_solve``, so the solve cannot undo it; the substep ``dt`` is
        what lets the held particles' velocity match the carry. In ``spring``
        mode it adds to ``particle_f`` and belongs in ``pre_solve``.
        :attr:`is_pin` tells a caller which slot to use.
        """
        if self.mode == "pin":
            wp.launch(
                _particle_pin_kernel,
                dim=self.max_particles,
                inputs=[
                    self._grabbed,
                    self._offsets,
                    self._count,
                    self._target,
                    self.pin_max_step,
                    0.0 if not dt else 1.0 / dt,
                ],
                outputs=[state.particle_q, state.particle_qd],
                device=self.model.device,
            )
            return
        wp.launch(
            _particle_drag_kernel,
            dim=self.max_particles,
            inputs=[
                state.particle_q,
                state.particle_qd,
                self.model.particle_mass,
                self._grabbed,
                self._offsets,
                self._count,
                self._target,
                self.stiffness,
                self.damping_ratio,
                self.max_accel,
            ],
            outputs=[state.particle_f],
            device=self.model.device,
        )
