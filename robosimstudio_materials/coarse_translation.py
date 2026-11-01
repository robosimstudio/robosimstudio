# SPDX-License-Identifier: Apache-2.0
"""Optional VBD coarse translation solve for one connected tetrahedral volume.

A stiff *connected* FEM has a whole-object translation mode the particle solve
reaches badly: on the labelled Clawhauser volume at 20 substeps and 12
iterations the mode's residual is 192.88 N (29.41 N at 512) against a 1.227 N
weight, so the object cannot balance itself against a local contact and sinks
or squashes instead of being supported. This module adds that one mode back as
a 3-DOF common-translation solve of VBD's own inertial/contact objective, once
per particle iteration, inside the same substep.

Use the RSS dotted backend path::

    rss.backend(
        "robosimstudio_materials.coarse_translation.SolverVBDCoarse",
        coarse_translation=True,
        coarse_contact_gap=0.02,     # the collision pipeline's own detection gap
        coarse_budget=0.005,         # <= gap / 4
    )

``coarse_contact_gap`` must be the gap the collision pipeline actually detects
at -- the scene's ``soft_contact_margin`` -- because a correction can only reach
contacts the pipeline stored. ``coarse_budget`` is the cumulative extra
translation allowed per physical substep; it is a budget on the block's own
motion, not a nonpenetration or collision-coverage guarantee.

Private compatibility surface: Newton 1.8.0.dev0 at this repository's
``third-party/newton`` (``454fc3e``). It reuses that revision's rigid-soft
force/Hessian and contact-eligibility functions, VBD's inertial target,
particle anchors and accumulated-displacement storage; revalidate those
contracts before changing Newton versions. The stock Newton solver and RSS core
are not modified.

Deliberately not ported from the branch's ``vbd_coarse.SolverVBDCoarse``: object
SE3 rotation, the rigid FREE-assembly block and its coupled 6D solve, multi-world
loops (this block owns exactly one world and refuses more), model-notification
profile revalidation, and the diagnostic ledgers. What remains is the
formulation that was qualified: the same hook, the same two blocks, the same
3x3 solve, the same position-only apply, the same cumulative per-substep budget
and the same DAT/self-contact refusal.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from newton._src.geometry.soft_contacts_mesh import (
    _make_soft_contact_filter_data,
    _soft_contact_force_enabled,
    _SoftContactFilterData,
)
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _eval_body_particle_contact,
    _eval_soft_ef_contact,
)
from newton.solvers import SolverVBD

__all__ = ["SolverVBDCoarse"]


@wp.kernel
def _check_contact_capacity(count: wp.array[int], capacity: int, invalid_world: wp.array[int]):
    w = wp.tid()
    # Overflow means omitted contacts may involve this object. Never infer
    # coverage from the prefix.
    if count[0] > capacity:
        invalid_world[w] = 1


@wp.kernel
def _inertia_block(
    dt: float,
    q: wp.array[wp.vec3],
    inertia: wp.array[wp.vec3],
    mass: wp.array[float],
    world: wp.array[int],
    residual: wp.array[wp.vec3],
    hessian: wp.array[wp.mat33],
):
    p = wp.tid()
    w = world[p]
    stiffness = mass[p] / (dt * dt)
    wp.atomic_add(residual, w, stiffness * (inertia[p] - q[p]))
    wp.atomic_add(hessian, w, stiffness * wp.identity(n=3, dtype=float))


@wp.kernel
def _contact_block(
    dt: float,
    q: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    world: wp.array[int],
    radius: wp.array[float],
    indices: wp.array[wp.vec3i],
    count: wp.array[int],
    capacity: int,
    barycentric: wp.array[wp.vec3],
    contact_filter: _SoftContactFilterData,
    ke: wp.array[float],
    kd: wp.array[float],
    mu: wp.array[float],
    friction_epsilon: float,
    use_log_barrier: bool,
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_previous: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    shape: wp.array[int],
    body_pos: wp.array[wp.vec3],
    body_velocity: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin: wp.array[float],
    residual: wp.array[wp.vec3],
    hessian: wp.array[wp.mat33],
    invalid_world: wp.array[int],
):
    c = wp.tid()
    if c >= wp.min(count[0], capacity) or not _soft_contact_force_enabled(c, contact_filter):
        return
    corners = indices[c]
    if corners[0] < 0:
        return
    w = world[corners[0]]
    if count[0] > capacity:
        invalid_world[w] = 1
        return
    f = wp.vec3(0.0)
    h = wp.mat33(0.0)
    weight_sum = float(1.0)
    if corners[1] < 0:
        p = corners[0]
        f, h = _eval_body_particle_contact(
            p,
            q[p],
            previous[p],
            c,
            ke[c],
            kd[c],
            mu[c],
            friction_epsilon,
            radius,
            shape_body,
            body_q,
            body_previous,
            body_qd,
            body_com,
            shape,
            body_pos,
            body_velocity,
            normal,
            margin,
            dt,
            use_log_barrier,
        )
    else:
        weights = barycentric[c]
        weight_sum = float(0.0)
        for k in range(3):
            p = corners[k]
            if p >= 0:
                if world[p] != w:
                    invalid_world[w] = 1
                    return
                weight_sum += weights[k]
        f, h, _point = _eval_soft_ef_contact(
            c,
            corners,
            weights,
            q,
            previous,
            radius,
            ke[c],
            kd[c],
            mu[c],
            friction_epsilon,
            shape_body,
            body_q,
            body_previous,
            body_qd,
            body_com,
            shape,
            body_pos,
            body_velocity,
            normal,
            margin,
            dt,
            use_log_barrier,
        )
    # Whole-object translation moves every corner together. Include this
    # contact Hessian ONCE, with (sum bary)^2, not sum of squared barycentrics.
    wp.atomic_add(residual, w, weight_sum * f)
    wp.atomic_add(hessian, w, weight_sum * weight_sum * h)


@wp.kernel
def _solve_block(
    residual: wp.array[wp.vec3],
    hessian: wp.array[wp.mat33],
    invalid_world: wp.array[int],
    budget: float,
    max_shift: float,
    relaxation: float,
    path_length: wp.array[float],
    step_path: wp.array[float],
    shift: wp.array[wp.vec3],
    applications: wp.array[int],
):
    w = wp.tid()
    shift[w] = wp.vec3(0.0)
    if invalid_world[w] != 0:
        return
    step = relaxation * (wp.inverse(hessian[w]) * residual[w])
    length = wp.length(step)
    remaining = wp.max(0.0, wp.min(max_shift, budget - path_length[w]))
    if length > remaining and length > 0.0:
        step *= remaining / length
    if wp.isfinite(step[0]) and wp.isfinite(step[1]) and wp.isfinite(step[2]):
        shift[w] = step
        path_length[w] += wp.length(step)
        # Cleared by the caller, not by ``begin_substep``: the substep budget
        # above resets every substep, so a per-frame reader of ``path_length``
        # would only ever see the last one.
        step_path[w] += wp.length(step)
        if wp.length(step) > 0.0:
            applications[w] += 1


@wp.kernel
def _apply_block(
    world: wp.array[int],
    shift: wp.array[wp.vec3],
    reference: wp.array[wp.vec3],
    q: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
):
    p = wp.tid()
    delta = shift[world[p]]
    # One writer owns each particle. Array += lowers to atomic_add, whose
    # deterministic reduction can run after this kernel's read of displacement.
    # Use the local value for both outputs so this iterate moves immediately.
    updated = displacement[p] + delta
    displacement[p] = updated
    q[p] = reference[p] + updated


class _TranslationBlock:
    """A restricted common translation solve, independent of desired robot poses."""

    def __init__(self, solver, *, contact_gap_m, substep_budget_m):
        self.solver = solver
        if solver.rigid_soft_enable_dat:
            raise ValueError("coarse translation does not support DAT")
        limits = np.asarray((contact_gap_m, substep_budget_m), dtype=float)
        if not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError("positive finite limits required")
        if substep_budget_m > contact_gap_m / 4:
            raise ValueError("require substep budget <= one quarter of contact detection gap")
        model = solver.model
        if model.attachment_particle_body_count or model.spring_count:
            raise ValueError("coarse translation refuses attachments and springs")
        mass, inv_mass = model.particle_mass.numpy(), model.particle_inv_mass.numpy()
        flags = model.particle_flags.numpy()
        if not len(mass) or not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError("all particles must have positive finite mass")
        if np.any(flags != 1) or not np.allclose(mass * inv_mass, 1):
            raise ValueError("all particles must be active, unpinned and have consistent inverse mass")
        source_worlds = model.particle_world
        if source_worlds is None:
            # A single-system builder allocates no world array at all: there is
            # exactly one world, which is the case this block is for.
            worlds = np.zeros(len(mass), dtype=int)
            source_worlds = worlds
        else:
            worlds = source_worlds.numpy()
        if np.all(worlds == -1):
            # Newton's nonreplicated builder stores a single global system as
            # world -1; this block owns that one object as its sole reduction.
            worlds = np.zeros_like(worlds)
        elif np.any(worlds < 0) or len(np.unique(worlds)) != 1:
            raise ValueError("coarse translation owns exactly one world")
        tets = model.tet_indices.numpy()
        if not len(tets) or not np.array_equal(np.unique(tets), np.arange(len(mass))):
            raise ValueError("every particle must belong to the tetrahedral volume")
        # Face connectivity decides whether one common translation is the
        # object's motion at all: two volumes may translate apart.
        parent = list(range(len(tets)))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        faces = {}
        for i, tet in enumerate(tets):
            for omit in range(4):
                face = tuple(sorted(np.delete(tet, omit).tolist()))
                if face in faces:
                    parent[root(i)] = root(faces[face])
                else:
                    faces[face] = i
        if len({root(i) for i in range(len(tets))}) != 1:
            raise ValueError("coarse translation needs exactly one face-connected volume")
        self.budget = float(substep_budget_m)
        self.max_shift = float(substep_budget_m)
        self.relaxation = 1.0
        device = model.device
        self.world = wp.array(worlds, dtype=int, device=device)
        self.residual = wp.zeros(1, dtype=wp.vec3, device=device)
        # The residual BEFORE the latest correction, not an independently
        # evaluated post-correction force-balance measurement.
        self.hessian = wp.zeros(1, dtype=wp.mat33, device=device)
        self.shift = wp.zeros(1, dtype=wp.vec3, device=device)
        self.path_length = wp.zeros(1, dtype=float, device=device)
        # Cumulative since the last ``reset``, for a per-frame reader.
        self.step_path = wp.zeros(1, dtype=float, device=device)
        self.applications = wp.zeros(1, dtype=int, device=device)
        self.invalid_world = wp.zeros(1, dtype=int, device=device)

    def begin_substep(self, world_mask=None):
        """Clear the per-substep budget and diagnostics; called once per substep."""
        if world_mask is not None and not bool(world_mask.numpy()[0]):
            # One world, so the mask is read at its own slot; a world that is
            # not being reset keeps its accumulated path.
            return
        self.path_length.zero_()
        self.applications.zero_()
        self.shift.zero_()
        self.residual.zero_()
        self.hessian.zero_()
        self.invalid_world.zero_()

    def apply(self, state_in, state_out, contacts, dt):
        solver, model = self.solver, self.solver.model
        self.residual.zero_()
        self.hessian.zero_()
        self.invalid_world.zero_()
        wp.launch(
            _inertia_block,
            dim=model.particle_count,
            inputs=[
                dt,
                state_in.particle_q,
                solver.inertia,
                model.particle_mass,
                self.world,
                self.residual,
                self.hessian,
            ],
            device=model.device,
        )
        if contacts is not None:
            wp.launch(
                _check_contact_capacity,
                dim=1,
                inputs=[contacts.soft_contact_count, contacts.soft_contact_max, self.invalid_world],
                device=model.device,
            )
        if contacts is not None and contacts.soft_contact_max:
            if solver.integrate_with_external_rigid_solver:
                body_q, previous, body_qd = state_out.body_q, state_in.body_q, state_out.body_qd
            else:
                body_q = state_in.body_q
                previous = solver.body_q_prev if model.body_count else None
                body_qd = state_in.body_qd
            contact_filter = _make_soft_contact_filter_data(model, contacts, state_in.particle_q, body_q)
            wp.launch(
                _contact_block,
                dim=contacts.soft_contact_max,
                inputs=[
                    dt,
                    state_in.particle_q,
                    solver.particle_q_prev,
                    self.world,
                    model.particle_radius,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_count,
                    contacts.soft_contact_max,
                    contacts.soft_contact_barycentric,
                    contact_filter,
                    solver.body_particle_contact_penalty_k,
                    solver.body_particle_contact_material_kd,
                    solver.body_particle_contact_material_mu,
                    solver.friction_epsilon,
                    solver.rigid_soft_contact_use_log_barrier,
                    model.shape_body,
                    body_q,
                    previous,
                    body_qd,
                    model.body_com,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    model.shape_margin,
                    self.residual,
                    self.hessian,
                    self.invalid_world,
                ],
                device=model.device,
            )
        wp.launch(
            _solve_block,
            dim=1,
            inputs=[
                self.residual,
                self.hessian,
                self.invalid_world,
                self.budget,
                self.max_shift,
                self.relaxation,
                self.path_length,
                self.step_path,
                self.shift,
                self.applications,
            ],
            device=model.device,
        )
        wp.launch(
            _apply_block,
            dim=model.particle_count,
            inputs=[
                self.world,
                self.shift,
                solver.pos_prev_collision_detection,
                state_in.particle_q,
                solver.particle_displacements,
            ],
            device=model.device,
        )
        wp.copy(state_out.particle_q, state_in.particle_q)


class SolverVBDCoarse(SolverVBD):
    """VBD with an opt-in translation block for one fused tetrahedral object.

    ``coarse_translation=False`` retains the base solver's stepping and reset
    behavior. Enabled, each particle iteration is followed by a fresh 3D solve
    of its existing inertial/contact objective along common object translation.
    The FEM material and deformation are preserved; no target object pose,
    attachment, or external manipulation force is introduced.

    Args:
        model: Newton Model with one complete labelled tetrahedral volume.
        coarse_translation: Enable the restricted translation block (default True).
        coarse_contact_gap: Actual collision pipeline detection gap in meters.
        coarse_budget: Maximum cumulative additional translation path per
            physical substep in meters, at most one quarter of
            ``coarse_contact_gap``.
        coarse_deterministic: Ordered coarse reductions (default True). False
            uses native floating atomics, reducing graph scratch and runtime;
            roundoff and repeated-rollout agreement must be validated separately.
        **kwargs: Passed unchanged to Newton's SolverVBD.

    Pins, attachments, springs, several worlds and disconnected volumes are
    refused when the block is enabled. Overflow disables the block, because an
    omitted contact may involve this object. The block's budget limits its own
    additional motion; it does not limit local VBD or rigid-body motion and is
    not a collision-coverage guarantee. Diagnostics and budget clear on reset
    and at every substep; DAT cannot be enabled with it. Self-contact is
    allowed: its pair forces are internal to the object and cancel in the
    common-translation solve, which only reads rigid-soft contacts.
    """

    def __init__(
        self,
        model,
        *,
        coarse_translation=True,
        coarse_contact_gap=0.02,
        coarse_budget=0.005,
        coarse_deterministic=True,
        **kwargs,
    ):
        self.coarse_block = None
        super().__init__(model, **kwargs)
        self.coarse_translation = bool(coarse_translation)
        self.coarse_contact_gap = float(coarse_contact_gap)
        self.coarse_budget = float(coarse_budget)
        if self.coarse_translation:
            # Preserve the ordered diagnostic baseline by default. Its per-launch
            # scratch is retained throughout CUDA graph lifetime; callers can
            # explicitly select floating atomics for throughput after validation.
            mode = wp.DeterministicMode.RUN_TO_RUN if coarse_deterministic else wp.DeterministicMode.NOT_GUARANTEED
            self._set_module_options(
                {"deterministic": mode, "deterministic_max_records": 0},
                module=_inertia_block.module,
            )
            self.coarse_block = _TranslationBlock(
                self,
                contact_gap_m=self.coarse_contact_gap,
                substep_budget_m=self.coarse_budget,
            )

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        if self.coarse_block is not None and self.rigid_soft_enable_dat:
            raise ValueError("coarse translation does not support enabling DAT; rebuild the solver")
        return super().step(state_in, state_out, control, contacts, dt, observables=observables)

    def reset(self, state, world_mask=None, flags=None):
        super().reset(state, world_mask=world_mask, flags=flags)
        if self.coarse_block is not None:
            self.coarse_block.begin_substep(world_mask)
            self.coarse_block.step_path.zero_()

    def _initialize_particles(self, state_in, state_out, contacts, dt):
        super()._initialize_particles(state_in, state_out, contacts, dt)
        if self.coarse_block is not None:
            self.coarse_block.begin_substep()

    def _solve_particle_iteration(self, state_in, state_out, contacts, dt):
        super()._solve_particle_iteration(state_in, state_out, contacts, dt)
        if self.coarse_block is not None:
            self.coarse_block.apply(state_in, state_out, contacts, dt)
