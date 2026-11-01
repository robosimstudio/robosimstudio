# SPDX-License-Identifier: Apache-2.0
"""VBD with an optional 3-DOF translation correction for one FEM volume.

The correction minimizes the inertial objective over a common translation.
It is only enabled for collision-free runs. Use the base VBD solver when a
collision pipeline or contact buffer is involved.

Compatible with the private Newton API at revision 454fc3e. Revalidate these
hooks when changing Newton versions.
"""
from __future__ import annotations

import numpy as np
import warp as wp
from newton.solvers import SolverVBD

__all__ = ["SolverVBDCoarse"]


@wp.kernel
def _accumulate_inertia(
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
def _solve_translation(
    residual: wp.array[wp.vec3],
    hessian: wp.array[wp.mat33],
    budget: float,
    path_length: wp.array[float],
    shift: wp.array[wp.vec3],
):
    w = wp.tid()
    step = wp.inverse(hessian[w]) * residual[w]
    length = wp.length(step)
    remaining = wp.max(0.0, budget - path_length[w])
    if length > remaining and length > 0.0:
        step *= remaining / length

    if wp.isfinite(step[0]) and wp.isfinite(step[1]) and wp.isfinite(step[2]):
        shift[w] = step
        path_length[w] += wp.length(step)
    else:
        shift[w] = wp.vec3(0.0)


@wp.kernel
def _apply_translation(
    world: wp.array[int],
    shift: wp.array[wp.vec3],
    reference: wp.array[wp.vec3],
    q: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
):
    p = wp.tid()
    updated = displacement[p] + shift[world[p]]
    displacement[p] = updated
    q[p] = reference[p] + updated


class _TranslationBlock:
    def __init__(self, solver, *, budget):
        model = solver.model
        if model.attachment_particle_body_count or model.spring_count:
            raise ValueError("coarse translation does not support attachments or springs")

        mass = model.particle_mass.numpy()
        inv_mass = model.particle_inv_mass.numpy()
        flags = model.particle_flags.numpy()
        if not len(mass) or not np.isfinite(mass).all() or np.any(mass <= 0.0):
            raise ValueError("all particles must have positive finite mass")
        if np.any(flags != 1) or not np.allclose(mass * inv_mass, 1.0):
            raise ValueError(
                "all particles must be active, unpinned and have consistent inverse mass"
            )

        worlds = model.particle_world
        if worlds is None:
            worlds = np.zeros(len(mass), dtype=int)
        else:
            worlds = worlds.numpy()
        if np.all(worlds == -1):
            worlds = np.zeros_like(worlds)
        elif np.any(worlds < 0) or len(np.unique(worlds)) != 1:
            raise ValueError("coarse translation supports exactly one world")

        tets = model.tet_indices.numpy()
        if not len(tets) or not np.array_equal(np.unique(tets), np.arange(len(mass))):
            raise ValueError("every particle must belong to the tetrahedral volume")
        if not self._is_connected(tets):
            raise ValueError("coarse translation requires one face-connected tetrahedral volume")

        self.solver = solver
        self.budget = budget
        self.world = wp.array(worlds, dtype=int, device=model.device)
        self.residual = wp.zeros(1, dtype=wp.vec3, device=model.device)
        self.hessian = wp.zeros(1, dtype=wp.mat33, device=model.device)
        self.shift = wp.zeros(1, dtype=wp.vec3, device=model.device)
        self.path_length = wp.zeros(1, dtype=float, device=model.device)

    @staticmethod
    def _is_connected(tets):
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
        return len({root(i) for i in range(len(tets))}) == 1

    def begin_substep(self, world_mask=None):
        if world_mask is not None and not bool(world_mask.numpy()[0]):
            return
        self.path_length.zero_()
        self.shift.zero_()
        self.residual.zero_()
        self.hessian.zero_()

    def apply(self, state_in, state_out, dt):
        solver = self.solver
        model = solver.model
        self.residual.zero_()
        self.hessian.zero_()

        wp.launch(
            _accumulate_inertia,
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
        wp.launch(
            _solve_translation,
            dim=1,
            inputs=[self.residual, self.hessian, self.budget, self.path_length, self.shift],
            device=model.device,
        )
        wp.launch(
            _apply_translation,
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
    """VBD with an optional coarse translation update for collision-free runs."""

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
        collision_enabled = (
            kwargs.get("collision_pipeline") is not None
            or bool(kwargs.get("particle_enable_self_contact", False))
            or bool(kwargs.get("rigid_soft_enable_dat", False))
        )
        self.coarse_translation = bool(coarse_translation and not collision_enabled)

        limits = np.asarray((coarse_contact_gap, coarse_budget), dtype=float)
        if not np.isfinite(limits).all() or np.any(limits <= 0.0):
            raise ValueError("coarse_contact_gap and coarse_budget must be positive and finite")
        if coarse_budget > coarse_contact_gap / 4.0:
            raise ValueError("coarse_budget must not exceed coarse_contact_gap / 4")

        self.coarse_block = None
        super().__init__(model, **kwargs)
        self.coarse_contact_gap = float(coarse_contact_gap)
        self.coarse_budget = float(coarse_budget)

        if self.coarse_translation:
            mode = (
                wp.DeterministicMode.RUN_TO_RUN
                if coarse_deterministic
                else wp.DeterministicMode.NOT_GUARANTEED
            )
            self._set_module_options(
                {"deterministic": mode, "deterministic_max_records": 0},
                module=_accumulate_inertia.module,
            )
            self.coarse_block = _TranslationBlock(self, budget=self.coarse_budget)

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        return super().step(state_in, state_out, control, contacts, dt, observables=observables)

    def reset(self, state, world_mask=None, flags=None):
        super().reset(state, world_mask=world_mask, flags=flags)
        if self.coarse_block is not None:
            self.coarse_block.begin_substep(world_mask)

    def _initialize_particles(self, state_in, state_out, contacts, dt):
        super()._initialize_particles(state_in, state_out, contacts, dt)
        if self.coarse_block is not None:
            self.coarse_block.begin_substep()

    def _solve_particle_iteration(self, state_in, state_out, contacts, dt):
        super()._solve_particle_iteration(state_in, state_out, contacts, dt)
        if self.coarse_block is not None and contacts is None:
            self.coarse_block.apply(state_in, state_out, dt)