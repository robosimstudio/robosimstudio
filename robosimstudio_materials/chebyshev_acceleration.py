# SPDX-License-Identifier: Apache-2.0
"""Chebyshev acceleration for VBD particle iterations."""

from __future__ import annotations

import warp as wp
from newton.solvers import SolverVBD

__all__ = ["SolverVBDChebyshev"]


@wp.func
def _mark_particle(
    mask: wp.array[wp.int32],
    mask_all: wp.array[wp.int32],
    particle: int,
    particle_count: int,
):
    if particle >= 0:
        if particle < particle_count:
            wp.atomic_max(mask, particle, 1)
        else:
            wp.atomic_max(mask_all, 0, 1)


@wp.kernel
def _mark_soft_contacts(
    indices: wp.array[wp.vec3i],
    count: wp.array[wp.int32],
    capacity: int,
    particle_count: int,
    mask: wp.array[wp.int32],
    mask_all: wp.array[wp.int32],
):
    contact = wp.tid()

    if contact == 0 and count[0] > 0:
        # Any particle contact at all: the sweep's contact response is not a
        # linear fixed point, so extrapolating it is extrapolating a penalty
        # force. Measured on the clawhauser rod scene: masking only the touching
        # particles left the plush 30 mm squashed into the bench by frame 16
        # (minimum tet volume ratio 0.68, bench load 69 N against a weight of
        # 1.23 N) and the pads closed above the loop, so nothing was lifted.
        mask_all[0] = 1

    if contact == 0 and count[0] > capacity:
        mask_all[0] = 1

    if contact >= wp.min(count[0], capacity):
        return

    particles = indices[contact]
    for k in range(3):
        _mark_particle(mask, mask_all, particles[k], particle_count)


@wp.kernel
def _mark_self_contacts(
    vertex_counts: wp.array[wp.int32],
    vertex_capacities: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    vertex_pairs: wp.array[wp.int32],
    triangle_indices: wp.array[wp.int32],
    edge_counts: wp.array[wp.int32],
    edge_capacities: wp.array[wp.int32],
    edge_offsets: wp.array[wp.int32],
    edge_pairs: wp.array[wp.int32],
    edge_indices: wp.array[wp.int32],
    particle_count: int,
    triangle_count: int,
    edge_count: int,
    mask: wp.array[wp.int32],
    mask_all: wp.array[wp.int32],
):
    index = wp.tid()

    if index < particle_count:
        count = vertex_counts[index]
        capacity = vertex_capacities[index]
        if count > capacity:
            wp.atomic_max(mask_all, 0, 1)
        if count > 0:
            _mark_particle(mask, mask_all, index, particle_count)
            wp.atomic_max(mask_all, 0, 1)

        for j in range(wp.min(count, capacity)):
            triangle = vertex_pairs[2 * (vertex_offsets[index] + j) + 1]
            if triangle >= 0 and triangle < triangle_count:
                base = 3 * triangle
                _mark_particle(mask, mask_all, triangle_indices[base], particle_count)
                _mark_particle(mask, mask_all, triangle_indices[base + 1], particle_count)
                _mark_particle(mask, mask_all, triangle_indices[base + 2], particle_count)
            else:
                wp.atomic_max(mask_all, 0, 1)

    if index < edge_count:
        count = edge_counts[index]
        capacity = edge_capacities[index]
        if count > capacity:
            wp.atomic_max(mask_all, 0, 1)

        if count > 0:
            base = 4 * index
            _mark_particle(mask, mask_all, edge_indices[base + 2], particle_count)
            _mark_particle(mask, mask_all, edge_indices[base + 3], particle_count)

        for j in range(wp.min(count, capacity)):
            other = edge_pairs[2 * (edge_offsets[index] + j) + 1]
            if other >= 0 and other < edge_count:
                base = 4 * other
                _mark_particle(mask, mask_all, edge_indices[base + 2], particle_count)
                _mark_particle(mask, mask_all, edge_indices[base + 3], particle_count)
            else:
                wp.atomic_max(mask_all, 0, 1)


@wp.kernel
def _mark_all_particles(mask_all: wp.array[wp.int32]):
    mask_all[0] = 1


@wp.kernel
def _chebyshev_blend(
    omega: float,
    use_collision_mask: int,
    mask: wp.array[wp.int32],
    mask_all: wp.array[wp.int32],
    previous: wp.array[wp.vec3],
    older: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    displacements: wp.array[wp.vec3],
):
    particle = wp.tid()
    swept = positions[particle]
    skip = use_collision_mask != 0 and (mask_all[0] != 0 or mask[particle] != 0)

    accelerated = swept
    if not skip:
        accelerated = older[particle] + omega * (swept - older[particle])

    displacements[particle] += accelerated - swept
    positions[particle] = accelerated
    older[particle] = previous[particle]
    previous[particle] = accelerated


class SolverVBDChebyshev(SolverVBD):
    """VBD with Chebyshev acceleration of non-contacting particles.

    ``chebyshev_collision_mask=False`` restores unmasked acceleration.
    """

    def __init__(
        self,
        model,
        *,
        chebyshev_rho: float = 0.9,
        chebyshev_collision_mask: bool = True,
        **kwargs,
    ):
        if not 0.0 <= chebyshev_rho < 1.0:
            raise ValueError("chebyshev_rho must be in [0, 1)")

        super().__init__(model, **kwargs)
        if self.rigid_soft_enable_dat:
            raise ValueError("Chebyshev acceleration does not support rigid-soft DAT")

        self.chebyshev_rho = float(chebyshev_rho)
        self.chebyshev_collision_mask = bool(chebyshev_collision_mask)
        self._previous = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self._older = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self._collision_mask = wp.zeros(model.particle_count, dtype=wp.int32, device=model.device)
        self._mask_all = wp.zeros(1, dtype=wp.int32, device=model.device)
        self._iteration = 0
        self._omega = 1.0

    def _initialize_particles(self, state_in, state_out, contacts, dt):
        super()._initialize_particles(state_in, state_out, contacts, dt)
        wp.copy(self._previous, state_in.particle_q)
        wp.copy(self._older, state_in.particle_q)
        self._collision_mask.zero_()
        self._mask_all.zero_()
        self._iteration = 0
        self._omega = 1.0

    def _update_collision_mask(self, contacts):
        if not self.chebyshev_collision_mask:
            return

        if contacts is None:
            contacts = getattr(self, "contacts", None)

        self_contact_enabled = getattr(self, "particle_enable_self_contact", False)

        if contacts is None:
            if self.model.shape_count or self_contact_enabled:
                wp.launch(
                    _mark_all_particles,
                    dim=1,
                    inputs=[self._mask_all],
                    device=self.model.device,
                )
            return

        indices = getattr(contacts, "soft_contact_indices", None)
        count = getattr(contacts, "soft_contact_count", None)
        capacity = getattr(contacts, "soft_contact_max", 0)
        if indices is not None and count is not None and capacity:
            wp.launch(
                _mark_soft_contacts,
                dim=capacity,
                inputs=[
                    indices,
                    count,
                    capacity,
                    self.model.particle_count,
                    self._collision_mask,
                    self._mask_all,
                ],
                device=self.model.device,
            )
        elif self.model.shape_count:
            wp.launch(
                _mark_all_particles,
                dim=1,
                inputs=[self._mask_all],
                device=self.model.device,
            )

        self_contact = getattr(contacts, "soft_self_contact_data", None)
        if self_contact is None:
            if self_contact_enabled:
                wp.launch(
                    _mark_all_particles,
                    dim=1,
                    inputs=[self._mask_all],
                    device=self.model.device,
                )
            self._report_mask(contacts)
            return

        fields = (
            "vertex_colliding_triangles_count",
            "vertex_colliding_triangles_buffer_sizes",
            "vertex_colliding_triangles_offsets",
            "vertex_colliding_triangles",
            "edge_colliding_edges_count",
            "edge_colliding_edges_buffer_sizes",
            "edge_colliding_edges_offsets",
            "edge_colliding_edges",
        )
        # This kernel uses Newton's flat int32 triangle and edge index buffers.
        if (
            not all(getattr(self_contact, name, None) is not None for name in fields)
            or self.model.tri_indices is None
            or self.model.edge_indices is None
            or self.model.tri_indices.dtype != wp.int32
            or self.model.edge_indices.dtype != wp.int32
        ):
            wp.launch(
                _mark_all_particles,
                dim=1,
                inputs=[self._mask_all],
                device=self.model.device,
            )
            return

        wp.launch(
            _mark_self_contacts,
            dim=max(self.model.particle_count, self.model.edge_count),
            inputs=[
                self_contact.vertex_colliding_triangles_count,
                self_contact.vertex_colliding_triangles_buffer_sizes,
                self_contact.vertex_colliding_triangles_offsets,
                self_contact.vertex_colliding_triangles,
                self.model.tri_indices,
                self_contact.edge_colliding_edges_count,
                self_contact.edge_colliding_edges_buffer_sizes,
                self_contact.edge_colliding_edges_offsets,
                self_contact.edge_colliding_edges,
                self.model.edge_indices,
                self.model.particle_count,
                self.model.tri_count,
                self.model.edge_count,
                self._collision_mask,
                self._mask_all,
            ],
            device=self.model.device,
        )

    def _report_mask(self, contacts) -> None:
        """One line, first substep only: how much of the mesh the mask covers."""
        if getattr(self, "_mask_reported", False):
            return
        self._mask_reported = True
        import json

        count = -1
        if contacts is not None:
            soft = getattr(contacts, "soft_contact_count", None)
            if soft is not None:
                count = int(soft.numpy()[0])
        print(
            json.dumps(
                {
                    "chebyshev_mask_particles": int(self._collision_mask.numpy().sum()),
                    "chebyshev_mask_all": int(self._mask_all.numpy()[0]),
                    "chebyshev_soft_contacts": count,
                    "chebyshev_particles": int(self.model.particle_count),
                }
            ),
            flush=True,
        )

    def _solve_particle_iteration(self, state_in, state_out, contacts, dt):
        super()._solve_particle_iteration(state_in, state_out, contacts, dt)
        self._update_collision_mask(contacts)

        rho2 = self.chebyshev_rho * self.chebyshev_rho
        self._iteration += 1
        if self._iteration == 2:
            self._omega = 2.0 / (2.0 - rho2)
        elif self._iteration > 2:
            self._omega = 4.0 / (4.0 - rho2 * self._omega)

        wp.launch(
            _chebyshev_blend,
            dim=self.model.particle_count,
            inputs=[
                self._omega,
                int(self.chebyshev_collision_mask),
                self._collision_mask,
                self._mask_all,
                self._previous,
                self._older,
                state_in.particle_q,
                self.particle_displacements,
            ],
            device=self.model.device,
        )
        wp.copy(state_out.particle_q, state_in.particle_q)
