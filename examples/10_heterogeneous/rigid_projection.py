# SPDX-License-Identifier: Apache-2.0
"""Keep one part of a soft mesh rigid by projecting it after every solver step.

A port of ``rigid_project`` from the Newton VBD handoff
(``newton_vbd_code_slim/clawhauser_franka/sim_franka.py``). After each substep:

1. the part's nodes get their best rigid pose by mass-weighted Kabsch -- the
   rotation and translation that best map their rest positions onto where the
   solver left them;
2. the nodes are snapped onto that pose, which removes the part's deformation
   and keeps its centre of mass where the solver put it;
3. their velocities become the rigid-body velocity with the same linear and
   angular momentum.

Nodes the part shares with the rest of the mesh are not snapped (``free`` is
false): the handoff found that snapping them diverges the solve. This is a
correction outside the solver, not a rigid body: the part's reaction on the rest
of the mesh is not rigid-body dynamics. It reads the state back to the host
every substep, so it is slow; it is an experiment, off unless a scene asks.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RigidProjection", "enable", "part_nodes"]


def part_nodes(tets: np.ndarray, labels: np.ndarray, part: int, *, attached_to: int) -> tuple[np.ndarray, np.ndarray]:
    """The local node ids of ``part`` and a mask of those not shared with part ``attached_to``."""
    nodes = np.unique(tets[labels == part])
    return nodes, ~np.isin(nodes, np.unique(tets[labels == attached_to]))


class RigidProjection:
    """Mass-weighted Kabsch projection of a set of particles onto their rest shape."""

    def __init__(self, model, particles: np.ndarray, free: np.ndarray) -> None:
        self.particles = np.asarray(particles, dtype=np.int64)
        self.free = np.asarray(free, dtype=bool)
        self.rest = model.particle_q.numpy()[self.particles].astype(np.float64)
        self.mass = model.particle_mass.numpy()[self.particles].astype(np.float64)

    def apply(self, state) -> None:
        """Snap the free particles in ``state`` onto the part's best rigid pose, with rigid velocities."""
        q, v = state.particle_q.numpy(), state.particle_qd.numpy()
        x, vx = q[self.particles].astype(np.float64), v[self.particles].astype(np.float64)
        m, w = self.mass, self.mass / self.mass.sum()
        centre, rest_centre = w @ x, w @ self.rest
        u, _, vt = np.linalg.svd(((self.rest - rest_centre) * w[:, None]).T @ (x - centre))
        rotation = vt.T @ np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)]) @ u.T  # the last axis flips a mirror fit
        snapped = (self.rest - rest_centre) @ rotation.T + centre
        r = snapped - centre
        momentum = (m[:, None] * np.cross(r, vx)).sum(axis=0)
        inertia = (
            m[:, None, None] * ((r * r).sum(axis=1)[:, None, None] * np.eye(3) - r[:, :, None] * r[:, None, :])
        ).sum(axis=0)
        omega = np.linalg.solve(inertia, momentum)
        ids = self.particles[self.free]
        q[ids] = snapped[self.free]
        v[ids] = (w @ vx + np.cross(omega, r))[self.free]
        state.particle_q.assign(q)
        state.particle_qd.assign(v)


def enable(solver, model, particles: np.ndarray, free: np.ndarray) -> RigidProjection:
    """Project ``particles`` after every ``solver.step``; the solver is otherwise unchanged."""
    projection = RigidProjection(model, particles, free)
    step = solver.step

    def projected_step(state_in, state_out, *args, **kwargs):
        step(state_in, state_out, *args, **kwargs)
        projection.apply(state_out)

    solver.step = projected_step
    return projection
