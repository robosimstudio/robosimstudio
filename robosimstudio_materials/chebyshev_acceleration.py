# SPDX-License-Identifier: Apache-2.0
"""Chebyshev acceleration for VBD particle iterations.

After each particle sweep s_k (computed from x_{k-1}) the iterate is blended with
the one from two sweeps earlier, using the Chebyshev semi-iterative weights from
the VBD paper (after Wang 2015)::

    w_1 = 1,  w_2 = 2 / (2 - rho^2),  w_k = 4 / (4 - rho^2 * w_{k-1})
    x_k = x_{k-2} + w_k * (g * (s_k - x_{k-1}) + x_{k-1} - x_{k-2})

``chebyshev_rho`` estimates the sweep's spectral radius and ``g``
(``chebyshev_gamma``) under-relaxes the sweep. ``chebyshev_rho=0`` with
``chebyshev_gamma=1`` reproduces stock VBD.

Acceleration runs in contact too. The sweep's contact response is a penalty
force rather than a linear fixed point, so the extrapolation is no longer a
fixed-point iteration there; ``chebyshev_restart`` (drop the history when the
sweep step grows), ``chebyshev_start`` (plain sweeps first) and
``chebyshev_gamma`` below 1 damp that. None of these is tuned.

Private compatibility surface: Newton 1.8.0.dev0 at this repository's
``third-party/newton``. It overrides VBD's ``_initialize_particles`` and
``_solve_particle_iteration`` hooks, assumes the particle iterate lives in
``state_in.particle_q`` with ``particle_displacements`` accumulating its
movement. Positions are rewritten through ``_penetration_free_truncation`` so
self-contact truncation still holds. Revalidate these contracts before changing
Newton versions. Not supported: rigid-soft DAT.
"""

from __future__ import annotations

import warp as wp
from newton.solvers import SolverVBD

__all__ = ["SolverVBDChebyshev"]


@wp.kernel
def _sweep_norm(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3], norm2: wp.array[float]):
    p = wp.tid()
    d = positions[p] - previous[p]
    wp.atomic_add(norm2, 0, wp.dot(d, d))


@wp.kernel
def _update_schedule(
    norm2: wp.array[float],
    rho2: float,
    start: int,
    restart_enabled: int,
    restart_growth: float,
    state: wp.array[float],
    omega: wp.array[float],
    restart: wp.array[int],
):
    # state: [iteration, omega, previous sweep step norm^2]. On device so the
    # restart decision needs no host sync and the sweep loop stays graph-safe.
    k = int(state[0]) + 1
    j = k - start
    w = float(1.0)
    if j == 2:
        w = 2.0 / (2.0 - rho2)
    elif j > 2:
        w = 4.0 / (4.0 - rho2 * state[1])
    do_restart = int(0)
    if restart_enabled != 0 and k > 1 and norm2[0] > restart_growth * state[2]:
        do_restart = 1
        w = 1.0
        k = start
    state[0] = float(k)
    state[1] = w
    state[2] = norm2[0]
    omega[0] = w
    restart[0] = do_restart


@wp.kernel
def _blend(
    omega: wp.array[float],
    restart: wp.array[int],
    gamma: float,
    previous: wp.array[wp.vec3],
    older: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    displacements: wp.array[wp.vec3],
):
    p = wp.tid()
    swept = positions[p]
    prev = previous[p]
    old = older[p]
    if restart[0] != 0:
        accelerated = prev + gamma * (swept - prev)
        older[p] = accelerated
    else:
        accelerated = old + omega[0] * (gamma * (swept - prev) + prev - old)
        older[p] = prev
    previous[p] = accelerated
    # One writer owns each particle; assign instead of +=, which lowers to an atomic add.
    displacements[p] = displacements[p] + (accelerated - swept)
    positions[p] = accelerated


class SolverVBDChebyshev(SolverVBD):
    """VBD with Chebyshev acceleration of the particle iteration.

    Args:
        model: Newton Model.
        chebyshev_rho: Estimated spectral radius of the sweep, in [0, 1).
        chebyshev_gamma: Sweep under-relaxation in (0, 1]. 1 is the plain scheme.
        chebyshev_start: Plain sweeps at the start of each substep before the
            acceleration begins.
        chebyshev_restart: Drop the extrapolation history when the sweep step
            norm grows by more than ``chebyshev_restart_growth`` between
            consecutive iterations.
        chebyshev_restart_growth: Growth factor that triggers a restart.
        **kwargs: Passed unchanged to Newton's SolverVBD.
    """

    def __init__(
        self,
        model,
        *,
        chebyshev_rho: float = 0.9,
        chebyshev_gamma: float = 1.0,
        chebyshev_start: int = 0,
        chebyshev_restart: bool = False,
        chebyshev_restart_growth: float = 1.0,
        **kwargs,
    ):
        if not 0.0 <= chebyshev_rho < 1.0:
            raise ValueError("chebyshev_rho must be in [0, 1)")
        if not 0.0 < chebyshev_gamma <= 1.0:
            raise ValueError("chebyshev_gamma must be in (0, 1]")
        if chebyshev_start < 0 or chebyshev_restart_growth <= 0.0:
            raise ValueError("chebyshev_start must be >= 0 and chebyshev_restart_growth > 0")

        super().__init__(model, **kwargs)
        if self.rigid_soft_enable_dat:
            raise ValueError("Chebyshev acceleration does not support rigid-soft DAT")

        self.chebyshev_rho = float(chebyshev_rho)
        self.chebyshev_gamma = float(chebyshev_gamma)
        self.chebyshev_start = int(chebyshev_start)
        self.chebyshev_restart = bool(chebyshev_restart)
        self.chebyshev_restart_growth = float(chebyshev_restart_growth)
        device = model.device
        self._previous = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)
        self._older = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)
        self._norm2 = wp.zeros(1, dtype=float, device=device)
        self._state = wp.zeros(3, dtype=float, device=device)
        self._omega = wp.ones(1, dtype=float, device=device)
        self._restart = wp.zeros(1, dtype=int, device=device)

    def _initialize_particles(self, state_in, state_out, contacts, dt):
        super()._initialize_particles(state_in, state_out, contacts, dt)
        wp.copy(self._previous, state_in.particle_q)
        wp.copy(self._older, state_in.particle_q)
        self._norm2.zero_()
        self._state.zero_()
        self._omega.fill_(1.0)
        self._restart.zero_()

    def _solve_particle_iteration(self, state_in, state_out, contacts, dt):
        super()._solve_particle_iteration(state_in, state_out, contacts, dt)
        device = self.model.device

        if self.chebyshev_restart:
            self._norm2.zero_()
            wp.launch(
                _sweep_norm,
                dim=self.model.particle_count,
                inputs=[state_in.particle_q, self._previous, self._norm2],
                device=device,
            )
        wp.launch(
            _update_schedule,
            dim=1,
            inputs=[
                self._norm2,
                self.chebyshev_rho * self.chebyshev_rho,
                self.chebyshev_start,
                int(self.chebyshev_restart),
                self.chebyshev_restart_growth,
                self._state,
                self._omega,
                self._restart,
            ],
            device=device,
        )
        wp.launch(
            _blend,
            dim=self.model.particle_count,
            inputs=[
                self._omega,
                self._restart,
                self.chebyshev_gamma,
                self._previous,
                self._older,
                state_in.particle_q,
                self.particle_displacements,
            ],
            device=device,
        )
        truncate = getattr(self, "_penetration_free_truncation", None)
        if truncate is not None:
            truncate(state_in, contacts)
        wp.copy(state_out.particle_q, state_in.particle_q)