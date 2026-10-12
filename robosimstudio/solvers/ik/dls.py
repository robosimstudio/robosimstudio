# SPDX-License-Identifier: Apache-2.0
"""The damped-least-squares reference: today's host solver behind the solver protocol.

:class:`IKSolverDLS` answers :class:`~robosimstudio.solvers.ik.solver.IKSolver`
with exactly the solves the graphs were measured on -- :meth:`ArmIK.solve_q`
(position plus approach axis, wrist roll free) and :meth:`ArmIK.solve_q_full`
(the roll closed on the last joint afterwards) for a batch of one, and their
batched twins in :mod:`~robosimstudio.solvers.ik.fleet_dls` for a wave -- so
every other family is an A/B against it and not against a memory. Its cost is
the reason the consolidation exists: nine finite-difference FK evaluations per
iteration, 22 ms per solve on the device path, 254.8 ms of a 255.7 ms wave
step before the fusers. It ships to be compared with, not to be chosen.
"""

from __future__ import annotations

import numpy as np

from .arm import ArmKinematics
from .fleet_dls import pose_errors_batched, solve_q_batched, solve_q_full_batched
from .solver import Roll, approach_axes, pose_errors

__all__ = ["IKSolverDLS"]

DEFAULT_ITERATIONS = 300
"""``ArmIK.solve_q``'s own budget."""


class _Poses:
    """The ``fk.poses`` object the batched DLS walks, over an :class:`ArmKinematics`."""

    def __init__(self, kinematics: ArmKinematics, template: np.ndarray | None) -> None:
        self.kinematics = kinematics
        self.template = template

    def poses(self, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.kinematics.poses(arm_q, self.template)


class IKSolverDLS:
    """Damped least squares over finite-difference Jacobians, batched over worlds.

    Args:
        kinematics: The arm.
        iterations: The default budget per solve; ``ArmIK.solve_q``'s 300.
    """

    capabilities = frozenset({"batched"})

    def __init__(self, kinematics: ArmKinematics, *, iterations: int = DEFAULT_ITERATIONS) -> None:
        self.kinematics = kinematics
        self.iterations = int(iterations)

    def solve(
        self,
        seed: np.ndarray,
        position: np.ndarray,
        quat_xyzw: np.ndarray,
        *,
        roll: Roll = "free",
        active: np.ndarray | None = None,
        iterations: int = 0,
        template: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        kin = self.kinematics
        seed = np.asarray(seed, dtype=np.float64).reshape(-1, kin.dof)
        n_worlds = seed.shape[0]
        position = np.asarray(position, dtype=np.float64).reshape(n_worlds, 3)
        quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(n_worlds, 4)
        budget = int(iterations) or self.iterations
        live = np.ones(n_worlds, dtype=bool) if active is None else np.asarray(active, dtype=bool)
        arm = kin.arm
        if n_worlds == 1 and arm is not None and kin.fk is None:
            # A batch of one on the arm's own solve: the exact path the serial
            # backend runs, so a serial parity test compares like with like.
            if not live[0]:
                return seed.copy(), np.zeros((1, 2))
            full = kin.full(seed[0], template)
            if roll == "held":
                solved = arm.solve_q_full(full, position[0], quat[0], iterations=budget)
            else:
                solved = arm.solve_q(full, position[0], approach_axes(quat)[0], iterations=budget)
            solved = np.asarray(solved, dtype=np.float64)
            q = solved[kin.arm_coords].reshape(1, -1)
            pos, got = kin.poses(q, template)
            return q, pose_errors(pos, got, position, quat, roll=roll)
        fk = _Poses(kin, template)
        if roll == "held":
            q = solve_q_full_batched(
                fk, seed, position, quat, lower=kin.lower, upper=kin.upper, active=live, iterations=budget
            )
            distance, angle = pose_errors_batched(fk, q, position, quat)
            return q, np.stack([distance, angle], axis=1)
        q = solve_q_batched(
            fk, seed, position, approach_axes(quat), lower=kin.lower, upper=kin.upper, active=live, iterations=budget
        )
        pos, got = fk.poses(q)
        return q, pose_errors(pos, got, position, quat, roll="free")
