# SPDX-License-Identifier: Apache-2.0
"""ROBOSIMSTUDIO_PLANNER_CACHE on PyRoKi: one compiled solve per arm per process.

The jaxls build is stubbed: what is tested is which solvers share a build.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from robosimstudio.solvers.ik import pyroki
from robosimstudio.solvers.ik.pyroki import COMPILED_CACHE_ENV, IKSolverPyroki

TIER = "fast"


def _solver(urdf: str = "yam.urdf", t_ht: np.ndarray | None = None) -> IKSolverPyroki:
    solver = IKSolverPyroki.__new__(IKSolverPyroki)
    solver.urdf = urdf
    solver._link = 6
    solver._slots = np.arange(6)
    solver._n_actuated = 8
    solver._t_ht = np.eye(4) if t_ht is None else t_ht
    solver._solvers = {}
    return solver


class CompiledCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builds: list[tuple[int, str]] = []

        def build(solver, worlds, roll):
            self.builds.append((worlds, roll))
            return object()

        for patch in (
            mock.patch.object(IKSolverPyroki, "_build", build),
            mock.patch.dict(pyroki._COMPILED, clear=True),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def test_without_the_cache_every_solver_compiles_its_own(self) -> None:
        with mock.patch.dict(os.environ, {COMPILED_CACHE_ENV: ""}):
            first, second = _solver(), _solver()
            self.assertIsNot(first._solver(1, "free"), second._solver(1, "free"))
        self.assertEqual(len(self.builds), 2)

    def test_the_next_episodes_solver_runs_the_compiled_one(self) -> None:
        with mock.patch.dict(os.environ, {COMPILED_CACHE_ENV: "1"}):
            first, second = _solver(), _solver(t_ht=np.eye(4) + 1e-12)
            self.assertIs(first._solver(1, "free"), second._solver(1, "free"))
            self.assertIs(first._solver(1, "held"), second._solver(1, "held"))
            self.assertIsNot(first._solver(1, "free"), first._solver(1, "held"))
        self.assertEqual(self.builds, [(1, "free"), (1, "held")])

    def test_another_arm_or_calibration_is_another_compile(self) -> None:
        moved = np.eye(4)
        moved[0, 3] = 0.001
        with mock.patch.dict(os.environ, {COMPILED_CACHE_ENV: "1"}):
            base = _solver()._solver(1, "free")
            self.assertIsNot(_solver(urdf="panda.urdf")._solver(1, "free"), base)
            self.assertIsNot(_solver(t_ht=moved)._solver(1, "free"), base)
        self.assertEqual(len(self.builds), 3)


if __name__ == "__main__":
    unittest.main()
