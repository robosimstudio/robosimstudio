# SPDX-License-Identifier: Apache-2.0
"""`planned: true` is a gate, not an arrival, and the gate is adjustable.

`IKBackend.tolerance` defaults to 30 mm. A hammer's handle is 36 mm across, so
a plan can pass the default gate and put the fingers beside the object rather
than around it -- measured on the hammer grid at 21-24 mm on the episodes
that failed, against 1.7-2.0 mm on the ones that worked. Until this was
exposed a policy could neither ask for tighter nor find out what it got.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio_policy_agent.ik import IKBackend

TIER = "fast"
"""A stubbed solver behind a real backend: no GPU, no scene."""


class _Solver:
    """A solver that lands every solve ``error`` metres short, whatever it is asked."""

    capabilities = frozenset()

    class kinematics:
        dof = 1
        arm_coords = (0,)

    def __init__(self, error):
        self._error = error

    def solve(self, seed, position, quat_xyzw, *, roll="free", active=None, iterations=0, template=None):
        seed = np.asarray(seed, dtype=np.float64).reshape(-1, 1)
        return seed, np.tile([self._error, 0.0], (seed.shape[0], 1))


class _Bus:
    def call(self, fn, *args):
        return fn(*args)


def _backend(error):
    b = IKBackend.__new__(IKBackend)
    b.__dict__.update(
        env=None,
        bus=_Bus(),
        planner=None,
        prefer_planned=True,
        tolerance=0.03,
        waypoints=4,
        iterations=4,
        fallback=True,
        family="fake",
        roll="free",
        honours_roll=False,
        last_solve_error=None,
        last_position_error=None,
        last_refusal="",
        _own_refusal="",
        _solvers={"arm": _Solver(error)},
    )
    b._arm_name = lambda arm_id: "arm"
    b._seed = lambda arm_id, seed_joints: (np.zeros((1, 1)), np.zeros(1))
    return b


POSE = {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


class TestTheGateIsAdjustable(unittest.TestCase):
    def test_a_20mm_miss_passes_the_default_gate(self):
        self.assertIsNotNone(_backend(0.020).plan_linear(POSE, POSE))

    def test_and_is_refused_when_the_caller_asks_for_5mm(self):
        self.assertIsNone(_backend(0.020).plan_linear(POSE, POSE, tolerance=0.005))

    def test_a_plan_that_passes_still_reports_how_far_it_landed(self):
        b = _backend(0.020)
        b.plan_linear(POSE, POSE)
        self.assertAlmostEqual(b.last_position_error, 0.020, places=6)


if __name__ == "__main__":
    unittest.main()
