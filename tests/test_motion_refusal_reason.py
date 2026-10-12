# SPDX-License-Identifier: Apache-2.0
"""A refused linear plan says which of three different things went wrong.

`plan_linear` answers `planned: false` rather than raising, so a policy can
route an unreachable pose to its own exit. What it could not do until now is
say *why*, and the three causes want opposite fixes: an out-of-reach target has
to move, a reachable target at the end of an impossible straight line wants
`motion.plan_joint`, and a wrist pinned by `orientation="lock"` wants the lock
dropped. One study iteration was spent guessing between them, at a cost of
twelve episodes.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio_policy_agent.ik import IKBackend
from robosimstudio_policy_agent.motion_tools import MotionTools

TIER = "fast"
"""Pure functions and a stubbed backend: no GPU, no scene."""


class _Backend:
    """Just enough of a solver to answer a plan_linear call."""

    honours_roll = False

    def __init__(self, plan=None, refusal=""):
        self._plan = plan
        self.last_refusal = refusal

    def plan_linear(self, start, end, **kwargs):
        return self._plan

    def current_pose(self, arm_id=0):
        return {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


class _Mute(_Backend):
    """A backend that refuses without explaining — curobo, pyroki, a stub."""

    def __init__(self):
        super().__init__(plan=None)
        del self.last_refusal


_END = {"position": {"x": 0.4, "y": 0.0, "z": 0.3}, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


class TestTheRefusalCarriesTheBackendsReason(unittest.TestCase):
    def test_the_solvers_own_account_reaches_the_caller(self):
        detail = (
            "orientation='lock' holds the end pose's whole rotation along "
            "the leg; at waypoint 3 of 40 the solver was 2 mm and 41 deg off"
        )
        tools = MotionTools(_Backend(refusal=detail))
        result = tools.plan_linear(end=_END)
        self.assertFalse(result["planned"])
        self.assertEqual(result["reason"], detail)

    def test_a_backend_that_does_not_explain_keeps_the_generic_reason(self):
        tools = MotionTools(_Mute())
        result = tools.plan_linear(end=_END)
        self.assertFalse(result["planned"])
        self.assertEqual(result["reason"], "no linear plan holds this move's constraint")

    def test_a_plan_that_succeeded_carries_no_reason(self):
        plan = {"waypoints": [{"positions": [0.0]}]}
        # Stale text from an earlier refusal must not be read as this call's.
        tools = MotionTools(_Backend(plan=plan, refusal="something from before"))
        result = tools.plan_linear(end=_END)
        self.assertTrue(result["planned"])
        self.assertEqual(result["reason"], "")


class _StubSolver:
    """A solver that always lands a fixed miss."""

    capabilities = frozenset()

    class kinematics:
        dof = 1
        arm_coords = (0,)

    def __init__(self, position_error, rotation_error):
        self._errors = (position_error, rotation_error)

    def solve(self, seed, position, quat_xyzw, *, roll="free", active=None, iterations=0, template=None):
        seed = np.asarray(seed, dtype=np.float64).reshape(-1, 1)
        return seed, np.tile(self._errors, (seed.shape[0], 1))


class _Bus:
    def call(self, fn, *args):
        return fn(*args)


def _solver(position_error, rotation_error):
    """An IKBackend over a stub solver, without an env."""
    backend = IKBackend.__new__(IKBackend)
    backend.__dict__.update(
        env=None,
        bus=_Bus(),
        planner=None,
        prefer_planned=True,
        tolerance=0.03,
        waypoints=10,
        iterations=4,
        fallback=True,
        family="fake",
        roll="free",
        honours_roll=False,
        last_solve_error=None,
        last_position_error=None,
        last_refusal="",
        _own_refusal="",
        _solvers={"arm": _StubSolver(position_error, rotation_error)},
    )
    backend._arm_name = lambda arm_id: "arm"
    backend._seed = lambda arm_id, seed_joints: (np.zeros((1, 1)), np.zeros(1))
    return backend


_POSE = {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


class TestTheHeldRotationRefusalNamesTheBindingConstraint(unittest.TestCase):
    """`orientation="lock"` is the cause a caller is least able to guess.

    Six of twelve episodes in one study iteration died here, and the agent
    spent the next iteration inferring from node names alone that the wrist
    was the problem.
    """

    def _refuse(self, position_error, rotation_error):
        backend = _solver(position_error, rotation_error)
        result = backend.plan_linear(_POSE, _POSE, hold_rotation=True)
        self.assertIsNone(result)
        return backend.last_refusal

    def test_a_rotation_that_cannot_be_held_says_so(self):
        reason = self._refuse(position_error=0.001, rotation_error=0.7)
        self.assertIn("orientation='lock'", reason)
        self.assertIn("the held rotation", reason)
        self.assertIn("40 deg", reason)  # 0.7 rad

    def test_a_position_miss_is_not_blamed_on_the_rotation(self):
        reason = self._refuse(position_error=0.25, rotation_error=0.0)
        self.assertIn("the position", reason)
        self.assertNotIn("the held rotation is what", reason)

    def test_both_missing_is_reported_as_both(self):
        reason = self._refuse(position_error=0.25, rotation_error=0.7)
        self.assertIn("both position and the held rotation", reason)

    def test_it_names_which_waypoint_of_how_many(self):
        reason = self._refuse(position_error=0.25, rotation_error=0.0)
        self.assertIn("waypoint 1 of 10", reason)


if __name__ == "__main__":
    unittest.main()
