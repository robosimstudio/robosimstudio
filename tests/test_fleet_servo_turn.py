# SPDX-License-Identifier: Apache-2.0
"""A fleet servo turns at the serial servo's cadence: re-aimed and judged once per two steps.

Under the lockstep clock a serial :meth:`CartesianServo.move` turn is one bus
call that steps once, then the rollout's hold step: it measures once, advances
the setpoint law twice from that one measurement, and commands once. The
fleet's stepper re-aimed every step from a fresh measurement, so the lead clamp
bound later and the arm travelled faster -- measured on ``cable_route/port_easy``,
go_to_pose targets reached in a median 0.81 of the serial step count.

:attr:`ServoToPose.steps_per_turn` (2) restores the serial turn. These tests
pin the arithmetic on a stub stepper: no device, no simulator.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robosimstudio_policy_agent.fleet import ServoToPose, stepper

TIER = "fast"

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


class _Lead:
    """Counts advances and records the measurement each was taken from."""

    def __init__(self) -> None:
        self.advances: list[tuple[int, float]] = []
        self.tick = 0

    def advance(self, measured_pos, measured_quat, goal, quat):
        self.advances.append((self.tick, float(measured_pos[0])))
        # The setpoint carries the measurement it was taken from, so a test can
        # tell which turn's setpoint reached the solver when.
        return np.asarray(measured_pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)


class _Solver:
    def __init__(self) -> None:
        self.targets: list[tuple[int, float]] = []
        self.tick = 0

    def set_target(self, world, pos, quat) -> None:
        self.targets.append((self.tick, round(float(pos[0]), 6)))

    def enable_world(self, world, value) -> None:
        pass


def _service(solver):
    service = object.__new__(stepper.StepService)
    service.arm_names = ["arm"]
    service.cadence = 1
    service.step = 0
    service.fleet = SimpleNamespace(
        scene=SimpleNamespace(embodiment=SimpleNamespace(arms={"arm": SimpleNamespace(ee_body=0)})),
        ik=SimpleNamespace(arms={"arm": solver}),
    )
    service._offsets = lambda name: ()
    service._release_arm = lambda world, name: None
    service._beyond_reach = lambda *a: False
    return service


def _program(turn: int, *, deadline: int = 1000, goal_x: float = 1.0):
    return stepper._Program(
        "servo",
        goal=(np.array([goal_x, 0.0, 0.0]), "arm", IDENTITY),
        deadline=deadline,
        tolerance=0.004,
        lead=_Lead(),
        requested_quat=IDENTITY,
        solver_quat=IDENTITY,
        turn=turn,
    )


def _run(service, program, ticks: int, x_of_tick):
    """Settle's per-program body for ``ticks`` steps, the TCP at ``x_of_tick(t)``."""
    snapshot = SimpleNamespace(body=lambda world: None)
    solver = service.fleet.ik.arms["arm"]
    for tick in range(1, ticks + 1):
        service.step = tick
        program.lead.tick = solver.tick = tick
        pose = (np.array([x_of_tick(tick), 0.0, 0.0]), IDENTITY)
        with mock.patch.object(stepper, "tcp_pose", lambda *a, pose=pose: pose):
            service._count_turn(program)
            service._advance_lead(0, program, snapshot)
            done, _error, result = service._finished(0, 0, program, snapshot)
        if done:
            return tick, result
    return None, None


class TestTheIntent(unittest.TestCase):
    def test_the_default_turn_is_the_serial_one(self):
        intent = ServoToPose(arm=0, position=np.zeros(3), quat_xyzw=IDENTITY)
        self.assertEqual(intent.steps_per_turn, 2)


class TestATurnOfTwo(unittest.TestCase):
    def test_the_lead_is_re_aimed_every_second_step_from_one_measurement(self):
        solver = _Solver()
        service = _service(solver)
        program = _program(2)
        _run(service, program, 7, lambda t: 0.1 * t)
        # Once per turn, and a hold after the measurement it came from: the
        # setpoint taken at the end of step 2 is commanded for step 4 (set
        # after step 3), as a serial command is serviced after the hold step.
        self.assertEqual(solver.targets, [(3, 0.2), (5, 0.4), (7, 0.6)])
        # Two advances per turn, both from the measurement that ended it.
        self.assertEqual(
            [(t, round(x, 6)) for t, x in program.lead.advances],
            [(2, 0.2), (2, 0.2), (4, 0.4), (4, 0.4), (6, 0.6), (6, 0.6)],
        )

    def test_arrival_is_judged_at_the_end_of_a_turn(self):
        service = _service(_Solver())
        program = _program(2, goal_x=0.0)
        tick, result = _run(service, program, 6, lambda t: 0.0)  # at the goal from the start
        self.assertEqual(tick, 2, "not mid-turn")
        self.assertEqual(result["outcome"], "reached")
        self.assertEqual(result["steps"], 2, "a turn counts every step it spans")

    def test_the_stall_window_is_still_in_steps(self):
        service = _service(_Solver())
        program = _program(2)
        program.stop_on_stall = True
        program.stall_steps = 12
        tick, result = _run(service, program, 40, lambda t: 0.5)  # never moves
        self.assertEqual(result["outcome"], "contact")
        # First turn sets best_dist; the window then fills two steps a turn.
        self.assertEqual(tick, 14)
        self.assertEqual(result["steps"], 14)

    def test_the_budget_ends_a_move_mid_turn(self):
        service = _service(_Solver())
        program = _program(2, deadline=5)
        tick, result = _run(service, program, 10, lambda t: 0.5)
        self.assertEqual(tick, 5)
        self.assertEqual(result["outcome"], "timeout")


class TestATurnOfOneIsTheOldStepper(unittest.TestCase):
    def test_every_step_re_aims_and_judges(self):
        solver = _Solver()
        service = _service(solver)
        program = _program(1, goal_x=0.0)
        _run(service, program, 3, lambda t: 0.5)
        self.assertEqual([t for t, _x in solver.targets], [1, 2, 3], "commanded at once")
        self.assertEqual([t for t, _x in program.lead.advances], [1, 2, 3])
        tick, result = _run(_service(_Solver()), _program(1, goal_x=0.0), 4, lambda t: 0.0)
        self.assertEqual((tick, result["steps"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
