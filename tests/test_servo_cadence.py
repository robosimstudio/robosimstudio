# SPDX-License-Identifier: Apache-2.0
"""A serial servo turn counts the simulator steps it spans, and drives the law that many times.

:meth:`CartesianServo.move` budgets in simulator steps, read off the bus's
published ``snapshot.step``. It used to count one per turn that saw a *new*
step, however many had passed: on a loaded host (three processes per GPU) the
simulator ran three steps between the worker's turns, so a 400-turn leg cost
exactly 1200 simulator steps, and the setpoint law, advanced once per turn at
the control period, moved the arm at a third of its speed. Both the budget and
the arm's speed were functions of host load.

The batched runner already counts ``cadence`` per readback and advances the law
``cadence`` times from the one measurement (``fleet/stepper.py``,
``_advance_lead`` and the servo branch of ``_finished``). These tests hold the
serial loop to the same rule, with a bus stand-in that steps ``k`` times
between the worker's looks and no simulator behind it.
"""

from __future__ import annotations

import itertools
import types
import unittest
from unittest import mock

import numpy as np

import robosimstudio_policy_agent.servo as servo_module
from robosimstudio.solvers import OscSetpoint
from robosimstudio_policy_agent.servo import CartesianServo

TIER = "fast"

CONTROL_HZ = 20.0
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


class _Bus:
    """A bus whose simulator runs on between the worker's looks.

    Every read of ``snapshot`` finds the counter ``per_turn`` steps further on,
    which is what a loaded host does to a serial loop: the worker comes back
    from its turn and the world has moved several steps, not one. ``per_turn``
    may be negative (a counter that went backwards) or ``None`` (a bus with no
    counter at all); ``call`` runs inline, as the contract tests' bus does.
    """

    def __init__(self, per_turn: int | None, start: int = 100) -> None:
        self.per_turn = per_turn
        self.step = None if per_turn is None else start
        self.reads = 0

    @property
    def snapshot(self):
        self.reads += 1
        if self.step is not None:
            self.step += self.per_turn
        return types.SimpleNamespace(step=self.step, body_q=None)

    def call(self, fn, *a, **kw):
        return fn(*a, **kw)


class _Facade:
    def __init__(self, bus: _Bus) -> None:
        self.bus = bus
        self._held = {0: np.zeros(1)}
        self.applied: list[np.ndarray] = []

    def apply_joint_action(self, joints, arm_id=0) -> None:
        del arm_id
        self.applied.append(np.asarray(joints))


class _IK:
    """Solves everything, roll-free, and keeps every setpoint it was handed."""

    honours_roll = False

    def __init__(self, solves: bool = True) -> None:
        self.requests: list[dict] = []
        self.solves = solves

    def solve_ik(self, pose, *, arm_id=0, seed_joints=None):
        del arm_id, seed_joints
        self.requests.append(pose)
        return [0.0] if self.solves else None


class _Rig:
    """A servo built without ``__init__``, as the contract tests build one.

    ``tracking`` makes the measured arm an ideal one: it stands wherever the
    last setpoint told it to, so the setpoint's travel per turn is the arm's.
    Otherwise the arm never moves, which is a blocked one.
    """

    def __init__(self, per_turn: int | None, *, tracking: bool = False, solves: bool = True) -> None:
        servo = CartesianServo.__new__(CartesianServo)
        servo.law = OscSetpoint(control_dt=1.0 / CONTROL_HZ)
        servo.honours_roll = False
        servo._scaled = {}
        servo.env = types.SimpleNamespace(
            scene=types.SimpleNamespace(embodiment=types.SimpleNamespace(iks={"arm": None}))
        )
        self.bus = _Bus(per_turn)
        self.facade = servo.facade = _Facade(self.bus)
        self.ik = servo.ik = _IK(solves)
        self.servo = servo

        self.measures = 0
        self.advances_by_turn: list[int] = []

        def measure(_name):
            self.measures += 1
            if tracking and self.ik.requests:
                p = self.ik.requests[-1]["position"]
                return np.array([p["x"], p["y"], p["z"]]), IDENTITY.copy()
            return np.zeros(3), IDENTITY.copy()

        servo._measure = measure

        advance = servo.law.advance

        def counted_advance(p, q, gp, gq):
            # One measurement per turn, so the turn a tick belongs to is the
            # measurement count at the time, less the seeding read before the
            # loop and one for a zero-based index.
            turn = self.measures - 2
            while len(self.advances_by_turn) < turn + 1:
                self.advances_by_turn.append(0)
            self.advances_by_turn[turn] += 1
            return advance(p, q, gp, gq)

        servo.law.advance = counted_advance

    @property
    def turns(self) -> int:
        """Loop turns taken: one measurement each, after the seeding read."""
        return self.measures - 1

    @property
    def per_step(self) -> float:
        """How far the law moves the setpoint in one control period [m]."""
        return self.servo.law.pos_speed * self.servo.law.control_dt

    @property
    def sent_x(self) -> list[float]:
        return [float(r["position"]["x"]) for r in self.ik.requests]

    def move(self, distance: float, **kw):
        pose = {"position": {"x": distance, "y": 0.0, "z": 0.0}, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}
        return self.servo.move(pose, **kw)


class TestATurnCountsTheStepsItSpans(unittest.TestCase):
    def test_three_steps_per_turn_count_three_and_advance_the_law_three_times(self):
        """A blocked arm, a far goal, k = 3: 400 simulator steps in 134 turns, not 400."""
        rig = _Rig(3)
        result = rig.move(1.0, max_steps=400)

        self.assertEqual(result["outcome"], "timeout")
        self.assertEqual(result["steps"], 400)
        # First turn counts one (nothing to difference against), then three each:
        # 1 + 3 x 133 = 400.
        self.assertEqual(rig.turns, 134)
        self.assertEqual(rig.advances_by_turn[0], 1)
        self.assertEqual(set(rig.advances_by_turn[1:]), {3})
        self.assertEqual(sum(rig.advances_by_turn), 400)

        # The setpoint sent is the LAST of the k ticks. From a blocked arm the
        # first turn leads by one period; the second's three ticks would run to
        # four periods and are clamped to the lead, and nothing creeps after.
        lead = rig.servo.law.pos_lead
        self.assertGreater(4 * rig.per_step, lead)  # four periods overrun the lead at 20 Hz, so the clamp binds
        self.assertAlmostEqual(rig.sent_x[0], rig.per_step, places=12)
        self.assertAlmostEqual(rig.sent_x[1], min(4 * rig.per_step, lead), places=12)
        for x in rig.sent_x[2:]:
            self.assertAlmostEqual(x, lead, places=12)

    def test_the_setpoint_travels_k_control_periods_per_turn(self):
        """On an arm that keeps up, each turn moves the command 3 x pos_speed x dt, clamped by the lead."""
        rig = _Rig(3, tracking=True)
        result = rig.move(10.0, max_steps=60)

        self.assertEqual(result["outcome"], "timeout")
        self.assertGreaterEqual(result["steps"], 60)
        self.assertEqual(result["steps"], 1 + 3 * (rig.turns - 1))

        expected = min(3 * rig.per_step, rig.servo.law.pos_lead)
        sent = rig.sent_x
        self.assertAlmostEqual(sent[0], rig.per_step, places=12)
        for before, after in itertools.pairwise(sent[1:]):
            self.assertAlmostEqual(after - before, expected, places=12)
        # And the arm covers pos_speed x dt per *simulator step* -- the number
        # the law's docstring promises -- rather than per turn.
        self.assertAlmostEqual(sent[-1], rig.per_step * result["steps"], places=9)

    def test_the_same_leg_costs_the_same_simulator_steps_at_any_k(self):
        """k = 1 and k = 3 arrive at the same simulator step, in a third of the turns."""
        one = _Rig(1, tracking=True)
        three = _Rig(3, tracking=True)
        r1 = one.move(1.0, max_steps=400)
        r3 = three.move(1.0, max_steps=400)

        self.assertEqual(r1["outcome"], "reached")
        self.assertEqual(r3["outcome"], "reached")
        # Arrival is tested on a readback, and a readback at step s measures the
        # travel commanded through step s - k, so k = 3 registers arrival up to
        # three steps after k = 1 does (70 against 67 at 20 Hz).
        self.assertLessEqual(abs(r3["steps"] - r1["steps"]), 3)
        self.assertEqual(r1["steps"], one.turns)
        self.assertEqual(r3["steps"], 1 + 3 * (three.turns - 1))
        # The defect had the k = 3 arm needing as many *turns* as the k = 1 arm,
        # i.e. three times the simulator steps.
        self.assertLess(three.turns, one.turns / 2)

    def test_one_step_per_turn_is_unchanged(self):
        """The unloaded case is today's behaviour, number for number."""
        rig = _Rig(1)
        result = rig.move(1.0, max_steps=400)

        self.assertEqual(result["outcome"], "timeout")
        self.assertEqual(result["steps"], 400)
        self.assertEqual(rig.turns, 400)
        self.assertEqual(rig.advances_by_turn, [1] * 400)
        lead = rig.servo.law.pos_lead
        for n in (1, 2, 3):
            self.assertAlmostEqual(rig.sent_x[n - 1], min(n * rig.per_step, lead), places=12)
        self.assertAlmostEqual(rig.sent_x[3], lead, places=12)  # 4 x 15.3 mm is past the 50 mm lead

    def test_a_bus_without_a_step_counter_counts_one_per_turn(self):
        """A synchronous harness publishes no step; the loop counts its turns, as it always did."""
        rig = _Rig(None)
        result = rig.move(1.0, max_steps=6)
        self.assertEqual(result["outcome"], "timeout")
        self.assertEqual(result["steps"], 6)
        self.assertEqual(rig.turns, 6)
        self.assertEqual(rig.advances_by_turn, [1] * 6)

    def test_a_counter_that_went_backwards_is_one_step_not_a_negative_number(self):
        rig = _Rig(-5)
        result = rig.move(1.0, max_steps=4)
        self.assertEqual(result["steps"], 4)
        self.assertEqual(rig.turns, 4)
        self.assertEqual(rig.advances_by_turn, [1] * 4)

    def test_a_turn_released_by_the_stall_escape_still_counts_one(self):
        """A bus that publishes but never advances is waited on, then counted once per escape."""
        rig = _Rig(0)
        with mock.patch.object(servo_module, "STALL_ESCAPE_S", 0.005):
            result = rig.move(1.0, max_steps=5)
        self.assertEqual(result["outcome"], "timeout")
        self.assertEqual(result["steps"], 5)
        self.assertEqual(rig.turns, 5)
        self.assertEqual(rig.advances_by_turn, [1] * 5)


class TestTheBudgetIsNeverBelowOneStep(unittest.TestCase):
    def test_a_zero_or_negative_budget_is_the_default(self):
        """``int(max_steps) or MAX_STEPS`` let a negative through: a move that ended before its first turn."""
        for given in (0, -5):
            rig = _Rig(1)
            with mock.patch.object(servo_module, "MAX_STEPS", 7):
                result = rig.move(1.0, max_steps=given)
            self.assertEqual(result["outcome"], "timeout")
            self.assertEqual(result["steps"], 7)
            self.assertGreaterEqual(result["remaining_distance"], 0.0)


class TestTheStallWindowIsInSimulatorSteps(unittest.TestCase):
    def test_contact_comes_after_stall_steps_simulator_steps_not_turns(self):
        """``stall_steps=12`` is twelve simulator steps without progress at any k.

        The first turn is the baseline (it sets ``best_dist``; nothing has
        stalled yet), so a blocked arm reports contact on the turn that carries
        the count past baseline + 12: simulator step 13 at k = 1 and at k = 3
        alike -- the thirteenth turn unloaded, the fifth at k = 3. The defect
        took thirteen turns at k = 3 too, which was 39 simulator steps.
        """
        one = _Rig(1)
        three = _Rig(3)
        r1 = one.move(1.0, max_steps=400, stop_on_stall=True, stall_steps=12)
        r3 = three.move(1.0, max_steps=400, stop_on_stall=True, stall_steps=12)

        self.assertEqual(r1["outcome"], "contact")
        self.assertEqual(r3["outcome"], "contact")
        self.assertEqual(r1["steps"], 13)
        self.assertEqual(r3["steps"], 13)
        self.assertEqual(one.turns, 13)
        self.assertEqual(three.turns, 5)

    def test_progress_resets_the_window_in_steps_too(self):
        """An arm that keeps up never stalls, however many steps a turn spans."""
        rig = _Rig(3, tracking=True)
        result = rig.move(1.0, max_steps=400, stop_on_stall=True, stall_steps=12)
        self.assertEqual(result["outcome"], "reached")


class TestUnreachableIsStillThreeConsecutiveSolveFailures(unittest.TestCase):
    def test_three_failed_turns_raise_whatever_k_is(self):
        """The solver's verdict is per attempt, not per simulator step."""
        for per_turn in (1, 3):
            with self.subTest(per_turn=per_turn):
                rig = _Rig(per_turn, solves=False)
                with self.assertRaises(RuntimeError):
                    rig.move(1.0, max_steps=400)
                self.assertEqual(rig.turns, 3)


if __name__ == "__main__":
    unittest.main()
