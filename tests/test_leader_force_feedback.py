# SPDX-License-Identifier: Apache-2.0
"""Leader force feedback: the mismatch signal and the per-joint torque gates.

:mod:`robosimstudio.robots.teleop.force_feedback` renders the gap between a
leader arm and the simulated arm it drags as torque on the leader's own servos.
This module is the pure half of that -- the arithmetic and the state machine,
with no bus in sight:

* the **signal** -- deadband, ramp and cap on one joint's mismatch, and which
  way the resulting correction points;
* the **gates** -- which motors energize, which let go, and the dwell a
  suspended joint has to serve before it may come back;
* the **layout** -- the ``[q_0 .. q_{n-1}, grip]`` slots every one of those
  indexes by, and the tick conversions at each end of it.

The servo writes are in ``test_leader_force_feedback_servos.py``, and the
two-clock exchange in ``test_leader_force_feedback_session.py``.
"""

from __future__ import annotations

import math
import unittest

from robosimstudio.robots.teleop import force_feedback as ffb
from robosimstudio.robots.teleop.lerobot_controller import (
    ArmCalibration,
    GripperCalibration,
    JointCalibration,
    raw_delta,
)
from tests.fixtures.leader_feedback import layout as _layout

TIER = "fast"
"""No GPU, no scene, no serial port."""


class TestTickInverses(unittest.TestCase):
    """The conversions the feedback needs that reading a leader never did."""

    def test_a_joint_angle_round_trips_through_ticks(self):
        cal = JointCalibration(name="j", ticks_per_radian=651.9, sign=-1.0, offset_ticks=2048.0)
        self.assertAlmostEqual(cal.to_radians(cal.to_ticks(0.37)), 0.37, places=9)

    def test_the_inverse_holds_for_a_positive_sign_too(self):
        cal = JointCalibration(name="j", ticks_per_radian=100.0, sign=1.0, offset_ticks=50.0)
        self.assertAlmostEqual(cal.to_ticks(2.0), 250.0)

    def test_a_grip_command_round_trips_over_the_calibrated_span(self):
        grip = GripperCalibration(name="g", open_ticks=1390.0, closed_ticks=1703.0)
        self.assertAlmostEqual(grip.to_command(grip.to_ticks(0.4)), 0.4, places=9)

    def test_a_command_past_an_endpoint_saturates_rather_than_overshooting(self):
        grip = GripperCalibration(name="g", open_ticks=0.0, closed_ticks=100.0)
        self.assertEqual(grip.to_ticks(1.5), 100.0)
        self.assertEqual(grip.to_ticks(-0.5), 0.0)


class TestTheEncoderSeam(unittest.TestCase):
    """A count is a position on a circle; 4095 and 0 are neighbours."""

    def test_the_shortest_way_round_is_the_displacement(self):
        self.assertEqual(raw_delta(10.0, 4090.0), 16.0)
        self.assertEqual(raw_delta(4090.0, 10.0), -16.0)
        self.assertEqual(raw_delta(2100.0, 2048.0), 52.0, "away from the seam it is the plain subtraction")

    def test_a_joint_whose_zero_sits_by_the_seam_does_not_read_a_full_turn(self):
        cal = JointCalibration(name="j", ticks_per_radian=651.9, offset_ticks=4090.0)
        self.assertAlmostEqual(cal.to_radians(10.0), 16.0 / 651.9)
        self.assertLess(abs(cal.to_radians(10.0)), 0.05, "not the 6.26 rad a subtraction gives")

    def test_a_hand_whose_span_straddles_the_seam_still_runs_forwards(self):
        """Open at 4000, closed 196 counts later at 100: the long way round is
        the wrong sign and twenty times the size."""
        grip = GripperCalibration(name="g", open_ticks=4000.0, closed_ticks=100.0)
        self.assertAlmostEqual(grip.to_command(4098.0 % 4096), 0.5)
        self.assertEqual(grip.to_command(4000.0), 0.0)
        self.assertEqual(grip.to_command(100.0), 1.0)

    def test_the_grip_inverse_crosses_the_seam_the_way_the_reading_does(self):
        """``to_ticks`` must invert ``to_command`` there too, or a goal written
        for a half-closed hand lands half a turn away from it."""
        grip = GripperCalibration(name="g", open_ticks=4000.0, closed_ticks=100.0)
        layout = ffb.ArmLayout(ArmCalibration(joints=(), gripper=grip))
        ticks = layout.ticks_of(0, 0.5)
        self.assertEqual(ticks, 2, "4000 + 98 counts, wrapped onto the turn")
        self.assertAlmostEqual(grip.to_command(ticks), 0.5)

    def test_a_joint_goal_past_the_seam_is_wrapped_onto_the_turn(self):
        cal = JointCalibration(name="j", ticks_per_radian=100.0, offset_ticks=4090.0)
        layout = ffb.ArmLayout(ArmCalibration(joints=(cal,)))
        ticks = layout.ticks_of(0, 0.2)
        self.assertEqual(ticks, 14)
        self.assertAlmostEqual(cal.to_radians(ticks), 0.2)


class TestTheSignal(unittest.TestCase):
    """Deadband, ramp, cap -- and which way the correction points."""

    def setUp(self):
        self.config = ffb.ComplianceConfig()

    def test_free_space_renders_nothing(self):
        target, mismatch, effective = ffb.correction(1.0, 1.02, self.config)
        self.assertAlmostEqual(mismatch, 0.02)
        self.assertEqual(effective, 0.0)
        self.assertEqual(target, 1.0, "a target equal to the leader is no pull at all")

    def test_the_ramp_starts_at_the_deadband_edge_not_at_zero(self):
        """Continuity: the first correction past the deadband is infinitesimal.

        A ramp applied to the whole mismatch instead of the excess would jump
        to ``ramp_gain * deadband`` -- a 0.2 rad step -- the instant a joint
        crossed the threshold, which is a kick rather than a contact.
        """
        _, _, effective = ffb.correction(0.0, self.config.deadband + 1e-6, self.config)
        self.assertLess(abs(effective), 1e-4)

    def test_the_correction_pulls_the_hand_toward_the_simulated_arm(self):
        """The sign that makes this feel like the scene pushing back."""
        _, _, ahead = ffb.correction(0.0, 0.2, self.config)
        _, _, behind = ffb.correction(0.0, -0.2, self.config)
        self.assertGreater(ahead, 0.0, "a follower left behind pulls the leader forward")
        self.assertLess(behind, 0.0)

    def test_the_cap_bounds_how_far_a_goal_can_lead_the_hand(self):
        target, _, effective = ffb.correction(0.5, 40.0, self.config)
        self.assertEqual(effective, self.config.max_error)
        self.assertAlmostEqual(target, 0.5 + self.config.max_error)

    def test_the_gripper_target_stays_inside_its_own_stroke(self):
        target, _, _ = ffb.correction(0.95, 1.0, self.config, limits=(0.0, 1.0))
        self.assertLessEqual(target, 1.0)

    def test_a_nonfinite_position_is_refused(self):
        with self.assertRaises(ValueError):
            ffb.correction(math.nan, 0.0, self.config)

    def test_a_reference_moves_the_mismatch_and_not_the_target_s_origin(self):
        """The hand has moved 0.1 rad on since the arm was measured against the
        pose it was chasing: the arm is on that pose, so nothing is rendered --
        and when there is something to render, the goal is still built on where
        the hand is *now*."""
        target, mismatch, effective = ffb.correction(1.1, 1.0, self.config, reference=1.0)
        self.assertEqual((mismatch, effective, target), (0.0, 0.0, 1.1))
        target, mismatch, effective = ffb.correction(1.1, 0.8, self.config, reference=1.0)
        self.assertAlmostEqual(mismatch, -0.2)
        self.assertAlmostEqual(target, 1.1 + effective)
        self.assertLess(effective, 0.0)

    def test_without_a_reference_the_hand_s_own_pose_is_the_reference(self):
        self.assertEqual(ffb.correction(1.1, 1.0, self.config), ffb.correction(1.1, 1.0, self.config, reference=1.1))

    def test_the_config_refuses_hysteresis_wider_than_the_deadband(self):
        """Otherwise the release threshold goes negative and nothing lets go."""
        with self.assertRaises(ValueError):
            ffb.ComplianceConfig(deadband=0.02, release_hysteresis=0.05)

    def test_the_config_refuses_a_torque_limit_out_of_register_range(self):
        with self.assertRaises(ValueError):
            ffb.ComplianceConfig(torque_limit=2000)


class TestTheLiveDeadband(unittest.TestCase):
    """The threshold rises with the hand's own speed, because lag does."""

    def setUp(self):
        self.config = ffb.ComplianceConfig()

    def test_a_hand_at_rest_gets_the_floor(self):
        self.assertEqual(self.config.deadband_at(0.0), self.config.deadband)

    def test_the_slope_is_the_measured_delay(self):
        """1 rad/s of hand is 45 ms of lag is 0.045 rad nobody should feel."""
        self.assertAlmostEqual(self.config.deadband_at(1.0), 0.04 + 0.045)

    def test_the_ceiling_keeps_a_fast_swing_into_something_solid_felt(self):
        self.assertEqual(self.config.deadband_at(50.0), self.config.max_deadband)

    def test_the_grip_slot_has_its_own_slope_and_a_tighter_ceiling(self):
        """A squeeze is 3 fraction/s; the arm's slope would silence the grasp."""
        self.assertAlmostEqual(self.config.deadband_at(1.0, grip=True), 0.04 + 0.020)
        self.assertEqual(self.config.deadband_at(50.0, grip=True), self.config.max_deadband_grip)
        self.assertLess(self.config.max_deadband_grip, self.config.max_deadband)

    def test_a_lag_sized_mismatch_at_speed_renders_nothing(self):
        """0.07 rad behind is contact for a still hand and lag for a fast one."""
        _, _, still = ffb.correction(0.0, 0.07, self.config, speed=0.0)
        _, _, moving = ffb.correction(0.0, 0.07, self.config, speed=1.0)
        self.assertGreater(still, 0.0)
        self.assertEqual(moving, 0.0)

    def test_the_release_threshold_scales_with_the_deadband(self):
        """A fixed 0.01 band under a 0.12 threshold is a rounding error."""
        at_rest = self.config.release_at(0.0)
        self.assertAlmostEqual(at_rest, self.config.release_threshold)
        fast = self.config.release_at(1.0)
        self.assertAlmostEqual(fast / self.config.deadband_at(1.0), at_rest / self.config.deadband)

    def test_a_zero_slope_restores_the_fixed_deadband(self):
        fixed = ffb.ComplianceConfig(velocity_deadband=0.0)
        self.assertEqual(fixed.deadband_at(3.0), fixed.deadband)


class TestTheGates(unittest.TestCase):
    """Which motors energize, and the dwell a suspended one has to serve."""

    def setUp(self):
        self.config = ffb.ComplianceConfig()
        self.layout = _layout()
        self.live = [ffb.JointGate(active=False, neutralized=True, neutral_since=None)] * self.layout.width

    def _propose(self, leader, follower, gates, now=0.0):
        return ffb.propose(
            self.config,
            self.layout,
            leader=leader,
            follower=follower,
            gates=list(gates),
            now=now,
            neutral_dwell_s=0.5,
        )

    def test_a_joint_inside_the_deadband_stays_dark(self):
        proposal = self._propose([0.0, 0.0, 0.0], [0.01, 0.0, 0.0], self.live)
        self.assertEqual(proposal.newly_active, [])
        self.assertFalse(any(gate.active for gate in proposal.gates))

    def test_only_the_joint_in_contact_energizes(self):
        proposal = self._propose([0.0, 0.0, 0.0], [0.0, 0.3, 0.0], self.live)
        self.assertEqual(proposal.newly_active, [1])
        self.assertEqual(proposal.decisions[1].motor, "j1")

    def test_hysteresis_holds_an_energized_joint_past_the_arming_threshold(self):
        """Between the release and arming thresholds, state decides -- so a
        joint resting on the edge does not chatter on and off every tick."""
        armed = [ffb.JointGate(active=True, neutralized=True) for _ in range(self.layout.width)]
        between = self.config.deadband - self.config.release_hysteresis / 2.0
        self.assertTrue(self._propose([0.0, 0.0, 0.0], [between, 0.0, 0.0], armed).gates[0].active)
        self.assertFalse(self._propose([0.0, 0.0, 0.0], [between, 0.0, 0.0], self.live).gates[0].active)

    def test_letting_go_is_reported_so_the_caller_can_disarm_it(self):
        armed = [ffb.JointGate(active=True, neutralized=True) for _ in range(self.layout.width)]
        proposal = self._propose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], armed)
        self.assertEqual(proposal.newly_inactive, [0, 1, 2])
        self.assertEqual(proposal.continuing, {})

    def test_a_joint_already_energized_is_re_targeted_not_re_armed(self):
        armed = [ffb.JointGate(active=True, neutralized=True) for _ in range(self.layout.width)]
        proposal = self._propose([0.0, 0.0, 0.0], [0.3, 0.0, 0.0], armed)
        self.assertEqual(proposal.newly_active, [])
        self.assertIn("j0", proposal.continuing)

    def test_a_suspended_joint_may_not_energize_however_large_the_mismatch(self):
        """The staleness case: the page came back and the arms disagree wildly."""
        proposal = self._propose([0.0, 0.0, 0.0], [2.0, 2.0, 1.0], self.layout.gates())
        self.assertEqual(proposal.newly_active, [])
        self.assertFalse(any(gate.neutralized for gate in proposal.gates))

    def test_alignment_must_be_continuous_to_count(self):
        """One aligned frame between two that are not restarts the dwell.

        A scene mid-settle passes through alignment on its way somewhere else,
        and a dwell that accepted that would arm off a coincidence.
        """
        gates = self.layout.gates()
        gates = self._propose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], gates, now=0.0).gates
        self.assertIsNotNone(gates[0].neutral_since)
        gates = self._propose([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], gates, now=0.2).gates
        self.assertIsNone(gates[0].neutral_since, "the broken frame restarts the clock")
        gates = self._propose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], gates, now=0.4).gates
        gates = self._propose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], gates, now=0.6).gates
        self.assertFalse(gates[0].neutralized, "0.2 s of dwell is not 0.5 s")

    def test_the_dwell_arms_a_joint_but_does_not_energize_it(self):
        """Completing the dwell and switching torque on may not be one tick."""
        gates = self.layout.gates()
        gates = self._propose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], gates, now=0.0).gates
        proposal = self._propose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], gates, now=1.0)
        self.assertTrue(all(gate.neutralized for gate in proposal.gates))
        self.assertEqual(proposal.newly_active, [])
        self.assertTrue(proposal.rearmed, "the caller says so out loud, once")
        after = self._propose([0.0, 0.0, 0.0], [0.4, 0.0, 0.0], proposal.gates, now=1.02)
        self.assertEqual(after.newly_active, [0], "the next real mismatch is what arms it")
        self.assertFalse(after.rearmed, "and it is not announced twice")

    def test_a_reference_is_taken_slot_by_slot(self):
        proposal = ffb.propose(
            self.config,
            self.layout,
            leader=[0.3, 0.3, 0.0],
            follower=[0.0, 0.0, 0.0],
            gates=list(self.live),
            now=0.0,
            neutral_dwell_s=0.5,
            reference=[0.0, 0.3, 0.0],
        )
        self.assertEqual(proposal.newly_active, [1], "slot 0 is on the pose it was chasing; slot 1 is 0.3 short of it")
        self.assertEqual(proposal.decisions[0].leader, 0.3, "the decision still records where the hand is")

    def test_the_grip_slot_is_measured_against_its_own_stroke(self):
        """The grip slot is a fraction, not radians, and converts by its own map."""
        proposal = self._propose([0.0, 0.0, 0.2], [0.0, 0.0, 0.9], self.live)
        self.assertEqual(proposal.decisions[2].motor, "grip")
        self.assertEqual(proposal.newly_active, [2])
        self.assertEqual(proposal.decisions[2].target_ticks, 40, "0.2 + 0.2 cap, over a 100-tick span")


class TestTheLayout(unittest.TestCase):
    def test_a_reading_lands_in_the_action_layout(self):
        layout = _layout()
        values = layout.read({"j0": 1002.0, "j1": 999.0, "grip": 25.0})
        self.assertAlmostEqual(values[0], 2.0)
        self.assertAlmostEqual(values[1], -1.0)
        self.assertAlmostEqual(values[2], 0.25)

    def test_an_arm_with_no_gripper_still_has_a_grip_slot(self):
        """``ActionJointPosition`` reserves one per arm whether or not it exists."""
        layout = _layout(joints=3, gripper=False)
        self.assertEqual(layout.width, 4)
        self.assertEqual(layout.read({f"j{i}": 1000.0 for i in range(3)})[-1], 0.0)
        self.assertEqual(layout.motors, ["j0", "j1", "j2"], "and no motor behind it")

    def test_a_goal_wraps_onto_the_encoder_turn(self):
        layout = _layout()
        self.assertEqual(layout.ticks_of(0, 3100.0), (1000 + 3100) % 4096)

    def test_fresh_gates_start_suspended(self):
        """Nothing may energize off the very first sample after a connect."""
        self.assertTrue(all(not gate.neutralized for gate in _layout().gates()))


if __name__ == "__main__":
    unittest.main()
