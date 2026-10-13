# SPDX-License-Identifier: Apache-2.0
"""The robot library, where it can be tested without a scene.

Operational-space control, the waypoint runner, and the action spaces a policy
writes into -- all of them over stub arms, because what is being tested is the
arithmetic and the contract rather than any particular robot. Anything needing a
built model belongs to the medium tier.

Split out of ``test_core.py`` when the control stack moved to
``robosimstudio.solvers``: a test module follows the code it tests.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.env import EnvBase
from robosimstudio.core.math import quat_from_axis_angle, quat_multiply
from robosimstudio.solvers import OSC_OUTPUT_MAX, OscController, OscGains, OscWaypoints, Waypoint, osc_action

TIER = "fast"
"""No GPU. Gates every commit."""


class _StubArm:
    """The slice of :class:`ArmIK` an :class:`OscController` reads.

    The controller's law is host-side numpy over a measured pose, so it can be
    exercised without a finalized model -- which is the point of testing it here
    rather than only through a scene: the gains and the clipping are the part
    that has to agree with the reference, and agreement is checkable in
    arithmetic.
    """

    class _Model:
        def __init__(self):
            self.body_com = SimpleNamespace(numpy=lambda: np.zeros((1, 3)))

    def __init__(self):
        self.model = self._Model()
        self.ee_body = 0
        self.ee_offset = np.zeros(3)
        self.ee_offset_rot = np.array([0.0, 0.0, 0.0, 1.0])
        self.home_pose = (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))


def _stub_state(position, quat=(0.0, 0.0, 0.0, 1.0), twist=(0.0,) * 6):
    """A ``State``-shaped stand-in holding one body at a pose."""
    body_q = np.array([[*position, *quat]], dtype=np.float64)
    body_qd = np.array([list(twist)], dtype=np.float64)
    return SimpleNamespace(
        body_q=SimpleNamespace(numpy=lambda: body_q),
        body_qd=SimpleNamespace(numpy=lambda: body_qd),
    )


class TestOscGains(unittest.TestCase):
    """The reference's parameterization, and what it implies about speed."""

    def test_damping_is_derived_from_the_ratio(self):
        gains = OscGains(kp=150.0, damping_ratio=1.0)
        self.assertAlmostEqual(gains.kd, 2.0 * np.sqrt(150.0))

    def test_rotation_stiffness_falls_back_to_kp(self):
        """``OSC_POSE``'s kp is one scalar over all six axes."""
        self.assertEqual(OscGains(kp=150.0).stiffness_rot, 150.0)
        self.assertEqual(OscGains(kp=150.0, kp_rot=40.0).stiffness_rot, 40.0)

    def test_cruise_speed_is_not_the_per_step_clip_over_dt(self):
        """The clip bounds the error; the gains turn that into a velocity.

        0.05 m per 20 Hz step reads as 1.0 m/s and is not: a saturated error
        settles where ``kp*e`` meets ``kd*v``, which for the reference's gains is
        about 0.31 m/s. A step budget sized on the wrong one of those is wrong
        by a factor of three.
        """
        self.assertAlmostEqual(OscGains().cruise_speed(0.05), 0.05 * np.sqrt(150.0) / 2.0, places=6)
        self.assertLess(OscGains().cruise_speed(0.05), 0.4)

    def test_degenerate_gains_raise(self):
        for kwargs in ({"kp": 0.0}, {"damping_ratio": 0.0}, {"kp_rot": -1.0}):
            with self.assertRaises(ValueError):
                OscGains(**kwargs)


class TestOscAction(unittest.TestCase):
    """:func:`osc_action`, against the reference's own formula."""

    IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])

    def test_a_far_waypoint_saturates(self):
        action = osc_action(np.zeros(3), self.IDENTITY, [1.0, 0.0, 0.0], self.IDENTITY)
        np.testing.assert_allclose(action[:3], [1.0, 0.0, 0.0])

    def test_a_near_waypoint_is_proportional(self):
        """Inside one step's reach the command tapers -- where the precision is."""
        action = osc_action(np.zeros(3), self.IDENTITY, [0.02, 0.0, 0.0], self.IDENTITY)
        self.assertAlmostEqual(action[0], 0.02 / OSC_OUTPUT_MAX[0])

    def test_arriving_commands_nothing(self):
        action = osc_action([0.3, 0.2, 0.1], self.IDENTITY, [0.3, 0.2, 0.1], self.IDENTITY)
        np.testing.assert_allclose(action, np.zeros(6), atol=1e-12)

    def test_rotation_normalizes_against_its_own_ceiling(self):
        target = quat_multiply(quat_from_axis_angle([0.0, 0.0, 1.0], 0.25), self.IDENTITY)
        action = osc_action(np.zeros(3), self.IDENTITY, np.zeros(3), target)
        self.assertAlmostEqual(action[5], 0.25 / OSC_OUTPUT_MAX[3])

    def test_the_error_is_taken_the_short_way_round(self):
        """A 190-degree error turns the near way, not through its own limits."""
        target = quat_multiply(quat_from_axis_angle([0.0, 0.0, 1.0], np.radians(190.0)), self.IDENTITY)
        action = osc_action(np.zeros(3), self.IDENTITY, np.zeros(3), target)
        self.assertLess(action[5], 0.0)


class TestOscController(unittest.TestCase):
    """The scaling contract, and the closed loop the scaling feeds."""

    IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])

    def _controller(self, **kwargs):
        return OscController(_StubArm(), control_dt=1.0 / 20.0, **kwargs)

    def test_a_unit_action_is_exactly_output_max(self):
        """What a checkpoint trained against the reference depends on."""
        scaled = self._controller().scale_action([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(scaled, OSC_OUTPUT_MAX)

    def test_an_out_of_range_action_is_clipped_before_scaling(self):
        scaled = self._controller().scale_action([5.0, -5.0, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(scaled[0], OSC_OUTPUT_MAX[0])
        self.assertAlmostEqual(scaled[1], -OSC_OUTPUT_MAX[0])

    def test_a_zero_action_asks_for_no_displacement(self):
        np.testing.assert_allclose(self._controller().scale_action(np.zeros(6)), np.zeros(6), atol=1e-15)

    def test_a_wrong_width_action_raises(self):
        with self.assertRaises(ValueError):
            self._controller().scale_action(np.zeros(7))

    def test_the_goal_is_the_measurement_plus_the_delta(self):
        """Not an accumulated setpoint -- the whole difference from ee_delta."""
        controller = self._controller()
        state = _stub_state([0.4, 0.1, 0.3])
        goal, _ = controller.goal([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], state)
        np.testing.assert_allclose(goal, [0.4 + OSC_OUTPUT_MAX[0], 0.1, 0.3])
        # Re-asking with the arm elsewhere re-anchors, rather than accumulating.
        goal, _ = controller.goal([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], _stub_state([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(goal, [OSC_OUTPUT_MAX[0], 0.0, 0.0])

    def test_the_tool_centre_twist_includes_the_offset_term(self):
        """A purely rotating wrist moves a tool centre that is offset from it."""
        arm = _StubArm()
        arm.ee_offset = np.array([0.0, 0.0, 0.1])
        controller = OscController(arm, control_dt=1.0 / 20.0)
        _, _, linear, _ = controller.measure(_stub_state(np.zeros(3), twist=(0.0, 0.0, 0.0, 0.0, 2.0, 0.0)))
        # omega x r = (0, 2, 0) x (0, 0, 0.1) = (0.2, 0, 0)
        np.testing.assert_allclose(linear, [0.2, 0.0, 0.0], atol=1e-12)

    def _run_perfect_servo(self, goal, steps, control_hz=20.0):
        """Close the loop against an ideal servo: the arm reaches the setpoint.

        The stiff-servo limit, which is the regime the position realization is
        chosen for. Reports the trajectory of the measured position so the shape
        of the response can be asserted rather than eyeballed.
        """
        controller = OscController(_StubArm(), control_dt=1.0 / control_hz)
        position = np.zeros(3)
        controller.reset(_stub_state(position))
        track = [position.copy()]
        for _ in range(steps):
            state = _stub_state(position)
            action = osc_action(position, self.IDENTITY, goal, self.IDENTITY)
            position, _ = controller.advance(action, state)
            track.append(position.copy())
        return np.array(track)

    def test_a_far_goal_is_crossed_at_the_gains_cruise_speed(self):
        """The saturated regime settles at ``(kp/kd) * output_max``, not at the clip."""
        track = self._run_perfect_servo(np.array([2.0, 0.0, 0.0]), steps=40)
        speed = float(np.diff(track[-10:, 0]).mean() * 20.0)
        self.assertAlmostEqual(speed, OscGains().cruise_speed(OSC_OUTPUT_MAX[0]), places=2)

    def test_a_near_goal_is_converged_on_without_overshoot(self):
        """Critically damped means it arrives and stays -- no hunting on a waypoint."""
        goal = np.array([0.03, 0.0, 0.0])
        track = self._run_perfect_servo(goal, steps=120)
        self.assertLess(abs(track[-1, 0] - goal[0]), 1.0e-3)
        self.assertLessEqual(track[:, 0].max(), goal[0] + 1.0e-3)

    def test_a_blocked_arm_does_not_wind_up(self):
        """Held in place, the setpoint stops one lead clamp away rather than running off.

        This is the whole of the measured feedback in the realized command, and
        it is what bounds the force a blocked arm pushes with: the reference's
        law tolerates a standing error of ``output_max`` at cruise, so that is
        how far ahead of a stuck arm this one is allowed to command.
        """
        controller = OscController(_StubArm(), control_dt=1.0 / 20.0)
        blocked = _stub_state(np.zeros(3))
        controller.reset(blocked)
        for _ in range(200):
            controller.advance([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], blocked)
        setpoint, _ = controller.setpoint
        self.assertAlmostEqual(float(np.linalg.norm(setpoint)), controller.pos_lead, places=9)

    def test_the_setpoint_leads_the_arm_rather_than_tracking_it(self):
        """Anchoring the command to the measurement is what made the arm crawl.

        A setpoint recomputed as "measurement plus one step" can never be more
        than one step of authority ahead, which caps the arm at whatever the
        servo does with a 6 mm error -- measured at an eighth of the reference's
        speed. So a stationary arm must see its command run away from it.
        """
        controller = OscController(_StubArm(), control_dt=1.0 / 20.0)
        stuck = _stub_state(np.zeros(3))
        controller.reset(stuck)
        first, _ = controller.advance([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], stuck)
        second, _ = controller.advance([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], stuck)
        self.assertGreater(float(np.linalg.norm(second)), float(np.linalg.norm(first)))

    def test_reset_reseeds_the_setpoint_from_the_arm(self):
        controller = self._controller()
        controller.reset(_stub_state(np.zeros(3)))
        for _ in range(20):
            controller.advance([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], _stub_state(np.zeros(3)))
        self.assertGreater(float(np.linalg.norm(controller.setpoint[0])), 1.0e-3)
        controller.reset(_stub_state([0.3, 0.2, 0.1]))
        np.testing.assert_allclose(controller.setpoint[0], [0.3, 0.2, 0.1])

    def test_the_speed_and_lead_limits_are_derived_from_the_gains(self):
        """Neither is a tuned constant: both fall out of kp, kd and output_max."""
        controller = self._controller()
        self.assertAlmostEqual(controller.pos_speed, OscGains().cruise_speed(OSC_OUTPUT_MAX[0]))
        self.assertAlmostEqual(controller.pos_lead, OSC_OUTPUT_MAX[0])
        self.assertAlmostEqual(controller.rot_lead, OSC_OUTPUT_MAX[3])

    def test_a_non_positive_control_period_raises(self):
        with self.assertRaises(ValueError):
            OscController(_StubArm(), control_dt=0.0)


class _RecordingSpace:
    """An action space that records what was asked of it, and nothing else."""

    per_arm = 3

    def __init__(self, dim: int = 3) -> None:
        self.action_dim = dim
        self.resets = 0
        self.applied: list[list[float]] = []

    def reset(self, state) -> None:
        del state
        self.resets += 1

    def apply(self, action, control, state) -> None:
        del control, state
        self.applied.append(list(action))

    def hold_action(self, state) -> np.ndarray:
        del state
        return np.zeros(self.action_dim, dtype=np.float32)


class TestControllerSet(unittest.TestCase):
    """An arm has several ways to be commanded, and a command names one.

    The behaviour under test is the one the reference runtime's own controller
    swap documents: a controller carries a *setpoint*, so handing the arm to one
    that still holds a stale setpoint drags the arm back there on the first step.
    Selecting a mode therefore re-seeds it.
    """

    def _env(self, **kwargs):
        pipeline = SimpleNamespace(
            model=SimpleNamespace(particle_count=0, body_count=1),
            state=_stub_state(np.zeros(3)),
            control=None,
            sim_time=0.0,
            step_count=0,
            step=lambda: None,
        )
        return EnvBase(pipeline, **kwargs)

    def test_a_single_space_still_works(self):
        """The old one-space construction is the one-entry case of the new one."""
        space = _RecordingSpace()
        env = self._env(action_space=space)
        self.assertIs(env.action_space, space)
        self.assertEqual(env.action_dim, 3)

    def test_the_declared_mode_starts_active(self):
        spaces = {"joint_position": _RecordingSpace(), "osc_pose": _RecordingSpace(7)}
        env = self._env(action_space=spaces, control_mode="osc_pose")
        self.assertEqual(env.control_mode, "osc_pose")
        self.assertEqual(env.action_dim, 7)

    def test_an_unsupported_declared_mode_raises_at_construction(self):
        with self.assertRaises(ValueError):
            self._env(action_space={"joint_position": _RecordingSpace()}, control_mode="osc_pose")

    def test_naming_a_mode_switches_and_reseeds_it(self):
        """The load-bearing behaviour: a handover re-seeds the incoming setpoint."""
        joint, osc = _RecordingSpace(), _RecordingSpace(7)
        env = self._env(action_space={"joint_position": joint, "osc_pose": osc}, control_mode="joint_position")
        before = osc.resets
        env.apply_action([0.0] * 7, mode="osc_pose")
        self.assertEqual(env.control_mode, "osc_pose")
        self.assertEqual(osc.resets, before + 1)
        self.assertEqual(len(osc.applied), 1)
        self.assertEqual(joint.applied, [])

    def test_renaming_the_active_mode_does_not_reseed(self):
        """So a caller may tag every command without paying for it."""
        osc = _RecordingSpace(7)
        env = self._env(action_space={"osc_pose": osc}, control_mode="osc_pose")
        seeded = osc.resets
        for _ in range(3):
            env.apply_action([0.0] * 7, mode="osc_pose")
        self.assertEqual(osc.resets, seeded)
        self.assertEqual(len(osc.applied), 3)

    def test_an_unknown_mode_names_what_there_is(self):
        env = self._env(action_space={"joint_position": _RecordingSpace()})
        with self.assertRaises(KeyError) as caught:
            env.use_control("osc_pose")
        self.assertIn("joint_position", str(caught.exception))

    def test_an_omitted_mode_keeps_the_active_one(self):
        joint, osc = _RecordingSpace(), _RecordingSpace(7)
        env = self._env(action_space={"joint_position": joint, "osc_pose": osc}, control_mode="joint_position")
        env.apply_action([0.0, 0.0, 0.0])
        self.assertEqual(env.control_mode, "joint_position")
        self.assertEqual(len(joint.applied), 1)

    def test_idle_action_may_be_asked_of_a_mode_without_switching(self):
        """Asking what a hold looks like elsewhere is not asking to hand over."""
        joint, osc = _RecordingSpace(), _RecordingSpace(7)
        env = self._env(action_space={"joint_position": joint, "osc_pose": osc}, control_mode="joint_position")
        env.idle_action("osc_pose")
        self.assertEqual(env.control_mode, "joint_position")


class TestOscWaypoints(unittest.TestCase):
    """The reference's ``go_to`` loop, as a sequencer."""

    IDENTITY = (0.0, 0.0, 0.0, 1.0)

    def _legs(self):
        return [
            Waypoint("approach", (0.5, 0.0, 0.3), self.IDENTITY, 0.0, 2),
            Waypoint("close", (0.5, 0.0, 0.2), self.IDENTITY, 1.0, 3),
        ]

    def test_each_leg_is_held_for_its_own_step_count(self):
        run = OscWaypoints(self._legs())
        labels = []
        for _ in range(5):
            labels.append(run.current.label)
            run.act(np.zeros(3), self.IDENTITY)
        self.assertEqual(labels, ["approach", "approach", "close", "close", "close"])
        self.assertTrue(run.done)

    def test_the_gripper_command_comes_from_the_leg(self):
        run = OscWaypoints(self._legs())
        self.assertEqual(run.act(np.zeros(3), self.IDENTITY)[6], 0.0)
        run.act(np.zeros(3), self.IDENTITY)
        self.assertEqual(run.act(np.zeros(3), self.IDENTITY)[6], 1.0)

    def test_running_out_of_waypoints_holds_the_last_grip(self):
        """An episode outlives its script, and a released object needs to settle."""
        run = OscWaypoints(self._legs())
        for _ in range(5):
            run.act(np.zeros(3), self.IDENTITY)
        action = run.act(np.zeros(3), self.IDENTITY)
        np.testing.assert_allclose(action[:6], np.zeros(6))
        self.assertEqual(action[6], 1.0)

    def test_finishing_reports_the_last_step_of_a_leg(self):
        run = OscWaypoints(self._legs())
        self.assertFalse(run.finishing)
        run.act(np.zeros(3), self.IDENTITY)
        self.assertTrue(run.finishing)

    def test_load_restarts_the_sequence(self):
        run = OscWaypoints(self._legs())
        run.act(np.zeros(3), self.IDENTITY)
        run.load(self._legs())
        self.assertEqual(run.index, 0)
        self.assertEqual(run.frame, 0)
        self.assertEqual(run.current.label, "approach")


if __name__ == "__main__":
    unittest.main()


class _DeltaArm:
    """The three things ``ActionEEDelta`` asks of an ``ArmIK``, recorded."""

    def __init__(self, pos, quat) -> None:
        self.home_pose = (np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64))
        self.target = None

    def ee_pose(self, state):
        return self.home_pose

    def set_target(self, pos, quat) -> None:
        self.target = (np.asarray(pos), np.asarray(quat))

    def solve_into(self, joint_q, joint_target_q) -> None:
        pass


class TestEEDeltaToolConvention(unittest.TestCase):
    """``"tool"`` is ManiSkill's ``ee`` frame: ``prev_pose * delta``, translating along the pre-step axes."""

    def test_a_combined_move_translates_along_the_pre_step_orientation(self):
        from robosimstudio.solvers import ActionEEDelta

        arm = _DeltaArm((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        space = ActionEEDelta([(arm, None)], convention="tool")
        # 10 cm along the tool's x while turning a quarter about its z.
        space.apply(
            [0.1, 0.0, 0.0, 0.0, 0.0, np.pi / 2, 0.0],
            SimpleNamespace(joint_target_q=None),
            SimpleNamespace(joint_q=None),
        )
        pos, quat = arm.target
        np.testing.assert_allclose(pos, [0.1, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(np.abs(quat), np.abs(quat_from_axis_angle([0, 0, 1], np.pi / 2)), atol=1e-9)
