# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the policy package's privilege boundary and facade stepping.

* ``NonPrivilegedRegistry`` filters ``register_rpc`` too, refuses the other
  registration doors, and answers ``in``.
* ``go_to_pose`` with ``z_approach`` stops on a failed pre-approach.
* ``assert_non_privileged`` accepts the frame the facade actually builds.
* A facade step observes once and reads ``body_q`` back once; a blocking joint
  move is one bus round trip per step.
* ``GapEnvFacade.reset`` forgets the last episode's held targets.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

TIER = "fast"


class _FakeRegistry:
    def __init__(self) -> None:
        self.names: list[str] = []

    def register_callable(self, name, fn, **kwargs):
        self.names.append(name)

    def register_rpc(self, name, client, **kwargs):
        self.names.append(name)

    def discover_pending(self, *a, **k):
        raise AssertionError("reached the real registry")

    def __contains__(self, name):
        return name in self.names

    def __len__(self):
        return len(self.names)


class TestNonPrivilegedRegistry(unittest.TestCase):
    def setUp(self) -> None:
        from robosimstudio_policy_agent.connector import NonPrivilegedRegistry

        self.inner = _FakeRegistry()
        self.reg = NonPrivilegedRegistry(self.inner)

    def test_register_rpc_drops_sim_and_privileged(self) -> None:
        self.reg.register_rpc("sim.get_object_pose", object(), summary="")
        self.reg.register_rpc("look", object(), tags=("privileged",))
        self.reg.register_rpc("robot.home", object(), tags=("robot",))
        self.assertEqual(self.inner.names, ["robot.home"])
        self.assertEqual(self.reg.dropped, ["sim.get_object_pose", "look"])

    def test_other_registration_doors_are_refused(self) -> None:
        for attr in (
            "discover_pending",
            "discover",
            "_discover_python",
            "_register_python",
            "register",
            "_tools",
            "python_adapter",
            "rpc_adapter",
        ):
            with self.assertRaises(AttributeError, msg=attr):
                getattr(self.reg, attr)

    def test_readers_pass_through(self) -> None:
        self.inner.get = lambda name: name
        self.assertEqual(self.reg.get("robot.home"), "robot.home")
        self.reg.register_callable("robot.home", lambda: None)
        self.assertEqual(len(self.reg), 1)

    def test_contains_delegates(self) -> None:
        self.reg.register_callable("robot.home", lambda: None)
        self.assertIn("robot.home", self.reg)
        self.assertNotIn("sim.x", self.reg)


class TestApproachFailureStops(unittest.TestCase):
    def test_contact_on_the_approach_is_reported(self) -> None:
        try:
            import robosimstudio_policy_agent.connector as C
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))

        class Servo:
            def __init__(self) -> None:
                self.calls: list[float] = []

            def move(self, pose, arm_id=0, **kw):
                self.calls.append(pose["position"]["z"])
                if len(self.calls) == 1:
                    return {"outcome": "contact", "remaining_distance": 0.08, "steps": 40}
                return {"outcome": "reached", "remaining_distance": 0.001, "steps": 90}

        class Base:
            def __init__(self, *a, **k) -> None:
                pass

            def go_to_pose(self, *a, **k):
                return "base"

        servo = Servo()
        cfg = mock.MagicMock()
        cfg.default_cameras = []
        with mock.patch.object(C, "_connector_base", lambda: Base):
            try:
                conn = C.make_connector(
                    mock.MagicMock(),
                    config=cfg,
                    facade=mock.MagicMock(),
                    ik=mock.MagicMock(),
                    servo=servo,
                    bus=mock.MagicMock(),
                )
            except ImportError as exc:  # pragma: no cover
                self.skipTest(str(exc))
        pose = {"position": {"x": 0.4, "y": 0.0, "z": 0.1}, "rotation": {"x": 1, "y": 0, "z": 0, "w": 0}}
        out = conn.go_to_pose(pose, z_approach=0.1)
        self.assertEqual(len(servo.calls), 1)
        self.assertEqual(out["outcome"], "contact")
        self.assertEqual(out["phase"], "approach")
        self.assertFalse(out["reached"])


class _Bus:
    def __init__(self) -> None:
        self.calls = 0
        self.snapshot = None

    def call(self, fn, *a, **kw):
        self.calls += 1
        return fn(*a, **kw)

    def publish(self, snapshot) -> None:
        self.snapshot = snapshot


def _facade():
    """A real GapEnvFacade over a stub env with one arm and one camera, counting readbacks."""
    from robosimstudio_policy_agent.facade import GapEnvFacade

    counts = {"observe": 0, "body_q": 0, "embodiment": 0, "step": 0}
    joints = np.zeros(7)

    def observe():
        counts["observe"] += 1
        return {
            "proprio": {"arm/joint_q": joints.copy(), "arm/gripper": 0.0, "arm/ee_pose": np.r_[0, 0, 0, 0, 0, 0, 1.0]},
            "cameras": {"overhead": {"rgb": np.zeros((2, 2, 3))}},
            "instruction": "go",
        }

    def body_q_numpy():
        counts["body_q"] += 1
        return np.zeros((1, 7), dtype=np.float32)

    def emb_observe(state):
        counts["embodiment"] += 1
        return {"arm/joint_q": joints.copy()}

    def step(action, mode=""):
        counts["step"] += 1
        joints[:] = action[:7]
        return observe(), 0.0, False, False, {}

    space = SimpleNamespace(per_arm=8)
    spec = SimpleNamespace(name="overhead", intrinsics=np.eye(3), pose_on_body=lambda body_q: np.eye(4))
    env = SimpleNamespace(
        max_steps=10,
        obs_render_enabled=True,
        renderer=None,
        cameras=[spec],
        frames={},
        step_count=0,
        sim_time=0.0,
        state=SimpleNamespace(body_q=SimpleNamespace(numpy=body_q_numpy)),
        observe=observe,
        step=step,
        use_control=lambda mode: space,
        idle_action=lambda mode: np.zeros(8, dtype=np.float32),
        controllers={"joint_position": space},
        scene=SimpleNamespace(embodiment=SimpleNamespace(grippers={}, observe=emb_observe)),
        reset=lambda episode_id: ({}, {}),
    )

    class OneArm(GapEnvFacade):
        @property
        def arm_names(self):
            return ["arm"]

        def _slot(self, name):
            return 0

        def _arm(self, arm_id):
            return "arm"

    facade = OneArm(env, _Bus())
    return facade, counts


class TestBridgeContractAcceptsTheFacadeFrame(unittest.TestCase):
    def test_full_observation_passes(self) -> None:
        from robosimstudio_policy_agent.contract import assert_non_privileged

        facade, _ = _facade()
        frame = facade._observation()
        self.assertIn("robot_joint_pos_0", frame)
        assert_non_privileged(frame, cameras=["overhead"])

    def test_format_runtime_obs_passes(self) -> None:
        from robosimstudio_policy_agent.contract import assert_non_privileged
        from robosimstudio_policy_agent.fleet.obs_format import format_runtime_obs

        snapshot = SimpleNamespace(
            obs={"proprio": {"a/joint_q": np.zeros(7), "a/ee_pose": np.r_[0, 0, 0, 0, 0, 0, 1.0]}},
            frames={"wrist": {"rgb": np.zeros((2, 2, 3)), "depth": np.zeros((2, 2))}},
            body_q=None,
        )
        frame = format_runtime_obs(snapshot, arm_count=1)
        assert_non_privileged(frame, cameras=["wrist"])

    def test_still_refuses_ground_truth(self) -> None:
        from robosimstudio_policy_agent.contract import assert_non_privileged

        with self.assertRaises(ValueError):
            assert_non_privileged({"robot_joint_pos_0": [0.0], "object_poses": {}})
        # An undeclared camera name is not a pass, and a camera may not be named for ground truth.
        with self.assertRaises(ValueError):
            assert_non_privileged({"overhead": {"images": {}}})
        with self.assertRaises(ValueError):
            assert_non_privileged({"objects": {"images": {}}}, cameras=["objects"])
        with self.assertRaises(ValueError):
            assert_non_privileged({"overhead": {"body_q": []}}, cameras=["overhead"])

    def test_refusal_names_what_is_allowed(self) -> None:
        from robosimstudio_policy_agent.contract import ALLOWED_PATTERNS, __all__, assert_non_privileged

        self.assertIn("ALLOWED_PATTERNS", __all__)
        with self.assertRaises(ValueError) as caught:
            assert_non_privileged({"overhead": {"images": {}}})
        message = str(caught.exception)
        self.assertIn("cameras=", message)
        for pattern in ALLOWED_PATTERNS:
            self.assertIn(pattern.pattern, message)
        # Ground truth gets no camera hint: it is refused for what it is.
        with self.assertRaises(ValueError) as caught:
            assert_non_privileged({"object_poses": {}})
        self.assertNotIn("cameras=", str(caught.exception))


class TestFacadeStepReadsOnce(unittest.TestCase):
    def test_blocking_move_counts(self) -> None:
        facade, counts = _facade()
        facade.move_to_joints_blocking(np.full(7, 0.5), tolerance=-1.0, max_steps=10)
        self.assertEqual(counts["step"], 10)
        self.assertEqual(counts["observe"], 10)
        self.assertEqual(counts["body_q"], 10)
        self.assertEqual(counts["embodiment"], 0)
        self.assertEqual(facade.bus.calls, 10)

    def test_blocking_move_still_converges(self) -> None:
        facade, _ = _facade()
        self.assertTrue(facade.move_to_joints_blocking(np.full(7, 0.5), tolerance=1e-3, max_steps=5))
        snap = facade.bus.snapshot
        np.testing.assert_allclose(snap.obs["robot_joint_pos_0"][:7], 0.5)


class TestStepPublishesTheFreshFrame(unittest.TestCase):
    def test_camera_block_is_this_steps_render(self) -> None:
        """``env.step`` observes, then renders in ``evaluate`` by rebinding ``env.frames``."""
        facade, _ = _facade()
        env = facade.env
        stale = {"overhead": {"rgb": np.zeros((2, 2, 3))}}
        fresh = {"overhead": {"rgb": np.ones((2, 2, 3))}}
        env.frames = stale
        inner_step = env.step

        def step(action, mode=""):
            obs, *rest = inner_step(action, mode)
            obs["cameras"] = env.frames  # the dict observe handed back
            env.frames = fresh  # evaluate's render
            return (obs, *rest)

        env.step = step
        facade.move_to_joints_blocking(np.full(7, 0.5), tolerance=-1.0, max_steps=1)
        snap = facade.bus.snapshot
        np.testing.assert_array_equal(snap.obs["overhead"]["images"]["rgb"], fresh["overhead"]["rgb"])
        self.assertIs(snap.frames["overhead"], fresh["overhead"])


class TestResetForgetsTheHold(unittest.TestCase):
    def test_hold_after_reset_is_idle(self) -> None:
        facade, _ = _facade()
        facade.apply_joint_action(np.full(7, 1.5), arm_id=0)
        self.assertEqual(float(facade.hold_action()[0]), 1.5)
        facade._gripper_fraction[0] = 0.0
        facade.reset(1)
        self.assertEqual(facade._held, {})
        self.assertEqual(facade._gripper_fraction, {})
        np.testing.assert_array_equal(facade.hold_action()[:7], np.zeros(7))


if __name__ == "__main__":
    unittest.main()
