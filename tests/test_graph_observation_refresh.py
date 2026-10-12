"""Perception must see current pixels when continuous rendering is disabled."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from robosimstudio_policy_agent.connector import _ProprioceptiveFeedback
from robosimstudio_policy_agent.facade import GapEnvFacade
from robosimstudio_policy_agent.graph_solution import _OnDemandConnector

TIER = "fast"
"""Perception sees current pixels when per-step rendering is off; fakes, no renderer. Gates every commit."""


class TestObservationRefresh(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from gap import execute
            from gap_core.tools import ToolRegistry
        except ImportError:
            self.skipTest("gap is behind the [gap] extra")
        self.execute, self.ToolRegistry = execute, ToolRegistry

    def test_on_demand_runtime_still_executes_explicit_camera_reads(self):
        reads = []

        def camera():
            reads.append("fresh RGB-D")
            return {"frame": len(reads)}

        registry = self.ToolRegistry()
        registry.register_callable("robot.get_observation", camera, summary="Read a fresh camera frame")
        connector = SimpleNamespace(tool_registry=registry, get_observation=camera, world_snapshot=None)
        workflow = {
            "version": 3,
            "meta": {"name": "on-demand observation test"},
            "nodes": {
                "read": {"type": "tool", "tool": "robot.get_observation"},
                "done": {"type": "end", "status": "success"},
            },
            "edges": [["START", "read"], ["read", "done"]],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "workflow.json").write_text(json.dumps(workflow))
            result = self.execute(path, connector=_OnDemandConnector(connector), skills=[], checkpoints="off")
        self.assertTrue(result.success, result.error)
        self.assertEqual(reads, ["fresh RGB-D"])

    def test_motion_feedback_does_not_render_cameras(self):
        reads = []

        def sensors():
            reads.append("proprioception")
            return {
                "robot_joint_pos_1": [0.1] * 6 + [0.35],
                "robot_cartesian_pos_1": [0.4, 0.2, 0.8, 1.0, 0.0, 0.0, 0.0],
            }

        def cameras():
            raise AssertionError("motion feedback rendered RGB-D")

        feedback = _ProprioceptiveFeedback()
        feedback.env = SimpleNamespace(get_proprioception=sensors, get_observation=cameras)
        feedback._arm_dof = 6
        feedback._gripper_fraction = 0.0
        feedback._base_frame_to_world = lambda point, arm: point
        self.assertEqual(feedback._current_joints(1), [0.1] * 6)
        self.assertEqual(feedback.get_gripper_fraction(1), 0.35)
        self.assertEqual(feedback.get_ee_pose(1)["position"], {"x": 0.4, "y": 0.2, "z": 0.8})
        self.assertEqual(reads, ["proprioception"] * 3)

    def test_on_request_render_uses_current_state_before_observing(self):
        calls = []
        state = object()
        env = SimpleNamespace(obs_render_enabled=False, state=state, frames={"old": {}}, cameras=[])

        def render(observed_state):
            self.assertIs(observed_state, state)
            calls.append("render")
            return {"fresh": {"rgb": "current pixels"}}

        def observe():
            calls.append("observe")
            return {"cameras": env.frames}

        env.renderer = SimpleNamespace(render=render)
        env.observe = observe
        facade = GapEnvFacade.__new__(GapEnvFacade)
        facade.env = env

        # No robot is needed to exercise the camera ownership boundary.
        class CameraFacade(GapEnvFacade):
            @property
            def arm_names(self):
                return []

        facade.__class__ = CameraFacade
        result = facade._observation(refresh_cameras=True)
        self.assertEqual(calls, ["render", "observe"])
        self.assertEqual(result["fresh"]["images"]["rgb"], "current pixels")
        self.assertNotIn("old", result)

        # Publishing a motion step must not turn continuous rendering back on.
        calls.clear()
        facade._observation()
        self.assertEqual(calls, ["observe"])

        env.obs_render_enabled = True
        calls.clear()
        facade._observation(refresh_cameras=True)
        self.assertEqual(calls, ["observe"])


if __name__ == "__main__":
    unittest.main()
