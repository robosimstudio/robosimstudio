"""Caches that outlived what they described, and a bodiless exclusion.

``fingertip_reach`` kept finger corners in a module dict keyed by ``id(env)``,
so a later env reusing a freed one's id read the old hand's corners; the servo
cached the backend's ``honours_roll`` once, so a solver fallback that dropped
the roll declaration left the servo testing arrival on a roll nothing closes;
and ``get_collision_world`` excluded body -1 for a bodiless handle, which drops
every world-static shape.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from robosimstudio_policy_agent import robot_tools, sim_tools
from robosimstudio_policy_agent.servo import CartesianServo

TIER = "fast"


def _env(size: float):
    """A stub env whose one finger collider has a corner *size* past the TCP along +z."""
    arm = types.SimpleNamespace(
        finger_groups=(((1, np.array([0])),),),
        ee_body=0,
        ee_offset=np.zeros(3),
        ee_offset_rot=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    builder = types.SimpleNamespace(shape_flags=[0xFFFF], size=size)
    return types.SimpleNamespace(
        handles=types.SimpleNamespace(arms={"arm0": arm}),
        scene=types.SimpleNamespace(builder=builder),
    )


class FingertipReachCache(unittest.TestCase):
    def test_each_env_reads_its_own_corners(self):
        body_q = np.tile(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64), (2, 1))
        try:
            import robosimstudio.solvers.grasp.obb as obb
        except Exception as error:  # pragma: no cover - environment without the solver stack
            self.skipTest(f"grasp solvers unavailable: {error}")
        with mock.patch.object(obb, "_shape_corners_local", lambda builder, s: [(0.0, 0.0, builder.size)]):
            a, b = _env(0.01), _env(0.05)
            # Force the collision the id-keyed cache could not survive.
            with mock.patch.object(robot_tools, "id", create=True, side_effect=lambda _obj: 1):
                reach_a = robot_tools.fingertip_reach(a, "arm0", body_q=body_q)
                reach_b = robot_tools.fingertip_reach(b, "arm0", body_q=body_q)
        self.assertAlmostEqual(abs(reach_a), 0.01)
        self.assertAlmostEqual(abs(reach_b), 0.05)


class ServoRollContract(unittest.TestCase):
    def test_follows_the_backend_after_a_fallback(self):
        ik = types.SimpleNamespace(honours_roll=True)
        servo = CartesianServo.__new__(CartesianServo)
        servo.ik = ik
        self.assertTrue(servo.honours_roll)
        ik.honours_roll = False  # what IKBackend._adopt does on a roll-dropping fallback
        self.assertFalse(servo.honours_roll)

    def test_a_direct_assignment_still_overrides(self):
        servo = CartesianServo.__new__(CartesianServo)
        servo.ik = types.SimpleNamespace(honours_roll=True)
        servo.honours_roll = False
        self.assertFalse(servo.honours_roll)


class CollisionWorldExclusion(unittest.TestCase):
    def test_bodiless_handle_does_not_drop_the_statics(self):
        def shape(index, body):
            return types.SimpleNamespace(
                index=index,
                body=body,
                local=np.eye(4),
                vertices=np.zeros((3, 3)),
                faces=np.array([[0, 1, 2]]),
            )

        shapes = [shape(0, -1), shape(1, -1), shape(2, 0)]
        tools = sim_tools.SimTools.__new__(sim_tools.SimTools)
        tools.env = types.SimpleNamespace(
            scene=types.SimpleNamespace(embodiment=None, index_model=types.SimpleNamespace(shape_label=None)),
            handles=types.SimpleNamespace(
                objects={
                    "cube": types.SimpleNamespace(category="cube", body=0),
                    "marker": types.SimpleNamespace(category="cube_marker", body=-1),
                }
            ),
        )
        tools.bus = types.SimpleNamespace(
            snapshot=types.SimpleNamespace(body_q=np.array([[0, 0, 0, 1, 0, 0, 0]], dtype=np.float64))
        )
        try:
            import newton  # noqa: F401
        except Exception as error:  # pragma: no cover
            self.skipTest(f"newton unavailable: {error}")
        with mock.patch.object(sim_tools, "shape_meshes", lambda *a, **k: shapes):
            world = tools.get_collision_world(exclude_categories=["cube"])
        self.assertEqual(world["mesh_names"], ["shape_0_collider", "shape_1_collider"])


if __name__ == "__main__":
    unittest.main()
