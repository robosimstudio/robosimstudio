# SPDX-License-Identifier: Apache-2.0
"""The IK backend routes each verb to the member that owns it, in the member's own dialect.

Two members: the solver family behind the protocol, which answers every
``solve_ik`` and the solved line; and an optional collision-aware planner,
which ``plan_linear`` prefers and ``plan_to_pose`` needs. Three rules the
runs were lost to before they were written down:

* ``honours_roll`` is the **solver's**, never the planner's. A servo that
  tested arrival on the planner's full-pose promise while the solver aligned
  an axis waited on a 4.3-degree roll error against a 3-degree gate, forever.
* each member gets the keywords its own signature declares and no others.
  Forwarding the union killed ``plan_linear`` with ``TypeError`` on the first
  descent of every hanging episode under the planner.
* a planner that finds no path falls back to the solved line -- unless the
  request carried world geometry or a contact allowance the line cannot
  honour, when a trajectory reported under a constraint nobody applied is
  worse than a refusal.

Hand-written stand-ins for both members; no GPU, no engine, no scene.
"""

from __future__ import annotations

import unittest

from robosimstudio_policy_agent.ik import IKBackend

TIER = "fast"


class _Planner:
    """Stands in for a cuRobo planner: world-aware, full-pose, its own refusal text."""

    honours_roll = True
    trajectory_needs_joint_reverse = False
    supports_world_aware_plan = True

    def __init__(self, answer=None):
        self.answer = answer
        self.calls: list[dict] = []
        self.last_refusal = "planner: no path"
        self.planned_to_pose = 0

    def plan_linear(self, start, end, *, arm_id=0, axis="", allow_start_contact=False):
        self.calls.append({"arm_id": arm_id, "axis": axis, "allow_start_contact": allow_start_contact})
        return self.answer

    def plan_to_pose(self, pose, **kwargs):
        self.planned_to_pose += 1
        return {"waypoints": [], "via": "planner"}

    def robot_collision_spheres(self, arm_id=-1):
        return [{"radius": 0.01}]

    def build_world_tsdf(self, *args, **kwargs):
        return {"voxels": 0}


def _backend(planner=None, *, honours_roll=False, line=None, prefer_planned=True) -> IKBackend:
    """An :class:`IKBackend` with no environment: the routing under test needs none."""
    backend = IKBackend.__new__(IKBackend)
    backend.__dict__.update(
        env=None,
        bus=None,
        planner=planner,
        prefer_planned=prefer_planned,
        tolerance=0.03,
        waypoints=40,
        iterations=80,
        fallback=True,
        family="fake",
        roll="held" if honours_roll else "free",
        honours_roll=honours_roll,
        last_solve_error=None,
        last_position_error=None,
        last_refusal="",
        _own_refusal="",
        _solvers={},
        line_calls=[],
    )

    def own(
        start, end, *, arm_id=0, tcp_offset=None, seed_joints=None, num_waypoints=0, hold_rotation=False, tolerance=None
    ):
        backend.line_calls.append(
            {"arm_id": arm_id, "num_waypoints": num_waypoints, "hold_rotation": hold_rotation, "tolerance": tolerance}
        )
        return line

    backend._own_plan_linear = own
    return backend


class TestTheRollIsTheSolvers(unittest.TestCase):
    def test_the_planner_cannot_make_the_backend_honour_roll(self):
        self.assertFalse(_backend(_Planner()).honours_roll)

    def test_a_held_roll_is_reported_whether_or_not_a_planner_is_installed(self):
        self.assertTrue(_backend(_Planner(), honours_roll=True).honours_roll)
        self.assertTrue(_backend(None, honours_roll=True).honours_roll)


class TestEachMemberGetsItsOwnKeywords(unittest.TestCase):
    def test_the_planner_gets_only_what_it_declares(self):
        planner = _Planner({"waypoints": [1]})
        backend = _backend(planner, line={"waypoints": [2]})
        out = backend.plan_linear({}, {}, arm_id=1, tolerance=0.01, hold_rotation=True, axis="z")
        self.assertEqual(out, {"waypoints": [1]})
        self.assertEqual(planner.calls, [{"arm_id": 1, "axis": "z", "allow_start_contact": False}])
        self.assertEqual(backend.line_calls, [])

    def test_the_fallback_gets_the_lines_keywords(self):
        planner = _Planner(None)
        backend = _backend(planner, line={"waypoints": [2]})
        out = backend.plan_linear({}, {}, arm_id=1, tolerance=0.01, hold_rotation=True, axis="z")
        self.assertEqual(out, {"waypoints": [2]})
        self.assertEqual(
            backend.line_calls, [{"arm_id": 1, "num_waypoints": 0, "hold_rotation": True, "tolerance": 0.01}]
        )

    def test_no_planner_means_the_line_alone(self):
        backend = _backend(None, line={"waypoints": [2]})
        self.assertEqual(backend.plan_linear({}, {}, axis="z", tolerance=0.02), {"waypoints": [2]})
        self.assertEqual(
            backend.line_calls, [{"arm_id": 0, "num_waypoints": 0, "hold_rotation": False, "tolerance": 0.02}]
        )


class TestTheFallbackIsRefusedWhenItCannotHonourTheRequest(unittest.TestCase):
    def test_world_geometry_is_never_discarded(self):
        backend = _backend(_Planner(None), line={"waypoints": [2]})
        self.assertIsNone(backend.plan_linear({}, {}, world_config={"cuboid": {}}))
        self.assertIn("world/attachment", backend.last_refusal)
        self.assertEqual(backend.line_calls, [])

    def test_a_contact_allowance_is_refused_rather_than_dropped(self):
        backend = _backend(_Planner(None), line={"waypoints": [2]})
        self.assertIsNone(backend.plan_linear({}, {}, allow_start_contact=True))
        self.assertIn("allow_start_contact", backend.last_refusal)
        self.assertEqual(backend.line_calls, [])

    def test_the_lines_own_refusal_is_not_shadowed(self):
        backend = _backend(_Planner(None), line=None)

        def own(start, end, **kwargs):
            backend.last_refusal = "line: waypoint 3 missed by 41 mm"
            return None

        backend._own_plan_linear = own
        self.assertIsNone(backend.plan_linear({}, {}, tolerance=0.01))
        self.assertEqual(backend.last_refusal, "line: waypoint 3 missed by 41 mm")

    def test_the_planners_refusal_is_reported_when_nothing_else_answered(self):
        backend = _backend(_Planner(None), line=None)
        self.assertIsNone(backend.plan_linear({}, {}))
        self.assertEqual(backend.last_refusal, "planner: no path")


class TestThePlannersVerbsAreThePlanners(unittest.TestCase):
    def test_plan_to_pose_is_none_without_a_planner_and_never_falls_back(self):
        self.assertIsNone(_backend(None).plan_to_pose({}))
        planner = _Planner()
        self.assertEqual(_backend(planner).plan_to_pose({})["via"], "planner")
        self.assertEqual(planner.planned_to_pose, 1)

    def test_world_awareness_follows_the_planner(self):
        self.assertTrue(_backend(_Planner()).supports_world_aware_plan)
        self.assertFalse(_backend(None).supports_world_aware_plan)
        self.assertFalse(_backend(_Planner(), prefer_planned=False).supports_world_aware_plan)

    def test_prefer_planned_false_routes_everything_to_the_line(self):
        planner = _Planner({"waypoints": [1]})
        backend = _backend(planner, line={"waypoints": [2]}, prefer_planned=False)
        self.assertEqual(backend.plan_linear({}, {}), {"waypoints": [2]})
        self.assertEqual(planner.calls, [])
        self.assertIsNone(backend.plan_to_pose({}))

    def test_the_world_builder_and_the_batch_solve_are_absent_without_a_planner(self):
        backend = _backend(None)
        with self.assertRaises(AttributeError):
            _ = backend.build_world_tsdf
        with self.assertRaises(AttributeError):
            _ = backend.solve_ik_batch
        self.assertEqual(_backend(_Planner()).build_world_tsdf(), {"voxels": 0})

    def test_a_planners_own_attribute_is_reachable_and_an_unknown_one_is_not(self):
        self.assertEqual(_backend(_Planner()).planned_to_pose, 0)
        with self.assertRaises(AttributeError):
            _ = _backend(None).no_such_thing


if __name__ == "__main__":
    unittest.main()
