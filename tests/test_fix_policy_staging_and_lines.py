# SPDX-License-Identifier: Apache-2.0
"""Staging leaves the graph's own tree alone, provenance reads a relative graph
directory, grasp tools refuse a name or role they do not know, and a line
reports the arrival it plans and the reason it was refused at the caller's
tolerance."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robosimstudio_policy_agent.graphs import NODES, outside_links, stage_graph
from robosimstudio_policy_agent.grasp_tools import GraspTools
from robosimstudio_policy_agent.ik import IKBackend
from robosimstudio_policy_agent.refusal import ToolRefusal
from robosimstudio_policy_agent.robot_tools import RobotTools

TIER = "fast"
"""Files in a scratch directory and stubbed solvers: no GPU, no scene."""


class _Scratch(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="rss-fix-policy-"))
        self.addCleanup(shutil.rmtree, self.root, True)


class TestStagingANestedScript(_Scratch):
    def test_the_graph_source_tree_is_not_written_through(self):
        graph = self.root / "graph"
        (graph / "scripts").mkdir(parents=True)
        (graph / "scripts" / "own.py").write_text("# own\n")
        node = sorted(NODES.glob("*.py"))[0].name
        workflow = {
            "nodes": {
                "a": {"type": "script", "script": "scripts/own.py"},
                "b": {"type": "script", "script": f"scripts/sub/{node}"},
            }
        }
        (graph / "workflow.json").write_text(json.dumps(workflow))
        before = sorted(str(p.relative_to(graph)) for p in graph.rglob("*"))
        staged = Path(stage_graph(graph, registries=[]))
        self.addCleanup(shutil.rmtree, staged, True)
        self.assertEqual(sorted(str(p.relative_to(graph)) for p in graph.rglob("*")), before)
        self.assertTrue((staged / "scripts" / "own.py").is_file())
        self.assertTrue((staged / "scripts" / "sub" / node).is_file())


class TestProvenanceOfARelativeGraph(_Scratch):
    def test_the_graphs_own_files_are_not_outside_it(self):
        graph = self.root / "g"
        graph.mkdir()
        (graph / "workflow.json").write_text("{}")
        staged = self.root / "st"
        staged.mkdir()
        os.symlink((graph / "workflow.json").resolve(), staged / "workflow.json")
        cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, cwd)
        self.assertEqual(outside_links("st", "g"), [])


def _handles(**arms):
    return SimpleNamespace(
        objects={"cube": SimpleNamespace(body=3)},
        statics={"table": SimpleNamespace(body=-1)},
        arms={"left": None, "right": None},
        work_arm=arms.get("work", "left"),
        support_arm=arms.get("support", "right"),
        entity=lambda name: {"cube": SimpleNamespace(body=3), "table": SimpleNamespace(body=-1)}[name],
    )


class TestGraspNamesAndRoles(unittest.TestCase):
    def _tools(self, cls):
        tools = cls.__new__(cls)
        tools.env = SimpleNamespace(handles=_handles())
        return tools

    def test_an_unknown_object_is_a_refusal_naming_the_choices(self):
        tools = self._tools(GraspTools)
        tools.env.handles.entity = lambda name: (_ for _ in ()).throw(KeyError(name))
        with self.assertRaises(ToolRefusal) as caught:
            tools.grasp_candidates("cueb", role="work")
        self.assertIn("cube", str(caught.exception))

    def test_a_static_reaches_the_bodiless_refusal(self):
        with self.assertRaisesRegex(ToolRefusal, "bodiless static"):
            self._tools(GraspTools).grasp_candidates("table", role="work")

    def test_a_mistyped_role_is_refused_not_mapped_to_the_support_arm(self):
        for cls in (GraspTools, RobotTools):
            tools = self._tools(cls)
            self.assertEqual(tools._arm(0, "support"), "right")
            for role in ("Work", "left", "worker"):
                with self.assertRaises(ValueError):
                    tools._arm(0, role)


class _Kinematics:
    dof = 1
    arm_coords = (0,)


class _Bus:
    def call(self, fn, *args):
        return fn(*args)


def _backend(solver):
    b = IKBackend.__new__(IKBackend)
    b.__dict__.update(
        env=None,
        bus=_Bus(),
        planner=None,
        prefer_planned=True,
        tolerance=0.03,
        rot_tolerance=np.radians(5.0),
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
        _solvers={"arm": solver},
    )
    b._arm_name = lambda arm_id: "arm"
    b._seed = lambda arm_id, seed_joints: (np.zeros((1, 1)), np.zeros(1))
    return b


def _pose(x):
    return {"position": {"x": x, "y": 0.0, "z": 0.0}, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


class _Converging:
    """Each solve lands closer than the one before, as an unconverged one does."""

    capabilities = frozenset()
    kinematics = _Kinematics

    def __init__(self):
        self.calls = 0

    def solve(self, seed, position, quat_xyzw, *, roll="free", active=None, iterations=0):
        self.calls += 1
        seed = np.asarray(seed, dtype=np.float64).reshape(-1, 1)
        return seed, np.tile([0.02 / self.calls, 0.0], (seed.shape[0], 1))


class _ByPlace:
    """Misses by ``mid`` between the ends, ``end_error`` at x=1, and 0.5 m whenever roll is held."""

    capabilities = frozenset()
    kinematics = _Kinematics

    def __init__(self, mid, end_error):
        self.mid, self.end_error = mid, end_error

    def solve(self, seed, position, quat_xyzw, *, roll="free", active=None, iterations=0):
        seed = np.asarray(seed, dtype=np.float64).reshape(-1, 1)
        x = float(np.asarray(position).reshape(-1, 3)[0, 0])
        error = 0.5 if roll == "held" else (self.end_error if x >= 1.0 else self.mid)
        return seed, np.tile([error, 0.0], (seed.shape[0], 1))


class TestALinesArrival(unittest.TestCase):
    def test_is_the_last_waypoints_error_not_a_fresh_solves(self):
        b = _backend(_Converging())
        self.assertIsNotNone(b.plan_linear(_pose(0.0), _pose(1.0), num_waypoints=4))
        self.assertAlmostEqual(b.last_position_error, 0.02 / 4, places=9)


class TestWhyNoLine(unittest.TestCase):
    def test_a_roll_free_end_is_judged_roll_free_at_the_callers_tolerance(self):
        b = _backend(_ByPlace(mid=0.0, end_error=0.02))
        self.assertIsNone(b.plan_linear(_pose(0.0), _pose(1.0), tolerance=0.01))
        self.assertIn("out of reach", b.last_refusal)
        self.assertIn("20 mm against a 10 mm tolerance", b.last_refusal)

    def test_a_reachable_end_behind_an_impossible_line_says_so(self):
        b = _backend(_ByPlace(mid=0.2, end_error=0.04))
        self.assertIsNone(b.plan_linear(_pose(0.0), _pose(1.0), tolerance=0.05))
        self.assertIn("reachable on its own", b.last_refusal)


if __name__ == "__main__":
    unittest.main()
