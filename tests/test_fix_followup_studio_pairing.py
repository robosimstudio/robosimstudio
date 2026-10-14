# SPDX-License-Identifier: Apache-2.0
"""The studio and the run script find a declared binding however its hand is
spelled, an untried pairing reaches the fleet as a spec that declares it, and the
gripper menu does not offer a declared binding's own hand as untried."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from robosimstudio.robots.registry import get_embodiment, hands_of
from robosimstudio.specs import Binding, TaskRequirements, TaskSpec
from robosimstudio_studio import bench_session
from robosimstudio_studio.bench_panels import PanelTask
from robosimstudio_studio.session import Refused

TIER = "fast"

ROOT = Path(__file__).resolve().parents[1]


def _task(**kw):
    native = Binding(scene="bench_0728", embodiment="panda", graphs=("pick",))
    return TaskSpec(name="t", scene="bench_0728", embodiment="panda", bindings=(native,), **kw)


class TestResolvePairing(unittest.TestCase):
    def test_the_own_hand_is_the_declared_key(self):
        task = _task()
        (own,) = hands_of(get_embodiment("panda"))
        resolved, *key = bench_session.resolve_pairing(task, "", "", own)
        self.assertIs(resolved, task)
        self.assertEqual(tuple(key), ("bench_0728", "panda", ""))
        self.assertIs(task.binding_for(*key), task.bindings[0])

    def test_no_overrides_is_the_task_as_it_stands(self):
        task = _task()
        self.assertEqual(bench_session.resolve_pairing(task), (task, "", "", ""))

    def test_an_admitted_pairing_is_a_spec_that_declares_it(self):
        task = _task()
        with mock.patch.object(bench_session, "admits", return_value=(True, "")):
            resolved, *key = bench_session.resolve_pairing(task, "", "panda_robotiq", "")
        self.assertEqual(tuple(key), ("", "", ""))
        self.assertIn(("bench_0728", "panda_robotiq", resolved.gripper), resolved.binding_keys)
        self.assertEqual(len(resolved.bindings), 2)

    def test_a_refused_pairing_raises_refused(self):
        task = _task(requires=TaskRequirements(monolithic=True))
        with self.assertRaises(Refused):
            bench_session.resolve_pairing(task, "", "panda_robotiq", "")

    def test_the_fleet_gets_the_resolved_spec(self):
        task = _task()
        session = SimpleNamespace(
            config=SimpleNamespace(
                fleet_solution="",
                fleet_solution_kwargs={},
                show_worlds=4,
                device=None,
                fleet_loop=False,
            ),
            num_worlds=2,
            viewer=None,
            task_panel=None,
        )
        seen = {}

        def fleet(task, **kw):
            seen.update(task=task, **kw)
            raise RuntimeError("stop here")

        with (
            mock.patch("robosimstudio_studio.fleet_session.FleetSession", fleet),
            mock.patch.object(bench_session, "admits", return_value=(True, "")),
            mock.patch("traceback.print_exc"),
        ):
            ok = bench_session.load_fleet(session, task, 0, "t", 0.0, "", "panda_robotiq", "")
        self.assertFalse(ok)
        self.assertIsInstance(seen["task"], TaskSpec)
        self.assertEqual((seen["scene"], seen["embodiment"], seen["gripper"]), ("", "", ""))
        self.assertEqual(seen["task"].embodiment, "panda_robotiq")


class TestGripperMenu(unittest.TestCase):
    def test_the_own_hand_is_not_untried(self):
        task = _task()
        (own,) = hands_of(get_embodiment("panda"))
        with (
            mock.patch("robosimstudio.robots.registry.attachable_grippers", return_value=[own]),
            mock.patch("robosimstudio_bench.taskbuild.matrix.admits", return_value=(True, "")),
        ):
            options = PanelTask._axis_options(None, task, axis="gripper")
        self.assertNotIn(f"{own}  (untried)", options)


class TestRunScriptGraphClaims(unittest.TestCase):
    def test_the_own_hand_finds_the_declared_graphs(self):
        spec = importlib.util.spec_from_file_location("bench_run_followup", ROOT / "scripts" / "bench_run.py")
        run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run)
        task = _task()
        (own,) = hands_of(get_embodiment("panda"))
        job = SimpleNamespace(
            solution_kwargs={"graph": "pick"},
            allow_undeclared_graph=False,
            task="t",
            scene="",
            embodiment="",
            gripper=own,
        )
        with (
            mock.patch("robosimstudio_bench.taskbuild.registry.get_task", return_value=task),
            mock.patch.object(TaskSpec, "graph_path", return_value="pick"),
        ):
            run.check_graph_claims([job])  # raises SystemExit if the bare binding were found


if __name__ == "__main__":
    unittest.main()
