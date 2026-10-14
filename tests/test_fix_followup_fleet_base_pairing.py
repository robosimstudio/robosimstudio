# SPDX-License-Identifier: Apache-2.0
"""An untried pairing on a task with a fleet base narrows the page to one world
instead of dying on the base's ``binding_for`` as a ``KeyError``."""

import unittest
from types import SimpleNamespace
from unittest import mock

from robosimstudio.scene.compiler import CannotReplicate
from robosimstudio.specs import Binding, TaskSpec, replace
from robosimstudio_studio import bench_session

TIER = "fast"


def _base():
    native = Binding(scene="bench_0728", embodiment="panda", graphs=("pick",))
    return TaskSpec(name="base", scene="bench_0728", embodiment="panda", bindings=(native,))


def _session():
    return SimpleNamespace(
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


class TestFleetBasePairing(unittest.TestCase):
    def test_an_undeclared_pairing_on_a_fleet_base_cannot_replicate(self):
        base = _base()
        instance = replace(base, name="instance", fleet_base="base")
        fleet = mock.Mock(side_effect=AssertionError("FleetSession must not be reached"))
        with (
            mock.patch("robosimstudio_studio.fleet_session.FleetSession", fleet),
            mock.patch("robosimstudio_bench.taskbuild.registry.get_task", return_value=base),
            mock.patch.object(bench_session, "admits", return_value=(True, "")),
            self.assertRaises(CannotReplicate) as caught,
        ):
            bench_session.load_fleet(_session(), instance, 0, "instance", 0.0, "", "panda_robotiq", "")
        self.assertIn("fleet base 'base'", str(caught.exception))
        fleet.assert_not_called()

    def test_a_pairing_the_base_declares_reaches_the_fleet(self):
        base = _base()
        instance = replace(base, name="instance", fleet_base="base")
        fleet = mock.Mock(side_effect=RuntimeError("stop here"))
        with (
            mock.patch("robosimstudio_studio.fleet_session.FleetSession", fleet),
            mock.patch("robosimstudio_bench.taskbuild.registry.get_task", return_value=base),
            mock.patch("traceback.print_exc"),
        ):
            ok = bench_session.load_fleet(_session(), instance, 0, "instance", 0.0)
        self.assertFalse(ok)
        fleet.assert_called_once()

    def test_a_task_without_a_base_is_not_checked(self):
        bench_session.base_declares(_base().with_pairing("", "panda_robotiq", ""))


if __name__ == "__main__":
    unittest.main()
