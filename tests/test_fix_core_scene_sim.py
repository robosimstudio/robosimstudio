# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the scene and sim fixes: parking, binding, snapshot, names."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

TIER = "fast"
"""No GPU: specs, fakes and numpy arrays."""


def _arr(values):
    return SimpleNamespace(numpy=lambda: np.asarray(values, dtype=float))


class TestOneParkLattice(unittest.TestCase):
    def test_a_hand_assembled_scene_parks_on_the_task_lattice(self) -> None:
        from robosimstudio.scene.parking import scene_park_poses
        from robosimstudio.specs import park_lattice
        from robosimstudio.specs import task as task_module

        names = [f"o{i}" for i in range(11)]
        scene = SimpleNamespace(request=SimpleNamespace(), objects=dict.fromkeys(names))
        # A row length other than the shipped one: a hard-coded copy would not follow it.
        row = task_module._PARK_ROW
        task_module._PARK_ROW = 5
        try:
            poses = scene_park_poses(scene)
            expected = {name: park_lattice(index) for index, name in enumerate(names)}
        finally:
            task_module._PARK_ROW = row
        self.assertAlmostEqual(task_module.PARK_PITCH, expected["o5"][1], msg="the sixth object starts row two")
        for name in names:
            np.testing.assert_allclose(poses[name], expected[name])


class TestABindingKeepsTheCell(unittest.TestCase):
    def test_an_empty_scene_on_a_binding_is_the_requests_cell(self) -> None:
        from robosimstudio.scene.request import BuildRequest
        from robosimstudio.specs import Binding

        request = BuildRequest(cell="bench_075", embodiment="arm", binding=Binding(cross_arm_collisions=True))
        have = request.binding_for()
        self.assertEqual("bench_075", have.scene)
        self.assertEqual("arm", have.embodiment)
        self.assertTrue(have.cross_arm_collisions)

    def test_a_binding_that_names_a_cell_keeps_it(self) -> None:
        from robosimstudio.scene.request import BuildRequest
        from robosimstudio.specs import Binding

        request = BuildRequest(cell="bench_075", binding=Binding(scene="other"))
        self.assertEqual("other", request.binding_for().scene)


class TestSnapshotSpeedsAreNorms(unittest.TestCase):
    def test_a_diagonal_velocity_is_its_magnitude(self) -> None:
        from robosimstudio.scene.snapshot import _objects

        env = SimpleNamespace(
            state=SimpleNamespace(
                body_q=_arr([[0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]]),
                body_qd=_arr([[0.1, 0.1, 0.1, 3.0, 4.0, 0.0]]),
            ),
            handles=SimpleNamespace(objects={"cube": SimpleNamespace(body=0, static=False)}),
        )
        (row,) = _objects(env, SimpleNamespace(body_mass=_arr([0.1])))
        self.assertAlmostEqual(0.1 * np.sqrt(3.0), row["linear_speed"], places=6)
        self.assertAlmostEqual(5.0, row["angular_speed"], places=6)


class TestPenetrationCopiesShapeBodyOnce(unittest.TestCase):
    def test_many_pairs_one_copy(self) -> None:
        from robosimstudio.scene.snapshot import _penetration

        copies = []

        def shape_body():
            copies.append(1)
            return np.arange(6)

        model = SimpleNamespace(shape_body=SimpleNamespace(numpy=shape_body), shape_label=[f"s{i}" for i in range(6)])
        probe = SimpleNamespace(update=lambda: None, penetrations=lambda: [(0, 1, 0.01), (2, 3, 0.02), (4, 5, 0.03)])
        out = _penetration(SimpleNamespace(contacts=probe), model, {0: "cube"})
        self.assertEqual(3, len(out))
        self.assertEqual(["cube", "s1"], out[0]["between"])
        self.assertEqual(1, len(copies))


class TestNamesAgreeWithGetitem(unittest.TestCase):
    def test_a_region_is_in_the_sim(self) -> None:
        from robosimstudio.sim.sim import Sim

        handles = SimpleNamespace(
            objects={"cube": object()}, deformables={}, statics={"post": object()}, regions={"goal": object()}
        )
        handles.find = lambda name: next(
            (t[name] for t in (handles.objects, handles.deformables, handles.statics, handles.regions) if name in t),
            None,
        )
        sim = Sim.__new__(Sim)
        sim.scene = SimpleNamespace(by_name={}, handles=handles)
        sim.refs = {}
        self.assertIn("goal", sim)
        self.assertIn("post", sim)
        self.assertNotIn("nothing", sim)
        self.assertEqual(["cube", "goal", "post"], sim.names())


if __name__ == "__main__":
    unittest.main()
