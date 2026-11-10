# SPDX-License-Identifier: Apache-2.0
"""A rod's material pass writes every world's capsules on a replicated model, not world 0's.

:meth:`EntityRod.apply_model_materials` runs on the REPLICATED model in a wave
(``_compile_multi``: "every entity writes its own rows per world"), but read
the rod's template shape indices as rows -- so only world 0's capsules got
``mu_cable``. Every other world's rod fell through to ``mu_base``: measured on a
``cable_route/port_easy`` wave, mu 60 in world 0 and 2.0 in worlds 1+, which is
a pad-on-cable grip of sqrt(2 x 60) = 11 instead of 60. The terminal carry's
pinch let go in those worlds, and the cell scored 5/30 against 15-18/30 serially.

No device here: a stub model and a stub world map, the arithmetic only.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio_materials.rod.entity import EntityRod
from robosimstudio_materials.rod.spec import RodSpec

TIER = "fast"

PER_WORLD = 6
"""Shapes per world in the stub: 0 the table, 1 a gripper pad, 2-4 the rod, 5 a fixture."""

ROD = (2, 3, 4)


class _Array:
    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def numpy(self):
        return self.values.copy()

    def assign(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float32)


def _model(worlds: int):
    n = PER_WORLD * worlds
    labels = ["table", "pad", "cable_0", "cable_1", "cable_2", "port"] * worlds
    return SimpleNamespace(
        shape_count=n,
        shape_label=labels,
        shape_material_mu=_Array(np.full(n, 0.5)),
        shape_material_ke=_Array(np.zeros(n)),
        shape_material_kd=_Array(np.zeros(n)),
    )


def _rod(worlds: int | None, *, scene_friction: bool = True):
    rod = object.__new__(EntityRod)
    # ``scene_friction`` on, as the cable suites stage it, so the pass also
    # writes the pads, the bench and everything else -- by label, which was
    # already every world's.
    rod.spec = RodSpec(name="cable", scene_friction=scene_friction)
    rod.shapes = np.asarray(ROD, dtype=np.int32)
    rod.worlds = (
        None
        if worlds is None
        else SimpleNamespace(
            num_worlds=worlds,
            shapes_per_world=PER_WORLD,
            shape=lambda index, world: int(index) + int(world) * PER_WORLD,
        )
    )
    # The pads are found by the hand's body labels; one per world here.
    rod._blade_shapes = lambda model: {1 + w * PER_WORLD for w in range(max(1, worlds or 1))}
    rod._apply_dahl = lambda model: None
    return rod


class TestEveryWorldsRodIsTheRod(unittest.TestCase):
    def test_a_replicated_model_gets_the_cable_friction_in_every_world(self):
        model = _model(3)
        _rod(3).apply_model_materials(model)
        mu = model.shape_material_mu.values.reshape(3, PER_WORLD)
        spec = RodSpec(name="cable")
        for world in range(3):
            with self.subTest(world=world):
                np.testing.assert_allclose(mu[world, list(ROD)], spec.mu_cable)
                self.assertAlmostEqual(float(mu[world, 1]), spec.mu_gripper)
                self.assertAlmostEqual(float(mu[world, 0]), spec.mu_surface)
                self.assertAlmostEqual(float(mu[world, 5]), spec.mu_base)

    def test_every_world_is_world_zero(self):
        model = _model(4)
        _rod(4).apply_model_materials(model)
        for name in ("shape_material_mu", "shape_material_ke", "shape_material_kd"):
            rows = getattr(model, name).values.reshape(4, PER_WORLD)
            for world in range(1, 4):
                np.testing.assert_array_equal(rows[world], rows[0], err_msg=f"{name}, world {world}")

    def test_without_scene_friction_only_the_rod_is_written_in_every_world(self):
        model = _model(2)
        _rod(2, scene_friction=False).apply_model_materials(model)
        mu = model.shape_material_mu.values.reshape(2, PER_WORLD)
        for world in range(2):
            np.testing.assert_allclose(mu[world, list(ROD)], RodSpec(name="cable").mu_cable)
            np.testing.assert_allclose(mu[world, [0, 1, 5]], 0.5)

    def test_a_single_world_build_is_unchanged(self):
        model = _model(1)
        _rod(None).apply_model_materials(model)
        np.testing.assert_allclose(model.shape_material_mu.values[list(ROD)], RodSpec(name="cable").mu_cable)

    def test_the_template_model_is_read_as_one_world(self):
        """A map handed with a per-world-sized model writes the template rows only."""
        model = _model(1)
        _rod(3).apply_model_materials(model)
        np.testing.assert_allclose(model.shape_material_mu.values[list(ROD)], RodSpec(name="cable").mu_cable)


if __name__ == "__main__":
    unittest.main()
