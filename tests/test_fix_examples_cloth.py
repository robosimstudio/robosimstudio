# SPDX-License-Identifier: Apache-2.0
"""``examples/02_cloth/cloth.py``: a membrane mesh is simplified, and a sheet zeroes only its own creases.

No GPU: a recording builder and a model made of CPU warp arrays.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

TIER = "fast"
"""No GPU. A stand-in builder and model."""


class _Builder:
    """Records what ``add_cloth_mesh`` was handed."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add_cloth_mesh(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestTheMembraneMeshIsSimplified(unittest.TestCase):
    def test_simplify_faces_and_scale_reach_the_builder(self) -> None:
        import trimesh
        from robosimstudio.domains.cloth import ClothMaterialSpec, ClothSpec, EntityCloth

        import robosimstudio as rss

        sphere = trimesh.creation.icosphere(subdivisions=2)  # 320 faces
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sheet.obj"
            sphere.export(path)
            spec = ClothSpec(
                name="shirt",
                mesh="mesh",
                asset=rss.AssetSpec(path=str(path), scale=2.0),
                material=ClothMaterialSpec(shell="membrane"),
                simplify_faces=40,
            )
            entity = EntityCloth(spec)
            builder = _Builder()
            # ``fast_simplification`` is optional: stand in a coarser sphere for the decimated mesh.
            coarse = trimesh.creation.icosphere(subdivisions=1)  # 80 faces
            with mock.patch.object(trimesh.Trimesh, "simplify_quadric_decimation", return_value=coarse):
                entity.build(builder, None)
                want_vertices, want_faces = entity.surface()

        (call,) = builder.calls
        self.assertLess(len(call["indices"]) // 3, len(sphere.faces), "simplify_faces held on a membrane")
        self.assertEqual(len(want_faces) * 3, len(call["indices"]))
        self.assertEqual(len(want_vertices), entity.surface_count)
        # Scaled once: by ``surface``, not again by the builder.
        self.assertEqual(1.0, call["scale"])
        radius = max(float(np.linalg.norm([v[0], v[1], v[2]])) for v in call["vertices"])
        self.assertAlmostEqual(2.0, radius, delta=0.05)


class TestZeroRestAngleIsPerSheet(unittest.TestCase):
    def test_one_sheet_does_not_flatten_anothers_creases(self) -> None:
        import warp as wp
        from robosimstudio.domains.cloth import ClothMaterialSpec, ClothSpec, EntityCloth

        wp.init()
        # Two sheets of four particles each; one edge per sheet, column 2 on the sheet.
        edges = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
        model = SimpleNamespace(
            edge_indices=wp.array(edges.reshape(-1), dtype=wp.int32, device="cpu"),
            edge_rest_angle=wp.array(np.array([0.5, 0.7], dtype=np.float32), dtype=wp.float32, device="cpu"),
        )
        flat = EntityCloth(ClothSpec(name="flat", material=ClothMaterialSpec(zero_rest_angle=True)))
        flat.range.particles = (0, 4)
        creased = EntityCloth(ClothSpec(name="creased", material=ClothMaterialSpec(zero_rest_angle=False)))
        creased.range.particles = (4, 8)

        flat.apply_material(model)
        creased.apply_material(model)

        np.testing.assert_allclose([0.0, 0.7], model.edge_rest_angle.numpy())

    def test_every_world_of_a_replicated_sheet_is_flattened(self) -> None:
        import warp as wp
        from robosimstudio.domains.cloth import ClothMaterialSpec, ClothSpec, EntityCloth

        wp.init()
        # Two worlds of eight particles: in each, a flat sheet on 0..3 and a creased one on 4..7.
        edges = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], dtype=np.int32)
        model = SimpleNamespace(
            edge_indices=wp.array(edges.reshape(-1), dtype=wp.int32, device="cpu"),
            edge_rest_angle=wp.array(np.array([0.5, 0.7, 0.5, 0.7], dtype=np.float32), dtype=wp.float32, device="cpu"),
        )
        worlds = SimpleNamespace(num_worlds=2, particles_per_world=8)
        flat = EntityCloth(ClothSpec(name="flat", material=ClothMaterialSpec(zero_rest_angle=True)))
        flat.range.particles = (0, 4)
        flat.worlds = worlds
        creased = EntityCloth(ClothSpec(name="creased", material=ClothMaterialSpec(zero_rest_angle=False)))
        creased.range.particles = (4, 8)
        creased.worlds = worlds

        flat.apply_material(model)
        creased.apply_material(model)

        np.testing.assert_allclose([0.0, 0.7, 0.0, 0.7], model.edge_rest_angle.numpy())


class TestSpecHintsResolve(unittest.TestCase):
    def test_get_type_hints_on_the_cloth_and_soft_specs(self) -> None:
        import typing

        from robosimstudio.domains.cloth import ClothSpec, HeldClothSpec
        from robosimstudio.domains.soft import SoftSpec

        for spec in (ClothSpec, HeldClothSpec, SoftSpec):
            with self.subTest(spec=spec.__name__):
                self.assertIn("name", typing.get_type_hints(spec))


if __name__ == "__main__":
    unittest.main()
