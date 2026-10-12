# SPDX-License-Identifier: Apache-2.0
"""A :class:`Finish` becoming glTF, which is how the page shades what RTX shades.

The encoder is worth its own module because two of its three jobs are places the
obvious version is quietly wrong -- a colour space that differs from the one
viser's flat path wants, and an exporter that rounds a field glTF defines as a
float -- and both failures look like "the lighting is off" rather than like a
bug in an encoder.
"""

from __future__ import annotations

import json
import struct
import unittest

import numpy as np

from robosimstudio.core.render.finish import DEFAULT, GLOSS, MATTE, Finish, register_texture
from robosimstudio.core.render.palette import srgb_to_linear
from robosimstudio_studio.finish_gltf import (
    clear_cache,
    glb_for,
    group_by_material,
    material_key,
    unweld,
    wants_pbr,
)

TIER = "fast"

QUAD = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
FACES = np.array([[0, 1, 2], [0, 2, 3]])


def _materials(blob: bytes) -> list[dict]:
    """The material block out of a GLB's JSON chunk."""
    size, _kind = struct.unpack_from("<II", blob, 12)
    return json.loads(blob[20 : 20 + size].decode("utf-8")).get("materials", [])


class TestTheColourIsLinear(unittest.TestCase):
    """glTF's ``baseColorFactor`` is linear; a viser ``color=`` triple is not."""

    def test_the_palette_value_is_decoded_on_the_way_in(self) -> None:
        color = (0.8, 0.2, 0.2)
        pbr = _materials(glb_for(QUAD, FACES, color, GLOSS))[0]["pbrMetallicRoughness"]
        self.assertEqual(pbr["baseColorFactor"][:3], list(srgb_to_linear(color)))

    def test_it_is_not_rounded_through_eight_bits(self) -> None:
        """trimesh keeps the factor as uint8; a display 0.2 would come back 5 % low."""
        pbr = _materials(glb_for(QUAD, FACES, (0.8, 0.2, 0.2), GLOSS))[0]["pbrMetallicRoughness"]
        self.assertNotAlmostEqual(pbr["baseColorFactor"][1], 8 / 255, places=6)
        self.assertAlmostEqual(pbr["baseColorFactor"][1], srgb_to_linear((0, 0.2, 0))[1], places=12)

    def test_the_mesh_survives_the_factor_rewrite(self) -> None:
        """The patch edits one chunk of a container the rest of the file indexes into."""
        import trimesh

        scene = trimesh.load(__import__("io").BytesIO(glb_for(QUAD, FACES, (0.5, 0.5, 0.5), MATTE)), file_type="glb")
        mesh = next(iter(scene.geometry.values()))
        self.assertEqual(len(mesh.faces), 2)
        self.assertEqual(len(mesh.vertices), 4)


class TestTheFinishReachesTheMaterial(unittest.TestCase):
    def test_roughness_and_metalness_are_the_finish_s(self) -> None:
        pbr = _materials(glb_for(QUAD, FACES, (0.5, 0.5, 0.5), Finish(roughness=0.37, metallic=0.82)))[0]
        self.assertAlmostEqual(pbr["pbrMetallicRoughness"]["roughnessFactor"], 0.37, places=6)
        self.assertAlmostEqual(pbr["pbrMetallicRoughness"]["metallicFactor"], 0.82, places=6)

    def test_a_translucent_finish_says_so(self) -> None:
        """Or a loader draws the glazing and the region boxes as walls."""
        self.assertEqual(
            _materials(glb_for(QUAD, FACES, (0.5, 0.5, 0.5), Finish(opacity=0.3)))[0]["alphaMode"], "BLEND"
        )

    def test_an_opaque_one_says_that(self) -> None:
        self.assertEqual(_materials(glb_for(QUAD, FACES, (0.5, 0.5, 0.5), DEFAULT))[0]["alphaMode"], "OPAQUE")


class TestTheCache(unittest.TestCase):
    def setUp(self) -> None:
        clear_cache()

    def test_the_same_material_encodes_once(self) -> None:
        self.assertIs(glb_for(QUAD, FACES, (0.4, 0.4, 0.4), MATTE), glb_for(QUAD, FACES, (0.4, 0.4, 0.4), MATTE))

    def test_a_different_colour_is_a_different_glb(self) -> None:
        """A GLB carries one material, which is why batches are cut by colour too."""
        self.assertIsNot(glb_for(QUAD, FACES, (0.4, 0.4, 0.4), MATTE), glb_for(QUAD, FACES, (0.9, 0.1, 0.1), MATTE))

    def test_a_different_finish_is_a_different_glb(self) -> None:
        self.assertIsNot(glb_for(QUAD, FACES, (0.4, 0.4, 0.4), MATTE), glb_for(QUAD, FACES, (0.4, 0.4, 0.4), GLOSS))

    def test_different_geometry_is_a_different_glb(self) -> None:
        other = QUAD * 2.0
        self.assertIsNot(glb_for(QUAD, FACES, (0.4, 0.4, 0.4), MATTE), glb_for(other, FACES, (0.4, 0.4, 0.4), MATTE))


class TestNormals(unittest.TestCase):
    """glTF flat-shades a mesh that carries none, which is right for the room's
    boxes and wrong for a cylinder or a draped sheet."""

    def _attrs(self, blob: bytes) -> list[str]:
        size, _kind = struct.unpack_from("<II", blob, 12)
        doc = json.loads(blob[20 : 20 + size].decode("utf-8"))
        return list(doc["meshes"][0]["primitives"][0]["attributes"])

    def test_none_by_default(self) -> None:
        self.assertNotIn("NORMAL", self._attrs(glb_for(QUAD, FACES, (0.5, 0.5, 0.5), MATTE)))

    def test_the_caller_s_are_carried(self) -> None:
        normals = np.tile([0.0, 0.0, 1.0], (len(QUAD), 1))
        self.assertIn("NORMAL", self._attrs(glb_for(QUAD, FACES, (0.5, 0.5, 0.5), MATTE, normals=normals)))

    def test_smooth_derives_them(self) -> None:
        """For a caller whose geometry is curved but whose normals do not line
        up with its vertices -- a garment shell has twice the vertices."""
        self.assertIn("NORMAL", self._attrs(glb_for(QUAD, FACES, (0.5, 0.5, 0.5), MATTE, smooth=True)))

    def test_two_normal_sets_are_two_glbs(self) -> None:
        up = np.tile([0.0, 0.0, 1.0], (len(QUAD), 1))
        side = np.tile([0.0, 1.0, 0.0], (len(QUAD), 1))
        self.assertIsNot(
            glb_for(QUAD, FACES, (0.4, 0.4, 0.4), MATTE, normals=up),
            glb_for(QUAD, FACES, (0.4, 0.4, 0.4), MATTE, normals=side),
        )


class TestTheRoughnessMap(unittest.TestCase):
    """The tracer wires the map's red channel *into* roughness, replacing the
    scalar; glTF reads green and *multiplies* by the factor."""

    MAP = "test_finish_gltf_ramp"

    @classmethod
    def setUpClass(cls) -> None:
        """A registered map of its own, so this runs rather than skipping on
        whether a task suite happens to be imported."""
        register_texture(cls.MAP, lambda: np.tile(np.linspace(0, 255, 16, dtype=np.uint8), (16, 1)))
        clear_cache()

    def _pbr(self, finish: Finish) -> dict:
        uvs = np.stack([QUAD[:, 0], QUAD[:, 1]], axis=1)
        blob = glb_for(QUAD, FACES, (0.5, 0.5, 0.5), finish, uvs=uvs)
        return _materials(blob)[0]["pbrMetallicRoughness"]

    def test_the_map_is_bound(self) -> None:
        self.assertIn("metallicRoughnessTexture", self._pbr(Finish(roughness_texture=self.MAP)))

    def test_the_factor_is_one_so_the_map_is_what_lands(self) -> None:
        """glTF multiplies; the tracer replaces. A factor of 1.0 is how the
        multiply comes out at the value the tracer would have used."""
        self.assertEqual(self._pbr(Finish(roughness=0.15, roughness_texture=self.MAP))["roughnessFactor"], 1.0)

    def test_without_a_map_the_stated_roughness_stands(self) -> None:
        self.assertAlmostEqual(self._pbr(Finish(roughness=0.15))["roughnessFactor"], 0.15, places=6)

    def test_two_maps_are_two_materials(self) -> None:
        """`material_key` used to drop `roughness_texture`, which would hand the
        second finish the first one's GLB."""
        self.assertNotEqual(
            material_key((0.5, 0.5, 0.5), Finish(roughness_texture="a")),
            material_key((0.5, 0.5, 0.5), Finish(roughness_texture="b")),
        )


class TestWhichPathAShapeTakes(unittest.TestCase):
    """A deforming mesh cannot use the cache, so this decides what is worth
    re-encoding every frame. See :data:`MATTE_ENOUGH`."""

    def test_a_garment_nobody_dressed_stays_flat(self) -> None:
        self.assertFalse(wants_pbr(MATTE))

    def test_and_so_does_something_rougher_still(self) -> None:
        self.assertFalse(wants_pbr(Finish(roughness=0.95)))

    def test_gloss_is_worth_a_glb(self) -> None:
        self.assertTrue(wants_pbr(GLOSS))

    def test_so_is_metal(self) -> None:
        self.assertTrue(wants_pbr(Finish(roughness=0.95, metallic=0.9)))

    def test_so_is_either_texture_slot(self) -> None:
        self.assertTrue(wants_pbr(Finish(roughness=0.95, texture="x")))
        self.assertTrue(wants_pbr(Finish(roughness=0.95, roughness_texture="x")))

    def test_no_finish_at_all_is_the_flat_path(self) -> None:
        """A caller holding a scene with no handles draws the way it always did."""
        self.assertFalse(wants_pbr(None))


class TestCuttingABatchByMaterial(unittest.TestCase):
    def test_shapes_sharing_a_material_share_a_group(self) -> None:
        colors = np.array([[0.8, 0.2, 0.2], [0.8, 0.2, 0.2], [0.2, 0.2, 0.8]])
        groups = group_by_material([0, 1, 2], colors, {0: GLOSS, 1: GLOSS, 2: GLOSS}, DEFAULT)
        self.assertEqual(sorted(len(rows) for rows in groups.values()), [1, 2])

    def test_the_same_colour_in_two_finishes_is_two_groups(self) -> None:
        colors = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        groups = group_by_material([0, 1], colors, {0: GLOSS, 1: MATTE}, DEFAULT)
        self.assertEqual(len(groups), 2)

    def test_a_shape_nobody_dressed_gets_the_default(self) -> None:
        colors = np.array([[0.5, 0.5, 0.5]])
        groups = group_by_material([0], colors, {}, MATTE)
        self.assertEqual(next(iter(groups))[0], material_key((0.5, 0.5, 0.5), MATTE))

    def test_the_order_is_stable(self) -> None:
        """Groups become node names, so a batch that reshuffles them every frame
        would churn the scene tree for nothing."""
        colors = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.1, 0.1, 0.1]])
        keys = [list(group_by_material([0, 1, 2], colors, {}, DEFAULT)) for _ in range(3)]
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(keys[1], keys[2])

    def test_no_colours_at_all_is_still_one_group(self) -> None:
        self.assertEqual(len(group_by_material([0, 1], None, {}, DEFAULT)), 1)


class TestUnwelding(unittest.TestCase):
    def test_every_corner_gets_its_own_vertex(self) -> None:
        """A box corner needs three box projections and a vertex carries one."""
        points, faces = unweld(QUAD, FACES)
        self.assertEqual(len(points), 6)
        self.assertEqual(faces.tolist(), [[0, 1, 2], [3, 4, 5]])

    def test_the_surface_is_unchanged(self) -> None:
        points, faces = unweld(QUAD, FACES)
        self.assertTrue(np.allclose(points[faces].reshape(-1, 3), QUAD[FACES].reshape(-1, 3)))


if __name__ == "__main__":
    unittest.main()
