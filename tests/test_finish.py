# SPDX-License-Identifier: Apache-2.0
"""What a surface does with light: the ``Finish`` seam.

Static. A finish is a display statement resolved the way ``color`` is, and what
these pin is the resolution rule rather than any picture: a spec that states a
finish keeps it, one that states nothing gets the finish its painted colour
implies, and the table a scene hands the path tracer is complete for every
shape. The palette's own scenery ladder is what keys the defaults, so a change
there that forgets this module fails here first.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace

import numpy as np
from robosimstudio.domains.cloth import ClothSpec

from robosimstudio.core.render import finish as finish_mod
from robosimstudio.core.render.finish import (
    BRUSHED_METAL,
    DEFAULT,
    GLOSS,
    MATTE,
    SATIN,
    Finish,
    finish_for,
    register_color_finish,
    register_texture,
    texture_path,
)
from robosimstudio.core.render.palette import DEFORMABLE, FIXTURE, FLOOR, GROUND, METAL, OBJECT, ROBOT_DARK
from robosimstudio.core.render.scenery import SCENERY, get_scenery, register_scenery
from robosimstudio.core.render.spec import RenderSpec
from robosimstudio.solvers.render import look_kwargs
from robosimstudio.specs import EmbodimentSpec, ObjectSpec, StaticSpec

TIER = "fast"
"""No GPU. Gates every commit."""


class TestFinishIsValidated(unittest.TestCase):
    def test_defaults_are_plastic(self) -> None:
        f = Finish()
        self.assertEqual((f.roughness, f.metallic, f.opacity, f.ior, f.texture), (0.6, 0.0, 1.0, 1.5, ""))
        self.assertFalse(f.translucent)

    def test_out_of_range_is_refused(self) -> None:
        for bad in ({"roughness": 1.2}, {"metallic": -0.1}, {"opacity": 0.0}, {"opacity": 1.5}, {"ior": 0.9}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                Finish(**bad)

    def test_translucent_below_one(self) -> None:
        self.assertTrue(Finish(opacity=0.35).translucent)

    def test_hashable_so_a_material_library_can_key_on_it(self) -> None:
        self.assertEqual(hash(Finish(roughness=0.3)), hash(Finish(roughness=0.3)))
        self.assertEqual({Finish(roughness=0.3): 1}[replace(GLOSS)], 1)


class TestDefaultsFollowTheColour(unittest.TestCase):
    """The palette's ladder means something, and the meaning carries a finish."""

    def test_metal_is_metal_whatever_the_role(self) -> None:
        self.assertEqual(finish_for(METAL, "objects"), BRUSHED_METAL)
        self.assertEqual(finish_for(METAL, "fixtures"), BRUSHED_METAL)

    def test_scenery_and_deformables_are_matte(self) -> None:
        for color in (FLOOR, GROUND, DEFORMABLE):
            with self.subTest(color=color):
                self.assertEqual(finish_for(color).roughness, MATTE.roughness)
        self.assertEqual(finish_for(FIXTURE).roughness, 0.8)

    def test_the_robot_is_satin(self) -> None:
        self.assertEqual(finish_for(ROBOT_DARK, "objects"), SATIN)
        self.assertEqual(finish_for(OBJECT, "robot"), SATIN)

    def test_an_unkeyed_colour_falls_back_to_the_role(self) -> None:
        self.assertEqual(finish_for((0.5, 0.1, 0.9), "objects"), DEFAULT)
        self.assertEqual(finish_for((0.5, 0.1, 0.9), "deformables"), MATTE)
        self.assertEqual(finish_for(None, "surface"), MATTE)

    def test_a_colour_within_tolerance_still_matches(self) -> None:
        nudged = tuple(float(np.float32(v)) + 1.0e-3 for v in METAL)
        self.assertEqual(finish_for(nudged), BRUSHED_METAL)

    def test_a_module_may_key_its_own_colour(self) -> None:
        color = (0.123, 0.456, 0.789)
        try:
            register_color_finish(color, GLOSS)
            self.assertEqual(finish_for(color), GLOSS)
            register_color_finish(color, MATTE)
            self.assertEqual(finish_for(color), MATTE, "re-registering a colour replaces its finish")
        finally:
            finish_mod._BY_COLOR[:] = [(c, f) for c, f in finish_mod._BY_COLOR if c != color]


class TestSpecsCarryAFinish(unittest.TestCase):
    def test_every_display_spec_defaults_to_none(self) -> None:
        self.assertIsNone(ObjectSpec(name="a").finish)
        self.assertIsNone(StaticSpec(name="a", half_extents=(0.1, 0.1, 0.1)).finish)
        self.assertIsNone(ClothSpec(name="a").finish)
        self.assertIsNone(EmbodimentSpec.__dataclass_fields__["finish"].default)

    def test_a_stated_finish_round_trips(self) -> None:
        spec = ObjectSpec(name="a", finish=Finish(roughness=0.2, texture="x"))
        self.assertEqual(spec.finish.texture, "x")


class TestRenderSpecReachesRTX(unittest.TestCase):
    def test_a_bare_spec_says_nothing(self) -> None:
        self.assertEqual(look_kwargs("rtx", RenderSpec()), {})

    def test_scenery_quality_and_map_are_rtx_only(self) -> None:
        spec = RenderSpec(scenery="lab", quality="final", environment_map="sky.exr", environment="lab")
        self.assertEqual(
            look_kwargs("rtx", spec),
            {"environment": "lab", "environment_map": "sky.exr", "scenery": "lab", "quality": "final"},
        )
        self.assertEqual(look_kwargs("tiled", spec), {})


class TestTextures(unittest.TestCase):
    def test_a_registered_generator_renders_once_into_the_cache(self) -> None:
        calls = []

        def generator():
            calls.append(1)
            return np.full((4, 6, 3), 0.5)

        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("ROBOSIMSTUDIO_CACHE")
            os.environ["ROBOSIMSTUDIO_CACHE"] = tmp
            try:
                register_texture("_test_probe", generator, version=3)
                first = texture_path("_test_probe")
                second = texture_path("_test_probe")
            finally:
                if previous is None:
                    del os.environ["ROBOSIMSTUDIO_CACHE"]
                else:
                    os.environ["ROBOSIMSTUDIO_CACHE"] = previous
                finish_mod.TEXTURES._items.pop("_test_probe", None)
            self.assertEqual(first, second)
            self.assertTrue(first.endswith("_test_probe_v3.png"))
            self.assertEqual(len(calls), 1, "the second lookup is served from the cache")
            from PIL import Image

            with Image.open(first) as image:
                self.assertEqual(image.size, (6, 4))

    def test_a_file_path_is_returned_absolute(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            self.assertEqual(texture_path(f.name), os.path.abspath(f.name))

    def test_an_unknown_name_is_loud(self) -> None:
        with self.assertRaises(FileNotFoundError):
            texture_path("no_such_texture_anywhere")


class TestSceneryRegistry(unittest.TestCase):
    def test_the_same_builder_may_arrive_twice_but_not_a_different_one(self) -> None:
        name = "_test_scenery"

        def build(kit):
            return None

        def other(kit):
            return None

        try:
            register_scenery(name, build)
            register_scenery(name, build)
            self.assertIs(get_scenery(name), build)
            with self.assertRaises(ValueError):
                register_scenery(name, other)
        finally:
            SCENERY._items.pop(name, None)

    def test_an_unknown_name_is_loud(self) -> None:
        with self.assertRaises(KeyError):
            get_scenery("_no_such_scenery")


if __name__ == "__main__":
    unittest.main()


class TestStageHelpers(unittest.TestCase):
    """The two pieces of geometry arithmetic the stage decoration leans on."""

    def test_box_projection_puts_each_face_on_its_own_plane(self) -> None:
        from robosimstudio.core.render.uv import box_projected_uvs

        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        tris = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4]], dtype=np.int32)  # two on z, one on y
        uv = box_projected_uvs(points, tris)
        self.assertEqual(uv.shape, (9, 2))
        np.testing.assert_allclose(uv[:3], [[0, 0], [1, 0], [1, 1]])  # z-facing: (x, y)
        np.testing.assert_allclose(uv[6:9], [[0, 0], [1, 0], [0, 1]])  # y-facing: (x, z)

    def test_normals_of_a_flat_sheet_point_up(self) -> None:
        from robosimstudio.core.render.uv import planar_uvs
        from robosimstudio.solvers.render.rtx import _vertex_normals

        points = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], dtype=np.float32)
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        np.testing.assert_allclose(_vertex_normals(points, tris), np.tile([0, 0, 1], (4, 1)), atol=1e-6)
        # Metres along the sheet's principal axes: a 2 x 1 rectangle spans 2 by 1.
        uv = planar_uvs(points)
        self.assertEqual(uv.shape, (4, 2))
        np.testing.assert_allclose(np.sort(uv.max(axis=0)), [1.0, 2.0], atol=1e-5)
