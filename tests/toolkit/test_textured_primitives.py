# SPDX-License-Identifier: Apache-2.0
"""``texture_box_geoms`` draws a textured box under ``texuniform`` and a textured cylinder.

Newton's MJCF importer binds a material's image to mesh geoms only; every
primitive arrives as its material's flat colour with no warning. The helper
redraws those as visual meshes, and it used to stop short twice: a material
stating ``texuniform="true"`` was skipped outright, and a ``cylinder`` was never
considered. Both cases are the tool-sort workbench -- steel plate on the top at
``texrepeat="2 4" texuniform="true"``, brushed steel on four cylinder legs
declared as a one-file ``type="cube"`` texture -- and on the syringe suite the
skipped top was a flat slate slab where the reported numbers had seen a textured
bench. No device: a builder, a three-geom MJCF and a PNG in a temp dir.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import newton
import numpy as np

from robosimstudio.scene.build.imports import texture_box_geoms

TIER = "fast"

MJCF = """<mujoco model="bench">
  <compiler angle="radian" texturedir="." />
  <asset>
    <texture name="tex-plate" type="2d" file="tex.png" />
    <texture name="tex-brushed" type="cube" file="tex.png" />
    <texture name="tex-sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0 0 0" width="8" height="8" />
    <material name="plate" texture="tex-plate" texrepeat="2 4" texuniform="true" />
    <material name="brushed" texture="tex-brushed" texrepeat="1 1" />
    <material name="paint" rgba="0.5 0.5 0.5 1" />
  </asset>
  <worldbody>
    <geom name="top" type="box" size="0.3 0.5 0.02" material="plate" />
    <geom name="leg" type="cylinder" size="0.02 0.37" material="brushed" />
    <geom name="plain" type="box" size="0.1 0.1 0.1" material="paint" />
  </worldbody>
</mujoco>
"""


def _write(tmp: str) -> str:
    from PIL import Image

    Image.fromarray(np.full((8, 8, 3), 200, dtype=np.uint8)).save(os.path.join(tmp, "tex.png"))
    path = os.path.join(tmp, "bench.xml")
    with open(path, "w") as fh:
        fh.write(MJCF)
    return path


class TestTexturedPrimitives(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.source = _write(cls.tmp.name)
        cls.builder = newton.ModelBuilder()
        cls.builder.add_mjcf(cls.source)
        cls.before = cls.builder.shape_count
        cls.added = texture_box_geoms(cls.builder, cls.source)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def _drawn(self) -> dict[str, int]:
        return {
            str(self.builder.shape_label[s]).rpartition("/")[2]: s for s in range(self.before, self.builder.shape_count)
        }

    def test_the_uniform_box_and_the_cylinder_are_drawn_and_the_plain_box_is_not(self):
        self.assertEqual(self.added, 2)
        self.assertEqual(sorted(self._drawn()), ["leg_textured", "top_textured"])

    def test_the_originals_are_hidden_not_removed(self):
        visible = int(newton.ShapeFlags.VISIBLE)
        for s in range(self.before):
            leaf = str(self.builder.shape_label[s]).rpartition("/")[2]
            with self.subTest(geom=leaf):
                self.assertEqual(bool(int(self.builder.shape_flags[s]) & visible), leaf == "plain")

    def test_texuniform_scales_the_repeat_by_each_faces_extent(self):
        mesh = self.builder.shape_source[self._drawn()["top_textured"]]
        self.assertTrue(str(mesh.texture).endswith("tex.png"))
        uvs = np.asarray(mesh.uvs, dtype=np.float64)
        # texrepeat "2 4" per metre over a 0.6 x 1.0 x 0.04 m box: the widest
        # face runs 2 * 1.0 = 2.0 repeats along u and 4 * 1.0 = 4.0 along v.
        self.assertAlmostEqual(float(uvs[:, 0].max()), 2.0, places=5)
        self.assertAlmostEqual(float(uvs[:, 1].max()), 4.0, places=5)
        self.assertEqual(len(mesh.vertices), 24)

    def test_the_cylinder_wraps_the_image_once_around_and_once_along(self):
        mesh = self.builder.shape_source[self._drawn()["leg_textured"]]
        self.assertTrue(str(mesh.texture).endswith("tex.png"))
        uvs = np.asarray(mesh.uvs, dtype=np.float64)
        self.assertAlmostEqual(float(uvs[:, 0].max()), 1.0, places=5)
        self.assertAlmostEqual(float(uvs[:, 1].max()), 1.0, places=5)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        radial = np.linalg.norm(verts[:, :2], axis=1)
        self.assertAlmostEqual(float(radial.max()), 0.02, places=5)
        self.assertAlmostEqual(float(np.abs(verts[:, 2]).max()), 0.37, places=5)


if __name__ == "__main__":
    unittest.main()
