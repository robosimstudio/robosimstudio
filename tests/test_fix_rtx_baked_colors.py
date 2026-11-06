# SPDX-License-Identifier: Apache-2.0
"""An untextured shape's colour goes into its material, not through a primvar the renderer can lose.

Over a real USD stage and a stand-in for the viewer, because what matters is what
is authored: a live shape still reads its primvar, a static one holds a constant,
and shapes that agree about their colour and finish still share one material.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robosimstudio.core.render.finish import Finish
from robosimstudio.solvers.render import rtx

TIER = "fast"

_HAS_PXR = importlib.util.find_spec("pxr") is not None


def _stage_with(colors: list[list[tuple[float, float, float]]]):
    """A stage of one mesh per entry, each with ``displayColor`` as a time sample, as ViewerUSD authors it."""
    from pxr import Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateInMemory()
    batches = []
    for index, values in enumerate(colors):
        path = f"/root/shape_{index}/instance_0"
        UsdGeom.Xform.Define(stage, f"/root/shape_{index}")
        mesh = UsdGeom.Mesh.Define(stage, path)
        primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant, len(values)
        )
        primvar.Set(Vt.Vec3fArray(values), 0.0)
        batches.append(SimpleNamespace(model_shapes=[index], name=f"shape_{index}", mesh=f"m{index}", geo_type=-1))
    return stage, batches


def _viewer(stage, batches):
    return SimpleNamespace(stage=stage, _shape_to_batch=batches, _get_path=lambda name: f"/root/{name}", _meshes={})


def _material_of(stage, path: str):
    from pxr import UsdShade

    return UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).GetDirectBinding().GetMaterial()


def _diffuse(stage, material):
    from pxr import UsdShade

    surface = UsdShade.Shader(stage.GetPrimAtPath(f"{material.GetPath()}/PreviewSurface"))
    return surface.GetInput("diffuseColor")


@unittest.skipUnless(_HAS_PXR, "needs USD")
class TestBakedColors(unittest.TestCase):
    def _decorate(self, colors, *, finishes=None, live=()):
        stage, batches = _stage_with(colors)
        model = SimpleNamespace(shape_color=np.asarray([c[0] for c in colors], dtype=np.float32))
        rtx.decorate_stage(
            _viewer(stage, batches), model=model, finishes=finishes or {}, deformable_finishes={}, live_shapes=live
        )
        return stage

    def test_a_static_shape_holds_a_constant_with_no_primvar_reader(self) -> None:
        stage = self._decorate([[(0.5, 0.25, 0.75)]])
        material = _material_of(stage, "/root/shape_0/instance_0")
        diffuse = _diffuse(stage, material)
        self.assertFalse(diffuse.HasConnectedSource())
        # What the primvar reader would have read: the decoded albedo, to the digit.
        expected = rtx.srgb_to_linear((0.5, 0.25, 0.75)) if rtx.PALETTE_IS_DISPLAY else (0.5, 0.25, 0.75)
        np.testing.assert_allclose(tuple(diffuse.Get()), expected, atol=1.0e-5)
        self.assertFalse(stage.GetPrimAtPath(f"{material.GetPath()}/PrimvarColor"))

    def test_a_live_shape_still_reads_its_primvar(self) -> None:
        stage = self._decorate([[(0.5, 0.5, 0.5)], [(0.5, 0.5, 0.5)]], live=(1,))
        self.assertFalse(_diffuse(stage, _material_of(stage, "/root/shape_0/instance_0")).HasConnectedSource())
        self.assertTrue(_diffuse(stage, _material_of(stage, "/root/shape_1/instance_0")).HasConnectedSource())
        self.assertIn(1, _viewer_live(stage))

    def test_one_finish_and_colour_is_one_material(self) -> None:
        stage = self._decorate([[(0.5, 0.5, 0.5)], [(0.5, 0.5, 0.5)], [(0.1, 0.5, 0.5)]])
        paths = [str(_material_of(stage, f"/root/shape_{i}/instance_0").GetPath()) for i in range(3)]
        self.assertEqual(paths[0], paths[1])
        self.assertNotEqual(paths[0], paths[2])

    def test_a_primvar_of_several_values_keeps_its_reader(self) -> None:
        stage = self._decorate([[(0.5, 0.5, 0.5), (0.1, 0.1, 0.1)]])
        self.assertTrue(_diffuse(stage, _material_of(stage, "/root/shape_0/instance_0")).HasConnectedSource())

    def test_a_textured_finish_is_untouched(self) -> None:
        finish = Finish(texture="room_plaster")
        with mock.patch.object(rtx, "texture_path", lambda name: f"/textures/{name}.png"):
            stage = self._decorate([[(0.5, 0.5, 0.5)]], finishes={0: finish})
        diffuse = _diffuse(stage, _material_of(stage, "/root/shape_0/instance_0"))
        self.assertTrue(diffuse.HasConnectedSource())
        self.assertEqual(diffuse.GetConnectedSources()[0][0].source.GetPath().name, "DiffuseTexture")

    def test_the_environment_turns_it_off(self) -> None:
        with mock.patch.dict(os.environ, {rtx.BAKE_COLORS_ENV: "0"}):
            stage = self._decorate([[(0.5, 0.5, 0.5)], [(0.1, 0.1, 0.1)]])
        paths = [str(_material_of(stage, f"/root/shape_{i}/instance_0").GetPath()) for i in range(2)]
        self.assertEqual(paths[0], paths[1])
        self.assertTrue(_diffuse(stage, _material_of(stage, "/root/shape_0/instance_0")).HasConnectedSource())


def _viewer_live(stage) -> set[int]:
    """Which shapes a decorate of this stage left reading their primvar, by index."""
    out = set()
    for prim in stage.Traverse():
        name = prim.GetName()
        if name == "instance_0":
            if _diffuse(stage, _material_of(stage, str(prim.GetPath()))).HasConnectedSource():
                out.add(int(prim.GetParent().GetName().split("_")[1]))
    return out


if __name__ == "__main__":
    unittest.main()
