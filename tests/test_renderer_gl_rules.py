# SPDX-License-Identifier: Apache-2.0
"""The gl renderer's rules that hold before any OpenGL context is opened.

What it refuses (a fleet), which OpenGL platform it asks MuJoCo for on each
operating system, how a heap of particles is thinned to the sphere budget, and
the colour a textured mesh is drawn at. ``tests/test_gl_renderer`` draws, on a
GPU, against the tiled tracer.
"""

from __future__ import annotations

import types
import unittest

import numpy as np

TIER = "fast"
"""No context is opened: a refusal raised before one, the platform rule, and pure arithmetic."""


def _camera(name="c", *, dx=0.0, dy=0.0, **kwargs):
    import robosimstudio as rss

    w, h, f = 320, 180, 210.0
    return rss.CameraSpec(
        name=name, width=w, height=h, fx=f, fy=1.05 * f, cx=w / 2 - 0.5 + dx, cy=h / 2 - 0.5 + dy, **kwargs
    )


class TestWhatIsRefused(unittest.TestCase):
    """Before any context is opened, so it runs wherever the module does."""

    def test_a_fleet_is_refused(self) -> None:
        from robosimstudio.solvers.render.gl import RendererGL

        model = types.SimpleNamespace(particle_count=0, world_count=4)
        with self.assertRaisesRegex(ValueError, "one world"):
            RendererGL(model, [_camera(eye=(1, 1, 1), look_at=(0, 0, 0))])


class TestWhichOpenGL(unittest.TestCase):
    def test_a_mac_opens_cgl_and_linux_egl_unless_a_variable_names_one(self) -> None:
        import os
        from unittest import mock

        from robosimstudio.solvers.render import gl

        clear = {"MUJOCO_GL": "", "PYOPENGL_PLATFORM": ""}
        with mock.patch.dict(os.environ, clear), mock.patch.object(gl.sys, "platform", "darwin"):
            self.assertEqual(gl._platform(), "cgl", "a Mac has no EGL")
        with mock.patch.dict(os.environ, clear), mock.patch.object(gl.sys, "platform", "linux"):
            self.assertEqual(gl._platform(), "egl")
        with (
            mock.patch.dict(os.environ, {**clear, "MUJOCO_GL": "glfw"}),
            mock.patch.object(gl.sys, "platform", "darwin"),
        ):
            self.assertEqual(gl._platform(), "glfw")
        with (
            mock.patch.dict(os.environ, {**clear, "MUJOCO_GL": "osmesa"}),
            mock.patch.object(gl.sys, "platform", "linux"),
        ):
            self.assertEqual(gl._platform(), "osmesa")
        with (
            mock.patch.dict(os.environ, {**clear, "MUJOCO_GL": "glfw"}),
            mock.patch.object(gl.sys, "platform", "linux"),
        ):
            self.assertEqual(gl._platform(), "egl", "a headless server exporting glfw for MuJoCo's viewer keeps EGL")


class TestGrainBudget(unittest.TestCase):
    def test_only_a_heap_past_the_budget_is_thinned(self) -> None:
        from robosimstudio.solvers.render.gl import sphere_strides

        self.assertEqual(sphere_strides([10, 2000], budget=5000), [1, 1])
        self.assertEqual(sphere_strides([12, 200_000], budget=50_000), [1, 5], "the dozen loose particles whole")
        strides = sphere_strides([100_000, 60_000, 30], budget=50_000)
        self.assertEqual(strides[2], 1)
        self.assertLessEqual(100_000 // strides[0] + 60_000 // strides[1] + 30, 50_000)
        many = [900] * 80  # small bodies past the budget by themselves: thinned all alike
        strides = sphere_strides(many, budget=50_000)
        self.assertLessEqual(sum(-(-n // k) for n, k in zip(many, strides, strict=True)), 50_000 + len(many))


class TestColours(unittest.TestCase):
    def test_a_texture_is_drawn_at_its_own_brightness(self) -> None:
        """MuJoCo multiplies a texture by its geom's colour; trimesh's default for a texture with no factor is 40 % grey."""
        import trimesh
        from PIL import Image

        from robosimstudio.core.render.visual import mesh_rgba

        def textured(**factor):
            mesh = trimesh.creation.box()
            material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (4, 4), "white"), **factor)
            mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)), material=material)
            return mesh

        rgba, image, _ = mesh_rgba(textured())
        self.assertIsNotNone(image)
        self.assertEqual(rgba, [1.0, 1.0, 1.0, 1.0])
        rgba, _, _ = mesh_rgba(textured(baseColorFactor=[255, 0, 0, 255]))
        self.assertEqual(rgba, [1.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
