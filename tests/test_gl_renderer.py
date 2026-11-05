# SPDX-License-Identifier: Apache-2.0
"""The gl renderer draws what the tiled tracer draws: the same shapes, where they are, through the same ``K``.

It exists for speed -- MuJoCo's rasteriser draws a 1280x720 frame of a
bimanual YAM in about half a millisecond, where the tiled tracer takes two, and
a scene stepped on a CPU has nothing faster -- so what is held here is that the
speed bought no difference in what is pictured. Segmentation is the measure:
it is a per-pixel answer to "which shape is here", free of lighting, which the
two backends do not share. A calibrated camera's principal point is held too,
since MuJoCo takes it as an offset the other way round from ``K``.

The scene's GPU is only needed for the context; a machine with neither an EGL
device nor OSMesa skips the module.
"""

from __future__ import annotations

import types
import unittest

import numpy as np

TIER = "medium"
"""Opens an OpenGL context on the scene's GPU. The rules that need none are ``tests/test_renderer_gl_rules``."""


def _scene():
    import robosimstudio as rss

    scene = rss.Scene(cell="bench_075", physics=rss.MUJOCO_500HZ)
    scene.add(rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.0, 0.1, 0.03)))
    scene.add(rss.robot("yam_bimanual"))
    return scene.build()


def _camera(name="c", *, dx=0.0, dy=0.0, **kwargs):
    import robosimstudio as rss

    w, h, f = 320, 180, 210.0
    return rss.CameraSpec(
        name=name, width=w, height=h, fx=f, fy=1.05 * f, cx=w / 2 - 0.5 + dx, cy=h / 2 - 0.5 + dy, **kwargs
    )


class TestACollapsedDeformable(unittest.TestCase):
    def test_a_sheet_and_a_rope_built_on_one_point_still_draw(self) -> None:
        """MuJoCo refuses a mesh with no area; a deformable's vertices are rewritten every frame, so gl builds."""
        import newton

        from robosimstudio.core.cameras import CameraSpec
        from robosimstudio.solvers.render.gl import RendererGL

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for _ in range(6):
            builder.add_particle(pos=(0.0, 0.0, 0.3), vel=(0.0, 0.0, 0.0), mass=0.01, radius=0.01)
        model = builder.finalize(device="cpu")

        def handle(particles, triangles=(), tube=0.0):
            return types.SimpleNamespace(particles=np.array(particles), triangles=np.array(triangles).reshape(-1, 3),
                                         tube_radius=tube, point_radius=0.0, surface=None)  # fmt: skip

        handles = {"sheet": handle([0, 1, 2], [[0, 1, 2]]), "rope": handle([3, 4, 5], tube=0.005)}
        cam = CameraSpec(name="c", width=32, height=24, fov_y=60.0, eye=(1, 1, 1), look_at=(0, 0, 0))
        try:
            gl = RendererGL(model, [cam], channels=("rgb", "depth"), deformables=handles)
        except RuntimeError as exc:
            self.skipTest(f"no OpenGL context here: {exc}")
        try:
            state = model.state()
            q = state.particle_q.numpy()
            q[:3] = [[0.0, 0.0, 0.3], [0.1, 0.0, 0.3], [0.0, 0.1, 0.3]]
            q[3:] = [[0.0, -0.1, 0.2], [0.05, -0.1, 0.2], [0.1, -0.1, 0.2]]
            state.particle_q.assign(q)
            frame = gl.render(state)["c"]
        finally:
            gl.close()
        self.assertTrue(np.isfinite(frame["depth"]).all())

    def test_a_rope_is_drawn_by_its_outside(self) -> None:
        """A tube's faces wound inward were culled as backs: gl drew the inside of its far wall, a hairline at the
        bench's depth that a plate composite then trimmed away. Seen from above, the depth is the tube's top."""
        import newton

        from robosimstudio.core.cameras import CameraSpec
        from robosimstudio.solvers.render.gl import RendererGL

        builder = newton.ModelBuilder()
        for i in range(8):
            builder.add_particle(pos=(0.02 * i, 0.0, 0.2), vel=(0.0, 0.0, 0.0), mass=0.01, radius=0.01)
        model = builder.finalize(device="cpu")
        rope = types.SimpleNamespace(particles=np.arange(8), triangles=np.zeros((0, 3), np.int64), tube_radius=0.01,
                                     point_radius=0.0, surface=None)  # fmt: skip
        cam = CameraSpec(name="c", width=64, height=48, fov_y=30.0, eye=(0.07, 0.0, 1.0), look_at=(0.07, 0.0, 0.0))
        try:
            gl = RendererGL(model, [cam], channels=("rgb", "depth"), deformables={"rope": rope})
        except RuntimeError as exc:
            self.skipTest(f"no OpenGL context here: {exc}")
        try:
            frame = gl.render(model.state())["c"]
        finally:
            gl.close()
        self.assertAlmostEqual(float(frame["depth"][24, 32]), 1.0 - 0.2 - 0.01, delta=0.002)
        wide = int(np.count_nonzero(frame["depth"][:, 32] > 0))
        self.assertGreaterEqual(wide, 2, "as wide as a 20 mm tube is at 0.8 m through 90 px of focal length")


class TestWhichGpuItDrawsOn(unittest.TestCase):
    """EGL lists every GPU whatever ``CUDA_VISIBLE_DEVICES`` says; left to MuJoCo, a context lands on GPU 0."""

    def test_each_visible_gpu_is_its_own_egl_device(self) -> None:
        import warp as wp

        from robosimstudio.solvers.render.gl import egl_device_for

        found = [egl_device_for(k) for k in range(wp.get_cuda_device_count())]
        if not found or None in found:
            self.skipTest(f"this driver does not say which EGL device is which CUDA device: {found}")
        self.assertEqual(len(set(found)), len(found))

    def test_a_cpu_scene_draws_on_the_first_gpu_its_process_sees(self) -> None:
        """A collection worker is pinned to a card by ``CUDA_VISIBLE_DEVICES`` alone, so this is what places it."""
        import os
        import subprocess
        import sys
        from pathlib import Path

        child = (
            "import os, newton, warp as wp\n"
            "from robosimstudio.core.cameras import CameraSpec\n"
            "from robosimstudio.solvers.render.gl import RendererGL, egl_device_for\n"
            "b = newton.ModelBuilder(); b.add_ground_plane(); m = b.finalize(device='cpu')\n"
            "cam = CameraSpec(name='c', width=32, height=24, fov_y=60.0, eye=(1, 1, 1), look_at=(0, 0, 0))\n"
            "RendererGL(m, [cam]).close()\n"
            "print(os.environ.get('MUJOCO_EGL_DEVICE_ID'), egl_device_for(0))\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "MUJOCO_EGL_DEVICE_ID"}
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run(
            [sys.executable, "-c", child], env=env, cwd=root, capture_output=True, text=True, timeout=300, check=False
        )
        if out.returncode != 0 and "could not open an OpenGL context" in out.stderr:
            self.skipTest("no OpenGL context here")
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        chosen, expected = out.stdout.strip().splitlines()[-1].split()
        if expected == "None":
            self.skipTest("this driver does not say which EGL device is which CUDA device")
        self.assertEqual(chosen, expected)

    def test_a_process_that_sees_no_gpu_is_refused_rather_than_drawn_on_the_first(self) -> None:
        """``CUDA_VISIBLE_DEVICES=""`` on a shared machine: EGL would otherwise draw on physical GPU 0."""
        import os
        import subprocess
        import sys
        from pathlib import Path

        from robosimstudio.solvers.render.gl import _egl_devices

        if not any(nvidia for nvidia, _ in _egl_devices()):
            self.skipTest("no NVIDIA GPU is an EGL device here")
        child = (
            "import newton\n"
            "from robosimstudio.core.cameras import CameraSpec\n"
            "from robosimstudio.solvers.render.gl import RendererGL\n"
            "b = newton.ModelBuilder(); b.add_ground_plane(); m = b.finalize(device='cpu')\n"
            "cam = CameraSpec(name='c', width=32, height=24, fov_y=60.0, eye=(1, 1, 1), look_at=(0, 0, 0))\n"
            "RendererGL(m, [cam]).close()\n"
        )
        env = {
            k: v for k, v in os.environ.items() if k not in ("MUJOCO_EGL_DEVICE_ID", "MUJOCO_GL", "PYOPENGL_PLATFORM")
        }
        env["CUDA_VISIBLE_DEVICES"] = ""
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run(
            [sys.executable, "-c", child], env=env, cwd=root, capture_output=True, text=True, timeout=300, check=False
        )
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("can see no GPU here", out.stderr)


class TestASceneWithNoBodies(unittest.TestCase):
    def test_a_bare_floor_draws(self) -> None:
        import newton

        from robosimstudio.core.cameras import CameraSpec
        from robosimstudio.solvers.render.gl import RendererGL

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        cam = CameraSpec(name="c", width=32, height=24, fov_y=60.0, eye=(1, 1, 1), look_at=(0, 0, 0))
        try:
            gl = RendererGL(model, [cam], channels=("rgb", "seg"))
        except RuntimeError as exc:
            self.skipTest(f"no OpenGL context here: {exc}")
        try:
            frame = gl.render(model.state())["c"]
        finally:
            gl.close()
        self.assertEqual(frame["rgb"].shape, (24, 32, 3))
        self.assertGreater(int((frame["seg"] > 0).sum()), 0, "the floor is not in the picture")


class TestItDrawsWhatTiledDraws(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _scene()
        mode = "joint_position"
        base = np.asarray(cls.sim.idle_action(mode), dtype=np.float32)
        for k in range(30):
            cls.sim.step(base + 0.3 * np.sin(0.1 * k + np.arange(len(base))), mode)
        try:
            cls.sim.make_renderer("gl", cameras=[_camera(eye=(1, 1, 1.5), look_at=(0, 0, 0.8))]).close()
        except RuntimeError as exc:
            cls.sim.close()
            raise unittest.SkipTest(f"no OpenGL context here: {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def _both(self, cameras, channels=("rgb", "depth", "seg")):
        tiled = self.sim.make_renderer("tiled", cameras=cameras, channels=channels).render(self.sim.state)
        gl = self.sim.make_renderer("gl", cameras=cameras, channels=channels)
        try:
            return tiled, gl.render(self.sim.state)
        finally:
            gl.close()

    def test_every_camera_pictures_the_same_shapes_in_the_same_places(self) -> None:
        cameras = [
            _camera("side", eye=(0.45, -0.75, 1.2), look_at=(0.0, 0.0, 0.8)),
            _camera("top", eye=(0.0, -0.25, 1.75), look_at=(0.0, 0.0, 0.75)),
            _camera("wrist", parent_frame="left", xform=(0, 0, 0.05, 0, 0, 0, 1)),
        ]
        tiled, gl = self._both(cameras)
        for cam in cameras:
            a, b = tiled[cam.name], gl[cam.name]
            self.assertEqual(b["rgb"].shape, (180, 320, 3))
            self.assertGreater(len(np.unique(b["rgb"].reshape(-1, 3), axis=0)), 50, f"{cam.name}: drew nothing")
            self.assertGreater((a["seg"] == b["seg"]).mean(), 0.97, cam.name)
            both = (a["depth"] > 0) & (b["depth"] > 0)
            self.assertLess(float(np.median(np.abs(a["depth"][both] - b["depth"][both]))), 0.01, cam.name)

    def test_a_calibrated_principal_point_is_where_k_puts_it(self) -> None:
        """MuJoCo moves the picture the other way from ``cx``: with K's sign it lands two offsets away."""
        for dx, dy in ((40.0, 0.0), (0.0, 20.0), (-30.0, -15.0)):
            cam = _camera(dx=dx, dy=dy, eye=(0.45, -0.75, 1.2), look_at=(0.0, 0.0, 0.8))
            tiled, gl = self._both([cam], channels=("seg",))
            self.assertGreater((tiled["c"]["seg"] == gl["c"]["seg"]).mean(), 0.99, (dx, dy))

    def test_each_light_is_as_bright_as_its_stated_intensity(self) -> None:
        """The key and the fills on one scale, as the spec's rig puts them."""
        cam = _camera(eye=(0.45, -0.75, 1.2), look_at=(0.0, 0.0, 0.8))
        rig = (((1.0, 0.0, -0.2), 0.5), ((0.0, 1.0, -0.2), 0.25))
        gl = self.sim.make_renderer("gl", cameras=[cam], channels=("rgb",), key_intensity=2.0, fill_lights=rig)
        try:
            diffuse = gl.mj_model.light_diffuse[:, 0]
        finally:
            gl.close()
        np.testing.assert_allclose(diffuse / diffuse[0], [1.0, 0.25, 0.125], rtol=1e-6)

    def test_a_picture_body_takes_no_inertia_from_its_meshes(self) -> None:
        """A flat mesh (a photo on a slab) has A + B = C to the rounding, which MuJoCo refuses when it infers."""
        cam = _camera(eye=(0.45, -0.75, 1.2), look_at=(0.0, 0.0, 0.8))
        gl = self.sim.make_renderer("gl", cameras=[cam], channels=("rgb",))
        try:
            ids = [gl.mj_model.body(f"body{b}").id for b in gl._bodies]
            np.testing.assert_allclose(gl.mj_model.body_inertia[ids], 1.0)
        finally:
            gl.close()

    def test_a_kept_frame_survives_the_next_render(self) -> None:
        cam = _camera(eye=(0.45, -0.75, 1.2), look_at=(0.0, 0.0, 0.8))
        gl = self.sim.make_renderer("gl", cameras=[cam], channels=("rgb",))
        try:
            kept = gl.render(self.sim.state)["c"]["rgb"]
            copy = kept.copy()
            gl.render(self.sim.state)
            np.testing.assert_array_equal(kept, copy)
        finally:
            gl.close()


class TestDeformables(unittest.TestCase):
    """A cloth is drawn where its particles are, moved every frame, as the tiled tracer draws it."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss
        from robosimstudio_materials.cloth import sheet

        scene = rss.Scene(cell="bench_075")
        scene.add(rss.robot("yam_bimanual"))
        scene.add(sheet(name="towel", at=(-0.15, -0.2, 0.08), size=(0.12, 0.12), cells=(8, 8), color=(0.8, 0.2, 0.2)))
        cls.sim = scene.build()
        cls.camera = _camera("top", eye=(0.51, -0.14, 1.3), look_at=(0.51, -0.13, 0.75))  # world: the bench is at x 0.6
        try:
            cls.sim.make_renderer("gl", cameras=[cls.camera]).close()
        except RuntimeError as exc:
            cls.sim.close()
            raise unittest.SkipTest(f"no OpenGL context here: {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def _both(self):
        channels = ("rgb", "depth", "seg")
        tiled = self.sim.make_renderer("tiled", cameras=[self.camera], channels=channels).render(self.sim.state)
        gl = self.sim.make_renderer("gl", cameras=[self.camera], channels=channels)
        try:
            return tiled["top"], gl.render(self.sim.state)["top"]
        finally:
            gl.close()

    def test_the_cloth_falls_in_the_picture_as_it_does_in_the_tracers(self) -> None:
        from robosimstudio.solvers.render.tiled import SEG_SOFT_MESH

        for _ in range(2):  # where it was built, and after it fell onto the bench
            tiled, gl = self._both()
            cloth = tiled["seg"] == SEG_SOFT_MESH
            self.assertGreater(int(cloth.sum()), 200, "the towel is in the picture")
            self.assertGreater(((gl["seg"] == SEG_SOFT_MESH) == cloth).mean(), 0.99)
            self.assertLess(float(np.median(np.abs(tiled["depth"][cloth] - gl["depth"][cloth]))), 0.002)
            red = gl["rgb"][cloth].astype(np.float64).mean(axis=0)
            self.assertGreater(red[0], 1.5 * red[2], "in its own colour")
            self.sim.run(60)

    def test_a_particle_body_is_spheres_labelled_as_the_tracers_label_them(self) -> None:
        from robosimstudio.solvers.render.tiled import SEG_PARTICLES, SEG_SOFT_MESH

        towel = self.sim.scene.handles.deformables["towel"]
        grains = types.SimpleNamespace(particles=towel.particles, triangles=np.zeros((0, 3)), tube_radius=0.0,
                                       point_radius=0.006, surface=None)  # fmt: skip
        gl = self.sim.make_renderer("gl", cameras=[self.camera], channels=("seg",), deformables={"towel": grains})
        try:
            seg = gl.render(self.sim.state)["top"]["seg"]
        finally:
            gl.close()
        self.assertGreater(int((seg == SEG_PARTICLES).sum()), 50, "drawn as its particles")
        self.assertEqual(int((seg == SEG_SOFT_MESH).sum()), 0, "and not as the model's triangles over them")


if __name__ == "__main__":
    unittest.main()
