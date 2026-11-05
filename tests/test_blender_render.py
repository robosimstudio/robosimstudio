# SPDX-License-Identifier: Apache-2.0
"""The blender renderer draws what the tiled tracer draws, through the same ``K``, lit as the path tracer lights it.

Through a real Blender (``ROBOSIMSTUDIO_BLENDER``, ``blender`` on the
``PATH``, or the Mac app); the module skips where there is none. Depth is
the measure of geometry: it is free of lighting, which the two renderers do
not share, and a camera over a clean plate composites by it. A calibrated
camera -- the principal point off centre, ``fx != fy`` -- is held to the pixel,
a cloth is drawn where its particles are, and a picture follows the state.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "medium"


def _camera(name="c", *, dx=0.0, dy=0.0, fy_ratio=1.0, **kwargs):
    import robosimstudio as rss

    w, h, f = (180, 320, 210.0) if name == "portrait" else (320, 180, 210.0)
    return rss.CameraSpec(
        name=name, width=w, height=h, fx=f, fy=fy_ratio * f, cx=w / 2 - 0.5 + dx, cy=h / 2 - 0.5 + dy, **kwargs
    )


class TestItDrawsWhatTiledDraws(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from dataclasses import replace

        import robosimstudio as rss
        from robosimstudio.core.render.room import LAB
        from robosimstudio.solvers.render.blender import find_blender
        from robosimstudio_materials.cloth import sheet

        try:
            find_blender()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        scene = rss.Scene(cell="bench_075", render=replace(LAB, scenery=""))
        scene.add(rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.0, 0.1, 0.03), color=(0.2, 0.4, 0.9)))
        scene.add(sheet(name="towel", at=(-0.15, -0.2, 0.05), size=(0.12, 0.12), cells=(8, 8), color=(0.8, 0.2, 0.2)))
        scene.add(rss.robot("yam_bimanual"))
        cls.sim = scene.build()
        cls.sim.run(20)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def _both(self, cameras):
        tiled = self.sim.make_renderer("tiled", cameras=cameras, channels=("rgb", "depth", "seg")).render(
            self.sim.state
        )
        blender = self.sim.make_renderer("blender", cameras=cameras, channels=("rgb", "depth"))
        try:
            return tiled, blender.render(self.sim.state)
        finally:
            blender.close()

    def test_a_calibrated_camera_sees_the_same_surfaces_at_the_same_depths(self) -> None:
        cameras = [
            _camera("side", eye=(0.45, -0.75, 1.2), look_at=(0.0, 0.0, 0.8)),
            _camera("offset", dx=40.0, dy=-20.0, fy_ratio=1.05, eye=(0.2, -0.6, 1.3), look_at=(0.0, -0.1, 0.78)),
            _camera("portrait", dx=30.0, dy=40.0, fy_ratio=0.95, eye=(0.45, -0.75, 1.2), look_at=(0.6, 0.0, 0.8)),
        ]
        tiled, blender = self._both(cameras)
        for cam in cameras:
            a, b = tiled[cam.name], blender[cam.name]
            self.assertEqual(b["rgb"].shape, (cam.height, cam.width, 3))
            self.assertEqual(b["rgb"].dtype, np.uint8)
            # the tiled tracer's depth runs past the camera's far; every other backend's is a miss there
            hit_a, hit_b = (a["depth"] > 0) & (a["depth"] <= cam.far), b["depth"] > 0
            self.assertGreater((hit_a == hit_b).mean(), 0.98, cam.name)
            both = hit_a & hit_b
            self.assertLess(float(np.median(np.abs(a["depth"][both] - b["depth"][both]))), 0.001, cam.name)
            self.assertGreater(float(np.percentile(b["rgb"], 90)), 40.0, f"{cam.name}: lit")

    def test_the_cloth_is_drawn_where_its_particles_are(self) -> None:
        from robosimstudio.solvers.render.tiled import SEG_SOFT_MESH

        cam = _camera("top", eye=(0.51, -0.14, 1.3), look_at=(0.51, -0.13, 0.75))  # world: the bench is at x 0.6
        tiled, blender = self._both([cam])
        cloth = tiled["top"]["seg"] == SEG_SOFT_MESH
        self.assertGreater(int(cloth.sum()), 200, "the towel is in the picture")
        a, b = tiled["top"]["depth"][cloth], blender["top"]["depth"][cloth]
        self.assertLess(float(np.median(np.abs(a - b))), 0.002)
        red = blender["top"]["rgb"][cloth].astype(np.float64).mean(axis=0)
        self.assertGreater(red[0], 1.5 * red[2], "in its own colour")

    def test_grains_and_a_regrown_surface_are_drawn_where_the_particles_are(self) -> None:
        import types

        towel = self.sim.scene.handles.deformables["towel"]
        cam = _camera("top", eye=(0.51, -0.14, 1.3), look_at=(0.51, -0.13, 0.75))
        tiled = self.sim.make_renderer("tiled", cameras=[cam], channels=("depth",)).render(self.sim.state)["top"]
        cloth = tiled["depth"] > 0

        def surface(positions):  # the sheet's own triangles, as a particle body's surface callable returns them
            return positions, towel.triangles

        for drawn in ("grains", "surface"):
            body = types.SimpleNamespace(
                particles=towel.particles, triangles=np.zeros((0, 3)), tube_radius=0.0, point_radius=0.004,
                surface=surface if drawn == "surface" else None,
            )  # fmt: skip
            blender = self.sim.make_renderer("blender", cameras=[cam], channels=("depth",), deformables={"towel": body})
            try:
                depth = blender.render(self.sim.state)["top"]["depth"]
            finally:
                blender.close()
            near = (depth > 0) & (depth < tiled["depth"].max() - 0.01)
            self.assertGreater(int(near.sum()), 200, f"the towel as {drawn}")
            self.assertLess(float(np.median(np.abs(depth[near & cloth] - tiled["depth"][near & cloth]))), 0.006, drawn)

    def test_a_still_keeps_one_blender_across_cameras(self) -> None:
        import os
        from unittest import mock

        side = _camera("side", eye=(0.45, -0.75, 1.2), look_at=(0.0, 0.0, 0.8))
        top = _camera("top", eye=(0.51, -0.14, 1.3), look_at=(0.51, -0.13, 0.75))
        with mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_RENDERER": "blender"}):
            first = self.sim.render(side, channels=("depth",))["depth"]
            renderer = self.sim._still_renderer
            second = self.sim.render(top, channels=("depth",))["depth"]
            self.assertIs(self.sim._still_renderer, renderer, "re-aimed, not started again")
            self.assertEqual(renderer.cameras, [top])
            self.assertFalse(np.allclose(first, second))
            # a camera over a plate asks depth beside rgb and another does not: still the one Blender
            picture = self.sim.render(side, channels=("rgb",))
            self.assertIs(self.sim._still_renderer, renderer)
            self.assertEqual(sorted(picture), ["rgb"])
            self.assertEqual(renderer.channels, {"side": ("rgb",)})

    def test_a_picture_follows_the_state(self) -> None:
        cam = _camera("side", eye=(0.45, -0.75, 1.2), look_at=(0.0, 0.0, 0.8))
        blender = self.sim.make_renderer("blender", cameras=[cam], channels=("depth",))
        try:
            frame = blender.render(self.sim.state)["side"]
            self.assertEqual(sorted(frame), ["depth"], "only the channel asked for")
            before = frame["depth"]
            mode = "joint_position"
            base = np.asarray(self.sim.idle_action(mode), dtype=np.float32)
            for k in range(40):
                self.sim.step(base + 0.4 * np.sin(0.1 * k + np.arange(len(base))), mode)
            after = blender.render(self.sim.state)["side"]["depth"]
        finally:
            blender.close()
        self.assertGreater(float((np.abs(after - before) > 0.005).mean()), 0.005, "the arms moved in the picture")

    def test_a_closed_renderer_says_so(self) -> None:
        cam = _camera("side", eye=(0.45, -0.75, 1.2), look_at=(0.0, 0.0, 0.8))
        blender = self.sim.make_renderer("blender", cameras=[cam], channels=("rgb",))
        blender.close()
        blender.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            blender.render(self.sim.state)


if __name__ == "__main__":
    unittest.main()
