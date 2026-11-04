# SPDX-License-Identifier: Apache-2.0
"""``play --record TAKE --also R,R``: one run filmed by the page's renderer and drawn by each other renderer named.

Needs the GPU: it builds a scene and draws it with gl and the tiled tracer.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

TIER = "medium"


def _frames(path: Path) -> int:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        return sum(1 for _ in reader)
    finally:
        reader.close()


class AlsoRenderers(unittest.TestCase):
    def test_one_run_is_drawn_by_each_renderer(self) -> None:
        import robosimstudio as rss
        from robosimstudio_real2sim_agent import checks

        scene = rss.Scene(cell="bench_075", name="take_renderers")
        scene.add(rss.box(name="cube", half=(0.02,) * 3, at=(0.0, 0.0, 0.2)))  # falls onto the bench as it is filmed
        with mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_RENDERER": "gl"}), redirect_stdout(io.StringIO()):
            sim = scene.build()
            with tempfile.TemporaryDirectory() as tmp:
                server = types.SimpleNamespace(sim=sim, scene=scene, run_dir=tmp)
                rec = checks.Recorder(server, "fall", every=2, also="tiled,gl")  # gl is the page's own: filmed once
                self.assertEqual(list(rec.also), ["tiled"])
                rec.probe()
                rec.start()
                for _ in range(10):
                    sim.run(1)
                    rec.maybe()
                out = rec.finish()
                meta = json.loads((Path(tmp) / "fall.json").read_text())
                take, tiled, grid = Path(out["video"]), Path(out["also"]["tiled"]), Path(out["renderers"])
                n = len(meta["frames"])
                self.assertGreater(n, 3)
                self.assertEqual((_frames(take), _frames(tiled), _frames(grid)), (n, n, n), "one frame each per state")
                import imageio.v2 as imageio

                first = imageio.get_reader(str(grid), "ffmpeg").get_data(0)
                self.assertEqual(first.shape[1], 2 * int(rec.cam.width), "the two renderers side by side")
                self.assertTrue(np.isfinite(first).all())

    def test_a_renderer_that_dies_mid_take_leaves_the_take_and_a_black_tile(self) -> None:
        import robosimstudio as rss
        from robosimstudio_real2sim_agent import checks

        scene = rss.Scene(cell="bench_075", name="take_renderers_dies")
        scene.add(rss.box(name="cube", half=(0.02,) * 3, at=(0.0, 0.0, 0.2)))
        with mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_RENDERER": "gl"}), redirect_stdout(io.StringIO()):
            sim = scene.build()
            with tempfile.TemporaryDirectory() as tmp:
                server = types.SimpleNamespace(sim=sim, scene=scene, run_dir=tmp)
                rec = checks.Recorder(server, "dies", every=2, also="tiled")
                rec.probe()
                rec.start()
                real = rec.also["tiled"]["renderer"]
                calls = {"n": 0}

                def flaky(state):
                    calls["n"] += 1
                    if calls["n"] > 2:
                        raise RuntimeError("the card went away")
                    return real.render(state)

                rec.also["tiled"]["renderer"] = types.SimpleNamespace(render=flaky, close=real.close)
                for _ in range(10):
                    sim.run(1)
                    rec.maybe()
                out = rec.finish()
                self.assertIn("tiled", out["also_failed"])
                n = len(json.loads((Path(tmp) / "dies.json").read_text())["frames"])
                self.assertEqual((_frames(Path(out["video"])), _frames(Path(out["renderers"]))), (n, n))
                self.assertLess(_frames(Path(out["also"]["tiled"])), n, "its video, ended where it stopped")

    def test_over_a_plate_each_renderer_is_composited_on_one_pass(self) -> None:
        from PIL import Image

        import robosimstudio as rss
        from robosimstudio_real2sim_agent import checks

        with tempfile.TemporaryDirectory() as tmp:
            Image.new("RGB", (320, 180), (20, 60, 230)).save(Path(tmp) / "plate.png")
            scene = rss.Scene(cell="bench_075", name="take_renderers_plate")
            scene.add(rss.box(name="cube", half=(0.05,) * 3, at=(0.0, 0.0, 0.05), color=(0.9, 0.15, 0.1)))
            scene.add(
                rss.CameraSpec(
                    name="front",
                    width=320,
                    height=180,
                    fov_y=40.0,
                    eye=(1.5, 0.0, 1.1),
                    look_at=(0.6, 0.0, 0.78),
                    background=str(Path(tmp) / "plate.png"),
                )
            )
            old = Path(tmp) / "plated.renderers" / "blender.mp4"
            old.parent.mkdir()
            old.write_bytes(b"an earlier take's, left by a crash")
            (Path(tmp) / "plated.gl.mp4").write_bytes(b"the take named plated.gl")  # its own video, not a side one
            with mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_RENDERER": "gl"}), redirect_stdout(io.StringIO()):
                sim = scene.build()
                server = types.SimpleNamespace(sim=sim, scene=scene, run_dir=tmp)
                rec = checks.Recorder(server, "plated", camera="front", every=2, also="tiled")
                rec.probe()
                self.assertTrue(old.exists(), "kept until the take begins")
                rec.start()
                self.assertFalse(old.exists(), "an earlier take of this name's folder is replaced")
                self.assertTrue((Path(tmp) / "plated.gl.mp4").exists(), "another take's own video stays")
                passes = []
                real = sim.plate_pass
                sim.plate_pass = lambda spec: passes.append(1) or real(spec)
                for _ in range(6):
                    sim.run(1)
                    rec.maybe()
                out = rec.finish()
                import imageio.v2 as imageio

                n = len(json.loads((Path(tmp) / "plated.json").read_text())["frames"])
                self.assertEqual(len(passes), n - 1, "one plate pass a frame filmed after start, for every renderer")
                last = imageio.get_reader(out["also"]["tiled"], "ffmpeg").get_data(n - 1).astype(int)
                corner = last[5, 5]
                self.assertLess(abs(corner[2] - 230) + abs(corner[0] - 20), 40, "the tiled tile is over the plate")
                u, v = rec.cam.project(np.array([[0.6, 0.0, 0.8]]))[0][0]
                self.assertGreater(last[int(v), int(u), 0], 120, "and the cube is the render's")

    def test_a_frame_missed_is_black_and_the_page_renderer_is_left_out(self) -> None:
        import robosimstudio as rss
        from robosimstudio_real2sim_agent import checks

        scene = rss.Scene(cell="bench_075", name="take_renderers_miss")
        scene.add(rss.box(name="cube", half=(0.02,) * 3, at=(0.0, 0.0, 0.2)))
        with mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_RENDERER": "gl"}), redirect_stdout(io.StringIO()):
            sim = scene.build()
            with tempfile.TemporaryDirectory() as tmp:
                server = types.SimpleNamespace(sim=sim, scene=scene, run_dir=tmp)
                own = checks.Recorder(server, "own", every=2, also="gl")
                self.assertIsNone(own.grid)
                own.probe()
                own.start()
                self.assertEqual(own.finish()["also_skipped"], ["gl"], "the page's own renderer, said")
                rec = checks.Recorder(server, "miss", every=2, also="tiled")
                rec.probe()
                rec.start()
                real = rec.also["tiled"]["renderer"]
                calls = {"n": 0}

                def once(state):
                    calls["n"] += 1
                    if calls["n"] == 2:
                        raise RuntimeError("one frame lost")
                    return real.render(state)

                rec.also["tiled"]["renderer"] = types.SimpleNamespace(render=once, close=real.close)
                for _ in range(8):
                    sim.run(1)
                    rec.maybe()
                out = rec.finish()
                self.assertEqual(out["also_missed"], {"tiled": 1})
                self.assertIn("tiled", out["also"], "a renderer that missed one frame carries on")
                n = len(json.loads((Path(tmp) / "miss.json").read_text())["frames"])
                self.assertEqual(_frames(Path(out["also"]["tiled"])), n)

    def test_an_unknown_renderer_is_refused_before_filming(self) -> None:
        import robosimstudio as rss
        from robosimstudio_real2sim_agent import checks

        scene = rss.Scene(cell="bench_075", name="take_renderers_refused")
        scene.add(rss.box(name="cube", half=(0.02,) * 3, at=(0.0, 0.0, 0.03)))
        with mock.patch.dict(os.environ, {"ROBOSIMSTUDIO_RENDERER": "gl"}), redirect_stdout(io.StringIO()):
            sim = scene.build()
            with tempfile.TemporaryDirectory() as tmp:
                server = types.SimpleNamespace(sim=sim, scene=scene, run_dir=tmp)
                with self.assertRaises(ValueError):
                    checks.Recorder(server, "nope", also="povray")
                self.assertFalse(any(Path(tmp).iterdir()), "nothing written for a take refused")


if __name__ == "__main__":
    unittest.main()
