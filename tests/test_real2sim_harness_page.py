# SPDX-License-Identifier: Apache-2.0
"""The live page's fixes on a GPU: a routine written with the studio's helpers moves the arm, and a towel is named.

:mod:`tests.test_real2sim_harness_fixes` holds them with no page. Here they are
held on one: a routine built from :func:`robosimstudio_studio.routine.reach`
yields without stepping, so the page has to step it -- before, the page held
its own stepping for every routine and the hand "stopped 278 mm short" at the
start -- even with Pause ticked; and a towel in view, which the ray tracer
reports only as "a deformable's triangles", comes out of the segmentation under
its own name rather than raising.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

TIER = "medium"
"""Builds a YAM with a towel and starts a viser server."""

PORT = 9361

SCENE = """import robosimstudio as rss
from robosimstudio_studio.routine import hand, reach

scene = rss.Scene(cell="bench_075", name="live-reach")
scene.add(rss.robot("yam_bimanual"))
scene.add(rss.sheet(name="towel", size=(0.2, 0.2), cells=(10, 10), at=(-0.1, -0.1, 0.02)))
scene.add(rss.CameraSpec(name="top", eye=(0.6, 0.0, 1.6), look_at=(0.6, 0.0, 0.75), width=320, height=240))
LIFT = 0.05


def routine(sim, session):
    start, _ = hand(session, "right")
    yield from reach(session, "right", start + [0.0, 0.0, LIFT])
"""


class TestTheFixesOnAPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio_real2sim_agent.server import LiveServer
        from robosimstudio_studio.session import StudioConfig

        cls.tmp = Path(tempfile.mkdtemp(prefix="live-fix-"))
        (cls.tmp / "scene.py").write_text(SCENE)
        config = StudioConfig(port=PORT, panels="all", render_every=1, live=False, out=str(cls.tmp / "studio"))
        cls.live = LiveServer(
            scene_path=cls.tmp / "scene.py",
            sock_path=cls.tmp / "ctl.sock",
            run_dir=cls.tmp / "run",
            config=config,
            watch=False,
        )
        cls.live.open_page()
        assert not cls.live.build_error, cls.live.build_error

    @classmethod
    def tearDownClass(cls) -> None:
        cls.live.bus.close()
        cls.live.session.close()

    def setUp(self) -> None:
        self.live.reload("next test")

    def test_a_reach_routine_moves_the_hand_even_on_a_paused_page(self) -> None:
        from robosimstudio_studio.routine import hand

        self.live.session.set_paused(True)
        before, _ = hand(self.live.session, "right")
        reply = self.live.cmd_play("routine")
        self.assertTrue(reply["ok"], reply)
        for _ in range(2000):
            self.live.tick_once()
            if not self.live.script:
                break
        ended = self.live.last_script
        self.assertEqual(ended["serial"], reply["serial"])
        self.assertEqual(ended["how"], "finished", ended.get("error"))
        after, _ = hand(self.live.session, "right")
        self.assertGreater(float(after[2] - before[2]), 0.04, "the hand did not rise")
        self.assertTrue(self.live.session.paused, "Pause is left as it was")
        self.assertFalse(any(p.suspended for p in self.live.session.arms.values()), "the arms drive the routine")

    def test_the_towel_is_named_in_the_segmentation(self) -> None:
        from robosimstudio_real2sim_agent import checks, stills

        sim = self.live.sim
        cam = stills.camera_for(sim, self.live.scene, name="top")
        _, labels = checks.seg_names(sim)
        renderer = sim.make_renderer("tiled", cameras=[cam], channels=["seg"])
        try:
            raw = np.asarray(renderer.render(sim.state)[cam.name]["seg"]).astype(np.int32)
        finally:
            renderer.close()
        seg = checks.name_deformables(sim, cam, raw, labels)
        self.assertIn("towel", labels)
        self.assertGreater(int((seg == labels["towel"]).sum()), 100, "the towel is in view and named")
        self.assertFalse((seg < 0).any(), "no hit is left unnamed")


if __name__ == "__main__":
    unittest.main()
