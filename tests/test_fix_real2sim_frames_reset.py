# SPDX-License-Identifier: Apache-2.0
"""The clip's stills and a reset under ``run N``, held to what they say they do."""

from __future__ import annotations

import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from robosimstudio_real2sim_agent import ctl, frames
from robosimstudio_real2sim_agent.server import LiveServer
from tests.test_agent import _FakePage

TIER = "fast"


def _clip(path: Path, seconds: float, colour: str, fmt: str = "mp4") -> Path:
    """A clip of one flat colour, at 10 fps."""
    import imageio_ffmpeg

    argv = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi"]
    argv += ["-i", f"color=c={colour}:size=64x48:rate=10:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if fmt == "h264":
        argv += ["-f", "h264"]
    subprocess.run([*argv, str(path)], check=True, stdin=subprocess.DEVNULL)
    return path


def _cut(out: Path, clip: Path, at: str) -> dict:
    host = types.SimpleNamespace(ref_dir=lambda: out)
    return LiveServer.cut_frames(host, str(clip), at=at, out=str(out))


class TestStills(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="fx-"))

    def test_a_clip_with_no_duration_in_its_header_is_refused(self) -> None:
        clip = _clip(self.tmp / "clip.h264", 3, "red", fmt="h264")
        self.assertEqual(frames.probe(clip)["duration"], 0.0, "ffmpeg reads Duration: N/A")
        reply = _cut(self.tmp / "ref", clip, "0,end")
        self.assertFalse(reply["ok"])
        self.assertIn("re-mux", reply["error"])
        self.assertFalse((self.tmp / "ref").exists(), "nothing cut, no stray still")

    def test_two_times_inside_one_still_name_are_one_anchor(self) -> None:
        clip = _clip(self.tmp / "clip.mp4", 2, "red")
        reply = _cut(self.tmp / "ref", clip, "0,1.701,1.704:b")
        self.assertTrue(reply["ok"], reply)
        self.assertEqual(len(reply["frames"]), len(set(reply["frames"])), "one still is cut once")
        anchors = json.loads((self.tmp / "ref" / frames.TIMELINE).read_text())["anchors"]
        files = [a["file"] for a in anchors]
        self.assertEqual(len(files), len(set(files)), "no two anchors share a still")
        for a in anchors:
            self.assertEqual(a["file"], f"t{a['t']:06.2f}.jpg", "the anchor's time is the still's")
        self.assertIn("b", [a.get("label") for a in anchors], "the one label in the still is kept")

    def test_two_labels_inside_one_still_name_are_refused(self) -> None:
        clip = _clip(self.tmp / "clip.mp4", 2, "red")
        reply = _cut(self.tmp / "ref", clip, "0,1.701:a,1.704:b")
        self.assertFalse(reply["ok"])
        self.assertIn("1.701:a and 1.704:b", reply["error"])
        self.assertIn("t001.70.jpg", reply["error"])
        self.assertFalse((self.tmp / "ref" / frames.TIMELINE).exists(), "no label is dropped into a timeline")

    def test_a_still_left_by_another_clip_is_not_taken_for_a_new_one(self) -> None:
        from PIL import Image

        ref = self.tmp / "ref"
        long, short = _clip(self.tmp / "long.mp4", 5, "red"), _clip(self.tmp / "short.mp4", 2, "blue")
        (stale,) = frames.extract(long, [4.0], ref)
        self.assertGreater(np.asarray(Image.open(stale))[..., 0].mean(), 200)
        with self.assertRaises(ValueError):
            frames.extract(short, [4.0], ref)
        self.assertFalse(stale.exists(), "the red still from the long clip is gone")


class _Session:
    def __init__(self) -> None:
        self.paused = False

    def set_paused(self, paused: bool) -> None:
        self.paused = paused


def _running(budget: int) -> LiveServer:
    host = LiveServer.__new__(LiveServer)
    host.script, host.run_budget, host.runs, host.run_target = [], budget, 1, 100
    host.runs_ended = {}
    host.session = _Session()
    host.sim = types.SimpleNamespace(step_count=50, state_is_finite=lambda: True)
    host._finish_recorder = lambda why: None  # type: ignore[method-assign]
    host._end_scripts = lambda how: None  # type: ignore[method-assign]
    return host


class TestResetUnderARun(unittest.TestCase):
    def test_a_reset_ends_a_run_held(self) -> None:
        host = _running(50)
        host.stop_scripts("reset")
        self.assertEqual(host.run_budget, 0)
        self.assertTrue(host.session.paused, "the run ends paused, as it would have at its last step")
        self.assertEqual((host.last_run["how"], host.last_run["step"]), ("cut short: reset", 50))

    def test_a_free_running_page_is_left_running(self) -> None:
        host = _running(-1)
        host.stop_scripts("reset")
        self.assertEqual(host.run_budget, -1)
        self.assertFalse(host.session.paused)
        self.assertEqual(host.last_run, {})

    def test_the_wait_says_the_run_was_reset(self) -> None:
        path = Path(tempfile.mkdtemp(prefix="run-")) / "ctl.sock"
        ended = {"serial": 1, "how": "cut short: reset", "step": 0, "target": 10, "finite": None}
        page = _FakePage(path, {"step": 0, "time": 0.0, "runs_ended": {"1": ended}})
        self.addCleanup(page.close)
        reply = ctl.wait_for_steps(path, {"ok": True, "run": 1, "step": 0, "target": 10}, 30.0)
        self.assertFalse(reply["ok"])
        self.assertIn("reset", reply["error"])


if __name__ == "__main__":
    unittest.main()
