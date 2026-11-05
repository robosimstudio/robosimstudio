# SPDX-License-Identifier: Apache-2.0
"""A reference clip turned into what the page compares against: stills, the timeline, the clean plate.

A still is cut once per name and a label is never dropped; a clip with no
duration in its header is refused; a label marked again moves; the plate is
the background with what passed through it gone.
"""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from robosimstudio_real2sim_agent import frames
from robosimstudio_real2sim_agent.server import LiveServer
from tests.fixtures.real2sim_clips import flat_clip

TIER = "fast"
"""No GPU: small clips ffmpeg writes for the purpose."""


def _cut(out: Path, clip: Path, at: str) -> dict:
    host = types.SimpleNamespace(ref_dir=lambda: out)
    return LiveServer.cut_frames(host, str(clip), at=at, out=str(out))


class TestStills(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="fx-"))

    def test_a_clip_with_no_duration_in_its_header_is_refused(self) -> None:
        clip = flat_clip(self.tmp / "clip.h264", 3, "red", fmt="h264")
        self.assertEqual(frames.probe(clip)["duration"], 0.0, "ffmpeg reads Duration: N/A")
        reply = _cut(self.tmp / "ref", clip, "0,end")
        self.assertFalse(reply["ok"])
        self.assertIn("re-mux", reply["error"])
        self.assertFalse((self.tmp / "ref").exists(), "nothing cut, no stray still")

    def test_two_times_inside_one_still_name_are_one_anchor(self) -> None:
        clip = flat_clip(self.tmp / "clip.mp4", 2, "red")
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
        clip = flat_clip(self.tmp / "clip.mp4", 2, "red")
        reply = _cut(self.tmp / "ref", clip, "0,1.701:a,1.704:b")
        self.assertFalse(reply["ok"])
        self.assertIn("1.701:a and 1.704:b", reply["error"])
        self.assertIn("t001.70.jpg", reply["error"])
        self.assertFalse((self.tmp / "ref" / frames.TIMELINE).exists(), "no label is dropped into a timeline")

    def test_a_still_left_by_another_clip_is_not_taken_for_a_new_one(self) -> None:
        from PIL import Image

        ref = self.tmp / "ref"
        long, short = flat_clip(self.tmp / "long.mp4", 5, "red"), flat_clip(self.tmp / "short.mp4", 2, "blue")
        (stale,) = frames.extract(long, [4.0], ref)
        self.assertGreater(np.asarray(Image.open(stale))[..., 0].mean(), 200)
        with self.assertRaises(ValueError):
            frames.extract(short, [4.0], ref)
        self.assertFalse(stale.exists(), "the red still from the long clip is gone")


class TestRemark(unittest.TestCase):
    def test_a_label_marked_again_replaces_its_old_moment(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="tl-"))
        for name in ("t000.00.jpg", "t004.50.jpg", "t004.60.jpg", "t009.97.jpg"):
            (tmp / name).write_bytes(b"")
        frames.write_timeline(
            tmp,
            {"clip": "c.mov"},
            [
                {"t": 0.0, "file": "t000.00.jpg", "label": "start"},
                {"t": 4.5, "file": "t004.50.jpg", "label": "lid_open"},
                {"t": 9.97, "file": "t009.97.jpg", "label": "end"},
            ],
        )
        frames.write_timeline(tmp, {"clip": "c.mov"}, [{"t": 4.6, "file": "t004.60.jpg", "label": "lid_open"}])
        anchors = frames.read_timeline(tmp)["anchors"]
        self.assertEqual([(a["t"], a["label"]) for a in anchors], [(0.0, "start"), (4.6, "lid_open"), (9.97, "end")])

    def test_an_unlabelled_anchor_leaves_the_labels_alone(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="tl-"))
        frames.write_timeline(tmp, {}, [{"t": 4.5, "file": "t004.50.jpg", "label": "lid_open"}])
        frames.write_timeline(tmp, {}, [{"t": 2.0, "file": "t002.00.jpg", "label": ""}])
        self.assertEqual([a["t"] for a in frames.read_timeline(tmp)["anchors"]], [2.0, 4.5])


def _clip(path: Path, frames_n: int = 30, w: int = 96, h: int = 64) -> np.ndarray:
    """A grey-blue background with a red square sweeping across it once; returns the background."""
    import imageio.v2 as imageio

    background = np.zeros((h, w, 3), np.uint8)
    background[..., 0], background[..., 1], background[..., 2] = 60, 110, 170
    background[h // 2 :, :] = (200, 190, 170)  # a table under a wall
    writer = imageio.get_writer(str(path), fps=10, codec="libx264", quality=10, macro_block_size=16)
    for i in range(frames_n):
        frame = background.copy()
        x = int(i * (w - 16) / (frames_n - 1))
        frame[20:36, x : x + 16] = (220, 30, 30)
        writer.append_data(frame)
    writer.close()
    return background


class TestPlate(unittest.TestCase):
    def test_what_passes_through_is_gone(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="plate-"))
        background = _clip(tmp / "clip.mp4")
        reply = frames.plate(tmp / "clip.mp4", count=20, out_dir=tmp, name="plate.png")
        dotted = frames.plate(tmp / "clip.mp4", count=6, out_dir=tmp, name="plate_cam.v2")
        self.assertEqual(Path(dotted["plate"]).name, "plate_cam.v2.png", "only a trailing .png is dropped")
        self.assertTrue(reply["ok"], reply)
        self.assertEqual(Path(reply["plate"]).name, "plate.png", "a name given with .png keeps one")
        from PIL import Image

        got = np.asarray(Image.open(reply["plate"]).convert("RGB")).astype(int)
        self.assertLess(
            np.abs(got - background).mean(), 6.0, "the square swept past every pixel in under half the frames"
        )
        self.assertTrue(Path(reply["spread"]).exists())

    def test_windows_and_paint(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="plate-"))
        _clip(tmp / "clip.mp4")
        self.assertFalse(frames.plate(tmp / "clip.mp4", windows="2:1", out_dir=tmp)["ok"], "an empty window is refused")
        from PIL import Image

        mask = np.zeros((64, 96), np.uint8)
        mask[40:50, 40:50] = 255
        Image.fromarray(mask).save(tmp / "mask.png")
        reply = frames.plate(tmp / "clip.mp4", windows="0:1,2:2.9", count=12, out_dir=tmp, paint=tmp / "mask.png")
        self.assertTrue(reply["ok"], reply)
        self.assertIn("painted", reply["painted"])
        self.assertIn("0 to 1 s", reply["windows"])


if __name__ == "__main__":
    unittest.main()
