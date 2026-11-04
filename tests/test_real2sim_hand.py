# SPDX-License-Identifier: Apache-2.0
"""The hand line: a camera on a wrist held to the hand its clip shows, and the path tracer's cache.

A wrist camera's clip has the robot's own palm and fingers in the same pixels in
every frame. Two twins of one two-arm clip mounted the right wrist camera the
way round the left one was, though the clip has that camera turned 180 degrees;
nothing they ran could see it. These are the pure parts of the line that does:
the still pixels of a clip, their overlap with the drawn robot and the turn
that would fit it, the frames it samples, the ``check --clip`` request, and the
warning a page gives when the path tracer cannot keep its cache.
"""

from __future__ import annotations

import os
import stat
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from robosimstudio_real2sim_agent import checks, ctl, frames, stills

TIER = "fast"

H, W = 60, 80


def _clip(seed: int = 0) -> np.ndarray:
    """Twenty frames of a world that never repeats, with a hand that never moves at the bottom middle."""
    out = np.random.default_rng(seed).integers(0, 255, (20, H, W, 3), dtype=np.uint8)
    out[:, 45:, 20:60] = 12
    return out


def _robot(rows: slice) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[rows, 20:60] = True
    return mask


class ChangeMap(unittest.TestCase):
    def test_the_hand_stays_and_the_world_changes(self) -> None:
        change = checks.change_map(_clip())
        self.assertEqual(change.shape, (H, W))
        self.assertLess(float(change[50, 40]), 0.05, "the hand's pixels never change")
        self.assertGreater(float(change[10, 10]), 0.8, "the world's change in nearly every frame")


class HandReading(unittest.TestCase):
    def test_a_robot_drawn_on_the_hand_passes(self) -> None:
        reading = checks.hand_reading(checks.change_map(_clip()), _robot(slice(45, None)))
        self.assertTrue(reading["ok"], reading["detail"])
        self.assertGreater(reading["iou"], 0.9)
        self.assertNotIn("turned", reading["detail"])

    def test_a_camera_turned_180_degrees_fails_and_is_named(self) -> None:
        reading = checks.hand_reading(checks.change_map(_clip()), _robot(slice(0, 15)))
        self.assertFalse(reading["ok"])
        self.assertEqual(reading["iou"], 0.0)
        self.assertIn("turned 180 degrees", reading["detail"], "a turn beats the mirror that fits as well")
        self.assertGreater(reading["orientations"]["turned 180 degrees"], 0.9)
        self.assertGreater(reading["clip_centre"][1], 0.8, "the clip's hand at the bottom")
        self.assertLess(reading["robot_centre"][1], 0.2, "the drawn robot at the top")

    def test_a_robot_drawn_elsewhere_fails_without_a_turn(self) -> None:
        robot = np.zeros((H, W), dtype=bool)
        robot[20:30, 0:10] = True
        reading = checks.hand_reading(checks.change_map(_clip()), robot)
        self.assertFalse(reading["ok"])
        self.assertNotIn("is the camera mounted", reading["detail"], "no orientation explains it")

    def test_no_robot_drawn_fails_with_where_the_hand_is(self) -> None:
        reading = checks.hand_reading(checks.change_map(_clip()), np.zeros((H, W), dtype=bool))
        self.assertFalse(reading["ok"])
        self.assertIn("draws none of the robot", reading["detail"])

    def test_a_clip_with_nothing_still_is_a_skip(self) -> None:
        moving = np.random.default_rng(1).integers(0, 255, (20, H, W, 3), dtype=np.uint8)
        reading = checks.hand_reading(checks.change_map(moving), _robot(slice(45, None)))
        self.assertIsNone(reading["ok"])


class HandLineInTheGate(unittest.TestCase):
    def test_a_camera_on_no_body_is_skipped_and_the_clip_is_not_read(self) -> None:
        sim = mock.MagicMock()
        sim.cameras = [types.SimpleNamespace(name="phone", width=W, height=H, fov_y=40.0, parent_body=-1)]
        server = types.SimpleNamespace(sim=sim, build_error="")
        with (
            mock.patch.object(checks, "hand_line") as line,
            mock.patch.object(checks, "_penetration", return_value=(0.0, [])),
        ):
            sim.state_is_finite.return_value = True
            sim.names.return_value = []
            result = checks.gate(server, camera="phone", clip="/nowhere.mp4", steps=1)
        line.assert_not_called()
        hand = [c for c in result["checks"] if c["name"] == "hand"]
        self.assertEqual(len(hand), 1)
        self.assertIsNone(hand[0]["ok"])

    def test_a_clip_that_cannot_be_read_fails_the_line(self) -> None:
        sim = mock.MagicMock()
        sim.cameras = [types.SimpleNamespace(name="wrist", width=W, height=H, fov_y=58.0, parent_body=7)]
        server = types.SimpleNamespace(sim=sim, build_error="", run_dir=tempfile.mkdtemp(prefix="hand-gate-"))
        with mock.patch.object(checks, "_penetration", return_value=(0.0, [])):
            sim.state_is_finite.return_value = True
            sim.names.return_value = []
            result = checks.gate(server, camera="wrist", clip="/nowhere/wrist.mp4", steps=1)
        hand = next(c for c in result["checks"] if c["name"] == "hand")
        self.assertFalse(hand["ok"])
        self.assertIn("wrist.mp4", hand["detail"])


class SampleFrames(unittest.TestCase):
    def test_frames_spread_over_the_clip_at_the_size_asked(self) -> None:
        import imageio.v2 as imageio

        clip = Path(tempfile.mkdtemp(prefix="hand-sample-")) / "c.mp4"
        with imageio.get_writer(str(clip), fps=10, macro_block_size=1) as w:
            for i in range(30):
                w.append_data(np.full((48, 64, 3), i * 8, dtype=np.uint8))
        got = frames.sample(clip, count=6, width=32, height=24)
        self.assertEqual(got.shape[1:], (24, 32, 3))
        self.assertGreaterEqual(len(got), 5)
        self.assertLess(int(got[0].mean()), int(got[-1].mean()), "from the start of the clip to its end")

    def test_a_missing_clip_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            frames.sample("/nowhere/at/all.mp4")


class CheckClipRequest(unittest.TestCase):
    def test_the_clip_goes_to_the_page_as_an_absolute_path(self) -> None:
        parser = ctl.build_parser()
        req, _ = ctl.build_request(parser.parse_args(["check", "--camera", "wrist", "--clip", "ref/wrist.mp4"]), parser)
        self.assertEqual(req["cmd"], "check")
        self.assertTrue(Path(req["clip"]).is_absolute())
        self.assertEqual(Path(req["clip"]).name, "wrist.mp4")
        req, _ = ctl.build_request(parser.parse_args(["check"]), parser)
        self.assertEqual(req["clip"], "", "no clip, no hand line")


@unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root writes a read-only folder")
class RtxCacheWarning(unittest.TestCase):
    def _package(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="ovrtx-pkg-"))
        (root / "bin" / "cache").mkdir(parents=True)
        (root / "__init__.py").write_text("")
        return root

    def test_a_read_only_cache_is_said(self) -> None:
        root = self._package()
        cache = root / "bin" / "cache"
        cache.chmod(stat.S_IRUSR | stat.S_IXUSR)
        self.addCleanup(cache.chmod, stat.S_IRWXU)
        spec = types.SimpleNamespace(origin=str(root / "__init__.py"))
        with mock.patch("importlib.util.find_spec", return_value=spec):
            warning = stills.rtx_cache_warning()
        self.assertIn(str(cache), warning)
        self.assertIn("minutes", warning)

    def test_a_writable_cache_or_no_path_tracer_says_nothing(self) -> None:
        spec = types.SimpleNamespace(origin=str(self._package() / "__init__.py"))
        with mock.patch("importlib.util.find_spec", return_value=spec):
            self.assertEqual(stills.rtx_cache_warning(), "")
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertEqual(stills.rtx_cache_warning(), "")


if __name__ == "__main__":
    unittest.main()
