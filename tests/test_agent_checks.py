# SPDX-License-Identifier: Apache-2.0
"""The numbers the live page puts beside a picture, checked on pictures drawn for the purpose.

Keys parse, a colour finds its patch, the offsets and the size ratio come out
as drawn, a clip's timeline is cut and merged, a recorded take is held against
its anchors. No GPU and no page: the render here is an array with a
segmentation drawn beside it, which is all the arithmetic ever sees. What
needs a scene -- the gate's rest and overlap checks, the recorder -- is
exercised on the page by hand and documented in ``examples/live/REFERENCE.md``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robosimstudio_real2sim_agent import checks, compare, frames
from robosimstudio_real2sim_agent import compare as C

TIER = "fast"
"""No GPU: arrays stand in for renders."""


def _photo(w: int = 640, h: int = 360) -> np.ndarray:
    """A beige table with a red box at (100..300, 80..200) and a grey bolt at (400..420, 150..170)."""
    photo = np.full((h, w, 3), (190, 170, 140), np.uint8)
    photo[80:200, 100:300] = (210, 30, 20)
    photo[150:170, 400:420] = (180, 182, 188)
    return photo


def _render(w: int = 640, h: int = 360) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """The same scene rendered 0.8x and 40 px right, with its segmentation."""
    rgb = np.full((h, w, 3), (60, 60, 60), np.uint8)
    seg = np.zeros((h, w), np.int32)
    rgb[90:186, 140:300] = (220, 40, 30)
    seg[90:186, 140:300] = 2
    rgb[152:168, 442:458] = (170, 170, 175)
    seg[152:168, 442:458] = 3
    return rgb, seg, {1: "table", 2: "case", 3: "bolt", 4: "__robot__"}


class TestKeys(unittest.TestCase):
    def test_the_grammar(self) -> None:
        k = compare.parse_key("case=red")
        self.assertEqual((k.name, k.colour, k.seed), ("case", "red", None))
        self.assertEqual(compare.parse_key("bolt=chrome").colour, "grey", "aliases resolve")
        k = compare.parse_key("case=#d21e14~0.05")
        self.assertEqual((k.colour, k.tolerance), ("#d21e14", 0.05))
        k = compare.parse_key("bolt=@410,160")
        self.assertEqual((k.colour, k.seed), ("", (410, 160)))
        k = compare.parse_key("case=red@200,140")
        self.assertEqual((k.colour, k.seed), ("red", (200, 140)))
        self.assertIn("around (410, 160)", compare.parse_key("bolt=@410,160").describe())
        for bad in (
            "bad",
            "=red",
            "case=",
            "case=@1",
            "case=red tool case",
            "bolt=socket@410,160",
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                compare.parse_key(bad)


class TestMasks(unittest.TestCase):
    def test_a_colour_finds_its_patch_and_a_seed_its_component(self) -> None:
        photo = _photo()
        case = compare.photo_mask(photo, compare.parse_key("case=red"))
        self.assertEqual(compare.mask_stats(case)["bbox"], [100, 80, 300, 200])
        bolt = compare.photo_mask(photo, compare.parse_key("bolt=@410,160"))
        self.assertEqual(compare.mask_stats(bolt)["bbox"], [400, 150, 420, 170])
        near = compare.photo_mask(photo, compare.parse_key("bolt=grey@425,160"))
        self.assertEqual(
            compare.mask_stats(near)["bbox"],
            [400, 150, 420, 170],
            "a seed a few px off still lands",
        )
        self.assertIsNone(compare.mask_stats(compare.photo_mask(photo, compare.parse_key("sky=blue"))))
        self.assertIsNone(compare.mask_stats(None))

    def test_the_largest_patch_wins_without_a_seed(self) -> None:
        photo = _photo()
        photo[300:310, 500:510] = (210, 30, 20)  # a red speck elsewhere
        case = compare.photo_mask(photo, compare.parse_key("case=red"))
        self.assertEqual(compare.mask_stats(case)["bbox"], [100, 80, 300, 200])

    def test_the_readings_come_out_as_drawn(self) -> None:
        photo, (_rgb, seg, names) = _photo(), _render()
        masks = compare.sim_masks(seg, names, ["case", "bolt", "ghost"])
        self.assertIsNone(masks["ghost"], "a name the scene has no label for")
        case = compare.compare_masks(compare.photo_mask(photo, compare.parse_key("case=red")), masks["case"])
        self.assertEqual(case["offset_px"], [20.0, -2.0])
        self.assertAlmostEqual(case["size_ratio"], 0.8, places=2)
        self.assertAlmostEqual(case["width_ratio"], 0.8, places=2)
        self.assertGreater(case["iou"], 0.6)
        line = compare.advice("case", case, 640, compare.parse_key("case=red"))
        self.assertIn("0.80x", line)
        self.assertIn("20 px right", line)
        self.assertIn("IoU", line)
        missing = compare.compare_masks(compare.photo_mask(photo, compare.parse_key("x=blue")), masks["case"])
        self.assertIsNone(missing["iou"])
        self.assertIn("found nothing in the photograph", compare.advice("x", missing, 640))
        self.assertIn(
            "not in the render",
            compare.advice("ghost", compare.compare_masks(masks["case"], None), 640),
        )

    def test_fitting_and_pictures(self) -> None:
        photo = _photo(1280, 720)
        fitted, stretch = compare.fit_photo(photo, 640, 360)
        self.assertEqual((fitted.shape, stretch), ((360, 640, 3), 1.0))
        _, stretch = compare.fit_photo(photo, 640, 480)
        self.assertAlmostEqual(stretch, 4.0 / 3.0, places=3)
        mask = compare.fit_mask(compare.photo_mask(photo, compare.parse_key("case=red")), 640, 360)
        self.assertEqual(mask.shape, (360, 640))
        _rgb, seg, _names = _render()
        pairs = {
            "case": (
                compare.photo_mask(fitted, compare.parse_key("case=red")),
                seg == 2,
            )
        }
        over = compare.overlay(fitted, pairs)
        self.assertEqual(over.shape, fitted.shape)
        self.assertTrue(
            (over == compare.SIM_COLOUR).all(axis=-1).any(),
            "the render's outline is drawn",
        )
        self.assertTrue((over == compare.PHOTO_COLOUR).all(axis=-1).any(), "and the photo's")


class TestWhatAnEyeSees(unittest.TestCase):
    def test_crops_and_the_sheet(self) -> None:
        photo, (render, seg, _) = _photo(), _render()
        pm = C.photo_mask(photo, C.parse_key("case=red"))
        tile = C.crop_pair(photo, render, pm, seg == 2, height=120)
        self.assertEqual(tile.shape[0], 120)
        self.assertIsNone(C.crop_pair(photo, render, None, None))
        behind = C.without_things(photo, pm)
        self.assertTrue((behind[140, 200] == 40).all(), "the thing is painted out")
        self.assertTrue((behind[300, 500] == photo[300, 500]).all(), "the rest is kept")
        sheet = C.look_sheet(photo, render, [("case up close", tile), ("none", None)], width=640)
        self.assertEqual(sheet.shape[1], 640)
        self.assertGreater(sheet.shape[0], 120)


class TestCompareStill(unittest.TestCase):
    def test_the_readings_and_the_sheet(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="cmp-"))
        from PIL import Image

        Image.fromarray(_photo()).save(tmp / "ref.png")
        rgb, seg, names = _render()
        keys = [compare.parse_key("case=red"), compare.parse_key("bolt=@410,160")]
        result = checks.compare_still(rgb, seg, names, tmp / "ref.png", keys)
        self.assertEqual(set(result["objects"]), {"case", "bolt"})
        self.assertAlmostEqual(
            result["score"],
            np.mean([o["iou"] for o in result["objects"].values()]),
            places=4,
        )
        self.assertEqual(len(result["lines"]), 2, "one line per key, nothing else")
        pictures = checks.write_compare_pictures(tmp / "cmp_001.png", rgb, result)
        self.assertTrue(Path(pictures["look"]).exists() and pictures["look"].endswith("_look.png"))
        beside = Image.open(pictures["beside"])
        self.assertEqual(
            beside.size, (2 * rgb.shape[1], rgb.shape[0]), "the render and the photo side by side, full size"
        )
        none = checks.compare_still(rgb, seg, names, tmp / "ref.png", [])
        self.assertEqual(none["score"], 0.0)
        self.assertIn("no --key", none["lines"][0])


class TestFrames(unittest.TestCase):
    def test_times_and_the_timeline(self) -> None:
        self.assertEqual(
            frames.parse_times("0,3.7:lid_up, end", 24.9),
            [(0.0, ""), (3.7, "lid_up"), (24.899, "")],
        )
        tmp = Path(tempfile.mkdtemp(prefix="tl-"))
        self.assertEqual(frames.read_timeline(tmp), {})
        meta = {
            "clip": "c.mov",
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "duration": 24.9,
            "frames": 747,
        }
        frames.write_timeline(
            tmp,
            meta,
            [
                {"t": 3.7, "file": "t003.70.jpg", "label": "lid_up"},
                {"t": 0.0, "file": "t000.00.jpg", "label": ""},
            ],
        )
        line = frames.read_timeline(tmp)
        self.assertEqual([a["t"] for a in line["anchors"]], [0.0, 3.7], "sorted by time")
        frames.write_timeline(tmp, {}, [{"t": 3.7, "file": "t003.70.jpg", "sim_t": 8.0}])
        line = frames.read_timeline(tmp)
        self.assertEqual(
            (line["width"], line["anchors"][1]["label"], line["anchors"][1]["sim_t"]),
            (1920, "lid_up", 8.0),
            "merged, nothing lost",
        )
        for name in ("t003.70.jpg", "t024.87.jpg"):
            (tmp / name).write_bytes(b"")
        frames.write_timeline(tmp, {}, [{"t": 24.87, "file": "t024.87.jpg", "label": "end"}])
        self.assertEqual(
            [a["t"] for a in frames.read_timeline(tmp)["anchors"]],
            [3.7, 24.87],
            "an anchor whose still is gone is dropped",
        )
        frames.write_timeline(
            tmp,
            {**meta, "clip": "other.mov", "duration": 5.0},
            [{"t": 0.0, "file": "t000.00.jpg"}],
        )
        line = frames.read_timeline(tmp)
        self.assertEqual(
            ([a["t"] for a in line["anchors"]], line["clip"]),
            ([0.0], "other.mov"),
            "a cut from another clip keeps none of the first clip's moments",
        )

    def test_a_clip_is_probed_and_cut(self) -> None:
        from robosimstudio.core.render.video import VideoWriter

        tmp = Path(tempfile.mkdtemp(prefix="clip-"))
        writer = VideoWriter(tmp / "clip.mp4", fps=10)
        for i in range(20):
            frame = np.zeros((72, 128, 3), np.uint8)
            frame[:, : 6 * (i + 1)] = (200, 30, 30)  # a red bar growing across
            writer.append(frame)
        writer.close()
        meta = frames.probe(tmp / "clip.mp4")
        self.assertEqual((meta["fps"], meta["width"], meta["height"]), (10.0, 128, 72))
        self.assertAlmostEqual(meta["duration"], 2.0, places=1)
        cut = frames.extract(tmp / "clip.mp4", [0.0, 1.0, 1.9], tmp / "ref")
        self.assertEqual([p.name for p in cut], ["t000.00.jpg", "t001.00.jpg", "t001.90.jpg"])
        sheet, tiles = frames.contact_sheet(tmp / "clip.mp4", 0.5, tmp / "ref" / "contact.jpg", columns=3, width=64)
        self.assertTrue(sheet.exists())
        self.assertEqual(tiles, 4)


class TestTheLog(unittest.TestCase):
    def test_append_read_report(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="log-"))
        self.assertIn("nothing logged", checks.log_report(checks.log_read(tmp)))
        self.assertEqual(
            checks.log_append(tmp, {"cmd": "compare", "scene": "abc", "score": 0.31, "note": "first"}),
            1,
        )
        self.assertEqual(checks.log_append(tmp, {"cmd": "check", "scene": "abc", "passed": False}), 2)
        rows = checks.log_read(tmp)
        self.assertEqual([r["n"] for r in rows], [1, 2])
        report = checks.log_report(rows)
        self.assertIn("0.310", report)
        self.assertIn("FAIL", report)
        self.assertEqual(checks.scene_hash(tmp / "nowhere.py"), "missing")


class TestATake(unittest.TestCase):
    def _take(self) -> tuple[Path, Path, int]:
        """A clip's two anchors and a recorded take of the same box moving right."""
        from PIL import Image

        from robosimstudio.core.render.video import VideoWriter

        tmp = Path(tempfile.mkdtemp(prefix="take-"))
        run, ref = tmp / "run", tmp / "ref"
        run.mkdir()
        ref.mkdir()
        w, h, n = 128, 72, 6
        # the clip: a red box moving right by 30 px over the clip; the take: the same box moving right by 32 px
        for i, name in ((0, "t000.00.jpg"), (n - 1, "t002.50.jpg")):
            photo = np.full((h, w, 3), (190, 170, 140), np.uint8)
            x = 10 + 6 * i
            photo[20:50, x : x + 30] = (210, 30, 20)
            Image.fromarray(photo).save(ref / name)
        frames.write_timeline(
            ref,
            {
                "clip": "c.mov",
                "fps": 30.0,
                "width": w,
                "height": h,
                "duration": 2.5,
                "frames": 75,
            },
            [
                {"t": 0.0, "file": "t000.00.jpg", "label": "start"},
                {"t": 2.5, "file": "t002.50.jpg", "label": "end"},
            ],
        )
        writer = VideoWriter(run / "take.mp4", fps=4)
        segs, rows = [], []
        for i in range(n):
            rgb = np.full((h, w, 3), (60, 60, 60), np.uint8)
            seg = np.zeros((h, w), np.int32)
            x = 12 + int(round(32 * i / (n - 1)))
            rgb[20:50, x : x + 30] = (220, 40, 30)
            seg[20:50, x : x + 30] = 2
            writer.append(rgb)
            segs.append(seg[:: checks.SEG_SCALE, :: checks.SEG_SCALE].astype(np.uint8))
            rows.append({"i": i, "step": 4 * i, "t": i * 0.25, "objects": {}, "positions": {}})
        writer.close()
        np.savez_compressed(run / "take_seg.npz", seg=np.stack(segs))
        (run / "take.json").write_text(
            json.dumps(
                {
                    "name": "take",
                    "camera": "cam",
                    "width": w,
                    "height": h,
                    "every": 4,
                    "fps": 4,
                    "names": {"1": "table", "2": "case"},
                    "video": str(run / "take.mp4"),
                    "frames": rows,
                }
            )
        )
        return run, ref, n

    def test_held_against_its_anchors(self) -> None:
        run, ref, n = self._take()
        result = checks.compare_take(run, "take", frames.read_timeline(ref), [compare.parse_key("case=red")], ref)
        self.assertEqual(
            [a["frame"] for a in result["anchors"]],
            [0, n - 1],
            "the first and last anchors meet the first and last frames",
        )
        motion = result["motion"]["case"]
        self.assertTrue(motion["photo"]["moved"] and motion["sim"]["moved"])
        self.assertAlmostEqual(motion["photo"]["dx"], 30.0, delta=1.0)
        self.assertAlmostEqual(motion["sim"]["dx"], 32.0, delta=1.0)
        self.assertTrue(any("moved in both" in line for line in result["lines"]))
        self.assertGreater(result["score"], 0.5)
        strip = checks.write_strip(run / "cmp_001.png", result["tiles"], tile_width=128)
        self.assertTrue(Path(strip).exists())
        self.assertEqual(checks.motion_summary(result["motion"]), "1 of 1 did what the clip did")

    def test_verdict_names_what_differed(self) -> None:
        motion = {
            "lid": {"photo": {"dx": 40.0, "dy": 0.0, "moved": True}, "sim": {"dx": 1.0, "dy": 0.0, "moved": False}},
            "socket": {"photo": {"dx": 30.0, "dy": 5.0, "moved": True}, "sim": {"dx": 28.0, "dy": 4.0, "moved": True}},
            "bar": {"photo": None, "sim": {"dx": 0.0, "dy": 0.0, "moved": False}},
        }
        self.assertEqual(
            checks.motion_summary(motion),
            "1 of 3 did what the clip did; lid stayed (clip moved); not found at both ends: bar",
        )


class TestNumbering(unittest.TestCase):
    """The page restarts; the files on disk are the count, so nothing is overwritten."""

    def test_continues_from_the_files_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.assertEqual(checks.next_number(run, "cmp"), 1, "an empty run folder starts at 1")
            for name in ("cmp_001.json", "cmp_001_look.png", "cmp_004_beside.png", "snap_002.png", "cmp_x.png"):
                (run / name).write_bytes(b"")
            self.assertEqual(checks.next_number(run, "cmp"), 5, "after the highest compare on disk")
            self.assertEqual(checks.next_number(run, "snap"), 3, "snapshots are counted apart")
            self.assertEqual(checks.next_number(run, "cmp", seen=7), 8, "and never below what this process wrote")


if __name__ == "__main__":
    unittest.main()
