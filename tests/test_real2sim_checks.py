# SPDX-License-Identifier: Apache-2.0
"""The numbers the live page puts beside a picture, checked on pictures drawn for the purpose.

Keys parse, a colour finds its patch, the offsets and the size ratio come out
as drawn, a clip's timeline is cut and merged, a recorded take is held against
its anchors, and the gate reads its lines off a sim that is a stand-in. No GPU
and no page: the render here is an array with a segmentation drawn beside it,
which is all the arithmetic ever sees. What needs a real scene -- the overlap
check's contacts, the recorder's renders -- is exercised on the page by hand
and documented in ``examples/live/REFERENCE.md``.
"""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.solvers.render.tiled import SEG_PARTICLES, SEG_SOFT_MESH, _labels
from robosimstudio_real2sim_agent import checks, compare, frames
from robosimstudio_real2sim_agent import compare as C
from tests.fixtures.gate_sim import GateSim, gate_server

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
        self.assertNotIn("score", result, "no mean over the keys: it said nothing about any one thing")
        self.assertEqual(len(result["lines"]), 3, "one line per key, and the picture's")
        self.assertTrue(result["lines"][-1].startswith("picture: "))
        picture = result["picture"]
        self.assertGreater(picture["delta_e"], 0.0, "a render unlike its photo differs in colour")
        same = checks.picture_reading(result["fitted"], result["fitted"])
        self.assertEqual((same["delta_e"], same["delta_e_behind"]), (0.0, 0.0), "identical pixels read dE 0")
        pictures = checks.write_compare_pictures(tmp / "cmp_001.png", rgb, result)
        self.assertTrue(Path(pictures["look"]).exists() and pictures["look"].endswith("_look.png"))
        with Image.open(pictures["beside"]) as beside:
            self.assertEqual(
                beside.size, (2 * rgb.shape[1], rgb.shape[0]), "the render and the photo side by side, full size"
            )
        none = checks.compare_still(rgb, seg, names, tmp / "ref.png", [])
        self.assertEqual(none["objects"], {})
        self.assertIn("no --key", none["lines"][0])
        masked = checks.compare_still(
            rgb, seg, names, tmp / "ref.png", [compare.parse_key("case=mask:" + str(_mask_png(tmp)))]
        )
        self.assertAlmostEqual(
            masked["objects"]["case"]["iou"],
            result["objects"]["case"]["iou"],
            places=2,
            msg="a mask of the case keys what red keyed",
        )


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
            checks.log_append(tmp, {"cmd": "compare", "scene": "abc", "look": "/r/cmp_012_look.png", "note": "first"}),
            1,
        )
        self.assertEqual(checks.log_append(tmp, {"cmd": "check", "scene": "abc", "passed": False}), 2)
        rows = checks.log_read(tmp)
        self.assertEqual([r["n"] for r in rows], [1, 2])
        report = checks.log_report(rows)
        self.assertIn("cmp_012_look.png", report, "the picture a report cites, tied to the scene it measured")
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
            rows.append(
                {"i": i, "step": 4 * i, "t": i * 0.25, "objects": {}, "positions": {"case": [0.004 * i, 0.0, 0.1]}}
            )
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

    def test_beside_its_anchors_with_what_moved_in_the_take(self) -> None:
        run, ref, n = self._take()
        result = checks.compare_take(run, "take", frames.read_timeline(ref), [compare.parse_key("case=red")], ref)
        self.assertEqual(
            [a["frame"] for a in result["anchors"]],
            [0, n - 1],
            "the first and last anchors meet the first and last frames",
        )
        self.assertEqual(result["moved"]["case"], {"ended_mm": 4.0 * (n - 1), "farthest_mm": 4.0 * (n - 1)})
        self.assertIn(f"case went {4 * (n - 1):.0f} mm", result["lines"][-1])
        self.assertFalse(any("did what the clip did" in line for line in result["lines"]), "no verdict")
        strip = checks.write_strip(run / "cmp_001.png", result["tiles"], tile_width=128)
        self.assertTrue(Path(strip).exists())
        unkeyed = checks.compare_take(run, "take", frames.read_timeline(ref), [], ref)
        self.assertEqual(len(unkeyed["tiles"]), 2, "no key needed: the strip is the pictures")

    def test_a_key_turning_in_a_case_whose_base_stays_is_seen(self) -> None:
        frames_ = [
            {"positions": {"case": [0.0, 0.0, 0.0]}, "parts": {"case": [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]}},
            {"positions": {"case": [0.0, 0.0, 0.0]}, "parts": {"case": [[0.0, 0.0, 0.0], [0.0, 0.05, 0.0]]}},
        ]
        self.assertAlmostEqual(checks.take_motion(frames_)["case"]["farthest_mm"], 70.7, delta=0.1)

    def test_an_anchor_pinned_past_the_end_is_said(self) -> None:
        run, ref, n = self._take()
        line = frames.read_timeline(ref)
        line["anchors"][1]["sim_step"] = 10_000
        result = checks.compare_take(run, "take", line, [], ref)
        self.assertTrue(any("pinned past the take's end" in x for x in result["lines"]))
        self.assertEqual(result["anchors"][1]["frame"], n - 1)

    def test_a_part_that_swung_back_does_not_hide_one_that_stayed_moved(self) -> None:
        swing = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]  # the lid went 100 mm and came back
        slide = [[0.0, 0.0, 0.0], [0.0, 0.07, 0.0], [0.0, 0.07, 0.0]]  # the base went 70 mm and stayed
        frames_ = [{"positions": {"case": [0, 0, 0]}, "parts": {"case": [swing[i], slide[i]]}} for i in range(3)]
        self.assertEqual(checks.take_motion(frames_)["case"], {"ended_mm": 70.0, "farthest_mm": 100.0})

    def test_a_deformable_with_no_position_is_read_off_its_particles(self) -> None:
        frames_ = [
            {"positions": {}, "parts": {"rope": [[0, 0, 0], [0.1, 0, 0]]}},
            {"positions": {}, "parts": {"rope": [[0, 0, 0.03], [0.1, 0, 0]]}},
        ]
        self.assertEqual(checks.take_motion(frames_)["rope"]["farthest_mm"], 30.0)

    def test_a_lid_that_opened_and_shut_went_somewhere(self) -> None:
        track = [{"positions": {"lid": [0.0, 0.0, z]}} for z in (0.0, 0.05, 0.12, 0.05, 0.0)]
        moved = checks.take_motion(track)
        self.assertEqual(moved["lid"], {"ended_mm": 0.0, "farthest_mm": 120.0})
        self.assertEqual(checks.motion_line(moved), "in the take: lid went 120 mm and ended 0 mm from its start")


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


def _mask_png(tmp: Path) -> Path:
    """The photo's case as another tool would draw it: white on black, at the photo's size."""
    from PIL import Image

    photo = _photo()
    mask = (photo[..., 0] > 180) & (photo[..., 1] < 90)
    path = tmp / "case_mask.png"
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    return path


def _render_bolt_nearly_on(w: int = 640, h: int = 360) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """The box a little off, the bolt nearly on."""
    rgb = np.full((h, w, 3), (60, 60, 60), np.uint8)
    seg = np.zeros((h, w), np.int32)
    rgb[90:186, 140:300] = (220, 40, 30)
    seg[90:186, 140:300] = 2
    rgb[150:170, 402:422] = (170, 170, 175)
    seg[150:170, 402:422] = 3
    return rgb, seg, {1: "table", 2: "case", 3: "bolt"}


class TestAMissingThing(unittest.TestCase):
    def test_a_thing_gone_from_the_render_or_unlabelled_reads_as_not_in_the_render(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "photo.png"
            Image.fromarray(_photo()).save(path)
            keys = [compare.parse_key("case=red"), compare.parse_key("bolt=@410,160")]
            rgb, seg, names = _render_bolt_nearly_on()
            gone = seg.copy()
            gone[gone == 2] = 0  # the case off camera
            without = checks.compare_still(rgb, gone, names, path, keys)
            self.assertIsNone(without["objects"]["case"]["iou"])
            self.assertIn("case: not in the render", without["lines"][0])
            unlabelled = checks.compare_still(rgb, seg, {1: "table", 3: "bolt"}, path, keys)
            self.assertIsNone(unlabelled["objects"]["case"]["sim"], "so does a name the scene has no label for")


class TestTake(unittest.TestCase):
    def _take(self, tmp: Path, names: dict[str, str]) -> tuple[Path, Path]:
        from PIL import Image

        from robosimstudio.core.render.video import VideoWriter

        run, ref = tmp / "run", tmp / "ref"
        run.mkdir()
        ref.mkdir()
        w, h, n = 128, 72, 4
        for i, name in ((0, "t000.00.jpg"), (n - 1, "t001.00.jpg")):
            photo = np.full((h, w, 3), (190, 170, 140), np.uint8)
            photo[20:50, 10 + 6 * i : 40 + 6 * i] = (210, 30, 20)
            Image.fromarray(photo).save(ref / name)
        frames.write_timeline(
            ref,
            {"clip": "c.mov", "fps": 30.0, "width": w, "height": h, "duration": 1.0, "frames": 30},
            [
                {"t": 0.0, "file": "t000.00.jpg", "label": "start"},
                {"t": 1.0, "file": "t001.00.jpg", "label": "end"},
            ],
        )
        writer = VideoWriter(run / "take.mp4", fps=4)
        segs, rows = [], []
        for i in range(n):
            rgb = np.full((h, w, 3), (60, 60, 60), np.uint8)
            seg = np.zeros((h, w), np.int32)
            rgb[20:50, 10 + 6 * i : 40 + 6 * i] = (220, 40, 30)
            seg[20:50, 10 + 6 * i : 40 + 6 * i] = 2
            writer.append(rgb)
            segs.append(seg[:: checks.SEG_SCALE, :: checks.SEG_SCALE].astype(np.uint8))
            rows.append({"i": i, "step": i, "t": i / (n - 1), "objects": {}, "positions": {}})
        writer.close()
        np.savez_compressed(run / "take_seg.npz", seg=np.stack(segs))
        (run / "take.json").write_text(
            json.dumps(
                {
                    "name": "take",
                    "camera": "cam",
                    "width": w,
                    "height": h,
                    "every": 1,
                    "fps": 4,
                    "names": names,
                    "video": str(run / "take.mp4"),
                    "frames": rows,
                }
            )
        )
        return run, ref

    def test_a_thing_missing_from_the_take_is_still_drawn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, ref = self._take(Path(tmp), {"1": "table", "3": "other"})
            result = checks.compare_take(run, "take", frames.read_timeline(ref), [compare.parse_key("case=red")], ref)
            self.assertEqual(
                len(result["anchors"]), len(result["tiles"]), "the strip is drawn, the case outlined in the clip alone"
            )

    def test_the_strip_makes_its_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, ref = self._take(Path(tmp), {"1": "table", "2": "case"})
            result = checks.compare_take(run, "take", frames.read_timeline(ref), [compare.parse_key("case=red")], ref)
            strip = checks.write_strip(Path(tmp) / "new" / "dir" / "x.png", result["tiles"], tile_width=64)
            self.assertTrue(Path(strip).exists())


class TestGate(unittest.TestCase):
    def test_a_failed_routine_is_not_a_failed_build(self) -> None:
        server = gate_server(GateSim(), last_error="Traceback\nRuntimeError: routine broke")
        builds = checks.gate(server, steps=1)["checks"][0]
        self.assertEqual(builds["name"], "builds")
        self.assertTrue(builds["ok"], builds["detail"])
        server.build_error = "Traceback\nSyntaxError: bad scene"
        builds = checks.gate(server, steps=1)["checks"][0]
        self.assertFalse(builds["ok"])
        self.assertEqual(builds["detail"], "SyntaxError: bad scene")


class _Probe:
    def update(self) -> None:
        raise RuntimeError("probe update failed")

    def penetrations(self) -> list:
        return []


class TestGateProbe(unittest.TestCase):
    def test_a_probe_that_raises_is_a_failed_line(self) -> None:
        server = gate_server(GateSim(contacts=_Probe(), handles=types.SimpleNamespace(objects={})))
        result = checks.gate(server, steps=1)
        line = next(c for c in result["checks"] if c["name"] == "penetration")
        self.assertFalse(line["ok"])
        self.assertIn("probe update failed", line["detail"])
        self.assertFalse(result["passed"])
        self.assertIn("finite_after", [c["name"] for c in result["checks"]], "the gate ran to the end")


class TiledLabels(unittest.TestCase):
    def test_reserved_ids_are_their_own_labels_and_a_miss_is_zero(self) -> None:
        lut = np.array([0, 5, 7], np.int32)
        idx = np.array([1, 2, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD], np.int64)
        np.testing.assert_array_equal(_labels(idx, lut), [5, 7, 0, SEG_PARTICLES, SEG_SOFT_MESH])

    def test_without_a_table_shapes_are_their_own_index(self) -> None:
        idx = np.array([3, 0xFFFFFFFF, 0xFFFFFFFE], np.int64)
        np.testing.assert_array_equal(_labels(idx, None), [3, 0, SEG_PARTICLES])

    def test_a_shape_past_the_table_is_still_an_error(self) -> None:
        with self.assertRaisesRegex(IndexError, "not this model's"):
            _labels(np.array([1, 9], np.int64), np.array([0, 5, 7], np.int32))


def _camera() -> CameraSpec:
    return CameraSpec(name="c", width=100, height=80, fov_y=60.0, eye=(0.0, 0.0, 1.0), look_at=(0.0, 0.0, 0.0))


class CameraProject(unittest.TestCase):
    def test_the_point_looked_at_lands_on_the_principal_point(self) -> None:
        uv, depth = _camera().project(np.array([[0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(uv[0], [50.0, 40.0], atol=1e-9)
        self.assertAlmostEqual(float(depth[0]), 1.0)

    def test_a_point_behind_the_camera_has_no_depth(self) -> None:
        _, depth = _camera().project(np.array([[0.0, 0.0, 2.0]]))
        self.assertLess(float(depth[0]), 0.0)


def _deformable_sim(points: np.ndarray, handles: dict) -> types.SimpleNamespace:
    state = types.SimpleNamespace(particle_q=types.SimpleNamespace(numpy=lambda: points), body_q=None)
    scene = types.SimpleNamespace(
        handles=types.SimpleNamespace(segmentation_names=lambda: {1: "table", 2: "__robot__"}, deformables=handles)
    )
    return types.SimpleNamespace(scene=scene, state=state)


def _handle(particles, triangles=()) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        particles=np.asarray(particles), triangles=np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    )


class NameDeformables(unittest.TestCase):
    def test_labels_follow_the_rigid_ones_in_name_order(self) -> None:
        sim = _deformable_sim(np.zeros((0, 3)), {"towel": _handle([]), "rice": _handle([])})
        names, labels = checks.seg_names(sim)
        self.assertEqual(labels, {"rice": 3, "towel": 4})
        self.assertEqual(names[3], "rice")

    def test_one_of_each_kind_takes_every_hit_of_its_kind(self) -> None:
        sim = _deformable_sim(np.zeros((3, 3)), {"towel": _handle([0, 1, 2], [[0, 1, 2]]), "rice": _handle([0])})
        _, labels = checks.seg_names(sim)
        seg = np.array([[1, SEG_SOFT_MESH], [SEG_PARTICLES, 0]], np.int32)
        out = checks.name_deformables(sim, _camera(), seg, labels)
        np.testing.assert_array_equal(out, [[1, labels["towel"]], [labels["rice"], 0]])

    def test_two_heaps_split_by_where_their_particles_are_drawn(self) -> None:
        left = np.array([[-0.3, 0.0, 0.0], [-0.3, 0.05, 0.0]])
        right = np.array([[0.3, 0.0, 0.0], [0.3, 0.05, 0.0]])
        sim = _deformable_sim(np.concatenate([left, right]), {"beans": _handle([0, 1]), "rice": _handle([2, 3])})
        _, labels = checks.seg_names(sim)
        seg = np.full((80, 100), SEG_PARTICLES, np.int32)
        cam = _camera()
        out = checks.name_deformables(sim, cam, seg, labels)
        for name, points in (("beans", left), ("rice", right)):
            uv, _ = cam.project(points[:1])
            self.assertEqual(out[int(uv[0, 1]), int(uv[0, 0])], labels[name], name)
        self.assertEqual(set(np.unique(out)), {labels["beans"], labels["rice"]}, "every hit goes to one of them")


class GateSeesDeformables(unittest.TestCase):
    def test_a_rope_that_sinks_fails_rest_and_a_thing_with_no_position_is_named(self) -> None:
        points = np.zeros((4, 3))

        def run(steps: int) -> None:
            points[2, 2] -= 0.0079  # one node sinks 7.9 mm; the centre moves 2 mm

        sim = GateSim(
            names=["rope", "ghost"],
            positions={"rope": lambda: points.mean(axis=0)},  # the ghost has none
            on_run=run,
            scene=_deformable_sim(points, {"rope": _handle([0, 1, 2, 3])}).scene,
            state=types.SimpleNamespace(particle_q=types.SimpleNamespace(numpy=points.copy)),
        )
        server = gate_server(sim)
        with mock.patch.object(checks, "_penetration", return_value=(0.0, [])):
            result = checks.gate(server, steps=1)
        rest = next(c for c in result["checks"] if c["name"] == "rest")
        self.assertFalse(rest["ok"])
        self.assertIn("rope drifted 7.90 mm in 1 steps left alone (bar 2 mm)", rest["detail"])
        self.assertIn("not measured (no position to read): ghost", rest["detail"])
        self.assertIn("SKIP  objects", result["report"], "no names given: a line that looked at nothing")


class RestOfAHeap(unittest.TestCase):
    def test_one_jittering_grain_does_not_fail_a_heap(self) -> None:
        points = np.zeros((100, 3))
        heap = types.SimpleNamespace(
            particles=np.arange(100), triangles=np.zeros((0, 3)), tube_radius=0.0, point_radius=0.002
        )

        def run(steps: int) -> None:
            points[7, 2] += 0.01  # one grain bounces 10 mm

        sim = GateSim(
            names=["rice"],
            on_run=run,
            scene=_deformable_sim(points, {"rice": heap}).scene,
            state=types.SimpleNamespace(particle_q=types.SimpleNamespace(numpy=points.copy)),
        )

        with mock.patch.object(checks, "_penetration", return_value=(0.0, [])):
            rest = next(c for c in checks.gate(gate_server(sim), steps=1)["checks"] if c["name"] == "rest")
        self.assertTrue(rest["ok"], rest["detail"])


def _case_rest(lid_after: tuple) -> dict:
    """The gate's rest line for a case whose base stays and whose lid ends at the quaternion ``lid_after``."""
    body_q = np.zeros((3, 7))
    body_q[:, 6] = 1.0
    com = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    handles = types.SimpleNamespace(
        objects={"case": None}, statics={}, deformables={}, entity=lambda name: types.SimpleNamespace(bodies=(1, 2))
    )

    def run(steps: int) -> None:
        body_q[2, 3:7] = lid_after

    sim = GateSim(
        names=["case"],
        positions={"case": body_q[1, :3].copy},  # the base, which stays: a view, copied when read
        finite=lambda: bool(np.isfinite(body_q).all()),
        on_run=run,
        scene=types.SimpleNamespace(handles=handles),
        state=types.SimpleNamespace(body_q=types.SimpleNamespace(numpy=body_q.copy)),
        model=types.SimpleNamespace(body_com=types.SimpleNamespace(numpy=lambda: com)),
    )
    with mock.patch.object(checks, "_penetration", return_value=(0.0, [])):
        result = checks.gate(gate_server(sim), steps=1)
    return next(c for c in result["checks"] if c["name"] == "rest")


class GateSeesJointedThings(unittest.TestCase):
    def test_a_lid_that_swings_while_its_base_stays_fails_rest(self) -> None:
        rest = _case_rest((0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)))  # the lid swings 45 degrees
        self.assertFalse(rest["ok"], rest["detail"])
        self.assertIn("case drifted", rest["detail"])

    def test_a_lid_that_blows_up_while_its_base_stays_fails_rest(self) -> None:
        rest = _case_rest((np.nan,) * 4)
        self.assertFalse(rest["ok"], rest["detail"])
        self.assertIn("case drifted nan", rest["detail"])


class TakeAccounting(unittest.TestCase):
    def test_a_reset_under_the_take_does_not_stop_the_count(self) -> None:
        rec = checks.Recorder.__new__(checks.Recorder)
        rec.sim = types.SimpleNamespace(step_count=100)
        rec.steps, rec._raw, rec.last_capture, rec.every = 0, 100, -1, 4
        rec._mark = checks.reset_mark(rec.sim)
        captured: list[int] = []
        rec.capture = lambda: (rec._advance(), captured.append(rec.steps), setattr(rec, "last_capture", rec.steps))
        for step in (100, 104, 110, 2, 6, 9):  # the routine resets the scene after step 110
            rec.sim.step_count = step
            rec.maybe()
        self.assertEqual(captured, [0, 4, 10, 16], "a frame every four steps, the reset at 12 included")
        self.assertEqual(rec.steps, 19)

    def test_a_reset_then_more_steps_than_before_in_one_go_is_still_a_reset(self) -> None:
        sim = types.SimpleNamespace(step_count=100, reset_count=4)
        mark = checks.reset_mark(sim)
        sim.reset_count, sim.step_count = 5, 500  # sim.reset(); sim.run(500)
        self.assertEqual(checks.steps_since(sim, 100, mark), 500)

    def test_a_long_chain_keeps_a_sample_of_its_links(self) -> None:
        body_q = np.zeros((130, 7))
        handles = types.SimpleNamespace(
            objects={"cable": None}, statics={}, entity=lambda name: types.SimpleNamespace(bodies=tuple(range(120)))
        )
        body_q[:, 6] = 1.0
        sim = types.SimpleNamespace(
            scene=types.SimpleNamespace(handles=handles),
            state=types.SimpleNamespace(body_q=types.SimpleNamespace(numpy=lambda: body_q)),
            model=types.SimpleNamespace(body_com=types.SimpleNamespace(numpy=lambda: np.zeros((130, 3)))),
        )
        self.assertEqual(len(checks._part_positions(sim)["cable"]), checks.PARTS_KEPT)
        sim.state.body_q = None
        self.assertEqual(checks._part_positions(sim), {}, "a take is never lost for want of these")

    def test_a_jointed_thing_records_every_body(self) -> None:
        body_q = np.zeros((4, 7))
        body_q[2, :3] = (0.1, 0.2, 0.3)
        handles = types.SimpleNamespace(
            objects={"case": None, "cube": None},
            statics={},
            entity=lambda name: types.SimpleNamespace(bodies=(1, 2) if name == "case" else (), body=3),
        )
        body_q[:, 6] = 1.0  # unrotated
        com = np.zeros((4, 3))
        sim = types.SimpleNamespace(
            scene=types.SimpleNamespace(handles=handles),
            state=types.SimpleNamespace(body_q=types.SimpleNamespace(numpy=lambda: body_q)),
            model=types.SimpleNamespace(body_com=types.SimpleNamespace(numpy=lambda: com)),
        )
        self.assertEqual(
            checks._part_positions(sim), {"case": [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]], "cube": [[0.0, 0.0, 0.0]]}
        )
        com[2] = (0.1, 0.0, 0.0)  # the lid's mass 100 mm along its own x from the hinge at its origin
        body_q[2, 3:7] = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))  # swung 90 degrees about z
        np.testing.assert_allclose(checks._part_positions(sim)["case"][1], [0.1, 0.3, 0.3], atol=1e-4)


class MaskKeys(unittest.TestCase):
    def test_any_pixel_not_black_is_the_thing_and_an_empty_mask_says_so(self) -> None:
        from PIL import Image

        from robosimstudio_real2sim_agent import compare

        tmp = Path(tempfile.mkdtemp())
        mask = np.zeros((4, 4, 3), np.uint8)
        mask[1, 1] = (0, 0, 1)  # nearly black, and not black
        Image.fromarray(mask).save(tmp / "m.png")
        key = compare.parse_key(f"lid=mask:{tmp / 'm.png'}")
        self.assertEqual(int(compare.photo_mask(np.zeros((4, 4, 3), np.uint8), key).sum()), 1)
        reading = compare.compare_masks(None, np.ones((4, 4), bool))
        self.assertIn("the mask holds nothing at this size", compare.advice("lid", reading, 4, key))

    def test_a_deformable_is_sampled_for_the_take(self) -> None:
        points = np.arange(300, dtype=float).reshape(100, 3)
        sim = _deformable_sim(points, {"rope": _handle(list(range(100)))})
        sim.scene.handles.objects, sim.scene.handles.statics = {}, {}
        self.assertEqual(len(checks._part_positions(sim)["rope"]), checks.PARTS_KEPT)


class TakeProbe(unittest.TestCase):
    def _recorder(self) -> checks.Recorder:
        rec = checks.Recorder.__new__(checks.Recorder)
        rec.sim = types.SimpleNamespace(step_count=7, reset_count=0)
        rec._probed = None
        rec.also, rec.grid = {}, None  # no --also renderers
        rec.run_dir, rec.name = Path(tempfile.mkdtemp(prefix="take-probe-")), "take"
        rec.also_dir = rec.run_dir / "take.renderers"
        rec.looks, rec.encoded, rec.captured = [], [], []
        rec._look = lambda strict=False: (rec.looks.append(1), (np.zeros((540, 960, 3), np.uint8), "seg", None, {}))[1]
        rec.writer = types.SimpleNamespace(try_encode=lambda frame: rec.encoded.append(frame.shape))
        rec.capture = lambda look=None: rec.captured.append(look)
        return rec

    def test_the_probe_encodes_its_frame_once(self) -> None:
        rec = self._recorder()
        rec.probe()
        self.assertEqual(rec.encoded, [(540, 960, 3)])

        def refuse(frame) -> None:
            raise RuntimeError("ffmpeg cannot encode take.mp4")

        rec.writer.try_encode = refuse
        with self.assertRaisesRegex(RuntimeError, "cannot encode"):
            rec.probe()

    def test_frame_0_is_the_probes_unless_the_scene_may_have_moved_since(self) -> None:
        rec = self._recorder()
        rec.probe()
        rec.start()
        self.assertEqual((len(rec.looks), rec.captured[-1][1]), (1, "seg"), "rendered once")
        rec.probe()
        rec.sim.step_count += 1  # what was dropped stepped the scene as it cleaned up
        rec.start()
        self.assertIsNone(rec.captured[-1], "filmed afresh")
        rec.probe()
        rec.start(fresh=True)  # a dropped routine's cleanup may move things without a step
        self.assertIsNone(rec.captured[-1], "filmed afresh")


def _gate_line(name: str, sim: GateSim, **kwargs) -> dict:
    """One of the gate's lines, with the contact probe answering "no overlap"."""
    with mock.patch.object(checks, "_penetration", return_value=(0.0, [])):
        result = checks.gate(gate_server(sim), steps=1, **kwargs)
    return next(c for c in result["checks"] if c["name"] == name)


PHONE = CameraSpec(name="phone", width=1280, height=720, fov_y=50.0, eye=(1.0, 0.0, 1.0), look_at=(0.0, 0.0, 0.0))
"""A 16:9 camera, 1.778 wide per unit of height."""


class TheCameraLine(unittest.TestCase):
    """The camera the comparison renders through must have the clip's shape: within 2% of its aspect."""

    def _camera(self, width: int = 0, height: int = 0, **kwargs) -> dict:
        timeline = {"width": width, "height": height} if width or height else None
        return _gate_line("camera", GateSim(cameras=(PHONE,)), timeline=timeline, **kwargs)

    def test_a_clip_of_the_cameras_own_shape_passes_and_is_named(self) -> None:
        line = self._camera(1920, 1080)
        self.assertTrue(line["ok"], line["detail"])
        self.assertEqual("phone 1280x720, fov 50 deg; clip 1920x1080", line["detail"])

    def test_a_clip_within_two_percent_passes(self) -> None:
        line = self._camera(1300, 720)  # 1.6% wider
        self.assertTrue(line["ok"], line["detail"])

    def test_a_clip_three_percent_off_fails_saying_by_how_much(self) -> None:
        line = self._camera(1320, 720)  # 3.1% wider
        self.assertFalse(line["ok"])
        self.assertIn("aspect off by 3%", line["detail"])

    def test_a_portrait_clip_on_a_landscape_camera_fails(self) -> None:
        line = self._camera(1080, 1920)  # 0.5625 against 1.778: off by 68%
        self.assertFalse(line["ok"])
        self.assertIn("clip 1080x1920 -- aspect off by 68%", line["detail"])

    def test_with_no_clip_size_the_aspect_is_not_asked(self) -> None:
        for timeline in (None, {"width": 0, "height": 1080}):
            with self.subTest(timeline=timeline):
                line = _gate_line("camera", GateSim(cameras=(PHONE,)), timeline=timeline)
                self.assertTrue(line["ok"])
                self.assertNotIn("clip", line["detail"])

    def test_a_camera_the_scene_does_not_declare_fails_naming_the_ones_it_does(self) -> None:
        line = _gate_line("camera", GateSim(cameras=(PHONE,)), camera="overhead")
        self.assertFalse(line["ok"])
        self.assertEqual("no camera 'overhead' declared; the scene has ['phone']", line["detail"])


class TheRestBar(unittest.TestCase):
    """``rest`` passes a thing that drifted as far as the bar and fails one that drifted further."""

    def _rest(self, drift_m: float, settle_mm: float = 2.0) -> dict:
        where = {"cube": np.zeros(3)}
        sim = GateSim(
            names=["cube"],
            positions={"cube": lambda: where["cube"]},
            on_run=lambda steps: where.update(cube=np.array([drift_m, 0.0, 0.0])),
        )
        return _gate_line("rest", sim, settle_mm=settle_mm)

    def test_a_drift_exactly_at_the_bar_passes(self) -> None:
        line = self._rest(0.002)
        self.assertTrue(line["ok"], line["detail"])
        self.assertEqual("nothing moved more than 2.00 mm in 1 steps (bar 2 mm)", line["detail"])

    def test_a_drift_just_past_the_bar_fails(self) -> None:
        line = self._rest(0.0021)
        self.assertFalse(line["ok"])
        self.assertIn("cube drifted 2.10 mm in 1 steps left alone (bar 2 mm)", line["detail"])

    def test_the_bar_is_the_one_asked_for(self) -> None:
        self.assertTrue(self._rest(0.0021, settle_mm=2.5)["ok"])


class GridOfRenderers(unittest.TestCase):
    def test_frames_go_two_to_a_row_each_labelled(self) -> None:
        from robosimstudio_real2sim_agent import checks

        frames = [np.full((200, 300, 3), v, np.uint8) for v in (200, 120, 60)]
        grid = checks.grid_frame(frames, ["rtx", "gl", "tiled"])
        self.assertEqual(grid.shape, (200, 300, 3), "three or more go at half size: about one frame's pixels")
        self.assertEqual(tuple(grid[80, 140]), (200, 200, 200), "the first tile's own pixels past its label")
        self.assertEqual(tuple(grid[180, 280]), (16, 20, 24), "an empty fourth place")
        self.assertLess(int(grid[9, 6].max()), 60, "a dark label box at each tile's top left")
        self.assertLess(int(grid[9, 156].max()), 60)
        two = checks.grid_frame(frames[:2], ["rtx", "gl"])
        self.assertEqual(two.shape, (200, 600, 3), "two side by side at full size")


if __name__ == "__main__":
    unittest.main()
