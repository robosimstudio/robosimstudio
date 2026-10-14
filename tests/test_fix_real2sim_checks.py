"""The compare score, the take's strip, and the gate's builds line, held to what they say they measure."""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from robosimstudio_real2sim_agent import checks, compare, frames

TIER = "fast"


def _photo(w: int = 640, h: int = 360) -> np.ndarray:
    """A beige table with a red box and a grey bolt."""
    photo = np.full((h, w, 3), (190, 170, 140), np.uint8)
    photo[80:200, 100:300] = (210, 30, 20)
    photo[150:170, 400:420] = (180, 182, 188)
    return photo


def _render(w: int = 640, h: int = 360) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """The box a little off, the bolt nearly on."""
    rgb = np.full((h, w, 3), (60, 60, 60), np.uint8)
    seg = np.zeros((h, w), np.int32)
    rgb[90:186, 140:300] = (220, 40, 30)
    seg[90:186, 140:300] = 2
    rgb[150:170, 402:422] = (170, 170, 175)
    seg[150:170, 402:422] = 3
    return rgb, seg, {1: "table", 2: "case", 3: "bolt"}


class TestScore(unittest.TestCase):
    def test_a_thing_gone_from_the_render_lowers_the_score(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "photo.png"
            Image.fromarray(_photo()).save(path)
            keys = [compare.parse_key("case=red"), compare.parse_key("bolt=@410,160")]
            rgb, seg, names = _render()
            both = checks.compare_still(rgb, seg, names, path, keys)
            gone = seg.copy()
            gone[gone == 2] = 0  # the case off camera
            without = checks.compare_still(rgb, gone, names, path, keys)
            self.assertIsNone(without["objects"]["case"]["iou"])
            self.assertLess(without["score"], both["score"], "a missing thing counts as no overlap")
            unlabelled = checks.compare_still(rgb, seg, {1: "table", 3: "bolt"}, path, keys)
            self.assertLess(unlabelled["score"], both["score"], "so does a name the scene has no label for")


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

    def test_a_thing_missing_from_the_take_scores_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, ref = self._take(Path(tmp), {"1": "table", "3": "other"})
            result = checks.compare_take(run, "take", frames.read_timeline(ref), [compare.parse_key("case=red")], ref)
            self.assertEqual(result["score"], 0.0)
            self.assertTrue(all(a["objects"]["case"]["photo"] is not None for a in result["anchors"]))

    def test_the_strip_makes_its_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, ref = self._take(Path(tmp), {"1": "table", "2": "case"})
            result = checks.compare_take(run, "take", frames.read_timeline(ref), [compare.parse_key("case=red")], ref)
            self.assertGreater(result["score"], 0.5)
            strip = checks.write_strip(Path(tmp) / "new" / "dir" / "x.png", result["tiles"], tile_width=64)
            self.assertTrue(Path(strip).exists())


class _Sim:
    cameras = ()

    def state_is_finite(self) -> bool:
        return True

    def names(self) -> list[str]:
        return []

    def run(self, steps: int) -> None:
        pass

    def sync(self) -> None:
        pass


class TestGate(unittest.TestCase):
    def test_a_failed_routine_is_not_a_failed_build(self) -> None:
        server = types.SimpleNamespace(sim=_Sim(), last_error="Traceback\nRuntimeError: routine broke", build_error="")
        builds = checks.gate(server, steps=1)["checks"][0]
        self.assertEqual(builds["name"], "builds")
        self.assertTrue(builds["ok"], builds["detail"])
        server.build_error = "Traceback\nSyntaxError: bad scene"
        builds = checks.gate(server, steps=1)["checks"][0]
        self.assertFalse(builds["ok"])
        self.assertEqual(builds["detail"], "SyntaxError: bad scene")


if __name__ == "__main__":
    unittest.main()
