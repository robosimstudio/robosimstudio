# SPDX-License-Identifier: Apache-2.0
"""The stills the page writes: parsed sizes and vectors, the pair beside a reference, the camera and the browser's frame."""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from robosimstudio_real2sim_agent import stills

TIER = "fast"
"""No GPU: arrays stand in for renders, and a namespace for the browser."""


class TestStills(unittest.TestCase):
    def test_parsing(self) -> None:
        self.assertEqual(stills.parse_vec("1, 2.5,-3"), (1.0, 2.5, -3.0))
        with self.assertRaises(ValueError):
            stills.parse_vec("1,2")
        self.assertEqual(stills.parse_size("", (4, 3)), (4, 3))
        self.assertEqual(stills.parse_size("320X180", (4, 3)), (320, 180))

    def test_to_uint8(self) -> None:
        self.assertEqual(stills.to_uint8(np.full((2, 2, 3), 0.5)).tolist()[0][0], [127, 127, 127])
        kept = np.full((2, 2, 3), 200, dtype=np.uint8)
        self.assertIs(stills.to_uint8(kept).dtype.type, np.uint8)
        self.assertEqual(
            int(stills.to_uint8(np.full((1, 1, 3), 300.0))[0, 0, 0]),
            255,
            "clipped, not wrapped",
        )

    def test_the_reference_stands_beside_the_render_at_its_height(self) -> None:
        from PIL import Image

        tmp = Path(tempfile.mkdtemp(prefix="pair-"))
        ref = tmp / "ref.png"
        Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(ref)
        render = np.full((8, 8, 3), 255, dtype=np.uint8)
        pair = stills.side_by_side(render, str(ref))
        self.assertEqual(pair.shape, (8, 8 + 12, 3), "the reference is scaled to the render's height")
        self.assertEqual(int(pair[0, 0, 0]), 255)
        image, label = stills.load_reference(str(ref), width=42)
        self.assertEqual((image.shape[1], label), (42, "ref.png"))
        out = stills.save_png(pair, tmp / "deep" / "pair.png")
        self.assertTrue(out.exists(), "save_png makes the directory")


class TestTheCamera(unittest.TestCase):
    def test_eye_without_look_at_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            stills.camera_for(None, None, eye="1,0,1")
        with self.assertRaises(ValueError):
            stills.camera_for(None, None, look_at="0,0,0")
        with self.assertRaises(ValueError):
            stills.camera_for(None, None, name="top", eye="1,0,1", look_at="0,0,0")


class TestTheBrowserStill(unittest.TestCase):
    def test_the_frame_is_asked_of_a_client_as_opaque_rgb(self) -> None:
        asked = {}

        def get_render(height, width, transport_format="jpeg", timeout=None):
            asked.update(format=transport_format, timeout=timeout)
            return np.full((height, width, 3), 200, np.uint8)

        client = types.SimpleNamespace(get_render=get_render)
        server = types.SimpleNamespace(get_clients=lambda: {0: client})
        rgb = stills.browser_still(server, "64x32")
        self.assertEqual(rgb.shape, (32, 64, 3))
        self.assertEqual(asked["format"], "jpeg")
        self.assertIsNotNone(asked["timeout"])
        self.assertIsNone(stills.browser_still(types.SimpleNamespace(get_clients=dict)))

    def test_a_browser_that_never_answers_times_out(self) -> None:
        def get_render(height, width, transport_format="jpeg", timeout=None):
            raise TimeoutError

        server = types.SimpleNamespace(get_clients=lambda: {0: types.SimpleNamespace(get_render=get_render)})
        with self.assertRaisesRegex(TimeoutError, "no frame"):
            stills.browser_still(server)


if __name__ == "__main__":
    unittest.main()
