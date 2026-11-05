# SPDX-License-Identifier: Apache-2.0
"""The camera fit's whole search, on the set it exists to refuse: points mirrored in one world axis.

A mirrored set fits as well from behind the camera as a true set does from in
front, and only a search started from cameras all round the points -- 26
directions, two distances, three fields of view -- meets that fit from behind
and so can say the set is mirrored. :mod:`tests.test_real2sim_calib` cuts the
search to four directions to stay fast; this is the one place it runs whole.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.core.cameras import CameraSpec
from robosimstudio_real2sim_agent import calib

TIER = "medium"
"""Needs no GPU, but 156 least-squares solves take about twenty seconds: too slow to gate every commit."""


class TestCameraFit(unittest.TestCase):
    CAM = CameraSpec(name="c", width=1280, height=720, fov_y=48, eye=(1.9, -0.7, 1.6), look_at=(0.6, 0.1, 0.75))

    def _points(self, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(3)
        world = np.c_[rng.uniform(0.2, 1.0, n), rng.uniform(-0.6, 0.6, n), rng.uniform(0.75, 1.1, n)]
        return world, self.CAM.project(world)[0]

    def test_a_mirrored_set_is_refused(self) -> None:
        world, pixels = self._points()
        with self.assertRaisesRegex(ValueError, "mirror"):
            calib.fit_camera(world * np.array([1.0, -1.0, 1.0]), pixels, 1280, 720)


if __name__ == "__main__":
    unittest.main()
