# SPDX-License-Identifier: Apache-2.0
"""Fitting a camera to points read off a still: the camera comes back, a mirror is refused, a misread point stands out.

The fit starts from cameras all round the points -- 26 directions, two
distances and three fields of view, 156 least-squares solves -- because nothing
about it is convex. Every test here but the one that needs the whole search
asks for four directions (or names the field of view), which finds the same
camera in a tenth of the time; the search itself, on the mirrored set it exists
to refuse, is :mod:`tests.test_real2sim_calib_search`'s.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robosimstudio.core.cameras import CameraSpec
from robosimstudio_real2sim_agent import calib

TIER = "fast"
"""No GPU: points a known camera projected."""


FEW_STARTS = 4
"""Directions the fit starts from here, of the 26 it tries: enough to find these cameras."""


def few_starts() -> mock._patch:
    """The fit's search, cut to :data:`FEW_STARTS` directions round the points."""
    sphere = calib._sphere
    return mock.patch.object(calib, "_sphere", lambda n: sphere(min(n, FEW_STARTS)))


class TestCameraFit(unittest.TestCase):
    CAM = CameraSpec(name="c", width=1280, height=720, fov_y=48, eye=(1.9, -0.7, 1.6), look_at=(0.6, 0.1, 0.75))

    def setUp(self) -> None:
        self.enterContext(few_starts())

    def _points(self, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(3)
        world = np.c_[rng.uniform(0.2, 1.0, n), rng.uniform(-0.6, 0.6, n), rng.uniform(0.75, 1.1, n)]
        return world, self.CAM.project(world)[0]

    def test_the_camera_comes_back(self) -> None:
        world, pixels = self._points()
        fit = calib.fit_camera(world, pixels, 1280, 720)
        self.assertLess(fit["rms_px"], 0.05)
        self.assertAlmostEqual(fit["fov_y"], 48.0, places=1)
        spec = CameraSpec(name="fit", **fit["spec"])
        self.assertLess(np.abs(spec.project(world)[0] - pixels).max(), 0.5, "the printed keywords make that camera")

    def test_a_mirrored_set_is_refused_with_its_field_of_view_given(self) -> None:
        """Named, the field of view leaves the search no path through a camera from behind: it must look there."""
        world, pixels = self._points()
        with self.assertRaisesRegex(ValueError, "mirror"):
            calib.fit_camera(world * np.array([1.0, -1.0, 1.0]), pixels, 1280, 720, fov_y=48.0)

    def test_the_camera_comes_back_with_its_field_of_view_given(self) -> None:
        world, pixels = self._points()
        fit = calib.fit_camera(world, pixels, 1280, 720, fov_y=48.0)
        self.assertLess(fit["rms_px"], 0.05)
        self.assertEqual(0, fit["behind"])

    def test_a_poor_fit_from_behind_does_not_stand_in_for_looking_there(self) -> None:
        """A start that drifts behind the points and stops far from any fit says nothing about a mirror."""
        from scipy import optimize

        solve, calls = optimize.least_squares, []

        def first_start_drifts_behind(fun, x0, **kwargs):
            calls.append(1)
            if len(calls) > 1:
                return solve(fun, x0, **kwargs)
            rot = calib._rodrigues(x0[1:4])
            eye = -rot.T @ x0[4:7]
            turned = np.diag([-1.0, 1.0, -1.0]) @ rot  # half a turn about its own y: every point behind it
            x = np.concatenate([[x0[0]], calib._rotvec(turned), -turned @ eye])
            return SimpleNamespace(x=x, fun=fun(x))  # stopped there, nowhere near a fit

        world, pixels = self._points()
        with (
            mock.patch.object(optimize, "least_squares", first_start_drifts_behind),
            self.assertRaisesRegex(ValueError, "mirror"),
        ):
            calib.fit_camera(world * np.array([1.0, -1.0, 1.0]), pixels, 1280, 720, fov_y=48.0)

    def test_a_hand_clicked_set_pays_for_no_search_from_behind(self) -> None:
        """A pixel or two of clicking noise is a fit some pixels out, not a mirror: one search, not two."""
        from scipy import optimize

        solve, calls = optimize.least_squares, []

        def counted(fun, x0, **kwargs):
            calls.append(1)
            return solve(fun, x0, **kwargs)

        world, pixels = self._points(12)
        clicked = pixels + np.random.default_rng(5).normal(0.0, 2.0, pixels.shape)
        with mock.patch.object(optimize, "least_squares", counted):
            fit = calib.fit_camera(world, clicked, 1280, 720, fov_y=48.0)
        self.assertGreater(fit["rms_px"], 1.0, "a fit poor enough that the old rule looked from behind")
        self.assertEqual(FEW_STARTS * 2, len(calls), "one start per direction and distance, from in front")

    def test_a_mirrored_set_stops_looking_from_behind_at_the_first_fit_there(self) -> None:
        """One fit from behind settles the verdict; the rest of the round from behind would cost and say nothing."""
        from scipy import optimize

        solve, calls = optimize.least_squares, []

        def counted(fun, x0, **kwargs):
            calls.append(1)
            return solve(fun, x0, **kwargs)

        world, pixels = self._points()
        with mock.patch.object(optimize, "least_squares", counted), self.assertRaisesRegex(ValueError, "mirror"):
            calib.fit_camera(world * np.array([1.0, -1.0, 1.0]), pixels, 1280, 720, fov_y=48.0)
        self.assertLess(len(calls), FEW_STARTS * 2 + FEW_STARTS, "a whole round from behind was searched")

    def test_a_point_read_off_wrong_stands_out(self) -> None:
        world, pixels = self._points(12)
        pixels = pixels.copy()
        pixels[4] += (40.0, -25.0)
        errors = calib.fit_camera(world, pixels, 1280, 720)["errors_px"]
        self.assertEqual(int(np.argmax(errors)), 4)

    def test_a_point_behind_the_origin_parses_on_the_command_line(self) -> None:
        from robosimstudio_real2sim_agent import ctl

        argv = ctl._join_points(["camera", "fit", "--point", "-0.3,0.2,0.75=812,455", "--size", "640x480"])
        args = ctl.build_parser().parse_args(argv)
        self.assertEqual(args.point, ["-0.3,0.2,0.75=812,455"])
        kept = ctl._join_points(["camera", "fit", "--point", "--ref", "still.png"])
        self.assertEqual(kept[2:], ["--point", "--ref", "still.png"], "an option after --point is left an option")

    def test_parse_point(self) -> None:
        self.assertEqual(calib.parse_point("0.6,0.3,0.75=812,455"), ((0.6, 0.3, 0.75), (812.0, 455.0)))
        with self.assertRaises(ValueError):
            calib.parse_point("0.6,0.3=812,455")


if __name__ == "__main__":
    unittest.main()
