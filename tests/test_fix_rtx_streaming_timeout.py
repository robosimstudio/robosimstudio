# SPDX-License-Identifier: Apache-2.0
"""The RTX backend bounds a first frame's wait on a stuck streaming client, unless the caller chose otherwise."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from robosimstudio.solvers.render import rtx

TIER = "fast"


class StreamingTimeoutTest(unittest.TestCase):
    def test_sets_the_timeout_when_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(rtx.STREAMING_TIMEOUT_ENV, None)
            rtx.bound_first_frame_wait()
            self.assertEqual(os.environ[rtx.STREAMING_TIMEOUT_ENV], str(rtx.STREAMING_TIMEOUT_S))

    def test_a_callers_choice_wins(self) -> None:
        with mock.patch.dict(os.environ, {rtx.STREAMING_TIMEOUT_ENV: "1800"}):
            rtx.bound_first_frame_wait()
            self.assertEqual(os.environ[rtx.STREAMING_TIMEOUT_ENV], "1800")

    def test_the_variable_names_the_setting(self) -> None:
        # OVRTX maps OVRTX_<path with "/" written "_"> onto a Carbonite setting; a
        # typo here is a variable nothing reads, and the 30-minute wait comes back.
        setting = "/rtx-transient/gpu/streamingStatusManager/timeoutS"
        self.assertEqual(rtx.STREAMING_TIMEOUT_ENV, "OVRTX_" + setting.lstrip("/").replace("/", "_"))

    def test_the_bound_is_far_below_ovrtxs_own(self) -> None:
        # OVRTX's own bound is the 1,800 s the stalled builds waited.
        self.assertLess(rtx.STREAMING_TIMEOUT_S, 1800 // 4)


if __name__ == "__main__":
    unittest.main()
