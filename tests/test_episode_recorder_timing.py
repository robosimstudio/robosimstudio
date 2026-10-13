# SPDX-License-Identifier: Apache-2.0
"""When the recorder emits a frame, and when it starts.

Sampling against simulation time rather than wall clock is what keeps a slow
render or a slow plan from changing an mp4's playback speed; ``start_on_motion``
is what keeps a perception model's warm-up out of the front of the video.

``unittest`` cases, because ``python -m unittest`` is the runner
(``scripts/testrun.py``). This module was once two bare functions taking a
``tmp_path`` argument, which that runner does not collect: it reported "Ran 0
tests" and exit code 5, and for as long as it did the two claims below were
held by nothing.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robosimstudio.record.recorder import EpisodeRecorder
from robosimstudio.specs import RecordSpec

TIER = "fast"
"""No GPU: the recorder's timing arithmetic, read directly."""


class TestVideoIsSampledOnTheSimulationClock(unittest.TestCase):
    def setUp(self) -> None:
        root = tempfile.TemporaryDirectory(prefix="recorder-")
        self.addCleanup(root.cleanup)
        self.recorder = EpisodeRecorder(Path(root.name), RecordSpec(video_fps=30))

    def test_video_sampling_uses_simulation_time(self) -> None:
        """A 60 Hz controller into a 30 fps container: every other step, not every step."""
        emitted = [self.recorder._video_frame_due(step / 60.0) for step in range(60)]
        self.assertEqual(sum(emitted), 30)

    def _frames(self, hz: int, seconds: float = 10.0) -> int:
        return sum(self.recorder._video_frame_due(step / hz) for step in range(int(hz * seconds)))

    def test_a_controller_rate_not_a_multiple_of_fps_keeps_the_frame_rate(self) -> None:
        """50 Hz into 30 fps: 300 frames in ten seconds, not 250 (a video 1.2x fast)."""
        self.assertAlmostEqual(self._frames(50), 300, delta=1)

    def test_40_hz_into_30_fps(self) -> None:
        self.assertAlmostEqual(self._frames(40), 300, delta=1)

    def test_a_long_gap_emits_one_frame_not_a_burst(self) -> None:
        """Skipping many deadlines (``every`` > 1, a slow tool) yields one frame and resumes on the grid."""
        self.assertTrue(self.recorder._video_frame_due(0.0))
        self.assertTrue(self.recorder._video_frame_due(1.0))
        self.assertFalse(self.recorder._video_frame_due(1.0 + 0.001))
        self.assertTrue(self.recorder._video_frame_due(1.0 + 1.0 / 30.0))

    def test_wall_clock_pause_cannot_add_video_frames(self) -> None:
        """Asked twice at one simulation time -- a plan that took a while -- it answers once."""
        self.assertTrue(self.recorder._video_frame_due(1.0))
        self.assertFalse(self.recorder._video_frame_due(1.0))
        self.assertTrue(self.recorder._video_frame_due(1.0 + 1.0 / 30.0))


if __name__ == "__main__":
    unittest.main()
