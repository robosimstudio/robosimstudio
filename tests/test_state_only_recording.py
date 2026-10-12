import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robosimstudio.record import EpisodeRecorder
from robosimstudio.specs import RecordSpec

TIER = "fast"
"""RecordSpec.record_video=False writes no video, in a tmpdir. Gates every commit."""


class TestStateOnlyRecording(unittest.TestCase):
    def test_state_arrays_are_saved_without_opening_video_encoders(self):
        episode = SimpleNamespace(episode_id=0, seed=42, instruction="test", model_key="fixture")
        env = SimpleNamespace(
            episode=episode,
            model=SimpleNamespace(particle_count=0),
            scene=SimpleNamespace(cloths={}),
            step_count=1,
            sim_time=1 / 60,
            task=SimpleNamespace(full_name="test", scene="bench", embodiment="robot"),
        )
        with TemporaryDirectory() as directory, patch("robosimstudio.record.recorder.VideoSet") as video:
            recorder = EpisodeRecorder(directory, RecordSpec(record_video=False, privileged=False))
            recorder.begin(episode, env)
            recorder.step(
                env,
                np.array([0.1]),
                {"proprio": {"joint": [1.0]}, "cameras": {"camera": {"rgb": np.zeros((2, 2, 3))}}},
                1.0,
                {"success": True},
            )
            recorder.end(env, {"success": True})
            recorder.close()
            video.assert_not_called()
            self.assertEqual(recorder.failures, [])
            arrays = np.load(Path(directory) / "0000/steps.npz")
            np.testing.assert_equal(arrays["success"], [1])
            np.testing.assert_allclose(arrays["action"], [[0.1]])
