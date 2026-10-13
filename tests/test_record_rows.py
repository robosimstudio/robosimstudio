# SPDX-License-Identifier: Apache-2.0
"""One episode row, whichever recorder wrote it.

``EpisodeStore`` readers, ``render_offline`` and the exporters open a fleet run
and a serial run the same way, so an episode's ``meta.json`` has to carry the
same keys from both. The fleet's used to omit the timing, termination and
evidence counts the serial recorder writes, and wrote an embodiment spec raw.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from robosimstudio.record import EpisodeRecorder
from robosimstudio.record.fleet import FleetRecorder
from robosimstudio.record.store import EpisodeStore
from robosimstudio.specs import RecordSpec

TIER = "fast"
"""No GPU: both recorders over plain namespaces, in a tmpdir."""


def _episode() -> SimpleNamespace:
    return SimpleNamespace(episode_id=3, seed=7, instruction="tidy", model_key="fixture")


def _task() -> SimpleNamespace:
    return SimpleNamespace(
        full_name="tidy/bench",
        scene="bench",
        embodiment=SimpleNamespace(name="arm"),
        physics=SimpleNamespace(control_hz=50.0),
    )


class TestFleetAndSerialRowsMatch(unittest.TestCase):
    def _serial(self, directory: str) -> dict:
        episode = _episode()
        env = SimpleNamespace(
            episode=episode,
            model=SimpleNamespace(particle_count=0),
            scene=SimpleNamespace(cloths={}),
            step_count=100,
            sim_time=2.0,
            task=_task(),
        )
        recorder = EpisodeRecorder(directory, RecordSpec(record_video=False, privileged=False))
        recorder.begin(episode, env)
        recorder.step(env, np.array([0.1]), {}, 1.0, {"success": True})
        return recorder.end(env, {"success": True, "termination_reason": "goal"})

    def _fleet(self, directory: str) -> dict:
        slot = SimpleNamespace(
            world=1,
            episode=_episode(),
            task=_task(),
            parked=False,
            success=True,
            reward=1.0,
            steps=100,
            close_kind="success",
            metrics={},
            info={"termination_reason": "goal"},
            wall_s=1.0,
        )
        recorder = FleetRecorder(directory, RecordSpec(record_video=False, privileged=False), num_worlds=2)
        recorder.begin_wave([slot])
        recorder.finish_world(1, slot)
        recorder.close()
        self.assertEqual({}, recorder.failures)
        return json.loads((Path(directory) / "0003" / "meta.json").read_text())

    def test_the_same_keys_and_the_embodiment_by_name(self):
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            serial, fleet = self._serial(a), self._fleet(b)
        self.assertEqual(set(serial), set(fleet) - {"world"})
        self.assertEqual("arm", fleet["embodiment"])
        self.assertEqual("goal", fleet["termination_reason"])
        self.assertAlmostEqual(2.0, fleet["sim_s"])
        self.assertAlmostEqual(2.0, fleet["realtime_factor"])


class TestAStoreNotYetWritten(unittest.TestCase):
    def test_reads_as_empty_rather_than_raising(self):
        with TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "missing")
            self.assertEqual([], store.episode_ids())
            self.assertEqual([], store.results())
            self.assertEqual({}, store.run_meta)


if __name__ == "__main__":
    unittest.main()
