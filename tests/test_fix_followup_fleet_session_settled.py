"""The studio's fleet page takes each world's start pose from the fleet, as the runner does."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

TIER = "fast"


class _Snapshot:
    def body(self, world):
        # A pose that differs from the fleet's settled one, so a session that
        # read the snapshot itself would be caught.
        return np.full((2, 7), -1.0, dtype=np.float32)


class _Fleet:
    def __init__(self, num_worlds):
        self.settled = [np.full((2, 7), float(w), dtype=np.float32) for w in range(num_worlds)]
        self.contexts = []

    def reset_wave(self, episodes):
        self.episodes = list(episodes)

    def snapshot(self):
        return _Snapshot()

    def settled_body_q(self, world):
        return self.settled[world]

    def context(self, world, snapshot, *, settled_body_q):
        self.contexts.append((world, settled_body_q))
        return (world, settled_body_q)


class _Evaluator:
    def reset(self, context):
        self.context = context


class _Policy:
    def reset_wave(self, slots):
        self.slots = slots


class TestFleetSessionSettled(unittest.TestCase):
    def test_reset_wave_reads_the_fleets_settled_pose(self):
        from robosimstudio_studio import fleet_session

        session = object.__new__(fleet_session.FleetSession)
        session.episode0 = 0
        session.num_worlds = 3
        session.task = object()
        session.fleet = _Fleet(3)
        session.policy = _Policy()
        session._evaluator = _Evaluator
        with mock.patch.object(fleet_session, "plan_waves", return_value=[[0, 1]]):
            session.reset_wave()

        self.assertEqual(len(session.slots), 3)
        for slot in session.slots:
            self.assertIs(slot.settled_body_q, session.fleet.settled[slot.world])
            self.assertIs(slot.evaluator.context[1], session.fleet.settled[slot.world])


if __name__ == "__main__":
    unittest.main()
