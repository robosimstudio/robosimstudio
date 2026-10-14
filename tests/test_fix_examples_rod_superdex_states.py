# SPDX-License-Identifier: Apache-2.0
"""The SuperDex rod publishes what the capsule chain publishes, and its
``INACTIVE`` flags are ``add_particle``'s default with ``ACTIVE`` cleared.

No GPU: the entity is a stand-in built without a scene, read off a snapshot
of numpy node positions.
"""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

import numpy as np

TIER = "fast"
"""numpy and a stand-in entity: no GPU."""


def _rod(n_nodes: int):
    from robosimstudio.domains.rod.superdex import EntityRodSuperdex

    rod = EntityRodSuperdex.__new__(EntityRodSuperdex)
    rod.particles = np.arange(n_nodes, dtype=np.int32)
    rod._pins = {}
    rod.bridge = None
    return rod


class TestPublishedStates(unittest.TestCase):
    def test_a_readable_snapshot_publishes_length_and_segments(self) -> None:
        rod = _rod(5)
        q = np.zeros((5, 3))
        q[:, 0] = np.linspace(0.0, 0.4, 5)
        got = rod.published_states(SimpleNamespace(particle_q=q))
        self.assertEqual({"held", "anchor_held", "length_m", "n_segments"}, set(got))
        self.assertAlmostEqual(got["length_m"], 0.4)
        self.assertEqual(got["n_segments"], 4.0)

    def test_no_snapshot_publishes_the_flags_only(self) -> None:
        self.assertEqual({"held": 0.0, "anchor_held": 0.0}, _rod(5).published_states(None))


class _Thing:
    """Another particle thing on the pipeline's shared record: counts its snapshots."""

    def __init__(self) -> None:
        self.snapshots = 0

    def snapshot_rest(self, pipeline) -> None:
        del pipeline
        self.snapshots += 1


class TestSceneReadyJoinsTheSharedRecord(unittest.TestCase):
    def test_the_rod_snapshots_every_thing_and_is_snapshotted_after_a_later_settle(self) -> None:
        from unittest import mock

        from robosimstudio.domains.rod import superdex

        sheet = _Thing()
        pipeline = SimpleNamespace(particle_settles={"runs": {(10, 4)}, "things": [sheet]})
        rod = _rod(3)
        rod.spec = SimpleNamespace(name="cable")
        with (
            mock.patch.object(superdex, "host_bridge", return_value=None),
            mock.patch.object(type(rod), "snapshot_rest", autospec=True) as snap,
        ):
            rod.on_scene_ready(pipeline)
            self.assertEqual([sheet, rod], pipeline.particle_settles["things"])
            self.assertEqual(1, sheet.snapshots, "a rod joining did not re-snapshot the sheet")
            snap.assert_called_once_with(rod, pipeline)
            # A later settle walks the record, so the rod's rest is taken after it too.
            for thing in pipeline.particle_settles["things"]:
                thing.snapshot_rest(pipeline)
            self.assertEqual(2, snap.call_count)

    def test_a_rod_alone_opens_the_record(self) -> None:
        from unittest import mock

        from robosimstudio.domains.rod import superdex

        pipeline = SimpleNamespace()
        rod = _rod(3)
        rod.spec = SimpleNamespace(name="cable")
        with (
            mock.patch.object(superdex, "host_bridge", return_value=None),
            mock.patch.object(type(rod), "snapshot_rest", autospec=True),
        ):
            rod.on_scene_ready(pipeline)
        self.assertEqual({"runs": set(), "things": [rod]}, pipeline.particle_settles)


class TestInactiveFlags(unittest.TestCase):
    def test_inactive_is_the_default_without_active(self) -> None:
        import newton
        from robosimstudio.domains.rod.superdex import INACTIVE

        default = int(inspect.signature(newton.ModelBuilder.add_particle).parameters["flags"].default)
        self.assertEqual(INACTIVE, default & ~int(newton.ParticleFlags.ACTIVE))
        self.assertFalse(INACTIVE & int(newton.ParticleFlags.ACTIVE))


if __name__ == "__main__":
    unittest.main()
