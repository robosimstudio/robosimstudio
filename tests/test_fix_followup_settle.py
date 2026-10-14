# SPDX-License-Identifier: Apache-2.0
"""Particle things share one settle per pipeline through ``ParticleMixin.join_settle``,
and the studio's re-settle re-baselines every thing on that record, not only the cloth.

No GPU: stand-in things and a stand-in pipeline.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

TIER = "fast"
"""Stand-ins only: no GPU."""


def _thing():
    from robosimstudio.domains.capabilities import ParticleMixin

    class _Stub(ParticleMixin):
        def __init__(self) -> None:
            self.settles = 0
            self.snapshots = 0

        def settle(self, pipeline) -> None:
            del pipeline
            self.settles += 1

        def snapshot_rest(self, pipeline) -> None:
            del pipeline
            self.snapshots += 1

    return _Stub()


class TestJoinSettle(unittest.TestCase):
    def test_a_run_settles_once_and_every_thing_snapshots_after_each_join(self) -> None:
        pipeline = SimpleNamespace()
        sheet, block, cable = _thing(), _thing(), _thing()
        sheet.join_settle(pipeline, (10, 4))
        block.join_settle(pipeline, (10, 4))
        cable.join_settle(pipeline)
        self.assertEqual((1, 0, 0), (sheet.settles, block.settles, cable.settles))
        self.assertEqual({(10, 4)}, pipeline.particle_settles["runs"])
        self.assertEqual([sheet, block, cable], pipeline.particle_settles["things"])
        self.assertEqual((3, 2, 1), (sheet.snapshots, block.snapshots, cable.snapshots))

    def test_a_different_run_settles_again(self) -> None:
        pipeline = SimpleNamespace()
        a, b = _thing(), _thing()
        a.join_settle(pipeline, (10, 4))
        b.join_settle(pipeline, (20, 4))
        self.assertEqual((1, 1), (a.settles, b.settles))


class TestStudioResettle(unittest.TestCase):
    def _panel(self, pipeline, cloth):
        from robosimstudio_studio.tuning.panels import PanelReadings

        panel = PanelReadings(SimpleNamespace(env=SimpleNamespace(pipeline=pipeline)))
        panel._note = SimpleNamespace(value="")
        panel.resettle_requested = True
        return panel

    def test_every_thing_on_the_record_is_rebaselined(self) -> None:
        cloth, cable = _thing(), _thing()
        pipeline = SimpleNamespace(particle_settles={"runs": set(), "things": [cloth, cable]})
        panel = self._panel(pipeline, cloth)
        with (
            mock.patch.object(type(panel), "_busy", return_value=""),
            mock.patch.object(type(panel), "_cloth", return_value=cloth),
            mock.patch.object(type(panel), "refresh"),
        ):
            panel.drain()
        self.assertEqual(1, cloth.settles)
        self.assertEqual((1, 1), (cloth.snapshots, cable.snapshots))

    def test_without_a_record_the_cloth_alone_is_rebaselined(self) -> None:
        cloth = _thing()
        panel = self._panel(SimpleNamespace(), cloth)
        with (
            mock.patch.object(type(panel), "_busy", return_value=""),
            mock.patch.object(type(panel), "_cloth", return_value=cloth),
            mock.patch.object(type(panel), "refresh"),
        ):
            panel.drain()
        self.assertEqual((1, 1), (cloth.settles, cloth.snapshots))


if __name__ == "__main__":
    unittest.main()
