# SPDX-License-Identifier: Apache-2.0
"""The check gate reads penetration through the snapshot's public rows."""

import unittest
from types import SimpleNamespace

import numpy as np

TIER = "fast"


class _Array:
    def __init__(self, values):
        self._values = np.asarray(values)

    def numpy(self):
        return self._values


class _Probe:
    def __init__(self, rows):
        self.rows = rows
        self.updated = 0

    def update(self):
        self.updated += 1

    def penetrations(self):
        return self.rows


def _env(probe):
    model = SimpleNamespace(shape_body=_Array([0, 1, -1]), shape_label=["arena/a", "arena/cube_geom", "arena/floor"])
    handles = SimpleNamespace(objects={"cube": SimpleNamespace(body=1), "loose": SimpleNamespace()})
    return SimpleNamespace(contacts=probe, model=model, handles=handles)


class TestPenetrationRows(unittest.TestCase):
    def test_rows_are_named_worst_first(self):
        from robosimstudio.scene.snapshot import penetration_rows

        probe = _Probe([(1, 2, 0.004), (0, 1, 0.001)])
        rows = penetration_rows(_env(probe))
        self.assertEqual(probe.updated, 1)
        self.assertEqual(rows[0]["depth_mm"], 4.0)
        self.assertEqual(rows[0]["between"], ["cube", "floor"])
        self.assertEqual(rows[1]["between"], ["a", "cube"])

    def test_no_probe_is_no_rows(self):
        from robosimstudio.scene.snapshot import penetration_rows

        self.assertEqual(penetration_rows(_env(None)), [])

    def test_checks_reads_the_public_rows(self):
        from robosimstudio_real2sim_agent.checks import _penetration

        self.assertEqual(_penetration(_env(_Probe([(1, 2, 0.004)]))), (4.0, ["cube and floor"]))
        self.assertEqual(_penetration(_env(_Probe([]))), (0.0, []))
        self.assertEqual(_penetration(_env(None)), (0.0, []))


if __name__ == "__main__":
    unittest.main()
