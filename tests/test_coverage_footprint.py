# SPDX-License-Identifier: Apache-2.0
"""``covers`` over a cloth, which was registered and unreachable.

The atom declares three pairs — ``cloth+rigid``, ``cloth+cloth``,
``rigid+rigid`` — and only the last one could ever run. ``footprint`` asked
``snapshot.particles`` for a ``sets`` attribute, but that is a **dict of probes
keyed by deformable**, not a probe; ``getattr(dict, "sets", ())`` is ``()``
every time, so every cloth fell through to the rigid path and raised a
``KeyError`` from ``handles.entity``.

Nothing caught it because no shipped document scored a `covers` with a cloth
operand — the two registrations were calibrated prose with no caller. This test
is the caller.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""Stubs. No GPU, no scene."""


class _Probe:
    """A ``ParticleProbe``'s two-method surface, as ``footprint`` uses it."""

    def __init__(self, points, sets):
        self._points = np.asarray(points, dtype=np.float64)
        self.sets = dict(sets)

    def points(self, name=None):
        if name is None:
            return self._points
        return self._points[self.sets[name]]


class _Handles:
    def __init__(self, objects=(), statics=()):
        self.objects, self.statics = dict.fromkeys(objects), dict.fromkeys(statics)

    def entity(self, name):
        raise KeyError(f"no entity {name!r}; objects {sorted(self.objects)}, statics {sorted(self.statics)}")


class _Ctx:
    def __init__(self, particles, handles):
        self.particles, self.handles = particles, handles


class _Snapshot:
    def __init__(self, ctx):
        self.ctx = ctx


def _sheet(n=6, side=0.2, origin=(0.0, 0.0), z=0.0):
    xs = np.linspace(0, side, n) + origin[0]
    ys = np.linspace(0, side, n) + origin[1]
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)


class TestAClothHasAFootprint(unittest.TestCase):
    def _snap(self, **deformables):
        probes = {n: _Probe(p, {n: np.arange(len(p))}) for n, p in deformables.items()}
        return _Snapshot(_Ctx(probes, _Handles(statics=("board",))))

    def test_a_cloth_resolves_through_its_own_probe(self):
        """The regression. Before the fix this raised from ``handles.entity``."""
        from robosimstudio.assetlib.vocabulary.coverage import footprint

        points = _sheet()
        got = footprint(self._snap(towel=points), "towel")
        np.testing.assert_allclose(got, points)

    def test_a_probe_with_no_set_of_its_own_name_still_answers(self):
        """``points(None)`` is every particle it owns, which is the same answer
        for a sheet that declares no subsets."""
        from robosimstudio.assetlib.vocabulary.coverage import footprint

        points = _sheet()
        snap = _Snapshot(_Ctx({"towel": _Probe(points, {})}, _Handles(statics=("board",))))
        np.testing.assert_allclose(footprint(snap, "towel"), points)

    def test_a_name_that_is_no_deformable_still_reaches_the_handles(self):
        """The rigid path is unchanged: a miss must still be the handles' error."""
        from robosimstudio.assetlib.vocabulary.coverage import footprint

        with self.assertRaises(KeyError):
            footprint(self._snap(towel=_sheet()), "board")

    def test_covers_scores_a_cloth_over_a_cloth(self):
        """End to end through the registered handler, not just the helper."""
        from robosimstudio.assetlib.vocabulary.coverage import _covers

        over = _sheet(origin=(0.0, 0.0))
        under = _sheet(origin=(0.0, 0.0))
        same = _covers(self._snap(towel=over, mat=under), "towel", "mat")
        self.assertGreater(float(same["fraction"]), 0.9, "a sheet exactly over another covers it")

        clear = _sheet(origin=(5.0, 5.0))
        apart = _covers(self._snap(towel=over, mat=clear), "towel", "mat")
        self.assertLess(float(apart["fraction"]), 0.05, "and covers nothing when it is elsewhere")


if __name__ == "__main__":
    unittest.main()
