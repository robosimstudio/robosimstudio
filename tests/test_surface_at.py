# SPDX-License-Identifier: Apache-2.0
"""The work-surface frame on the public API, and a Ref that reads in it.

Constructors take ``at=`` in the cell's work-surface frame; ``Ref.position``
stays world because that is what the solver holds. These tests are the
arithmetic that sits between the two -- a name, a spec, a round-trip -- so
they need no model. The compiler honouring the same numbers on a live body is
:mod:`tests.test_surface_at_cpu`.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""No GPU. Gates every commit."""


class TestSurfaceFrameHelpers(unittest.TestCase):
    def test_void_is_the_world_frame(self) -> None:
        import robosimstudio as rss

        np.testing.assert_allclose(rss.surface_frame("void"), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        np.testing.assert_allclose(rss.surface_frame(None), rss.surface_frame("void"))
        np.testing.assert_allclose(rss.surface_frame(""), rss.surface_frame(None))

    def test_bench_075_origin_is_the_middle_of_the_top(self) -> None:
        import robosimstudio as rss

        np.testing.assert_allclose(rss.surface_frame("bench_075")[:3], (0.6, 0.0, 0.75))
        spec = rss.get_cell("bench_075")
        np.testing.assert_allclose(rss.surface_frame("bench_075"), rss.surface_frame(spec))

    def test_in_surface_and_from_surface_round_trip(self) -> None:
        import robosimstudio as rss

        posed = (0.05, -0.10, 0.20, 0.0, 0.0, 0.0, 1.0)
        world = rss.in_surface("bench_075", posed)
        np.testing.assert_allclose(world[:3], (0.65, -0.10, 0.95))
        back = rss.from_surface("bench_075", world)
        np.testing.assert_allclose(back, posed, atol=1e-12)

    def test_a_world_point_on_the_bench_origin_is_the_surface_origin(self) -> None:
        import robosimstudio as rss

        np.testing.assert_allclose(rss.point_from_surface("bench_075", (0.6, 0.0, 0.75)), (0.0, 0.0, 0.0))
        np.testing.assert_allclose(rss.point_from_surface("void", (0.6, 0.0, 0.75)), (0.6, 0.0, 0.75))

    def test_the_helpers_are_on_the_flat_namespace(self) -> None:
        import robosimstudio as rss

        for name in ("in_surface", "from_surface", "surface_frame", "point_from_surface"):
            self.assertIn(name, rss._EXPORTS)
            self.assertTrue(callable(getattr(rss, name)))


class TestANameThatIsNotACell(unittest.TestCase):
    def test_an_unknown_cell_is_the_registry_error(self) -> None:
        import robosimstudio as rss

        with self.assertRaisesRegex(KeyError, "unknown cell"):
            rss.surface_frame("not_a_cell")


if __name__ == "__main__":
    unittest.main()
