# SPDX-License-Identifier: Apache-2.0
"""A live Ref reads ``at`` in the same frame the constructor wrote it.

The fast tests in :mod:`tests.test_surface_at` never build a scene. This one
does, on a CPU: after ``build()`` and before anything moves, ``cube.at`` is the
``at=`` that placed it, and after it lands on the bench that number is the
half-extent above the work surface while ``position`` stays world.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "cpu"
"""No GPU, a real scene; the runner hides every GPU."""


class TestRefReadsAtInTheWorkSurfaceFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        cls.placed = (0.0, 0.10, 0.20)
        scene = rss.Scene(cell="bench_075", name="surface_at")
        cls.cube = scene.add(rss.box(name="cube", half=(0.025,) * 3, at=cls.placed))
        scene.add(rss.robot("panda"))
        cls.sim = scene.build()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def test_after_build_at_is_the_constructor_at(self) -> None:
        np.testing.assert_allclose(self.cube.at, self.placed, atol=1e-5)
        np.testing.assert_allclose(self.cube.position, (0.6, 0.10, 0.95), atol=1e-5)
        state = self.sim.read()
        np.testing.assert_allclose(state.surface_position("cube"), self.placed, atol=1e-5)
        np.testing.assert_allclose(state.position("cube"), self.cube.position, atol=1e-5)

    def test_after_it_lands_at_is_the_half_extent_above_the_bench(self) -> None:
        self.sim.run(150).sync()
        np.testing.assert_allclose(self.cube.at[:2], (0.0, 0.10), atol=2e-3)
        self.assertAlmostEqual(float(self.cube.at[2]), 0.025, delta=2e-3)
        np.testing.assert_allclose(self.cube.position[:2], (0.6, 0.10), atol=2e-3)
        self.assertAlmostEqual(float(self.cube.position[2]), 0.775, delta=2e-3)
        state = self.sim.read()
        np.testing.assert_allclose(state.surface_position("cube"), self.cube.at, atol=1e-6)

    def test_void_makes_at_and_position_the_same(self) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="void", name="floor")
        cube = scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0.0, 0.10, 0.20)))
        scene.add(rss.robot("panda"))
        sim = scene.build()
        try:
            np.testing.assert_allclose(cube.at, cube.position, atol=1e-6)
            np.testing.assert_allclose(cube.at, (0.0, 0.10, 0.20), atol=1e-5)
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()
