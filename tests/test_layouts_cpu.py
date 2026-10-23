# SPDX-License-Identifier: Apache-2.0
"""A layout's declared pose is the pose the compiler puts the body at.

The fast tests in :mod:`tests.test_layouts` never build a scene. This one does,
on a CPU, so a draw that only rewrote the spec would still fail here.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "cpu"
"""No GPU, a real scene; the runner hides every GPU."""


class TestALayoutLandsWhereItSays(unittest.TestCase):
    def test_the_cube_starts_at_the_variant_pose(self) -> None:
        import robosimstudio as rss
        from robosimstudio.scene.frames import in_surface

        scene = rss.Scene(cell="bench_075", name="layout")
        scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0.0, 0.10, 0.20)))
        scene.add(rss.robot("panda"))
        variant = rss.layouts(scene, 1, seed=3, xy=0.04)[0]
        spec = next(entity for entity in variant.request.entities if entity.name == "cube")
        sim = variant.build()
        try:
            world = in_surface(rss.get_cell("bench_075"), (*spec.pose, *spec.quat))
            np.testing.assert_allclose(sim["cube"].position, world[:3], atol=1e-5)
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()
