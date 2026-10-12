# SPDX-License-Identifier: Apache-2.0
"""The studio draws a rope it holds only as a centerline, as the tube its handle says it is.

A rod routed to SuperDex is a curve: particles flagged inactive, one per node,
and no triangle. The page drew a deformable by its triangles, so it drew this
one as nothing at all -- an empty bench under a cable the solver was moving and
the path tracer was rendering. The fact that says how to draw it was already on
the handle (``DeformableHandle.tube_radius``) and only the path tracer read it.

So the page reads it too, through the same seam it dresses a towel by
(:func:`~robosimstudio_studio.garments.garment_visuals`): one mesh per rope,
faces fixed, vertices rewritten each frame -- the tube the path tracer sweeps,
from the same function. Held here: the tube is there, it sits on the nodes at
the cable's radius, it follows them by writing vertices into the one node
rather than adding another, it survives the ``flat`` look, and a capsule chain
-- a rope on Newton, drawn as its capsules -- gets no tube.
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

TIER = "medium"
"""Compiles two scenes and starts a viser server on a spare port for each."""

RADIUS = 0.006
SEGMENTS = 24


def _rope_scene(engine: str):
    import robosimstudio as rss

    physics = rss.replace(rss.GENERIC, backends=(("rod", rss.backend(engine, substeps=2)), *rss.GENERIC.backends))
    scene = rss.Scene(cell="bench_075", physics=physics, name=f"rope_on_{engine}")
    # Dropped from 6 cm, so it is still moving when the page draws it.
    scene.add(rss.rod(name="rope", length=0.4, radius=RADIUS, segments=SEGMENTS, at=(0.0, -0.2, 0.06), along=(0, 1, 0)))
    return scene.build(control_mode="joint_position")


def _session(sim, port: int):
    from robosimstudio_studio.session import StudioConfig, StudioSession

    return StudioSession(sim, config=StudioConfig(port=port, panels="drive"))


def _has_rope() -> bool:
    import robosimstudio as rss

    return hasattr(rss, "rod")


@unittest.skipUnless(_has_rope(), "the rope (rss.rod) is not in this tree")
@unittest.skipUnless(importlib.util.find_spec("superdex") is not None, "the [superdex] extra is not installed")
class TestACenterlineIsDrawnAsATube(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _rope_scene("superdex")
        cls.session = _session(cls.sim, 8973)
        cls.viewer = cls.session.viewer
        cls.key = cls.viewer._qualify("/model/garments/rope")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()
        cls.viewer.close()

    def _rings(self) -> np.ndarray:
        """The drawn tube's rings, ``[nodes, sides, 3]``, as the browser was last sent them."""
        self.session.draw()
        vertices = np.asarray(self.viewer._garment_nodes[self.key].vertices, dtype=np.float64)
        return vertices[:-2].reshape(SEGMENTS + 1, -1, 3)

    def test_the_premise_is_a_curve_with_no_surface(self) -> None:
        self.assertEqual(0, self.sim.model.tri_count, "if the rope has triangles, this is not the case under test")
        self.assertEqual(SEGMENTS + 1, self.sim.model.particle_count)

    def test_the_tube_sits_on_the_nodes_at_the_radius(self) -> None:
        self.session.reset_env()
        rings = self._rings()
        nodes = np.asarray(self.sim["rope"].particle_q, dtype=np.float64)
        np.testing.assert_allclose(rings.mean(axis=1), nodes, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(rings - nodes[:, None, :], axis=2), RADIUS, rtol=1e-3)

    def test_it_follows_the_rope_in_one_node(self) -> None:
        self.session.reset_env()
        before = self._rings()
        node = self.viewer._garment_nodes[self.key]
        for _ in range(20):
            self.sim.step_teleop()
        after = self._rings()
        self.assertIs(node, self.viewer._garment_nodes[self.key], "a frame re-added the mesh instead of moving it")
        self.assertGreater(float(before[..., 2].mean() - after[..., 2].mean()), 0.005, "the drawn rope did not fall")
        nodes = np.asarray(self.sim["rope"].particle_q, dtype=np.float64)
        np.testing.assert_allclose(after.mean(axis=1), nodes, atol=1e-5)

    def test_the_flat_look_still_draws_it(self) -> None:
        """``flat`` is the viewer's own triangle draw, and a centerline has no triangles to fall back on."""
        style = self.viewer.garment_style
        try:
            self.viewer.set_garment_style("flat")
            self.assertNotIn(self.key, self.viewer._garment_nodes)
            self.assertEqual(SEGMENTS + 1, len(self._rings()))
        finally:
            self.viewer.set_garment_style(style)


@unittest.skipUnless(_has_rope(), "the rope (rss.rod) is not in this tree")
class TestACapsuleChainIsNotATube(unittest.TestCase):
    def test_a_rope_on_newton_is_drawn_by_its_capsules_alone(self) -> None:
        sim = _rope_scene("vbd")
        session = _session(sim, 8974)
        try:
            self.assertNotIn("rope", sim.handles.deformables, "a capsule chain is bodies, not particles")
            session.draw()
            self.assertEqual([], session.viewer._tubes)
            self.assertEqual({}, session.viewer._garment_nodes)
        finally:
            session.teardown()
            session.viewer.close()


if __name__ == "__main__":
    unittest.main()
