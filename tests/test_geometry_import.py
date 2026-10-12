# SPDX-License-Identifier: Apache-2.0
"""A ported shape can exist: a polyline cable, a tet mesh from a file, a kind
an engine was never taught to build.

Two of the three deformable domains were *generated-only*. ``RodSpec`` laid a
straight line and ``SoftSpec`` a rectangular lattice, and neither had a field
that could say otherwise -- so a coiled cable, a pre-routed harness, a foam
block digitised from a real one or a tet body exported from another simulator
could not be expressed without a new entity class. This file is the seam that
changed, exercised end to end on both engines where both can serve the kind.

Medium tier: each case builds a scene.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tests.fixtures.packs import HAS_ROD_PACK, needs_rod_pack

TIER = "medium"

# A unit cube as five tetrahedra: the smallest volume worth calling a mesh.
_CUBE_VERTS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64
)
_CUBE_TETS = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6]], dtype=np.int64)

# A quarter-circle arc in the work plane, lifted slightly: not a line by any reading.
_ARC = tuple((0.15 * float(np.cos(t)), 0.15 * float(np.sin(t)), 0.03) for t in np.linspace(0.0, np.pi / 2.0, 13))
_ARC_LENGTH = float(np.linalg.norm(np.diff(np.asarray(_ARC), axis=0), axis=1).sum())


def _superdex_for(domain: str):
    import robosimstudio as rss
    from robosimstudio.specs.physics import GENERIC, backend

    return rss.replace(
        GENERIC,
        backends=(("articulated", backend("mujoco")), (domain, backend("superdex", substeps=2)), ("*", backend("vbd"))),
    )


@needs_rod_pack
class TestAPolylineCable(unittest.TestCase):
    """``RodSpec(nodes=...)``: the same arc on both implementations."""

    def test_the_echo_fields_follow_the_polyline(self) -> None:
        import robosimstudio as rss

        spec = rss.RodSpec(name="arc", nodes=_ARC)
        self.assertEqual(spec.segments, len(_ARC) - 1)
        self.assertAlmostEqual(spec.length, _ARC_LENGTH, places=9)
        np.testing.assert_allclose(spec.start, _ARC[0])
        self.assertAlmostEqual(float(np.linalg.norm(spec.direction)), 1.0, places=9)

    def test_fewer_than_two_nodes_is_refused(self) -> None:
        import robosimstudio as rss

        with self.assertRaises(ValueError):
            rss.RodSpec(name="dot", nodes=((0.0, 0.0, 0.0),))

    def _built_arc(self, physics=None) -> float:
        """Arc length of the built centerline, off the same accessor a measure uses."""
        import robosimstudio as rss
        from robosimstudio.assetlib.semantics.snapshot import Snapshot

        scene = rss.Scene(cell="void", name="arc", **({"physics": physics} if physics else {}))
        scene.add(rss.RodSpec(name="arc", nodes=_ARC, radius=0.006))
        sim = scene.build()
        try:
            sim.run(1).sync()
            pts = sim.scene.by_name["arc"].centerline(Snapshot(ctx=sim.read()))
            # Both implementations answer with the polyline's nodes: the
            # Cosserat rod its particles, the capsule chain its endpoints. The
            # chain used to answer with capsule *centres* -- one point fewer and
            # a few percent shorter -- and the two disagreed about one cable.
            self.assertEqual(len(pts), len(_ARC), "one point per stated node, on either engine")
            return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        finally:
            sim.close()

    def test_newton_builds_the_arc_as_stated(self) -> None:
        self.assertAlmostEqual(self._built_arc(), _ARC_LENGTH, delta=0.02 * _ARC_LENGTH)

    def test_superdex_builds_the_same_arc(self) -> None:
        try:
            import superdex  # noqa: F401
        except ImportError:
            self.skipTest("needs the [superdex] extra")
        self.assertAlmostEqual(self._built_arc(_superdex_for("rod")), _ARC_LENGTH, delta=0.02 * _ARC_LENGTH)


class TestATetMeshFromAFile(unittest.TestCase):
    """``SoftSpec(asset=...)``: a body the lattice could never have been."""

    def _build(self, path: Path, physics=None) -> tuple[int, int]:
        import robosimstudio as rss

        scene = rss.Scene(cell="void", name="tet", **({"physics": physics} if physics else {}))
        scene.add(rss.SoftSpec(name="cube", asset=rss.AssetSpec(path=str(path), scale=0.05), pose=(0.0, 0.0, 0.10)))
        sim = scene.build()
        try:
            sim.run(1).sync()
            return int(sim.model.particle_count), int(sim.model.tet_count)
        finally:
            sim.close()

    def _needs_the_default_soft_engine(self) -> None:
        """A soft body is solved by SuperDex unless the scene says otherwise, and that is an extra."""
        try:
            import superdex  # noqa: F401
        except ImportError:
            self.skipTest("needs the [superdex] extra")

    def test_an_npz_builds_its_own_tets(self) -> None:
        self._needs_the_default_soft_engine()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cube.npz"
            np.savez(path, vertices=_CUBE_VERTS, tets=_CUBE_TETS)
            particles, tets = self._build(path)
        self.assertEqual(particles, 8)
        self.assertEqual(tets, 5)

    def test_a_meshio_format_builds_the_same(self) -> None:
        try:
            import meshio
        except ImportError:
            self.skipTest("needs meshio")
        self._needs_the_default_soft_engine()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cube.vtk"
            meshio.write(path, meshio.Mesh(points=_CUBE_VERTS, cells=[("tetra", _CUBE_TETS)]))
            particles, tets = self._build(path)
        self.assertEqual((particles, tets), (8, 5))

    def test_a_file_with_no_tetrahedra_is_refused_by_name(self) -> None:
        from robosimstudio.domains.soft import read_tets

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "flat.npz"
            np.savez(path, vertices=_CUBE_VERTS, tets=np.zeros((0, 4), np.int64))
            with self.assertRaises(ValueError) as caught:
                read_tets(path)
        self.assertIn("not a volume", str(caught.exception))

    def test_superdex_serves_the_file_body(self) -> None:
        try:
            import superdex  # noqa: F401
        except ImportError:
            self.skipTest("needs the [superdex] extra")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cube.npz"
            np.savez(path, vertices=_CUBE_VERTS, tets=_CUBE_TETS)
            particles, tets = self._build(path, _superdex_for("soft"))
        self.assertEqual((particles, tets), (8, 5))


class TestAnEngineCanBeTaughtAKind(unittest.TestCase):
    """``register_actor_builder``: the build half of the seam is open."""

    def test_a_kind_nobody_taught_is_refused_naming_what_is_built(self) -> None:
        # The engine's own kinds register when its bridge is imported, which a scene on it does.
        import robosimstudio.engines.superdex.bridge  # noqa: F401
        from robosimstudio.engines.geometry import actor_builder, builders_for

        self.assertIsNone(actor_builder("superdex", "strand"))
        # Two kinds are the library's own; the rope's is registered by its pack, where that is installed.
        shipped = {"shell", "tet"} | ({"rod"} if HAS_ROD_PACK else set())
        self.assertEqual(shipped, set(builders_for("superdex")), "exactly the installed kinds register")

    def test_a_third_party_can_teach_one(self) -> None:
        from robosimstudio.engines.geometry import (
            ACTOR_BUILDERS,
            actor_builder,
            register_actor_builder,
        )

        def build(bridge, record, particle_q, tet_indices, tet_materials):
            return {"kind": "strand"}

        register_actor_builder("superdex", "strand", build)
        try:
            self.assertIs(actor_builder("superdex", "strand"), build)
        finally:
            ACTOR_BUILDERS._items.pop("superdex:strand", None)
            ACTOR_BUILDERS._origins.pop("superdex:strand", None)


if __name__ == "__main__":
    unittest.main()
