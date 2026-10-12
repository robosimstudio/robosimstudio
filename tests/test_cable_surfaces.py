# SPDX-License-Identifier: Apache-2.0
"""The geometry a centerline cable is given so it can be touched and seen.

A Cosserat rod is a polyline. To be *touched* at its radius it gets a tubular
skin whose cross-section frames the engine reads its rest twist from; to be
*seen* by the path tracer and the studio page, which draw deformables as
triangles, it gets a tube swept along it every frame. Both turn on one property
-- frames that do not flip where the cable turns -- and both are pure functions,
so they are pinned here without an engine. So is the check a cable's join runs
before it welds.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

TIER = "fast"
"""No GPU: numpy over polylines, and one spec."""


def _l_shaped(step: float = 0.005) -> np.ndarray:
    """Down a post, then along the bench: the routed cable that diverged."""
    down = [(0.0, 0.0, 0.2 - step * i) for i in range(40)]
    along = [(-step * i, 0.0, 0.0) for i in range(1, 40)]
    return np.asarray(down + along, dtype=np.float64)


class TestTheSkinFramesAreTransported(unittest.TestCase):
    def test_every_axis_is_a_unit_normal_to_its_element(self):
        from robosimstudio.domains.rod.actor import transported_frames

        nodes = _l_shaped()
        axes = transported_frames(nodes)
        tangents = np.diff(nodes, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
        self.assertEqual(len(axes), len(nodes) - 1)
        np.testing.assert_allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(np.einsum("ij,ij->i", axes, tangents), 0.0, atol=1e-9)

    def test_no_axis_flips_through_the_turn(self):
        """A per-element rule flipped 90 degrees at the corner; the engine read that as rest twist and diverged."""
        from robosimstudio.domains.rod.actor import transported_frames

        axes = transported_frames(_l_shaped())
        turn = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", axes[:-1], axes[1:]), -1.0, 1.0)))
        self.assertLess(float(turn.max()), 1.0)


class TestTheRenderedTube(unittest.TestCase):
    def test_the_tube_sits_one_radius_off_the_centerline(self):
        from robosimstudio.core.render.visual import tube_points, tube_triangles

        nodes, radius, sides = _l_shaped(), 0.003, 12
        points, normals = tube_points(nodes, radius, sides)
        ring = points[:-2].reshape(len(nodes), sides, 3)
        np.testing.assert_allclose(np.linalg.norm(ring - nodes[:, None, :], axis=2), radius, rtol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
        tris = tube_triangles(len(nodes), sides)
        self.assertEqual(int(tris.max()), len(points) - 1)
        self.assertEqual(len(tris), 2 * sides * (len(nodes) - 1) + 2 * sides)

    def test_the_tube_does_not_pinch_at_the_turn(self):
        from robosimstudio.core.render.visual import tube_points

        nodes = _l_shaped()
        points, _ = tube_points(nodes, 0.003, 12)
        ring = points[:-2].reshape(len(nodes), 12, 3)
        # A twisting frame folds a ring onto itself; transported, every ring keeps its full width.
        widths = np.linalg.norm(ring[:, 0] - ring[:, 6], axis=1)
        np.testing.assert_allclose(widths, 0.006, rtol=1e-6)


class TestTheCenterlineReadsEitherSnapshot(unittest.TestCase):
    def test_a_host_snapshot_answers(self):
        """``HostSnapshot`` carries ``particle_q`` and no ``particles``; reading one raised."""
        from robosimstudio.domains.rod import RodSpec
        from robosimstudio.domains.rod.superdex import EntityRodSuperdex

        entity = EntityRodSuperdex(RodSpec(name="cable"))
        entity.particles = np.arange(2, 7, dtype=np.int32)

        class _HostSnapshot:
            particle_q = np.arange(30, dtype=np.float64).reshape(10, 3)

        np.testing.assert_allclose(entity.centerline(_HostSnapshot()), _HostSnapshot.particle_q[2:7])


class TestAStatedModulusIsNotReportedLost(unittest.TestCase):
    def test_young_rides_the_model(self):
        """A clip's E goes to SuperDex through ``tet_materials``; the composer said it never arrived."""
        from types import SimpleNamespace

        from robosimstudio.domains.soft import SoftMaterialSpec, SoftSpec

        from robosimstudio.scene.physics.compose import _warn_lost_material
        from robosimstudio.specs.physics import HostActor

        spec = SoftSpec(name="clip", material=SoftMaterialSpec(young=3.0e8, poisson=0.38))
        node = SimpleNamespace(name="clip", entity=SimpleNamespace(spec=spec))
        island = SimpleNamespace(backend=SimpleNamespace(solver="superdex"))
        declared = HostActor(kind="tet", params=spec.material.actor_params())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_lost_material(node, declared, island)
        self.assertEqual([str(w.message) for w in caught], [])


class TestAWeldToleranceHasAFloat32Floor(unittest.TestCase):
    """Nodes built at one point sit up to a float32 ULP apart per axis, which a fine mesh's thousandth exceeded.

    A row of nodes 0.05 mm apart around (0.6, 0.3, 0.95) m: a thousandth of
    its spacing is 0.05 um, and one ULP on each axis there is 0.089 um.
    """

    def _check(self, offset: np.ndarray) -> None:
        from robosimstudio.engines.superdex.bridge import SuperdexBridge

        row = np.array([0.6, 0.3, 0.95]) + np.arange(3)[:, None] * np.array([0.0, 0.0, 5.0e-5])
        mine = row.astype(np.float32).astype(np.float64)  # a rest layout is float32
        theirs = mine.copy()
        theirs[0] = offset(mine[0])
        nodes = np.array([0])
        SuperdexBridge._check_coincident({"name": "cable"}, {"name": "weight"}, nodes, nodes, mine, theirs)

    def test_float32_rounding_is_accepted(self):
        self._check(lambda p: np.nextafter(p.astype(np.float32), np.float32(2.0)).astype(np.float64))

    def test_a_micrometre_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"'cable' joins 'weight'.*0\.001 mm"):
            self._check(lambda p: p + np.array([1.0e-6, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
