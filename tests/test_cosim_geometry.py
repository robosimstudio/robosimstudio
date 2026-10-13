# SPDX-License-Identifier: Apache-2.0
"""Newton's shapes, read into the engine-neutral form both engines build from.

``robosimstudio/engines/geometry.py`` is the *reading* half of what used to be
``engines/superdex/mirror.py``'s ``make_collider_shape`` -- the half that was the
same work for every engine and is now done once. The SuperDex half stayed where
it was, and ``tests/test_superdex_contact.py`` covers it end to end with the engine in
the loop; what is checked here is the part that has no engine in it, against a
stub model, so it runs in the fast tier.

The property that matters most is the one the split exists for: **primitive
intent and tessellation are separate**. An engine with a native box uses the
box; an engine without one asks for triangles. Reading a shape must not force
the second on an engine that wanted the first.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""A stub model. No GPU, no engine, no scene."""


class _Arr:
    def __init__(self, a):
        self._a = np.asarray(a)

    def numpy(self):
        return self._a


class _Source:
    """What ``model.shape_source`` holds for a mesh: warp-ish arrays."""

    def __init__(self, vertices, indices):
        self.vertices = _Arr(vertices)
        self.indices = _Arr(indices)


class _Model:
    """The six things :func:`read_shape_geometry` asks a finalized model for."""

    def __init__(self, types, scales, transforms, bodies, mus, sources):
        self.shape_type = _Arr(types)
        self.shape_scale = _Arr(scales)
        self.shape_transform = _Arr(transforms)
        self.shape_body = _Arr(bodies)
        self.shape_material_mu = _Arr(mus)
        self.shape_source = sources


def _model():
    from newton import GeoType

    tri = _Source(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        indices=np.array([0, 1, 2], dtype=np.int64),
    )
    kinds = [
        GeoType.PLANE,
        GeoType.SPHERE,
        GeoType.BOX,
        GeoType.CAPSULE,
        GeoType.CYLINDER,
        GeoType.CONE,
        GeoType.MESH,
        GeoType.CONVEX_MESH,
        99,  # a kind nothing can mirror
    ]
    n = len(kinds)
    ident = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (n, 1))
    return _Model(
        types=[int(k) for k in kinds],
        scales=np.tile(np.array([0.05, 0.1, 0.05]), (n, 1)),
        transforms=ident,
        bodies=[-1, 3, 3, 4, 4, 5, 6, 6, 7],
        mus=np.full(n, 0.8),
        sources=[None, None, None, None, None, None, tri, tri, None],
    )


NAMES = ["plane", "sphere", "box", "capsule", "cylinder", "cone", "mesh", "mesh"]


class TestReadShapeGeometry(unittest.TestCase):
    def test_every_newton_kind_maps_to_a_name(self):
        from robosimstudio.engines.geometry import read_shape_geometry

        model = _model()
        for index, want in enumerate(NAMES):
            with self.subTest(shape=index):
                geom = read_shape_geometry(model, index)
                self.assertIsNotNone(geom)
                self.assertEqual(geom.kind, want)
        self.assertIsNone(read_shape_geometry(model, 8), "an unmirrorable kind is a skip, not a crash")

    def test_both_mesh_types_collapse_to_one_name(self):
        """An adapter can do exactly one thing with either, so it sees one name."""
        from robosimstudio.engines.geometry import read_shape_geometry

        model = _model()
        self.assertEqual(read_shape_geometry(model, 6).kind, read_shape_geometry(model, 7).kind)

    def test_it_carries_the_fields_a_mirror_needs(self):
        from robosimstudio.engines.geometry import read_shape_geometry

        model = _model()
        geom = read_shape_geometry(model, 2, two_way=True)
        self.assertEqual(geom.shape, 2)
        self.assertEqual(geom.body, 3)
        self.assertTrue(geom.two_way)
        self.assertTrue(geom.moving, "a body carries it, so a tick has to drive it")
        self.assertAlmostEqual(geom.mu, 0.8)
        np.testing.assert_allclose(geom.scale, [0.05, 0.1, 0.05])
        self.assertFalse(read_shape_geometry(model, 0).moving, "a world-fixed shape is posed once")

    def test_the_recipe_body_overrides_the_model(self):
        """The recipe states the owning body; the model's is the fallback."""
        from robosimstudio.engines.geometry import read_shape_geometry

        model = _model()
        self.assertEqual(read_shape_geometry(model, 2, body=11).body, 11)
        self.assertEqual(read_shape_geometry(model, 2).body, 3)

    def test_source_rides_along_for_meshes_only(self):
        from robosimstudio.engines.geometry import read_shape_geometry

        model = _model()
        self.assertIsNotNone(read_shape_geometry(model, 6).source)
        self.assertIsNone(read_shape_geometry(model, 2).source, "a box needs no source")


class TestTessellationIsSeparate(unittest.TestCase):
    """The split's whole point: intent first, triangles only if asked."""

    def test_the_analytic_kinds_tessellate_on_demand(self):
        from robosimstudio.engines.geometry import read_shape_geometry

        model = _model()
        for index in (1, 2, 3, 4, 5):
            with self.subTest(kind=NAMES[index]):
                verts, faces = read_shape_geometry(model, index).mesh()
                self.assertGreater(len(verts), 3)
                self.assertGreater(len(faces), 3)
                self.assertEqual(faces.shape[1], 3)
                self.assertLess(int(faces.max()), len(verts), "no face indexes past the vertices")

    def test_a_box_is_its_full_extent(self):
        """Newton stores half-extents; a mirror must be the whole box."""
        from robosimstudio.engines.geometry import read_shape_geometry

        verts, _ = read_shape_geometry(_model(), 2).mesh()
        np.testing.assert_allclose(verts.max(axis=0) - verts.min(axis=0), [0.1, 0.2, 0.1], atol=1e-9)

    def test_a_mesh_comes_back_scaled_as_collision_saw_it(self):
        from robosimstudio.engines.geometry import read_shape_geometry

        verts, faces = read_shape_geometry(_model(), 6).mesh()
        np.testing.assert_allclose(verts, np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.1, 0.0]]))
        np.testing.assert_array_equal(faces, [[0, 1, 2]])

    def test_a_plane_has_no_triangles(self):
        """It is a half-space; an engine that cannot say so has no business here."""
        from robosimstudio.engines.geometry import read_shape_geometry

        self.assertIsNone(read_shape_geometry(_model(), 0).mesh())

    def test_a_cone_is_centred_as_newton_builds_it(self):
        """Newton's cone runs base -h to apex +h about its origin; trimesh's runs 0 to 2h."""
        from robosimstudio.engines.geometry import ShapeGeometry

        geom = ShapeGeometry(
            shape=0,
            body=-1,
            kind="cone",
            scale=np.array([0.03, 0.05, 0.0]),
            local=np.array([0.0, 0, 0, 0, 0, 0, 1.0]),
            mu=0.5,
        )
        verts, _ = geom.mesh()
        self.assertAlmostEqual(-0.05, float(verts[:, 2].min()))
        self.assertAlmostEqual(0.05, float(verts[:, 2].max()))
        base = verts[np.isclose(verts[:, 2], -0.05)]
        self.assertAlmostEqual(0.03, float(np.linalg.norm(base[:, :2], axis=1).max()), places=6)

    def test_a_mesh_without_a_source_is_a_skip(self):
        from robosimstudio.engines.geometry import ShapeGeometry

        geom = ShapeGeometry(
            shape=0,
            body=-1,
            kind="mesh",
            scale=np.ones(3),
            local=np.array([0.0, 0, 0, 0, 0, 0, 1.0]),
            mu=0.5,
            source=None,
        )
        self.assertIsNone(geom.mesh())


class TestIterMirrors(unittest.TestCase):
    def test_it_reads_the_recipe_records(self):
        from robosimstudio.engines.geometry import iter_mirrors

        model = _model()
        records = [
            {"shape": 0, "body": -1, "two_way": False},
            {"shape": 2, "body": 3, "two_way": True},
            {"shape": 8, "body": 7, "two_way": True},  # unmirrorable
        ]
        mirrors = iter_mirrors(model, records)
        self.assertEqual([m.shape for m in mirrors], [0, 2], "the unmirrorable one is dropped")
        self.assertEqual([m.two_way for m in mirrors], [False, True])

    def test_it_survives_a_record_with_no_body(self):
        from robosimstudio.engines.geometry import iter_mirrors

        mirrors = iter_mirrors(_model(), [{"shape": 2}])
        self.assertEqual(mirrors[0].body, 3, "the model's owning body is the fallback")


class _DeformableModel:
    """The particle/connectivity half of a finalized model, as an actor reader sees it."""

    def __init__(self, *, particles, tris=(), tets=(), tet_materials=()):
        self.particle_q = _Arr(np.asarray(particles, dtype=np.float64))
        self.tri_indices = _Arr(np.asarray(tris, dtype=np.int64).reshape(-1))
        self.tri_count = len(np.asarray(tris, dtype=np.int64).reshape(-1, 3)) if len(tris) else 0
        self.tet_indices = _Arr(np.asarray(tets, dtype=np.int64).reshape(-1))
        self.tet_count = len(np.asarray(tets, dtype=np.int64).reshape(-1, 4)) if len(tets) else 0
        self.tet_materials = _Arr(
            np.asarray(tet_materials, dtype=np.float64).reshape(-1, 3) if len(tet_materials) else np.zeros((0, 3))
        )


class TestTheActorKindSetIsOpen(unittest.TestCase):
    """A foreign engine could simulate exactly two shapes of thing, and the set
    was closed by an ``if``.

    ``actor_geometry`` branched ``shell``/``tet``, and the cost was not the
    branch -- it was that both of those are particle *meshes*, so a rod could
    not be handed to a foreign engine at all however good that engine's rod was.
    A rod's centerline is an ordered run of particles owning neither a triangle
    nor a tetrahedron, so no amount of reading the model can recognise it.
    """

    def test_the_shipped_kinds_are_registered_rather_than_branched(self):
        """Three where there were two, and the third is the evidence: ``rod``
        was added without touching ``actor_geometry`` at all.

        It is added from *outside* now: the library registers the shell and the
        tet, and the rod pack registers its own reader when it is installed --
        which the registry finds by scanning the packs, not by anyone having
        imported one first. A checkout without the pack has the library's two.
        """
        from robosimstudio.engines.geometry import ACTOR_KINDS
        from tests.fixtures.packs import HAS_ROD_PACK

        self.assertEqual(sorted(["shell", "tet", *(["rod"] if HAS_ROD_PACK else [])]), ACTOR_KINDS.names())

    def test_a_shell_is_read_as_its_triangles_reindexed_to_zero(self):
        from robosimstudio.engines.geometry import actor_geometry

        model = _DeformableModel(particles=np.zeros((9, 3)), tris=[[4, 5, 6], [5, 6, 7]])
        geom = actor_geometry(model, {"name": "sheet", "kind": "shell", "particle_start": 4, "particle_count": 5})
        self.assertEqual([[0, 1, 2], [1, 2, 3]], geom.tris.tolist())
        self.assertIsNone(geom.tets)

    def test_a_tet_carries_the_elasticity_the_model_states(self):
        from robosimstudio.engines.geometry import actor_geometry

        model = _DeformableModel(particles=np.zeros((4, 3)), tets=[[0, 1, 2, 3]], tet_materials=[[1.0e5, 1.5e5, 0.0]])
        geom = actor_geometry(model, {"name": "blob", "kind": "tet", "particle_start": 0, "particle_count": 4})
        self.assertEqual([[0, 1, 2, 3]], geom.tets.tolist())
        self.assertGreater(geom.young, 0.0)
        self.assertIsNotNone(geom.poisson)

    def test_a_centerline_reads_with_no_table_to_cut(self):
        """A rod: the kind that could not be expressed at all until the set opened.

        Its reader cuts nothing -- the composer already refuses a
        non-contiguous particle range, so for a centerline the node *order* is
        the connectivity. That makes it the cheapest of the three readers rather
        than the most expensive, which is the shape of evidence that the old
        branch was closed for no good reason.

        Registered here under a name of this module's own as well, to check that
        the door is open from outside and not merely that the library walked
        through it.
        """
        from robosimstudio.engines.geometry import ACTOR_KINDS, actor_geometry, register_actor_kind

        register_actor_kind("centerline_probe", lambda model, base: base)
        self.addCleanup(lambda: ACTOR_KINDS._items.pop("centerline_probe", None))

        nodes = np.linspace([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], 8)
        model = _DeformableModel(particles=nodes)
        geom = actor_geometry(
            model,
            {
                "name": "cord",
                "kind": "centerline_probe",
                "particle_start": 0,
                "particle_count": 8,
                "params": {"radius": 0.011},
            },
        )
        self.assertEqual("centerline_probe", geom.kind)
        self.assertEqual(8, len(geom.rest))
        self.assertIsNone(geom.tris)
        self.assertIsNone(geom.tets)
        self.assertEqual(0.011, geom.params["radius"])

    def test_an_unregistered_kind_refuses_rather_than_defaulting(self):
        """Both available defaults would build the wrong thing silently."""
        from robosimstudio.engines.geometry import actor_geometry

        model = _DeformableModel(particles=np.zeros((4, 3)))
        with self.assertRaises(KeyError) as caught:
            actor_geometry(model, {"name": "x", "kind": "nosuch", "particle_start": 0, "particle_count": 4})
        self.assertIn("nosuch", str(caught.exception))
        self.assertIn("shell", str(caught.exception), "the message names what is registered")


if __name__ == "__main__":
    unittest.main()
