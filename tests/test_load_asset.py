# SPDX-License-Identifier: Apache-2.0
"""A third party can put a mesh file into a builder.

The one step of writing an entity that had no public answer. Everything else was
already open -- ``steps_of`` finds phases by method name, the compiler resolves
entity classes by dotted path, a solver may be named by bare dotted path -- and
the geometry import existed only as two private methods inside the library's own
entities, each with its own drift.

What is checked here is the contract a caller depends on: the ranges come back
right, the world case and the body case both work, the proxy handling matches
what the shipped entities do, and every refusal names what to do instead.

:mod:`robosimstudio.scene.assets` argues at length why this is **not** those two
private methods factored out. This module is the other half of that argument:
the mesh path is small enough to state completely, and the tests below are what
"completely" means.
"""

from __future__ import annotations

import unittest

TIER = "fast"
"""Builds into a ``ModelBuilder``; no model, no solver, no GPU."""

MESH = "embodiments/panda/assets/link1.stl"
"""A vendored mesh with no texture and no proxies -- the plain case.

The Panda's, because that robot ships with every distribution of the library;
a mesh borrowed from a robot whose description may be left out would make this
a test of what happens to be installed."""


def _builder():
    import newton

    return newton.ModelBuilder()


class TestTheMeshPath(unittest.TestCase):
    def test_a_mesh_welds_to_the_world_by_default(self) -> None:
        from robosimstudio.scene import load_asset
        from robosimstudio.specs import AssetSpec

        builder = _builder()
        added = load_asset(builder, AssetSpec(path=MESH), label="bracket")
        self.assertEqual(0, builder.body_count, "body=-1 is the default: scenery, welded")
        self.assertEqual((0, 1), added.shapes)
        self.assertEqual(added.visual, 0)
        self.assertEqual(-1, int(builder.shape_body[added.visual]))

    def test_the_reported_range_is_the_range(self) -> None:
        """The reason :class:`~robosimstudio.scene.assets.Added` exists. An
        entity that brackets ``builder.shape_count`` itself gets it wrong once
        and is then wrong quietly: an unclaimed shape is a shape the interaction
        graph gives to nobody, and a thing that collides with nothing looks
        exactly like a thing at rest."""
        from robosimstudio.scene import load_asset
        from robosimstudio.specs import AssetSpec

        builder = _builder()
        load_asset(builder, AssetSpec(path=MESH), label="one")
        added = load_asset(builder, AssetSpec(path=MESH), label="two")
        self.assertEqual((1, 2), added.shapes, "the second call reports its own shapes, not the first's")
        self.assertEqual(builder.shape_count, added.shapes[1])

    def test_a_body_carries_the_shapes(self) -> None:
        """The caller creates the body. That is the line: recording the free
        joint's coordinate rows for a reset is a decision only the entity can
        make, so this function will not make it silently."""
        import warp as wp

        from robosimstudio.scene import load_asset
        from robosimstudio.specs import AssetSpec

        builder = _builder()
        body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat(0, 0, 0, 1)), label="part")
        added = load_asset(builder, AssetSpec(path=MESH), body=body, label="part")
        self.assertEqual(body, int(builder.shape_body[added.visual]))

    def test_the_xform_takes_seven_numbers_or_a_transform(self) -> None:
        import warp as wp

        from robosimstudio.scene import load_asset
        from robosimstudio.specs import AssetSpec

        for xform in ((0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0), wp.transform((0.1, 0.2, 0.3), wp.quat(0, 0, 0, 1))):
            with self.subTest(kind=type(xform).__name__):
                builder = _builder()
                added = load_asset(builder, AssetSpec(path=MESH), xform=xform, label="at")
                placed = builder.shape_transform[added.visual]
                self.assertAlmostEqual(0.2, float(placed[1]), places=5)

    def test_a_bad_xform_says_what_it_wanted(self) -> None:
        from robosimstudio.scene import load_asset
        from robosimstudio.specs import AssetSpec

        with self.assertRaises(ValueError) as caught:
            load_asset(_builder(), AssetSpec(path=MESH), xform=(0.0, 0.0, 0.0), label="short")
        self.assertIn("7 numbers", str(caught.exception))


class TestWhatItRefuses(unittest.TestCase):
    """Every refusal names the thing to do instead. A message that only says no
    is a message that sends the reader into the library's source."""

    def test_a_stage_or_an_mjcf_points_at_the_entity_that_owns_it(self) -> None:
        from robosimstudio.scene import load_asset
        from robosimstudio.specs import AssetSpec

        with self.assertRaises(ValueError) as caught:
            load_asset(_builder(), AssetSpec(path="embodiments/panda/panda_standalone.xml"), label="arm")
        message = str(caught.exception)
        self.assertIn("EntityStatic", message)
        self.assertIn("add_usd_stage", message)

    def test_proxies_and_an_approximation_are_refused_together(self) -> None:
        """The shipped entities both refuse this and both say why: the
        approximation would swallow the proxies into the very hull they exist to
        avoid. Checked here so a third party gets the same answer."""
        from robosimstudio.scene import load_asset
        from robosimstudio.specs import AssetSpec

        spec = AssetSpec(path=MESH, collision_path=MESH, approximate="convex_hull")
        with self.assertRaises(ValueError) as caught:
            load_asset(_builder(), spec, label="both")
        self.assertIn("swallow", str(caught.exception))


class TestProxiesAndApproximation(unittest.TestCase):
    def test_a_proxy_collides_and_the_visual_does_not(self) -> None:
        """What a collision proxy *is*: the authored mesh keeps its look and its
        mass, and contact goes through the cheap stand-in."""
        import newton

        from robosimstudio.scene import load_asset
        from robosimstudio.specs import AssetSpec

        builder = _builder()
        added = load_asset(builder, AssetSpec(path=MESH, collision_path=MESH), label="part")
        self.assertEqual(1, len(added.proxies))
        visual_flags = int(builder.shape_flags[added.visual])
        proxy_flags = int(builder.shape_flags[added.proxies[0]])
        self.assertFalse(visual_flags & int(newton.ShapeFlags.COLLIDE_SHAPES), "the visual still collides")
        self.assertFalse(proxy_flags & int(newton.ShapeFlags.VISIBLE), "the proxy is drawn")
        self.assertTrue(proxy_flags & int(newton.ShapeFlags.COLLIDE_SHAPES))
        self.assertEqual("part_collision", builder.shape_label[added.proxies[0]])

    def test_several_proxies_are_numbered(self) -> None:
        from robosimstudio.scene import load_asset
        from robosimstudio.specs import AssetSpec

        builder = _builder()
        added = load_asset(builder, AssetSpec(path=MESH, collision_paths=(MESH, MESH)), label="part")
        self.assertEqual(2, len(added.proxies))
        self.assertEqual(["part_collision_00", "part_collision_01"], [builder.shape_label[s] for s in added.proxies])

    def test_the_collider_argument_overrides_the_spec(self) -> None:
        """``collider=`` is the call site's word and ``AssetSpec.approximate``
        is the asset's; the call site is the more specific of the two."""
        from robosimstudio.scene import load_asset
        from robosimstudio.specs import AssetSpec

        builder = _builder()
        added = load_asset(builder, AssetSpec(path=MESH, keep_visual=False), collider="bounding_box", label="boxed")
        self.assertEqual(builder.shape_count, added.shapes[1], "the reported range covers the approximation")
        self.assertEqual(1, builder.shape_count, "keep_visual=False leaves only the box")


if __name__ == "__main__":
    unittest.main()
