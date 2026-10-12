# SPDX-License-Identifier: Apache-2.0
"""Reading an asset's interface without assuming it has a shape.

The claim under test is the one :mod:`~robosimstudio.assetlib.semantics` makes:
an interface here is the Python kind -- duck-typed, per object, freely defined --
so **no two assets need have anything in common beyond the substrate**. A
receptacle's bore is five numbers with frames on two of them; a towel's grasp
strip is a selector expression; a shirt's collar is a reference to registered
code; a fold panel's fields are the names of *other* features. Four features,
four different things to do with a payload, and a reader that walks one shape
gets three of them wrong.

So these tests are mostly about the *differences*. What a uniform reader would
lose is exactly what is asserted here: which fields are positions and which are
directions (translate one and it stops being a unit vector), whether a payload is
constants or an expression, whether the kind says anything about the feature at
all, and whether a key the kind does not name was dropped on the floor.

Synthetic records throughout, plus one sweep over the real corpus. The synthetic
half needs no registry writes -- every kind named is either already registered or
deliberately is not -- which keeps the tests from leaking a vocabulary into
whatever runs after them.
"""

from __future__ import annotations

import unittest

from robosimstudio.assetlib.interface import DIRECTION, POINT, facet, interface
from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.registry import get_asset, list_assets
from robosimstudio.assetlib.semantics.features import FeatureSpec
from robosimstudio.assetlib.sim import SimRequirements
from tests.fixtures.content import needs_content_assets

TIER = "fast"
"""No GPU, no scene. Gates every commit."""


def _record(*features: FeatureSpec, domain: str = "rigid") -> AssetRecord:
    # The substrate is derived from the domain now, so a test that wants a
    # particular reading states the domain that produces it.
    return AssetRecord(name="thing", summary="A thing.", features=features, sim=SimRequirements(domain=domain))


class TestEachTierParsesIntoItsOwnCarrier(unittest.TestCase):
    """A tier names the mechanism, and the three mechanisms share no content.

    Reading `payload` uniformly across them is the failure this prevents: a
    recipe's payload is an expression and a resolver's is a key, so a reader
    pairing either against a kind's declared fields reports a feature with
    nothing in it.
    """

    def test_a_literal_carries_members_and_nothing_else(self) -> None:
        parsed = facet(
            FeatureSpec(
                name="bore",
                kind="cavity",
                tier="literal",
                payload={"axis": [0.0, 0.0, -1.0], "mouth": [0.0, 0.0, 0.013], "depth": 0.013},
            )
        )
        self.assertEqual([m.name for m in parsed.members], ["axis", "mouth", "depth"])
        self.assertIsNone(parsed.select)
        self.assertEqual(parsed.resolver, "")

    def test_a_recipe_carries_a_canonical_selector_and_no_members(self) -> None:
        parsed = facet(
            FeatureSpec(
                name="grasp",
                kind="set",
                tier="recipe",
                payload={"select": {"op": "extreme", "axis": "y", "band": 0.025, "high": True}},
            )
        )
        self.assertEqual(parsed.members, ())
        # Canonicalised through the selector's own form, so an axis *name*
        # arrives as the index every consumer of a recipe actually uses.
        self.assertEqual(parsed.select["axis"], 1)

    def test_a_resolver_carries_the_key_it_references(self) -> None:
        parsed = facet(
            FeatureSpec(name="collar", kind="set", tier="resolver", payload={"resolver": "garments.collar_curve@v1"})
        )
        self.assertEqual(parsed.resolver, "garments.collar_curve@v1")
        self.assertEqual(parsed.members, ())
        self.assertIsNone(parsed.select)


class TestAFieldArrivesWithItsFrame(unittest.TestCase):
    """Which fields reach world how, which is the one thing a drawer must know.

    `FeatureKind` splits `points` from `directions` precisely so a reader does
    not have to guess, and its own docstring says what guessing costs: translate
    a direction and a unit vector becomes a position, which then normalises to
    something pointing at the body rather than along the bore.
    """

    def setUp(self) -> None:
        self.parsed = facet(
            FeatureSpec(
                name="bore",
                kind="cavity",
                tier="literal",
                payload={"axis": [0.0, 0.0, -1.0], "mouth": [0.0, 0.0, 0.013], "depth": 0.013},
            )
        )
        self.frames = {m.name: m.frame for m in self.parsed.members}

    def test_a_direction_is_marked_a_direction(self) -> None:
        self.assertEqual(self.frames["axis"], DIRECTION)

    def test_a_position_is_marked_a_position(self) -> None:
        self.assertEqual(self.frames["mouth"], POINT)

    def test_a_scalar_is_marked_neither(self) -> None:
        self.assertEqual(self.frames["depth"], "")

    def test_a_field_carries_the_kinds_own_words_for_it(self) -> None:
        """The meaning is the kind's, never restated beside the value."""
        meanings = {m.name: m.meaning for m in self.parsed.members}
        self.assertIn("insertion direction", meanings["axis"])
        self.assertTrue(all(meanings.values()), "a declared field with no meaning is a kind that did not say")

    def test_carrying_geometry_is_what_makes_a_facet_drawable(self) -> None:
        self.assertTrue(self.parsed.drawable)
        template = facet(FeatureSpec(name="fold", kind="template", tier="literal", payload={"steps": []}))
        self.assertFalse(template.drawable, "a step list has no place in the world")


class TestMembersReadInTheKindsOrder(unittest.TestCase):
    def test_two_assets_writing_one_kind_read_the_same_way(self) -> None:
        """The kind's order, not the author's, so a page is comparable down it.

        Two people annotating a cavity will not agree on key order, and there is
        no reason they should have to -- the payload is a mapping. Ordering here
        is what makes that irrelevant.
        """
        forwards = facet(
            FeatureSpec(
                name="a", kind="cavity", tier="literal", payload={"axis": [0, 0, -1], "mouth": [0, 0, 0], "depth": 1.0}
            )
        )
        backwards = facet(
            FeatureSpec(
                name="b", kind="cavity", tier="literal", payload={"depth": 1.0, "mouth": [0, 0, 0], "axis": [0, 0, -1]}
            )
        )
        self.assertEqual([m.name for m in forwards.members], [m.name for m in backwards.members])

    def test_required_fields_come_before_optional_ones(self) -> None:
        parsed = facet(
            FeatureSpec(
                name="bore",
                kind="cavity",
                tier="literal",
                payload={"tip_offset": 0.01, "axis": [0, 0, -1], "mouth": [0, 0, 0], "depth": 1.0},
            )
        )
        required = [m.required for m in parsed.members]
        self.assertEqual(required, sorted(required, reverse=True))


class TestAGapIsReportedRatherThanFilled(unittest.TestCase):
    """What the kind does not account for, said out loud.

    Both directions matter and for the same reason: a reader that silently drops
    an unknown key and a reader that silently defaults a missing one both produce
    a feature that looks complete and is not.
    """

    def test_a_missing_required_field_is_named(self) -> None:
        parsed = facet(FeatureSpec(name="bore", kind="cavity", tier="literal", payload={"axis": [0, 0, -1]}))
        self.assertIn("mouth", parsed.missing)
        self.assertIn("depth", parsed.missing)

    def test_a_field_the_kind_does_not_define_is_surfaced_not_dropped(self) -> None:
        parsed = facet(
            FeatureSpec(
                name="bore",
                kind="cavity",
                tier="literal",
                payload={"axis": [0, 0, -1], "mouth": [0, 0, 0], "depth": 1.0, "colour": "red"},
            )
        )
        self.assertEqual(parsed.extra, ("colour",))
        undeclared = [m for m in parsed.members if not m.declared]
        self.assertEqual([m.name for m in undeclared], ["colour"])

    def test_a_baked_index_without_a_mesh_hash_is_flagged(self) -> None:
        parsed = facet(FeatureSpec(name="patch", kind="set", tier="literal", payload={"indices": [1, 2, 3]}))
        self.assertTrue(parsed.pins_mesh)
        self.assertEqual(parsed.mesh_sha, "")


class TestABespokeKindIsLegalAndSaysSo(unittest.TestCase):
    """An unregistered kind is the duck-typed freedom, not a mistake.

    `is_registered_kind` returning False means nothing states what the feature's
    handlers owe -- which is a real difference from a contracted kind and worth
    showing, and is not the same as the feature being wrong.
    """

    def setUp(self) -> None:
        self.iface = interface(_record(FeatureSpec(name="spout", kind="pouring_lip", tier="literal", payload={"r": 1})))

    def test_it_parses(self) -> None:
        self.assertEqual(self.iface.facets[0].kind, "pouring_lip")

    def test_it_is_marked_uncontracted(self) -> None:
        self.assertFalse(self.iface.facets[0].contracted)
        self.assertEqual(self.iface.uncontracted, ("pouring_lip",))

    def test_its_payload_survives_with_nothing_claimed_about_it(self) -> None:
        member = self.iface.facets[0].members[0]
        self.assertEqual((member.name, member.value), ("r", 1))
        self.assertFalse(member.declared, "an unregistered kind declares nothing, so nothing is declared")
        self.assertEqual(self.iface.facets[0].contracts, ())


@needs_content_assets
class TestTheWholeCorpusReadsWithoutAssumingAShape(unittest.TestCase):
    """The sweep. Every shipped asset, parsed on its own terms.

    The corpus is content's -- the library declares no record of its own -- so
    on a tree with no collection this class skips rather than sweep nothing and
    call it a pass; the synthetic half above is the whole of the claim there.

    Structural on purpose: the point is not that any one asset is right, it is
    that a single reader survives twelve kinds and three tiers across assets that
    have nothing in common -- which is the property the page depends on and the
    one that breaks the first time somebody assumes a payload has fields in it.
    """

    def test_every_registered_asset_parses(self) -> None:
        for name in list_assets():
            with self.subTest(name):
                iface = interface(get_asset(name))
                self.assertEqual(len(iface.facets), len(get_asset(name).features))

    def test_the_corpus_declares_nothing_undeclared(self) -> None:
        """What the checker already refuses, asserted where it is visible."""
        for name in list_assets():
            for parsed in interface(get_asset(name)).facets:
                with self.subTest(f"{name}.{parsed.name}"):
                    self.assertEqual(parsed.extra, ())
                    self.assertEqual(parsed.missing, ())

    def test_a_facet_is_reachable_by_name_and_by_kind(self) -> None:
        port = interface(get_asset("usb_c_port"))
        self.assertEqual(port.facet("bore").kind, "cavity")
        self.assertEqual([f.name for f in port.of_kind("cavity")], ["bore"])

    def test_asking_for_a_feature_an_asset_lacks_names_what_it_has(self) -> None:
        port = interface(get_asset("usb_c_port"))
        with self.assertRaises(KeyError) as caught:
            port.facet("handle")
        self.assertIn("bore", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
