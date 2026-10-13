# SPDX-License-Identifier: Apache-2.0
"""The shipped assets are well formed, and malformed ones are refused.

Two halves. The first walks every registered asset and asserts the properties the
catalog depends on -- that it validates, that its kinds are ones something can
dispatch on, that its thresholds say where they came from. The second feeds the
validator things that are wrong on purpose, because a validator nobody has seen
refuse anything is a validator nobody should trust.

The refusals are the interesting half, and each one caught a real mistake while
these assets were being written rather than being invented afterwards: a payload
field the kind does not define, an arc-length point annotated on a rigid body, a
physics domain nothing registered.

**The catalog is content's, so the first half states two answers.** The library
declares no asset record of its own; a tree with a task collection on a root has
a corpus, and a tree that ships the library alone has none. On the second the
sweeps skip and say why -- a sweep over nothing proves nothing -- and the catalog
is asserted to be exactly empty, with lookups that still answer usefully.
"""

from __future__ import annotations

import unittest

from robosimstudio.assetlib import AssetRecord, SimRequirements, asset_to_dict, get_asset, list_assets
from robosimstudio.assetlib.registry import find_asset
from robosimstudio.assetlib.semantics import (
    KIND_NAMES,
    FeatureSpec,
    StateSpec,
    is_registered_kind,
    is_substrate,
    list_kinds,
)
from robosimstudio.assetlib.sim import contacts_with
from tests.fixtures.content import HAS_CONTENT_ASSETS, needs_content_assets
from tests.fixtures.packs import needs_rod_pack

TIER = "fast"
"""No GPU. Gates every commit."""


class TestTheShippedAssets(unittest.TestCase):
    """Everything in the catalog, checked the way the catalog will be read."""

    def test_the_catalog_is_contents_or_it_is_empty(self) -> None:
        """Both answers. With a collection on a root there is a corpus; with the
        library alone there is exactly nothing -- no library default, no stale
        copy -- and the three lookups still answer: an empty listing, ``None``
        from :func:`find_asset`, and a ``KeyError`` that says the registry is
        empty rather than that the name was misspelt."""
        if HAS_CONTENT_ASSETS:
            self.assertGreater(len(list_assets()), 3, "the registry loaded suspiciously few assets")
            return
        self.assertEqual(list_assets(), [])
        self.assertIsNone(find_asset("towel"))
        with self.assertRaises(KeyError) as caught:
            get_asset("towel")
        self.assertIn("registered: []", str(caught.exception))

    @needs_content_assets
    def test_every_asset_validates(self) -> None:
        """Registration already refuses a malformed asset; this proves the
        registry did not simply fail to load any."""
        for name in list_assets():
            with self.subTest(asset=name):
                self.assertEqual(get_asset(name).problems(), [])

    @needs_content_assets
    def test_every_asset_has_a_known_substrate(self) -> None:
        for name in list_assets():
            with self.subTest(asset=name):
                self.assertTrue(is_substrate(get_asset(name).substrate))

    @needs_content_assets
    def test_every_feature_kind_is_registered(self) -> None:
        """An unregistered kind is *legal* -- bespoke per-object features are the
        point of an open set -- but nothing shipped should need one. A shipped
        asset reaching for an uncontracted kind means a kind is missing from the
        library, not that the asset is special."""
        for name in list_assets():
            for spec in get_asset(name).features:
                with self.subTest(asset=name, feature=spec.name):
                    self.assertTrue(is_registered_kind(spec.kind), f"{spec.kind!r} has no contract")

    @needs_content_assets
    def test_every_state_says_where_its_number_came_from(self) -> None:
        """A threshold whose justification survives only as a comment is a
        threshold nothing can check -- and these will be read by an agent
        deciding whether its task is expressible, which is a worse audience for
        folklore than a human reviewer."""
        for name in list_assets():
            for state in get_asset(name).states:
                with self.subTest(asset=name, state=state.name):
                    self.assertTrue(state.provenance, "state gates carry provenance")

    @needs_content_assets
    def test_every_asset_is_json_able(self) -> None:
        """The catalog form is what an agent reads, so it must survive the trip."""
        import json

        for name in list_assets():
            with self.subTest(asset=name):
                self.assertIsInstance(json.dumps(asset_to_dict(get_asset(name))), str)

    @needs_content_assets
    def test_the_catalog_names_no_machine_path(self) -> None:
        """Files resolved beside their declaration load absolute; the catalog lists them relative.

        A catalog is pasted into an LLM prompt, and an absolute path there is the
        author's home directory.
        """
        from pathlib import Path

        for name in list_assets():
            for key, path in asset_to_dict(get_asset(name)).get("files", {}).items():
                with self.subTest(asset=name, file=key):
                    self.assertFalse(Path(path).is_absolute(), path)

    def test_a_file_beside_its_declaration_is_listed_relative_to_it(self) -> None:
        from types import SimpleNamespace

        from robosimstudio.assetlib.record import _portable_path

        record = SimpleNamespace(home="/home/someone/tasks/mug")
        self.assertEqual("meshes/mug.obj", _portable_path(record, "/home/someone/tasks/mug/meshes/mug.obj"))
        self.assertEqual("objects/mug.obj", _portable_path(record, "objects/mug.obj"))
        self.assertEqual("/elsewhere/mug.obj", _portable_path(record, "/elsewhere/mug.obj"))
        self.assertEqual("/x/mug.obj", _portable_path(SimpleNamespace(home=None), "/x/mug.obj"))

    def test_the_shipped_kinds_are_the_declared_ones(self) -> None:
        """``KIND_NAMES`` is the deliberate list; the registry is what happened.

        Kinds prefixed ``test_`` are excluded. Registries are process-global by
        design -- that is what makes lazy discovery work -- so a test that
        registers a fixture kind is visible to every other test in the same run.
        Comparing the two lists exactly would therefore make this test depend on
        which *other* tests ran first, which is a worse property than the one it
        is trying to check. The convention is the fix: a fixture kind says so in
        its name.

        The declared list has two halves. The asset library's ``KIND_NAMES`` is
        the general kinds; a kind only one kind of thing can carry is registered
        by that thing's domain pack, which keeps a ``KIND_NAMES`` of its own --
        the arc-length ``point`` is the rod's. So the registry is checked
        against the library's tuple plus every *installed* pack's, and a
        checkout without a pack is not expected to hold that pack's kinds.
        """
        import importlib
        import pkgutil

        import robosimstudio.domains as domains

        declared = set(KIND_NAMES)
        for info in pkgutil.iter_modules(domains.__path__):
            if not info.name.isidentifier():
                continue  # ``04_your_own_cloth.py``: an example beside a pack, as the registries' own scan skips it
            pack = importlib.import_module(f"robosimstudio.domains.{info.name}")
            declared |= set(getattr(pack, "KIND_NAMES", ()))
        shipped = [name for name in list_kinds() if not name.startswith("test_")]
        self.assertEqual(sorted(declared), shipped)
        self.assertNotIn("point", KIND_NAMES, "the arc-length point is the rod pack's kind, not the library's")


class TestTheCatalogAnswersUsefully(unittest.TestCase):
    """What an authoring agent actually asks it."""

    def test_a_missing_asset_names_what_exists(self) -> None:
        """The catalog-gap signal: 'the thing your instruction needs does not
        exist yet' is the single most useful thing an authoring loop is told.

        With a corpus the message lists it; with none it says the registry is
        empty, which is the same signal and the shortest list it can be."""
        with self.assertRaises(KeyError) as caught:
            get_asset("teapot")
        self.assertIn("cable_rod" if HAS_CONTENT_ASSETS else "registered: []", str(caught.exception))

    @needs_content_assets
    def test_a_missing_feature_names_what_the_asset_offers(self) -> None:
        with self.assertRaises(KeyError) as caught:
            get_asset("towel").feature("sleeve")
        message = str(caught.exception)
        self.assertIn("sleeve", message)
        self.assertIn("grasp", message)

    @needs_content_assets
    def test_features_are_findable_by_kind(self) -> None:
        """How a handler's dispatch asks the question."""
        self.assertTrue(get_asset("cable_peg").features_of_kind("rod"))
        self.assertEqual(get_asset("cable_peg").features_of_kind("bond_field"), ())


@needs_content_assets
class TestHangingGeneralisesByKind(unittest.TestCase):
    """The claim the whole prototype-style object model rests on.

    Nothing declares itself hangable. A rail carries a ``rod``; anything carrying
    a ``loop`` will thread onto it, and neither side names the other. Today no
    shipped asset has a loop -- which is exactly the state this test pins, so
    that adding a mug or a pair of scissors *later* needs no change here.
    """

    def test_the_fixtures_offer_rods_without_naming_what_hangs_on_them(self) -> None:
        """The structural surface only -- prose is deliberately exempt.

        What must not name a category is anything a dispatch or a document could
        depend on: tags, feature names, feature kinds. A ``provenance`` sentence
        explaining the design *by example* ("thin enough that a scissors hole, a
        mug handle and a keyring all clear it") is documentation, not coupling,
        and forbidding it would cost a useful explanation to buy nothing. An
        earlier version of this test compared the whole serialised blob and
        failed on exactly that sentence, which is the wrong lesson to teach.

        The asset's own registry name is exempt too, and one shipped fixture
        needs the exemption: the peg is called ``cable_peg``, inherited from the
        suite that first built it. A peg is a peg -- it would serve rope, hose or
        ribbon equally -- so the name is a mild instance of exactly the coupling
        this design objects to. It is an *identifier*, though, not a dispatch
        dependency: nothing in the peg's declaration references a cable, and a
        document naming it gets a peg. Recorded here rather than fixed, because
        renaming it means moving the entity registration the shipped suite still
        uses, which belongs to that suite's port.
        """
        # One fixture, where there used to be three. The rail and the wall hook
        # were invented to demonstrate that hanging generalises -- annotate a
        # `loop` on a mug and it hangs too -- and they have been removed along
        # with the scissors that made the demonstration concrete.
        #
        # What is left is weaker on purpose rather than by accident: the peg is a
        # SHIPPED asset, so this now asserts the property where it actually has
        # to hold. A demonstration proves a design can generalise; a shipped
        # asset proves this one does.
        for name in ("cable_peg",):
            with self.subTest(fixture=name):
                asset = get_asset(name)
                self.assertTrue(asset.features_of_kind("rod"), "a hanging fixture offers a rod")
                dispatch_surface = [*asset.tags, *(f.name for f in asset.features), *asset.kinds]
                for category in ("scissors", "mug", "keyring", "towel", "cable"):
                    for field in dispatch_surface:
                        self.assertNotIn(category, field, f"{name} couples to {category!r} via {field!r}")


class TestTheValidatorRefuses(unittest.TestCase):
    """Each of these was a real mistake made while writing the shipped assets."""

    def test_a_payload_field_the_kind_does_not_define(self) -> None:
        """Caught the cable's anchor declaring a grip the kind had no field for.

        Refused rather than dropped: a silently ignored key produces a feature
        that quietly kept a default, and that gets blamed on the physics.
        """
        record = AssetRecord(
            name="x",
            summary="s",
            features=(
                FeatureSpec(
                    name="r", kind="rod", payload={"axis": [[0, 0, 0], [1, 0, 0]], "radius": 0.01, "colour": "red"}
                ),
            ),
        )
        self.assertIn("colour", " ".join(record.problems()))

    @needs_rod_pack
    def test_a_kind_on_the_wrong_substrate(self) -> None:
        """Caught an arc-length ``point`` annotated on a rigid hook.

        The kind is the rod pack's, so this is asked only where the pack is: an
        unregistered kind is legal and uncontracted, and without the pack
        ``point`` is exactly that.

        Meaningless, and it would have resolved to nothing at scene-ready --
        the quietest possible failure. The kind knew which substrates it belongs
        on; until this check nobody asked it.
        """
        record = AssetRecord(
            name="x",
            summary="s",
            features=(FeatureSpec(name="tip", kind="point", payload={"s": 1.0}),),
        )
        self.assertIn("belongs on", " ".join(record.problems()))

    def test_a_domain_nothing_registered(self) -> None:
        """A domain is what the composer partitions on, so an unregistered one
        has no backend and no state kinds -- the asset would land in whatever
        island the fallback happened to give it."""
        record = AssetRecord(name="x", summary="s", sim=SimRequirements(domain="jelly"))
        self.assertIn("is not registered", " ".join(record.problems()))

    def test_self_contact_declared_two_ways_and_only_one(self) -> None:
        """A rule naming ``self`` and a ``self_contact`` flag are two spellings
        of one claim, and two spellings drift. The record says which."""
        record = AssetRecord(
            name="x",
            summary="s",
            sim=SimRequirements(domain="cloth", contacts=contacts_with("self"), self_contact=False),
        )
        self.assertIn("say one thing", " ".join(record.problems()))

    def test_two_features_with_one_name(self) -> None:
        record = AssetRecord(
            name="x",
            summary="s",
            sim=SimRequirements(domain="cloth"),
            features=(
                FeatureSpec(name="hem", kind="set", tier="recipe", payload={"select": {"op": "half", "axis": "y"}}),
                FeatureSpec(name="hem", kind="set", tier="recipe", payload={"select": {"op": "half", "axis": "x"}}),
            ),
        )
        self.assertIn("two features named", " ".join(record.problems()))

    def test_a_state_about_a_feature_that_is_not_there(self) -> None:
        record = AssetRecord(
            name="x",
            summary="s",
            sim=SimRequirements(domain="cloth"),
            states=(
                StateSpec(
                    name="folded",
                    measure="fold_extent",
                    op="<=",
                    value=0.6,
                    of="nonexistent",
                    summary="s",
                    provenance="p",
                ),
            ),
        )
        self.assertIn("no feature", " ".join(record.problems()))

    def test_an_unknown_domain_is_named_as_a_domain(self) -> None:
        """The substrate is derived, so an asset can no longer get one wrong --
        what it can still get wrong is the domain it derives from, and that is
        what the message has to name."""
        problems = " ".join(AssetRecord(name="x", summary="s", sim=SimRequirements(domain="jelly")).problems())
        self.assertIn("unknown physics domain", problems)

    def test_a_state_with_no_provenance(self) -> None:
        record = AssetRecord(
            name="x",
            summary="s",
            sim=SimRequirements(domain="cloth"),
            states=(StateSpec(name="flat", measure="cloth_coverage", value=0.9, summary="s"),),
        )
        self.assertIn("no provenance", " ".join(record.problems()))


if __name__ == "__main__":
    unittest.main()
