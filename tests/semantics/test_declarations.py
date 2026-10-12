# SPDX-License-Identifier: Apache-2.0
"""Features, kinds, handlers, atoms and published states, as declarations.

Everything here is structural -- no sheet, no model, no GPU. That is the point:
an agent-authored annotation should be refused for being malformed long before
anything is built, and these are the checks that do the refusing.

The two rules worth reading the tests for, because both fail *silently* when they
are wrong:

* **A feature referencing discretisation must pin the mesh hash.** A baked
  particle index means nothing after a re-mesh, and this repo's shirt was
  re-meshed deliberately. An unpinned index is how an annotation rots without
  anybody noticing.
* **A gate must cite the populations on both sides of it.** The best threshold in
  the repo records what its measure reads on outcomes that should pass, what it
  reads on outcomes that should fail, and where the valley between them is. A
  gate citing only its positives has not shown a separation -- it is a wish.
"""

from __future__ import annotations

import unittest

from robosimstudio.assetlib.semantics import (
    ArgSpec,
    AtomCalibration,
    AtomSpec,
    FeatureKind,
    FeatureSpec,
    PublishedState,
    atom_to_dict,
    crossable,
    dispatch,
    feature_from_dict,
    feature_to_dict,
    handler,
    handlers_for,
    is_substrate,
    register_kind,
)

TIER = "fast"
"""No GPU. Gates every commit."""

LOOP = FeatureKind(
    name="test_loop",
    summary="A closed opening something rod-shaped can pass through.",
    payload={"center": "centre in the body frame [m]", "normal": "disc normal", "inner_r": "clear radius [m]"},
    optional={"thickness": "material thickness [m]"},
    contracts=("threading is geometric, never contact attribution",),
    provenance="tests",
)
register_kind(LOOP)


class TestSubstrates(unittest.TestCase):
    def test_the_set_is_physics_not_taxonomy(self) -> None:
        for name in ("rigid", "articulated", "cloth"):
            self.assertTrue(is_substrate(name))
        self.assertFalse(is_substrate("scissors"), "a category is never a substrate")

    def test_a_layout_one_kind_owns_is_minted_by_its_pack(self) -> None:
        """The arc-length chain is the rod's: a layout exactly when the rod pack is
        installed, whatever has or has not been imported before the question."""
        from tests.fixtures.packs import HAS_ROD_PACK

        self.assertEqual(HAS_ROD_PACK, is_substrate("rod"))


class TestFeatureTiers(unittest.TestCase):
    def test_a_literal_feature_is_checked_against_its_kind(self) -> None:
        good = FeatureSpec(
            name="hole",
            kind="test_loop",
            payload={"center": [0.012, 0.0, -0.041], "normal": [0, 1, 0], "inner_r": 0.014},
        )
        self.assertEqual(good.problems(), [])

    def test_a_missing_payload_field_is_named(self) -> None:
        spec = FeatureSpec(name="hole", kind="test_loop", payload={"center": [0, 0, 0]})
        problems = " ".join(spec.problems())
        self.assertIn("requires", problems)
        self.assertIn("inner_r", problems)

    def test_an_unknown_payload_field_is_refused_not_ignored(self) -> None:
        """A silently dropped typo yields a feature that quietly kept a default."""
        spec = FeatureSpec(
            name="hole",
            kind="test_loop",
            payload={"center": [0, 0, 0], "normal": [0, 1, 0], "inner_r": 0.01, "raduis": 0.02},
        )
        self.assertIn("raduis", " ".join(spec.problems()))

    def test_an_unregistered_kind_is_legal_but_uncontracted(self) -> None:
        """Bespoke per-object features are the point of an open set. Nothing
        refuses them; they simply cannot claim a registered kind's generality."""
        spec = FeatureSpec(name="whatsit", kind="some_bespoke_kind", payload={"anything": 1})
        self.assertEqual(spec.problems(), [])

    def test_a_recipe_feature_validates_its_selector(self) -> None:
        good = FeatureSpec(name="hem", kind="set", tier="recipe", payload={"op": "extreme", "axis": 0, "band": 0.03})
        self.assertEqual(good.problems(), [])
        bad = FeatureSpec(name="hem", kind="set", tier="recipe", payload={"op": "nope"})
        self.assertIn("unknown selector op", " ".join(bad.problems()))

    def test_a_resolver_feature_must_name_its_resolver(self) -> None:
        spec = FeatureSpec(name="collar", kind="set", tier="resolver", payload={})
        self.assertIn("resolver", " ".join(spec.problems()))
        ok = FeatureSpec(name="collar", kind="set", tier="resolver", payload={"resolver": "collar_curve@v1"})
        self.assertEqual(ok.problems(), [])

    def test_an_unknown_tier_is_refused(self) -> None:
        self.assertIn("unknown tier", " ".join(FeatureSpec(name="x", kind="set", tier="magic").problems()))


class TestTheMeshShaRule(unittest.TestCase):
    """Anything referencing discretisation must pin the layout it was taken from."""

    def test_a_baked_index_list_without_a_hash_is_refused(self) -> None:
        spec = FeatureSpec(name="tab", kind="set", tier="literal", payload={"indices": [1, 2, 3]})
        self.assertTrue(spec.references_discretisation)
        self.assertIn("mesh_sha", " ".join(spec.problems()))

    def test_a_baked_index_list_with_a_hash_is_fine(self) -> None:
        spec = FeatureSpec(name="tab", kind="set", tier="literal", payload={"indices": [1, 2, 3]}, mesh_sha="91f2abcd")
        self.assertEqual(spec.problems(), [])

    def test_an_indices_op_nested_in_a_recipe_is_caught_too(self) -> None:
        """The case a shallow check would miss."""
        spec = FeatureSpec(
            name="panel",
            kind="set",
            tier="recipe",
            payload={"op": "any", "of": [{"op": "indices", "of": [4, 5]}]},
        )
        self.assertTrue(spec.references_discretisation)
        self.assertIn("mesh_sha", " ".join(spec.problems()))

    def test_a_geometric_recipe_needs_no_hash(self) -> None:
        """Stated in metres against the rest extent, so it survives a re-mesh --
        which is the entire reason to prefer recipes."""
        spec = FeatureSpec(name="left", kind="set", tier="recipe", payload={"op": "half", "axis": "y", "side": -1})
        self.assertFalse(spec.references_discretisation)
        self.assertEqual(spec.problems(), [])


class TestFeatureRoundTrip(unittest.TestCase):
    def test_to_dict_and_back(self) -> None:
        spec = FeatureSpec(
            name="hem", kind="set", tier="recipe", payload={"op": "half", "axis": 1, "side": -1}, provenance="measured"
        )
        self.assertEqual(feature_from_dict(feature_to_dict(spec)), spec)

    def test_unknown_fields_are_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            feature_from_dict({"name": "x", "kind": "set", "tiers": "recipe"})
        self.assertIn("tiers", str(caught.exception))


class TestHandlerDispatch(unittest.TestCase):
    """Most specific wins: task, then asset, then kind."""

    @classmethod
    def setUpClass(cls) -> None:
        @handler("test_hang", kinds=("test_loop", "rod"), summary="generic geometric threading")
        def _generic(*_a, **_k):
            return "kind"

        @handler("test_hang", kinds=("test_loop", "rod"), scope="asset", owner="scissors", summary="scissors-specific")
        def _asset(*_a, **_k):
            return "asset"

        @handler("test_hang", kinds=("test_loop", "rod"), scope="task", owner="demo/hang", summary="task-specific")
        def _task(*_a, **_k):
            return "task"

        @handler("test_hang", kinds=("test_loop", ""), summary="fallback with a wildcard operand")
        def _loose(*_a, **_k):
            return "loose"

    def test_kind_scope_is_the_default(self) -> None:
        self.assertEqual(dispatch("test_hang", ("test_loop", "rod")).fn(), "kind")

    def test_an_asset_shadows_the_kind(self) -> None:
        self.assertEqual(dispatch("test_hang", ("test_loop", "rod"), asset="scissors").fn(), "asset")

    def test_a_task_shadows_the_asset(self) -> None:
        got = dispatch("test_hang", ("test_loop", "rod"), asset="scissors", task="demo/hang")
        self.assertEqual(got.fn(), "task")

    def test_an_unrelated_owner_falls_back_to_the_kind(self) -> None:
        self.assertEqual(dispatch("test_hang", ("test_loop", "rod"), asset="mug").fn(), "kind")

    def test_the_more_constrained_signature_wins_within_a_scope(self) -> None:
        """A handler naming both operands beats one that wildcards an operand --
        what anybody writing a fallback expects to get."""
        self.assertEqual(dispatch("test_hang", ("test_loop", "rod")).fn(), "kind")
        self.assertEqual(dispatch("test_hang", ("test_loop", "hook")).fn(), "loose")

    def test_a_missing_implementation_says_what_it_has(self) -> None:
        """A common and legitimate authoring outcome: the object does not afford
        what the goal asked of it. The message is written for whoever wrote the
        goal, agent or human."""
        with self.assertRaises(LookupError) as caught:
            dispatch("test_hang", ("slot", "rod"))
        self.assertIn("no implementation", str(caught.exception))

    def test_an_unimplemented_atom_says_so(self) -> None:
        with self.assertRaises(LookupError) as caught:
            dispatch("test_nonexistent_atom", ("test_loop",))
        self.assertIn("no handler implements", str(caught.exception))

    def test_every_implementation_is_discoverable(self) -> None:
        """What the contract tests iterate -- generated from the registry, never
        from a list somebody has to remember to extend."""
        self.assertEqual(len(handlers_for("test_hang")), 4)


class TestAtomDeclarations(unittest.TestCase):
    def test_a_well_formed_atom_has_no_problems(self) -> None:
        spec = AtomSpec(
            name="test_winding",
            summary="Net turns of a chain about an axis.",
            args=(ArgSpec(name="body", substrates=("rod",)), ArgSpec(name="about", kinds=("axis",))),
            directions=("check", "observe"),
            returns=("turns", "engaged"),
            calibration=AtomCalibration(
                positives="0.55-0.96 turns retained over every shipped policy rollout",
                negatives="drag-pasts, slipped grasps and a sprung coil all measure 0.25 or less",
                gate=0.5,
                bench="scripts/invariance_check.py",
            ),
        )
        self.assertEqual(spec.problems(), [])
        self.assertTrue(spec.supports("check"))
        self.assertFalse(spec.supports("sample"))

    def test_a_gate_without_a_measured_separation_is_refused(self) -> None:
        """A threshold needs the populations on both sides of it, not just the
        ones it passes."""
        spec = AtomSpec(
            name="test_wishful",
            summary="something",
            returns=("value",),
            directions=("check",),
            calibration=AtomCalibration(positives="looked right in a few runs", gate=0.5),
        )
        self.assertIn("cites no measured separation", " ".join(spec.problems()))

    def test_an_atom_naming_an_instance_is_a_task_parameter_in_disguise(self) -> None:
        """The anti-overfit lint. ``hammer_in_correct_bin`` can never be reused by
        the next task or the next structured observation."""
        overfit = AtomSpec(name="hammer_in_correct_bin", summary="x", returns=("v",))
        general = AtomSpec(name="in_region", summary="x", returns=("v",))
        instances = ("hammer", "bin2", "tote")
        self.assertTrue(overfit.names_an_instance(instances))
        self.assertFalse(general.names_an_instance(instances))

    def test_the_catalog_form_is_json_able(self) -> None:
        spec = AtomSpec(name="test_x", summary="s", returns=("v",), directions=("check",))
        data = atom_to_dict(spec)
        self.assertEqual(data["name"], "test_x")
        self.assertEqual(data["returns"], ["v"])


class TestPublishedStates(unittest.TestCase):
    def test_a_well_formed_declaration(self) -> None:
        state = PublishedState(
            name="bond_fraction", type="float", summary="Share of adhesive bonds still intact.", range=(0.0, 1.0)
        )
        self.assertEqual(state.problems(), [])

    def test_the_tier_orders_observability(self) -> None:
        """Not a new access-control system -- the allowlist still refuses. This
        just lets it be stated per member rather than per tool."""
        internal = PublishedState(name="bond_force", type="float", summary="x", tier="internal")
        perceivable = PublishedState(name="lid_open", type="bool", summary="x", tier="perceivable")
        self.assertFalse(crossable(internal))
        self.assertTrue(crossable(perceivable))
        self.assertTrue(crossable(internal, allow="internal"))

    def test_an_unknown_tier_is_refused(self) -> None:
        state = PublishedState(name="x", type="float", summary="x", tier="secret")
        self.assertIn("unknown observability tier", " ".join(state.problems()))


if __name__ == "__main__":
    unittest.main()
