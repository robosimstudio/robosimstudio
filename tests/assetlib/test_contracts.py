# SPDX-License-Identifier: Apache-2.0
"""Handler contracts, generated from the registry rather than listed by hand.

This is what replaces interface conformance testing in a design with no
interfaces. There is no ``Containing`` to implement and no base class to inherit,
so "does this implementation behave like the others" cannot be asked of a type.
It is asked of a **kind**: the ``region`` kind declares that containment returns a
fraction, and every handler dispatching on it owes that, whoever wrote it and
whenever.

The obligation is discovered, not enumerated. These tests walk
:data:`~robosimstudio.assetlib.semantics.handlers.HANDLERS` and check what they
find, so an asset registering a new implementation next month inherits the checks
without anybody remembering to extend a list -- which is the only version of this
that survives contact with a growing catalog.

The contract that matters most here is the one that would be easiest to get
wrong: **containment returns a fraction everywhere.** A rigid body answering
``True`` and a garment answering ``0.82`` would make every caller branch and
would break the first goal pointed at a different substrate.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.assetlib.semantics import (
    ATOMS,
    KINDS,
    handlers_for,
    is_registered_kind,
    list_atoms,
    list_handlers,
)
from robosimstudio.assetlib.semantics.handlers import HANDLERS
from robosimstudio.assetlib.vocabulary.regions import containment, region_bounds
from tests.fixtures.grids import sheet

TIER = "fast"
"""No GPU. Gates every commit."""

BIN = {"xy_ranges": [-0.15, -0.15, 0.15, 0.15], "z_range": [0.0, 0.20]}
"""A 30 cm bin, 20 cm deep, for the containment contract."""


class TestTheRegistryIsCoherent(unittest.TestCase):
    """Cheap structural checks over everything registered."""

    def test_every_handler_implements_a_catalogued_atom(self) -> None:
        """A handler for an atom nobody declared cannot be reached by a goal, and
        would sit there looking implemented."""
        atoms = set(list_atoms())
        for _, spec in HANDLERS:
            with self.subTest(handler=spec.key):
                self.assertIn(spec.atom, atoms)

    def test_every_atom_that_names_a_kind_has_a_handler_for_it(self) -> None:
        """An atom whose operands name kinds nothing implements is vocabulary a
        document can use and the runtime cannot answer."""
        from robosimstudio.assetlib.semantics import get_atom

        for name in list_atoms():
            atom = get_atom(name)
            if not any(arg.kinds for arg in atom.args):
                continue
            with self.subTest(atom=name):
                self.assertTrue(handlers_for(name), f"atom {name!r} names kinds but nothing implements it")

    def test_every_handler_declares_what_it_dispatches_on(self) -> None:
        """A handler constraining neither kind nor substrate matches everything,
        which makes precedence meaningless."""
        for _, spec in HANDLERS:
            if spec.atom.startswith("test_"):
                continue
            with self.subTest(handler=spec.key):
                self.assertTrue(
                    any(spec.kinds) or any(spec.substrates),
                    "a handler must say what it is for",
                )

    def test_an_operand_naming_a_geometric_kind_reads_it_as_a_feature(self) -> None:
        """The check that would have caught four latent crashes at once.

        An operand constrained to a registered kind is naming a *declared
        feature*, and the only reader that resolves one into world coordinates is
        the ``feature`` reader. Leave ``reads`` unset and it defaults to
        ``position`` -- so the handler is handed the body's position array and
        subscripts it with a string.

        That is not hypothetical. ``reads`` arrived after four handler modules
        were written, and three of them -- ``threaded``, ``winding_turns``,
        ``open_fraction`` -- were never retrofitted. All three bound cleanly and
        died at step time. Generated from the registry rather than listed, so an
        atom added next year inherits the obligation.
        """
        for _, atom in ATOMS:
            if atom.name.startswith("test_"):
                continue
            for arg in atom.args:
                if not arg.kinds or not all(is_registered_kind(k) for k in arg.kinds):
                    continue
                # `region` is the one registered kind that is not an asset
                # feature. A BDDL `:regions` entry is declared by the *document*
                # and lowers to a RegionHandle, so it has its own reader and its
                # own form -- the frame it converts from is a scene body's, not
                # an asset's. Every other kind is annotated on an asset.
                if set(arg.kinds) == {"region"}:
                    continue
                with self.subTest(atom=atom.name, operand=arg.name):
                    self.assertEqual(
                        arg.reads,
                        "feature",
                        f"{atom.name}.{arg.name} names kinds {list(arg.kinds)} but reads {arg.reads or 'position'!r}",
                    )

    def test_a_kind_carrying_geometry_says_how_its_frame_converts(self) -> None:
        """A feature is authored in its body's frame; the kind says which of its
        payload fields must rotate, and which must rotate *and* translate.

        Declare neither and the feature reader passes the field through
        untouched, which puts it at the world origin. The cable peg's winding
        axis was exactly this: declared at the origin in the peg's frame, read as
        the world origin, so every cable segment sat 0.58 m away, no segment
        counted as near, and the wrap goal read 0.0 turns for ever. A number,
        plausible, and no error.

        Kinds whose payload is all scalars, indices or names are exempt -- there
        is nothing to convert. The list is explicit so that adding a geometric
        field to one of them fails here.
        """
        no_geometry = {"anchor", "bond_field", "joint", "panel", "point", "region", "set", "template"}
        for name, kind in KINDS:
            if name.startswith("test_") or name in no_geometry:
                continue
            with self.subTest(kind=name):
                self.assertTrue(
                    kind.points or kind.directions,
                    f"{name} carries geometry ({sorted(kind.payload)}) and declares no frame conversion",
                )

    def test_registered_kinds_state_their_contracts(self) -> None:
        """Registration is what earns a kind its generality. A registered kind
        with no contract claims one and owes nothing."""
        for name, kind in KINDS:
            if name.startswith("test_"):
                continue
            with self.subTest(kind=name):
                self.assertTrue(kind.summary, "a kind describes itself")

    def test_handler_keys_are_unique(self) -> None:
        """The registry refuses duplicates, so this is really a check that the
        key encodes enough to tell two implementations apart -- an earlier
        version collapsed every handler of one atom onto one key."""
        keys = list_handlers()
        self.assertEqual(len(keys), len(set(keys)))
        self.assertGreater(len(keys), 5)


class TestTheContainmentContract(unittest.TestCase):
    """ "Containment returns a fraction, always" -- checked over every implementation.

    Generated from the registry: whichever substrates have a containment handler,
    each is fed the operands its substrate implies and its answer is checked
    against the same three obligations.
    """

    def operands_for(self, substrate: str):
        """Inputs of the shape that substrate's handler expects, all inside the bin."""
        if substrate in ("rigid", "articulated"):
            return np.array([0.0, 0.0, 0.10])
        if substrate == "cloth":
            points = sheet(cells=9, size=0.10)
            points[:, 2] = 0.10
            return points
        if substrate == "rod":
            return np.stack([np.linspace(-0.05, 0.05, 8), np.zeros(8), np.full(8, 0.10)], axis=1)
        raise AssertionError(f"no operand fixture for substrate {substrate!r}")

    def test_every_implementation_returns_a_fraction_in_range(self) -> None:
        for spec in handlers_for("containment"):
            substrate = next((s for s in spec.substrates if s), "")
            with self.subTest(handler=spec.key):
                got = spec.fn(self.operands_for(substrate), BIN)
                self.assertIn("fraction", got, "the contract names the key")
                self.assertIsInstance(got["fraction"], float, "a fraction, never a bool")
                self.assertGreaterEqual(got["fraction"], 0.0)
                self.assertLessEqual(got["fraction"], 1.0)

    def test_fully_inside_is_one_for_every_substrate(self) -> None:
        for spec in handlers_for("containment"):
            substrate = next((s for s in spec.substrates if s), "")
            with self.subTest(handler=spec.key):
                self.assertAlmostEqual(spec.fn(self.operands_for(substrate), BIN)["fraction"], 1.0)

    def test_fully_outside_is_zero_for_every_substrate(self) -> None:
        far = {"xy_ranges": [5.0, 5.0, 5.3, 5.3], "z_range": [0.0, 0.2]}
        for spec in handlers_for("containment"):
            substrate = next((s for s in spec.substrates if s), "")
            with self.subTest(handler=spec.key):
                self.assertAlmostEqual(spec.fn(self.operands_for(substrate), far)["fraction"], 0.0)

    def test_the_rigid_case_is_the_easy_case_of_the_same_question(self) -> None:
        """It returns 0.0 or 1.0 and loses nothing -- it does not return a bool."""
        rigid = next(s for s in handlers_for("containment") if "rigid" in s.substrates)
        got = rigid.fn(np.array([0.0, 0.0, 0.10]), BIN)["fraction"]
        self.assertIn(got, (0.0, 1.0))
        self.assertNotIsInstance(got, bool)

    def test_the_cloth_case_reports_a_genuine_fraction(self) -> None:
        """The reason the contract is a fraction at all: a garment draped over a
        bin rim is partly in, and no boolean is honest about that."""
        cloth = next(s for s in handlers_for("containment") if "cloth" in s.substrates)
        points = sheet(cells=11, size=0.40)  # wider than the bin, centred on it
        points[:, 2] = 0.10
        fraction = cloth.fn(points, BIN)["fraction"]
        self.assertGreater(fraction, 0.0)
        self.assertLess(fraction, 1.0)


class TestRegionsRideTheirFrame(unittest.TestCase):
    """A drawer's interior only means anything because it moves with the drawer."""

    def test_the_box_translates_with_its_origin(self) -> None:
        lo_home, hi_home = region_bounds(BIN)
        lo_out, hi_out = region_bounds(BIN, origin=(0.0, 0.40, 0.0))
        np.testing.assert_allclose(lo_out - lo_home, [0.0, 0.40, 0.0])
        np.testing.assert_allclose(hi_out - hi_home, [0.0, 0.40, 0.0])

    def test_a_bowl_stays_contained_as_the_drawer_slides(self) -> None:
        """The claim in one assertion. The bowl moves with the drawer, so it is
        in the drawer at both ends of the travel -- and a world-anchored box
        would have lost it."""
        for travel in (0.0, 0.20, 0.45):
            with self.subTest(travel=travel):
                lo, hi = region_bounds(BIN, origin=(0.0, travel, 0.0))
                bowl = np.array([0.0, travel, 0.10])
                self.assertAlmostEqual(containment(bowl, lo, hi), 1.0)

    def test_the_height_window_starts_at_the_frame_origin(self) -> None:
        """The convention that makes 'in the bin' mean what people expect: an
        object held above an open bin is not in it."""
        lo, hi = region_bounds(BIN)
        self.assertAlmostEqual(containment(np.array([0.0, 0.0, -0.05]), lo, hi), 0.0)
        self.assertAlmostEqual(containment(np.array([0.0, 0.0, 0.30]), lo, hi), 0.0)


if __name__ == "__main__":
    unittest.main()
