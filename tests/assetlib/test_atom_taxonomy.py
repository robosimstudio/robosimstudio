# SPDX-License-Identifier: Apache-2.0
"""The measurement taxonomy: what a subclass means, and what it must not mean.

The atom taxonomy is the system's vocabulary, so the properties worth checking
are the ones that would make it stop being one: a family that registers itself,
a leaf with no summary, an implication nobody verified, or -- the one that would
quietly undo the whole design -- inheritance leaking from contracts into
dispatch.

**Why the ``isinstance`` lint is here rather than in a style config.** Dispatch is
structural: an operand satisfies ``threaded``'s loop because it *has* a loop, not
because it *is* a Loop, which is what lets scissors and a mug and a keyring all
answer. Now that atoms are classes, writing ``isinstance(operand, ...)`` in a
handler is one keystroke away and reads as reasonable. It is the exact thing the
duck-typed design exists to prevent, so it is refused mechanically.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import robosimstudio.assetlib.vocabulary
import robosimstudio.domains
from robosimstudio.assetlib.semantics.atom import Atom, Property, Relation, SpatialRelation, family_of, leaves_of
from robosimstudio.assetlib.semantics.atoms import ArgSpec, get_atom, list_atoms

TIER = "fast"
"""No GPU. Gates every commit."""


class TestAFamilyDescribesAndALeafRegisters(unittest.TestCase):
    def test_the_families_register_nothing_themselves(self) -> None:
        """An abstract class is a kind of question, not a question."""
        registered = set(list_atoms())
        for family in (Atom, Relation, SpatialRelation, Property):
            with self.subTest(family=family.__name__):
                self.assertTrue(family.abstract)
                self.assertNotIn(family.__name__.lower(), registered)

    def test_a_leaf_registers_by_existing(self) -> None:
        class _Probe(Property):
            name = "probe_taxonomy_leaf"
            summary = "A leaf minted by a test."
            args = (ArgSpec(name="thing"),)
            returns = ("value",)

        try:
            self.assertIn("probe_taxonomy_leaf", list_atoms())
            self.assertEqual(get_atom("probe_taxonomy_leaf").summary, _Probe.summary)
            self.assertIs(_Probe.spec, get_atom("probe_taxonomy_leaf"))
        finally:
            _unregister("probe_taxonomy_leaf")

    def test_a_leaf_lands_in_its_family(self) -> None:
        class _Probe(SpatialRelation):
            name = "probe_taxonomy_family"
            summary = "A spatial leaf minted by a test."
            args = (ArgSpec(name="a"), ArgSpec(name="b"))

        try:
            self.assertEqual(family_of(_Probe), "SpatialRelation")
            self.assertIn(_Probe, leaves_of("SpatialRelation"))
        finally:
            _unregister("probe_taxonomy_family")
            leaves_of("SpatialRelation").remove(_Probe) if _Probe in leaves_of("SpatialRelation") else None

    def test_a_leaf_without_a_summary_is_refused(self) -> None:
        """The catalog is read by people and by models; a blank line helps neither."""
        with self.assertRaises(ValueError):

            class _Nameless(Property):
                name = "probe_taxonomy_blank"

    def test_the_shipped_atoms_are_all_in_a_family(self) -> None:
        """No leaf hangs directly off ``Atom``.

        Not tidiness: ``sim.list_relations`` answers by family, so an atom in no
        family is one an agent cannot discover by asking what it may ask.
        """
        placed = {leaf.name for group in ("Relation", "SpatialRelation", "Property") for leaf in leaves_of(group)}
        stray = sorted(leaf.name for leaf in leaves_of("Atom"))
        self.assertEqual(stray, [], f"these hang off Atom directly; classified: {sorted(placed)}")


class TestSubsumptionIsAClaimAndIsChecked(unittest.TestCase):
    def test_every_implication_names_a_registered_atom(self) -> None:
        for group in ("Relation", "SpatialRelation", "Property"):
            for leaf in leaves_of(group):
                for implied in leaf.implies:
                    with self.subTest(atom=leaf.name, implies=implied.name):
                        self.assertIn(implied.name, list_atoms())

    def test_on_implies_contact_and_the_handlers_agree_on_the_contact_set(self) -> None:
        """The claim, checked where it is cheapest to check: the code path.

        ``on``'s handler delegates its touching half to ``Contact.measure``
        rather than reading the contact set again, so the two cannot disagree
        about what touching means. That delegation *is* the implication, and this
        asserts it rather than trusting the comment above it.
        """
        from robosimstudio.domains.rigid.measures import placement

        source = Path(placement.__file__).read_text()
        body = source[source.index("def _rigid_on(") :]
        self.assertIn("Contact.measure(", body, "on must delegate its touching half, not re-read the contact set")


class TestInheritanceDoesNotReachDispatch(unittest.TestCase):
    def test_no_handler_asks_what_an_operand_is(self) -> None:
        """Dispatch is on what a thing HAS. ``isinstance`` is on what it IS.

        The one line that would quietly turn a duck-typed vocabulary into a
        nominal one, and it now looks reasonable because atoms are classes. So:
        refused in the handlers, by reading them.
        """
        offenders: list[str] = []
        for label, path in _handler_sources():
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "isinstance":
                    offenders.append(f"{label}:{node.lineno}")
        self.assertEqual(
            offenders,
            [],
            "a handler dispatching on a Python type has stopped dispatching on what the object has",
        )

    def test_the_lint_is_reading_something(self) -> None:
        """The lint's failure mode is silence: point it at an empty tree and it passes.

        The handlers moved once already -- out of ``assetlib/handlers`` into the
        shared vocabulary and each domain's ``measures`` -- so the roster it
        walks is asserted rather than assumed.
        """
        read = {label for label, _ in _handler_sources()}
        self.assertIn("vocabulary/coverage.py", read)
        self.assertTrue(
            any(label.startswith("rigid/measures/") for label in read),
            f"the substrate measures are not being read; roster: {sorted(read)}",
        )


def _handler_sources() -> list[tuple[str, Path]]:
    """Every module implementing a measure, wherever the refactor put it.

    The handlers were one package; they are now the shared vocabulary plus a
    ``measures`` module per domain, found the same way ``HANDLERS`` finds them --
    by walking ``robosimstudio.domains.__path__``, so a domain root outside this
    checkout is linted too. Each file is labelled relative to the root it was
    found under, because ``measures.py`` is a name several domains use.
    """
    found: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    vocabulary = Path(robosimstudio.assetlib.vocabulary.__file__).parent
    candidates: list[tuple[Path, Path]] = [(vocabulary.parent, path) for path in sorted(vocabulary.rglob("*.py"))]
    for domain_root in robosimstudio.domains.__path__:
        base = Path(domain_root)
        for path in sorted(base.rglob("*.py")):
            if path.stem == "measures" or path.parent.name == "measures":
                candidates.append((base, path))
    for base, path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        found.append((path.relative_to(base).as_posix(), path))
    return found


def _unregister(name: str) -> None:
    """Drop a test-minted atom. The registry has no public removal, by design."""
    from robosimstudio.assetlib.semantics.atoms import ATOMS

    ATOMS._items.pop(name, None)
    ATOMS._origins.pop(name, None)


class TestObservationsStayStructured(unittest.TestCase):
    def test_atoms_expose_values_without_a_language_renderer(self) -> None:
        from robosimstudio.domains.rigid.measures.placement import On

        self.assertTrue(On.spec.supports("observe"))
        self.assertFalse(hasattr(On, "narrate"))
        self.assertNotIn("narration", On.spec.__dataclass_fields__)

    def test_no_atom_claims_a_narration_direction(self) -> None:
        """The direction left the contract; nothing may still be declaring it.

        Upstream kept ``narrate`` and this checkout does not, so the way the two
        collide is a shipped atom asking for a direction ``AtomSpec`` no longer
        knows -- which fails at import, far from anything that reads like a
        taxonomy problem.
        """
        for name in list_atoms():
            with self.subTest(atom=name):
                self.assertNotIn("narrate", get_atom(name).directions)


if __name__ == "__main__":
    unittest.main()
