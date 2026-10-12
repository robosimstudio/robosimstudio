# SPDX-License-Identifier: Apache-2.0
"""The composer knows no domain by name.

``domains/`` ships six kinds of thing, and for a long time the composer knew all
six by name. Eleven sites across four modules matched a *string*: ``static`` is
scenery and owns no dynamics, ``articulated`` is the robot and may have hands,
``cloth`` or ``soft`` or ``rod`` is a deformable, anything else is an object,
and a table of six enumerated what each touches by default.

The cost was not the eleven lines. It was that a domain registered by **content**
fell through every branch: it played the ``objects`` role whatever it was, it
touched nothing by default, and it was never world-fixed. None of that raised.
A scene composed, stepped, and silently never collided.

Those facts are declared at ``register_domain`` now -- ``role``, ``contacts``,
``world_fixed`` -- and the composer reads them. So a domain nobody here has heard
of is composed on exactly the terms ``cloth`` is, and this module is what keeps
that true: a name creeping back into the composer is how it would stop being.

Checked by AST rather than by grep, so a domain named in a docstring or a
comment -- which is prose, and often the clearest way to explain a mechanism --
does not trip it. What is refused is a domain name used as a *value*.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

TIER = "fast"
"""No GPU. Gates every commit."""

COMPOSER = Path(__file__).resolve().parents[1] / "robosimstudio" / "scene" / "physics"

DOMAIN_NAMES = frozenset({"rigid", "cloth", "rod", "soft", "articulated", "static"})

ALLOWED: dict[tuple[str, str], str] = {
    ("interactions.py", "rigid"): (
        "the fallback for an entity that declares no domain at all, mirroring "
        "Entity.domain's own default -- not a branch on which domain it is"
    ),
}
"""Domain names the composer may still use as a value, and why.

A ratchet, in the shape this repo already uses: state the exemption with its
reason rather than loosening the rule, so the list is a countdown and an entry
that stops being true fails rather than lingering."""


def _string_constants(path: Path) -> list[tuple[int, str]]:
    """Every string literal used as a value, docstrings excluded."""
    tree = ast.parse(path.read_text())
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    docstrings.add(id(body[0].value))
        # A bare string expression is a field docstring, which this tree uses heavily.
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            docstrings.add(id(node.value))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    ]


class TestTheComposerKnowsNoDomainByName(unittest.TestCase):
    def test_no_module_branches_on_a_domain_name(self) -> None:
        offenders: list[str] = []
        for path in sorted(COMPOSER.glob("*.py")):
            for line, value in _string_constants(path):
                if value in DOMAIN_NAMES and ALLOWED.get((path.name, value)) is None:
                    offenders.append(f"{path.name}:{line}: {value!r}")
        self.assertEqual(
            offenders,
            [],
            "the composer names a domain as a value; declare the fact at register_domain "
            "and read it off DomainInfo, or add an exemption to ALLOWED with its reason:\n  " + "\n  ".join(offenders),
        )

    def test_the_exemptions_are_still_real(self) -> None:
        """An exemption that no longer applies is a rule quietly relaxed."""
        for (filename, value), reason in ALLOWED.items():
            with self.subTest(file=filename, name=value):
                self.assertTrue(reason.strip(), "an exemption without a reason is folklore")
                found = [ln for ln, v in _string_constants(COMPOSER / filename) if v == value]
                self.assertTrue(found, f"{filename} no longer uses {value!r}; drop the exemption")


class TestEveryDomainDeclaresItsBehaviour(unittest.TestCase):
    def test_each_registered_domain_states_a_role(self) -> None:
        """A domain that declares nothing gets the defaults, and the defaults are
        a choice -- ``objects``, touching nothing, not world-fixed. That is right
        for a free body and wrong for anything else, so every shipped domain
        should have said which it is."""
        from robosimstudio.specs.physics import DOMAINS, ROLES, get_domain

        for name in DOMAINS.names():
            with self.subTest(domain=name):
                info = get_domain(name)
                self.assertIn(info.role, ROLES)
                for role in info.contacts:
                    self.assertIn(role, ROLES, f"{name} touches {role!r}, which is not a role")

    def test_a_content_domain_is_composed_on_the_same_terms(self) -> None:
        """The claim the eleven sites used to break: a domain the library has
        never heard of gets a role, a default contact set and a world-fixed flag
        exactly as ``cloth`` does, because it declared them the same way."""
        from robosimstudio.scene.physics.interactions import default_contacts
        from robosimstudio.specs.physics import DOMAINS, register_domain

        try:
            register_domain(
                "probe_granular",
                state=("particles",),
                reads="cloth",
                role="deformables",
                contacts=("surface", "hands"),
                summary="A probe domain.",
            )
            rules = default_contacts("probe_granular")
            self.assertEqual(("surface", "hands"), tuple(r.with_ for r in rules))
        finally:
            DOMAINS._items.pop("probe_granular", None)
            DOMAINS._origins.pop("probe_granular", None)


if __name__ == "__main__":
    unittest.main()
