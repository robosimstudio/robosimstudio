# SPDX-License-Identifier: Apache-2.0
"""Where the asset system divides, and why a ``[core]`` install depends on it holding.

``assetlib`` says what an asset **is** -- its record, its semantics, its
handlers, its materials -- and needs no document language to say it.
``robosimstudio_bench.goals`` says what a benchmark asks **of** an asset --
binding, goal trees, evaluation, the gauntlet, the catalogue -- and eight of its
nine document-facing modules name :mod:`robosimstudio_bench.taskdoc` at module
scope (``mint`` reaches it through ``gauntlet``); three goal-side modules sit
beside them and name no parser at all. That line was measured before the two
were separated, and separating them is what made a core install possible: until
it happened, reaching *any* module of the asset system ran a document parser.

Nothing enforced the line while the two halves shared a directory, and one
module-scope ``from robosimstudio_bench.taskdoc import ...`` added to
``record.py`` would put the coupling straight back. The only thing that would
notice is a subprocess test three files away reporting it as "a bare scene needs
the benchmark". This says it where it happens.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

TIER = "fast"
"""Reads source; imports nothing."""

LIBRARY = Path(__file__).resolve().parents[1] / "robosimstudio"
CORE_HALF = LIBRARY / "assetlib"
GOAL_HALF = LIBRARY.parent / "robosimstudio_bench" / "goals"

GOAL_MODULES = frozenset({"bind", "catalog", "evaluate", "gauntlet", "goal", "initial", "mint", "query", "simcheck"})
"""The nine that moved. Eight name the parser directly; ``mint`` reaches it via ``gauntlet``.

``describe`` moved with them and was then deleted: turning simulator state into
English is a research variable, and this checkout keeps the numbers and hands
language back to whoever is asking (see ``docs/raw_capture.md``)."""
GOAL_SIDE = frozenset({"combinators", "context", "reward"})
"""The three that need no parser and that nothing in assetlib reads: the combinator
algebra, the reward clauses and the evaluator protocol. Here because everything that
reads them is a goal; ``states.py`` keeps only the closed comparison set."""

PARSER = "robosimstudio_bench.taskdoc"


def _module(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return ".".join(p for p in rel.parts if p != "__init__") or "__init__"


def _names_parser(path: Path) -> bool:
    """Whether this file imports the document language at *module scope*.

    Module scope only: a function-local import is the sanctioned way to cross a
    layer, and ``assetlib/registry.py`` uses it deliberately for
    ``bench.taskbuild.discovery``.
    """
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(PARSER):
            return True
        if isinstance(node, ast.Import) and any(a.name.startswith(PARSER) for a in node.names):
            return True
    return False


class TestTheTwoHalvesAreWhereTheySay(unittest.TestCase):
    def test_the_goal_half_is_in_bench(self) -> None:
        have = {_module(GOAL_HALF, p) for p in GOAL_HALF.glob("*.py")} - {"__init__"}
        self.assertEqual(GOAL_MODULES | GOAL_SIDE, have, f"robosimstudio_bench/goals holds {sorted(have)}")

    def test_none_of_it_is_left_in_assetlib(self) -> None:
        stale = sorted(m for m in GOAL_MODULES | GOAL_SIDE if (CORE_HALF / f"{m}.py").exists())
        self.assertEqual([], stale, f"still in assetlib: {stale}")

    def test_the_core_half_is_still_the_shape_this_describes(self) -> None:
        """A counter-test: an emptied directory would pass everything below."""
        self.assertGreater(len(list(CORE_HALF.rglob("*.py"))), 20)


class TestTheCoreHalfNeedsNoParser(unittest.TestCase):
    def test_no_assetlib_module_names_the_document_language(self) -> None:
        offenders = sorted(_module(CORE_HALF, p) for p in CORE_HALF.rglob("*.py") if _names_parser(p))
        self.assertEqual(
            [],
            offenders,
            "these say what an asset *is* and should need no document parser to say it:\n  "
            + "\n  ".join(offenders)
            + "\nMove the import inside the function that needs it, or move the module to robosimstudio_bench/goals.",
        )

    def test_the_goal_half_still_needs_it(self) -> None:
        """The other direction: a goal module that stops naming the parser is a
        module that may belong back in ``assetlib``, and should be looked at
        rather than left."""
        names = sorted(m for m in GOAL_MODULES if m != "mint" and not _names_parser(GOAL_HALF / f"{m}.py"))
        self.assertEqual(
            [], names, f"these no longer name {PARSER}; do they still belong in robosimstudio_bench/goals? {names}"
        )

    def test_mint_is_here_for_the_stated_reason(self) -> None:
        source = (GOAL_HALF / "mint.py").read_text()
        self.assertIn("gauntlet", source, "mint no longer reaches the goal half through gauntlet")
        self.assertFalse(
            _names_parser(GOAL_HALF / "mint.py"), "mint names the parser directly now; update the docstring"
        )


if __name__ == "__main__":
    unittest.main()
