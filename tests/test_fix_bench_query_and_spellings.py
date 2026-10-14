# SPDX-License-Identifier: Apache-2.0
"""A query in the document's own spelling, dispatched the way bind dispatches;
one set of reward gate words; the mint gate listing the catalog once; and a
selector refused on an atom that reports nothing."""

import importlib
import unittest
from types import SimpleNamespace
from unittest import mock

from robosimstudio_bench.goals import mint, reward
from robosimstudio_bench.goals import query as query_module
from robosimstudio_bench.goals.bind import Operand
from robosimstudio_bench.goals.combinators import AllOf, Leaf
from robosimstudio_bench.taskdoc import typecheck
from robosimstudio_bench.taskdoc.ast import Ref
from robosimstudio_bench.taskdoc.parse import parse_condition
from robosimstudio_bench.taskdoc.reader import parse_sexp

TIER = "fast"

# The package re-exports a function named `gauntlet`, which shadows the module.
gauntlet = importlib.import_module("robosimstudio_bench.goals.gauntlet")


class TestAQueryReadsTheDocumentsSpelling(unittest.TestCase):
    def _query(self, atom, reported, *references, asset=""):
        spec = SimpleNamespace(args=(), reads_scene=False, returns=("depth", "lat"), primary="depth")
        handler = SimpleNamespace(fn=lambda *values: reported)
        with (
            mock.patch.object(query_module, "leaves_of", return_value=[]),
            mock.patch.object(query_module, "get_atom", return_value=spec) as get_atom,
            mock.patch.object(query_module, "dispatch", return_value=handler) as dispatch,
            mock.patch.object(query_module, "_layouts", return_value=()),
            mock.patch.object(query_module, "_operand", side_effect=lambda r, d, c: Operand(ref=Ref(r), asset=asset)),
            mock.patch.object(query_module, "read", return_value=None),
        ):
            out = query_module.query(SimpleNamespace(), atom, *references)
        return out, get_atom, dispatch

    def test_a_hyphen_is_the_registered_underscore(self):
        _, get_atom, dispatch = self._query("cloth-coverage", {"depth": 1.0})
        get_atom.assert_called_once_with("cloth_coverage")
        self.assertEqual(dispatch.call_args.args[0], "cloth_coverage")

    def test_a_selector_narrows_the_answer(self):
        out, get_atom, _ = self._query("insertion:lat", {"depth": 1.0, "lat": 0.2})
        get_atom.assert_called_once_with("insertion")
        self.assertEqual(out, {"lat": 0.2})

    def test_a_selector_the_atom_does_not_report_is_refused(self):
        with self.assertRaises(query_module.QueryRefused):
            self._query("insertion:mate", {"depth": 1.0})

    def test_a_scalar_return_is_keyed_by_the_primary(self):
        out, _, _ = self._query("insertion", 0.5)
        self.assertEqual(out, {"depth": 0.5})

    def test_the_asset_scope_is_passed_as_bind_passes_it(self):
        _, _, dispatch = self._query("insertion", {"depth": 1.0}, "plug", asset="widget")
        self.assertEqual(dispatch.call_args.kwargs.get("asset"), "widget")


class TestAQueryTakesTheTaskScope(unittest.TestCase):
    """bind dispatches under ``task or doc.name``; a query must, or a task-scoped
    override scores the goal with one handler and answers the query with another."""

    def _query(self, **kwargs):
        from robosimstudio.assetlib.semantics import handlers

        spec = SimpleNamespace(args=(), reads_scene=False, returns=("depth",), primary="depth")
        candidates = [
            handlers.HandlerSpec(atom="insertion", fn=lambda *v: {"depth": 0.0}),
            handlers.HandlerSpec(atom="insertion", fn=lambda *v: {"depth": 1.0}, scope="task", owner="peg_task"),
        ]
        with (
            mock.patch.object(query_module, "leaves_of", return_value=[]),
            mock.patch.object(query_module, "get_atom", return_value=spec),
            mock.patch.object(handlers, "handlers_for", return_value=candidates),
            mock.patch.object(query_module, "_layouts", return_value=()),
            mock.patch.object(query_module, "_operand", side_effect=lambda r, d, c: Operand(ref=Ref(r))),
            mock.patch.object(query_module, "read", return_value=None),
        ):
            return query_module.query(SimpleNamespace(), "insertion", "plug", **kwargs)

    def test_the_documents_name_is_the_task_scope(self):
        self.assertEqual(self._query(doc=SimpleNamespace(name="peg_task")), {"depth": 1.0})

    def test_an_explicit_task_wins(self):
        self.assertEqual(self._query(doc=SimpleNamespace(name="other"), task="peg_task"), {"depth": 1.0})

    def test_no_task_is_the_kind_scope(self):
        self.assertEqual(self._query(), {"depth": 0.0})


class TestOneSetOfGateWords(unittest.TestCase):
    def test_reward_reads_the_checkers_spellings(self):
        self.assertIs(reward._GATE_WORDS, typecheck._GATE_WORDS)
        self.assertIs(reward._MASK_WORDS, typecheck._MASK_WORDS)

    def test_the_checker_takes_a_gate_prefixed_atom_as_the_measure(self):
        # `gate_depth` is not a gate word; the checker used to skip it by prefix.
        seen = []
        with mock.patch.object(typecheck, "_visit_reward_form", side_effect=lambda form, *a: seen.append(form) or []):
            typecheck._check_reward(SimpleNamespace(reward={"progress": [["gate_depth", "plug"]]}), catalog=None)
        self.assertEqual(seen, [["gate_depth", "plug"]])

    def test_the_gauntlet_finds_the_measure_past_the_gate(self):
        section = {"progress": [[":gate-at", 0.1], [":gated_by", ["intact", "film"]], ["depth", "plug"]]}
        self.assertEqual(gauntlet._reward_measure(section), ("depth", ("plug",)))

    def test_the_gauntlet_leaves_are_the_goals(self):
        a, b = Leaf(fn=lambda ctx: (True, 1.0, {})), Leaf(fn=lambda ctx: (True, 1.0, {}))
        self.assertEqual(gauntlet._leaves(AllOf(of=(a, b))), (a, b))
        self.assertEqual(gauntlet._leaves(None), ())


class TestTheMintGateListsTheCatalogOnce(unittest.TestCase):
    def test_once(self):
        with (
            mock.patch.object(mint, "list_atoms", return_value=["a", "b", "c"]) as list_atoms,
            mock.patch.object(mint, "get_atom", return_value=SimpleNamespace()),
            mock.patch.object(mint, "mint_problems", return_value=[]) as problems,
            mock.patch.object(mint, "MintReport", side_effect=lambda *a, **k: None),
        ):
            mint.mint_gate(exempt={})
        self.assertEqual(list_atoms.call_count, 1)
        self.assertEqual({call.kwargs["among"] for call in problems.call_args_list}, {("a", "b", "c")})


class TestASelectorOnAnAtomThatReportsNothing(unittest.TestCase):
    def _problems(self, atom):
        condition = parse_condition(parse_sexp("(widget:lat)")[0])
        catalog = SimpleNamespace(atom=lambda name: atom)
        return typecheck._check_condition(condition, None, catalog, "goal", direction="check")

    def _atom(self, **extra):
        return SimpleNamespace(supports=lambda d: True, directions=("check",), args=(), **extra)

    def test_is_refused_as_bind_refuses_it(self):
        problems = self._problems(self._atom(returns=()))
        self.assertTrue(any(p.kind == typecheck.UNRESOLVED and "'lat'" in p.detail for p in problems), problems)

    def test_a_catalog_object_without_returns_is_not_checked(self):
        self.assertEqual(self._problems(self._atom()), [])


if __name__ == "__main__":
    unittest.main()
