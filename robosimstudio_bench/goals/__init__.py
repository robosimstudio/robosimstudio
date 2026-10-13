# SPDX-License-Identifier: Apache-2.0
"""The goal half of the asset system: what a benchmark asks *of* an asset.

Ten modules that used to live beside the asset records in
:mod:`robosimstudio.assetlib`, and were moved here for one measured reason:
nine of them name the ``.task`` language at module scope, and the tenth
(``mint``) reaches it through ``gauntlet``. An asset *record* -- its features,
its states, its materials, its handlers -- needs no document language to say
what a thing is, and a core install that had to carry a document parser to
build a cube was carrying this package by accident.

What is here:

* :mod:`~robosimstudio_bench.goals.bind` -- resolve a document's operands
  against a built scene's handles;
* :mod:`~robosimstudio_bench.goals.goal` -- compile a goal tree;
* :mod:`~robosimstudio_bench.goals.evaluate` -- the evaluator over it;
* :mod:`~robosimstudio_bench.goals.initial` -- whether a task starts where it
  says it does;
* :mod:`~robosimstudio_bench.goals.gauntlet`, :mod:`~robosimstudio_bench.goals.mint`
  -- lint a goal, gate a new task;
* :mod:`~robosimstudio_bench.goals.catalog`, :mod:`~robosimstudio_bench.goals.query`
  -- what an agent may ask;
* :mod:`~robosimstudio_bench.goals.simcheck` -- whether a staged arrangement
  can honour what the assets declare;
* :mod:`~robosimstudio_bench.goals.combinators` -- the closed set of nine
  ways to combine measures into a goal, and nothing else;
* :mod:`~robosimstudio_bench.goals.reward` -- reward, which is separate
  from the goal because many tasks score progress on a quantity, not on a
  share of satisfied predicates;
* :mod:`~robosimstudio_bench.goals.context` -- the evaluator protocol and the
  per-step verdict it returns.

The last three joined later, for the other measured reason: nothing in
``assetlib`` read them -- ``states.py`` needs only the closed comparison set,
which stayed beside the state it thresholds -- and everything that did is here
or above it.

They import the core half absolutely -- ``from robosimstudio.assetlib.record
import ...`` -- because a relative spelling from here would name a package
above the one it reaches into.
"""

from __future__ import annotations

_EXPORTS: dict[str, tuple[str, str]] = {
    "AssetCatalog": ("catalog", "AssetCatalog"),
    "BindError": ("bind", "BindError"),
    "Finding": ("gauntlet", "Finding"),
    "InitReport": ("initial", "InitReport"),
    "MintReport": ("mint", "MintReport"),
    "Operand": ("bind", "Operand"),
    "Report": ("gauntlet", "Report"),
    "bind": ("bind", "bind"),
    "build_goal": ("goal", "build_goal"),
    "catalog_json": ("catalog", "catalog_json"),
    "check_init": ("initial", "check_init"),
    "do_nothing_floor": ("gauntlet", "do_nothing_floor"),
    "gauntlet": ("gauntlet", "run"),
    "lint_goal": ("gauntlet", "lint_goal"),
    "make_evaluator": ("evaluate", "make_evaluator"),
    "mint_gate": ("mint", "mint_gate"),
    "mint_problems": ("mint", "mint_problems"),
    "resolve_operand": ("bind", "resolve_operand"),
    "starts_solved": ("initial", "starts_solved"),
}
"""Flat name -> (module, name in it), resolved on access.

The nineteen names ``robosimstudio.assetlib`` used to re-export and no
longer does -- they moved here with the modules that define them. Lazy for the
same reason the root package and ``assetlib`` are: importing this package
costs nothing, and ``from robosimstudio_bench.goals import build_goal`` costs
``goal.py`` and what it needs, not the gauntlet.
"""

__all__ = [
    "AssetCatalog",
    "BindError",
    "Finding",
    "InitReport",
    "MintReport",
    "Operand",
    "Report",
    "bind",
    "build_goal",
    "catalog_json",
    "check_init",
    "do_nothing_floor",
    "gauntlet",
    "lint_goal",
    "make_evaluator",
    "mint_gate",
    "mint_problems",
    "resolve_operand",
    "starts_solved",
]


def __getattr__(name: str):
    """Resolve one export, importing its module on first access (:pep:`562`)."""
    found = _EXPORTS.get(name)
    if found is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module, attribute = found
    value = getattr(importlib.import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(_EXPORTS)
