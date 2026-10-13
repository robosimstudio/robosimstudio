# SPDX-License-Identifier: Apache-2.0
"""The task language: reading a document, checking it, and writing one back.

A task is a **document**, not Python. This package is everything that turns the
text into a checked :class:`~robosimstudio_bench.taskdoc.ast.TaskDoc` -- and nothing
that turns a ``TaskDoc`` into a scene, which is the layer above's job. The split
is what lets an authoring agent parse, typecheck and lint a task it has just
written without building anything.

The language is BDDL, extended where standard BDDL cannot say what this stack
needs to say. Not a second surface beside a manifest -- one language, so there is
no canonical form for a dialect to lag behind.

The split follows PDDL's own, and it is the split this design already needed:

**Domain** -- what kinds of thing exist. Referenced, not written here: an asset's
attributes live on its ``AssetRecord`` in the asset library, in Python, because
an asset also carries *code* (handlers, resolvers, entity classes) and a
declaration that can be validated against its implementation beats one that
cannot. What the domain contributes to a task file is a vocabulary: the types,
measures and states a problem may name. :mod:`~robosimstudio_bench.taskdoc.typecheck`
reaches it through a ``Catalog`` protocol, so the language never imports the
asset library and the two are free to move independently.

**Problem** -- what to do. Objects, staging, init, goal, reward, artifacts. This
is the file an author writes and an agent emits.

The extensions are **additive**. A reader that understands only standard BDDL
still gets coherent ``:objects``, ``:init`` and ``:goal``; nothing we add changes
the meaning of anything standard. What they add is what BDDL genuinely lacks:
graded comparisons over measures (a garment is never entirely inside a bin),
temporal modals (turns *retained after release*), a reward clause separate from
the goal (a depth over its gate is not a share of satisfied predicates),
staging (which cell, which robot -- LIBERO supplies these by harness convention),
and dotted references into an object's own features.
"""

from __future__ import annotations

from .ast import Condition, Instance, Ref, Region, SuiteDoc, TaskDoc
from .bindings import VAB_INDUSTRIAL_V1, BindingTable, PredicateBinding, apply_bindings
from .parse import MODALS, merge_suite, parse_problem, parse_ref, parse_suite, read_problem, read_suite
from .reader import ParseError, Sexp, parse_sexp, read_sexp
from .sidecar import load_sidecar, wxyz_to_xyzw
from .typecheck import CATALOG_GAP, Catalog, CheckProblem, check

__all__ = [
    "CATALOG_GAP",
    "MODALS",
    "VAB_INDUSTRIAL_V1",
    "BindingTable",
    "Catalog",
    "CheckProblem",
    "Condition",
    "Instance",
    "ParseError",
    "PredicateBinding",
    "Ref",
    "Region",
    "Sexp",
    "SuiteDoc",
    "TaskDoc",
    "apply_bindings",
    "check",
    "load_sidecar",
    "merge_suite",
    "parse_problem",
    "parse_ref",
    "parse_sexp",
    "parse_suite",
    "read_problem",
    "read_sexp",
    "read_suite",
    "wxyz_to_xyzw",
]
