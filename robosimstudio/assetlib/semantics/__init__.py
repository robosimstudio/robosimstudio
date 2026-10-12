# SPDX-License-Identifier: Apache-2.0
"""The semantic layer: what things offer, and what a goal is written in.

This package holds the vocabulary the task language is built from, and nothing
that knows about any particular task. Three pieces, in dependency order:

* :mod:`~robosimstudio.assetlib.semantics.substrates` -- what kind of state the
  engine integrates for a thing. The only type an object must have.
* :mod:`~robosimstudio.assetlib.semantics.kinds`,
  :mod:`~robosimstudio.assetlib.semantics.features`,
  :mod:`~robosimstudio.assetlib.semantics.selectors`,
  :mod:`~robosimstudio.assetlib.semantics.resolve` -- features: what an object
  offers, declared per object, resolved once against the rest layout.
* :mod:`~robosimstudio.assetlib.semantics.atoms`,
  :mod:`~robosimstudio.assetlib.semantics.handlers` -- the measures, as free
  multimethod functions dispatching on feature kinds. Never methods on objects:
  ``on(a, b)`` belongs to neither ``a`` nor ``b``.

The closed set of ways to combine measures into a goal, and the reward clauses
over them, used to be the fourth piece; they are the benchmark tier's ``goals``
package now. Nothing here reads them, and everything that does is a goal.

**There is no capability-interface roster here.** No ``Graspable``, no
``Containing``, no ``Openable``. Objects carry features and handlers freely, in
the Python sense of "interface": duck-typed, per object, discovered rather than
declared. Recurrence lives in lowercase feature kinds, and handler contracts per
kind are the only conformance tests. Categories like "scissors" never generalise;
``loop`` does.

The vocabulary is deliberately small and grows sublinearly. The benchmark's whole
corpus -- rigid sorting, 0.15 mm insertion, adhesive peeling, cable winding,
garment folding -- runs on about twenty-five measures, because atoms ask
questions of *physics* and tasks recombine those questions rather than extending
them.
"""

from __future__ import annotations

from .atoms import (
    ATOMS,
    ArgSpec,
    AtomCalibration,
    AtomSpec,
    atom_to_dict,
    get_atom,
    list_atoms,
    register_atom,
)
from .features import TIERS, FeatureSpec, feature_from_dict, feature_to_dict
from .handlers import HANDLERS, HandlerSpec, dispatch, handler, handlers_for, list_handlers
from .kinds import KINDS, FeatureKind, get_kind, is_registered_kind, list_kinds, register_kind
from .library import KIND_NAMES
from .published import OBSERVABILITY, PublishedState, crossable, published_to_dict
from .resolve import (
    RESOLVERS,
    get_resolver,
    list_resolvers,
    register_resolver,
)
from .selectors import OPS, Selector, compile_selector, selector_to_dict, validate_selector
from .snapshot import Snapshot
from .states import StateSpec, state_to_dict
from .substrates import (
    SUBSTRATES,
    Jointed,
    Particled,
    Posed,
    PublishesState,
    is_substrate,
    layout_chain,
    reads_as,
)

__all__ = [
    "ATOMS",
    "HANDLERS",
    "KINDS",
    "KIND_NAMES",
    "OBSERVABILITY",
    "OPS",
    "RESOLVERS",
    "SUBSTRATES",
    "TIERS",
    "ArgSpec",
    "AtomCalibration",
    "AtomSpec",
    "FeatureKind",
    "FeatureSpec",
    "HandlerSpec",
    "Jointed",
    "Particled",
    "Posed",
    "PublishedState",
    "PublishesState",
    "Selector",
    "Snapshot",
    "StateSpec",
    "atom_to_dict",
    "compile_selector",
    "crossable",
    "dispatch",
    "feature_from_dict",
    "feature_to_dict",
    "get_atom",
    "get_kind",
    "get_resolver",
    "handler",
    "handlers_for",
    "is_registered_kind",
    "is_substrate",
    "layout_chain",
    "list_atoms",
    "list_handlers",
    "list_kinds",
    "list_resolvers",
    "published_to_dict",
    "reads_as",
    "register_atom",
    "register_kind",
    "register_resolver",
    "selector_to_dict",
    "state_to_dict",
    "validate_selector",
]
