# SPDX-License-Identifier: Apache-2.0
"""The parsed shape of a task file, before anything is resolved.

Frozen dataclasses, and **registry-free on purpose**. Parsing asks only "is this
a well-formed task file"; whether ``cable_rod`` is an asset that exists, whether
``winding-turns`` is an atom anything implements, and whether a peg has an axis
to wind about are all questions for the typechecker, which needs a catalog.

Keeping them apart buys two things. Schema errors are testable with no GPU, no
scene and no assets registered -- which is what lets the fast tier own them. And
the two failures stay legible as different things: "you wrote this file wrong" is
a different message from "the world does not contain what you asked for", and an
agent acts on them differently. The second is a *catalog gap*, and is the most
useful thing an authoring loop can be told.

Everything a parse cannot make sense of is kept rather than dropped. An
unrecognised section lands in ``extras`` with its source position, so a lowering
step can refuse it by name and a round-trip can put it back. Silently discarding
a clause would mean exporting a different task under the same name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Condition",
    "Instance",
    "Ref",
    "Region",
    "SuiteDoc",
    "TaskDoc",
]


@dataclass(frozen=True)
class Ref:
    """A reference into the world: ``cable``, ``peg0.axis``, ``cable.point(0.63)``.

    Kept as its parts rather than a string, because every consumer wants the
    parts: the typechecker resolves ``base`` against the catalog and ``path``
    against that asset's features, and a describe pass wants to name them
    separately.

    ``arg`` carries a parameterised feature's parameter -- the ``0.63`` in
    ``cable.point(0.63)``. Arc length is the only one today; it is a float rather
    than a general expression because a reference that can compute is a reference
    nobody can validate.
    """

    base: str
    path: tuple[str, ...] = ()
    arg: float | None = None

    def __str__(self) -> str:
        text = ".".join((self.base, *self.path))
        return f"{text}({self.arg})" if self.arg is not None else text


@dataclass(frozen=True)
class Condition:
    """One clause of an init or a goal.

    Covers both shapes BDDL needs and keeps them distinguishable:

    * a **predicate** -- ``(adhered tape)`` -- boolean, standard BDDL;
    * a **comparison** -- ``(>= (winding-turns cable peg0.axis) 0.5)`` -- graded,
      our extension, and the one deformables cannot do without.

    Attributes:
        head: The predicate or measure name.
        args: Its operands.
        op: The comparison, empty for a bare predicate.
        value: What it is compared against.
        modal: A temporal wrapper (``once``, ``hold-for``, ``after``, ``peak``),
            empty for none.
        modal_arg: The wrapper's own argument -- seconds for ``hold-for``, the
            event for ``after``.
        children: Sub-conditions, for ``and`` / ``or`` / a modal's body.
    """

    head: str
    args: tuple[Ref, ...] = ()
    op: str = ""
    value: float | None = None
    modal: str = ""
    modal_arg: Any = None
    children: tuple[Condition, ...] = ()

    @property
    def is_comparison(self) -> bool:
        return bool(self.op)

    # ----------------------------------------------------------- operators
    #
    # A goal written in Python and a goal parsed from a document meet *here*:
    # both are `Condition` trees, and `build_goal` binds either one without
    # knowing which it came from. That is what makes BDDL a layer over this
    # rather than a second surface beside it -- and it is why the operators
    # build conditions rather than evaluator nodes, which cannot exist until
    # there is a catalog and a scene to bind against.
    #
    # The vocabulary stays closed. `and`, `or` and `not` are heads `build_goal`
    # already handles; these spell them, they do not add to them.

    def __and__(self, other: Condition) -> Condition:
        """``a & b`` -- ``(and a b)``.

        Chains flatten. `AllOf`'s reward is the share of its children satisfied,
        so `(a & b) & c` and `a & b & c` agree on the verdict and disagree on
        the number -- 1/2 against 2/3 -- and the flat one is what the chain
        reads as. Only a bare ``and`` is absorbed: one carrying a modal or a
        comparison is a clause in its own right.
        """
        if self.head == "and" and not (self.modal or self.op):
            return Condition(head="and", children=(*self.children, other))
        return Condition(head="and", children=(self, other))

    def __or__(self, other: Condition) -> Condition:
        """``a | b`` -- ``(or a b)``. Flattens for the same reason."""
        if self.head == "or" and not (self.modal or self.op):
            return Condition(head="or", children=(*self.children, other))
        return Condition(head="or", children=(self, other))

    def __invert__(self) -> Condition:
        """``~a`` -- ``(not a)``, which is standard BDDL and already lowered."""
        return Condition(head="not", children=(self,))

    def __bool__(self) -> bool:
        """Always raises.

        `Condition` is frozen and compares by value, so ``if clause:`` would
        otherwise be quietly true and a chained comparison quietly wrong:
        Python reads ``0.2 <= x <= 0.8`` as ``(0.2 <= x) and (x <= 0.8)`` and
        ``and`` returns the *left* operand's truth, discarding the right half of
        the goal.
        """
        raise TypeError(
            "a goal clause is a tree, not a boolean: use `&`, `|` and `~`, and write "
            "`(a >= 1) & (a <= 2)` rather than `1 <= a <= 2`."
        )


@dataclass(frozen=True)
class Instance:
    """One object or fixture the task places."""

    name: str
    type: str = ""
    tags: tuple[str, ...] = ()
    static: bool = False


@dataclass(frozen=True)
class Region:
    """A BDDL region: a rectangle in its target's own frame.

    Ranges and the height window are kept exactly as written. The z window
    defaulting to 0-0.15 m above the frame origin is BDDL's own, and it is the
    convention that makes an object held *above* an open bin fail the test while
    one resting on its floor passes.
    """

    name: str
    target: str
    ranges: tuple[float, float, float, float]
    z_range: tuple[float, float] = (0.0, 0.15)


@dataclass(frozen=True)
class TaskDoc:
    """A parsed task file. Structure only -- nothing here has been resolved."""

    name: str
    language: str = ""
    objects: tuple[Instance, ...] = ()
    regions: tuple[Region, ...] = ()
    init: tuple[Condition, ...] = ()
    goal: Condition | None = None
    reward: dict[str, Any] = field(default_factory=dict)
    observe: tuple[Condition, ...] = ()
    staging: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    suite: str = ""
    """The ``(:suite NAME)`` this variant inherits from, or empty. See :class:`SuiteDoc`."""
    policy_params: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    extras: tuple[tuple[str, Any], ...] = ()
    hosts: tuple[dict[str, Any], ...] = ()
    """``(:hosts (:host ...) ...)``: foreign pairings that can stand in this scene, each a plain
    clause dict -- scene, embodiment (+arms), gripper, scene-resident, static-override, notes."""
    """Sections this parser does not model, kept verbatim.

    Dropping an unrecognised clause would mean a round-trip that silently
    exports a different task than it imported. Keeping it lets a lowering step
    refuse it *by name*, which is a message somebody can act on.
    """

    def instance(self, name: str) -> Instance:
        for item in self.objects:
            if item.name == name:
                return item
        raise KeyError(f"task {self.name!r} declares no object {name!r}; it has {[o.name for o in self.objects]}")


@dataclass(frozen=True, kw_only=True)
class SuiteDoc:
    """What every variant of one suite shares, stated once: a ``<name>.suite`` file.

    A multi-variant suite has three columns -- shared by all, shared by a job
    family, per variant -- and the document language had a construct for the
    second (a named arrangement) and the third (the ``.task`` itself) and none
    for the first.

    ``(define (suite NAME) ...)`` holds the first column. A variant that says
    ``(:suite NAME)`` -- or simply sits beside the file -- inherits it: its own
    ``:staging`` clauses override the suite's key by key, its ``:objects`` and
    ``:regions`` are added to the suite's ``:fixtures`` and ``:regions``, and its
    ``:artifacts`` override the suite's by kind. One level, no chains.

    Attributes:
        name: The suite's registry name.
        language: One line for listings -- the suite's description.
        staging: The shared ``:staging``, in the parser's plain shape.
        fixtures: Shared scenery every variant stands in.
        regions: Shared named regions.
        artifacts: Shared ``:artifacts`` -- typically ``(:tools ...)``.
        provenance: Where the benchmark this reproduces came from.
    """

    name: str
    language: str = ""
    staging: dict[str, Any] = field(default_factory=dict)
    fixtures: tuple[Instance, ...] = ()
    regions: tuple[Region, ...] = ()
    artifacts: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    variants: tuple[str, ...] = ()
    """``(:variants a b c)``: the suite's order, headline first. Empty is file order."""
    hosts: tuple[dict[str, Any], ...] = ()
    """Foreign pairings every variant inherits; see :attr:`TaskDoc.hosts`."""
