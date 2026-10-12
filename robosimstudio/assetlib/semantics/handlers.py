# SPDX-License-Identifier: Apache-2.0
"""Handlers: free multimethod functions, dispatching on what things have.

A handler is the implementation behind an atom for one combination of operands.
``holding`` is one name with three implementations -- finger contacts closing on
a grasp box, a kinematic latch on a chain segment, particles pinched inside a
radius -- and which one runs is decided by what the object *has*, never by what
it claims to be.

Two design commitments are worth stating because both are easy to get wrong.

**Handlers are free functions, never methods.** ``on(a, b)`` belongs to neither
``a`` nor ``b``, and forcing it onto one of them is how object models acquire
their worst asymmetries. This is multimethod dispatch on the operands' feature
kinds and substrates -- the same dispatch the repo already performs by hand every
time a call site chooses between the rigid grasp test and the cloth one. Making
it automatic is the only change.

**Precedence is three-level, most specific wins.** A generic kind handler is
written once (``threaded(loop, rod)``); an asset may shadow it when one object is
genuinely special; a task may shadow that as a last resort. Python's own model,
and the escape hatch that means the open set never has to anticipate everything.
The promotion path runs the other way: a bespoke handler written twice is a
candidate for its kind, through the same gate an atom mint uses.

Contracts attach to **kinds**, not to a capability roster -- there being no
roster. An object's interface is whatever it declares, per object and duck-typed,
so it is not the thing an obligation can be hung on: what recurs across objects
is the lowercase kind, and only something that recurs is general enough to owe
anything. When a kind declares "containment returns a fraction", every handler
dispatching on it owes that, and the test suite generates the check from the
registry rather than from a hand-maintained list. That is what replaces
conformance testing, and it is why registering a kind is what earns it
generality: an unregistered kind is usable, but nothing says what its handlers
owe.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from robosimstudio.registry import Registry
from robosimstudio.specs.physics import layout_chain

__all__ = [
    "HANDLERS",
    "HandlerSpec",
    "dispatch",
    "handler",
    "handlers_for",
    "list_handlers",
]

SCOPES: tuple[str, ...] = ("kind", "asset", "task")
"""Precedence, least specific first. A lookup walks these in reverse."""


@dataclass(frozen=True, kw_only=True)
class HandlerSpec:
    """One implementation of one atom, for one operand signature.

    Attributes:
        atom: Which atom this implements.
        fn: The implementation. Pure over its snapshot and operands.
        kinds: Required feature kind per operand, in order. ``""`` is a wildcard,
            so an operand that is a whole object rather than a feature matches
            anything.
        substrates: Required substrate per operand, in order. ``""`` is a
            wildcard.
        scope: One of :data:`SCOPES`.
        owner: The asset or task key this is scoped to, empty for kind scope.
        summary: What this particular implementation does, since the interesting
            differences between implementations of one atom are exactly what a
            reader needs -- a jaw-stall grasp test and a particle-pinch grasp
            test agree on the question and disagree on all the evidence.
    """

    atom: str
    fn: Callable[..., Any]
    kinds: tuple[str, ...] = ()
    substrates: tuple[str, ...] = ()
    scope: str = "kind"
    owner: str = ""
    summary: str = ""

    @property
    def key(self) -> str:
        """Registry key: unique per (atom, signature, scope, owner).

        Both operand lists are padded to the same length before pairing. Zipping
        them directly looks equivalent and is not: a handler that declares kinds
        but no substrates -- which is most of them -- would zip against an empty
        tuple and produce an empty signature, so every implementation of one atom
        would collide on one key. The registry refuses duplicates, so that failed
        loudly the first time two handlers were registered; in a registry that
        merely overwrote, it would have silently shadowed instead.
        """
        width = max(len(self.kinds), len(self.substrates))
        kinds = list(self.kinds) + [""] * (width - len(self.kinds))
        substrates = list(self.substrates) + [""] * (width - len(self.substrates))
        signature = "+".join(f"{k or '*'}:{s or '*'}" for k, s in zip(kinds, substrates, strict=True))
        return f"{self.atom}@{self.scope}:{self.owner or '-'}[{signature}]"

    def matches(self, kinds: tuple[str, ...], substrates: tuple[str, ...]) -> bool:
        """Whether this implementation accepts operands of these kinds."""
        if self.kinds and len(self.kinds) != len(kinds):
            return False
        for required, got in zip(self.kinds, kinds, strict=False):
            if required and required != got:
                return False
        for required, got in zip(self.substrates, substrates, strict=False):
            if required and got and required != got:
                return False
        return True

    @property
    def specificity(self) -> int:
        """How constrained this implementation is.

        Ties within one scope are broken toward the more constrained signature:
        a handler naming both operands' kinds beats one that wildcards an operand,
        which is the behaviour anyone writing a fallback expects to get.
        """
        return sum(1 for k in self.kinds if k) + sum(1 for s in self.substrates if s)


HANDLERS: Registry[HandlerSpec] = Registry("handler", ("robosimstudio.domains",))
"""Every registered handler, keyed by :attr:`HandlerSpec.key`.

**Scans the domains**, because that is where a measurement lives: a handler
reads one substrate, and the package that owns that substrate owns the code
that reads it. It loaded no packages at all while every implementation sat in
``assetlib/handlers/`` and was imported as a side effect of importing this
layer -- which stopped being true the moment a measurement moved beside the
kind it measures, and would have failed *quietly*: a missing handler is a
legitimate outcome (an object does not afford that question), so nothing would
have raised. ``tests/assetlib/test_contracts.py`` generates its checks from this
registry and would simply have checked fewer things.

Content adds more through the task-root populator, as it always did."""


def handler(
    atom: str,
    *,
    kinds: tuple[str, ...] = (),
    substrates: tuple[str, ...] = (),
    scope: str = "kind",
    owner: str = "",
    summary: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering an implementation of ``atom``.

    Args:
        atom: The atom this implements.
        kinds: Required feature kind per operand, ``""`` to accept anything.
        substrates: Required substrate per operand, ``""`` to accept anything.
        scope: ``"kind"`` for the general implementation, ``"asset"`` or
            ``"task"`` to shadow it for one owner.
        owner: The asset or task key, for the shadowing scopes.
        summary: What this implementation does, and on what evidence.

    Returns:
        The function, unchanged -- so a handler stays directly callable and
        directly testable, which is what keeps its unit tests honest.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown handler scope {scope!r}; one of {sorted(SCOPES)}")
    if scope != "kind" and not owner:
        raise ValueError(f"handler scope {scope!r} needs an owner to shadow for")

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        spec = HandlerSpec(
            atom=atom, fn=fn, kinds=kinds, substrates=substrates, scope=scope, owner=owner, summary=summary
        )
        HANDLERS.register(spec.key, spec, origin=fn.__module__)
        return fn

    return decorate


def handlers_for(atom: str) -> list[HandlerSpec]:
    """Every registered implementation of one atom, most specific first.

    What the contract tests iterate: a kind's obligations are checked against
    every handler that dispatches on it, generated from the registry rather than
    from a list somebody has to remember to extend.
    """
    found = [spec for _, spec in HANDLERS if spec.atom == atom]
    return sorted(found, key=lambda s: (SCOPES.index(s.scope), s.specificity), reverse=True)


def list_handlers() -> list[str]:
    """Every registered handler key, sorted."""
    return HANDLERS.names()


def dispatch(
    atom: str,
    kinds: tuple[str, ...] = (),
    substrates: tuple[str, ...] = (),
    *,
    layouts: tuple[tuple[str, ...], ...] = (),
    asset: str = "",
    task: str = "",
) -> HandlerSpec:
    """Choose the implementation for one call.

    Walks the scopes from most specific to least -- task, then asset, then kind
    -- and within a scope prefers the more constrained signature. The first match
    wins.

    Raises:
        LookupError: When nothing matches, naming what was asked for and what is
            registered. A missing handler is a legitimate and common outcome
            during authoring -- it means an object does not afford what a goal
            asked of it -- so the message is written to be read by whoever wrote
            the goal, agent or human.
    """
    candidates = handlers_for(atom)
    if not candidates:
        raise LookupError(
            f"no handler implements atom {atom!r}; registered atoms with handlers: {_atoms_with_handlers()}"
        )

    # Each operand's fallback chain: its layout's own, then whatever its live
    # entity's capabilities add (``layouts``, per operand, from the caller that
    # has a snapshot). A capability layout ranks *below* the domain's, so a
    # native rod still beats a soft-domain thing that merely reads as one.
    chains = tuple(
        layout_chain(name, extra=(layouts[i] if i < len(layouts) else ()))
        if (name or (i < len(layouts) and layouts[i]))
        else ()
        for i, name in enumerate(substrates)
    )

    def fallback_depth(spec: HandlerSpec) -> int | None:
        """How far down the operands' fallback chains this handler sits, or ``None``.

        ``None`` means it does not match at all. Zero means every substrate it
        constrains is the operand's own layout, which is what every handler
        scored before layouts had chains -- so a corpus with no chain longer than
        one measures exactly as it did.
        """
        if spec.kinds and len(spec.kinds) != len(kinds):
            return None
        for required, got in zip(spec.kinds, kinds, strict=False):
            if required and required != got:
                return None
        depth = 0
        for index, required in enumerate(spec.substrates):
            got = substrates[index] if index < len(substrates) else ""
            if not required or not got:
                continue
            chain = chains[index] if index < len(chains) else (got,)
            if required not in chain:
                return None
            depth = max(depth, chain.index(required))
        return depth

    for scope, owner in (("task", task), ("asset", asset), ("kind", "")):
        if scope != "kind" and not owner:
            continue
        # Most specific layout first, and only then the sort ``handlers_for``
        # already applied. A ``volume`` operand matches a volume handler and a
        # cloth one, and both constrain exactly one substrate -- so specificity
        # cannot separate them and, without this, the fallback would win as often
        # as not. ``min`` is stable, so within one depth the existing precedence
        # (scope, then the more constrained signature) is untouched.
        matched = [
            (depth, spec)
            for spec in candidates
            if spec.scope == scope
            and (scope == "kind" or spec.owner == owner)
            and (depth := fallback_depth(spec)) is not None
        ]
        if matched:
            return min(matched, key=lambda pair: pair[0])[1]

    offered = sorted({"+".join(s.kinds) or "(any)" for s in candidates})
    raise LookupError(
        f"atom {atom!r} has no implementation for operand kinds {list(kinds)}"
        + (f" on substrates {list(substrates)}" if any(substrates) else "")
        + f"; it is implemented for {offered}"
    )


def _atoms_with_handlers() -> list[str]:
    return sorted({spec.atom for _, spec in HANDLERS})
