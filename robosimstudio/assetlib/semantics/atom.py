# SPDX-License-Identifier: Apache-2.0
"""The measurement layer: structured questions that can be asked of a scene.

An **atom** is a question the system knows how to ask. ``containment``,
``winding_turns``, ``insertion`` -- each names one thing that can be measured,
states what it takes and what it reports, and carries the evidence behind any
threshold it suggests. This module is where an atom is *declared*.

**Three roles, and keeping them apart is the whole design.**

=============  ==============================================================
Contract       What a relation or property means: operands, structured return
               keys, cost, and calibration. **Here.**
Measurement    How that contract is computed for one substrate and kind
               combination -- particles versus a rigid body versus a rod.
               Handler registrations, dispatched structurally.
Judgment       Thresholds and temporal operators over a measure (a goal), or a
               direct structured question (an agent's query). Neither may
               invent vocabulary; both may only use what is declared here.
=============  ==============================================================

The separation lets one atom set run in three directions -- ``check`` scores a
goal, ``sample`` places a scene, and ``observe`` returns a structured reading --
without three implementations that drift. Rendering that reading into language
is deliberately outside RoboSimStudio.

**What subclassing does, and what it deliberately does not.**

Inheritance organises contracts. A family carries operand shape, return
conventions, default smoothing and cost, so a new leaf gets its family's
contract rather than restating it. That is all it does.

It does **not** organise measurements: ``On`` does not inherit ``Contact``'s
handlers, it registers its own, and where it wants the touching half it calls
``Contact.measure`` explicitly. And it does not organise *objects*: dispatch
stays structural, on what an operand HAS rather than on what it IS, so any body
carrying a ``loop`` satisfies ``threaded``'s first operand -- scissors, a mug, a
keyring -- without declaring an interface or inheriting anything. A lint refuses
``isinstance`` on an operand inside a handler, so nominal typing cannot creep in
through this door.

**Subsumption is a claim, and claims are tested.** ``implies`` states that
whenever this atom reads true, another must too. It is checked against the
calibration benches rather than assumed, because the value of a stated
implication is entirely in its being false when it is false: a measurement layer
that lets ``on`` hold where ``contact`` does not is a measurement layer nobody
can reason with.

**Anyone may extend it.** An atom is registered by defining a class, and a task
directory's ``assets/`` package is executed like any other module -- so a task
that needs a question this catalog cannot ask defines it beside the asset it is
about. The mint gate applies identically: a new atom states its calibration or
it does not register.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, ClassVar

from .atoms import ArgSpec, AtomCalibration, AtomSpec, register_atom
from .handlers import dispatch, handler

__all__ = ["Atom", "Property", "Relation", "SpatialRelation", "family_of", "leaves_of"]

_FAMILIES: dict[str, list[type[Atom]]] = {}
"""Registered leaves by family name, in declaration order."""


class Atom:
    """The root of the measurement taxonomy. Abstract; declares nothing itself.

    A subclass states what it measures by setting the class variables below, and
    registers itself by existing. The class is the *declaration*; the compiled
    :class:`~robosimstudio.assetlib.semantics.atoms.AtomSpec` it produces is what
    the registry, the typechecker, the goal binder and the catalog consume -- so
    nothing downstream had to learn a new shape when this layer arrived.

    Class variables mirror ``AtomSpec`` field for field, plus two that are new
    and are what makes this a taxonomy rather than a namespace:

    ``family``
        Which abstract class it hangs off, derived from the MRO rather than
        declared. What it buys is a question an agent can actually ask: "what
        spatial relations can I ask about these two things" is answerable;
        "list every atom" is not.
    ``implies``
        Other atoms that must read true wherever this one does. Tested, not
        assumed -- see the module docstring.
    """

    abstract: ClassVar[bool] = True
    """Set on families. An abstract class describes a *kind* of question and
    registers nothing; only leaves are atoms."""

    name: ClassVar[str] = ""
    summary: ClassVar[str] = ""
    args: ClassVar[tuple[ArgSpec, ...]] = ()
    returns: ClassVar[tuple[str, ...]] = ()
    unit: ClassVar[str] = ""
    cost: ClassVar[str] = "cheap"
    directions: ClassVar[tuple[str, ...]] = ("check", "observe")
    reads_scene: ClassVar[bool] = False
    smoothing: ClassVar[int] = 0
    provenance: ClassVar[str] = ""
    calibration: ClassVar[AtomCalibration | None] = None
    tags: ClassVar[tuple[str, ...]] = ()
    """Free-text tags on the catalogue record.

    Declared here because it is an ``AtomSpec`` field and the class side is
    meant to mirror that record field for field -- and it did not, which is the
    concrete cost the hand-written projection below used to carry: an atom
    written as a class could not set a tag that an atom written as a spec
    could, and nothing said so."""

    implies: ClassVar[tuple[type[Atom], ...]] = ()

    spec: ClassVar[AtomSpec | None] = None
    """The compiled declaration, set at subclass time. Leaves only."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A family sets ``abstract = True`` in its own body. A leaf does not, and
        # inheriting it would silently make every leaf abstract -- so the flag is
        # read off the class's own ``__dict__`` rather than off the MRO.
        cls.abstract = cls.__dict__.get("abstract", False)
        if cls.abstract:
            _FAMILIES.setdefault(cls.__name__, [])
            return
        if not cls.name:
            raise ValueError(f"{cls.__name__} declares no atom name")
        if not cls.summary:
            raise ValueError(f"atom {cls.name!r} declares no summary; the catalog is read by people and models")
        # Every field ``AtomSpec`` declares that this class also carries, read off
        # the dataclass rather than listed. Eleven of them were named by hand
        # here, and a hand-written projection of one type onto another is a list
        # that goes stale in exactly one direction: ``AtomSpec`` grew ``tags``
        # and no class-declared atom could set it, because nobody remembered
        # this call. The class keeps the fields it owns alone -- ``implies``,
        # which is about subsumption and has no place on the catalogue record --
        # and everything else travels by name.
        cls.spec = AtomSpec(**{f.name: getattr(cls, f.name) for f in fields(AtomSpec) if hasattr(cls, f.name)})
        register_atom(cls.spec)
        _FAMILIES.setdefault(family_of(cls), []).append(cls)

    @classmethod
    def implement(
        cls,
        *,
        kinds: tuple[str, ...] = (),
        substrates: tuple[str, ...] = (),
        scope: str = "kind",
        owner: str = "",
        summary: str = "",
    ):
        """Register one measurement of this atom, for one operand signature.

        The decorator form of ``@handler``, bound to the atom it implements so a
        typo cannot register a measurement of something else. Everything about
        dispatch is unchanged: most specific wins, an asset may shadow a kind
        handler, a task may shadow that.
        """
        return handler(cls.name, kinds=kinds, substrates=substrates, scope=scope, owner=owner, summary=summary)

    @classmethod
    def measure(cls, *values: Any, kinds: tuple[str, ...] = (), substrates: tuple[str, ...] = ()) -> dict[str, Any]:
        """Compute this atom for these already-read operands.

        The one evaluation path. A goal reaches it through the binder and an
        agent through the benchmark tier's ``query``; both dispatch
        the same way and land in the same handler, which is why a score and a
        structured reading cannot disagree about what happened.

        Args:
            *values: The operands, already read into the form this atom asked
                for. Reading is the caller's, because who resolves a reference
                differs -- a goal has a document, an agent may not.
            kinds: Feature kind per operand, for dispatch. Empty is a wildcard.
            substrates: Substrate per operand. Empty is a wildcard.
        """
        # Padded to the atom's own arity, because a handler that declares kinds
        # is matched on their *length* -- so asking with substrates alone would
        # miss every handler that wildcarded its kinds, which is most of them.
        # The arity is read off ``args`` rather than off ``values``: a
        # scene-reading atom is handed the snapshot first, and counting that as
        # an operand asks for a three-operand handler that nobody registered.
        width = max(len(cls.args), len(kinds), len(substrates))
        kinds = tuple(kinds) + ("",) * (width - len(kinds))
        substrates = tuple(substrates) + ("",) * (width - len(substrates))
        chosen = dispatch(cls.name, kinds, substrates)
        return chosen.fn(*values)


class Relation(Atom):
    """Something true *between* things. Two operands or more.

    The family that carries the asymmetry rule: a relation belongs to neither of
    its operands, which is why handlers are free functions rather than methods.
    ``on(a, b)`` forced onto ``a`` is how object models acquire their worst
    asymmetries -- and the answer to "is the tote under the bottle" stops being
    the answer to "is the bottle on the tote".
    """

    abstract = True
    symmetric: ClassVar[bool] = False
    """Whether swapping the operands must give the same reading. Declared so a
    test can check it rather than a reader assuming it."""


class SpatialRelation(Relation):
    """Where things are with respect to each other: touching, on, inside, over.

    Everything in this family answers a question about *arrangement* -- and
    arrangement is what a manipulation task is about, so this is the family an
    agent asks first and the one a goal is usually written in.
    """

    abstract = True


class Property(Atom):
    """Something true *of* one thing: its height, its speed, how wound it is.

    One operand. The distinction from a relation is not pedantry: a property is
    answerable from the object alone, so it survives the other object leaving
    the scene, and a structured observation can report it without pairing anything.
    """

    abstract = True


def family_of(cls: type[Atom]) -> str:
    """The nearest abstract ancestor's name, or ``"Atom"`` for a bare leaf."""
    for base in cls.__mro__[1:]:
        if base is Atom:
            return "Atom"
        if isinstance(base, type) and issubclass(base, Atom) and base.__dict__.get("abstract", False):
            return base.__name__
    return "Atom"


def leaves_of(family: str) -> list[type[Atom]]:
    """Every registered atom in one family, in declaration order.

    What ``sim.list_relations`` answers with, and what makes "ask every spatial
    relation about these two" a thing an agent can do rather than a list it has
    to keep.
    """
    return list(_FAMILIES.get(family, ()))
