# SPDX-License-Identifier: Apache-2.0
"""Atoms: catalogued measures shared by goals, initial constraints and queries.

An atom is a named measure over a snapshot, and it is the unit this whole design
economises on. The benchmark's corpus -- rigid sorting, 0.15 mm insertion, adhesive
peeling, cable winding, garment folding across seventeen variants -- runs on
about twenty-five of them, because an atom asks a question of *physics* (pose,
contact, attachment, joint state, deformation shape, field state) and a task
recombines those questions rather than extending them. Vocabularies of this shape
saturate; a new task in a familiar family mints none, a genuinely new family
mints one to three.

Three properties are structural rather than stylistic:

**One atom, three directions.** The same measure can *check* a goal, *sample* an
initial condition, and be *observed* as structured values. Init and goal are the
same statements pointed in opposite directions, and observation exposes the
same implementation without converting it into language. :attr:`AtomSpec.directions`
records which uses an atom actually supports.

**Graded always.** Every atom reports the quantity behind its boolean, because "a
threshold nobody has calibrated is worth less than the number it thresholds".
Where the two disagree about what is even possible, the harder implementer sets
the contract: containment returns a *fraction* for everything, since a garment is
never entirely inside a bin and a rigid object's 0-or-1 is the easy case of that,
not a different question.

**Calibration is a field, not a docstring.** The best threshold in this repo
records the two populations it separates and the valley between them -- every
wrap the shipped policy produces retains 0.55-0.96 turns, every non-wrap ever
traced measures 0.25 or less, so the gate sits at 0.5 where a drag-past cannot
fake it by a factor of three. That practice is what :class:`AtomCalibration`
makes machine-checkable, so the mint gate can *require* it rather than hope for
it.

Atoms are pure and read-only over a snapshot: deterministic, no stepping, no side
effects. Anything that changes the world is an actuator and belongs to a policy
artifact, not here.

Near-synonyms with different rigor are legitimate and expected. ``resting_on``
(a contact force from the destination, with the object at rest) and containment
(a box a *carried* object satisfies) are both correct and they measure different
claims; the catalog's job is to make the distinction legible so a goal chooses
deliberately, not to pretend there is one truth.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from robosimstudio.registry import Registry

__all__ = [
    "ATOMS",
    "COST_TIERS",
    "DIRECTIONS",
    "OPERAND_FORMS",
    "ArgSpec",
    "AtomCalibration",
    "AtomSpec",
    "atom_to_dict",
    "get_atom",
    "list_atoms",
    "register_atom",
]

OPERAND_FORMS: tuple[str, ...] = (
    "position",
    "pose",
    "points",
    "shapes",
    "velocity",
    "region",
    "feature",
    "handle",
)
"""What a handler may be handed for one operand.

Closed, and small on purpose. Structural typing decides *which objects* satisfy
an operand; this decides *what a handler gets* when one does, and they are
different questions -- ``containment`` and ``contact_force`` both accept a rigid
body and want a point and a set of collider indices respectively. An open
vocabulary here would be a second type system nobody validates.

``position`` and ``pose`` are separate because most measures want a point and
would rather not carry an orientation they do not use; the ones that need it
genuinely need it, and a plug's tab tip is a millimetre out at ten degrees of
tilt if you take the point and assume the rotation away.

``feature`` is the general one: a declared feature's payload, converted into
world coordinates using the rules its *kind* states. One reader serves every
geometric kind, which is what stops each handler re-deriving the same rotation.
"""

DIRECTIONS: tuple[str, ...] = ("check", "sample", "observe")
"""What an atom can be used for. Declared per atom and enforced by the
typechecker, so a goal written against a sample-only family is refused where it
was written rather than silently never firing."""

COST_TIERS: tuple[str, ...] = ("cheap", "expensive")
"""The relative cost of evaluating an atom for a structured query."""


@dataclass(frozen=True, kw_only=True)
class ArgSpec:
    """One operand of an atom, typed structurally.

    Structural, never nominal: an argument does not require an object to *be* a
    kind of thing, it requires the object to *have* a feature of some kind. Any
    object carrying a ``loop`` satisfies ``threaded``'s first operand -- scissors,
    a mug, a keyring, a washer -- without anything declaring an interface or
    inheriting anything.

    Attributes:
        name: Operand name, used in errors and structured result binding.
        kinds: Feature kinds that satisfy it, any one of them. Empty means the
            operand is a whole object rather than one of its features.
        substrates: Substrates that satisfy it, empty for any.
        optional: Whether a call may omit it.
        reads: **What form the handler wants this operand in.** Structural typing
            says which objects satisfy an operand; this says what a handler is
            handed when one does, and the two are genuinely different questions.
            ``containment`` and ``contact_force`` both accept a rigid body, and
            one needs its position while the other needs its collision shapes --
            so the form belongs to the *measure*, not to the object.

            The vocabulary is deliberately small and closed: ``position`` (a
            point), ``points`` (many, for a garment or a chain), ``shapes``
            (collider indices), ``velocity``, ``region`` (a box in world
            coordinates), ``handle`` (the scene entity itself, for a measure that
            needs more than one of the above). Empty means ``position``, which is
            what nearly every measure wants.
    """

    name: str
    kinds: tuple[str, ...] = ()
    substrates: tuple[str, ...] = ()
    optional: bool = False
    reads: str = ""

    def accepts(self, kind: str = "", substrate: str = "") -> bool:
        """Whether a feature of ``kind`` on a ``substrate`` body satisfies this."""
        from robosimstudio.specs.physics import reads_as

        if self.kinds and kind not in self.kinds:
            return False
        if not self.substrates or not substrate:
            return True
        return any(reads_as(substrate, entry) for entry in self.substrates)


@dataclass(frozen=True, kw_only=True)
class AtomCalibration:
    """The evidence behind a threshold.

    Mirrors ``materials.CalibrationSpec`` in intent: a number whose justification
    survives only as a comment beside it is a number nothing can check. Here the
    justification is the *separation* -- what the measure reads on outcomes that
    should pass, what it reads on outcomes that should fail, and where the gate
    sits between them.

    Attributes:
        positives: What the measure reads on outcomes that should pass, as prose
            with numbers ("0.55-0.96 turns retained, over every shipped policy
            rollout").
        negatives: What it reads on outcomes that should fail ("drag-pasts,
            slipped grasps and a sprung coil all measure 0.25 or less").
        gate: The threshold, sitting in the valley between them.
        bench: What measured it -- a tool, a suite, a published benchmark.
        notes: Anything a reader needs that the three fields above do not carry.
    """

    positives: str = ""
    negatives: str = ""
    gate: float | None = None
    bench: str = ""
    notes: str = ""

    @property
    def separated(self) -> bool:
        """Whether both populations were actually measured.

        A calibration naming only its positives has not shown a separation, and a
        gate with nothing below it is a wish. The mint gate reads this.
        """
        return bool(self.positives and self.negatives)


@dataclass(frozen=True, kw_only=True)
class AtomSpec:
    """One catalogued measure.

    Attributes:
        name: How a goal, an init or a structured query refers to it.
        summary: What it measures, in one line.
        fn: The implementation -- a pure function of a snapshot and its operands.
        args: Typed operands, in order.
        directions: Which of :data:`DIRECTIONS` it supports.
        returns: What it reports, by key, so a goal can threshold the right thing
            and a report can label it.
        unit: Physical unit of the primary quantity, empty for a ratio.
        cost: One of :data:`COST_TIERS`.
        provenance: Where it came from -- transcribed from which benchmark,
            designed here, ported from which reference.
        calibration: The evidence behind its default gate, if it has one.
    """

    name: str
    summary: str
    fn: Callable[..., Any] | None = None
    args: tuple[ArgSpec, ...] = ()
    directions: tuple[str, ...] = ("check", "observe")
    returns: tuple[str, ...] = ()
    unit: str = ""
    cost: str = "cheap"
    provenance: str = ""
    calibration: AtomCalibration | None = None
    reads_scene: bool = False
    """Whether the handler is handed the whole snapshot before its operands.

    Off for nearly every measure, and the exceptions are worth naming because the
    line is easy to blur. A measure of *geometry* -- is this point in that box,
    how many turns are on this rod -- is a function of its operands and nothing
    else, and keeping it that way is what lets it be tested against arrays a test
    wrote down.

    A measure of *contact* is not. Whether two things are pressing on each other
    is answered by the solver's contact set, which belongs to the scene rather
    than to either object, and a handler that took only positions could not ask.
    Same for anything reading velocities or the settled baseline.

    Declared rather than inferred from the signature, so that "this measure needs
    the world" is a decision somebody made and a reviewer can see.
    """
    smoothing: int = 0
    """Control steps to take the **largest** reading over, or 0 for none.

    Declared on the atom rather than written into a goal, because it is a
    property of the *measurement* and not of what the task wants. The case that
    forced it: a settled item's contact with a tote floor is not continuous --
    the solver's contact set churns, so the pair drops out of it for up to six
    consecutive steps while nothing has moved. Measured over 300 steps with five
    items at rest, each showed a 2-5 N median and a spurious zero on 6-10% of
    steps; with five items independently dropping, about 28% of steps carry a
    zero somewhere. A goal sampling one instant is a coin flip, and more settling
    never helps because there is nothing left to settle.

    The alternative would have been a tenth combinator -- a windowed maximum --
    and it is the wrong place for it twice over. The combinator set is closed on
    purpose, and a window written in a goal would let two tasks disagree about
    how to read one sensor while claiming to measure the same thing. Here it is
    the atom's own answer to "what does this instrument read", stated once.

    Not a latch: the window slides, so a reading that genuinely falls to zero and
    stays there reports zero after ``smoothing`` steps.
    """
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def primary(self) -> str:
        """The reading a clause means when it names no selector.

        The first declared return, always.
        """
        return self.returns[0] if self.returns else ""

    def supports(self, direction: str) -> bool:
        """Whether this atom can be used in ``direction``."""
        return direction in self.directions

    def problems(self, *, require_calibration: bool = False) -> list[str]:
        """Everything structurally wrong with this atom, or an empty list.

        What the mint gate runs. It checks the declaration; the tests check the
        arithmetic, and no amount of either substitutes for the other.

        Args:
            require_calibration: Whether an atom must carry an
                :class:`AtomCalibration` at all. Off by default, because this is
                also the check a *reader* runs over the catalog and the five
                atoms that predate the gate would drown a caller asking a
                different question. The benchmark tier's mint gate turns it
                on, and carries the shrinking list of atoms exempt from it.
        """
        out: list[str] = []
        where = f"atom {self.name!r}"
        if not self.name:
            out.append("atom has no name")
        if not self.summary:
            out.append(f"{where}: no summary, so the catalog cannot describe it")
        if not self.directions:
            out.append(f"{where}: supports no direction, so nothing may use it")
        for direction in self.directions:
            if direction not in DIRECTIONS:
                out.append(f"{where}: unknown direction {direction!r}; one of {sorted(DIRECTIONS)}")
        if self.cost not in COST_TIERS:
            out.append(f"{where}: unknown cost tier {self.cost!r}; one of {sorted(COST_TIERS)}")
        if not self.returns:
            out.append(f"{where}: declares no return keys, so nothing can threshold or report it")
        if len(set(self.returns)) != len(self.returns):
            out.append(
                f"{where}: repeats a return key in {list(self.returns)}; a selector could not choose between them"
            )
        out += self._signature_problems(where)
        if self.smoothing < 0:
            out.append(f"{where}: smoothing is a window in control steps and cannot be negative")
        if require_calibration and self.calibration is None:
            out.append(
                f"{where}: carries no calibration. Every threshold this atom is ever given is a "
                "number somebody chose; the block records what the measure reads on outcomes that "
                "should pass, what it reads on outcomes that should fail, and where the gate sits "
                "between them"
            )
        if self.calibration is not None:
            out += _calibration_problems(where, self.calibration, strict=require_calibration)
        return out

    def _signature_problems(self, where: str) -> list[str]:
        """The typed-signature half of the mint gate.

        Over the **declaration**, never over ``fn``'s own parameter list, and the
        difference is deliberate. ``fn`` is a reference implementation working on
        already-converted payloads -- ``containment(points, lo, hi)`` implements
        two declared operands, ``thing`` and ``region``, because a region reads
        as two corners. Comparing the two arities would report the one atom whose
        implementation is written most clearly as the one that is wrong.

        What is checkable is that the operands can be *addressed*: named, named
        distinctly, read in a form the closed set contains, and ordered so that a
        call omitting an optional operand cannot silently shift the ones after it.
        """
        out: list[str] = []
        seen: set[str] = set()
        optional_seen = ""
        for arg in self.args:
            if not arg.name:
                out.append(f"{where}: an operand has no name, so no error message can point at it")
            elif arg.name in seen:
                out.append(f"{where}: two operands are both called {arg.name!r}")
            seen.add(arg.name)
            if arg.reads and arg.reads not in OPERAND_FORMS:
                out.append(f"{where}: operand {arg.name!r} reads {arg.reads!r}; one of {sorted(OPERAND_FORMS)}")
            if arg.optional:
                optional_seen = arg.name
            elif optional_seen:
                out.append(
                    f"{where}: required operand {arg.name!r} follows optional {optional_seen!r}. "
                    "Operands bind by position, so a call that omits the optional one silently "
                    "hands this measure the wrong argument"
                )
        return out

    def names_an_instance(self, instances: tuple[str, ...]) -> bool:
        """Whether this atom's name mentions a specific scene instance.

        The anti-overfit lint. An atom called ``hammer_in_correct_bin`` is a task
        parameter wearing a costume: the generality belongs in the operands, and
        an atom that hard-codes what it is measuring can never be reused by the
        next task or the next observation.
        """
        lowered = self.name.lower()
        return any(instance.lower() in lowered for instance in instances if instance)


def _calibration_problems(where: str, block: AtomCalibration, *, strict: bool) -> list[str]:
    """What a calibration block owes, once it exists.

    The gate check is the original one and the one that matters: a threshold
    needs the populations on both sides of it. ``strict`` adds the two the mint
    gate wants and a reader does not -- a separation at all, and the name of
    whatever measured it, because evidence nobody can re-run is a paragraph
    rather than a calibration.
    """
    out: list[str] = []
    if block.gate is not None and not block.separated:
        out.append(
            f"{where}: gate {block.gate} cites no measured separation -- "
            "a threshold needs the populations on both sides of it, not just the ones it passes"
        )
    if strict and not block.separated:
        out.append(
            f"{where}: the calibration names "
            f"{'no negatives' if block.positives else 'no populations'}, so it shows no separation"
        )
    if strict and not block.bench:
        out.append(f"{where}: the calibration names no bench, so nobody can re-run the measurement behind it")
    return out


ATOMS: Registry[AtomSpec] = Registry("atom", ("robosimstudio.domains",))
"""Every registered atom. Loads no packages -- atoms register from the modules
that implement them, which the asset and semantics layers import."""


def register_atom(spec: AtomSpec) -> AtomSpec:
    """Register an atom under its own name.

    Duplicate names are refused with both origins named, which is the mint gate's
    dedup lint for free: a second atom answering a question the catalog already
    answers has to say so out loud.
    """
    return ATOMS.register(spec.name, spec, origin=getattr(spec.fn, "__module__", ""))


def get_atom(name: str) -> AtomSpec:
    """Look one up by name."""
    return ATOMS.get(name)


def list_atoms() -> list[str]:
    """Every registered atom name, sorted."""
    return ATOMS.names()


def atom_to_dict(spec: AtomSpec) -> dict[str, Any]:
    """An atom as plain JSON-able data, for the catalog an agent reads."""
    out: dict[str, Any] = {
        "name": spec.name,
        "summary": spec.summary,
        "args": [
            {"name": a.name, "kinds": list(a.kinds), "substrates": list(a.substrates), "optional": a.optional}
            for a in spec.args
        ],
        "directions": list(spec.directions),
        "returns": list(spec.returns),
        "cost": spec.cost,
    }
    if spec.unit:
        out["unit"] = spec.unit
    if spec.provenance:
        out["provenance"] = spec.provenance
    if spec.tags:
        out["tags"] = list(spec.tags)
    if spec.calibration is not None:
        out["calibration"] = {
            "positives": spec.calibration.positives,
            "negatives": spec.calibration.negatives,
            "gate": spec.calibration.gate,
            "bench": spec.calibration.bench,
            "notes": spec.calibration.notes,
        }
    return out
