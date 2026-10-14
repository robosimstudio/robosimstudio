# SPDX-License-Identifier: Apache-2.0
"""The mint gate: the gauntlet, pointed at an atom instead of at a task.

An authoring agent that cannot express a goal drafts the measure it needs. That
is the intended path -- small pure functions over a snapshot are an excellent
codegen target, and the alternative is a task language that grows a combinator
every time somebody wants to say something new. What must not happen is the
*silent registry write*: an atom that appears, gets thresholded, and is never
compared against anything.

So a mint is PR-shaped, and these are the checks it has to survive:

============================  ================================================
purity                        a measure reads a snapshot and returns a number.
                              It does not step, write, sample or remember
typed signature               the operands can be addressed: named, distinct,
                              read in a form the closed set contains, ordered
                              so an omitted optional cannot shift the rest
dedup                         does an atom answering this question already
                              exist? Near-synonyms are legitimate; *unremarked*
                              near-synonyms are how two goals come to claim
                              they measure the same thing
anti-overfit                  ``hammer_in_correct_bin`` is a task parameter
                              wearing a costume. The generality belongs in the
                              operands
required calibration          a gate needs the two populations it separates and
                              the valley between them
============================  ================================================

The declaration half is :meth:`~robosimstudio.assetlib.semantics.atoms.AtomSpec.problems`,
which has always described itself as "what the mint gate runs" and had nothing
running it. This module is the rest: the source-level purity lint, the dedup
sweep across the registry, the anti-overfit lint against the vocabulary the
corpus actually uses, and the ratchet that makes "every atom carries a
calibration" a rule the repository can state before it can satisfy it.

**The calibration rule is stated with its debt attached.** Eleven of the sixteen
shipped atoms carry a calibration block; five predate the gate. Those five are
listed in :data:`UNCALIBRATED` with the reading each one is missing, the list may
only shrink, and an entry that stops being needed must be deleted: a rule you
cannot satisfy yet is worth stating if the gap is a number that only goes down.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass

from robosimstudio.assetlib.semantics.atoms import AtomSpec, get_atom, list_atoms

from .gauntlet import Finding

__all__ = [
    "IMPURE_CALLS",
    "UNCALIBRATED",
    "MintReport",
    "mint_gate",
    "mint_problems",
    "near_synonyms",
    "purity_problems",
]

UNCALIBRATED: dict[str, str] = {
    "containment": "the fraction is BDDL's own box test and inherits its convention; what is "
    "uncalibrated is the >= 1.0 every goal compares it against",
    "height": "no shipped goal thresholds it; it is read as an observation",
    "straightness": "observed by cable_route and thresholded by nothing",
    "threaded": "geometric and exact -- but the clearance it reports has never been separated "
    "into hanging and merely-resting populations",
}
"""Atoms that predate the required-calibration rule, and what each is missing.

A ratchet, not an exemption. Every entry is debt; the dict may only shrink; an
entry whose atom has since grown a calibration must be deleted, so the countdown
keeps counting.

Note what the entries say. Four of the five are honest -- nothing thresholds
them, so there is no gate to justify. ``containment`` is the interesting one: it
*is* thresholded, by every placement goal in the corpus, at ``>= 1.0``. That
number came from BDDL and has never been separated against anything here, which
is exactly the shape this gate exists to make visible.
"""

IMPURE_CALLS: frozenset[str] = frozenset(
    {
        # writes the world
        "step",
        "assign",
        "zero_",
        "fill_",
        "launch",
        "synchronize",
        "apply",
        # makes the answer depend on when it was asked
        "random",
        "rand",
        "randn",
        "seed",
        "shuffle",
        "uniform",
        "choice",
        "time",
        "monotonic",
        "now",
    }
)
"""Call names a measure may not make.

Two groups, and they fail differently. The first writes something -- an atom
that steps or assigns has stopped being a reading of the world and has become
part of it, and two measures over one snapshot would then disagree depending on
which ran first. The second reads a clock or an entropy source, which makes the
same snapshot score differently twice and destroys the one property the whole
equivalence spine rests on: that a recorded tape re-scores bit-for-bit.

Matched on the attribute or function name alone, deliberately. ``np.random.rand``
and ``rng.rand`` are the same mistake and a dotted-path matcher would catch
neither reliably.
"""


@dataclass(frozen=True)
class MintReport:
    """One atom, and everything the gate has to say about it.

    Attributes:
        atom: The name.
        findings: Errors and notes. An empty tuple is a clean mint.
    """

    atom: str
    findings: tuple[Finding, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)

    def text(self) -> str:
        head = f"  {'ok  ' if self.ok else 'FAIL'} {self.atom}"
        return "\n".join([head, *(f"         {f}" for f in self.findings)])


def purity_problems(spec: AtomSpec) -> list[Finding]:
    """Whether the implementation reads the world rather than touching it.

    Source-level and therefore approximate, which is the honest description: it
    catches the shapes a generated measure actually takes -- a global counter, a
    write into an argument, a call to a clock or an RNG, a generator masquerading
    as a function -- and it cannot prove purity. The tests prove purity by
    calling a measure twice on one frozen snapshot; this is what refuses the
    obvious cases before anybody runs anything.

    An atom with no ``fn`` is not impure. Eleven of the sixteen shipped atoms
    have none: the atom is the *contract* and the handlers are its
    implementations, so there is nothing here to read. The handlers are covered
    by the per-kind contract tests, which is where a dispatching implementation
    belongs.
    """
    if spec.fn is None:
        return []
    where = f"atom {spec.name!r}"
    try:
        # ``dedent``, never ``cleandoc``: a nested implementation is indented and
        # ``ast`` refuses that, while ``cleandoc`` is for docstrings and strips
        # the first line's indent only -- which turns a method into a file that
        # will not parse and reports every such atom as unreadable.
        tree = ast.parse(textwrap.dedent(inspect.getsource(spec.fn)))
    except (OSError, TypeError, SyntaxError, IndentationError) as exc:
        return [
            Finding(
                "purity-unknown",
                f"the implementation's source could not be read ({type(exc).__name__}: {exc}), so "
                "the purity lint says nothing about it",
                where,
                severity="note",
            )
        ]

    function = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if function is None:
        return [Finding("purity-unknown", "no function definition found in the source", where, severity="note")]

    params = {a.arg for a in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)}
    out: list[Finding] = []

    for node in ast.walk(function):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            out.append(
                Finding(
                    "impure",
                    f"declares {'global' if isinstance(node, ast.Global) else 'nonlocal'} "
                    f"{', '.join(node.names)}; a measure that remembers is an instrument with state, "
                    "and this stack puts measurement state on the bound clause (see Leaf.on_reset)",
                    where,
                )
            )
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            out.append(Finding("impure", "is a generator; a measure returns its reading", where))
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                base = _root_name(target)
                if base in params and not isinstance(target, ast.Name):
                    out.append(
                        Finding(
                            "impure",
                            f"writes into its operand {base!r}. A snapshot's arrays are the "
                            "environment's own and are shared with every other measure this step",
                            where,
                        )
                    )
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in IMPURE_CALLS:
                out.append(
                    Finding(
                        "impure",
                        f"calls {name!r}. A measure is a pure function of a snapshot: it may not "
                        "advance the world, and it may not read a clock or an entropy source",
                        where,
                    )
                )
    return out


def near_synonyms(spec: AtomSpec, *, among: tuple[str, ...] = ()) -> list[str]:
    """Catalogued atoms asking the same question of the same operands.

    The dedup lint's evidence, not its verdict. Near-synonyms with different
    rigor are legitimate and documented -- ``resting_on`` wants a contact force
    with the object at rest, containment is a box a *carried* object satisfies,
    and both are correct about different claims. What the catalog owes a reader
    is that the distinction is legible, so a duplicate is reported and a
    duplicate that says why it exists passes.

    Matched on each operand's accepted kinds **and substrates**, plus the keys
    reported. The substrates are what stops the lint crying on the pair that
    decided its shape: ``speed`` reads a body's velocity magnitude for any
    substrate and ``settle_speed`` reads a high quantile over a garment's
    particles, both reporting ``speed`` over one operand. They are not two
    answers to one question -- a garment has no body -- and a lint that called
    them duplicates would fire on every legitimate specialisation, which is how
    a check gets muted.
    """
    names = among or tuple(list_atoms())
    mine = (_operand_signature(spec), frozenset(spec.returns))
    out = []
    for name in names:
        if name == spec.name:
            continue
        other = get_atom(name)
        if (_operand_signature(other), frozenset(other.returns)) == mine:
            out.append(name)
    return sorted(out)


def mint_problems(
    spec: AtomSpec,
    *,
    instances: tuple[str, ...] = (),
    among: tuple[str, ...] = (),
    require_calibration: bool = True,
) -> list[Finding]:
    """Everything the mint gate has against one atom.

    Args:
        spec: The atom, registered or drafted.
        instances: Scene-instance names the anti-overfit lint checks the atom's
            own name against -- ``hammer_0``, ``bin2``, ``tote``. Supplied by the
            caller rather than gathered here: the vocabulary belongs to a corpus
            and the library must not know which one it is standing next to.
        among: Which catalogued atoms to dedup against, defaulting to all of them.
        require_calibration: Whether a missing calibration block is an error.
            :data:`UNCALIBRATED` is consulted by :func:`mint_gate`, not here, so
            a caller checking a *draft* gets the rule undiluted.

    Returns:
        Findings. Errors fail the mint; notes are things a reviewer should read.
    """
    where = f"atom {spec.name!r}"
    out = [Finding("declaration", text, where) for text in spec.problems(require_calibration=require_calibration)]
    out += purity_problems(spec)

    if instances and spec.names_an_instance(instances):
        matched = sorted(i for i in instances if i and i.lower() in spec.name.lower())
        out.append(
            Finding(
                "overfit",
                f"the name mentions {', '.join(matched)}, which is a scene instance. An atom that "
                "hard-codes what it is measuring can never be reused by the next task or observe "
                "the next scene -- the generality belongs in the operands",
                where,
            )
        )

    twins = near_synonyms(spec, among=among)
    if twins:
        out.append(
            Finding(
                "duplicate",
                f"{', '.join(twins)} take the same operand kinds and report the same keys. "
                "A near-synonym with different rigor is legitimate; say which claim differs in "
                "`provenance` so a goal can choose deliberately",
                where,
                severity="note" if spec.provenance else "error",
            )
        )
    return out


def mint_gate(
    *,
    instances: tuple[str, ...] = (),
    atoms: tuple[str, ...] = (),
    exempt: dict[str, str] | None = None,
) -> list[MintReport]:
    """Run the gate over the registered catalog.

    Args:
        instances: The anti-overfit vocabulary; see :func:`mint_problems`.
        atoms: Which atoms, defaulting to every registered one.
        exempt: Atoms excused from the required-calibration rule, and why.
            Defaults to :data:`UNCALIBRATED`. Pass ``{}`` to see the whole debt.

    Returns:
        One report per atom, in catalog order. Reports rather than a raise: a
        reviewer wants the whole sheet, and the exempt atoms still carry their
        reason as a note so the debt is visible in the same output.
    """
    excused = UNCALIBRATED if exempt is None else exempt
    catalog = tuple(list_atoms())
    names = atoms or catalog
    reports: list[MintReport] = []
    for name in names:
        spec = get_atom(name)
        excuse = excused.get(name)
        findings = mint_problems(
            spec,
            instances=instances,
            among=catalog,
            require_calibration=excuse is None,
        )
        if excuse is not None:
            findings.append(
                Finding(
                    "uncalibrated",
                    f"exempt from the calibration rule: {excuse}",
                    f"atom {name!r}",
                    severity="note",
                )
            )
        reports.append(MintReport(atom=name, findings=tuple(findings)))
    return reports


def _operand_signature(spec: AtomSpec) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    """One atom's operands, as what each will accept.

    An operand naming no kinds is a whole object rather than one of its
    features, and its empty tuple is as much a signature as any other. Both
    halves are carried: two measures that take the same feature kinds on the
    same substrates and report the same keys are one measure with two names.
    """
    return tuple((tuple(sorted(arg.kinds)), tuple(sorted(arg.substrates))) for arg in spec.args)


def _root_name(node: ast.AST) -> str:
    """The name at the bottom of ``a.b[0].c``, or empty."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def _call_name(node: ast.AST) -> str:
    """What a call is called, ignoring whatever it hangs off."""
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""
