# SPDX-License-Identifier: Apache-2.0
"""Turning s-expression forms into a task, without resolving anything.

Handles the LIBERO/VAB dialect as it is actually written, plus the extensions
this stack needs. Standard sections mean exactly what they mean upstream;
extensions add sections rather than changing any.

**What standard BDDL cannot say, and what we add for it:**

``:staging``
    Which cell, which robot, which cameras, which physics, how many episodes.
    LIBERO supplies these through harness convention rather than in the file --
    which works until you want two harnesses, or want to know why a task ran the
    way it did. Recorded beats defaulted.
graded comparisons
    ``(>= (winding-turns cable peg0.axis) 0.5)``. Upstream predicates are
    boolean, and a deformable cannot be scored that way: a garment is never
    entirely inside a bin.
temporal modals
    ``once``, ``hold-for``, ``after``, ``peak``. A wrap is scored on turns
    *retained after release*; there is no way to say that in a static predicate.
``:reward``
    Separate from the goal, because reward is often not "the share of goal
    predicates satisfied" -- an insertion depth over its gate, a mating gap.
``:observe``
    Recorded every step, gating nothing. Reporting an uncalibrated quantity is
    how it gets calibrated; thresholding it uncalibrated is how a benchmark ends
    up measuring an arbitrary constant.
``:artifacts`` / ``:policy-params``
    A solution runs, so it stays code and is referenced. Its parameters are
    fenced off so they cannot drift back into being task semantics.

Unknown sections are **kept**, not dropped -- see :attr:`TaskDoc.extras`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .ast import Condition, Instance, Ref, Region, SuiteDoc, TaskDoc
from .reader import ParseError, Sexp, read_sexp

__all__ = ["MODALS", "merge_suite", "parse_problem", "parse_ref", "parse_suite", "read_problem", "read_suite"]

MODALS = ("once", "hold-for", "after", "peak", "integral")
"""Temporal wrappers a goal may use. Closed, and matching the combinator set --
expressivity grows by minting atoms, never by growing the grammar."""

COMPARISONS = (">=", ">", "<=", "<")
CONNECTIVES = ("and", "or", "any", "not", "worst_of")
"""Forms that take sub-clauses rather than references.

Every one of these is a combinator the goal builder already implements, and the
list had drifted to three of them. ``worst_of`` is the case that hurt: it is in
the closed set of nine, it is documented as the shirt's -- three creases, scored
on the weakest -- and no document could reach it, because a head the parser does
not recognise as a connective falls through to the predicate branch and
``parse_ref`` meets a nested form. The error is at least loud (that is what the
nested-form refusal is for), but it reads as a broken file rather than as a
grammar that never wired up its own combinator. ``any`` was the same gap,
silently: ``build_goal`` maps it, the parser did not.

Not a growth of the grammar. Nothing new can be said that the combinator set did
not already say; what changed is that a document can now say it.
"""


def parse_ref(token: Sexp) -> Ref:
    """Parse ``cable``, ``peg0.axis`` or ``cable.point(0.63)`` into a reference.

    Parameterised features are written the way anyone would write them --
    ``point(0.63)`` -- and split here rather than being given their own syntax,
    because a reference an author has to think about is a reference they get
    wrong.
    """
    if isinstance(token, list):
        # Refused rather than stringified. A list reaching here means a clause
        # was mistaken for a predicate -- which is exactly what happened when
        # BDDL's capitalised `And` missed a lowercase connective list: the four
        # sub-goals were each `str()`-ed into a reference named
        # "['In', 'hammer_0', ...]", and the file parsed "successfully" with its
        # entire goal silently discarded. A parser that stringifies structure
        # cannot fail loudly, and a task language that cannot fail loudly is
        # worse than one that cannot parse.
        raise ValueError(
            f"expected a reference, got a nested form {token!r} -- is a connective missing from the parser?"
        )

    text = str(token)
    arg: float | None = None
    if text.endswith(")") and "(" in text:
        text, _, tail = text.partition("(")
        try:
            arg = float(tail[:-1])
        except ValueError as exc:
            raise ValueError(f"reference {token!r}: {tail[:-1]!r} is not a number") from exc
    base, *path = text.split(".")
    if not base:
        raise ValueError(f"reference {token!r} has no object to hang off")
    return Ref(base=base, path=tuple(path), arg=arg)


def _tagged(form: Sexp, tag: str) -> list[Sexp] | None:
    """The body of ``(tag ...)``, or ``None`` when ``form`` is something else."""
    if isinstance(form, list) and form and str(form[0]) == tag:
        return form[1:]
    return None


def _sections(body: list[Sexp]) -> dict[str, list[Sexp]]:
    """Every ``(:keyword ...)`` section, by keyword.

    A repeated section is an error rather than a last-one-wins: two ``:goal``
    clauses is a file whose author believed something untrue about it, and
    quietly honouring one of them hides that.
    """
    out: dict[str, list[Sexp]] = {}
    for form in body:
        if isinstance(form, list) and form and str(form[0]).startswith(":"):
            key = str(form[0])
            if key in out:
                raise ValueError(f"section {key} appears more than once")
            out[key] = form[1:]
    return out


# ------------------------------------------------------------------ clauses
def parse_condition(form: Sexp) -> Condition:
    """Parse one init or goal clause."""
    if not isinstance(form, list) or not form:
        raise ValueError(f"expected a clause, got {form!r}")

    head = str(form[0])
    # Case-insensitive, because the dialect in the wild is not consistent:
    # LIBERO and VAB write `And` and `In`, our extended files write `and` and
    # `once`. Matching only one casing means the other silently becomes a
    # predicate nobody implements -- or worse, parses and means nothing.
    #
    folded = head.lower()
    # Hyphens fold to underscores for the connectives, and only for them: BDDL
    # is a lisp and `worst-of` is how anyone writes it, while the combinator it
    # names is a Python identifier. The modals keep their own spelling because
    # `hold-for` IS the name the goal builder branches on, and folding it here
    # would quietly stop matching.
    grouped = folded.replace("-", "_")

    if grouped in CONNECTIVES:
        return Condition(head=grouped, children=tuple(parse_condition(child) for child in form[1:]))

    if folded in MODALS:
        head = folded
        # `hold-for` and `after` take an argument before their body; `once`,
        # `peak` and `integral` wrap directly.
        if head in ("hold-for", "after"):
            if len(form) < 3:
                raise ValueError(f"({head} ...) needs an argument and a body")
            argument = form[1] if head == "hold-for" else parse_condition(form[1])
            body = parse_condition(form[2])
            return Condition(head=head, modal=head, modal_arg=argument, children=(body,))
        if len(form) != 2:
            raise ValueError(f"({head} ...) wraps exactly one clause")
        return Condition(head=head, modal=head, children=(parse_condition(form[1]),))

    if head in COMPARISONS:
        if len(form) != 3:
            raise ValueError(f"({head} ...) compares exactly two things")
        measure = form[1]
        if not isinstance(measure, list) or not measure:
            raise ValueError(f"({head} ...) expects a measure call on the left, got {measure!r}")
        if not isinstance(form[2], (int, float)):
            raise ValueError(f"({head} ...) expects a number on the right, got {form[2]!r}")
        # `(<= (peak (tension film)) 12.0)`. The left operand is a MODAL, not a
        # measure call: `peak` and `integral` report a quantity rather than a
        # verdict, so the only well-formed way to write one is inside the
        # comparison that thresholds it. The modal carries the quantity, the
        # comparison written around it carries the operator and the value, and
        # that is exactly the shape `build_goal` destructures -- it reads `op`
        # and `value` off the modal's own condition.
        modal = str(measure[0]).lower()
        if modal in ("peak", "integral"):
            return replace(parse_condition(measure), op=head, value=float(form[2]))
        if modal in MODALS:
            # `once`, `hold-for` and `after` are verdicts, and nothing reads a
            # threshold written around one -- the number would be dropped and
            # the clause would pass on any nonzero reading.
            raise ValueError(
                f"({head} ({modal} ...) ...) compares a verdict; put the comparison inside it, "
                f"as ({modal} ... ({head} MEASURE {form[2]}))"
            )
        return Condition(
            head=str(measure[0]),
            args=tuple(parse_ref(a) for a in measure[1:]),
            op=head,
            value=float(form[2]),
        )

    # A plain predicate: (adhered tape), (In hammer_0 bin2_hammer_region)
    return Condition(head=head, args=tuple(parse_ref(a) for a in form[1:]))


def _parse_objects(body: list[Sexp], *, static: bool) -> tuple[Instance, ...]:
    """``o1 o2 - type`` runs, BDDL's own typing syntax.

    Several names may share one type, so the parser accumulates until it meets
    the ``-`` and then applies the type backwards over everything pending.
    """
    out: list[Instance] = []
    pending: list[str] = []
    index = 0
    while index < len(body):
        token = str(body[index])
        if token == "-":
            if index + 1 >= len(body):
                raise ValueError("a '-' must be followed by a type")
            kind = str(body[index + 1])
            out.extend(Instance(name=name, type=kind, static=static) for name in pending)
            pending = []
            index += 2
            continue
        pending.append(token)
        index += 1
    # Untyped trailing names are legal: BDDL files declare bare objects.
    out.extend(Instance(name=name, static=static) for name in pending)
    return tuple(out)


def _parse_regions(body: list[Sexp]) -> tuple[Region, ...]:
    """``(name (:target t) (:ranges ((x0 y0 x1 y1))))``.

    The doubled bracket around ranges is BDDL's own -- a region may list several
    rectangles -- and only the first is taken here, with the rest kept out of the
    way rather than silently merged.
    """
    out: list[Region] = []
    for form in body:
        if not isinstance(form, list) or not form:
            continue
        name = str(form[0])
        clauses = _sections(form[1:])
        target = str(clauses.get(":target", [""])[0]) if ":target" in clauses else ""
        ranges_clause = clauses.get(":ranges")
        if not ranges_clause or not isinstance(ranges_clause[0], list):
            raise ValueError(f"region {name!r} has no (:ranges ((x0 y0 x1 y1)))")
        first = ranges_clause[0][0] if isinstance(ranges_clause[0][0], list) else ranges_clause[0]
        if len(first) != 4:
            raise ValueError(f"region {name!r}: a range is four numbers, got {first!r}")
        z_clause = clauses.get(":z-range") or clauses.get(":z_range")
        z_range = (float(z_clause[0]), float(z_clause[1])) if z_clause else (0.0, 0.15)
        out.append(
            Region(
                name=name,
                target=target,
                ranges=tuple(float(v) for v in first),  # type: ignore[arg-type]
                z_range=z_range,
            )
        )
    return tuple(out)


def _plain(body: list[Sexp]) -> dict[str, Any]:
    """A section of ``(key value...)`` clauses, as a dict.

    Used by the sections that are configuration rather than logic -- staging,
    artifacts, policy params, provenance. A single value unwraps; several stay a
    list; a nested form keeps its shape.

    **A repeated key accumulates rather than replacing.** ``(:graph gt "...")``
    and ``(:graph gap "...")`` are two graphs, not one graph written twice, and
    an earlier version of this kept only the last -- so a task shipping two
    baselines silently shipped one.

    Sections that are genuinely single-valued are protected a level up:
    ``_sections`` already refuses a repeated ``(:staging ...)``, so this only
    ever sees repeats *within* one section, where they mean a list.
    """
    out: dict[str, Any] = {}
    seen: dict[str, int] = {}
    for form in body:
        if not isinstance(form, list) or not form:
            continue
        key = str(form[0]).lstrip(":")
        values = form[1:]
        if not values:
            value: Any = True
        elif len(values) == 1:
            value = values[0]
        else:
            value = list(values)

        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            out[key] = value
        elif seen[key] == 2:
            # The second occurrence is what turns one value into a list. Wrapping
            # the first here rather than always building a list keeps the common
            # single-clause case reading as the value it is.
            out[key] = [out[key], value]
        else:
            out[key].append(value)
    return out


def _parse_hosts(body: list[Sexp]) -> tuple[dict[str, Any], ...]:
    """``(:host (:embodiment ur5 (:work-arm right)) (:static-override workcell scene_norobot.xml) ...)``, each as a plain dict."""
    out: list[dict[str, Any]] = []
    for form in body:
        host = _tagged(form, ":host")
        if host is None:
            raise ValueError(f":hosts holds (:host ...) clauses, got {str(form)[:60]}")
        out.append(_plain(host))
    return tuple(out)


# ------------------------------------------------------------------ problem
def parse_problem(forms: list[Sexp], *, where: str = "") -> TaskDoc:
    """Parse a ``(define (problem NAME) ...)`` form into a :class:`TaskDoc`."""
    define = next((f for f in forms if _tagged(f, "define") is not None), None)
    if define is None:
        raise ParseError("no (define ...) form in this file", line=1, column=1, where=where)

    body = _tagged(define, "define") or []
    header = body[0] if body else None
    if not isinstance(header, list) or len(header) < 2 or str(header[0]) != "problem":
        raise ParseError("a task file starts (define (problem NAME) ...)", line=1, column=1, where=where)

    sections = _sections(body[1:])
    doc_name = str(header[1])

    goal_clause = sections.get(":goal")
    if goal_clause is not None and len(goal_clause) != 1:
        raise ValueError(f"problem {doc_name!r}: :goal holds exactly one clause")

    return TaskDoc(
        name=doc_name,
        language=" ".join(str(t) for t in sections.get(":language", [])),
        # Fixtures first, then objects. The order is not cosmetic: it becomes
        # the order of body handles in the compiled scene, so it has to be one
        # thing consistently -- and scenery before the things that sit in it is
        # both how a scene is built and how every hand-written suite here
        # already ordered it.
        objects=(
            _parse_objects(sections.get(":fixtures", []), static=True)
            + _parse_objects(sections.get(":objects", []), static=False)
        ),
        regions=_parse_regions(sections.get(":regions", [])),
        init=tuple(parse_condition(f) for f in sections.get(":init", [])),
        goal=parse_condition(goal_clause[0]) if goal_clause else None,
        reward=_plain(sections[":reward"]) if ":reward" in sections else {},
        observe=tuple(parse_condition(f) for f in sections.get(":observe", [])),
        staging=_plain(sections.get(":staging", [])),
        suite=str(sections[":suite"][0]) if sections.get(":suite") else "",
        artifacts=_plain(sections.get(":artifacts", [])),
        policy_params=_plain(sections.get(":policy-params", [])),
        provenance=_plain(sections.get(":provenance", [])),
        hosts=_parse_hosts(sections.get(":hosts", [])),
        extras=tuple(
            (key, value)
            for key, value in sections.items()
            if key
            not in {
                ":domain",  # standard BDDL's upstream vocabulary: accepted, never loaded
                ":language",
                ":objects",
                ":fixtures",
                ":regions",
                ":init",
                ":goal",
                ":reward",
                ":observe",
                ":staging",
                ":suite",
                ":artifacts",
                ":policy-params",
                ":provenance",
                ":hosts",
            }
        ),
    )


def read_suite(path: str | Path) -> SuiteDoc:
    """Read one ``.suite`` file."""
    from .reader import read_sexp

    path = Path(path)
    return parse_suite(read_sexp(path), where=str(path))


def read_problem(path: str | Path) -> TaskDoc:
    """Read and parse a task file."""
    return parse_problem(read_sexp(path), where=str(path))


def parse_suite(forms: list[Sexp], *, where: str = "") -> SuiteDoc:
    """``(define (suite NAME) ...)`` to a :class:`SuiteDoc`."""
    if len(forms) != 1 or not isinstance(forms[0], list) or not forms[0]:
        raise ValueError(f"{where or 'suite'}: expected one (define (suite NAME) ...) form")
    form = forms[0]
    head = form[1] if len(form) > 1 else None
    if str(form[0]).lower() != "define" or not isinstance(head, list) or str(head[0]).lower() != "suite":
        raise ValueError(f"{where or 'suite'}: expected (define (suite NAME) ...), got {str(form[:2])[:80]}")
    # As written, the same as a problem's name: a variant's ``(:suite NAME)`` and
    # the directory the suite sits in are compared to it exactly.
    name = str(head[1])
    sections = _sections(form[2:])
    allowed = {
        ":language",
        ":staging",
        ":fixtures",
        ":regions",
        ":artifacts",
        ":provenance",
        ":hosts",
        ":variants",
    }
    unknown = sorted(set(sections) - allowed)
    if unknown:
        raise ValueError(f"{where or name}: a suite may carry {sorted(allowed)}, not {unknown}")
    language = sections.get(":language", [])
    return SuiteDoc(
        name=name,
        language=" ".join(str(t) for t in language),
        staging=_plain(sections.get(":staging", [])),
        fixtures=_parse_objects(sections.get(":fixtures", []), static=True),
        regions=_parse_regions(sections.get(":regions", [])),
        artifacts=_plain(sections.get(":artifacts", [])),
        provenance=_plain(sections.get(":provenance", [])),
        variants=tuple(str(t) for t in sections.get(":variants", [])),
        hosts=_parse_hosts(sections.get(":hosts", [])),
    )


def _clause_block(value: Any) -> list[Any] | None:
    """A staging value as its ``(:key ...)`` sub-clauses, or ``None`` if it is not only those.

    ``_plain`` unwraps one sub-clause, so ``(:episodes (:count 5))`` is the one
    clause ``[":count", 5]`` and two are a list of two. A value with a head of
    its own -- ``(:embodiment yam_bimanual (:work-arm left))`` -- is not a block:
    its clauses belong to that head, and a variant naming another head replaces
    them with it.
    """
    if not isinstance(value, list) or not value:
        return None
    clauses = value if isinstance(value[0], list) else [value]
    if all(isinstance(c, list) and c and str(c[0]).startswith(":") for c in clauses):
        return clauses
    return None


def _sub_key(clause: list[Any]) -> str:
    """A sub-clause's key, spelled once: ``:model-key-fields`` and ``:model_key_fields`` are one key."""
    return str(clause[0]).lstrip(":").replace("-", "_")


def _merge_staging(suite: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any]:
    """The variant's staging over its suite's, a clause block merged sub-key by sub-key.

    A variant stating ``(:episodes (:count 5))`` over a suite's ``(:episodes
    (:count 50) (:sampler layout) (:model-key-fields layout))`` changes the count
    and keeps the sampler: replacing the whole block would drop the sampler and
    the model-key-fields without a word, and the second is what makes the
    environment rebuild the model between episodes.
    """
    out = {**suite, **doc}
    for key in suite.keys() & doc.keys():
        base, mine = _clause_block(suite[key]), _clause_block(doc[key])
        if base is None or mine is None:
            continue
        stated = {_sub_key(c) for c in mine}
        out[key] = [*(c for c in base if _sub_key(c) not in stated), *mine]
    return out


def merge_suite(doc: TaskDoc, suite: SuiteDoc | None) -> TaskDoc:
    """A variant over its suite: the variant's clause wins key by key, the rest inherits.

    ``None`` is the one-variant case and returns the document untouched. A
    variant naming a *different* suite than the file beside it is refused -- one
    level, no chains, and no document quietly inheriting from the wrong place.
    """
    if suite is None:
        return doc
    if doc.suite and doc.suite != suite.name:
        raise ValueError(f"{doc.name}: names suite {doc.suite!r} but sits beside suite {suite.name!r}")
    from dataclasses import replace

    seen = {o.name for o in doc.objects}
    fixtures = tuple(f for f in suite.fixtures if f.name not in seen)
    return replace(
        doc,
        suite=suite.name,
        language=doc.language or suite.language,
        objects=(*fixtures, *doc.objects),
        regions=(*(r for r in suite.regions if r.name not in {x.name for x in doc.regions}), *doc.regions),
        staging=_merge_staging(suite.staging, doc.staging),
        artifacts={**suite.artifacts, **doc.artifacts},
        provenance={**suite.provenance, **doc.provenance},
        hosts=(*suite.hosts, *doc.hosts),
    )
