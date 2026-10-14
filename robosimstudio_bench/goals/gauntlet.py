# SPDX-License-Identifier: Apache-2.0
"""The gauntlet: eight staged checks over one task document, cheapest first.

An authoring loop needs to be told its task is **wrong**, in a form it can act
on. Nothing here computes anything new -- the reader, the binder, the checker,
the goal builder and the init verifier all already exist. What this adds is the
*sequence* and the *shape of the answer*: each stage runs only if the one before
it passed, and each reports findings rather than raising, so an agent gets the
whole list at the cheapest tier that can produce it.

Ordered by cost, and the ordering is the design:

======  ===============  ============================================================
Stage   Needs            What it establishes
======  ===============  ============================================================
1 read  a file           the document is well-formed s-expressions and sections
2 bind  a table name     an upstream dialect's predicates mean what they mean here
3 type  the catalog      everything it names exists and affords what is asked
4 eval  the catalog      the goal tree and the reward clause build
5 lint  nothing more     the goal is not trivially satisfiable or self-contradictory
6 comp  **a GPU**        the scene compiles and steps
7 start **a GPU**        init holds after settling, and the goal is not already true
8 floor **a GPU**        what a do-nothing policy is paid
======  ===============  ============================================================

**Stages 1-5 are free.** No scene, no GPU, milliseconds -- which matters because
they are the ones an agent iterates against. Seven of the eight failure modes
this benchmark has actually shipped are visible there.

**The non-trivial-start check has a documented trap, and it is honoured in
:mod:`~robosimstudio_bench.goals.initial` rather than here.** ``hold_for(5s, X)``
reads false on the first step however true ``X`` is, so a check that forgot to
strip the temporal wrappers would call every latched task non-trivial and let
the one broken task through -- passing loudly while doing nothing. Stage 7 goes
through :func:`~robosimstudio_bench.goals.initial.check_init`, which strips first.

**Stage 8 exists because every bug this benchmark found porting eight suites was
silent when wrong.** A measure that resolved to nothing and reported 0.0 for
ever; a metric key that drifted from what the atom reports; a flat conjunction of
fifteen clauses paying a do-nothing policy 0.333. None failed a run and most
produced a plausible number. Stepping the idle action for a few hundred steps and
printing the reward is one number that would have caught the last of those
immediately, and it costs a fraction of a rollout.

The stages are functions and the driver is :func:`run`, so a caller that wants
only the cheap half omits ``scene`` and gets stages 6-8 back as *skipped with a
reason* rather than as silence. A skipped stage is never a passed stage.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio.roots import SUITE_SUFFIX
from robosimstudio_bench.taskdoc import apply_bindings, check, merge_suite, read_problem, read_suite
from robosimstudio_bench.taskdoc.ast import Condition, TaskDoc
from robosimstudio_bench.taskdoc.typecheck import _GATE_WORDS, _MASK_WORDS

from .bind import bind
from .combinators import Leaf, Node
from .evaluate import catalog as asset_of
from .evaluate import make_evaluator
from .goal import about_key, leaves
from .initial import check_init
from .reward import measured_atom

__all__ = [
    "FLOOR_STEPS",
    "NEEDS_SCENE",
    "STAGES",
    "Finding",
    "Report",
    "StageResult",
    "do_nothing_floor",
    "goal_shape",
    "lint_goal",
    "run",
]

STAGES: tuple[str, ...] = (
    "read",
    "bind",
    "typecheck",
    "evaluate",
    "sim",
    "lint",
    "compile",
    "start",
    "floor",
)
"""The nine stages, in the order they run. Named rather than numbered in the
output, because "stage 6 failed" is not a message anybody acts on."""

NEEDS_SCENE: frozenset[str] = frozenset({"compile", "start", "floor"})
"""The stages that need a built scene, and therefore a GPU. Last, and skippable
-- an agent revising a document should not pay for a compile to be told its atom
has the wrong arity."""

FLOOR_STEPS = 300
"""Control steps the do-nothing floor runs for.

A few hundred, not a full episode. The question is "does this task pay a policy
that never moves", and a reward that is going to be paid for nothing is paid
within the first settle -- the grocery reward-grouping error read 0.333 on step
one. A full rollout would answer the same question for twenty times the cost.
"""

_VACUOUS_CEILING = 1e6
"""A ``<=`` threshold at or above this cannot fail for any measure this stack
reports. Deliberately far above every real gate (the largest in the corpus is a
12 N peel tension) so the lint cannot mistake a generous threshold for a vacuous
one."""

_WIDE_CONJUNCTION = 8
"""How many clauses under one ``and`` before the shape is worth remarking on.

Not a rule and not a failure: it is the number at which one satisfied clause is
worth less than an eighth, so a policy that solves one object of many looks like
a policy that is making progress on all of them. Grocery's fifteen is what this
remembers.
"""


@dataclass(frozen=True)
class Finding:
    """One thing the gauntlet has to say, phrased for whoever must act on it.

    Attributes:
        kind: A short machine-readable tag -- ``catalog-gap``, ``arity``,
            ``trivial-start``, ``do-nothing-reward``. An agent branches on this;
            a human reads ``detail``.
        detail: What is wrong and what would fix it.
        where: Which clause, in the checker's own dotted notation (``goal.1``,
            ``init[0]``, ``reward.progress``).
        severity: ``error`` fails the stage. ``note`` does not -- it is something
            an author should see and may legitimately have meant, and a gauntlet
            that failed on those would be one people learn to run with ``--force``.
    """

    kind: str
    detail: str
    where: str = ""
    severity: str = "error"

    def __str__(self) -> str:
        mark = "" if self.severity == "error" else "note "
        return f"[{mark}{self.kind}] {self.where + ': ' if self.where else ''}{self.detail}"


@dataclass
class StageResult:
    """What one stage concluded.

    Attributes:
        name: One of :data:`STAGES`.
        status: ``pass``, ``fail`` or ``skip``. A skipped stage is never a passed
            one: an agent told "the scene stages did not run" can decide whether
            to care, where a silent gap reads as approval.
        findings: Everything the stage has to say. A passing stage may still
            carry notes.
        data: Numbers worth carrying forward -- the goal's shape, the do-nothing
            reward, the init report's per-clause readings.
        reason: Why it was skipped, when it was.
    """

    name: str
    status: str
    findings: tuple[Finding, ...] = ()
    data: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def notes(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity != "error")


@dataclass
class Report:
    """Every stage's verdict for one document.

    Attributes:
        task: What was checked, as ``suite/variant`` or a path.
        stages: One entry per stage attempted or skipped, in order.
    """

    task: str
    stages: list[StageResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether nothing failed. A skipped stage does not make this false --
        it also does not make it true, which is what ``reason`` is for."""
        return not any(s.status == "fail" for s in self.stages)

    @property
    def failed(self) -> str:
        """The first stage that failed, or empty."""
        return next((s.name for s in self.stages if s.status == "fail"), "")

    def stage(self, name: str) -> StageResult | None:
        return next((s for s in self.stages if s.name == name), None)

    def findings(self) -> tuple[Finding, ...]:
        return tuple(f for s in self.stages for f in s.findings)

    def text(self) -> str:
        """The whole report, as the lines a terminal or an agent reads."""
        mark = {"pass": "ok  ", "fail": "FAIL", "skip": "skip"}
        lines = [f"{self.task}"]
        for stage in self.stages:
            head = f"  {mark.get(stage.status, '?'):<4} {stage.name:<10}"
            summary = stage.reason if stage.status == "skip" else _summarise(stage)
            lines.append(f"{head} {summary}".rstrip())
            for finding in stage.findings:
                lines.append(f"         {finding}")
        return "\n".join(lines)


def _summarise(stage: StageResult) -> str:
    """One line of numbers per stage, for the reader who is not debugging."""
    if not stage.data:
        return ""
    return "  ".join(f"{key}={value}" for key, value in stage.data.items())


# --------------------------------------------------------------------- stages
def _read(path: str | Path) -> tuple[StageResult, TaskDoc | None]:
    """Stage 1. The file is well-formed BDDL.

    Merged over a sibling ``.suite`` when there is one, as the loader merges it:
    a variant's fixtures and ``(:profile ...)`` live there, and checking the
    variant alone reports every one of them missing.
    """
    try:
        suites = sorted(Path(path).parent.glob(f"*{SUITE_SUFFIX}"))
        doc = merge_suite(read_problem(path), read_suite(suites[0]) if suites else None)
    except Exception as exc:
        return StageResult("read", "fail", (Finding("parse", f"{type(exc).__name__}: {exc}", str(path)),)), None
    return StageResult(
        "read",
        "pass",
        data={"objects": len(doc.objects), "regions": len(doc.regions), "init": len(doc.init)},
    ), doc


def _bind(doc: TaskDoc, profile: str) -> tuple[StageResult, TaskDoc]:
    """Stage 2. An upstream dialect's predicates, rewritten to this stack's measures.

    Unbound is the normal case and is not a failure: a document authored here
    writes ``containment`` directly, and a profile with no dialect binds
    nothing. What is a failure is *naming* a profile that does not exist,
    because the document then means something nobody can reconstruct -- and
    running the checker on the unbound document would report the dialect's own
    ``In`` as an atom nobody catalogued, which is eight false problems on a
    file that is in fact clean.
    """
    from robosimstudio_bench.taskbuild.staging import get_profile

    # The document's own profile when none was named, as the loader binds it.
    profile = profile or str(dict(doc.staging).get("profile", ""))
    if not profile:
        return StageResult("bind", "pass", data={"table": "none"}), doc
    try:
        table = get_profile(profile).bindings
    except KeyError as exc:
        return StageResult("bind", "fail", (Finding("unresolved", str(exc), "bindings"),)), doc
    if table is None:
        return StageResult("bind", "pass", data={"table": "none", "profile": profile}), doc
    bound = apply_bindings(doc, table)
    return StageResult("bind", "pass", data={"table": table.name, "tier": table.tier}), bound


def _typecheck(doc: TaskDoc, catalog: Any) -> StageResult:
    """Stage 3. Everything the document names exists and affords what is asked.

    A ``catalog-gap`` is kept distinct from everything else and is reported as
    what it is: the world does not contain a hook, which is not a mistake in the
    file. That distinction is the most useful thing an authoring loop can be
    told, because it converts "the task failed" into "the catalog needs a hook".
    """
    problems = check(doc, catalog)
    findings = tuple(Finding(p.kind, p.detail, p.where) for p in problems)
    gaps = sum(1 for p in problems if p.is_gap)
    return StageResult(
        "typecheck",
        "fail" if problems else "pass",
        findings,
        data={"problems": len(problems), "gaps": gaps},
    )


def _evaluate(doc: TaskDoc, task: str) -> tuple[StageResult, Any]:
    """Stage 4. The goal tree and the reward clause build.

    Separate from the typecheck because they refuse different things. The checker
    asks whether the words resolve; this asks whether the *forms* are ones the
    closed combinator and reward sets cover, and whether a handler exists for
    each clause's actual operands. A bare ``(peak M)`` resolves perfectly and
    still cannot be a verdict.
    """
    try:
        scored = make_evaluator(doc, task=task)
    except Exception as exc:
        kind = "unimplemented" if type(exc).__name__ == "BindError" else "malformed"
        return StageResult("evaluate", "fail", (Finding(kind, f"{type(exc).__name__}: {exc}", "goal"),)), None
    return StageResult(
        "evaluate",
        "pass",
        data={"leaves": len(_leaves(scored.goal)), "observe": len(scored.observe)},
    ), scored


def _resolver_for(task: str) -> Callable[[str], Any]:
    """The artifact resolver a registered suite's documents lower with.

    A document under a registered suite resolves ``routine:X`` and
    ``layout:X`` against that suite's package, as the loader does. A
    document checked on its own -- an agent's draft, a file outside any root --
    lowers with every artifact unresolved: this stage asks about physics, and
    a routine that does not exist yet is not a physics problem.
    """
    from robosimstudio_bench.taskbuild.artifacts import task_resolver
    from robosimstudio_bench.taskbuild.discovery import task_loader
    from robosimstudio_bench.taskbuild.registry import get_suite

    suite = task.partition("/")[0]
    try:
        home = get_suite(suite).home
        package = task_loader(suite).__name__
    except (KeyError, LookupError):
        return lambda _reference: None
    return task_resolver(Path(home), package)


def _record_of(name: str) -> Any | None:
    """The live catalog's record for ``name``, or ``None`` for an asset nothing declares."""
    from robosimstudio.assetlib.registry import get_asset

    try:
        return get_asset(name)
    except (KeyError, LookupError):
        return None


def _sim(doc: TaskDoc, catalog: Any, *, task: str) -> StageResult:
    """Stage 5. Does the staged arrangement give every asset the solver it needs?

    The one static check that asks the multi-engine question -- a backend for
    every domain the document's assets declare, and one that can own the state
    that domain carries. Cheap: it lowers the document (no scene) and reads the
    arrangement off it.
    """
    from robosimstudio_bench.taskbuild.lower import LowerError, lower

    from .simcheck import sim_problems

    suite, _, name = task.partition("/")
    try:
        spec = lower(doc, suite=suite, name=name or suite, resolve_artifact=_resolver_for(task)).spec
    except (LowerError, KeyError, ValueError) as exc:
        return StageResult("sim", "fail", (Finding("staging", f"{type(exc).__name__}: {exc}", "staging"),))
    # ``sim_problems`` wants a record lookup; the typechecker's catalog answers
    # different questions. A callable handed in (a test's fake) is used as is.
    problems = sim_problems(doc, spec, catalog if callable(catalog) else _record_of)
    findings = tuple(Finding("physics", problem, "staging") for problem in problems)
    return StageResult("sim", "fail" if problems else "pass", findings, data={"problems": len(problems)})


def _lint(doc: TaskDoc, scored: Any, catalog: Any) -> StageResult:
    """Stage 5. The goal builds -- but does it measure anything?"""
    findings = tuple(lint_goal(doc, scored, catalog))
    shape = goal_shape(doc, scored)
    return StageResult(
        "lint",
        "fail" if any(f.severity == "error" for f in findings) else "pass",
        findings,
        data=shape,
    )


def _compile(build: Callable[[], Any]) -> tuple[StageResult, Any]:
    """Stage 6. The scene builds and can be stepped."""
    try:
        env = build()
    except Exception as exc:
        return StageResult("compile", "fail", (Finding("build", f"{type(exc).__name__}: {exc}", "scene"),)), None
    model = getattr(env, "model", None)
    data = {
        "bodies": int(getattr(model, "body_count", 0) or 0),
        "shapes": int(getattr(model, "shape_count", 0) or 0),
        "particles": int(getattr(model, "particle_count", 0) or 0),
    }
    return StageResult("compile", "pass", data=data), env


def _start(doc: TaskDoc, env: Any, task: str) -> StageResult:
    """Stages 4 and 5 of the spec's own gauntlet, run together on one snapshot.

    Init verification and the non-trivial-start lint are answered from the same
    settled world, because they are asked of the same instant and because an
    author fixing one usually needs to know the other.

    The temporal stripping the second one needs is
    :func:`~robosimstudio_bench.goals.initial.strip_temporal`'s, not repeated here:
    ``hold_for(5s, X)`` is false on the first step however true ``X`` is, so a
    check that asked the goal directly would declare every latched task
    non-trivial and let the one broken task through.
    """
    try:
        env.reset()
        snapshot = Snapshot(ctx=env.make_context())
        # The episode's active set, as the env's own evaluator reads it: an
        # init clause about a parked object is not this episode's start.
        active = getattr(getattr(env, "episode", None), "active", None)
        report = check_init(doc, lambda c: bind(c, doc, asset_of, task=task or doc.name), snapshot, active=active)
    except Exception as exc:
        return StageResult("start", "fail", (Finding("start", f"{type(exc).__name__}: {exc}", "init"),))

    findings: list[Finding] = []
    for label in report.failed:
        findings.append(
            Finding(
                "init-violated",
                f"the document says this holds at the start; after settling it reports {report.details.get(label, {})}",
                label,
            )
        )
    if report.trivial:
        findings.append(
            Finding(
                "trivial-start",
                "the goal is already satisfied at step 0 with the temporal wrappers stripped, so "
                "this task scores 1.0 for a policy that never moves and reads in every log exactly "
                "like a task that was solved",
                "goal",
            )
        )
    return StageResult(
        "start",
        "fail" if findings else "pass",
        tuple(findings),
        data={"init_holds": report.holds, "trivial": report.trivial},
    )


def do_nothing_floor(env: Any, steps: int = FLOOR_STEPS, *, reset: bool = True) -> StageResult:
    """Stage 8. What the idle action is paid.

    A stage of its own because every bug this benchmark found porting eight
    suites was silent when wrong, and this is the one number that makes the
    loudest of them audible. A flat conjunction of fifteen clauses paid a
    do-nothing policy 0.333 and nothing anywhere said so; here it is the first
    line of the report.

    The reward reported is the **largest** over the run, not the last: a task
    that pays for the settle and then decays to zero has still paid for nothing,
    and a final-step reading would miss it.

    Public, and the only stage that is, because it is the one worth running on
    its own. A task that passes everything above still has a floor, and that
    number belongs beside any result ever reported against it.

    Args:
        env: A steppable environment.
        steps: How long to idle for.
        reset: Whether to restart the episode first. The driver passes ``False``
            because stage 7 has just reset and has not stepped, so the world is
            already at episode start -- and this repository has measured what a
            second reset can carry (``Env.reset`` does not fully restore
            simulation state, which is why the certifier runs each episode in
            a fresh process). A caller running this stage alone leaves it on.
    """
    try:
        if reset:
            env.reset()
        idle = env.idle_action()
        best = 0.0
        last = 0.0
        solved_at = -1
        taken = 0
        for step in range(max(1, int(steps))):
            _, reward, terminated, truncated, info = env.step(idle)
            taken = step + 1
            last = float(reward)
            best = max(best, last)
            if bool(info.get("success")) and solved_at < 0:
                solved_at = taken
            if terminated or truncated:
                break
    except Exception as exc:
        return StageResult("floor", "fail", (Finding("floor", f"{type(exc).__name__}: {exc}", "rollout"),))

    findings: list[Finding] = []
    if solved_at >= 0:
        findings.append(
            Finding(
                "solved-by-nothing",
                f"the idle action satisfies this goal at step {solved_at}. Whatever the task is "
                "measuring, it is not the policy",
                "goal",
            )
        )
    elif best > 0.0:
        findings.append(
            Finding(
                "do-nothing-reward",
                f"a policy that never moves is paid {best:.3f}. That is the floor every reported "
                "number sits on -- check it is a share you meant to give away",
                "reward",
                severity="note",
            )
        )
    return StageResult(
        "floor",
        "fail" if any(f.severity == "error" for f in findings) else "pass",
        tuple(findings),
        data={"steps": taken, "floor": round(best, 4), "final": round(last, 4)},
    )


# ----------------------------------------------------------------- the driver
def run(
    document: str | Path,
    *,
    task: str = "",
    profile: str = "",
    catalog: Any = None,
    scene: Callable[[], Any] | None = None,
    floor_steps: int = FLOOR_STEPS,
    close: bool = True,
) -> Report:
    """Run the gauntlet over one document, stopping at the first failure.

    Args:
        document: The ``.bddl`` file.
        task: What to call it in the report, and the name task-scoped handler
            overrides are resolved against. Defaults to the document's own.
        profile: The import profile the document reads under, for one written
            in an upstream dialect -- its predicate bindings are the profile's.
            Empty means the document's own ``(:profile ...)`` (merged over its
            suite's), and with none there, that it already speaks this stack's
            measures, which is true of everything authored here.
        catalog: What the typechecker asks about the world. ``None`` builds the
            live :class:`~robosimstudio_bench.goals.catalog.AssetCatalog`, which is
            what an agent wants; a test hands in a five-line fake.
        scene: Builds a steppable environment for stages 6-8. ``None`` reports
            them as skipped -- with a reason, never silently.
        floor_steps: How long the do-nothing floor runs.
        close: Whether to close the environment afterwards. False leaves it open
            for a caller that wants to keep looking at it.

    Returns:
        A report. Failures are findings, not exceptions: an agent revising a
        document wants the whole list, and an author fixing one error per round
        trip is the experience this exists to avoid.
    """
    report = Report(task=task or str(document))

    stage, doc = _read(document)
    report.stages.append(stage)
    if stage.status != "pass" or doc is None:
        return _skip_rest(report, "the document did not parse")

    stage, doc = _bind(doc, profile)
    report.stages.append(stage)
    if stage.status != "pass":
        return _skip_rest(report, "the predicate binding did not resolve")

    if catalog is None:
        from .catalog import AssetCatalog

        catalog = AssetCatalog()

    stage = _typecheck(doc, catalog)
    report.stages.append(stage)
    if stage.status != "pass":
        return _skip_rest(report, "the document does not typecheck")

    stage, scored = _evaluate(doc, task or doc.name)
    report.stages.append(stage)
    if stage.status != "pass" or scored is None:
        return _skip_rest(report, "there is no evaluator to run")

    stage = _sim(doc, catalog, task=task or doc.name)
    report.stages.append(stage)
    if stage.status != "pass":
        return _skip_rest(report, "the staged physics cannot serve what the document declares")

    stage = _lint(doc, scored, catalog)
    report.stages.append(stage)
    if stage.status != "pass":
        return _skip_rest(report, "the goal did not pass its lint")

    if scene is None:
        return _skip_rest(report, "no scene was offered (--no-scene, or no GPU)")

    stage, env = _compile(scene)
    report.stages.append(stage)
    if stage.status != "pass" or env is None:
        return _skip_rest(report, "the scene did not compile")

    try:
        report.stages.append(_start(doc, env, task or doc.name))
        if report.stages[-1].status == "pass":
            # No second reset: stage 7 has just reset and stepped nothing, so
            # the world is already where this stage wants it.
            report.stages.append(do_nothing_floor(env, floor_steps, reset=False))
        else:
            _skip_rest(report, "the start state is not what the document says it is")
    finally:
        if close:
            _close(env)
    return report


def _skip_rest(report: Report, reason: str) -> Report:
    """Record every stage that did not run, and why.

    Explicitly, rather than by absence. A stage missing from a report reads as a
    stage that passed, and the whole value of this tool is that it says what it
    did not check.
    """
    done = {s.name for s in report.stages}
    for name in STAGES:
        if name not in done:
            report.stages.append(StageResult(name, "skip", reason=reason))
    report.stages.sort(key=lambda s: STAGES.index(s.name))
    return report


def _close(env: Any) -> None:
    try:
        env.close()
    except Exception:
        pass


# ------------------------------------------------------------------- the lint
def lint_goal(doc: TaskDoc, scored: Any, catalog: Any) -> list[Finding]:
    """Everything structurally suspect about a goal that already builds.

    Six checks, and each one is a mistake this repository has shipped or come
    within one commit of shipping. Two are errors -- a goal that cannot fail
    measures nothing, and a clause written twice weights itself twice -- and four
    are notes, because each has a legitimate form and a lint that fails on those
    is one people learn to suppress.

    Args:
        doc: The document, which is where the thresholds are. The built tree
            deliberately does not carry them: a :class:`Leaf` knows what it is
            *about* and not what it was compared against.
        scored: The built evaluator, for the questions only the bound tree can
            answer -- whether anything under the goal actually measures.
        catalog: What the world contains, for the declarations no clause names.
    """
    findings: list[Finding] = []
    goal = doc.goal
    if goal is None:
        return [Finding("no-goal", "the task states no goal, so nothing can score it", "goal")]

    if not _leaves(getattr(scored, "goal", None)):
        findings.append(Finding("no-measure", "the goal combines nothing; it measures nothing", "goal"))

    leaves = list(_conditions(goal))
    findings += _vacuous(leaves)
    findings += _duplicates(leaves)
    findings += _wide(goal)
    findings += _reward_outside_goal(doc, leaves)
    findings += _unmeasured(doc)
    findings += _undeclared_types(doc, catalog)
    return findings


def goal_shape(doc: TaskDoc, scored: Any) -> dict[str, Any]:
    """The goal's dimensions, as numbers a report can print.

    Worth carrying even when nothing is wrong: ``leaves=15 widest=15 depth=2``
    is a task whose reward pays a fifteenth per clause, and seeing that beside
    the do-nothing floor is what makes the floor interpretable.
    """
    node = getattr(scored, "goal", None)
    widest = max((_arity(c) for c in _walk_conditions(doc.goal)), default=0) if doc.goal is not None else 0
    return {
        "leaves": len(_leaves(node)) if node is not None else 0,
        "depth": _depth(node) if node is not None else 0,
        "widest": widest,
    }


def _vacuous(leaves: list[Condition]) -> list[Finding]:
    """A threshold no reading can fail.

    ``(>= m 0.0)`` is satisfied by every non-negative measure, which is every
    measure in this catalog -- a fraction, a distance, a count of turns, a force.
    Stated as a threshold rule rather than as a claim about ranges, because
    nothing declares a range and inventing one would make this lint wrong in
    the direction that matters.
    """
    out: list[Finding] = []
    for index, leaf in enumerate(leaves):
        where = f"goal[{index}] ({leaf.head})"
        # `(> m 0)` is not vacuous: it fails at exactly zero, and says "strictly positive".
        if leaf.value is not None and (
            (leaf.op == ">=" and float(leaf.value) <= 0.0) or (leaf.op == ">" and float(leaf.value) < 0.0)
        ):
            out.append(
                Finding(
                    "vacuous-threshold",
                    f"({leaf.op} ({leaf.head} ...) {leaf.value}) cannot fail for a measure that is "
                    "never negative, so this clause contributes nothing but a satisfied conjunct",
                    where,
                )
            )
        if leaf.op in ("<=", "<") and leaf.value is not None and float(leaf.value) >= _VACUOUS_CEILING:
            out.append(
                Finding(
                    "vacuous-threshold",
                    f"({leaf.op} ({leaf.head} ...) {leaf.value}) is above anything this stack measures",
                    where,
                )
            )
    return out


def _duplicates(leaves: list[Condition]) -> list[Finding]:
    """The same clause twice.

    Not a stylistic complaint. ``all`` pays the *share* of its children
    satisfied, so a clause written twice is a clause weighted twice, and a goal
    of three distinct statements where one is duplicated pays 0.5 for half the
    work rather than a third.
    """
    seen: dict[tuple[Any, ...], int] = {}
    out: list[Finding] = []
    for index, leaf in enumerate(leaves):
        key = (leaf.head.replace("-", "_"), tuple(str(r) for r in leaf.args), leaf.op, leaf.value)
        if key in seen:
            out.append(
                Finding(
                    "duplicate-clause",
                    f"{about_key(key[0], key[1])} is stated twice; a conjunction pays the share of "
                    f"its children satisfied, so this clause carries double weight",
                    f"goal[{seen[key]}] and goal[{index}]",
                )
            )
        else:
            seen[key] = index
    return out


def _wide(goal: Condition) -> list[Finding]:
    """A conjunction wide enough that one clause is worth very little.

    A note, never an error: ``tool_sort/level2_1`` is a legitimate four-way flat
    conjunction and ``grocery_packing`` a legitimate grouped one. What is worth
    saying out loud is the arithmetic -- one of fifteen is 0.067, and a policy
    that solves one object looks, in the reward column, like one that is making
    progress on all of them.
    """
    widest = max((_arity(c) for c in _walk_conditions(goal)), default=0)
    if widest < _WIDE_CONJUNCTION:
        return []
    return [
        Finding(
            "flat-conjunction",
            f"the widest conjunction has {widest} clauses, so each is worth {1.0 / widest:.3f} of "
            "the reward. Check the grouping is the one you meant, and read the do-nothing floor",
            "goal",
            severity="note",
        )
    ]


def _reward_outside_goal(doc: TaskDoc, leaves: list[Condition]) -> list[Finding]:
    """The reward measures something the goal never mentions.

    Legitimate and shipped -- ``film_peel`` gates reward at its success fraction
    while its goal also demands the film survived, and the difference is the
    point. It is still worth saying, because the same shape is what a reward
    disagreeing with its goal about what counted looks like, and that one is
    silent.
    """
    measured = _reward_measure(doc.reward)
    if not measured:
        return []
    head, args = measured
    known = {(leaf.head.replace("-", "_"), tuple(str(r) for r in leaf.args)) for leaf in leaves}
    if (head, args) in known:
        return []
    return [
        Finding(
            "reward-outside-goal",
            f"reward is scored on {about_key(head, args)}, which no goal clause measures. "
            "Deliberate in two shipped suites; silent when it is not",
            "reward",
            severity="note",
        )
    ]


def _unmeasured(doc: TaskDoc) -> list[Finding]:
    """Objects the document declares and no clause ever mentions.

    A note, because scenery is real: ``cable_route/port_easy`` declares two pegs it
    never wraps, on purpose, because the rod can foul them and a document naming
    only the peg it uses would describe a different scene than the one that runs.
    The shape worth catching is the other one -- a sixth grocery item added to
    the objects list and forgotten in the goal, which reads as a task that got
    harder and scores as a task that did not.
    """
    named: set[str] = set()
    for condition in (*doc.init, *doc.observe, *((doc.goal,) if doc.goal is not None else ())):
        for leaf in _conditions(condition):
            named.update(ref.base for ref in leaf.args)
    for region in doc.regions:
        named.add(region.name)
        if region.target:
            named.add(region.target)

    silent = [o.name for o in doc.objects if not o.static and o.name not in named]
    if not silent:
        return []
    return [
        Finding(
            "declared-but-unmeasured",
            f"{', '.join(sorted(silent))} are placed in the scene and named by no clause -- "
            "neither init, goal nor observe measures them",
            "objects",
            severity="note",
        )
    ]


def _undeclared_types(doc: TaskDoc, catalog: Any) -> list[Finding]:
    """Types the document declares that the catalog has never heard of.

    The typechecker cannot see these, and the reason is structural rather than
    an oversight: it resolves *references*, so an instance no clause mentions is
    never looked up. Five of the seventeen shipped documents declare a fixture
    that way -- ``peel_box``, ``sharps_box``, ``bench_block`` -- and each is real
    scenery the loader builds procedurally, so the scene is not wrong.

    What is missing is the declaration, and the consequence is exact: nothing may
    be *asked* of them. A goal cannot name ``box.top`` because no asset says the
    box has a top, and ``describe`` reports such an instance as a position and a
    rest flag rather than as a thing. That is the same 42-gap shape this
    repository already found once, in the same way -- invisible until something
    asked -- so it is reported rather than left to be rediscovered.

    A note, not an error. The corpus's own documents say why a declaration may
    contribute a name and a frame without contributing a body, and a lint that
    failed on those would be one nobody runs.
    """
    missing = sorted({o.type for o in doc.objects if o.type and not catalog.has_asset(o.type)})
    if not missing:
        return []
    return [
        Finding(
            "undeclared-type",
            f"{', '.join(missing)} are declared here and catalogued nowhere. Nothing can be asked "
            "of them beyond where they are -- no features, no states, no description",
            "objects",
            severity="note",
        )
    ]


# ------------------------------------------------------------------- walking
def _conditions(condition: Condition | None) -> Iterator[Condition]:
    """Every measuring clause under one condition, in document order: the leaves of :func:`_walk_conditions`."""
    return (node for node in _walk_conditions(condition) if not node.children)


def _walk_conditions(condition: Condition | None) -> Iterator[Condition]:
    """Every condition, leaves and combinators alike."""
    if condition is None:
        return
    yield condition
    argument = condition.modal_arg
    if isinstance(argument, Condition):
        yield from _walk_conditions(argument)
    for child in condition.children:
        yield from _walk_conditions(child)


def _arity(condition: Condition) -> int:
    """How many clauses a conjunction combines, or 0 for anything else.

    ``worst_of`` counts too: it is a different reading of the same shape, and a
    garment scored on the worst of fifteen creases has the same legibility
    problem as one scored on the share of fifteen placements.
    """
    if condition.modal:
        return 0
    return len(condition.children) if condition.head in ("and", "any", "or", "worst_of") else 0


def _leaves(node: Node | None) -> tuple[Leaf, ...]:
    return leaves(node) if node is not None else ()


def _depth(node: Node) -> int:
    children = node.children()
    return 1 + max((_depth(child) for child in children), default=0) if children else 1


def _reward_measure(section: dict[str, Any]) -> tuple[str, tuple[str, ...]] | None:
    """The atom and operands a ``progress``/``shrinking`` clause is scored on.

    Read off the parsed section rather than off the built clause, because a
    built :class:`~robosimstudio_bench.goals.reward.Progress` carries the
    node and the key and deliberately not the reference it came from.
    """
    for word in ("progress", "shrinking"):
        body = section.get(word)
        if body is None:
            continue
        parts = body if isinstance(body, list) else [body]
        for part in parts:
            if not isinstance(part, list) or not part:
                continue
            if str(part[0]).lstrip(":") in _GATE_WORDS + _MASK_WORDS:
                continue
            atom = measured_atom(part)
            return str(atom[0]).replace("-", "_"), tuple(str(a) for a in atom[1:])
    return None
