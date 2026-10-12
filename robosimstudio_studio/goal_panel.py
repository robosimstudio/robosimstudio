# SPDX-License-Identifier: Apache-2.0
"""The task, while it runs: its goal as a tree, its init, and what it observes.

What the page said about a task was one string in a collapsed folder::

    reward 0.62  |  hammer_0/bin2_hammer_region 1.00  drill_0/... 0.00

That is a flat list of leaves, and a goal is not flat. ``cloth_fold/towel_half``
is ``once(and(<= fold-extent 0.6, >= fold-score 0.85, hold-for 5 s (settled and
not held)))`` -- four combinators, a duration and three thresholds, none of which
survived into that line. So a run that was not succeeding could be read as "0.62"
and not as "the fold is there, the sheet has been still for 1.2 of the 5 seconds
it owes".

Two things make the tree renderable, and both were already there:

* **Structure** comes from :func:`~robosimstudio_bench.goals.goal.outline`, which
  recovers the document's own clauses as data, labelled the way
  ``build_goal`` labels the nodes that measure them, so the two join.
* **Verdicts and readings** come from ``metrics``: every node writes its own
  boolean under its label, and
  :func:`~robosimstudio_bench.goals.goal.reading` knows the three keys a clause's
  number can land under.

**Three items to a line, and one of them moves.** A clause is its verdict, its
measure, and what it currently reads::

    ✗ fold-extent   1.00

Not its threshold, not its comparison, and not its operands. The document says
``(<= (fold-extent towel towel.half_fold) 0.6)`` and every part of that except
the reading is a constant -- printing them put the one number that changes fourth
in a line of things that never do. The tick and the cross carry the verdict, so
nothing has to be compared by eye to find out whether a clause holds.

**Operands come back only when a line would otherwise be a duplicate.** The
towel's clauses each name a different measure, so ``fold-extent`` is the whole
identity of one. The shirt's are three ``fold-score``s over three creases and
tool sort's are four ``containment``s over four bins -- there, dropping the
operands would print one line four times, and *which* one failed is exactly the
diagnostic those documents are written to give.

**The clauses are the whole of the open section.** Everything else this panel
knows is behind one fold, because everything else is either a constant of the
load or a second way of saying what the tree already says:

* the **instruction**, which is one sentence of English -- ``sort all the tools
  from the cluttered bin1 pile into the compartment of bin2 labelled for their
  type`` -- standing as two wrapped lines over the four ``containment`` clauses
  that state it in the terms the run is scored on, and the step count and reward
  that ride with it;
* ``(:init)``, which says whether the episode started where the document said it
  would -- already computed at every reset and, until now, thrown away into a
  console line;
* ``(:observe)``, what the document records while gating nothing, which is the
  uncalibrated half of a task and the part somebody is usually watching when they
  open the page at all;
* the **staging**, which the document left to convention.

**And the page unrolls once.** A fold inside a fold is a thing you have to find
twice.

**This panel owns the once-per-step evaluation.** Evaluators are stateful --
``once`` latches, ``hold-for`` accumulates simulated seconds, ``peak`` and
``integral`` accumulate, a smoothed leaf keeps a window -- so a second call in
one step advances a clock only one step passed on, and a five-second hold would
come out satisfied in half the time it was asked for. Hence: evaluate in
:meth:`drain`, never in :meth:`refresh`, and sit out of the session's drain
beside its ``CONTROL_PANELS`` while a policy drives the environment (which
evaluates as it steps), so the policy's step does not get a second one.

**Which episode is loaded is not here.** It is a property of what the page has
opened, like the scene and the robot, so it lives on the task menu beside them
(:class:`~robosimstudio_studio.bench_panels.PanelTask`) rather than as a
slider under the goal it has nothing to do with.
"""

from __future__ import annotations

from typing import Any

from .panels import Panel

__all__ = ["PanelGoal"]

TICK = "✓"
CROSS = "✗"
BULLET = "·"

MAX_DEPTH = 8
"""How deep a goal tree is drawn before it is elided.

No shipped goal is deeper than five (``cloth_fold/towel_half``'s ``held`` clause
sits at ``goal.0.2.0.1.0``). The bound is here so a malformed or generated
document cannot turn a refresh into an unbounded string build.
"""


class PanelGoal(Panel):
    """Task, goal tree, init report and observed terms.

    Args:
        session: The page.
    """

    def __init__(self, session) -> None:
        super().__init__(session)
        self._outline: Any = None
        self._qualify: frozenset[str] = frozenset()
        """Labels of clauses that need their operands printed to be told apart."""
        self._init: dict[str, Any] = {}
        self._observed: Any = None
        self._build_outline()

    # ------------------------------------------------------------------- build
    def _build_outline(self) -> None:
        """Walk the document's goal once, at load. Structure does not change."""
        doc = getattr(self.env, "doc", None)
        if doc is None or getattr(doc, "goal", None) is None:
            self._outline, self._qualify = None, frozenset()
            return
        from robosimstudio_bench.goals.goal import outline

        self._outline = outline(doc.goal)
        self._qualify = _ambiguous(self._outline)

    def build(self, gui) -> None:
        # `Goal`, not `Task`: the menu immediately above is called Task and is
        # where a task is *chosen*, and a sidebar with two sections of one name
        # makes you read both to find out which one you wanted.
        with gui.add_folder("Goal"):
            # The clauses, and nothing above them.
            self._tree = gui.add_markdown(self._tree_text())
            with gui.add_folder("Detail", expand_by_default=False):
                self._header = gui.add_markdown(self._header_text())
                if self._observed_terms():
                    self._observed = gui.add_markdown(self._observed_text())
                self._static = gui.add_markdown(self._static_text())

    # ------------------------------------------------------------------- drain
    def drain(self) -> None:
        """Score this step. **Exactly once**, and only from here.

        See the module docstring: the evaluator is stateful and a second call
        advances clocks that only one step passed on.
        """
        self.env.evaluate()

    # ----------------------------------------------------------------- refresh
    def refresh(self) -> None:
        self._header.content = self._header_text()
        self._tree.content = self._tree_text()
        if self._observed is not None:
            self._observed.content = self._observed_text()

    def on_reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        del obs
        self._init = dict(info.get("init") or {})
        self._static.content = self._static_text()
        self._header.content = self._header_text()

    # -------------------------------------------------------------- the header
    def _header_text(self) -> str:
        """What is being asked in English, and where the run has got to.

        Behind the fold, because the instruction is the same sentence for the
        life of a load and the tree it stands over says the same thing in the
        terms the run is actually scored on. The step count and the reward come
        with it, and this is the only place on the page that counts steps while
        nothing is driving -- the transport reports a *rate* and the policy
        panel's progress line exists only while a policy is running. Behind one
        click is the price of the two wrapped lines it cost above the clauses.

        No task name: the menu directly above says which task is loaded.
        """
        env = self.env
        limit = int(env.max_steps or env.task.max_steps or 0)
        result = env.last_result
        verdict = f"{TICK} success" if result.success else f"reward {result.reward:.2f}"
        step = f"{env.step_count}/{limit}" if limit else f"{env.step_count}"
        return f"*{env.episode.instruction}*\n\nstep {step} {BULLET} {verdict}"

    # ---------------------------------------------------------------- the tree
    def _tree_text(self) -> str:
        """The goal, one line per clause, in the document's own order.

        Falls back to the flat metric list for a task scored by a hand-written
        evaluator, which has no tree to walk. That is not a lesser rendering of
        the same thing -- there is nothing better to say about it, which is what
        the fallback admits.
        """
        metrics = self.env.last_result.metrics
        if self._outline is None:
            if not metrics:
                return "*no goal tree; this task is scored by a hand-written evaluator*"
            rows = [
                f"- `{k}` {v:.2f}" if isinstance(v, float) else f"- `{k}` {v}" for k, v in list(metrics.items())[:8]
            ]
            return "\n".join(rows)

        rows: list[str] = []
        self._render(self._outline, metrics, self._leaves_by_label(), rows, depth=0)
        return "\n".join(rows)

    def _leaves_by_label(self) -> dict[str, Any]:
        """The built tree's leaves, keyed the way the outline is.

        The join between what the document says and what is measuring it.
        """
        goal = getattr(self.env, "goal", None)
        if goal is None:
            return {}
        from robosimstudio_bench.goals.goal import leaves

        return {leaf.label: leaf for leaf in leaves(goal)}

    def _render(self, clause, metrics: dict, leaves: dict, rows: list[str], *, depth: int, note: str = "") -> None:
        if depth > MAX_DEPTH:
            rows.append(f"{'  ' * depth}- *(deeper clauses elided)*")
            return
        rows.append(f"{'  ' * depth}- {self._clause_line(clause, metrics, leaves)}{note}")
        # `worst_of` names its weakest member, by label, and the shirt's document
        # calls that the diagnostic: three creases score and the one dragging the
        # rest down is the thing to go and look at. It used to be printed as a
        # trailing `worst goal.0.1.2` line -- when it was printed at all, which
        # for the shirt it was not: the key is namespaced by the clause that
        # owns it, so the bare lookup found nothing on the one goal that has a
        # `worst_of` in it. A label is a row of this tree, so it is marked there.
        worst = _metric(metrics, clause.label, "worst") if clause.head == "worst_of" else None
        for child in clause.children:
            marked = " ← worst" if worst is not None and child.label == worst else ""
            self._render(child, metrics, leaves, rows, depth=depth + 1, note=marked)

    def _clause_line(self, clause, metrics: dict, leaves: dict) -> str:
        """One clause: whether it holds, what it is, and what it reads.

        A combinator says only its name and its verdict, because that is all it
        is -- except ``hold-for``, whose whole content is a clock, and which is
        the clause a fold most often fails on. It says the seconds it has banked,
        which is the one thing in this tree that moves without anything else
        moving.
        """
        row = [_mark(metrics.get(clause.label)), self._title(clause)]
        if clause.children:
            if clause.head == "hold-for":
                held = _metric(metrics, clause.label, "held_s")
                if held is not None:
                    row.append(f"{float(held):.1f} s")
            elif clause.modal_arg is not None:
                row.append(f"{clause.modal_arg:g} s")
            return " ".join(part for part in row if part)

        leaf = leaves.get(clause.label)
        if leaf is not None:
            from robosimstudio_bench.goals.goal import reading

            got = reading(metrics, leaf)
            # A predicate reports its own boolean and nothing else, and the tick
            # beside it already said that. Printing `True` next to `✓` is the
            # verdict twice.
            if got not in ("-", "True", "False"):
                row.append(got)
        return " ".join(part for part in row if part)

    def _title(self, clause) -> str:
        """What to call one clause: its measure, and its operands only if needed.

        See the module docstring. The operands are dropped because they are
        constants of the document, and put back for a measure the document uses
        more than once -- where they are the only thing telling two lines apart.
        An operand another operand reaches through goes either way: the shirt's
        creases are written over ``(shirt, shirt.left_side)`` and the word shirt
        twice is not a distinction.

        Not bold. A row is a bullet, a tick and three words; emphasis picks one
        thing out of many and every row here was picking itself out, which is
        the same as none of them doing it.
        """
        if clause.label not in self._qualify:
            return clause.head
        refs = _refs(clause.operands)
        return f"{clause.head} {'/'.join(refs)}" if refs else clause.head

    # ---------------------------------------------------------------- the fold
    def _static_text(self) -> str:
        """Where the episode started, and what the document left to convention.

        Both change only at load, so this is one string rebuilt on reset rather
        than on the refresh gate.
        """
        return f"{self._init_report()}\n\n**staged**\n\n{self._staging_text()}"

    def _init_report(self) -> str:
        """Whether the settled world is where the document said it would start.

        ``Env._verify_init`` has always computed this and written a summary into
        ``info``. It is the difference between a policy that failed and an
        episode that never started where it claimed to -- which are
        indistinguishable from a reward of zero.
        """
        doc = getattr(self.env, "doc", None)
        clauses = list(getattr(doc, "init", ()) or ())
        if not clauses:
            return "**start** *unconstrained*"
        report = getattr(self.env, "last_init", None)
        failed = set(getattr(report, "failed", ()) or ())
        holds = self._init.get("holds")
        rows = [f"**start** {'' if holds is None else _mark(holds)}".rstrip()]
        for clause in clauses:
            state = CROSS if clause.head in failed else (TICK if holds else BULLET)
            # Operands always, unlike the goal tree: an init is a list of where
            # things start and is almost always the same predicate over different
            # objects, so the objects are the whole of what each line says.
            refs = _refs(tuple(str(ref) for ref in getattr(clause, "args", ())))
            rows.append(f"- {state} {clause.head} {'/'.join(refs)}".rstrip())
        return "\n".join(rows)

    # ------------------------------------------------------------- the observed
    def _observed_terms(self) -> tuple:
        doc = getattr(self.env, "doc", None)
        return tuple(getattr(doc, "observe", ()) or ())

    def _observed_text(self) -> str:
        """The quantities the document records while gating nothing.

        Free: ``Scored.__call__`` already evaluates every observe node each step
        and merges its metrics, so these are live readings and not a second
        measurement. The uncalibrated half of a task, and reporting it is how it
        gets calibrated -- which is the document's own argument for having them.
        """
        metrics = self.env.last_result.metrics
        rows = ["**observed**"]
        for term in self._observed_terms():
            key = term.head.replace("-", "_")
            value = next((v for k, v in metrics.items() if k == key or k.endswith(f".{key}")), None)
            shown = f"{value:.3g}" if isinstance(value, float) else ("-" if value is None else str(value))
            rows.append(f"- {term.head} {shown}")
        return "\n".join(rows)

    # ------------------------------------------------------------- the staging
    def _staging_text(self) -> str:
        """What the document left to convention, and where it came from."""
        task = self.env.task
        rows = [
            f"- scene `{task.scene}`",
            f"- embodiment `{task.embodiment}`" + (f" gripper `{task.gripper}`" if task.gripper else ""),
            f"- physics `{'+'.join(dict.fromkeys(b.solver for _, b in task.physics.backends))}` at "
            f"{task.physics.control_hz:g} Hz, {task.physics.substeps} substeps",
            f"- control mode `{task.control_mode}`",
        ]
        handles = self.env.handles
        if handles.work_arm:
            support = f", support `{handles.support_arm}`" if handles.support_arm else ""
            rows.append(f"- work arm `{handles.work_arm}`{support}")
        doc = getattr(self.env, "doc", None)
        provenance = dict(getattr(doc, "provenance", {}) or {})
        if provenance.get("tier"):
            rows.append(f"- provenance `{provenance['tier']}`")
        return "\n".join(rows)


def _mark(verdict: Any) -> str:
    """``✓``/``✗`` for a clause that has been scored, nothing for one that has
    not -- which is every clause before the first step."""
    return "" if verdict is None else (TICK if verdict else CROSS)


def _refs(operands: tuple[str, ...]) -> list[str]:
    """A clause's operands, minus the ones another operand reaches through.

    ``(towel, towel.half_fold)`` is a sheet and a feature *of* that sheet, and
    printing both gives ``towel/towel.half_fold`` -- the word towel twice and one
    fact. A genuine pair like ``(hammer_0, bin2_hammer_region)`` keeps both
    halves, because neither implies the other and the pair is the whole of what
    the clause asks.
    """
    return [ref for ref in operands if not any(other.startswith(f"{ref}.") for other in operands)]


def _ambiguous(root) -> frozenset[str]:
    """Labels of the measuring clauses that a bare measure name cannot tell apart.

    A goal usually names each measure once -- the towel's four clauses are four
    different measures -- and there the measure alone is the clause's whole
    identity. The exceptions are the documents whose point *is* the repetition:
    the shirt scores three creases with ``fold-score`` and tool sort places four
    tools with ``containment``, and a rendering that dropped the operands there
    would print one line four times over.

    Computed once at load, off the outline, because that is where the document's
    own spelling of each clause is and it does not change while a task is open.
    """
    seen: dict[str, list[str]] = {}
    for clause in root.walk():
        if not clause.children:
            seen.setdefault(clause.head, []).append(clause.label)
    return frozenset(label for labels in seen.values() if len(labels) > 1 for label in labels)


def _metric(metrics: dict, label: str, name: str) -> Any:
    """A combinator's own reading, under its label or bare.

    Every node namespaces what its children reported by the child's label, so a
    ``hold-for`` at ``goal.0.2`` reports ``goal.0.2.hold_progress``. A modal does
    not namespace what it wraps, though -- ``Once`` passes its child's metrics
    through untouched -- so a ``hold-for`` sitting directly under one reports the
    bare key. Both are looked for, prefixed first, which is the same order
    :func:`~robosimstudio_bench.goals.goal.reading` resolves a leaf's number in.
    """
    value = metrics.get(f"{label}.{name}")
    return metrics.get(name) if value is None else value
