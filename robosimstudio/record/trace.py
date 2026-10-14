# SPDX-License-Identifier: Apache-2.0
"""Golden traces: what the evaluator said, step by step, and how to compare two.

The arbiter every port is judged against. A re-expressed evaluator must produce
the same verdicts on the same episodes as the one it replaces, and "the same" has
to mean something precise -- so this module records the verdict sequence and
reduces it to a fixed set of statistics that a later run can be checked against.

**Why a summary and not the raw sequence.** Per-step equality is not available.
Identical code has been measured scoring 0.84 and 0.90 on consecutive folds of
one towel: the nondeterminism is GPU-atomic, not seeded, and
no amount of care removes it. A raw trace diff would therefore be red on every
run and would teach everyone to ignore it. What *is* stable is the shape of the
episode, and each field below is chosen because a specific kind of mistake moves
it and little else does:

``metric_keys``
    Never banded. An evaluator that stops emitting a measure, or starts emitting
    one, is a hard diff -- that is a change to the benchmark's surface, not to a
    number in it.
``success_first_step``
    The sharpest probe available. A latch, a hold or an ``after`` re-expressed
    wrongly moves *this* and often nothing else, because the terminal verdict
    still ends up correct.
``reward_auc``
    Catches a reward re-expression that gets the ending right and the journey
    wrong -- which is exactly what swapping a progress term for a satisfied-share
    does.
``per-metric min / max / final / mean / argmax_step``
    A ``peak`` re-expression moves ``max`` and ``argmax_step`` and leaves
    ``final`` alone. Recording both is what tells those apart.
``per-bool true_fraction / first_true_step / n_transitions``
    The transition count is what catches a debounce or a hold-duration change; a
    flag that flickers twice as often is doing something different even when it
    ends in the same place.

**Bands are measured, not chosen.** A band comes from running the same variant
several times on one machine and recording the spread -- and where the repeats
agree exactly, the band collapses to equality, which is what happens for every
rigid suite and for anything driven by an idle policy. A band nobody measured is
a tolerance somebody wanted, and this module will not invent one.

The recorder duck-types into ``rollout``'s existing ``recorder`` argument, so
capturing a trace needs no change to the environment, the runner or any task.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "MIN_REPEATS",
    "TraceDiff",
    "TraceRecorder",
    "TraceSummary",
    "bands_from_repeats",
    "compare",
    "read_summary",
    "summarise",
    "write_summary",
]


# --------------------------------------------------------------------- record
class TraceRecorder:
    """Captures one episode's verdict sequence, as ``rollout`` runs it.

    Duck-types :class:`~robosimstudio.record.recorder.EpisodeRecorder`'s
    three methods, so it drops into ``rollout(recorder=...)`` unmodified. It
    reads only what the step already produced and asks the environment for
    nothing -- the same contract the episode recorder holds itself to, and for
    the same reason: a recorder that queries the simulator is a recorder that can
    change what it is recording.

    Failures disable it rather than raising. A trace is evidence *about* a run,
    and losing the run because the evidence-gathering broke is the wrong trade.
    """

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.enabled = True
        self.failures: list[str] = []
        self.episode: Any = None

    def begin(self, episode: Any, env: Any) -> None:
        del env
        self.episode = episode
        self.rows = []

    def step(self, env: Any, action: Any, obs: Any, reward: float, info: dict[str, Any], stage: Any = None) -> None:
        # `stage` is accepted and dropped: this class duck-types `EpisodeRecorder`,
        # which grew the parameter, and a trace that did not follow it
        # made the certifier's solve stage raise `TypeError` before its first
        # step, under every verb. A trace records
        # verdicts, not frames, so which stage produced a row is not a thing it
        # has anywhere to put.
        del action, obs, stage
        if not self.enabled:
            return
        try:
            self.rows.append(
                {
                    "step": int(getattr(env, "step_count", len(self.rows))),
                    "sim_time": float(getattr(env, "sim_time", 0.0)),
                    "success": bool(info.get("success", False)),
                    "reward": float(reward),
                    "metrics": {k: _scalar(v) for k, v in (info.get("metrics") or {}).items()},
                    "close_kind": info.get("close_kind"),
                }
            )
        except Exception as exc:
            self._disable("step", exc)

    def end(self, env: Any, info: dict[str, Any], extra: Any = None) -> dict[str, Any]:
        del env, extra
        return {"trace_rows": len(self.rows), "trace_enabled": self.enabled, "close_kind": info.get("close_kind")}

    def _disable(self, where: str, exc: Exception) -> None:
        self.enabled = False
        self.failures.append(f"{where}: {type(exc).__name__}: {exc}")


def _scalar(value: Any) -> Any:
    """Coerce a metric to something a trace can compare.

    Bools stay bools -- their transition count is a signal in its own right --
    numbers become floats, and anything else is refused at record time rather
    than stringified. A silently stringified dict is a golden that can never
    diff, which is worse than a recording that stops and says why.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"metric value {value!r} is not a scalar; put structured state in a sidecar, not in metrics")


# -------------------------------------------------------------------- reduce
@dataclass
class TraceSummary:
    """The committed golden: fixed-size statistics over one verdict sequence."""

    task: str = ""
    episode_id: int = 0
    solution: str = ""
    metric_keys: tuple[str, ...] = ()
    episode_length: int = 0
    success: bool = False
    success_first_step: int = -1
    reward_final: float = 0.0
    reward_auc: float = 0.0
    close_kind: str | None = None
    numeric: dict[str, dict[str, float]] = field(default_factory=dict)
    boolean: dict[str, dict[str, float]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "episode_id": self.episode_id,
            "solution": self.solution,
            "metric_keys": list(self.metric_keys),
            "episode_length": self.episode_length,
            "success": self.success,
            "success_first_step": self.success_first_step,
            "reward_final": self.reward_final,
            "reward_auc": self.reward_auc,
            "close_kind": self.close_kind,
            "numeric": self.numeric,
            "boolean": self.boolean,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceSummary:
        return cls(
            task=data.get("task", ""),
            episode_id=int(data.get("episode_id", 0)),
            solution=data.get("solution", data.get("policy", "")),
            metric_keys=tuple(data.get("metric_keys", ())),
            episode_length=int(data.get("episode_length", 0)),
            success=bool(data.get("success", False)),
            success_first_step=int(data.get("success_first_step", -1)),
            reward_final=float(data.get("reward_final", 0.0)),
            reward_auc=float(data.get("reward_auc", 0.0)),
            close_kind=data.get("close_kind"),
            numeric={k: dict(v) for k, v in (data.get("numeric") or {}).items()},
            boolean={k: dict(v) for k, v in (data.get("boolean") or {}).items()},
            provenance=dict(data.get("provenance") or {}),
        )


def summarise(rows: list[dict[str, Any]], *, task: str = "", episode_id: int = 0, solution: str = "") -> TraceSummary:
    """Reduce a verdict sequence to the statistics a golden compares.

    ``episode_length`` is the environment's own step count, taken off the last
    row, and **not** ``len(rows)``. The two are the same number for every policy
    that advances one control step per ``rollout`` iteration, which was every
    policy this module was written against -- and they are not the same number
    for a graph. ``SolutionGraph`` blocks while its runtime streams a trajectory,
    so one iteration of the rollout loop can cover tens of control steps:
    measured on ``tape_peel/handle``, a 427-step peel produced **39** rows. Under
    ``len(rows)`` a golden recorded with ``--solve graph`` would have banded the
    *rollout's observation count*, which is a property of how the graph batches
    its tool calls rather than of the episode, and would have read as an 11x
    speedup against the scripted golden beside it.
    """
    length = int(rows[-1].get("step", len(rows))) if rows else 0
    summary = TraceSummary(task=task, episode_id=episode_id, solution=solution, episode_length=length)
    if not rows:
        return summary

    keys = sorted({key for row in rows for key in row["metrics"]})
    summary.metric_keys = tuple(keys)
    summary.success = bool(rows[-1]["success"])
    summary.success_first_step = next((int(r["step"]) for r in rows if r["success"]), -1)
    summary.reward_final = float(rows[-1]["reward"])
    summary.reward_auc = float(statistics.fmean(float(r["reward"]) for r in rows))
    summary.close_kind = rows[-1].get("close_kind")

    for key in keys:
        # Paired with each row's step: an event-gated metric is absent from
        # the rows before its event, so a position in the series is not a
        # position in ``rows``.
        pairs = [(row["step"], row["metrics"][key]) for row in rows if key in row["metrics"]]
        if not pairs:
            continue
        steps = [step for step, _ in pairs]
        series = [value for _, value in pairs]
        if isinstance(series[0], bool):
            trues = [i for i, v in enumerate(series) if v]
            summary.boolean[key] = {
                "true_fraction": len(trues) / len(series),
                "first_true_step": float(steps[trues[0]]) if trues else -1.0,
                # How often it flips, which is what catches a debounce or a
                # hold-duration change that leaves the ending identical.
                "n_transitions": float(sum(1 for a, b in itertools.pairwise(series) if a != b)),
            }
        else:
            try:
                values = [float(v) for v in series]
            except (TypeError, ValueError):
                # A label such as ``awaiting`` or ``worst``: nothing to band.
                continue
            best = max(range(len(values)), key=lambda i: values[i])
            summary.numeric[key] = {
                "min": min(values),
                "max": max(values),
                "final": values[-1],
                "mean": statistics.fmean(values),
                "argmax_step": float(steps[best]),
            }
    return summary


# ------------------------------------------------------------------- compare
@dataclass
class TraceDiff:
    """What changed between a recorded golden and a fresh run."""

    hard: list[str] = field(default_factory=list)
    """Differences no band can excuse: a changed metric surface, a changed
    verdict. These are the benchmark itself moving."""
    banded: list[str] = field(default_factory=list)
    """Numeric fields outside their measured band."""
    within: list[str] = field(default_factory=list)
    """Fields that moved but stayed inside their band -- reported, not failed,
    because a reviewer should see the drift even when the gate passes."""

    @property
    def ok(self) -> bool:
        return not self.hard and not self.banded

    def report(self) -> str:
        lines = []
        for label, entries in (("HARD", self.hard), ("OUT OF BAND", self.banded), ("within band", self.within)):
            for entry in entries:
                lines.append(f"  [{label}] {entry}")
        return "\n".join(lines) or "  (identical)"


def compare(got: TraceSummary, want: TraceSummary, bands: dict[str, tuple[float, float]] | None = None) -> TraceDiff:
    """Diff a fresh summary against a golden, under measured bands.

    ``bands`` maps ``"<section>.<key>.<stat>"`` to an absolute low/high the
    difference may fall in. A field with no band must match exactly -- which is
    the right default, and is what most of the corpus ends up under.
    """
    bands = bands or {}
    diff = TraceDiff()

    if got.metric_keys != want.metric_keys:
        added = sorted(set(got.metric_keys) - set(want.metric_keys))
        removed = sorted(set(want.metric_keys) - set(got.metric_keys))
        diff.hard.append(f"metric surface changed: added {added}, removed {removed}")
    if got.success != want.success:
        diff.hard.append(f"success: {want.success} -> {got.success}")
    if got.close_kind != want.close_kind:
        diff.hard.append(f"close_kind: {want.close_kind!r} -> {got.close_kind!r}")

    _check(diff, bands, "success_first_step", float(want.success_first_step), float(got.success_first_step))
    _check(diff, bands, "reward_final", want.reward_final, got.reward_final)
    _check(diff, bands, "reward_auc", want.reward_auc, got.reward_auc)
    _check(diff, bands, "episode_length", float(want.episode_length), float(got.episode_length))

    for section, mine, theirs in (("numeric", got.numeric, want.numeric), ("boolean", got.boolean, want.boolean)):
        for key, stats in theirs.items():
            if key not in mine:
                diff.hard.append(f"{section} metric {key!r} is no longer reported")
                continue
            for stat, value in stats.items():
                _check(diff, bands, f"{section}.{key}.{stat}", float(value), float(mine[key].get(stat, math.nan)))
    return diff


def _check(diff: TraceDiff, bands: dict[str, tuple[float, float]], field_name: str, want: float, got: float) -> None:
    if math.isnan(got) and math.isnan(want):
        return
    delta = got - want
    band = bands.get(field_name)
    if band is None:
        if got != want:
            diff.banded.append(f"{field_name}: {want} -> {got} (no band; must match exactly)")
        return
    low, high = band
    if low <= delta <= high:
        if delta:
            diff.within.append(f"{field_name}: {want} -> {got} (delta {delta:+.4g}, band [{low:+.4g}, {high:+.4g}])")
    else:
        diff.banded.append(f"{field_name}: {want} -> {got} (delta {delta:+.4g}, outside [{low:+.4g}, {high:+.4g}])")


# --------------------------------------------------------------------- bands
MIN_REPEATS = 5
"""Runs below which a stability classification is not evidence.

Measured, painfully. A variant was recorded twice, agreed exactly, and was
reported as deterministic; a later pair of runs of the *same* variant disagreed
on three fields. GPU-atomic nondeterminism does not have to show up in every
pair, so two agreeing samples say almost nothing -- and the conclusion drawn from
them ("most of the corpus is exactly gated") was wrong in the other direction
too: at six repeats, 4 of 16 variants are exact and the rest need bands.

Five is a floor rather than a target. A variant whose verdict *flips* between
runs needs more than a spread; it needs a rate, and that means many more.
"""


def bands_from_repeats(summaries: list[TraceSummary], *, k: float = 0.5) -> dict[str, tuple[float, float]]:
    """Measure bands from repeated runs of one variant on one machine.

    The band is the observed spread, widened by ``k`` times itself on each side.
    Where the repeats agree exactly the band collapses to zero width, so a
    genuinely deterministic variant ends up exactly gated -- but see
    :data:`MIN_REPEATS` before believing that of any particular one.

    ``k`` is fixed in code rather than per variant. A per-variant ``k`` is
    tolerance-widening with extra steps, and the whole point of measuring is to
    stop that being a judgement call.
    """
    if len(summaries) < 2:
        return {}
    fields: dict[str, list[float]] = {}
    for summary in summaries:
        fields.setdefault("success_first_step", []).append(float(summary.success_first_step))
        fields.setdefault("reward_final", []).append(summary.reward_final)
        fields.setdefault("reward_auc", []).append(summary.reward_auc)
        fields.setdefault("episode_length", []).append(float(summary.episode_length))
        for section, table in (("numeric", summary.numeric), ("boolean", summary.boolean)):
            for key, stats in table.items():
                for stat, value in stats.items():
                    fields.setdefault(f"{section}.{key}.{stat}", []).append(float(value))

    bands: dict[str, tuple[float, float]] = {}
    for name, values in fields.items():
        spread = max(values) - min(values)
        margin = k * spread
        bands[name] = (min(values) - max(values) - margin, max(values) - min(values) + margin)
    return bands


# ---------------------------------------------------------------------- i/o
def write_summary(path: str | Path, summary: TraceSummary) -> None:
    """Write a golden, with stable key order so a diff is about content."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n")


def read_summary(path: str | Path) -> TraceSummary:
    """Read a golden back."""
    return TraceSummary.from_dict(json.loads(Path(path).read_text()))
