# SPDX-License-Identifier: Apache-2.0
"""Scripted contexts and probes: what a measure reads, dictated by the test.

A goal tree is a pure function of a snapshot, so every claim about goal semantics
can be made against inputs a test writes down. That is what makes the strict half
of the equivalence discipline possible at all: identical code has been measured
scoring 0.84 and 0.90 on consecutive folds of the same towel, so anything checked
through a live rollout can only ever be checked within a band -- while anything
checked against a scripted snapshot can be checked exactly.

Three things here:

* :func:`ctx` -- an ``WorldState`` carrying only what the caller sets. The
  environment's own fields (task, episode, handles, state) are left empty rather
  than built, because a combinator touches none of them and constructing a scene
  to satisfy a type annotation is how fast tests stop being fast.
* :func:`probe_for` -- a **real** ``ParticleProbe`` over synthetic positions.
  Real, not faked: the point is to test the production measure, and a fake probe
  would test the fake.
* :class:`Tape` -- a scripted sequence of verdicts, for driving temporal
  combinators through a story ("holds, holds, lapses, holds again") whose right
  answer is known by construction.
"""

from __future__ import annotations

import types
from collections.abc import Sequence
from typing import Any

import numpy as np

from robosimstudio.core.probes import ParticleProbe
from robosimstudio.scene.state import WorldState
from robosimstudio_bench.goals.context import EvalResult

__all__ = ["Tape", "ctx", "fold_context", "probe_for"]


def ctx(
    *,
    sim_time: float = 0.0,
    step: int = 0,
    body_q: np.ndarray | None = None,
    body_qd: np.ndarray | None = None,
    settled: np.ndarray | None = None,
    joint_target_q: np.ndarray | None = None,
    contacts: Any = None,
    particles: dict[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
    handles: Any = None,
    state: Any = None,
) -> WorldState:
    """An ``WorldState`` carrying only what this test needs.

    The real type rather than a stand-in, so a measure that reaches for a field
    nobody set fails here the same way it would fail in an environment.
    """
    return WorldState(
        handles=handles,
        state=state,
        step=step,
        sim_time=sim_time,
        body_q=np.zeros((0, 7)) if body_q is None else np.asarray(body_q, dtype=np.float64),
        body_qd=np.zeros((0, 6)) if body_qd is None else np.asarray(body_qd, dtype=np.float64),
        settled_body_q=np.zeros((0, 7)) if settled is None else np.asarray(settled, dtype=np.float64),
        joint_target_q=np.zeros(0) if joint_target_q is None else np.asarray(joint_target_q, dtype=np.float64),
        contacts=contacts,
        particles=dict(particles or {}),
        # The field is `entities` now; this fixture keeps the older parameter
        # name so its callers are untouched while the alias lives.
        entities=dict(extras or {}),
    )


def probe_for(rest: np.ndarray, sets: dict[str, np.ndarray] | None = None, *, baseline: bool = True) -> ParticleProbe:
    """A real ``ParticleProbe`` over synthetic positions.

    Args:
        rest: Positions, ``[n, 3]``. Used as both the live state and, unless
            ``baseline`` is false, the settled baseline -- which is what every
            material-frame measure is stated against.
        sets: Named subsets by *global* index. The whole body is registered under
            ``"c"`` when nothing is given.
        baseline: Whether to snapshot ``rest`` as the baseline. False leaves a
            probe with no baseline, which is how a measure's "you never settled
            me" path gets tested.
    """
    model = types.SimpleNamespace(particle_count=len(rest))
    probe = ParticleProbe(model, dict(sets or {"c": np.arange(len(rest))}))
    probe.positions = rest.copy()
    probe.velocities = np.zeros_like(rest)
    if baseline:
        probe.baseline = rest.copy()
    return probe


def fold_context(rest: np.ndarray, *, name: str = "c", axis: int = 1, origin: float = 0.0):
    """A probe split into moving and standing halves, plus a context reading it.

    The rig ``TestFoldMetric`` builds, promoted: a flat sheet, the two halves a
    fold maps onto each other, and a context whose ``probe`` returns it. Returns
    ``(probe, context)``.
    """
    moving = np.flatnonzero(rest[:, axis] > origin)
    standing = np.flatnonzero(rest[:, axis] <= origin)
    probe = probe_for(
        rest,
        {name: np.arange(len(rest)), f"{name}/moving": moving, f"{name}/base": standing},
    )
    context = ctx(particles={name: probe})
    # The production measures reach for ``ctx.probe(name)``; WorldState resolves
    # that through its ``particles`` dict, so nothing here needs to fake it.
    return probe, context


class Tape:
    """A scripted sequence of verdicts, replayed one call at a time.

    Drives a temporal combinator through a story whose right answer is known:
    "true for four steps, false for one, true again" is the shape that proves an
    interruption *restarts* a hold rather than resuming it, and no rollout can be
    asked to produce that reliably.

    The last entry repeats, so a tape can be shorter than the run that reads it.
    """

    def __init__(self, script: Sequence[tuple[bool, float]], label: str = "leaf") -> None:
        self.script = list(script)
        self.label = label
        self.calls = 0

    def __call__(self, _ctx: WorldState) -> EvalResult:
        success, reward = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return EvalResult(success=bool(success), reward=float(reward), metrics={"value": float(reward)})

    def rewind(self) -> None:
        """Start the story again -- what a reset test needs."""
        self.calls = 0
