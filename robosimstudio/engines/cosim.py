# SPDX-License-Identifier: Apache-2.0
"""The per-control-step tick, minus the engine.

A co-simulation tick is the same five moves whatever engine is under it: read
this step's rigid poses, drive the mirrors across the engine's substeps,
advance, harvest, write the deformable state back into Newton's particle
arrays. Only the middle of that is the engine's -- setting a transform, calling
its step -- and :class:`CosimRig` is the rest, held once so a second engine does
not arrive with a second copy.

The rig owns no engine handle and calls no engine method. An adapter drives it:

    targets = rig.body_poses(state)
    for _sub, poses in rig.substeps_of(mirrors, targets):
        for mirror, pose in poses:
            ...engine: put ``mirror`` at ``pose``, and zero its velocity if
               ``rig.teleport``...
        ...engine: advance by ``rig.substep_dt(dt)``...
    rig.commit(targets)
    ...engine: harvest...
    for actor in actors:
        actor.prev = rig.write_particles(state, actor.start, actor.count, positions, dt, actor.prev)

**Why the mirror drive is interpolated rather than jumped.** A control step is
16.7 ms at 60 Hz and a hand can cross several millimetres in one; a mirror
teleported that far between engine steps arrives already overlapping whatever it
was approaching, and contact detection at step endpoints tunnels straight
through a sheet. Sweeping it in ``substeps`` small aims is what makes the
approach-speed envelope usable, and it is also what gives a displacement-based
friction model the right surface velocity for free.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .pose import compose_many, lerp_poses

__all__ = [
    "CosimRig",
    "engine_dt",
]


def engine_dt(control_dt: float, substeps: int) -> float:
    """The one contract for a foreign engine's step: the control period over its substeps.

    Both now compute this; an engine that fixes its step at construction passes
    the arrangement's control period, one that steps to a ``dt`` it is handed
    passes that.
    """
    return float(control_dt) / max(1, int(substeps))


@dataclass(frozen=True)
class _Drive:
    """Which mirrors this rig poses, and how to lay their inputs out as arrays.

    Rebuilt only when the mirror set or the mirrored body set changes, which on
    a running island is never: an island's colliders are built once and its
    bodies come from the rig's own fixed list. Holding it is what turns the
    per-tick work from "walk every mirror" into two small array reads, and at
    2748 mirrors that is 4.1 ms a control step against 0.5 ms.

    The layout is by *body*, not by mirror, because a mirror set is much larger
    than the set of bodies under it -- a decomposed hull is dozens of shapes on
    one link, so twelve worlds of this bench are 2748 mirrors on 264 bodies.
    The sweep is a property of a body, so it is computed once per body and
    ``rows`` fans it out to the mirrors riding on it.

    Attributes:
        mirrors: The list that was passed in, held for an identity check. A
            caller that builds a fresh list every tick gets a fresh plan every
            tick -- correct, and exactly what the per-tick gather cost before.
        keys: The bodies ``targets`` carried, so a rig whose mirrored set
            changed underneath it rebuilds rather than indexing the old layout.
        kept: The mirrors that are posed here, in yield order. A mirror with no
            body, or one whose body is not in ``targets``, is world-fixed and
            was posed once at build.
        bodies: The distinct bodies ``kept`` rides on, in ``rows`` order.
        rows: For each kept mirror, its body's row in ``bodies`` ``[n]``.
        local: Each kept mirror's body-local transform ``[n, 7]``.
    """

    mirrors: Any
    keys: frozenset
    kept: list
    bodies: list
    rows: np.ndarray
    local: np.ndarray

    @classmethod
    def build(cls, mirrors, targets) -> _Drive:
        kept, rows = [], []
        slot: dict[int, int] = {}
        for mirror in mirrors:
            body = getattr(mirror, "body", -1)
            if body < 0 or body not in targets:
                continue
            kept.append(mirror)
            rows.append(slot.setdefault(body, len(slot)))
        return cls(
            mirrors=mirrors,
            keys=frozenset(targets),
            kept=kept,
            bodies=sorted(slot, key=slot.get),
            rows=np.asarray(rows, dtype=np.intp),
            local=np.asarray([m.local for m in kept], dtype=np.float64).reshape(-1, 7),
        )


class CosimRig:
    """The engine-neutral half of one host island's tick.

    Args:
        bodies: The Newton body indices the mirrors ride on. Only these are
            read back from the device, which is what keeps ``pose_source=
            "fresh"`` to one small copy rather than the whole body array.
        substeps: Engine steps per control step.
        pose_source: ``"fresh"`` reads this step's poses off the state, costing
            one device read and coupling with zero lag; ``"snapshot"`` reads the
            batched host copy, which is free and one control period stale.
    """

    def __init__(self, bodies, *, substeps: int = 1, pose_source: str = "fresh") -> None:
        self.bodies = tuple(sorted({int(b) for b in bodies if int(b) >= 0}))
        self.substeps = max(1, int(substeps))
        self.pose_source = str(pose_source)
        self.pipeline: Any = None
        self.extra_states: tuple[Any, ...] = ()
        self._prev: dict[int, np.ndarray] = {}
        self._teleport = True
        self._plan: _Drive | None = None

    # ------------------------------------------------------------- wiring
    def attach_pipeline(self, pipeline: Any) -> None:
        """Hand the rig the pipeline whose states it writes. Idempotent."""
        self.pipeline = pipeline

    def attach_states(self, states) -> None:
        """Pipeline-less wiring -- the repro runner's states, both of them."""
        self.extra_states = tuple(states)

    def states(self, state: Any) -> tuple[Any, ...]:
        """Every state the write-back must reach.

        Both halves of the double buffer when there is one: the tick runs in
        the eager ``after`` phase, outside the captured graph, so writing only
        one leaves the swap free to resurrect a stale copy next step.
        """
        if self.pipeline is not None:
            return (self.pipeline.state_0, self.pipeline.state_1)
        if self.extra_states:
            return self.extra_states
        return (state,)

    # -------------------------------------------------------------- teleport
    @property
    def teleport(self) -> bool:
        """Whether this tick is a jump rather than a sweep.

        True for the first tick and the one after a reset, where the previous
        pose is meaningless: interpolating from it would sweep a mirror across
        the scene and through anything in the way.
        """
        return self._teleport

    def mark_teleport(self) -> None:
        """The next tick jumps. What a reset calls."""
        self._teleport = True

    # ------------------------------------------------------------ pose feed
    def body_poses(self, state: Any, bodies=None) -> dict[int, np.ndarray]:
        """World pose per mirrored body -- or per one of ``bodies`` -- by the configured source."""
        bodies = self.bodies if bodies is None else tuple(int(b) for b in bodies)
        if not bodies:
            return {}
        pipeline = self.pipeline
        if self.pose_source == "snapshot" and pipeline is not None and not getattr(pipeline, "settling", False):
            snapshot = getattr(pipeline, "snapshot", None)
            body_q = getattr(snapshot, "body_q", None) if snapshot is not None else None
            if body_q is not None and len(body_q):
                return {b: np.asarray(body_q[b], dtype=np.float64) for b in bodies}
        body_q = state.body_q.numpy()
        return {b: np.asarray(body_q[b], dtype=np.float64) for b in bodies}

    def substep_dt(self, dt: float) -> float:
        """One engine step's worth of the control period: :func:`engine_dt` of this rig's substeps."""
        return engine_dt(dt, self.substeps)

    def substeps_of(self, mirrors, targets: dict[int, np.ndarray]) -> Iterator[tuple[int, list]]:
        """Yield ``(index, [(mirror, world_pose), ...])`` per engine substep.

        A mirror with no body, or one whose body is not in ``targets``, is
        skipped: it is world-fixed and was posed once at build.

        The arithmetic is done for the whole mirror set at once, one array
        operation per substep rather than two per mirror per substep. That is
        the same numbers -- the batched forms in :mod:`.pose` are the scalar
        ones with a leading axis, checked against each other in
        ``tests/test_pose_batched.py`` -- and it matters because the count is
        large: a three-world surface-preparation scene mirrors 2808 shapes, so the
        per-mirror form makes 11232 microsecond-scale numpy calls a control
        step, and the loop's own overhead comes to dominate the work in it.

        What is deliberately still per mirror is the yielded list. A caller
        wants a mirror and its pose together, and the poses are rows of one
        array: zipping is a C-level walk, and a row is a view, so nothing is
        copied on the way out.

        The *layout* -- which mirrors are posed, and which body each rides --
        is held across ticks in a :class:`_Drive` and rebuilt only when the
        question changes. The poses are not held: they are read from ``targets``
        every tick, which is what makes a body that moved move its mirrors.
        """
        if self._teleport:
            self._prev = {b: np.asarray(p, dtype=np.float64).copy() for b, p in targets.items()}

        plan = self._plan
        if plan is None or plan.mirrors is not mirrors or plan.keys != targets.keys():
            plan = self._plan = _Drive.build(mirrors, targets)
        if not plan.kept:
            for sub in range(self.substeps):
                yield sub, []
            return

        prev = np.asarray([self._prev.get(b, targets[b]) for b in plan.bodies], dtype=np.float64)
        goal = np.asarray([targets[b] for b in plan.bodies], dtype=np.float64)
        for sub in range(self.substeps):
            t = (sub + 1.0) / self.substeps
            here = lerp_poses(prev, goal, t)
            yield sub, list(zip(plan.kept, compose_many(here[plan.rows], plan.local), strict=True))

    def commit(self, targets: dict[int, np.ndarray]) -> None:
        """Close the tick: this step's poses become the next one's previous."""
        self._teleport = False
        self._prev = {b: np.asarray(p, dtype=np.float64).copy() for b, p in targets.items()}

    # --------------------------------------------------------- particle sink
    def write_particles(
        self,
        state: Any,
        start: int,
        count: int,
        positions: np.ndarray,
        dt: float,
        prev: np.ndarray,
    ) -> np.ndarray:
        """One actor's positions into ``particle_q``/``particle_qd``.

        Velocity is finite-differenced against ``prev`` rather than read from
        the engine: the engines this serves either expose no per-node velocity
        or expose one that is not the deformable's under their own coupler.
        Returns the positions, to be held as the next tick's ``prev``.
        """
        q_new = np.ascontiguousarray(np.asarray(positions).reshape(-1, 3), dtype=prev.dtype)
        inv_dt = 1.0 / float(dt) if dt > 0.0 else 0.0
        qd = np.ascontiguousarray((q_new - prev) * inv_dt, dtype=np.float32)
        q32 = np.ascontiguousarray(q_new, dtype=np.float32)
        for target in self.states(state):
            target.particle_q[start : start + count].assign(q32)
            target.particle_qd[start : start + count].assign(qd)
        return q_new
