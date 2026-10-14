# SPDX-License-Identifier: Apache-2.0
"""Deformable state, read back once per control step and queried on the host.

:class:`ParticleProbe` is the deformable counterpart of
:class:`~robosimstudio.core.probes.contact.ContactProbe`: one device-to-host
copy of ``particle_q`` per step, then everything an evaluator or an observation
wants answered out of that one array.

It is deliberately generic. Particles are addressed by explicit index arrays --
named *sets*, which the toolkit layer resolves from a spec on the rest layout --
and the queries are the ones that mean something for any deformable: where a
set is, how far it has moved from its settled baseline, how much of it lies in a
region, how large its footprint is. Nothing here knows about adhesion, bonds or
peel fronts; those are one task's physics and live with that task.

The settled baseline is the reference every displacement query is measured
against, and it is captured by the caller at the moment the scene declares
itself ready -- after the fine-dt settle, not at build time, since a cloth
built a centimetre above the table has not yet become the scene's start state.
"""

from __future__ import annotations

from collections.abc import Sequence

import newton
import numpy as np

__all__ = ["ParticleProbe"]


def _grid_area(pts: np.ndarray, cell: float) -> float:
    """Occupied area [m²] of ``pts`` on a ``cell``-sized xy grid; see :meth:`ParticleProbe.covered_area`."""
    if not len(pts):
        return 0.0
    size = float(cell)
    if not size > 0.0:
        raise ValueError(f"grid cell must be positive, got {cell}")
    # A diverged solver hands back NaN positions, and casting those to an
    # integer bin is undefined -- numpy warns and produces garbage indices,
    # so the metric would report a confident number for a scene that no
    # longer exists. NaN in, NaN out: the caller can then say "diverged"
    # instead of quoting a coverage.
    pts = pts[np.isfinite(pts[:, :2]).all(axis=1)]
    if not len(pts):
        return float("nan")
    keys = np.floor(pts[:, :2] / size).astype(np.int64)
    # One pass, no dict: pack the two indices into a single integer and
    # count the distinct ones. Cloth is thousands of particles per step and
    # this is called every step.
    packed = (keys[:, 0] - keys[:, 0].min()) * (int(np.ptp(keys[:, 1])) + 1) + (keys[:, 1] - keys[:, 1].min())
    return float(len(np.unique(packed))) * size * size


class ParticleProbe:
    """Particle positions for one step, plus named subsets and a baseline.

    Args:
        model: The finalized model, for the total particle count.
        sets: Named index arrays into ``particle_q``. A cloth entity registers
            its own full range plus whatever sub-regions its spec named (a tab,
            a hem, two corners), so a task refers to ``"tape/tab"`` rather than
            to raw indices.
        world: Which world these indices belong to. Recorded for handles and
            batching; single-world today.
    """

    def __init__(
        self,
        model: newton.Model,
        sets: dict[str, Sequence[int]] | None = None,
        *,
        world: int = 0,
    ) -> None:
        self.model = model
        self.world = int(world)
        self.sets: dict[str, np.ndarray] = {name: np.asarray(idx, dtype=np.int32) for name, idx in (sets or {}).items()}
        self.positions = np.zeros((model.particle_count, 3), dtype=np.float64)
        """This step's particle positions [m], shape ``[particle_count, 3]``."""
        self.velocities = np.zeros((model.particle_count, 3), dtype=np.float64)
        """This step's particle velocities [m/s], shape ``[particle_count, 3]``."""
        self.baseline: np.ndarray | None = None
        """The settled positions displacements are measured from, or ``None``."""
        self.step: int | None = None
        """Control step :attr:`positions` was read at, or ``None`` if never."""

    # ----------------------------------------------------------------- lifecycle
    def add_set(self, name: str, indices: Sequence[int]) -> np.ndarray:
        """Register a named particle set. Returns the stored index array."""
        idx = np.asarray(indices, dtype=np.int32)
        self.sets[name] = idx
        return idx

    def update(self, state: newton.State, step: int | None = None) -> None:
        """Read this step's positions and velocities. Once per control step, after physics.

        Both come back in the same call for the reason ``WorldState`` gathers
        ``body_qd`` alongside ``body_q``: a predicate asking whether a sheet has
        come to rest would otherwise pay its own device-to-host copy every step,
        and "is it still moving" is not answerable from positions alone.

        Args:
            state: The state to read.
            step: The control step this cloud belongs to, which makes the call
                idempotent within a step. An environment reads the cloud twice
                per step otherwise -- once for ``obs["deformables"]`` and once
                for the evaluator -- and on a garment that is two copies of
                every vertex for an identical answer. ``None`` always reads.
        """
        if step is not None and step == self.step:
            return
        if self.model.particle_count:
            self.update_arrays(state.particle_q.numpy(), state.particle_qd.numpy(), step)
        else:
            self.step = step

    def update_arrays(self, positions, velocities, step: int | None = None) -> None:
        """Take this step's cloud from host arrays already read back.

        What the pipeline's batched read-back hands over, so a probe never pays
        a second device-to-host copy for an array the step already copied.
        Same idempotence as :meth:`update`.
        """
        if step is not None and step == self.step:
            return
        self.step = step
        self.positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        self.velocities = np.asarray(velocities, dtype=np.float64).reshape(-1, 3)

    def snapshot_baseline(self, state: newton.State | None = None) -> np.ndarray:
        """Record the current positions as the settled reference.

        Args:
            state: Read from this state first. ``None`` uses whatever
                :meth:`update` last read, which is what an environment reset
                does after restoring its rest snapshot.
        """
        if state is not None:
            self.update(state)
        self.baseline = self.positions.copy()
        return self.baseline

    # -------------------------------------------------------------------- access
    def indices(self, name: str | None = None) -> np.ndarray:
        """The index array of a named set, or every particle when ``name`` is None."""
        if name is None:
            return np.arange(self.model.particle_count, dtype=np.int32)
        if name not in self.sets:
            raise KeyError(f"unknown particle set {name!r}; have {sorted(self.sets)}")
        return self.sets[name]

    def points(self, name: str | None = None) -> np.ndarray:
        """Positions of a named set [m], shape ``[n, 3]``."""
        return self.positions[self.indices(name)]

    def speeds(self, name: str | None = None) -> np.ndarray:
        """Speeds of a named set [m/s], shape ``[n]``."""
        vel = self.velocities[self.indices(name)]
        return np.linalg.norm(vel, axis=1) if len(vel) else np.zeros(0)

    def max_speed(self, name: str | None = None) -> float:
        """The fastest particle in a set [m/s]."""
        speeds = self.speeds(name)
        return float(speeds.max()) if len(speeds) else 0.0

    def settle_speed(self, name: str | None = None, *, quantile: float = 0.95) -> float:
        """A robust "is it still moving" speed for a set [m/s].

        A high quantile rather than :meth:`max_speed`, and the difference
        matters. Cloth under self-contact pops individual particles constantly:
        measured on a settling towel, the fastest particle spikes between 0.02
        and 0.46 m/s from step to step while the sheet is visibly still, so a
        maximum never falls below a rest threshold and "has it settled" is never
        answered yes. The mean is no good either -- it hides a corner that is
        genuinely whipping, which is the thing worth catching. A corner in
        motion is hundreds of particles and moves the 95th percentile; one
        popping vertex is not and does not.
        """
        speeds = self.speeds(name)
        return float(np.quantile(speeds, quantile)) if len(speeds) else 0.0

    def centroid(self, name: str | None = None) -> np.ndarray:
        """Mean position of a named set [m], shape ``[3]``."""
        pts = self.points(name)
        return pts.mean(axis=0) if len(pts) else np.zeros(3)

    def extent(self, name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds of a named set [m], as ``(lo, hi)``."""
        pts = self.points(name)
        if not len(pts):
            return np.zeros(3), np.zeros(3)
        return pts.min(axis=0), pts.max(axis=0)

    def height_band(self, name: str | None = None) -> tuple[float, float]:
        """Lowest and highest z of a named set [m]."""
        lo, hi = self.extent(name)
        return float(lo[2]), float(hi[2])

    # --------------------------------------------------------------- displacement
    def displacement(self, name: str | None = None) -> np.ndarray:
        """Per-particle displacement from the settled baseline [m], shape ``[n, 3]``."""
        if self.baseline is None:
            raise RuntimeError("no settled baseline; call snapshot_baseline() when the scene is ready")
        idx = self.indices(name)
        return self.positions[idx] - self.baseline[idx]

    def max_displacement(self, name: str | None = None) -> float:
        """Largest distance any particle of a set has moved from settled [m]."""
        d = self.displacement(name)
        return float(np.linalg.norm(d, axis=1).max()) if len(d) else 0.0

    def mean_displacement(self, name: str | None = None) -> float:
        """Mean distance a set's particles have moved from settled [m]."""
        d = self.displacement(name)
        return float(np.linalg.norm(d, axis=1).mean()) if len(d) else 0.0

    def lifted_fraction(self, name: str | None = None, *, height: float = 0.01) -> float:
        """Fraction of a set that has risen more than ``height`` above settled."""
        d = self.displacement(name)
        return float((d[:, 2] > height).mean()) if len(d) else 0.0

    # -------------------------------------------------------------------- regions
    def in_box(
        self,
        lo: Sequence[float],
        hi: Sequence[float],
        name: str | None = None,
    ) -> np.ndarray:
        """Boolean mask of which of a set's particles lie in an axis-aligned box."""
        pts = self.points(name)
        if not len(pts):
            return np.zeros(0, dtype=bool)
        lo_v = np.asarray(lo, dtype=np.float64)
        hi_v = np.asarray(hi, dtype=np.float64)
        return np.all((pts >= lo_v) & (pts <= hi_v), axis=1)

    def fraction_in_box(self, lo: Sequence[float], hi: Sequence[float], name: str | None = None) -> float:
        """Fraction of a set inside an axis-aligned box."""
        mask = self.in_box(lo, hi, name)
        return float(mask.mean()) if len(mask) else 0.0

    def footprint_area(self, name: str | None = None) -> float:
        """Area of the set's xy bounding rectangle [m²].

        The cheap coverage measure: a garment spread flat covers close to its
        rest footprint, one balled up covers a fraction of it. A convex hull
        would be tighter, and needs scipy; this needs nothing and moves in the
        same direction, which is what a metric wants.
        """
        lo, hi = self.extent(name)
        return float(max(hi[0] - lo[0], 0.0) * max(hi[1] - lo[1], 0.0))

    def covered_area(self, name: str | None = None, *, cell: float = 0.01) -> float:
        """Area the set actually covers from above [m²], on an occupancy grid.

        Particle positions are binned into ``cell``-sized squares in xy and the
        occupied squares counted. What "how flat is this garment" means, and
        what :meth:`footprint_area` does not: a bounding rectangle is defined by
        four extreme particles, so a garment lying in a ball with one sleeve
        flicked out scores nearly the same as one spread flat. An unfolding task
        scored on the rectangle can be solved by flicking the sleeve out, and
        will be.

        The cell size is the resolution of the answer and wants to be a few
        times the mesh spacing: below that, the gaps *between* particles start
        reading as uncovered and a flat sheet stops scoring 1; far above it, two
        different garments round to the same number. It is a parameter rather
        than a constant because that ratio is a property of the cloth, not of
        this function.

        Args:
            name: Which set, or the whole probe's default.
            cell: Grid square edge [m].

        Returns:
            Occupied area [m²]. Zero for an empty set.
        """
        return _grid_area(self.points(name), cell)

    def coverage(self, name: str | None = None, *, cell: float = 0.01) -> float:
        """Covered area as a fraction of what the set covered when it settled.

        The measure an unfolding task is scored on. 1.0 is "as spread as the
        garment was when it was laid out flat"; a crumpled start is a few tenths
        of that. Can exceed 1.0 slightly -- a sheet stretched taut between two
        grippers covers more than it did lying relaxed -- which is honest and is
        left unclamped.

        Uses the settled baseline as the reference rather than the garment's
        rest *area*, because those are different numbers for a garment: a shirt
        laid flat covers less table than its fabric would if you could iron it
        into a single layer, since it is two panels on top of each other.
        """
        if self.baseline is None:
            raise RuntimeError("no settled baseline; call snapshot_baseline() when the scene is ready")
        # The denominator is binned by the same function as the numerator, so a
        # baseline taken from a diverged state reads NaN rather than counting
        # its NaN rows as an occupied cell.
        flat = _grid_area(self.baseline[self.indices(name)], cell)
        return 0.0 if flat <= 1.0e-9 else self.covered_area(name, cell=cell) / flat

    def spread_ratio(self, name: str | None = None) -> float:
        """Footprint area as a fraction of the settled footprint.

        1.0 is "as spread as it settled"; folding halves it, balling it up
        drives it towards 0. Undefined without a baseline, and reported as 0
        when the baseline footprint is degenerate.
        """
        if self.baseline is None:
            raise RuntimeError("no settled baseline; call snapshot_baseline() when the scene is ready")
        idx = self.indices(name)
        base = self.baseline[idx]
        if not len(base):
            return 0.0
        base_area = float(max(base[:, 0].max() - base[:, 0].min(), 0.0) * max(base[:, 1].max() - base[:, 1].min(), 0.0))
        return 0.0 if base_area <= 1.0e-9 else self.footprint_area(name) / base_area

    def near(self, position: Sequence[float], radius: float, name: str | None = None) -> np.ndarray:
        """Indices of a set's particles within ``radius`` [m] of a point."""
        idx = self.indices(name)
        pts = self.positions[idx]
        if not len(pts):
            return np.zeros(0, dtype=np.int32)
        d = np.linalg.norm(pts - np.asarray(position, dtype=np.float64), axis=1)
        return idx[d < float(radius)]

    def summary(self, name: str | None = None) -> dict[str, float]:
        """A compact observation of one set: centroid, height band, footprint.

        This is what an environment puts in ``obs["deformables"]`` -- enough for
        a policy to reason about a cloth without shipping thousands of
        particles per step, and cheap enough to record every step.
        """
        centroid = self.centroid(name)
        z_lo, z_hi = self.height_band(name)
        out = {
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "z_min": z_lo,
            "z_max": z_hi,
            "footprint_area": self.footprint_area(name),
            "max_speed": self.max_speed(name),
        }
        if self.baseline is not None:
            out["max_displacement"] = self.max_displacement(name)
            out["spread_ratio"] = self.spread_ratio(name)
        return out
