# SPDX-License-Identifier: Apache-2.0
"""What a rig is telling you, in units cloth is specified in.

A stiffness slider on its own is not tuning, it is guessing -- ``edge_ke = 6.6``
means nothing to anybody, and the only way to know whether it is right is to
compare something measurable against real fabric. So every rig reports readings
rather than solver state, and the cantilever reports the one number textiles are
actually characterised by.

Readings are deliberately allowed to be unavailable. A cantilever whose droop
has run off the end of the well-conditioned band has *not* told you what the
fabric is, and :func:`~robosimstudio.domains.cloth.bending_length_from_droop`
returns NaN rather than a clamped number for exactly that reason. A hole that
says "out of band" is worth more than a plausible figure that is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from robosimstudio.domains.cloth import DROOP_BAND, bending_length_from_droop, resembles

__all__ = ["Reading", "measure_cantilever", "measure_sheet", "relax", "settle_speed"]


def relax(pipeline: Any, *, damp_every: int = 3, tol: float = 1.0e-6, max_steps: int = 3000) -> tuple[bool, int]:
    """Step until the cloth stops moving, and report whether it got there.

    A bending length is a property of an **equilibrium** shape -- that is what
    the elastica describes -- so reading one off a strip that is still ringing
    measures the swing rather than the fabric. This is the same kinetic damping
    the cloth drape bench settles with: velocities are zeroed periodically,
    which removes energy without touching any force, so the shape converged to
    is the one an undamped strip would eventually ring down to, reached in a
    fraction of the steps.

    Returns:
        ``(settled, steps)``. ``settled`` is False if it ran out of steps, which
        makes any measurement taken afterwards a lower bound and worth saying so
        rather than quietly reporting.
    """
    previous = np.asarray(pipeline.state.particle_q.numpy(), dtype=np.float64).copy()
    for step in range(max_steps):
        pipeline.step()
        if (step + 1) % damp_every:
            continue
        pipeline.state.particle_qd.zero_()
        current = np.asarray(pipeline.state.particle_q.numpy(), dtype=np.float64)
        moved = float(np.abs(current - previous).max())
        previous = current.copy()
        if moved < tol:
            return True, step + 1
    return False, max_steps


@dataclass(frozen=True)
class Reading:
    """One measured quantity, ready to be shown.

    Attributes:
        name: What it is.
        value: The number, or NaN when it could not be measured.
        unit: Its unit, empty for a ratio.
        note: Why it is unavailable, or what it is consistent with.
    """

    name: str
    value: float
    unit: str = ""
    note: str = ""

    @property
    def known(self) -> bool:
        return bool(np.isfinite(self.value))

    def text(self) -> str:
        """One line, for a read-only widget."""
        if not self.known:
            return f"-- {self.note}".strip()
        body = f"{self.value:.4g} {self.unit}".strip()
        return f"{body}  ({self.note})" if self.note else body


def _positions(entity: Any, state: Any) -> np.ndarray:
    """This cloth's particle positions, on the mid-surface for a slab.

    A slab has a particle on each face, so a raw z would read whichever face
    happened to be sampled and swing by a thickness. The mid-surface is what the
    membrane's single sheet already is, and what a droop should be measured on.
    """
    if hasattr(entity, "mid_surface_positions"):
        try:
            mid = entity.mid_surface_positions(state)
            if mid is not None and len(mid):
                return np.asarray(mid, dtype=np.float64)
        except (AttributeError, ValueError, TypeError):
            pass
    return np.asarray(state.particle_q.numpy(), dtype=np.float64)[entity.particles]


def _rest_positions(entity: Any) -> np.ndarray | None:
    """The settled rest layout this cloth resets to, or ``None``."""
    rest = getattr(entity, "rest", None)
    if not rest or "particle_q" not in rest:
        return None
    return np.asarray(rest["particle_q"], dtype=np.float64)[entity.particles]


def settle_speed(entity: Any, state: Any, *, percentile: float = 95.0) -> Reading:
    """How fast the cloth is still moving [m/s].

    A high percentile rather than a maximum: on self-contacting cloth a single
    particle is always being squeezed out from between two layers, so a maximum
    never falls below any threshold and stops being a settle test.
    """
    qd = np.asarray(state.particle_qd.numpy(), dtype=np.float64)[entity.particles]
    if not len(qd):
        return Reading("settle speed", float("nan"), "m/s", "no particles")
    speed = np.linalg.norm(qd, axis=1)
    return Reading("settle speed", float(np.percentile(speed, percentile)), "m/s", "95th pct")


def measure_sheet(entity: Any, state: Any) -> list[Reading]:
    """Readings any sheet can give: how far it spread, how high it stands.

    ``spread`` is measured against the **rest layout** and stays that way, which
    is worth saying because there is an atom that looks like it and is not.
    ``cloth_coverage`` divides by the *settled* footprint -- "as spread as it
    settled" -- which is the right denominator for scoring a fold and the wrong
    one for tuning a material, where the question is precisely how far the sheet
    moved from flat on its way to settling. Two ratios, two references; replacing
    one with the other would leave this page reporting 1.0 for a fabric that
    collapsed the moment it was let go.
    """
    q = _positions(entity, state)
    if not len(q):
        return []
    extent_x = float(q[:, 0].max() - q[:, 0].min())
    extent_y = float(q[:, 1].max() - q[:, 1].min())
    readings = [
        Reading("footprint", extent_x * extent_y, "m^2"),
        Reading("height", float(q[:, 2].max() - q[:, 2].min()), "m", "z extent"),
        settle_speed(entity, state),
    ]
    rest = _rest_positions(entity)
    if rest is not None and len(rest) == len(q):
        flat = float((rest[:, 0].max() - rest[:, 0].min()) * (rest[:, 1].max() - rest[:, 1].min()))
        if flat > 0.0:
            readings.insert(0, Reading("spread", extent_x * extent_y / flat, "", "vs laid out"))
    return readings


def declared_states(record: Any, snapshot: Any) -> list[Reading]:
    """Each semantic state the garment declares, and whether it holds now.

    The gates a material is *supposed* to reach. A towel declares ``flat`` as
    ``cloth_coverage >= 0.9`` and ``folded`` as ``fold_extent <= 0.6``, with a
    paragraph of provenance on each -- calibrated numbers, sitting in the asset,
    invisible on the page where somebody is dragging stiffness sliders trying to
    reach them.

    Reported as the measured value with its gate in the note, rather than as a
    bare yes: "0.94, gate >= 0.9" says how much room is left, and a tuning pass is
    entirely about how much room is left.

    Args:
        record: The garment's ``AssetRecord``, or ``None`` for a cloth nothing
            declares -- a scratch sheet in the material rig, which is most of
            what this page is pointed at.
        snapshot: A frozen snapshot to measure through.

    Returns:
        One reading per declared state measured on the sheet alone. **A state
        naming a feature is skipped**, and that is a scope decision rather than a
        gap: the towel's ``folded`` is ``fold_extent`` *of* its ``half_fold``
        template, and turning a feature name into the payload a handler wants is
        operand resolution, which the benchmark's ``goals.bind`` owns and
        does in eight hundred lines against a document. This page is often
        pointed at a scratch sheet with no document at all, and a second
        resolver living here to serve it would be the copy this whole exercise
        is about removing.

        The states that survive are the ones a *material* pass is about anyway.
        "Is this fabric still lying flat after it settled" is a question about
        the fabric; "did it fold along its declared crease" is a question about
        the task, and the task's page already answers it.
    """
    if record is None or snapshot is None:
        return []
    from robosimstudio.assetlib.semantics.atoms import get_atom
    from robosimstudio.assetlib.semantics.handlers import dispatch

    cloth = next(iter(getattr(snapshot, "particles", {}) or {}), "")
    if not cloth:
        return []

    out: list[Reading] = []
    for state in getattr(record, "states", ()) or ():
        if state.of:
            continue
        try:
            atom = get_atom(state.measure.replace("-", "_"))
            spec = dispatch(atom.name, ("",), (record.substrate,))
        except (KeyError, LookupError):
            continue
        try:
            reported = spec.fn(snapshot, cloth)
        except Exception:
            continue
        value = reported.get(atom.primary) if isinstance(reported, dict) else reported
        if value is None:
            continue
        held = "holds" if state.holds(float(value)) else "not yet"
        out.append(Reading(state.name, float(value), state.unit, f"{held}, gate {state.op} {state.value}"))
    return out


def measure_cantilever(entity: Any, state: Any, *, overhang: float, clamp_x: float) -> list[Reading]:
    """Turn a settled cantilever into a bending length.

    The droop is read at the tip relative to the clamp and divided by the
    overhang, which is the dimensionless quantity the elastica is inverted
    through -- so the areal density cancels and the answer depends on nothing
    but the shape the strip settled into.

    Args:
        entity: The built strip.
        state: Current state.
        overhang: How far the strip reaches past its clamp [m].
        clamp_x: Where the clamp ends [m], in the same frame.
    """
    q = _positions(entity, state)
    rest = _rest_positions(entity)
    if not len(q) or rest is None or len(rest) != len(q) or not overhang > 0.0:
        return [Reading("bending length", float("nan"), "cm", "rig not measurable")]

    # Tip and clamp are identified on the *rest* layout: once the strip has
    # drooped its tip is no longer the largest x, and a set defined by where
    # particles are now would wander as the shape changed.
    tip = np.flatnonzero(rest[:, 0] >= rest[:, 0].max() - 1e-9)
    root = np.flatnonzero(rest[:, 0] <= clamp_x + 1e-9)
    if not len(tip) or not len(root):
        return [Reading("bending length", float("nan"), "cm", "no tip or clamp row")]

    # The mid-width of the tip row, not its mean: a strip's free corners curl,
    # and averaging them in reads the curl as though it were less droop -- which
    # comes back as a stiffer fabric than the one being measured. This is the
    # sample the cloth drape bench takes, and taking a different one is
    # enough on its own to disagree with the number recorded beside a material.
    tip_mid = tip[np.argsort(q[tip][:, 1])[len(tip) // 2]]
    droop = float(np.mean(q[root][:, 2]) - q[tip_mid][2])
    ratio = droop / float(overhang)
    readings = [Reading("tip droop", droop, "m"), Reading("droop / overhang", ratio)]

    low, high = DROOP_BAND
    if not low <= ratio <= high:
        where = "barely bent" if ratio < low else "hung straight down"
        readings.append(Reading("bending length", float("nan"), "cm", f"droop {ratio:.2f} out of band -- {where}"))
        return readings

    bending_cm = bending_length_from_droop(ratio, float(overhang)) * 100.0
    like = resembles(bending_cm)
    note = f"like {', '.join(like)}" if like else "no reference fabric in this band"
    readings.append(Reading("bending length", bending_cm, "cm", note))
    return readings
