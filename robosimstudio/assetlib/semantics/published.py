# SPDX-License-Identifier: Apache-2.0
"""Published states: internal state, declared instead of reached for.

Some things only an asset can know. A tape's bond fraction is not derivable from
poses or contacts -- it is the state of a cohesive-zone field the tape itself
owns -- and the same is true of a latch flag, a peel front, a seat drive's
engagement. Today those travel through an untyped ``extras`` dict that an
evaluator reaches into by name and hopes.

This is the grown-up form of that: an asset **declares** what it publishes, with
a name, a type, a unit and a range. Nothing changes about who may read it inside
the simulator -- evaluators and privileged tools still see everything, because
they live inside the privilege boundary and nothing they see can cross it. What
changes is that reaching into an array becomes reading a declared member, so the
catalog can list it and the schema can validate a goal against it.

The **observability tier** is the part worth the extra field. It answers a
question the repo currently answers per *tool*: could a real rig know this?

``internal``
    Sim-side only. An evaluator may score on it; no policy may ever be handed it.
    Bond forces between particles are the honest example -- no instrument reads
    them.
``measurable``
    A real rig could instrument this, so a privileged tool may legitimately
    expose it and a de-privileged one may be built later.
``perceivable``
    A camera or a proprioceptive sensor could estimate it, so a non-privileged
    policy may be given it without the benchmark lying about what it knows.

That tiering is not a new access-control system. The privilege boundary is
already structural -- the in-process executor offers ground truth, the bridge
carries only what a real rig has, and an allowlist enforces it. Tiers just let
the allowlist be stated per member rather than per tool, which is what turns
"de-privilege the perception stack" from a rewrite into a member-by-member
migration with a visible frontier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["OBSERVABILITY", "PublishedState", "crossable", "published_to_dict"]

OBSERVABILITY: tuple[str, ...] = ("internal", "measurable", "perceivable")
"""Tiers, least observable first. Ordered so a comparison means something: a
bridge admitting ``measurable`` admits ``perceivable`` too."""


@dataclass(frozen=True, kw_only=True)
class PublishedState:
    """One state an asset exposes under a declared name.

    Attributes:
        name: How a goal, a structured observation or a tool refers to it.
        type: ``"float"``, ``"int"``, ``"bool"`` or ``"vector"``.
        summary: What it means, in one line.
        tier: One of :data:`OBSERVABILITY`.
        unit: Physical unit, empty for a ratio or a count.
        range: Low and high bounds, when the quantity has them. A fraction
            declaring ``(0.0, 1.0)`` lets a report plot it without guessing and
            lets a schema refuse a goal thresholding it at 1.4.
    """

    name: str
    type: str
    summary: str
    tier: str = "internal"
    unit: str = ""
    range: tuple[float, float] | None = None

    def problems(self) -> list[str]:
        """Everything structurally wrong with this declaration."""
        out: list[str] = []
        where = f"published state {self.name!r}"
        if not self.name:
            out.append("published state has no name")
        if self.tier not in OBSERVABILITY:
            out.append(f"{where}: unknown observability tier {self.tier!r}; one of {sorted(OBSERVABILITY)}")
        if self.type not in ("float", "int", "bool", "vector"):
            out.append(f"{where}: unknown type {self.type!r}")
        if not self.summary:
            out.append(f"{where}: no summary, so nothing can describe it")
        if self.range is not None and self.range[0] > self.range[1]:
            out.append(f"{where}: range low {self.range[0]} is above high {self.range[1]}")
        return out


def crossable(state: PublishedState, *, allow: str = "perceivable") -> bool:
    """Whether this member may cross the privilege boundary at ``allow``.

    Advisory to the transport, never a replacement for it: the allowlist in
    the graph-policy runtime's contract remains the thing that refuses,
    because a boundary enforced by whoever remembers to call a helper is not a
    boundary. What this gives the allowlist is a per-member answer to base its
    refusal on.
    """
    return OBSERVABILITY.index(state.tier) >= OBSERVABILITY.index(allow)


def published_to_dict(state: PublishedState) -> dict[str, Any]:
    """A declaration as plain JSON-able data, for the catalog."""
    out: dict[str, Any] = {
        "name": state.name,
        "type": state.type,
        "summary": state.summary,
        "tier": state.tier,
    }
    if state.unit:
        out["unit"] = state.unit
    if state.range is not None:
        out["range"] = list(state.range)
    return out
