# SPDX-License-Identifier: Apache-2.0
"""What a spec field *is*, beside its value -- and what changing it costs.

A dataclass field has always carried its value and its type. What it has not
carried is everything a tool needs to offer that value to somebody: the unit it
is in, the range worth sweeping, whether it reads linearly or across decades,
and -- the load-bearing one -- **what has to happen before a change takes
effect**.

That last one is not presentation. Newton keeps most cloth constitutive
parameters as Warp arrays the solver re-reads on every substep, so writing one
mid-run lands on the next step and does so even inside a captured CUDA graph:
the graph bakes the array's pointer, not the numbers in it. Others are host
scalars passed by value into kernels, which a capture *does* bake, so changing
one costs a recapture. Others still are consumed at build time -- turned into
tetrahedra, or into a solver's pre-allocated buffers -- and cannot move at all
without rebuilding the model. Three quite different costs, none of them visible
from the field declaration, and every tool that wants to expose a parameter has
to know which it is looking at.

Recording it here states it once, next to the value, instead of once per tool.
And attaching it to the field rather than keeping a table alongside is what
stops the two drifting: a field added without a description is visibly missing
one, where a table simply fails to mention it and nobody notices.

The tiers, in the order they cost:

* :data:`LIVE` -- a Warp array on the model. Assign into it and the next substep
  reads it. Safe under graph capture.
* :data:`RECAPTURE` -- a host scalar. Writable any time, but a captured graph
  holds the old value until it is recaptured. Free when running eager.
* :data:`REBUILD` -- consumed at build or solver construction. Needs a fresh
  model.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import Field, dataclass, fields
from typing import Any

__all__ = [
    "LIVE",
    "REBUILD",
    "RECAPTURE",
    "TIERS",
    "ParamUI",
    "iter_params",
    "param_ui",
    "ui",
    "undescribed",
]

LIVE = "live"
"""Takes effect on the next substep. A Warp array the solver re-reads."""

RECAPTURE = "recapture"
"""Takes effect once the CUDA graph is recaptured, or immediately when eager."""

REBUILD = "rebuild"
"""Takes effect only in a freshly built model."""

TIERS = (LIVE, RECAPTURE, REBUILD)
"""Every tier, cheapest first."""

_KEY = "ui"
"""Key this module owns inside ``field.metadata``, which is shared."""


@dataclass(frozen=True, kw_only=True)
class ParamUI:
    """How one spec field should be presented, and what changing it costs.

    Attributes:
        tier: One of :data:`TIERS` -- what has to happen for a change to land.
        unit: SI unit for display, e.g. ``"N/m"``. Empty for dimensionless.
        log: Whether the field spans decades and wants a logarithmic control.
            Stiffnesses almost always do: the interesting range of ``edge_ke``
            is six orders of magnitude wide and a linear slider spends all of
            its travel in the top decade.
        range: ``(low, high)`` worth sweeping, or ``None`` to leave it open.
            Not a validity bound -- a spec's own ``__post_init__`` is where
            invalid lives. This is where *useful* is, which is a different and
            softer claim.
        step: Increment for a linear control, or ``None`` to let the tool pick.
        group: Which cluster of related parameters this belongs to, so a tool
            can lay a spec out the way somebody thinks about it rather than in
            declaration order.
        doc: One line, when the field name alone does not carry it. The
            attribute's own docstring stays the long-form home; this is what
            fits beside a slider.
        choices: Allowed values for a string field, which turns it into a
            dropdown rather than a text box.
    """

    tier: str = REBUILD
    unit: str = ""
    log: bool = False
    range: tuple[float, float] | None = None
    step: float | None = None
    group: str = ""
    doc: str = ""
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {self.tier!r}")
        if self.range is not None:
            low, high = self.range
            if not low < high:
                raise ValueError(f"range must be increasing, got {self.range}")
            if self.log and low <= 0.0:
                raise ValueError(f"a log-scaled range needs a positive low end, got {self.range}")


def ui(
    *,
    tier: str = REBUILD,
    unit: str = "",
    log: bool = False,
    range: tuple[float, float] | None = None,
    step: float | None = None,
    group: str = "",
    doc: str = "",
    choices: tuple[str, ...] = (),
) -> dict[str, ParamUI]:
    """Describe a field, for :func:`dataclasses.field`'s ``metadata``.

    Returns a single-key mapping so it composes with any other metadata a field
    carries::

        tri_ke: float = field(
            default=1.0e4,
            metadata=ui(tier=LIVE, unit="N/m", log=True, range=(1e2, 1e6), group="in-plane"),
        )
    """
    return {_KEY: ParamUI(tier=tier, unit=unit, log=log, range=range, step=step, group=group, doc=doc, choices=choices)}


def param_ui(spec: Any, name: str) -> ParamUI | None:
    """The description of one field, or ``None`` if it carries none.

    Args:
        spec: A dataclass or an instance of one.
        name: Field name.

    Raises:
        KeyError: If ``spec`` has no such field. Undescribed and absent are
            different answers and worth telling apart -- the first is a field
            somebody has not got to yet, the second is a typo.
    """
    for f in fields(spec):
        if f.name == name:
            return f.metadata.get(_KEY)
    raise KeyError(f"{getattr(spec, '__name__', type(spec).__name__)} has no field {name!r}")


def iter_params(spec: Any, *, tier: str = "", group: str = "") -> Iterator[tuple[Field, ParamUI]]:
    """Every described field of ``spec``, in declaration order.

    Args:
        spec: A dataclass or an instance of one.
        tier: Keep only this tier, or all of them when empty.
        group: Keep only this group, or all of them when empty.
    """
    for f in fields(spec):
        described = f.metadata.get(_KEY)
        if described is None:
            continue
        if tier and described.tier != tier:
            continue
        if group and described.group != group:
            continue
        yield f, described


def undescribed(spec: Any) -> tuple[str, ...]:
    """Names of the fields carrying none -- what a coverage test asserts is empty.

    The whole reason the description lives on the field is that this question
    has an answer at all. A table beside the spec cannot be asked it.
    """
    return tuple(f.name for f in fields(spec) if _KEY not in f.metadata)
