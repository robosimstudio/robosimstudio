# SPDX-License-Identifier: Apache-2.0
"""Semantic states: a named threshold on a measure, owned by the asset.

"Open", "closed", "flat", "folded", "adhered", "straight" -- the words a task
document and a structured query use. Each is a
measure plus a comparison plus a number, and the number is the interesting part:
it is a claim about *this asset*, not about the task using it. A drawer is open
past 80% of its travel whether the task is putting a bowl away or taking one out,
and stating that once on the drawer is the difference between a vocabulary and a
pile of task-local constants.

The unification worth noticing: a drawer's ``open`` and a shirt's ``folded`` are
the same construct. One is backed by a joint coordinate and the other by a shape
functional over particles, but both are a named, thresholded, provenance-carrying
quantity, and a structured consumer does not care which. That is why
this lives beside the feature machinery rather than inside any one substrate.

States are **derived, never stored**. Nothing latches here and nothing caches: a
state is evaluated from the world as it currently is, so a drawer pushed shut
stops being open. Goals that need "it happened at least once" say so with the
``once`` combinator, which is the layer that owns that question.

Like every threshold in this stack, a state carries its provenance. A number
whose justification survives only as a comment beside it is a number nothing can
check, and the states here will be read by an agent deciding whether its task is
expressible -- which is a worse audience for folklore than a human reviewer.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["COMPARISONS", "StateSpec", "state_to_dict"]

COMPARISONS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
}
"""Comparisons a thresholded term may use. Closed for the same reason the
combinator set is: an open comparison vocabulary is an expression language.

Here rather than beside the combinators, which moved to the benchmark tier's
``goals`` package: a state *is* a measure, one of these and a number, and an
asset saying it is ``open`` should not need the goal machinery to say what
``>=`` means."""


@dataclass(frozen=True, kw_only=True)
class StateSpec:
    """One named state an asset can be in.

    Attributes:
        name: The word a document and a structured query use -- ``open``, ``flat``.
        measure: Which atom or published state it thresholds.
        op: One of ``>=``, ``>``, ``<=``, ``<``.
        value: The threshold.
        summary: What being in this state means, in one line.
        unit: Physical unit of ``value``, empty for a ratio or fraction.
        provenance: Where the number came from. Required in spirit; a state that
            cannot say why its gate sits where it does is a guess wearing a name.
        of: Which feature the measure applies to, when an asset has more than one
            that could answer -- a cabinet with two drawers needs to say which.
    """

    name: str
    measure: str
    op: str = ">="
    value: float = 0.0
    summary: str = ""
    unit: str = ""
    provenance: str = ""
    of: str = ""

    def holds(self, measured: float) -> bool:
        """Whether a measured value puts the asset in this state."""
        return COMPARISONS[self.op](float(measured), float(self.value))

    def problems(self) -> list[str]:
        """Everything structurally wrong with this declaration."""
        out: list[str] = []
        where = f"state {self.name!r}"
        if not self.name:
            out.append("state has no name")
        if not self.measure:
            out.append(f"{where}: names no measure, so nothing can evaluate it")
        if self.op not in COMPARISONS:
            out.append(f"{where}: unknown comparison {self.op!r}; one of {sorted(COMPARISONS)}")
        if not self.summary:
            out.append(f"{where}: no summary, so the catalog cannot explain it")
        if not self.provenance:
            out.append(f"{where}: gate {self.value} has no provenance -- say where the number came from")
        return out


def state_to_dict(spec: StateSpec) -> dict[str, Any]:
    """A state as plain JSON-able data, for the catalog an agent reads."""
    out: dict[str, Any] = {
        "name": spec.name,
        "measure": spec.measure,
        "op": spec.op,
        "value": spec.value,
        "summary": spec.summary,
    }
    if spec.of:
        out["of"] = spec.of
    if spec.unit:
        out["unit"] = spec.unit
    if spec.provenance:
        out["provenance"] = spec.provenance
    return out
