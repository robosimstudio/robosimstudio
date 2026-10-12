# SPDX-License-Identifier: Apache-2.0
"""The rate a planned path is densified at, which is the rate it runs at.

The densifiers themselves stay with their tasks and should: the weave solves at
4 cm cut points and smoothsteps as a joint chain because solving every stream
frame cost 157-485 s a station and poisoned its own seeds, and that reasoning is
about the weave and nothing else. What all three copies shared was the *rate
lookup* and the hazard behind it.

The hazard, in the tool-hanging suite's words, because it put it best: the streaming
path feeds one waypoint per control step, so densifying at a nominal 30 while
the cell runs at 60 "would run every stage in half its stated time, which on the
turn to vertical is the difference between a tool that rotates in the jaws and
one that is flung out of them." A stage's waypoint count *is* its duration.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PLAN_HZ", "stream_hz"]

PLAN_HZ = 30.0
"""Densification rate for a cell that declares no control rate [Hz].

**A floor, not the rate.** It shipped as "half the control rate, dense enough
that streaming is indistinguishable from per-step interpolation", and that is
only true if something clocks the waypoints out at 30. Nothing does. It survives
only so a cell stating no rate still gets a finite one.
"""


def stream_hz(physics: Any, *, default: float = PLAN_HZ) -> float:
    """The rate to densify a planned path to, read off the arrangement.

    Args:
        physics: The task's arrangement, or anything carrying ``control_hz``.
            Read with ``getattr`` rather than by attribute, because a cell may
            state no arrangement at all and a planner still has to pick a rate.
        default: What to use when it states no rate. The floor, not a
            preference -- see :data:`PLAN_HZ`.

    Returns:
        The cell's control rate, or ``default`` when it declares none. A
        declared rate of 0 counts as declaring none, which is why this is
        ``or`` and not a ``None`` check: an arrangement carrying 0.0 would
        otherwise densify a stage into no waypoints at all.
    """
    return float(getattr(physics, "control_hz", 0.0) or default)
