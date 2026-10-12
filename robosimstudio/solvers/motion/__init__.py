# SPDX-License-Identifier: Apache-2.0
"""Getting somewhere: gripper commands, OSC, servoing, and planning.

What all four have in common is that they run *over time*. An IK solve is a
function; a motion is a sequence with a rate, a tolerance and a way of failing
part-way, which is why they live apart from :mod:`robosimstudio.solvers.ik` even
where one wraps the other.
"""

from __future__ import annotations

__all__: list[str] = []
