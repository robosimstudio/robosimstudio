# SPDX-License-Identifier: Apache-2.0
"""Action spaces, and the drivers underneath them.

An action space is the contract between a policy and an arm: how many numbers,
in what order, meaning what. Four of them ship, they differ only in what the
seven numbers *mean*, and all four know about DOF indices, an IK solver and a
gripper -- never about a task.
"""

from __future__ import annotations

__all__: list[str] = []
