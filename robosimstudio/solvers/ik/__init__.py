# SPDX-License-Identifier: Apache-2.0
"""Inverse kinematics, one module per backend.

A backend answers one question -- what joint angles put the tool here -- and
they disagree about how, which is why they are separate modules rather than
flags: a damped least-squares solve on device, a full-orientation solve on the
host, and a JAX solver that measures the base rather than assuming it.
"""

from __future__ import annotations

__all__: list[str] = []
