# SPDX-License-Identifier: Apache-2.0
"""Who acts: the robots, and the entity that builds one.

A robot is the one thing in a scene whose declaration and whose builder are worth
keeping together -- adding one should be a matter of saying which joints are an
arm, not of writing a builder -- so ``EntityEmbodiment`` lives here rather than
under ``domains/articulated``, which registers the domain and nothing else.

The compiler reaches this by dotted path, never by import. It has to: this
imports the entity contract from ``scene``, so an import the other way would
close a cycle.
"""

from __future__ import annotations

from .entity import ArmIndex, EntityEmbodiment
from .registry import (
    EMBODIMENTS,
    GRIPPERS,
    attachable_grippers,
    get_embodiment,
    get_gripper,
    hands_of,
    list_embodiments,
    list_grippers,
    register_embodiment,
    register_gripper,
    resolve_gripper,
)

__all__ = [
    "EMBODIMENTS",
    "GRIPPERS",
    "ArmIndex",
    "EntityEmbodiment",
    "attachable_grippers",
    "get_embodiment",
    "get_gripper",
    "hands_of",
    "list_embodiments",
    "list_grippers",
    "register_embodiment",
    "register_gripper",
    "resolve_gripper",
]
