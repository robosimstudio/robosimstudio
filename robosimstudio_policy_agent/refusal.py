# SPDX-License-Identifier: Apache-2.0
"""The exception a tool raises when the catalog cannot answer.

One class, because the *type name* is the contract. GaP's executor records a
failed node as ``Node '<id>' failed: <TypeName>: <message>`` (see gap's
``runtime/executor.py::_typed_error_text``), and downstream consumers — most
concretely a self-learning loop's pre-flight — separate "this policy is broken"
from "this policy tried and failed" by that name. A bare ``ValueError`` cannot
carry the distinction: a policy's own deliberate check (``raise ValueError("the
graspable segment does not fit the gripper")``) and a tool's "object 'bin1' has
no bounding box" arrive as the same type, and the 2026-08-23 syringe run spent
two of its three scored iterations on exactly that ambiguity — twelve episodes
each on a policy that asked a static for a bounding box and never moved the arm.

What belongs here is a **refusal of the question**: the scene has no such
entity, the entity has no such property, the sensor has produced nothing to
read. What does not belong here is a malformed *argument* (a heading in the
wrong unit, two exclusive parameters at once) — those stay ``ValueError``,
because they are about the call, not the catalog.

Subclasses ``ValueError`` so every existing ``except ValueError`` handler and
test keeps working; the only observable change is the name in the trace.
"""

from __future__ import annotations

__all__ = ["ToolRefusal"]


class ToolRefusal(ValueError):
    """A privileged/proprioceptive tool declining a question the catalog cannot answer."""
