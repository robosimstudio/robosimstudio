# SPDX-License-Identifier: Apache-2.0
"""GaP: a graph as the policy that drives a task.

**What this package is.** An adapter over the pinned, git-sourced graph runtime
(``gap``, ``gap_core``): the connector that subclasses its ``Connector``, the
``robot.*`` / ``sim.*`` / ``motion.*`` tools a graph calls, the facade those tools
drive an ``Env`` through, and the two policies -- single-world
(:class:`~robosimstudio_policy_agent.graph_solution.SolutionGraph`) and fleet
(:class:`~robosimstudio_policy_agent.fleet.solution.SolutionGraphBatched`) -- that run a
workflow against it. A graph is a folder: a ``workflow.json`` and the
``scripts/`` its script nodes name. A task declares one with ``(:graph LABEL
"DIR")``, and :mod:`robosimstudio_policy_agent.graphs` says how a workflow reaches its
scripts and how a subgraph's ``skill`` reaches a bundle in open-robot-skills.

It stands beside the library as its own top-level package because it is a
*policy*, one of the things that drives an environment, and not an asset or task
surface: the benchmark keeps what a policy is (``robosimstudio_bench.solutions.base``)
and the dotted-path loader, and content reaches this runtime by naming it.

The line that matters is the **privilege boundary**. With ``privileged=True`` (the
default) a graph may also call the ground-truth ``sim.*`` tools -- where an object
is, whether the goal holds -- which is what building and debugging a graph wants.
With ``privileged=False`` it gets only what a real rig would have: camera frames,
proprioception, calibration, and the ``robot.*`` and ``motion.*`` verbs. A frame
sent to a policy in another process is to be held to the same line by
:func:`~robosimstudio_policy_agent.contract.assert_non_privileged`, which a
bridge sender calls before it sends; this package ships none.

Only the boundary-enforcing pieces are imported eagerly here, and the policies
are imported on first use. Everything that runs a graph needs the ``[gap]``
extra -- dependencies, not code: this package ships in every wheel and imports
``gap`` only inside the functions that need it, so ``import robosimstudio_policy_agent``
costs nothing a base install lacks.
"""

from robosimstudio.core.bus import BusTimeout, SimBus, SimSnapshot

from .contract import ALLOWED_KEYS, PRIVILEGED_KEYS, assert_non_privileged
from .refusal import ToolRefusal

_POLICIES = {
    "SolutionGraph": "robosimstudio_policy_agent.graph_solution",
    "SolutionGraphBatched": "robosimstudio_policy_agent.fleet.solution",
    "check_graph": "robosimstudio_policy_agent.graphs",
}
"""Imported on first use: each reaches the runtime or the benchmark."""

__all__ = [
    "ALLOWED_KEYS",
    "PRIVILEGED_KEYS",
    "BusTimeout",
    "SimBus",
    "SimSnapshot",
    "SolutionGraph",
    "SolutionGraphBatched",
    "ToolRefusal",
    "assert_non_privileged",
    "check_graph",
]


def __getattr__(name: str):
    module = _POLICIES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_POLICIES})
