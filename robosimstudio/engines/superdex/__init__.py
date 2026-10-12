# SPDX-License-Identifier: Apache-2.0
"""The ``superdex`` backend: Meta's SuperDex FEM engine, co-simulated on the host.

Importing this registers the alias and nothing else -- the engine itself loads
the first time a scene actually builds a SuperDex island, so a checkout without
the ``[superdex]`` extra composes, lists and runs every shipped task untouched.
"""

from __future__ import annotations

from robosimstudio.specs.physics import register_backend

__all__ = ["BACKEND"]

BACKEND = register_backend(
    "superdex",
    cls="robosimstudio.engines.superdex.solver.SolverSuperdex",
    supports=("particles",),
    summary="Meta's SuperDex FEM engine, co-simulated on the host; Newton islands lead, wrenches feed back.",
    host_engine=True,
    # What each builder reads is declared beside it, per kind, in ``bridge.py``
    # (``register_actor_builder(..., consumes=...)``); nothing engine-wide.
)

from . import bridge as _bridge  # noqa: E402, F401 -- registers the shell and tet builders
