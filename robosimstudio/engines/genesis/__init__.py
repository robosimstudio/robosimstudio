# SPDX-License-Identifier: Apache-2.0
"""The ``genesis`` backend: Genesis's IPC coupler, co-simulated beside Newton.

Importing this registers the alias and nothing else --- the engine itself loads
the first time a scene actually builds a Genesis island, so a checkout without
the ``[genesis]`` extra composes, lists and runs every shipped task untouched.

Named ``genesis`` inside ``robosimstudio.engines``, which is
``robosimstudio.engines.genesis`` and shadows nothing: the adapter's own
``import genesis`` is absolute and resolves to the wheel.
"""

from __future__ import annotations

from robosimstudio.specs.physics import register_backend

__all__ = ["BACKEND"]

BACKEND = register_backend(
    "genesis",
    cls="robosimstudio.engines.genesis.solver.SolverGenesis",
    supports=("particles",),
    summary="Genesis's IPC coupler (libuipc): cloth and FEM bodies, co-simulated on the host; "
    "Newton islands lead, constraint reactions feed back.",
    host_engine=True,
    # What each builder reads is declared beside it, per kind, in ``bridge.py``.
)

from . import bridge as _bridge  # noqa: E402, F401 -- registers the shell and tet builders
