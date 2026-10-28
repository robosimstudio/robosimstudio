# SPDX-License-Identifier: Apache-2.0
"""How far a joint is open: the one question a drawer, a door and a lid share.

The atom is vocabulary; the ``articulated`` domain registers its measurement
beside the ``joint`` feature kind it reads.
"""

from __future__ import annotations

from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomSpec, register_atom

__all__: list[str] = []


register_atom(
    AtomSpec(
        name="opening",
        summary="How far a joint is along its travel: 0 at its lower limit, 1 at its upper.",
        args=(ArgSpec(name="joint", kinds=("joint",), substrates=("articulated",), reads="feature"),),
        directions=("check", "observe"),
        returns=("fraction", "coordinate"),
        unit="",
        cost="cheap",
        provenance=(
            "Not borrowed. The open_fraction of a drawer goal, minted with the articulated-object "
            "entity: a fraction of the joint's own limits, so 'open' is one threshold for a prismatic "
            "drawer [m] and a revolute door [rad]. The raw coordinate rides along."
        ),
    )
)
