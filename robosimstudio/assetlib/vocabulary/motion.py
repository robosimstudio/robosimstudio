# SPDX-License-Identifier: Apache-2.0
"""Two questions every substrate answers: where is it, and is it moving.

The atoms stay here, in ``assetlib``, because they are **vocabulary** -- a goal
may ask the height of a sheet, a hammer or a cable, and the description of what
"height" means cannot belong to any one of them. The *measurements* do not stay:
each domain registers these two for its own substrate, beside the kind it reads.

The two helpers below are what most domains register: the mean is the honest
reading of "where is this" for something with no single position. A kind whose
answer differs writes its own and does not import these.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomCalibration, AtomSpec, register_atom

__all__ = ["body_height", "linear_speed"]


register_atom(
    AtomSpec(
        name="speed",
        summary="How fast a body is moving [m/s], linear only.",
        args=(ArgSpec(name="object", reads="velocity"),),
        directions=("check", "observe"),
        returns=("speed",),
        unit="m/s",
        cost="cheap",
        provenance=(
            "The at-rest half of isaaclab_arena.tasks.predicates.spatial."
            "object_on_destination. Angular velocity is deliberately excluded: a can that has "
            "landed and is still spinning about its own axis has landed."
        ),
        calibration=AtomCalibration(
            positives="Settled grocery items read 0.0001-0.005 m/s over 300 steps.",
            negatives="An item in transport under the arm reads two orders of magnitude more.",
            gate=None,
            bench="isaaclab_arena, grocery_packing/five_object",
            notes="Arena's own threshold is 0.1 m/s for both of its task shapes.",
        ),
    )
)

register_atom(
    AtomSpec(
        name="height",
        summary="How high a body sits [m], in world coordinates.",
        args=(ArgSpec(name="object", reads="position"),),
        directions=("check", "observe"),
        returns=("height",),
        unit="m",
        cost="cheap",
        provenance=(
            "Not borrowed. World z of a body's origin -- the simplest possible measure, minted "
            "because the alternative was every task that wanted 'did it fall off the table' "
            "writing its own. Arena expresses that as object_min_z on the table, a number the "
            "task states rather than the measure."
        ),
    )
)


def body_height(position) -> dict[str, Any]:
    """The z of a point, or the mean z of many.

    The mean for a deformable, which is the only reading that means anything for
    something with no single position -- and the case it is asked about, a
    garment that slid off the table, moves all of it together.
    """
    array = np.atleast_2d(np.asarray(position, dtype=np.float64))
    return {"height": float(array[:, 2].mean())}


def linear_speed(velocity) -> dict[str, Any]:
    """Linear only -- see the atom's provenance for why the spin is dropped.

    A deformable's velocity reader hands back many rows and the norm is taken
    over their mean, which is the honest reading of "is this thing moving" for
    something that has no single velocity.
    """
    array = np.atleast_2d(np.asarray(velocity, dtype=np.float64))
    return {"speed": float(np.linalg.norm(array[:, :3].mean(axis=0)))}
