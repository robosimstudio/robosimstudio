# SPDX-License-Identifier: Apache-2.0
"""A rigid body against a region, and the two questions every substrate answers.

The implementations, beside the kind that builds one. The *atoms* -- what
``containment``, ``height`` and ``speed`` mean -- stay in ``assetlib``, because a
goal may ask any of them of a sheet, a hammer or a cable and the description
belongs to none of those.

A body's centre against the box is the easy case of the fraction: a point is
inside or it is not, so the share is 0 or 1. A sheet answers the same atom with a
real fraction, which is the whole reason dispatch is per substrate.
"""

from __future__ import annotations

import numpy as np

from robosimstudio.assetlib.semantics.handlers import handler
from robosimstudio.assetlib.vocabulary.motion import body_height, linear_speed
from robosimstudio.assetlib.vocabulary.regions import containment, region_bounds

__all__: list[str] = []


@handler(
    "containment",
    kinds=("", "region"),
    substrates=("rigid", ""),
    summary="A body's centre against the box -- the easy case of the fraction, returning 0 or 1.",
)
def _rigid_in_region(position: np.ndarray, region: dict, *, origin=(0.0, 0.0, 0.0)) -> dict:
    lo, hi = region_bounds(region, origin)
    return {"fraction": containment(np.atleast_2d(position), lo, hi)}


handler("height", kinds=("",), substrates=("rigid",), summary="World z of a rigid body.")(body_height)
handler("speed", kinds=("",), substrates=("rigid",), summary="Linear speed of a rigid body.")(linear_speed)


# The third ``covers`` pairing: one body's footprint over another's. The two a
# sheet answers are in ``domains/cloth/measures.py``.
from robosimstudio.assetlib.vocabulary.coverage import Covers, _covers  # noqa: E402

Covers.implement(
    kinds=("", ""),
    substrates=("rigid", "rigid"),
    summary="A rigid body's footprint over a rigid one's, from above.",
)(_covers)
