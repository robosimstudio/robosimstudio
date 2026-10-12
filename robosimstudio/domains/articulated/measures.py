# SPDX-License-Identifier: Apache-2.0
"""What may be measured of a jointed body: a robot, a cabinet, a drawer.

A domain that answers three questions should be the module that says so, or
"which measures does this kind afford" is a question you answer by grepping
somebody else's loop.
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
    substrates=("articulated", ""),
    summary="Same test for a jointed body's own frame.",
)
def _articulated_in_region(position: np.ndarray, region: dict, *, origin=(0.0, 0.0, 0.0)) -> dict:
    lo, hi = region_bounds(region, origin)
    return {"fraction": containment(np.atleast_2d(position), lo, hi)}


handler("height", kinds=("",), substrates=("articulated",), summary="World z of a jointed body.")(body_height)
handler("speed", kinds=("",), substrates=("articulated",), summary="Linear speed of a jointed body.")(linear_speed)
