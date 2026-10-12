# SPDX-License-Identifier: Apache-2.0
"""Four hand poses from where the cube and the square are: above it, at it, over the square, into it."""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    above: dict
    grasp: dict
    over: dict
    drop: dict


def _at(x: float, y: float, z: float, rotation: dict) -> dict:
    return {"position": {"x": x, "y": y, "z": z}, "rotation": rotation}


def run(ctx: NodeContext, cube: dict, square: dict, rotation: dict, clearance: float = 0.12) -> Output:
    del ctx
    c, s = cube["position"], square["center"]
    return {
        "above": _at(c["x"], c["y"], c["z"] + clearance, rotation),
        "grasp": _at(c["x"], c["y"], c["z"], rotation),
        "over": _at(s["x"], s["y"], c["z"] + clearance, rotation),
        "drop": _at(s["x"], s["y"], c["z"] + 0.01, rotation),
    }
