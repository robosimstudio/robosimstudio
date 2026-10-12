# SPDX-License-Identifier: Apache-2.0
"""Hand poses over and into the square, keeping the hand's current rotation."""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    over: dict
    drop: dict


def run(ctx: NodeContext, square: dict, hand: dict) -> Output:
    s, h = square["center"], hand["position"]
    at = lambda z: {"position": {"x": s["x"], "y": s["y"], "z": z}, "rotation": hand["rotation"]}  # noqa: E731
    return {"over": at(h["z"]), "drop": at(s["z"] + 0.01)}
