# SPDX-License-Identifier: Apache-2.0
"""Hand poses over and at the cube, with the jaws turned to close across its faces."""

import math
from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    above: dict
    grasp: dict


def run(ctx: NodeContext, cube: dict, clearance: float = 0.12) -> Output:
    p, q = cube["position"], cube["rotation"]
    yaw = math.degrees(math.atan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]), 1 - 2 * (q["y"] ** 2 + q["z"] ** 2)))
    heading = (yaw + 45.0) % 90.0 - 45.0  # a cube's faces repeat every 90 degrees
    rotation = ctx.tool("robot.grasp_frame", close_heading_deg=heading)["rotation"]
    at = lambda z: {"position": {"x": p["x"], "y": p["y"], "z": z}, "rotation": rotation}  # noqa: E731
    return {"above": at(p["z"] + clearance), "grasp": at(p["z"])}
