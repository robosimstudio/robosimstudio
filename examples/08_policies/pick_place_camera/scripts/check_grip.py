# SPDX-License-Identifier: Apache-2.0
"""Raise unless the jaws stopped on something: a hand shut on air reads fully closed."""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    open_fraction: float


def run(ctx: NodeContext, least: float = 0.1) -> Output:
    reading = float(ctx.tool("robot.get_gripper")["position"])  # 1 is open, 0 is shut
    if reading < least:
        raise RuntimeError(f"the jaws closed on nothing ({reading:.2f} open)")
    return {"open_fraction": reading}
