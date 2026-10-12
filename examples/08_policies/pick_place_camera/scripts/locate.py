# SPDX-License-Identifier: Apache-2.0
"""The cube and the pad, from the overhead depth image alone.

Every pixel is back-projected to a world point. The bench is the most common
height; points 30 to 60 mm above it are the cube's top face, and points 2 to
8 mm above it are the pad. The cube's yaw is the angle whose bounding box
around its top face is smallest. ``want`` names the one the node needs: the
other may be out of sight, as the cube is once the hand holds it.
"""

import math
from typing import TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    cube: dict
    target: dict


def _world_points(cam: dict) -> np.ndarray:
    depth = np.asarray(cam["depth"], dtype=np.float64).reshape(np.shape(cam["depth"])[:2])
    K = np.asarray(cam["intrinsics"], dtype=np.float64).reshape(3, 3)
    v, u = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]]
    ok = np.isfinite(depth) & (depth > 0.0)
    d = depth[ok]
    local = np.stack([(u[ok] - K[0, 2]) / K[0, 0] * d, (v[ok] - K[1, 2]) / K[1, 1] * d, d], axis=1)
    w, x, y, z = (float(cam["pose"]["rotation"][k]) for k in "wxyz")
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip
    t = cam["pose"]["position"]
    return local @ R.T + np.array([float(t["x"]), float(t["y"]), float(t["z"])])


def _yaw(xy: np.ndarray) -> float:
    def area(deg: int) -> float:
        a = math.radians(deg)
        return float(np.ptp(xy @ [math.cos(a), math.sin(a)]) * np.ptp(xy @ [-math.sin(a), math.cos(a)]))

    return math.radians(min(range(90), key=area))


def run(ctx: NodeContext, cameras: list, want: str = "cube", camera: str = "overhead") -> Output:
    points = _world_points(next(c for c in cameras if c["name"] == camera))
    bench = float(np.median(points[:, 2]))
    found = {"cube": None, "target": None}
    top = points[(points[:, 2] > bench + 0.03) & (points[:, 2] < bench + 0.06)]
    if len(top) >= 20:
        cx, cy = top[:, :2].mean(axis=0)
        z = (bench + float(np.percentile(top[:, 2], 90))) / 2
        yaw = _yaw(top[:, :2])
        rotation = {"w": math.cos(yaw / 2), "x": 0.0, "y": 0.0, "z": math.sin(yaw / 2)}
        found["cube"] = {"position": {"x": float(cx), "y": float(cy), "z": z}, "rotation": rotation}
    pad = points[(points[:, 2] > bench + 0.002) & (points[:, 2] < bench + 0.008)]
    if len(pad) >= 20:
        px, py = pad[:, :2].mean(axis=0)
        # Where a held 50 mm cube's centre goes to rest on the pad.
        found["target"] = {"center": {"x": float(px), "y": float(py), "z": float(np.percentile(pad[:, 2], 90)) + 0.025}}
    if found[want] is None:
        raise RuntimeError(f"the camera does not see the {want}")
    return found
