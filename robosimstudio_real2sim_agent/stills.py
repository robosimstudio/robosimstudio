# SPDX-License-Identifier: Apache-2.0
"""One still of the live scene, however it is taken, and the reference beside it.

Two ways to take a picture of what the page is showing and one shape coming
out. A camera the scene declares, or a pose given on the command line, goes
through :meth:`Sim.render`: the path tracer, which draws the room, the mat and
the textures, or the collider-only tracer where it is not installed. What the
connected browser is looking at, from its own viewpoint, is viser's own render
and needs no renderer of ours. Either way the result is an RGB
``uint8`` array, and :func:`side_by_side` is the picture an agent actually
looks at: the render on the left, the photograph on the right, scaled to the
render's height.

Nothing here is named ``snapshot``: ``robosimstudio/scene/snapshot.py`` is a
JSON account of a scene's penetrations and materials, and the two should not
be confused. The ``ctl`` command keeps the word because a picture is what a
person asking for a snapshot means.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SNAPSHOT_SIZE",
    "browser_still",
    "camera_for",
    "load_reference",
    "parse_size",
    "parse_vec",
    "render_still",
    "save_png",
    "side_by_side",
    "to_uint8",
]

SNAPSHOT_SIZE = (960, 540)
"""Pixels of a still taken from a pose or the default view; a declared camera keeps its own."""


def parse_vec(text: str) -> tuple[float, float, float]:
    parts = [float(v) for v in str(text).replace(" ", "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected x,y,z, got {text!r}")
    return parts[0], parts[1], parts[2]


def parse_size(text: str, default: tuple[int, int]) -> tuple[int, int]:
    if not text:
        return default
    w, h = (int(v) for v in str(text).lower().split("x"))
    return w, h


def camera_for(sim: Any, scene: Any, *, name: str = "", size: str = "", eye: str = "", look_at: str = ""):
    """The camera a still is taken through: named, posed, the first declared, or a default three-quarter view."""
    if bool(eye) != bool(look_at):
        raise ValueError("--eye and --look-at go together; got only one")
    if name and eye:
        raise ValueError("--camera or --eye/--look-at, not both")
    import robosimstudio as rss

    cameras = {c.name: c for c in sim.cameras}
    if name and name in cameras:
        cam = cameras[name]
    elif name:
        raise KeyError(f"no camera named {name!r}; the scene declares {sorted(cameras) or 'none'}")
    elif eye and look_at:
        cam = rss.CameraSpec(
            name="view",
            width=SNAPSHOT_SIZE[0],
            height=SNAPSHOT_SIZE[1],
            fov_y=45.0,
            eye=parse_vec(eye),
            look_at=parse_vec(look_at),
            up=(0.0, 0.0, 1.0),
            near=0.01,
            far=8.0,
        )
    elif cameras:
        cam = next(iter(cameras.values()))
    else:
        # A three-quarter view from behind the arms' side of the bench.
        cell = rss.get_cell(scene.cell)
        cx, cy, z = float(cell.center[0]), float(cell.center[1]), float(cell.surface_z)
        cam = rss.CameraSpec(
            name="default",
            width=SNAPSHOT_SIZE[0],
            height=SNAPSHOT_SIZE[1],
            fov_y=45.0,
            eye=(cx - 1.1, cy - 0.9, z + 0.85),
            look_at=(cx, cy, z + 0.1),
            up=(0.0, 0.0, 1.0),
            near=0.01,
            far=8.0,
        )
    if size:
        cam = cam.scaled(*parse_size(size, (cam.width, cam.height)))
    return cam


def to_uint8(rgb: Any) -> np.ndarray:
    """An RGB array as ``uint8``: floats in [0, 1] are scaled, anything else clipped."""
    rgb = np.ascontiguousarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * (255.0 if rgb.size and rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return rgb


def render_still(sim: Any, cam: Any) -> np.ndarray:
    """The scene's current state through ``cam``, as RGB ``uint8``. Syncs the sim first."""
    sim.sync()
    return to_uint8(sim.render(cam)["rgb"][..., :3])


BROWSER_TIMEOUT = 10.0
"""Seconds to wait for a connected browser to send its frame back."""


def browser_still(server: Any, size: str = "") -> np.ndarray | None:
    """What the connected browser shows from its own viewpoint, or None when nobody is connected.

    The render is asked of a client, not the server (viser's server has no
    ``get_render``), as JPEG so it comes back opaque RGB rather than RGBA over
    a transparent background.
    """
    clients = server.get_clients()
    if not clients:
        return None
    client = next(iter(clients.values()))
    w, h = parse_size(size, (1280, 720))
    try:
        rgb = client.get_render(h, w, transport_format="jpeg", timeout=BROWSER_TIMEOUT)
    except TimeoutError as exc:
        raise TimeoutError(f"the browser sent no frame within {BROWSER_TIMEOUT:.0f} s") from exc
    return to_uint8(np.asarray(rgb)[..., :3])


def side_by_side(sim_rgb: np.ndarray, reference_path: str) -> np.ndarray:
    """The sim frame with the reference picture beside it, scaled to the frame's height."""
    from PIL import Image

    with Image.open(reference_path) as opened:
        ref = opened.convert("RGB")
    h = sim_rgb.shape[0]
    ref = ref.resize((int(round(ref.width * h / ref.height)), h))
    return np.concatenate([sim_rgb, np.asarray(ref)], axis=1)


def save_png(rgb: np.ndarray, path: Path | str) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path


def load_reference(path: str, width: int = 420) -> tuple[np.ndarray, str]:
    """A picture at sidebar width, and the name to label it with."""
    from PIL import Image

    with Image.open(path) as opened:
        image = opened.convert("RGB")
    image = image.resize((width, max(1, int(round(image.height * width / image.width)))))
    return np.asarray(image, dtype=np.uint8), Path(path).name
