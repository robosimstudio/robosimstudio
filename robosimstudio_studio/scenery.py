# SPDX-License-Identifier: Apache-2.0
"""The room, drawn into the browser page.

:class:`~robosimstudio.core.render.scenery.SceneryKit`'s docstring says
"Implemented by the RTX backend", and that was the whole problem: the concrete
floor, the four plaster walls, the lab's slat wall and glazing -- everything
that makes a render look like a photograph of somewhere -- existed only for the
path tracer, and the page showed its subject floating in a void. This is the
second implementation, so `house_room_closed`, `lab`, `rgbs_lab` and
`tool_sort_shop` draw the same room in both without a line of their own
changing.

It is worth saying why the room matters more than it sounds. ``room.py``'s own
docstring records a measured negative result: two dome HDRIs of quite different
radiance rendered a scene to 86.77 and 86.95, because **the renderer ignores the
dome's texture entirely**. What reaches the picture is geometry. A page with the
right lights and no room is still a page of a thing in a void; a page with the
room is most of the way there.

Every prop arrives with a :class:`Finish` and a palette colour, which is exactly
what :mod:`robosimstudio_studio.finish_gltf` turns into a material -- so the
walls carry their plaster, the mat its grid and the glazing its 30 % opacity,
rather than being flat boxes in the right places.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.core.render.finish import Finish
from robosimstudio.core.render.scenery import box_mesh, get_scenery

from .finish_gltf import glb_for

__all__ = ["ViserKit", "draw_scenery"]


class ViserKit:
    """What a scenery builder is handed when the page is doing the drawing.

    The same two methods and two attributes :class:`_StageKit` offers the path
    tracer. Nodes go under one parent so the whole backdrop can be taken down in
    one call when the scene changes or somebody turns it off.
    """

    def __init__(self, scene: Any, *, cell: Any, surface_z: float, prefix: str = "/studio/scenery") -> None:
        self.scene = scene
        self.cell = cell
        self.surface_z = float(surface_z)
        self.prefix = prefix
        self.handles: dict[str, Any] = {}
        self.skipped = 0
        """Props that could not be encoded. Reported rather than raised: a room
        missing one slat is a room; a page that refuses to open is not."""

    # ------------------------------------------------------------- the kit
    def box(self, name: str, center, half, *, color, finish: Finish, quat=None) -> None:
        points, tris, uvs = box_mesh(center, half, quat)
        self.mesh(name, points, tris, color=color, finish=finish, uvs=uvs)

    def mesh(self, name: str, points, faces, *, color, finish: Finish, uvs=None) -> None:
        blob = glb_for(
            np.asarray(points, dtype=np.float64),
            np.asarray(faces, dtype=np.int64).reshape(-1, 3),
            color,
            finish,
            uvs=None if uvs is None else np.asarray(uvs, dtype=np.float64),
        )
        if blob is None:
            self.skipped += 1
            return
        node = f"{self.prefix}/{name.strip('/').replace('/', '_')}"
        try:
            # Scenery is furniture, not a subject: it casts and receives shadow
            # and is never clickable, which is what stops the bench swallowing a
            # gizmo the way the model's own statics used to.
            self.handles[node] = self.scene.add_glb(node, glb_data=blob, cast_shadow=True, receive_shadow=True)
        except Exception:
            self.skipped += 1

    # ------------------------------------------------------------ teardown
    def remove(self) -> None:
        """Take the backdrop down. Idempotent."""
        for handle in self.handles.values():
            try:
                handle.remove()
            except Exception:
                pass
        self.handles = {}


def draw_scenery(scene: Any, name: str, *, cell: Any, surface_z: float) -> ViserKit | None:
    """Run a registered scenery builder against the page.

    Args:
        scene: ``server.scene``.
        name: a registered builder's name, as ``RenderSpec.scenery`` states it.
        cell: the cell the scene stands in, or ``None``.
        surface_z: the work surface height [m].

    Returns:
        The kit holding the nodes it drew, for a later :meth:`ViserKit.remove`,
        or ``None`` when the name is not registered -- which is a scene with no
        backdrop rather than an error.
    """
    if not name:
        return None
    try:
        builder = get_scenery(name)
    except KeyError:
        return None
    kit = ViserKit(scene, cell=cell, surface_z=surface_z)
    try:
        builder(kit)
    except Exception:
        # A half-drawn room is worse than none: take back what went up.
        kit.remove()
        return None
    return kit
