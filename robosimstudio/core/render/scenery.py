# SPDX-License-Identifier: Apache-2.0
"""Backdrops: geometry that exists for the camera and for nothing else.

A scene's statics are physics -- a table is stood on, a fixture is pushed
against -- and every one of them is also a landmark the framing tests project
and a thing every backend draws. What a *picture* of the bench wants in
addition is none of that: a gridded mat laid over the table top, the oak ledge
and the slatted wall the real cell has behind its arms. Nothing rests on them,
nothing is scored against them, and a policy trained on tiled pixels must not
start seeing them. So they are not specs. They are a named builder a
:class:`~robosimstudio.core.render.spec.RenderSpec` may point at, honoured by
the one backend that photographs a scene for people, and registered here so a
task states a *name* and the geometry lives beside the look that needs it.

The builder receives a :class:`SceneryKit`-shaped object from the backend --
the USD stage, a way to bind a finish, helpers for boxes and meshes -- and the
cell it is dressing, so a ledge can be placed at the edge of whatever bench the
task actually stands in rather than at a number copied from one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from robosimstudio.registry import Registry, same_source

__all__ = ["SCENERY", "SceneryKit", "box_mesh", "get_scenery", "register_scenery"]


class SceneryKit(Protocol):
    """What a scenery builder is handed. Implemented by the RTX backend.

    ``box(name, center, half, *, color, finish, quat=None)`` draws an axis-aligned
    (or rotated) block in world metres; ``mesh(name, points, faces, *, color,
    finish, uvs=None)`` an arbitrary triangle mesh; ``cell`` is the cell the
    scene stands in -- whatever declares its ``center``, ``half`` and
    ``surface_z`` -- or ``None``; ``surface_z`` the work-surface height [m].
    """

    cell: Any
    surface_z: float

    def box(self, name: str, center, half, *, color, finish, quat=None) -> None: ...

    def mesh(self, name: str, points, faces, *, color, finish, uvs=None) -> None: ...


def box_mesh(center, half, quat=None):
    """A rotated box as ``(points, triangles, per-vertex UVs)``, in metres.

    Every backdrop in this repo is mostly boxes -- a floor, four walls, a ledge,
    a slat -- so the one thing a :class:`SceneryKit` cannot be asked to
    reimplement is what a box *is*. Both kits build it here and differ only in
    where they put it: the path tracer writes a USD mesh, the browser page an
    instanced glTF.

    The six faces carry four unique vertices each, which is what makes the UVs
    per-vertex rather than face-varying: a corner of this mesh belongs to
    exactly one face, so the box projection has one answer per vertex and no
    unwelding is needed.
    """
    import numpy as np

    from robosimstudio.core.render.uv import box_projected_uvs

    hx, hy, hz = (float(v) for v in half)
    corners = np.array(
        [[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64
    )
    faces = [
        (0, 1, 3, 2),  # -x
        (4, 6, 7, 5),  # +x
        (0, 4, 5, 1),  # -y
        (2, 3, 7, 6),  # +y
        (0, 2, 6, 4),  # -z
        (1, 5, 7, 3),  # +z
    ]
    points, tris = [], []
    for quad in faces:
        base = len(points)
        points.extend(corners[list(quad)])
        tris.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
    points = np.asarray(points, dtype=np.float64)
    if quat is not None:
        from robosimstudio.core.math import transform_matrix

        rot = np.asarray(transform_matrix((0.0, 0.0, 0.0, *[float(v) for v in quat])), dtype=np.float64)[:3, :3]
        points = points @ rot.T
    points += np.array([float(v) for v in center])
    tris = np.asarray(tris, dtype=np.int32)
    per_vertex = np.zeros((len(points), 2), dtype=np.float32)
    per_vertex[tris.reshape(-1)] = box_projected_uvs(points, tris)
    return points, tris, per_vertex


SCENERY: Registry[Callable[[SceneryKit], None]] = Registry("scenery", same=same_source)
"""``name -> builder``. Content registers; the toolkit only holds the table."""


def register_scenery(name: str, builder: Callable[[SceneryKit], None] | None = None):
    """Register ``builder`` under ``name``; usable as a decorator.

    Re-registering a name with a *different* builder is an error rather than a
    silent replacement -- two suites that both think they own ``"lab"`` would
    otherwise dress each other's scenes depending on import order. The same
    builder arriving twice is not that: a task module is imported once by the
    registry and once more under the environment's own copy of the package, so
    a function of the same name from a module of the same name is the same
    statement and the first one stands.
    """

    def _register(fn: Callable[[SceneryKit], None]) -> Callable[[SceneryKit], None]:
        SCENERY.register(name, fn, origin=f"{fn.__module__}.{fn.__qualname__}")
        return fn

    return _register if builder is None else _register(builder)


def get_scenery(name: str) -> Callable[[SceneryKit], None]:
    return SCENERY.get(name)
