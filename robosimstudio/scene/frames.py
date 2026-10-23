# SPDX-License-Identifier: Apache-2.0
"""Composing the scene's work-surface frame, in one place.

The three axes -- scene, embodiment, task -- only compose because they agree on
a coordinate system, and there is exactly one: the *work-surface frame*, whose
origin is the scene's :attr:`~robosimstudio.specs.CellSpec.center` in
x and y with z at its ``surface_z``. A task states its fixtures and object poses
there; an embodiment states its mount there.

This module is the only thing that turns those statements into world transforms.
That matters more than it looks: the alternative is each caller writing
``z + surface_z``, which is right until one of them writes it against the
*ground* instead and produces a scene that is 22 mm wrong and looks fine.
"""

from __future__ import annotations

import numpy as np

from robosimstudio.core.math import transform_matrix, xform_from_matrix
from robosimstudio.specs import IDENTITY, CellSpec, Xform

__all__ = ["compose", "from_surface", "in_surface", "point_from_surface", "surface_frame"]

Cell = CellSpec | str | None
"""A cell as these helpers take it: a spec, a registered name, or ``None`` for the world frame."""


def _cell(scene: Cell) -> CellSpec | None:
    """Resolve a cell name, treating ``None`` and an empty name as no cell (the world frame)."""
    if scene is None or scene == "":
        return None
    if isinstance(scene, str):
        from robosimstudio.scene.registry import get_cell

        return get_cell(scene)
    return scene


def surface_frame(scene: Cell = None) -> Xform:
    """The scene's work-surface frame as a world transform.

    ``None`` -- a task compiled without a scene -- is the world frame itself, so
    a coordinate stated "in the surface frame" of no scene is simply a world
    coordinate. That is what lets the frame be introduced without rewriting the
    suites that have not moved onto it yet. A registered cell name is looked up
    the same way :func:`~robosimstudio.scene.registry.get_cell` does.
    """
    spec = _cell(scene)
    return IDENTITY if spec is None else spec.surface_frame


def compose(parent: Xform, child: Xform) -> Xform:
    """``parent ∘ child``: the child transform expressed in the parent's parent."""
    return tuple(
        float(v) for v in xform_from_matrix(transform_matrix(np.asarray(parent)) @ transform_matrix(np.asarray(child)))
    )


def in_surface(scene: Cell, xform: Xform) -> Xform:
    """A surface-frame transform, as world coordinates."""
    return compose(surface_frame(scene), xform)


def from_surface(scene: Cell, xform: Xform) -> Xform:
    """A world transform, restated in the scene's work-surface frame.

    The inverse of :func:`in_surface`, and the one a *suite* reaches for. A port
    suite's geometry is quoted from a reference sim in that sim's world frame,
    and those numbers are the claim being reproduced -- so they stay world, and
    this is where they are converted, once, at the point they become a spec.
    Doing it here rather than by hand keeps the reference numbers legible next
    to the offsets they turn into.
    """
    frame = np.asarray(surface_frame(scene), dtype=np.float64)
    inverse = np.linalg.inv(transform_matrix(frame))
    return tuple(float(v) for v in xform_from_matrix(inverse @ transform_matrix(np.asarray(xform))))


def point_from_surface(scene: Cell, point) -> tuple[float, float, float]:
    """A world position, restated in the surface frame. Rotation-free convenience."""
    x, y, z = (float(v) for v in tuple(point)[:3])
    return from_surface(scene, (x, y, z, 0.0, 0.0, 0.0, 1.0))[:3]
