# SPDX-License-Identifier: Apache-2.0
"""Texture coordinates in the mesh's own metres, for any backend that shades.

A :class:`~robosimstudio.core.render.finish.Finish` states ``uv_scale`` as
*repeats per metre*, which only means anything if the UVs under it are metres.
Newton's primitive prototypes are not: they arrive with one unit square per
face, so a 25 mm jigsaw mat and a 2 m bench would take the same number of
repeats and the mat's grid would come out the size of the bench's. Both
functions here answer the same question -- where on this surface, in metres, am
I -- and they live in ``core`` because the answer is a property of the geometry
rather than of whichever renderer asked.

They were the path tracer's private helpers until the browser page needed the
same numbers. A second implementation would have been two implementations of
one fact, and the way that shows up is a texture that tiles differently in the
preview than in the render -- which is the exact thing a preview exists to rule
out.
"""

from __future__ import annotations

import numpy as np

__all__ = ["box_projected_uvs", "planar_uvs"]


def box_projected_uvs(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Face-varying UVs in the mesh's own units: each triangle projected onto
    the axis plane its normal faces. Seams land on edges, where they belong.

    Face-varying rather than per-vertex because a box's corner belongs to three
    faces with three different projections; a vertex can only carry one, so a
    caller that needs per-vertex UVs has to unweld first.

    Args:
        points: ``(V, 3)`` vertex positions, in metres.
        tris: ``(F, 3)`` triangle indices.

    Returns:
        ``(F * 3, 2)`` UVs, ordered corner-major to match the triangle list.
    """
    p = np.asarray(points, dtype=np.float64)
    a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    n = np.abs(np.cross(b - a, c - a))
    axis = np.argmax(n, axis=1)
    planes = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    uv = np.zeros((len(tris) * 3, 2), dtype=np.float32)
    for k in range(3):
        corner = p[tris[:, k]]
        for ax, (u, v) in planes.items():
            mask = axis == ax
            uv[k::3][mask] = corner[mask][:, [u, v]]
    return uv


def planar_uvs(points: np.ndarray) -> np.ndarray:
    """UVs in metres along a point cloud's two principal axes.

    A sheet at rest is flat; a sponge is a block, whose top face is what a
    texture would be read on. Either way the first two principal directions are
    the surface, and metres are what a finish's ``uv_scale`` expects.

    Args:
        points: ``(V, 3)`` positions, in metres -- a deformable's rest state.

    Returns:
        ``(V, 2)`` UVs, origin at the cloud's own minimum.
    """
    p = np.asarray(points, dtype=np.float64)
    centred = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    uv = centred @ vt[:2].T
    uv -= uv.min(axis=0)
    return uv.astype(np.float32)
