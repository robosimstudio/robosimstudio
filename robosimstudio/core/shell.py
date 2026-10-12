# SPDX-License-Identifier: Apache-2.0
"""Turning a zero-thickness surface into a slab that actually has a thickness.

A cloth mid-surface is a membrane: one layer of particles, triangles between
them, and no geometry at all in the third direction. That is the right model
for a plastic sheet and the wrong one for a bath towel, which is 5 mm of pile
and behaves like it -- it stacks when folded, it squashes under a finger, and
it will not take a crease sharper than its own thickness.

This module offsets a surface into a **closed tetrahedral slab**: the input
mesh pushed to +/- h/2 along its own smoothed normals, with the space between
filled by tetrahedra. Newton's :meth:`~newton.ModelBuilder.add_soft_mesh`
consumes exactly that -- it lumps mass per tet volume, derives the boundary
triangles itself, and hands the interior to VBD's volumetric solve.

Two details are load-bearing and both are easy to get subtly wrong:

* **Neighbouring prisms must agree about their shared diagonal.** Each triangle
  becomes a prism, each prism splits into three tets, and every quad wall is
  shared with the prism next door. Split the two sides on opposite diagonals
  and the mesh is no longer conforming: the tets meet along a crack, VBD
  happily simulates the crack, and the slab quietly delaminates under load.
  :func:`prism_slab` sorts each triangle's corners by global index before
  splitting, which forces every quad's diagonal to run from its lower-indexed
  bottom corner to its higher-indexed top corner -- a rule both neighbours
  derive independently and therefore agree on.
* **Every tet has to come out positively oriented.** ``add_tetrahedron``
  computes the rest pose from ``det[x1-x0, x2-x0, x3-x0]`` and rejects a
  non-positive volume, so a slab built off an inward-facing normal fails at
  build rather than at run time. The winding is fixed per tet here instead of
  being assumed from the input mesh's.

The vertex layout is part of the contract: **bottom face first, then top**, so
particle ``i`` and particle ``i + n`` are the two faces of the same point on
the original surface and their midpoint is that point. That pairing is what
lets a caller recover the mid-surface it started with -- which is the thing a
fold should be scored against, since the two faces of a folded slab swap sides
and neither one alone is where the cloth "is".

The same offsetting exists once more as triangles rather than tetrahedra --
:func:`shell_faces` and :func:`shell_vertices` -- for a renderer that wants a
membrane *drawn* at its fabric's thickness without the simulation carrying one.
A membrane is the right physics for a fling and the wrong picture for a towel:
zero thickness renders as a film whose silhouette is a line. The visual shell
closes the surface with a rim along every boundary loop, shares the slab's
bottom-then-top vertex layout, and is recomputed from live positions each
frame, so the picture has the thickness the material declares while the solver
keeps the membrane it was calibrated on.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "boundary_edges",
    "grid_surface",
    "mid_surface",
    "prism_slab",
    "shell_faces",
    "shell_vertices",
    "surface_normals",
]


def grid_surface(size: tuple[float, float], cells: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """A flat rectangular triangle grid in the z = 0 plane.

    Matches :meth:`~newton.ModelBuilder.add_cloth_grid`'s layout -- vertex
    ``y * (nx + 1) + x``, two triangles per cell -- so a sheet built as a slab
    is indexed the same way as the membrane it replaces, and a particle-set
    selector written against one keeps working against the other.

    Args:
        size: ``(length, width)`` [m].
        cells: Subdivisions ``(nx, ny)``.

    Returns:
        ``(vertices [n, 3], faces [m, 3])``, wound counter-clockwise seen from
        +z so the surface normal points up.
    """
    length, width = float(size[0]), float(size[1])
    nx, ny = int(cells[0]), int(cells[1])
    if nx < 1 or ny < 1:
        raise ValueError(f"grid needs at least one cell per side, got {cells}")

    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, width, ny + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    vertices = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)

    x0, y0 = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    x0, y0 = x0.ravel(), y0.ravel()
    v00 = y0 * (nx + 1) + x0
    v10, v01, v11 = v00 + 1, v00 + (nx + 1), v00 + (nx + 2)
    faces = np.concatenate([np.stack([v00, v10, v11], axis=1), np.stack([v00, v11, v01], axis=1)])
    return vertices, faces.astype(np.int64)


def surface_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, unit length.

    Area weighting rather than a plain face average because the offset has to
    survive an irregular mesh: on a garment converted from a scan, a vertex can
    sit on one large triangle and a fan of slivers, and an unweighted average
    lets the slivers outvote the surface the vertex is actually on.

    A vertex whose normals cancel -- a fold pinched flat, a degenerate fan --
    has no meaningful offset direction and comes back as ``+z`` rather than as
    a division by zero. :func:`prism_slab` checks tet volumes afterwards, which
    is where such a vertex actually gets caught.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    # Un-normalised cross products are already area-weighted (twice the area).
    face_normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    normals = np.zeros_like(v)
    for corner in range(3):
        np.add.at(normals, f[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    degenerate = lengths < 1e-12
    normals[degenerate] = (0.0, 0.0, 1.0)
    lengths[degenerate] = 1.0
    return normals / lengths[:, None]


def prism_slab(
    vertices: np.ndarray,
    faces: np.ndarray,
    thickness: float,
    *,
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Offset a triangle surface into a closed tetrahedral slab.

    Args:
        vertices: The mid-surface ``[n, 3]`` [m].
        faces: Triangles into it, ``[m, 3]``.
        thickness: Slab thickness [m]. The surface ends up at the middle of it,
            so the input geometry is preserved rather than shifted to one side.
        normals: Offset directions ``[n, 3]``. Defaults to
            :func:`surface_normals`. Worth supplying for a surface whose own
            normals are noisy -- a flat sheet's are exactly ``+z`` and saying
            so beats averaging a hundred triangles that all agree.

    Returns:
        ``(vertices [2n, 3], tets [3m, 4])``. Bottom face is ``0 .. n-1``, top
        is ``n .. 2n-1``, and tets are positively oriented for
        :meth:`~newton.ModelBuilder.add_soft_mesh`. Boundary triangles -- the
        two faces and the rim around every hole -- are not returned because
        Newton derives them itself from the tets, which is also the only way
        they are guaranteed to agree with the tets.

    Raises:
        ValueError: If ``thickness`` is not positive, if the surface is not
            two-manifold, or if the offset turned some prism inside out.

            The manifold check is not pedantry. An edge shared by three or more
            triangles becomes a wall shared by three or more prisms, and the
            diagonal rule that keeps neighbours agreeing is only defined for
            two -- so the slab comes out non-conforming, with tets meeting
            along a crack instead of a face. Nothing downstream notices: the
            model finalizes, VBD simulates the crack, and the sheet quietly
            delaminates under load. Mesh decimation is a common way to acquire
            such an edge without meaning to.

            An inverted prism means the surface folds tighter than its own
            half-thickness somewhere, so no slab of that thickness exists
            there -- the fix is a thinner slab or a smoother surface.
    """
    h = float(thickness)
    if not h > 0.0:
        raise ValueError(f"slab thickness must be positive, got {h}")

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    n = len(v)

    edges = np.sort(f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, shared = np.unique(edges, axis=0, return_counts=True)
    if shared.max() > 2:
        raise ValueError(
            f"surface is not two-manifold: {int((shared > 2).sum())} edges are shared by more than two "
            f"triangles (worst is {int(shared.max())}). A slab built on one would not conform."
        )
    offset = (surface_normals(v, f) if normals is None else np.asarray(normals, dtype=np.float64)) * (0.5 * h)
    slab = np.concatenate([v - offset, v + offset])

    # Sorting each triangle's corners is what makes neighbouring prisms agree
    # about the diagonal of the wall they share -- see the module docstring.
    base = np.sort(f, axis=1)
    a, b, c = base[:, 0], base[:, 1], base[:, 2]
    top_a, top_b, top_c = a + n, b + n, c + n
    tets = np.concatenate(
        [
            np.stack([a, b, c, top_c], axis=1),
            np.stack([a, b, top_c, top_b], axis=1),
            np.stack([a, top_b, top_c, top_a], axis=1),
        ]
    )

    # Orientation is a property of the surface's winding, which an arbitrary
    # input mesh is under no obligation to have got right, so it is measured
    # rather than assumed. Swapping two corners flips the sign and changes
    # nothing else about the element.
    edges = slab[tets[:, 1:]] - slab[tets[:, [0]]]
    volumes = np.linalg.det(edges) / 6.0
    flip = volumes < 0.0
    tets[flip] = tets[flip][:, [0, 1, 3, 2]]

    degenerate = np.abs(volumes) <= 1e-18
    if degenerate.any():
        raise ValueError(
            f"{int(degenerate.sum())} of {len(tets)} tets came out with no volume: the offset by "
            f"{h * 1000:.2f} mm collapsed the surface somewhere. Try a thinner slab or smoother normals."
        )
    return slab, tets


def boundary_edges(faces: np.ndarray) -> np.ndarray:
    """The directed boundary edges of a triangle surface, wound as their faces wind.

    An edge is boundary when exactly one triangle uses it. The *direction* is
    kept as that triangle walks it -- ``a -> b`` in face order -- because the
    direction encodes which side the surface is on: counter-clockwise winding
    keeps the interior on the edge's left, and that is what lets a rim built
    along the boundary face outward without ever consulting the geometry.

    Raises:
        ValueError: If any edge is shared by more than two triangles, the same
            rule as :func:`prism_slab` and for the same reason -- along such an
            edge "the boundary" is not a defined thing.
    """
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    directed = f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    undirected = np.sort(directed, axis=1)
    _, inverse, shared = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    if shared.max(initial=0) > 2:
        raise ValueError(
            f"surface is not two-manifold: {int((shared > 2).sum())} edges are shared by more than "
            f"two triangles (worst is {int(shared.max())})."
        )
    return directed[shared[inverse] == 1]


def shell_faces(faces: np.ndarray, count: int) -> np.ndarray:
    """The closed triangle surface of a shell over ``count`` mid-surface vertices.

    The *visual* counterpart of :func:`prism_slab`: the same two offset copies
    of the surface -- bottom ``0 .. n-1``, top ``n .. 2n-1``, the same layout,
    so :func:`mid_surface` recovers the sheet from either construction -- but
    as boundary triangles rather than tetrahedra, because a renderer wants a
    silhouette and not a stiffness. Topology only, computed once; the vertices
    move every frame and come from :func:`shell_vertices`.

    Winding puts every normal outward: the top copy keeps the input's winding,
    the bottom copy reverses it, and the rim walks each boundary edge in face
    order -- which, for a counter-clockwise surface, is exactly the order that
    faces the wall away from the interior.

    Args:
        faces: Mid-surface triangles ``[m, 3]``, indices in ``0 .. count-1``.
        count: Mid-surface vertex count ``n``.

    Returns:
        Triangles ``[2m + 2k, 3]`` into a ``2n``-vertex shell, where ``k`` is
        the boundary edge count. Watertight whenever the input is manifold.

    Raises:
        ValueError: If the surface is not two-manifold; see
            :func:`boundary_edges`.
    """
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    n = int(count)
    rim_ab = boundary_edges(f)
    a, b = rim_ab[:, 0], rim_ab[:, 1]
    return np.concatenate(
        [
            f[:, [0, 2, 1]],  # bottom, outward is -normal
            f + n,  # top, outward is +normal
            np.stack([a, b, b + n], axis=1),  # rim, two triangles per wall
            np.stack([a, b + n, a + n], axis=1),
        ]
    )


def shell_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    thickness: float,
    *,
    normals: np.ndarray | None = None,
) -> np.ndarray:
    """Both faces of a shell, from the live mid-surface. Every frame.

    The per-frame half of :func:`shell_faces`: offset the current positions to
    either side of the surface's *current* normals, so the drawn thickness
    follows the sheet through whatever the solver did to it. Unlike
    :func:`prism_slab` nothing is validated here -- a crumple that folds the
    sheet tighter than its half-thickness makes the shell self-intersect for a
    frame, which drawn double-sided is invisible, and a per-frame path that can
    raise mid-episode is worse than the artefact it would be protecting.

    Args:
        vertices: Live mid-surface positions ``[n, 3]`` [m].
        faces: Mid-surface triangles, for the normals.
        thickness: Shell thickness [m].
        normals: Offset directions ``[n, 3]``, if the caller already has them.

    Returns:
        ``[2n, 3]`` -- bottom copy then top copy, :func:`shell_faces`'s layout.
    """
    v = np.asarray(vertices, dtype=np.float64)
    offset = (surface_normals(v, faces) if normals is None else np.asarray(normals, dtype=np.float64)) * (
        0.5 * float(thickness)
    )
    return np.concatenate([v - offset, v + offset])


def mid_surface(particle_q: np.ndarray, count: int) -> np.ndarray:
    """The original surface, recovered from a slab's two faces.

    Where a slab "is". A fold turns the sheet over, so the face that was on top
    is underneath afterwards and is half a thickness off in the *opposite*
    direction from before -- scoring a fold against either face alone builds
    that flip into the error, which for a bath towel is 5 mm of pure artefact.
    The midpoint has no such bias.

    Args:
        particle_q: Positions ``[2n, 3]``, or the whole scene's array sliced to
            this slab's particles.
        count: ``n``, the mid-surface's vertex count.

    Returns:
        ``[n, 3]``.
    """
    q = np.asarray(particle_q, dtype=np.float64)
    if len(q) != 2 * int(count):
        raise ValueError(f"a slab of {count} surface vertices has 2n = {2 * int(count)} particles, got {len(q)}")
    return 0.5 * (q[: int(count)] + q[int(count) :])
