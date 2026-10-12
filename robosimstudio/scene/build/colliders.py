# SPDX-License-Identifier: Apache-2.0
"""Collision geometry that stands in for a mesh, and what the stand-in has to carry.

A substitute collider is a substitute for the *material* as well as the shape
(:func:`copy_shape_material`); a mechanism that gets its geometry back must not
collide with itself (:func:`filter_self_collisions`); a tapered object wants a
stack of boxes rather than one bound (:func:`collider_vertices`,
:func:`slab_decomposition`); and a shape collided hydroelastically must carry a
signed-distance field (:func:`build_hydroelastic_sdf`). Each is a measurement
first and a function second -- the numbers are in the docstrings.
"""

from __future__ import annotations

from typing import Any

import newton

__all__ = [
    "SDF_MAX_RESOLUTION",
    "SDF_NARROW_BAND_RANGE",
    "build_hydroelastic_sdf",
    "collider_vertices",
    "copy_shape_material",
    "filter_self_collisions",
    "slab_decomposition",
]

_MATERIAL_FIELDS = (
    "shape_material_ke",
    "shape_material_kd",
    "shape_material_kf",
    "shape_material_ka",
    "shape_material_kh",
    "shape_material_mu",
    "shape_material_mu_rolling",
    "shape_material_mu_torsional",
    "shape_material_restitution",
    "shape_thickness",
    # Not materials in Newton's sense -- both are narrow-phase margins -- and
    # they are here because a stated one was being dropped. A collider
    # approximation disables the imported mesh shape and adds a box, and
    # `ObjectSpec.contact_gap` is applied to the *mesh*: so an object that
    # states a gap and also states a collider had the gap set on a shape that
    # had already stopped colliding, silently. Found while measuring astro's
    # wire spool. A substitute collider stands in for the margin as much as for
    # the friction.
    "shape_gap",
    "shape_margin",
)


def copy_shape_material(builder: newton.ModelBuilder, source: int, target: int) -> int:
    """Give ``target`` the contact material ``source`` carries.

    Every ``add_shape_*`` call takes the builder's *default* material unless it is
    handed a config, so a shape that stands in for an imported one silently loses
    whatever the asset authored. That is not a cosmetic loss: the grocery meshes
    import at ``mu = 2.0`` and the builder's default is 1.0, so replacing a mesh
    collider with a box halved the friction at every contact it makes -- measured
    as a grasped can sliding 27 mm in the fingers during a lift and dropping out
    on the traverse. A substitute collider has to be a substitute for the
    *material* as well as the shape.

    Args:
        builder: The builder both shapes live on.
        source: Shape to copy from.
        target: Shape to copy onto.

    Returns:
        How many material fields were copied, so a caller can assert it did
        something if the builder's field set ever changes under it.
    """
    copied = 0
    for field in _MATERIAL_FIELDS:
        values = getattr(builder, field, None)
        if values is None or len(values) <= max(int(source), int(target)):
            continue
        values[int(target)] = values[int(source)]
        copied += 1
    return copied


def _np():
    import numpy

    return numpy


def filter_self_collisions(builder: newton.ModelBuilder, shapes) -> int:
    """Stop a mechanism's own shapes from colliding with each other.

    **A ModelBuilder starts with no collision filtering at all** -- measured on the
    grocery stage, ``shape_collision_filter_pairs`` was empty for the whole model --
    and for most imports that is harmless, because the parts that would touch have no
    colliders. It stops being harmless the moment a linkage gets its geometry back.

    The Robotiq 2F-85 is the case this exists for. Recovering the eleven collision
    prims the export hides behind an ``Xform`` (see :func:`expose_declared_colliders`)
    takes the hand from three fingertip pads to thirteen shapes -- and with all 78
    pairs of them live, the linkage collides with *itself*. The inner knuckle meets
    the housing, the outer knuckle meets the outer finger, and the drive winds up
    against its own geometry. Measured on a closing-force bench since retired:
    contact forces went from 2-5 N to **51-675 N** and the hand stalled at 9-16% of its travel
    against objects it had been closing fully on. Six hundred newtons from a 100-gain
    servo on a 0.3 kg bottle is not a grasp getting harder, it is a mechanism fighting
    itself.

    Filtering *every* pair rather than only the adjacent ones is deliberate: a gripper's
    fingers are held apart by its joints and its mimic constraints, not by its
    collision geometry, and the pads have to be able to meet when the hand closes on
    nothing. This is what a hand model is expected to do -- the same thing MJCF's
    ``contype``/``conaffinity`` and PhysX's filter groups are used for on the same
    part.

    Args:
        builder: The builder mid-import.
        shapes: Shape indices belonging to one mechanism.

    Returns:
        Pairs filtered.
    """
    ids = sorted({int(s) for s in shapes})
    n = 0
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            builder.add_shape_collision_filter_pair(a, b)
            n += 1
    return n


def collider_vertices(builder: newton.ModelBuilder, shapes) -> Any:
    """Every collider mesh vertex of ``shapes``, in the body's own frame [m].

    ``compute_obb`` only needs each mesh's eight bounding corners, so it stops
    there. A decomposition needs the vertices themselves -- the whole point is to
    see where the object is *narrower* than its bound.
    """
    np = _np()
    from robosimstudio.core.math import quat_to_mat

    out = []
    for s in shapes:
        mesh = builder.shape_source[s]
        if mesh is None or getattr(mesh, "vertices", None) is None or len(mesh.vertices) == 0:
            continue
        scale = np.asarray(builder.shape_scale[s], dtype=np.float64)[:3]
        verts = np.asarray(mesh.vertices, dtype=np.float64) * scale
        tf = builder.shape_transform[s]
        rot = quat_to_mat([tf[3], tf[4], tf[5], tf[6]])
        pos = np.array([tf[0], tf[1], tf[2]], dtype=np.float64)
        out.append(pos[None, :] + verts @ rot.T)
    if not out:
        return np.zeros((0, 3))
    return np.concatenate(out, axis=0)


def slab_decomposition(vertices, *, slabs: int = 8, merge_tol: float = 0.0015):
    """A stack of axis-aligned boxes along the object's longest axis.

    One box at the mesh's bound -- the ``"aabb"`` collider -- settles like a box
    and stops the creep a many-vertex hull causes, which is why it is there. What
    it cannot do is be *narrower anywhere*: a mustard bottle becomes a full-width
    block up to its cap, and a gable-top carton becomes a block up to its ridge.
    Both are then grasped and scored against a shape they do not have.

    So the bound is taken per slab instead of once. The structure is LIBERO's --
    a solid approximated by a handful of authored boxes rather than by its hull --
    generated from our own meshes rather than borrowed, because LIBERO ships only
    three of this suite's five groceries and its boxes carry its own scale.

    **It degenerates to the AABB exactly when the object is a prism.** Adjacent
    slabs whose cross-sections agree within ``merge_tol`` are merged, so a can
    comes back out as the single box it already was -- same collider, same contact
    count, no regression -- and only a genuinely tapered object spends more shapes.

    Args:
        vertices: ``(N, 3)`` body-frame collider vertices.
        slabs: How many bands to cut before merging. The ceiling on detail.
        merge_tol: Cross-section agreement below which two bands are one box [m].

    Returns:
        ``(boxes, axis)``, boxes as ``(centre, half)`` pairs in the body frame.
    """
    np = _np()
    pts = np.asarray(vertices, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return [], 0

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    axis = int(np.argmax(hi - lo))
    edges = np.linspace(lo[axis], hi[axis], int(max(slabs, 1)) + 1)

    sliced: list[tuple[Any, Any] | None] = []
    for k in range(len(edges) - 1):
        a, b = edges[k], edges[k + 1]
        # The last band takes its own upper edge, so the top vertices are in it.
        sel = (pts[:, axis] >= a) & (pts[:, axis] <= b if k == len(edges) - 2 else pts[:, axis] < b)
        band = pts[sel]
        if band.shape[0] < 3:
            sliced.append(None)
            continue
        blo, bhi = band.min(axis=0), band.max(axis=0)
        # The band spans its whole slice along the axis even where no vertex sits
        # exactly on the cut, or consecutive boxes would leave gaps to fall through.
        blo[axis], bhi[axis] = a, b
        sliced.append((blo, bhi))

    if all(band is None for band in sliced):
        return [], axis

    # **An empty band is not a hole in the object.** A low-poly barrel carries
    # vertices only at its two end rings, and 60 mm of a tool-sorting syringe lies
    # between a pair of them -- so dropping the bands between them left the
    # collider open along the middle of the tool, which the jaws closed straight
    # through. An empty band takes the cross-section of its nearest populated
    # neighbour instead, preferring the one below, so the stack still tiles the
    # whole axis. An object whose bands are all populated is untouched by this.
    for order in (range(len(sliced)), reversed(range(len(sliced)))):
        neighbour = None
        for k in order:
            if sliced[k] is not None:
                neighbour = sliced[k]
            elif neighbour is not None:
                blo, bhi = neighbour[0].copy(), neighbour[1].copy()
                blo[axis], bhi[axis] = edges[k], edges[k + 1]
                sliced[k] = (blo, bhi)

    bands = [band for band in sliced if band is not None]

    merged = [bands[0]]
    for blo, bhi in bands[1:]:
        plo, phi = merged[-1]
        cross = [i for i in range(3) if i != axis]
        same = all(abs(blo[i] - plo[i]) <= merge_tol and abs(bhi[i] - phi[i]) <= merge_tol for i in cross)
        if same:
            merged[-1] = (np.minimum(plo, blo), np.maximum(phi, bhi))
        else:
            merged.append((blo, bhi))

    boxes = [((blo + bhi) / 2.0, (bhi - blo) / 2.0) for blo, bhi in merged]
    return boxes, axis


# ------------------------------------------------------------- hydroelastic
SDF_MAX_RESOLUTION = 128
"""Voxel resolution cap for a cooked hydroelastic SDF. Newton's own example
value, and a ceiling rather than a target: the cook is superlinear in it."""

SDF_MARGIN = 0.006
"""How far past the mesh the cooked field's box reaches [m].

**Newton's own default is 0.05, and paying it here is paying for nothing.** The
field stores only :data:`SDF_NARROW_BAND_RANGE` -- five millimetres either side
of the surface -- so a box padded fifty millimetres past the mesh is 45 mm of
grid per side that can never hold a sample the narrow band keeps. It is not free:
the box is cubed. On the 66 mm bulkhead the default turns a 78 mm box into a
166 mm one, which is 9.6x the voxels, and the cost lands as resolution because
:data:`SDF_MAX_RESOLUTION` divides whatever box it is given -- the cap's cell
goes from 0.61 mm to 1.30 mm on the same part.

That mattered. The bulkhead's mating sleeve is a double wall whose skins sit
0.552 mm apart and whose fit clears the plug's tab by 0.15 mm; neither survives
a 1.30 mm cell, and asking for a cell fine enough to hold them cost 29 GB of
device memory and held the runner to one worker. Six millimetres covers the band
with a millimetre to spare and gives that memory back.
"""

SDF_NARROW_BAND_RANGE = (-0.005, 0.005)
"""Signed-distance band the SDF stores [m].

Hydroelastic only ever reads pressure near the surface, so a narrow band is what
keeps the cook affordable. Wide enough to cover a 0.15 mm-class fit's whole
approach and every interpenetration a stiff pressure law will allow.
"""


def build_hydroelastic_sdf(
    mesh: newton.Mesh,
    *,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cache_dir: str | None = None,
    target_voxel_size: float | None = None,
) -> None:
    """Cook the SDF a hydroelastic shape needs, in place, with a disk cache.

    Hydroelastic contact is a **pressure field** rather than a point set: two
    overlapping bodies get an iso-pressure patch and contacts are emitted across
    it, so a tab in a slot is carried by a distributed surface rather than by
    whichever handful of points the narrow phase picked that step. That field is
    sampled from a signed-distance function, so a mesh collided this way must
    carry one.

    Cooking is slow -- seconds to minutes for a fine mesh -- so it is cached
    content-addressed on disk and paid once per machine rather than once per
    build. The cache is shared and keyed by content, which is what makes
    leftovers from another task or an older run harmless rather than a source of
    a stale field.

    **Scale has to be baked in**, and matching the ``add_shape_mesh`` call is not
    optional: the SDF is sampled in the shape's own space, so a field cooked at
    x1 under a shape built at x2 reports distances half what they are and a
    pressure law reads an interpenetration that is not there.

    Args:
        mesh: The mesh to cook, modified in place.
        scale: Shape scale to bake in, matching the ``add_shape_mesh`` call.
        cache_dir: Where to persist cooked SDFs; ``None`` uses a shared
            temporary directory.
        target_voxel_size: Voxel size [m] to cook at, overriding the
            :data:`SDF_MAX_RESOLUTION` cap; ``None`` keeps the cap. The knob a
            *fit* needs and a free part does not: the field is exact only where
            surfaces are isolated at the voxel scale, so a functional surface
            within a cell of another -- a double-walled slot, a liner let into
            a wall -- collides wherever the smear lands. See
            ``ObjectSpec.sdf_voxel`` for the case that forced this through.
    """
    import tempfile
    from pathlib import Path

    directory = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "robosimstudio_sdf_cache"
    directory.mkdir(parents=True, exist_ok=True)
    mesh.build_sdf(
        max_resolution=SDF_MAX_RESOLUTION,
        narrow_band_range=SDF_NARROW_BAND_RANGE,
        target_voxel_size=target_voxel_size,
        margin=SDF_MARGIN,
        scale=scale,
        cache_dir=directory,
    )
