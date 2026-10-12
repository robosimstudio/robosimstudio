# SPDX-License-Identifier: Apache-2.0
"""Putting a mesh file into a builder, for an entity the library has never heard of.

A third party writing an entity has geometry in a file and needs it in a
:class:`newton.ModelBuilder`. Everything else about writing an entity is already
open -- the lifecycle finds phases by method name, the compiler reaches classes
by dotted path -- and this was the one step where the only worked examples were
two private methods inside the library's own entities.

**This is not those two methods factored out, and the reason is worth stating.**
``EntityStatic._add_asset`` and ``EntityRigid.build`` look like one function
written twice with drift, and they are not: they are the world-weld case and a
strict superset of it. A static hangs its shapes on body ``-1``; a rigid creates
a free body *and records that joint's coordinate rows*, because a reset that
cannot find them silently leaves the object wherever the last episode put it.
Everything that differs after that follows from owning a body -- a shape config
with density and contact stiffness, a hydroelastic field cooked at the import
scale, inertia preserved across a proxy, mass restated after an approximation.
And the format options are on the **entity specs**, not on
:class:`~robosimstudio.specs.AssetSpec`: ``collapse_fixed_joints`` is
``StaticSpec``'s; ``hydroelastic``, ``ke``, ``kd`` and ``visual_markers`` are
``ObjectSpec``'s. A function unifying the two would take a dozen optional
arguments to save forty lines, and half of them would be no-ops on either side.
That is a worse abstraction than the duplication, which at least says plainly
which entity owns which knob.

So what is here is the part a third party actually needs and cannot write from
outside: **the mesh path**, with the collision-proxy and approximation handling
that goes with it. MJCF and USD are deliberately absent -- an MJCF carries its
own bodies and joints and an entity importing one is making decisions about
articulation collapse that belong to it, and ``add_usd`` has no scale argument
at all. Both are one line of :mod:`robosimstudio.scene.build.imports` away for an
entity that wants them, and :class:`~robosimstudio.domains.static.EntityStatic`
is the worked example.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.paths import resolve_asset, resolve_path
from robosimstudio.specs import AssetSpec

from .build.bodies import preserve_body_inertia
from .build.colliders import copy_shape_material

__all__ = ["MESH_SUFFIXES", "Added", "load_asset", "mesh_texturing"]

MESH_SUFFIXES = (".obj", ".stl", ".ply", ".glb", ".gltf", ".dae")
"""Mesh formats :func:`load_asset` reads, via ``trimesh``."""


@dataclass(frozen=True)
class Added:
    """What one :func:`load_asset` call put in the builder, as half-open ranges.

    Returned rather than left for the caller to bracket, because an entity that
    has to diff ``builder.shape_count`` around a call is an entity that will get
    it wrong once and then be wrong quietly: the shapes it failed to claim are
    the shapes the interaction graph gives to nobody, and a thing that collides
    with nothing looks exactly like a thing at rest.

    Attributes:
        visual: The shape carrying the authored geometry.
        shapes: ``(start, end)`` over every shape added, the visual included.
        proxies: The collision-only shapes, if the asset named any.
    """

    visual: int
    shapes: tuple[int, int]
    proxies: tuple[int, ...] = ()


def mesh_texturing(asset: AssetSpec, mesh: Any, *, label: str = "") -> dict:
    """``uvs`` and ``texture`` for :class:`newton.Mesh`, or an empty dict.

    A vendored mesh carries its UVs and leaves its diffuse map in a sibling file,
    so the two are joined here: the coordinates come off the geometry and the
    image off :attr:`~robosimstudio.specs.AssetSpec.texture`.

    **A named texture with no UVs raises.** Passing an image through with no
    coordinates to sample it at is the failure this exists to stop being silent
    -- the object arrives with a material, renders flat, and reads as a texture
    that did not load rather than as a mesh that cannot carry one.
    ``trimesh.load(force="mesh")`` drops UVs when it concatenates several
    materials, which is exactly when a vendored asset needs re-exporting rather
    than shipping unnoticed.
    """
    if not asset.texture:
        return {}
    uvs = getattr(getattr(mesh, "visual", None), "uv", None)
    if uvs is None or len(uvs) != len(mesh.vertices):
        raise ValueError(
            f"{label or asset.path}: names texture {asset.texture!r} but {asset.path!r} "
            "carries no per-vertex UVs to sample it at"
        )
    return {"uvs": np.asarray(uvs, dtype=np.float32), "texture": str(resolve_path(asset.texture))}


def load_asset(
    builder: newton.ModelBuilder,
    asset: AssetSpec,
    *,
    xform: Any = None,
    body: int = -1,
    label: str = "",
    cfg: Any = None,
    collider: str | None = None,
) -> Added:
    """Add one mesh asset to ``builder`` and report what it added.

    Args:
        builder: The builder mid-build.
        asset: What to load. :attr:`~robosimstudio.specs.AssetSpec.scale`,
            ``texture``, ``collision_path``/``collision_paths``, ``approximate``
            and ``keep_visual`` are all honoured.
        xform: Where to put it. A ``wp.transform``, or a 7-sequence
            ``(x, y, z, qx, qy, qz, qw)``, or ``None`` for the identity.
            **An entity should pass** :meth:`~robosimstudio.scene.entity.Entity.placed`
            of its pose, not the raw pose: that is what puts a spec written in
            the work-surface frame onto a bench of a different height, and a
            body that skips it lands inside the table.
        body: Which body to hang the shapes on. ``-1`` welds them to the world,
            which is what scenery wants. An entity that wants a *free* body
            calls ``builder.add_body`` itself and passes the index -- and must
            record ``joint_q_start``/``joint_qd_start`` for that joint if it
            intends to reset, which is a decision this function will not make
            on its behalf.
        label: Name for the shapes. The proxies get ``f"{label}_collision"``.
        cfg: A ``ShapeConfig``, for density, friction, gap and contact
            stiffness. ``None`` takes the builder's default, which is right for
            a world-welded fixture and rarely right for a body with mass.
        collider: Approximation to apply to everything added --
            ``"coacd"``, ``"convex_hull"``, ``"bounding_box"``. Overrides
            :attr:`~robosimstudio.specs.AssetSpec.approximate`.

    Raises:
        ValueError: On a format this does not read (see :data:`MESH_SUFFIXES`
            and the module docstring for why MJCF and USD are not here), or on
            collision proxies combined with an approximation -- the
            approximation would swallow the proxies into the very hull they
            exist to avoid.
    """
    import trimesh

    # The suffix before the resolution, deliberately: a caller who hands this an
    # MJCF should be told that MJCF is not read here, not that a file it was
    # never going to open is missing.
    if not str(asset.path).lower().endswith(MESH_SUFFIXES):
        raise ValueError(
            f"{label or asset.path}: load_asset reads {list(MESH_SUFFIXES)}, and {asset.path!r} is none of them. "
            "An MJCF or a USD stage carries its own bodies and joints, and the options that govern "
            "how they arrive belong to the entity importing them -- see EntityStatic, and "
            "robosimstudio.scene.build.imports's add_mjcf_collapsed and add_usd_stage."
        )
    path = resolve_asset(asset)
    placed = _transform(xform)
    scale = (float(asset.scale),) * 3
    proxy_paths = ((asset.collision_path,) if asset.collision_path else ()) + tuple(asset.collision_paths)
    approximate = collider or asset.approximate
    if proxy_paths and approximate:
        raise ValueError(
            f"{label or asset.path}: collision proxies cannot be combined with "
            f"approximate={approximate!r} -- the approximation would swallow the proxies into "
            "the very hull they exist to avoid"
        )

    first_shape = builder.shape_count
    mesh = trimesh.load(str(path), force="mesh")
    visual = builder.add_shape_mesh(
        body,
        xform=placed,
        mesh=newton.Mesh(
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int32).reshape(-1),
            **mesh_texturing(asset, mesh, label=label),
        ),
        scale=scale,
        label=label or asset.path,
        **({"cfg": cfg} if cfg is not None else {}),
    )

    proxies: list[int] = []
    if proxy_paths:
        builder.shape_flags[visual] = int(builder.shape_flags[visual]) & ~int(newton.ShapeFlags.COLLIDE_SHAPES)
    for index, proxy_path in enumerate(proxy_paths):
        # ``process=False``: a collision proxy is already an authored simulation
        # mesh, and trimesh's normal processing merges coincident vertices after
        # decimation -- which turns two separate surface layers into one
        # non-manifold contact seam.
        proxy = trimesh.load(str(resolve_path(proxy_path)), force="mesh", process=False)
        name = f"{label}_collision" if len(proxy_paths) == 1 else f"{label}_collision_{index:02d}"
        with preserve_body_inertia(builder, body) if body >= 0 else _nothing():
            shape = builder.add_shape_mesh(
                body,
                xform=placed,
                mesh=newton.Mesh(
                    np.asarray(proxy.vertices, dtype=np.float32),
                    np.asarray(proxy.faces, dtype=np.int32).reshape(-1),
                ),
                scale=scale,
                label=name,
                **({"cfg": cfg} if cfg is not None else {}),
            )
        builder.shape_flags[shape] = int(builder.shape_flags[shape]) & ~int(newton.ShapeFlags.VISIBLE)
        copy_shape_material(builder, visual, shape)
        proxies.append(shape)

    if approximate:
        builder.approximate_meshes(
            method=approximate,
            shape_indices=list(range(first_shape, builder.shape_count)),
            keep_visual_shapes=asset.keep_visual,
        )
    return Added(visual=visual, shapes=(first_shape, builder.shape_count), proxies=tuple(proxies))


class _nothing:
    """A no-op context manager. ``preserve_body_inertia`` needs a body to preserve."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return False


def _transform(xform: Any) -> wp.transform:
    """A ``wp.transform`` from one, from a 7-sequence, or from nothing."""
    if xform is None:
        return wp.transform((0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))
    if isinstance(xform, wp.transform):
        return xform
    values = [float(v) for v in xform]
    if len(values) != 7:
        raise ValueError(f"xform must be a wp.transform or 7 numbers (x y z qx qy qz qw), got {len(values)}")
    return wp.transform(tuple(values[:3]), wp.quat(*values[3:7]))
