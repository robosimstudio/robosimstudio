# SPDX-License-Identifier: Apache-2.0
"""An MJCF or a USD stage into a builder, with what the importers drop restored.

:func:`add_mjcf_collapsed` and :func:`add_usd_stage` are the two doors. Both
apply :func:`~robosimstudio.core.builder.rekey_custom_attributes` to their
fixed-joint collapse, which is the fixup every scene with a collapsed arm
needs and the reason a caller should not reach for ``add_mjcf`` directly.

:func:`add_usd_stage` is the USD half
of the same job. An Isaac-exported stage arrives with a *geared* gripper whose
coupling ``add_usd`` does not import, and with a scene graph in which the same
mesh appears twice -- once as render geometry and once as a collider -- so both
of those need fixing before the model is steppable and photographable.

:func:`texture_box_geoms` is the same genre one importer over: ``add_mjcf``
binds a material's texture to mesh geoms and silently drops it from every
primitive, so an arena that *labels* anything with a textured box imports unable
to state what the label said.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import newton

from robosimstudio.core.builder import rekey_custom_attributes

__all__ = [
    "add_mjcf_collapsed",
    "add_usd_stage",
    "clear_texture_tints",
    "expose_declared_colliders",
    "is_textured",
    "strip_free_bodies",
    "texture_box_geoms",
]


def expose_declared_colliders(source: str, *, verbose: bool = False) -> tuple[Any, dict[str, list[str]]]:
    """Open a stage with the colliders it declares actually reachable by ``add_usd``.

    **An Isaac export can declare a collider in two ways ``add_usd`` cannot
    follow, and a stage that mixes them imports an asymmetric robot.** Both are
    about *where* the ``UsdPhysics.CollisionAPI`` is relative to the mesh:

    ``CollisionAPI on an Xform, mesh underneath``
        The importer's collider pass keys on the ``Mesh`` prim, so an ``Xform``
        that says "I collide" over a ``Mesh`` that does not say so contributes
        nothing. PhysX walks the subtree and honours it, which is why the export
        is valid for Isaac and silently lossy here. This function copies the
        schema down onto the mesh the Xform is speaking for.
    ``the prim is instanceable``
        Then it has no composed children at all -- the subtree lives in a
        prototype reachable only through an instance proxy -- so there is no mesh
        to key on and nothing to copy the schema to either. Cleared first.

    On the grocery stage's DROID gripper, **eight of the eleven** gripper
    collision prims hit both problems at once -- instanceable *and* declaring the
    schema on the Xform -- so they import as nothing. The other three declare it
    on their mesh and import fine, and they do not divide evenly:
    ``left_inner_finger`` gets two of them (its pad and its pad backing) while
    ``right_inner_finger`` gets one (its pad alone). So the hand imports
    asymmetric, with two colliding shapes on the left against one on the right.

    Worth stating plainly, since the asymmetry is easy to over-read: on this stage
    it turned out to be **harmless**. It is the object *masses* that made grasps
    slip, and recovering these eight colliders measured worse than leaving them
    out -- the recovered hulls over-approximate the Robotiq's thin bars. See the
    grocery suite's ``.stage`` for the measured comparison. This function is what
    reproduces it.

    Only prims already carrying a ``CollisionAPI`` are touched, so this adds no
    collider the stage did not ask for; a mesh named ``*_visual`` is left alone,
    being the render half of a pair whose other half is the collider. Every edit
    goes to a throwaway session layer, so the checked-in asset is never modified,
    and render-only instances (the table's ``visuals``) keep their instancing.

    Args:
        source: Path to the USD stage.
        verbose: Print each prim changed.

    Returns:
        ``(stage, changed)`` -- the opened stage to hand to ``add_usd``, and what
        was done, keyed ``"deinstanced"`` and ``"collision_applied"``.

    Raises:
        ValueError: If the stage cannot be opened.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    # Opened with an explicit anonymous session layer, and that is not a detail.
    # ``Usd.Stage.Open(path)`` gives a stage with **no** session layer, so
    # ``SetEditTarget(GetSessionLayer())`` falls back to the *root* layer -- and
    # root layers are shared process-wide by Sdf's layer cache. Editing there
    # mutates the checked-in asset for every stage opened afterwards (measured:
    # a second call found 0 instanceable prims because the first had already
    # cleared them, and the root layer came back ``dirty``). Nothing writes it to
    # disk, but a scene built later in the same process would silently get
    # different collision geometry.
    root = Sdf.Layer.FindOrOpen(source)
    if not root:
        raise ValueError(f"could not open USD stage {source!r}")
    session = Sdf.Layer.CreateAnonymous("robosimstudio_expose_declared_colliders")
    stage = Usd.Stage.Open(root, session, Usd.Stage.LoadAll)
    if not stage:
        raise ValueError(f"could not open USD stage {source!r}")
    stage.SetEditTarget(session)
    if stage.GetEditTarget().GetLayer() != session:
        raise ValueError(
            f"USD refused the anonymous session layer for {source!r}; refusing to edit the root layer, "
            "which is shared process-wide and would change every scene built after this one"
        )

    # Collected before anything is edited: clearing `instanceable` restructures
    # the composed scene graph, and mutating it mid-traversal skips prims.
    declared = [
        prim.GetPath()
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsInstanceProxy()
    ]

    deinstanced: list[str] = []
    for path in declared:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsInstanceable():
            prim.SetInstanceable(False)
            deinstanced.append(str(path))
            if verbose:
                print(f"[usd] de-instanced collider {path}")

    applied: list[str] = []
    for path in declared:
        prim = stage.GetPrimAtPath(path)
        if not prim or prim.IsA(UsdGeom.Mesh):
            continue  # already where the importer looks
        meshes = [d for d in Usd.PrimRange(prim) if d.IsA(UsdGeom.Mesh) and not d.GetName().endswith("_visual")]
        if any(d.HasAPI(UsdPhysics.CollisionAPI) for d in meshes):
            continue  # the stage already says it where it counts
        for mesh in meshes:
            UsdPhysics.CollisionAPI.Apply(mesh)
            applied.append(str(mesh.GetPath()))
            if verbose:
                print(f"[usd] applied CollisionAPI to {mesh.GetPath()}")
    return stage, {"deinstanced": deinstanced, "collision_applied": applied}


#: Module-local alias, because ``add_usd_stage`` takes a keyword of the same name.
_expose_declared_colliders = expose_declared_colliders


def add_usd_stage(
    builder: newton.ModelBuilder,
    source: str,
    *,
    collapse_fixed_joints: bool = False,
    expose_declared_colliders: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """``add_usd`` with the custom-attribute re-key applied to its collapse.

    The USD sibling of :func:`add_mjcf_collapsed`. ``parse_usd`` *does* hand back
    the collapse tables (under ``"collapse_results"``), so unlike the MJCF path
    there is nothing to work around -- but the re-key still has to happen, and a
    caller should not have to know that.

    Args:
        builder: Builder to import into.
        source: Path to the USD stage.
        collapse_fixed_joints: Fold the stage's fixed joints. Worth doing on an
            Isaac export, whose robot carries a wrist flange and a gripper base
            welded on: keeping them hands the solver zero-DOF articulations to
            maintain for nothing.
        expose_declared_colliders: Make the stage's declared colliders reachable
            by the importer -- colliders inside instance prototypes, and
            ``CollisionAPI`` declared on an Xform rather than on the mesh under
            it, are both otherwise dropped silently. See
            :func:`expose_declared_colliders` for what that costs on a gripper.
        verbose: Forwarded to the import and the re-key.
        **kwargs: Forwarded to :meth:`newton.ModelBuilder.add_usd`.

    Returns:
        ``parse_usd``'s own result dict.
    """
    if expose_declared_colliders:
        source, changed = _expose_declared_colliders(source, verbose=verbose)
        if verbose:
            counts = ", ".join(f"{len(v)} {k}" for k, v in changed.items())
            print(f"[usd] exposed declared colliders: {counts}")
    results = builder.add_usd(source, collapse_fixed_joints=collapse_fixed_joints, verbose=verbose, **kwargs)
    collapse = (results or {}).get("collapse_results")
    if collapse:
        rekey_custom_attributes(builder, collapse, verbose=verbose)
    return results or {}


def clear_texture_tints(builder: newton.ModelBuilder, shapes=None) -> int:
    """Set every textured shape's colour to white, so its texture renders as authored.

    Newton's tiled ray tracer multiplies a sampled texel by the shape's own
    ``shape_color``; the visual-matching backends let the texture win outright.
    That difference only shows when the two disagree about what the colour is,
    and for a textured shape they nearly always do -- an importer that binds a
    texture usually states no separate colour, so the shape falls to the
    build-order palette and the tiled render comes out tinted by an arbitrary
    hue while pyrender shows the texture clean.

    The grocery tote is the measured case: a grey plastic albedo (mean RGB
    91, 95, 90) times a palette red (0.96, 0.37, 0.37) renders a dark red bin
    in a scene whose provenance calls it grey.

    White is the identity of that multiply, so this makes the two backends agree
    on the one class of shape where the texture, not the colour, is the claim.

    Args:
        builder: The builder, after every import.
        shapes: Shape indices to consider; ``None`` is every shape.

    Returns:
        How many shapes were whitened.
    """
    cleared = 0
    for s in range(builder.shape_count) if shapes is None else shapes:
        if not is_textured(builder, s):
            continue
        if tuple(float(v) for v in builder.shape_color[s]) == (1.0, 1.0, 1.0):
            continue
        builder.shape_color[s] = (1.0, 1.0, 1.0)
        cleared += 1
    return cleared


def is_textured(builder: newton.ModelBuilder, shape: int) -> bool:
    """Whether this shape carries a texture, and so states no colour of its own.

    Split out of :func:`clear_texture_tints` because two passes need the same
    question and a second copy of it is a second thing to get wrong. The other
    caller is the per-entity display pass, which must not paint a role default
    onto a textured shape for the reason :func:`clear_texture_tints` exists:
    under the tiled tracer a colour on a texture is a multiply, not a choice.
    """
    src = builder.shape_source[shape]
    return src is not None and getattr(src, "texture", None) is not None


def texture_box_geoms(builder: newton.ModelBuilder, source: str, *, verbose: bool = False) -> int:
    """Draw the textures Newton's MJCF importer binds to meshes and to nothing else.

    **The gap this fills.** ``import_mjcf`` resolves a geom's ``material`` to a
    texture file and threads it through as ``override_texture`` -- but only on
    its ``geom_type == "mesh"`` branch. Every primitive branch calls
    ``add_shape_box`` (and its siblings) with the material's *colour* and drops
    the image on the floor, silently: no warning, because from the importer's
    side nothing failed. The shape arrives with ``shape_source is None``, and
    with no source there is nothing for any backend to sample, so it falls to
    the build-order palette and renders as a flat coloured slab.

    The measured case is the tool-sort workcell, whose bin2 compartments are
    labelled by four textured box geoms 2 mm thick -- ``BATTERY``,
    ``MULTIMETER``, ``FLASHLIGHT``, ``SAFETY GLASSES``. In MuJoCo they are
    readable plates; imported, they are a cyan, a green, a yellow and a red
    rectangle. That is not a cosmetic loss on a task whose whole instruction is
    *"into the compartment whose label names each one"*: the scene stops stating
    the thing the policy is asked to read, and a VLM shown the render is being
    asked a question the picture no longer contains.

    What this adds is **visual only** -- a textured mesh per affected geom, with
    ``has_shape_collision=False`` -- and the original box is hidden rather than
    removed, so shape indices, contact behaviour and every handle that already
    points at one are untouched.

    **Cylinders too, for the same reason boxes needed it.** The tool-sort
    workbench's own four legs are ``type="cylinder"`` wearing a brushed-steel
    material, dropped exactly as the compartment plates were -- a flat grey
    pole where MuJoCo shows chrome. See :func:`_textured_cylinder_mesh`.

    **Placement is read back from the builder, not recomputed.** Only the
    ``geom name -> texture file`` mapping comes from the XML; the body, the
    transform and the half-extents come from the shape the importer already
    placed. So a scaled import, an ``xform`` argument and a fixed-joint collapse
    all land correctly by construction rather than by re-deriving them here and
    hoping the two agree.

    **UVs follow MuJoCo's own box convention**, which was read off a render
    rather than assumed: on the ``+z`` face, viewed down the axis, image-right
    is local ``+x`` and image-down is local ``-y``, with the full ``[0, 1]^2``
    across each face (``texuniform="false"``). Each of the six faces gets the
    whole image, which is what MuJoCo draws.

    Args:
        builder: The builder, immediately after the MJCF import.
        source: The same MJCF path.
        verbose: Print one line per geom re-textured.

    Returns:
        How many textured meshes were added.
    """
    import os
    import xml.etree.ElementTree as ET

    import numpy as np
    import warp as wp

    root = ET.parse(source).getroot()
    base = os.path.dirname(os.path.abspath(source))
    compiler = root.find("compiler")
    texdir = (
        (compiler.attrib.get("texturedir") or compiler.attrib.get("assetdir") or ".") if compiler is not None else "."
    )

    face_attrs = ("fileright", "fileleft", "filetop", "filebottom", "filefront", "fileback")
    textures: dict[str, str] = {}
    for tex in root.iter("texture"):
        name, file = tex.attrib.get("name"), tex.attrib.get("file")
        # A builtin (the skybox) has no file. A cube map is normally six
        # images this single-image path has no business guessing at -- but
        # MuJoCo also lets ``type="cube"`` name just one ``file`` and paints
        # it on all six faces (no per-face attribute), which is exactly a 2D
        # texture for any purpose that doesn't care which face is which, and
        # is how this file's own brushed-steel table legs are declared.
        if not (name and file):
            continue
        kind = tex.attrib.get("type", "2d")
        if kind == "2d" or (kind == "cube" and not any(a in tex.attrib for a in face_attrs)):
            textures[name] = os.path.normpath(os.path.join(base, texdir, file))

    materials: dict[str, tuple[str, tuple[float, float], bool]] = {}
    for m in root.iter("material"):
        name, tex = m.attrib.get("name"), m.attrib.get("texture")
        if not name or tex not in textures:
            continue
        # texuniform="true" asks for repeats per unit *length* -- a tiling in
        # world space rather than across the face -- so it is carried through
        # as a flag and the per-face extents scale the repeats at mesh build.
        # It used to be skipped outright, which left the one geom this whole
        # function was measured on a sibling of -- the workbench top, steel
        # plate at texrepeat "2 4" texuniform -- as a flat slate slab.
        uniform = m.attrib.get("texuniform", "false").lower() in ("true", "1")
        repeat = [float(v) for v in m.attrib.get("texrepeat", "1 1").split()]
        materials[name] = (textures[tex], (repeat[0], repeat[-1]), uniform)

    wanted = {
        g.attrib["name"]: materials[g.attrib["material"]]
        for g in root.iter("geom")
        if g.attrib.get("name")
        and g.attrib.get("type") in ("box", "cylinder")
        and g.attrib.get("material") in materials
    }
    if not wanted:
        return 0

    visible = int(newton.ShapeFlags.VISIBLE)
    box_type, cylinder_type = int(newton.GeoType.BOX), int(newton.GeoType.CYLINDER)
    added = 0
    # Snapshotted: add_shape_mesh appends, and iterating a growing range would
    # walk into the meshes this loop is producing.
    for s in range(builder.shape_count):
        shape_type = int(builder.shape_type[s])
        if shape_type not in (box_type, cylinder_type):
            continue
        # The importer's labels are slash-joined paths; the geom name is the leaf.
        found = wanted.get(str(builder.shape_label[s]).rpartition("/")[2])
        # A source means the importer already bound something -- leave it alone.
        if found is None or builder.shape_source[s] is not None:
            continue
        if not (int(builder.shape_flags[s]) & visible):
            continue
        texture, repeat, uniform = found
        if not os.path.exists(texture):
            raise ValueError(f"{source}: geom {builder.shape_label[s]!r} names a texture that is not there: {texture}")

        if shape_type == box_type:
            hx, hy, hz = (float(v) for v in builder.shape_scale[s])
            mesh = _textured_box_mesh(np, hx, hy, hz, texture, repeat, uniform=uniform)
        else:
            radius, half_height, _ = (float(v) for v in builder.shape_scale[s])
            mesh = _textured_cylinder_mesh(np, radius, half_height, texture, repeat, uniform=uniform)
        cfg = builder.default_shape_cfg.copy()
        cfg.has_shape_collision = False
        cfg.has_particle_collision = False
        cfg.density = 0.0
        xf = builder.shape_transform[s]
        builder.add_shape_mesh(
            body=int(builder.shape_body[s]),
            xform=wp.transform(wp.vec3(*(float(v) for v in xf[:3])), wp.quat(*(float(v) for v in xf[3:7]))),
            mesh=mesh,
            cfg=cfg,
            # White, so the texture is what shows -- the same identity
            # clear_texture_tints restores for every other textured shape.
            color=(1.0, 1.0, 1.0),
            label=f"{builder.shape_label[s]}_textured",
        )
        # Hidden, not deleted: the box keeps its index, and anything already
        # holding one keeps pointing at the same shape.
        builder.shape_flags[s] = int(builder.shape_flags[s]) & ~visible
        added += 1
        if verbose:
            print(f"re-textured {builder.shape_label[s]} from {os.path.basename(texture)}")
    return added


def _textured_box_mesh(
    np, hx: float, hy: float, hz: float, texture: str, repeat: tuple[float, float], *, uniform: bool = False
):
    """A box carrying the image on each of its six faces, ``texrepeat`` times.

    Split into 24 vertices rather than 8 because a UV is per-vertex: a corner
    shared by three faces would have to hold three different texture
    coordinates, and a cube with 8 vertices can state only one.

    Args:
        uniform: MuJoCo's ``texuniform``: ``repeat`` counts repeats per metre
            rather than per face, so each face multiplies it by its own extent
            along that face's u and v.
    """
    # (axis, sign, u axis, v axis) -- v runs against the image's own row order,
    # which is why each v axis is negated when the frame is built below.
    faces = (
        (2, +1, (1, 0, 0), (0, -1, 0)),  # +z: image-right is +x, image-down is -y
        (2, -1, (-1, 0, 0), (0, -1, 0)),
        (1, +1, (1, 0, 0), (0, 0, -1)),
        (1, -1, (-1, 0, 0), (0, 0, -1)),
        (0, +1, (0, -1, 0), (0, 0, -1)),
        (0, -1, (0, 1, 0), (0, 0, -1)),
    )
    half = np.array([hx, hy, hz], dtype=np.float32)
    vertices, normals, uvs, indices = [], [], [], []
    for axis, sign, u_axis, v_axis in faces:
        normal = np.zeros(3, dtype=np.float32)
        normal[axis] = sign
        u = np.array(u_axis, dtype=np.float32)
        v = np.array(v_axis, dtype=np.float32)
        centre = normal * half
        base = len(vertices)
        face_repeat = repeat
        if uniform:
            # Repeats per metre: scale by this face's full extent along u and v.
            face_repeat = (
                repeat[0] * 2.0 * float(np.abs(u * half).sum()),
                repeat[1] * 2.0 * float(np.abs(v * half).sum()),
            )
        # Corners in image order: (u=0,v=0) top-left, then across, then down.
        for du, dv, tu, tv in ((-1, 1, 0.0, 0.0), (1, 1, 1.0, 0.0), (1, -1, 1.0, 1.0), (-1, -1, 0.0, 1.0)):
            vertices.append(centre + u * half * du + v * half * dv)
            normals.append(normal)
            uvs.append((tu * face_repeat[0], tv * face_repeat[1]))
        indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))
    return newton.Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        normals=np.asarray(normals, dtype=np.float32),
        uvs=np.asarray(uvs, dtype=np.float32),
        texture=texture,
        color=(1.0, 1.0, 1.0),
        compute_inertia=False,
    )


def _textured_cylinder_mesh(
    np, radius: float, half_height: float, texture: str, repeat: tuple[float, float], *, uniform: bool = False
):
    """A cylinder's side wall carrying one image, wrapped once around like a label.

    MuJoCo states no UV convention for a textured cylinder primitive the way
    it does for a box's six faces (``_textured_box_mesh``'s docstring), so this
    picks the one every renderer uses for a can or a pole: image left-to-right
    is the circumference (angle 0 to 2*pi), image top-to-bottom is the height
    (``-half_height`` to ``+half_height``). ``texuniform`` scales the repeat by
    the true circumference and height, the same convention the box mesh's
    per-face scaling uses, so a material stating ``texrepeat="1 1"`` with no
    ``texuniform`` wraps the image exactly once around and once along --
    which is what makes a brushed-steel image read as a chrome pole rather
    than as a smear or an illegible tile.

    The end caps are triangle fans sampled at the image's own centre rather
    than genuinely unwrapped: a table leg's caps are butted against the floor
    and the underside of the top, so no camera this suite ships ever sees one,
    and a flat sample is what keeps a stray view reading as "the same
    material" instead of a seam of the raw palette colour.
    """
    segments = 32
    height = 2.0 * half_height
    circumference = 2.0 * float(np.pi) * radius
    u_repeat = repeat[0] * circumference if uniform else repeat[0]
    v_repeat = repeat[1] * height if uniform else repeat[1]

    vertices: list = []
    normals: list = []
    uvs: list = []
    indices: list = []

    # Side wall: one ring of quads, wound so the outward normal faces out.
    for i in range(segments):
        a0 = 2.0 * np.pi * i / segments
        a1 = 2.0 * np.pi * (i + 1) / segments
        n0 = np.array([np.cos(a0), np.sin(a0), 0.0], dtype=np.float32)
        n1 = np.array([np.cos(a1), np.sin(a1), 0.0], dtype=np.float32)
        lo = np.array([0.0, 0.0, -half_height], dtype=np.float32)
        hi = np.array([0.0, 0.0, half_height], dtype=np.float32)
        base = len(vertices)
        vertices += [n0 * radius + lo, n1 * radius + lo, n1 * radius + hi, n0 * radius + hi]
        normals += [n0, n1, n1, n0]
        u0, u1 = u_repeat * i / segments, u_repeat * (i + 1) / segments
        uvs += [(u0, 0.0), (u1, 0.0), (u1, v_repeat), (u0, v_repeat)]
        indices += [base, base + 1, base + 2, base, base + 2, base + 3]

    # End caps: a centre vertex plus the rim, both sampled at the image centre.
    centre_uv = (0.5 * u_repeat, 0.5 * v_repeat)
    for sign in (-1.0, 1.0):
        z = sign * half_height
        normal = np.array([0.0, 0.0, sign], dtype=np.float32)
        centre = len(vertices)
        vertices.append(np.array([0.0, 0.0, z], dtype=np.float32))
        normals.append(normal)
        uvs.append(centre_uv)
        ring_start = len(vertices)
        for i in range(segments):
            a = 2.0 * np.pi * i / segments
            vertices.append(np.array([radius * np.cos(a), radius * np.sin(a), z], dtype=np.float32))
            normals.append(normal)
            uvs.append(centre_uv)
        for i in range(segments):
            a_index, b_index = ring_start + i, ring_start + (i + 1) % segments
            # Wound the opposite way on the two caps, so both face outward.
            tri = (centre, a_index, b_index) if sign > 0 else (centre, b_index, a_index)
            indices.extend(tri)

    return newton.Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        normals=np.asarray(normals, dtype=np.float32),
        uvs=np.asarray(uvs, dtype=np.float32),
        texture=texture,
        color=(1.0, 1.0, 1.0),
        compute_inertia=False,
    )


def add_mjcf_collapsed(
    builder: newton.ModelBuilder,
    source: str,
    *,
    verbose: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """``add_mjcf`` with the fixed-joint collapse run afterwards, and re-keyed.

    ``add_mjcf(collapse_fixed_joints=True)`` throws away the remap tables the
    collapse returns, so the re-key cannot be applied to its result. Importing
    uncollapsed and collapsing explicitly keeps them.

    Args:
        builder: Builder to import into.
        source: Path to the MJCF, or its XML text.
        verbose: Forwarded to both the import and the collapse.
        **kwargs: Forwarded to :meth:`newton.ModelBuilder.add_mjcf`. Passing
            ``collapse_fixed_joints`` is an error -- that is what this function
            is for.

    Returns:
        The collapse results dict, so a caller can remap its own indices.
    """
    if "collapse_fixed_joints" in kwargs:
        raise ValueError("add_mjcf_collapsed collapses itself; drop the collapse_fixed_joints argument")
    builder.add_mjcf(source, verbose=verbose, collapse_fixed_joints=False, **kwargs)
    results = builder.collapse_fixed_joints(verbose=verbose)
    rekey_custom_attributes(builder, results, verbose=verbose)
    return results


def strip_free_bodies(scene_xml: str | Path) -> str:
    """Write a sibling MJCF with every free-jointed body and keyframe removed.

    The stripped file goes in the *source's* directory so its relative
    ``meshdir`` still resolves. The caller owns the temp file.

    **Both spellings of a free joint count.** MJCF says the same thing two ways
    -- ``<freejoint/>`` and ``<joint type="free"/>`` -- and a check for one of
    them fails open on a file that uses the other: nothing raises, nothing is
    removed, and the caller gets back a copy of the scene it asked to have
    emptied. Every robosuite export in this repo uses the attribute form, so
    this read only the spelling they do not use.
    """
    src = Path(scene_xml)
    tree = ET.parse(src)
    root = tree.getroot()

    def free_jointed(body) -> bool:
        return body.find("freejoint") is not None or any(j.get("type") == "free" for j in body.findall("joint"))

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for body in list(worldbody.findall("body")):
            if free_jointed(body):
                worldbody.remove(body)
    # A keyframe indexes the qpos of the bodies just removed.
    for keyframe in root.findall("keyframe"):
        root.remove(keyframe)

    fd, path = tempfile.mkstemp(prefix="robosimstudio_stripped_", suffix=".xml", dir=str(src.parent))
    os.close(fd)
    tree.write(path, xml_declaration=False, encoding="unicode")
    return path
