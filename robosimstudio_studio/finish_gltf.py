# SPDX-License-Identifier: Apache-2.0
"""A :class:`Finish` as glTF, so the browser page shades what the tracer shades.

viser's own mesh primitives carry a flat ``color`` and nothing else -- no
roughness, no metalness, no texture. Its ``MeshProps`` has three material names
(``standard``, ``toon3``, ``toon5``) and a colour, which is why every object on
the page has looked like chalk regardless of what the scene said it was made of.
What viser *can* do is instance a GLB with full metallic-roughness PBR, through
``add_batched_glb``. So the page's materials arrive as glTF rather than as viser
material arguments, and this module is the encoder.

Three things it has to get right, and each is a place the obvious version is
wrong:

* **``baseColorFactor`` is linear; a viser ``color=`` triple is not.** The
  palette stores display values, ``add_mesh_simple`` hands them to three.js
  which converts on ingest, and glTF does not -- so the flat path passes the
  number through and this one decodes it
  (:func:`~robosimstudio.core.render.palette.srgb_to_linear`). Skip that and the
  page draws every surface several stops too bright, which reads as "the
  lighting is wrong" and is not.
* **UV transforms are baked into the UVs.** glTF says texture transforms with
  ``KHR_texture_transform``, which not every loader carries. The UVs here are
  generated rather than authored, so a finish's ``uv_scale`` (repeats per metre)
  and ``uv_offset`` are applied to the coordinates instead -- same picture, no
  extension.
* **trimesh quantises the base colour, so the factors are rewritten after
  export.** ``PBRMaterial`` keeps ``baseColorFactor`` as 8-bit RGBA internally,
  and glTF defines it as a float. Measured, a display 0.2 decodes to a linear
  0.0331 and comes back out of the exporter as 0.03137 -- 5 % low, and worst
  exactly where the palette lives, since most of this scheme is dark. The mesh
  half of trimesh's export is what this module wants; the one field it rounds is
  patched back to the exact value in :func:`_with_exact_factors`.
* **Encoding is cached, because it is not cheap and not varied.** A scene has
  far fewer distinct ``(geometry, colour, finish)`` triples than it has shapes:
  a tool sort bench is hundreds of instances over a handful of materials. The
  cache key is that triple, which is also the key the instanced batches are cut
  along.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
from functools import lru_cache
from typing import Any

import numpy as np

from robosimstudio.core.render.finish import Finish, texture_path
from robosimstudio.core.render.palette import srgb_to_linear

__all__ = ["GlbKey", "glb_for", "group_by_material", "material_key", "unweld", "wants_pbr"]

GlbKey = tuple

_CACHE: dict[GlbKey, bytes] = {}
"""Encoded GLBs by :func:`material_key`. Plain dict rather than ``lru_cache``
because the values are megabytes and a page holds one scene at a time: it is
cleared when the model changes rather than evicted by count."""


def material_key(color, finish: Finish) -> tuple:
    """What makes two shapes shareable as one instanced batch.

    Geometry is the caller's half of the key -- it already has a name for the
    prototype. This is the other half: a batch carries one material, so two
    shapes of the same shape in different colours are two batches.
    """
    rgb = tuple(round(float(c), 6) for c in tuple(color)[:3])
    return (
        rgb,
        round(float(finish.roughness), 6),
        round(float(finish.metallic), 6),
        round(float(finish.opacity), 6),
        finish.texture,
        # Two finishes alike but for their roughness map are two materials, and
        # a key that cannot tell them apart hands the second one the first's GLB.
        finish.roughness_texture,
        tuple(float(v) for v in finish.uv_scale),
        tuple(float(v) for v in finish.uv_offset),
    )


def clear_cache() -> None:
    """Forget every encoded GLB. Called when the page changes scene."""
    _CACHE.clear()
    _texture_image.cache_clear()
    _roughness_image.cache_clear()


@lru_cache(maxsize=32)
def _texture_image(name: str):
    """A finish's diffuse map as a PIL image, resolved and loaded once.

    ``texture_path`` renders a procedural texture into the same cache directory
    the path tracer reads, so the page and the render use one file rather than
    two generators that agree by accident.
    """
    if not name:
        return None
    try:
        from PIL import Image

        path = texture_path(name)
        return Image.open(path).convert("RGB") if path else None
    except Exception:
        # A texture that will not load is a duller picture, not a dead page.
        return None


@lru_cache(maxsize=16)
def _roughness_image(name: str):
    """A finish's roughness map, packed the way glTF reads one.

    glTF puts roughness in a texture's **green** channel and metalness in its
    **blue** one, and multiplies each by the matching factor. The path tracer
    does neither: it wires the map's **red** channel straight into
    ``UsdPreviewSurface``'s ``roughness``, in ``raw`` colour space, which
    *replaces* the scalar (``rtx.py``, ``_MaterialLibrary._build``). So the red
    channel is copied into green, blue is filled with 1.0 so ``metallicFactor``
    passes through untouched, and the caller sets ``roughnessFactor`` to 1.0 --
    which makes glTF's multiply come out at the map's own value, the number the
    tracer uses.

    The same replaces-versus-multiplies difference as the base colour, in the
    one other place a `Finish` can carry an image. It is worth stating twice:
    both times, getting it wrong applies the material's own number on top of a
    map that already is that number.
    """
    if not name:
        return None
    try:
        from PIL import Image

        path = texture_path(name)
        if not path:
            return None
        red = Image.open(path).convert("RGB").split()[0]
        return Image.merge("RGB", (Image.new("L", red.size, 0), red, Image.new("L", red.size, 255)))
    except Exception:
        return None


def glb_for(
    points: np.ndarray,
    faces: np.ndarray,
    color,
    finish: Finish,
    *,
    uvs: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    smooth: bool = False,
    cache: bool = True,
) -> bytes | None:
    """One prototype mesh, with this colour and finish, as GLB bytes.

    Args:
        points: ``(V, 3)`` vertices in metres.
        faces: ``(F, 3)`` triangle indices.
        color: display RGB in 0..1 -- the palette's own values, decoded here.
        finish: the material the scene states for this shape.
        uvs: ``(V, 2)`` per-vertex coordinates in metres, or ``None`` for an
            untextured material. ``uv_scale`` and ``uv_offset`` are applied here.
        normals: ``(V, 3)`` per-vertex normals, or ``None``. **glTF says a mesh
            without them is flat-shaded**, so a curved surface handed nothing
            comes back faceted -- which is the right answer for the room's boxes
            and the wrong one for a cylinder or a draped sheet. Passed rather
            than derived because the caller that needs them usually has better
            ones than a re-derivation would give: a garment's come off the same
            ``surface()`` call that produced the vertices.
        smooth: Derive per-vertex normals when ``normals`` is ``None``. For a
            caller whose geometry is curved but whose normals do not line up
            with its vertices -- a garment's shell has twice the vertices the
            surface it was built from has. Costs about 0.45 ms per 640
            vertices, so it is asked for rather than assumed.
        cache: Keep the blob for the next caller with the same geometry.
            ``False`` for a mesh that deforms: its vertices are new every frame,
            so a kept blob would never be read again and the cache -- cleared
            only when the model changes -- would grow by one every frame.

    Returns:
        The encoded GLB, or ``None`` when trimesh is absent or the export fails
        -- in which case the caller should fall back to a flat batch rather than
        show nothing.
    """
    if not cache:
        return _encode(points, faces, color, finish, uvs, normals, smooth)
    key = (_shape_key(points, faces, normals), material_key(color, finish), smooth)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    blob = _encode(points, faces, color, finish, uvs, normals, smooth)
    if blob is not None:
        _CACHE[key] = blob
    return blob


def _shape_key(points: np.ndarray, faces: np.ndarray, normals: np.ndarray | None = None) -> str:
    """A cheap content hash of one prototype's geometry."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(points, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(faces, dtype=np.uint32).tobytes())
    if normals is not None:
        digest.update(np.ascontiguousarray(normals, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _encode(points, faces, color, finish: Finish, uvs, normals=None, smooth=False) -> bytes | None:
    try:
        import trimesh
        from trimesh.visual.material import PBRMaterial
    except Exception:
        return None

    points = np.asarray(points, dtype=np.float64)
    mesh = trimesh.Trimesh(
        vertices=points,
        faces=np.asarray(faces, dtype=np.int64).reshape(-1, 3),
        vertex_normals=None if normals is None else np.asarray(normals, dtype=np.float64),
        process=False,
    )
    if normals is None and smooth:
        # Touching the property is what puts NORMAL in the export: trimesh
        # writes the attribute only when the mesh already carries it, and glTF
        # says a mesh without it is flat-shaded.
        _ = mesh.vertex_normals
    image = _texture_image(finish.texture)
    # glTF multiplies the base colour texture by the base colour factor;
    # `UsdPreviewSurface` *replaces* diffuseColor with the texture and uses the
    # colour only when there is none (`rtx.py`, `_MaterialLibrary._build`). So a
    # textured finish takes white here, or the colour is applied twice -- which
    # is what turned the lab's oak slats orange and darkened the mat's grid.
    linear = (1.0, 1.0, 1.0) if image is not None else srgb_to_linear(tuple(color)[:3])
    rough_map = _roughness_image(finish.roughness_texture)
    # 1.0 so glTF's `roughnessFactor * texture.g` lands on the map's own value,
    # which is what the tracer's `roughness <- texture.r` connection gives.
    roughness = 1.0 if rough_map is not None else float(finish.roughness)
    opacity = float(finish.opacity)
    material: Any = PBRMaterial(
        baseColorFactor=[*(float(c) for c in linear), opacity],
        metallicFactor=float(finish.metallic),
        roughnessFactor=roughness,
        # A translucent finish has to say so in the glTF, or a loader draws it
        # opaque and the region boxes and glazing come out as walls.
        alphaMode="BLEND" if opacity < 1.0 else "OPAQUE",
        doubleSided=True,
    )
    uv = None
    if (image is not None or rough_map is not None) and uvs is not None and len(uvs) == len(mesh.vertices):
        scale = np.asarray(finish.uv_scale, dtype=np.float64)
        offset = np.asarray(finish.uv_offset, dtype=np.float64)
        uv = np.asarray(uvs, dtype=np.float64) * scale + offset
        if image is not None:
            material.baseColorTexture = image
        if rough_map is not None:
            material.metallicRoughnessTexture = rough_map
    try:
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        with io.BytesIO() as buffer:
            mesh.export(buffer, file_type="glb")
            blob = buffer.getvalue()
    except Exception:
        return None
    return _with_exact_factors(blob, (*linear, opacity), roughness, finish.metallic)


_JSON_CHUNK = 0x4E4F534A


def _with_exact_factors(blob: bytes, rgba, roughness: float, metallic: float) -> bytes:
    """Put the unrounded PBR factors back into an exported GLB.

    trimesh stores ``baseColorFactor`` as 8-bit RGBA, so a linear 0.0331 leaves
    the exporter as 0.03137. That is a colour error of a couple of display
    levels on a mid tone and proportionally much worse on the dark end, which is
    where most of this palette sits. Everything else the exporter wrote --
    accessors, buffers, the image -- is left exactly as it is; only the three
    numbers glTF defines as floats are rewritten.

    Returns the blob unchanged if it is not a GLB this can parse, because a
    slightly-wrong colour beats no mesh.
    """
    try:
        magic, version, _total = struct.unpack_from("<III", blob, 0)
        chunks, offset = [], 12
        while offset < len(blob):
            size, kind = struct.unpack_from("<II", blob, offset)
            chunks.append([kind, blob[offset + 8 : offset + 8 + size]])
            offset += 8 + size
        for chunk in chunks:
            if chunk[0] != _JSON_CHUNK:
                continue
            doc = json.loads(chunk[1].decode("utf-8"))
            for material in doc.get("materials", []):
                pbr = material.setdefault("pbrMetallicRoughness", {})
                pbr["baseColorFactor"] = [float(v) for v in rgba]
                pbr["roughnessFactor"] = float(roughness)
                pbr["metallicFactor"] = float(metallic)
            raw = json.dumps(doc, separators=(",", ":")).encode("utf-8")
            raw += b" " * (-len(raw) % 4)  # a JSON chunk pads with spaces
            chunk[1] = raw
            break
        body = b"".join(struct.pack("<II", len(data), kind) + data for kind, data in chunks)
        return struct.pack("<III", magic, version, 12 + len(body)) + body
    except Exception:
        return blob


MATTE_ENOUGH = 0.9
"""At or above this roughness, a surface is drawn on the cheap flat path.

The number is ``MATTE``'s own, which is what the compiler gives a deformable
nobody dressed -- so the common case keeps the in-place vertex update and only a
scene that *asked* for gloss pays for a GLB.

It is a cost threshold and not an equivalence. Measured on a lit sphere, viser's
flat path matches no GLB roughness at all: it sits about 8 grey levels brighter
in the mean at every one of them, and its highlight is broader and dimmer than
even roughness 0.2 (99th percentile 221 against 211, widening to a 29-level gap
by 1.0). So there is no default to match, only a judgement about where the
difference stops being worth 0.4 ms a frame.
"""


def wants_pbr(finish: Finish | None) -> bool:
    """Whether a finish states anything ``add_mesh_simple`` throws away.

    Colour and opacity it carries; roughness, metalness and both texture slots
    it does not. A deforming mesh cannot use the cached GLB path -- its geometry
    changes every frame -- so this decides which shapes are worth re-encoding.
    """
    if finish is None:
        return False
    return bool(
        finish.texture
        or finish.roughness_texture
        or float(finish.metallic) > 0.0
        or float(finish.roughness) < MATTE_ENOUGH
    )


def group_by_material(shapes, colors, finishes, default: Finish) -> dict[tuple, list[int]]:
    """Instance indices of one batch, cut into groups that can share a GLB.

    Newton batches by geometry alone, so one batch is one shape in however many
    colours and materials the scene gave it. A GLB carries a single material, so
    the batch has to be cut again along :func:`material_key` before it can be
    instanced -- which is cheap, because a scene has few materials and many
    shapes: a tool-sort bench is hundreds of instances over a handful of them.

    Args:
        shapes: the model shape index behind each instance, in instance order.
        colors: ``model.shape_color``, or ``None`` to treat every shape as white.
        finishes: shape index -> :class:`Finish`, as the compiler resolved them.
        default: what a shape nobody gave a finish is made of.

    Returns:
        ``material_key -> [instance index, ...]``, insertion-ordered so the
        groups a batch produces are stable between frames.
    """
    groups: dict[tuple, list[int]] = {}
    for index, shape in enumerate(shapes):
        color = (1.0, 1.0, 1.0) if colors is None else tuple(float(c) for c in colors[shape][:3])
        finish = finishes.get(int(shape), default)
        groups.setdefault((material_key(color, finish), color, finish), []).append(index)
    return groups


def unweld(points, faces):
    """One vertex per triangle corner, so face-varying UVs can be per-vertex.

    A box's corner belongs to three faces with three different box projections
    and a vertex can only carry one UV, so the mesh is split before
    :func:`~robosimstudio.core.render.uv.box_projected_uvs` is applied. Only
    worth paying for a textured finish, which is why the caller decides.
    """
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    verts = np.asarray(points, dtype=np.float64)[faces.reshape(-1)]
    return verts, np.arange(len(verts), dtype=np.int64).reshape(-1, 3)
