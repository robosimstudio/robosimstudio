# SPDX-License-Identifier: Apache-2.0
"""Drawing the deformables so they read as fluffy cloth rather than as film.

Newton's viser viewer draws every triangle mesh in the model as one node at one
grey -- ``Viewer._log_triangles`` passes no colour -- and for a membrane that
node is *literally* a film: a zero-thickness surface whose silhouette is a
line, whose edge is invisible from every angle, and whose colour is the same as
the bench it lies on. No lighting rig fixes that, because the problem is not
the light; it is that the geometry being drawn is not the geometry of a towel.

What makes real cloth look soft is specific and reproducible: it has **loft**
(a body thicker than its mid-surface), a **silhouette that fades** instead of
cutting (pile scatters the edge), and a **grain** (thousands of slightly
different tufts, where a uniform surface reads as vinyl). Each of those is
built here from the live mesh, every frame, out of primitives the viewer
already has -- so the solver's cloth is untouched and there is nothing to
train, load or bake:

* **A core shell.** The fabric's real thickness as solid geometry: a membrane
  is offset to both sides of its live normals and closed with a rim
  (:func:`~robosimstudio.core.shell.shell_faces`); a slab already carries its
  two faces and rim in the model's triangles. Opaque, so it occludes, casts
  and receives shadows like the object it is standing in for.
* **Fur shells.** The classic real-time pile trick: the same closed surface
  regrown slightly outward a couple of times and drawn translucent, tips
  lighter than roots. At the silhouette the layers stack into a soft gradient,
  which is what "fluffy" mostly *is* at bench distance.
* **Fuzz.** A fixed area-weighted sprinkle of soft round points through the
  pile depth -- a dense fine undercoat and a sparser layer of taller tufts,
  root-heavy the way real pile is, colour-jittered so the surface has grain.
  viser's points are unlit, so a Lambert term against the studio key is baked
  into their colours every frame -- that is what keeps the fuzz shading with
  the folds instead of glowing through them.
* **Weave.** A fixed, seeded sub-millimetre relief along the live normals.
  A mathematically flat surface under smooth shading reads as injection-
  moulded plastic whatever is sprinkled on it; looped terry is not flat at
  exactly the scale a raking key light shows.
* **Bands.** A cloth whose spec declares ``decor="towel"`` gets the flat woven
  border bands a bath towel has near its two ends -- same yarn, slightly
  deeper tone, no pile -- which is the single strongest "this is a towel" cue
  a silhouette cannot give.

The styles stay a live control rather than a config field, because
"presentable" is a judgement made by looking: **fluffy** is the above,
**shell** is the core alone (the honest solid, and what an export renders),
**flat** is exactly what the viewer always drew. Every deformable that is
*not* a declared garment keeps the flat draw too -- the tape suite's film is a
film, and drawing it thick or furry would be a lie about the physics. What a
film does honour is its spec's ``opacity``: a polymer film is a thing you see
the substrate through, and drawing it as solid white was the whole reason the
peel suites read as paper.

A **centerline** -- a rope on an engine that solves it as a curve, whose
particles are its nodes and which owns no triangle at all -- is drawn as what
its handle says it is: a tube of
:attr:`~robosimstudio.scene.handles.DeformableHandle.tube_radius` swept along
the nodes, the same tube the path tracer draws (one mesh, its faces fixed, its
vertices rewritten each frame). Never fluffy: a cable has no pile.

What decides thickness is the material, not this module:
``ClothMaterialSpec.thickness`` is real fabric data (5 mm terry, 3 mm fleece)
that a membrane records even though its solver geometry cannot carry it. A
material that records none -- the tape film -- draws at zero and stays a film,
which is correct.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import newton
import numpy as np

from robosimstudio.core.profile import NULL as NULL_PROFILE
from robosimstudio.core.render.finish import DEFAULT as DEFAULT_FINISH
from robosimstudio.core.render.uv import box_projected_uvs
from robosimstudio.core.render.visual import TUBE_SIDES, tube_points, tube_triangles
from robosimstudio.core.shell import shell_faces
from robosimstudio.sim.draw import PAGE_TITLE, title_page

from .finish_gltf import clear_cache as clear_glb_cache
from .finish_gltf import glb_for, group_by_material, unweld, wants_pbr

__all__ = [
    "CLOTH_COLOR",
    "GARMENT_STYLES",
    "GarmentVisual",
    "ViewerStudio",
    "garment_visuals",
    "scene_repaints_live",
    "viewer_studio",
]

CLOTH_COLOR = (232, 226, 214)
"""Off-white, warm, for deformables nothing declared a colour for. 0-255.

The viewer draws every triangle mesh in the model at ``(180, 180, 180)``, the
same grey it draws the bench and the robot in, so an undeclared cloth is
invisible against its own table until something separates them.
"""

GARMENT_STYLES = ("fluffy", "shell", "flat")
"""How the garments may be drawn, best-dressed first. See the module docstring."""

MIN_VISUAL_THICKNESS = 0.001
"""Below this [m], a membrane draws as the film it is.

A shell thinner than a millimetre is sub-pixel at bench distance: all it buys
is double the vertices. The tape film records no thickness at all and a
garment records millimetres, so in practice this separates exactly those two.
"""

FUR_LAYERS = ((0.55, 0.20), (1.0, 0.10))
"""``(pile fraction, opacity)`` per fur shell, root to tip.

Two layers, not five: each is a full translucent copy of the cloth, and
translucency in three.js is blended in object order rather than resolved per
triangle, so every extra layer buys less softness and more sorting shimmer on
a folded sheet. Two reads as pile; one reads as a halo.
"""

FUR_TIP_LIGHTEN = 0.35
"""How far tip colour moves toward white, at full pile height.

Pile tips catch light that roots do not -- the reason velvet and towelling
have that bright rim. Applied per layer by its pile fraction, and to the fuzz
by each point's own height.
"""

FUZZ_PER_M2 = 320_000.0
FUZZ_CAP = 48_000
"""Fuzz density and its ceiling. A third of a million per square metre puts a
point every ~1.8 mm, which at the undercoat's point size closes into nap
rather than polka dots; the cap keeps a big garment from swamping the
websocket -- beyond it the points thin evenly rather than clustering. 48k is
sized so the 30 cm towel (0.09 m^2, both faces) keeps nearly its full
density: at the old 36k it visibly thinned into speckle."""

FUZZ_TUFT_FRACTION = 0.28
"""How much of the sprinkle is taller, larger tufts rather than undercoat.

One point size was the polka-dot look: identical dots at identical spacing
read as a pattern, not a pile. Two populations -- a fine dense undercoat
hugging the surface and sparse bigger tufts standing through it -- is the
minimum that reads as nap.
"""

FUZZ_POINT_SIZE = 0.0017
"""Undercoat point diameter [m]. Sized just above the sprinkle's ~1.8 mm
spacing so neighbouring points fuse into nap instead of reading as dots."""

FUZZ_TUFT_SIZE = 0.0024
"""Tuft point diameter [m]. About a real terry loop."""

_SHADE_AMBIENT = 0.52
_SHADE_KEY = 0.50
_SHADE_BOUNCE = 0.14
"""The Lambert rig baked into fuzz colours. Ambient keeps fold interiors
legible, the key term shades by orientation, the bounce lifts faces pointing
away from the key so an underside is dim rather than dead. Close to the
lighting rig's own balance -- the fuzz has to sit in the same room as the
meshes, which *are* lit by that rig."""

_TONE_JITTER = 0.13
"""Per-point colour variation, fixed at build. The difference between fabric
and plastic at this scale *is* variation."""

WEAVE_RATIO = 0.09
WEAVE_MIN, WEAVE_MAX = 0.00025, 0.0006
"""Weave relief amplitude as a fraction of fabric thickness, clamped [m].

0.45 mm on the 5 mm towel, 0.27 mm on the 3 mm fleece: enough that smooth
shading picks up an undulation and the silhouette stops being a ruled line,
small enough that the drawn cloth never strays a visible distance from where
the solver says it is. Seeded per garment and fixed, like the fuzz -- it is
the same towel every reload, and only the physics moves it.
"""

BAND_START, BAND_END = 0.12, 0.19
"""Where a ``decor="towel"`` border band sits, as fractions of the sheet's
length from each end. On the 30 cm towel that is a 2 cm band about 4 cm in --
where a dobby border actually sits -- and stated as fractions so a resized
grid keeps the proportion rather than the millimetres."""

BAND_TONE = 0.78
"""Band colour as a fraction of the garment's. Self-coloured jacquard: the
same yarn woven flat sits visibly deeper than its pile -- and it has to hold
its own against pile that the fuzz and fur layers actively brighten."""


def _display_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals for the *drawn* cloth, per frame.

    A deliberate double of :func:`~robosimstudio.core.shell.surface_normals`,
    and the duplication is the point. That function is on the build path --
    ``prism_slab`` grows the solver's own slab along it -- so its summation
    order is part of what gets simulated and must not move. This one runs per
    drawn frame on the display side, where the only requirement is that the
    towel looks like the towel, and there ``np.add.at``'s unbuffered scatter is
    just slow: replacing it with per-axis ``np.bincount`` accumulation measured
    2.4x faster at shirt scale (3.6 -> 1.5 ms), different from the core
    function by reordered float addition -- parts in 1e-15, invisible even
    before viser casts the result to float32.

    Same contract otherwise: un-normalised cross products are already
    area-weighted, and a vertex whose normals cancel comes back ``+z`` rather
    than as a division by zero.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    face_normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = len(v)
    normals = np.zeros_like(v)
    for corner in range(3):
        fc = f[:, corner]
        for axis in range(3):
            normals[:, axis] += np.bincount(fc, weights=face_normals[:, axis], minlength=n)
    lengths = np.linalg.norm(normals, axis=1)
    degenerate = lengths < 1e-12
    normals[degenerate] = (0.0, 0.0, 1.0)
    lengths[degenerate] = 1.0
    return normals / lengths[:, None]


@dataclass
class GarmentVisual:
    """One deformable's appearance, resolved from its spec and handles.

    Attributes:
        name: The cloth's scene name.
        particles: Global particle indices, the handle's own.
        triangles: Triangles into the local numbering, the handle's own. For a
            slab this is the full skin -- both faces and the rim.
        color: Base RGB in [0, 1].
        opacity: Alpha in (0, 1]. Below 1 the core draws translucent -- the
            peel suites' polymer films. The pile layers never apply: a fabric
            thick enough to be fluffy is a fabric you cannot see through.
        thickness: The fabric's thickness [m], straight from the material.
            Sizes the visual shell on a membrane and the pile depth on both.
        is_slab: Whether the model geometry is already a closed slab, in which
            case the thickness is in the triangles and never re-synthesised.
        decor: The spec's named detailing; ``"towel"`` grows border bands.
        grid_cells: The spec's ``cells`` for a grid-built sheet, or ``None``.
            What lets the bands be *derived*: rows of a grid are addressable
            where positions on an arbitrary mesh are not.
        crease_axis: Which grid axis the garment folds along, from the asset's
            own ``panel`` declaration. The bands have to run *across* the fold so
            the two of them land on each other when it does. ``1`` when nothing
            says.
        band_drawn: Which of the core topology's faces are band, or ``None``.
        band_mid: The same bands over :attr:`triangles` -- the numbering the
            fuzz samples in -- so the bands can be left bald.
        finish: What the scene says this cloth is made of, as the compiler
            resolved it -- the same table the path tracer binds. ``None`` for a
            caller holding only a compiled scene with no handles, which draws
            the way it always did.
        tube_radius: The handle's own: for a centerline -- ordered nodes and no
            triangles -- the radius it is drawn as a tube of. Then
            :attr:`triangles` become the tube's faces, over
            :data:`~robosimstudio.core.render.visual.TUBE_SIDES` vertices per
            node, and :meth:`surface` sweeps the vertices. Ignored by anything
            that has triangles of its own.
    """

    name: str
    particles: np.ndarray
    triangles: np.ndarray
    color: tuple[float, float, float] = (0.85, 0.85, 0.88)
    opacity: float = 1.0
    thickness: float = 0.0
    is_slab: bool = False
    decor: str = ""
    grid_cells: tuple[int, int] | None = None
    crease_axis: int = 1
    finish: Any = None
    tube_radius: float = 0.0

    is_tube: bool = field(default=False, init=False)
    shell_triangles: np.ndarray | None = field(default=None, repr=False)
    band_drawn: np.ndarray | None = field(default=None, repr=False)
    band_mid: np.ndarray | None = field(default=None, repr=False)
    _fuzz: dict[str, Any] | None = field(default=None, repr=False)
    _weave: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.particles = np.asarray(self.particles, dtype=np.int64).reshape(-1)
        self.triangles = np.asarray(self.triangles, dtype=np.int64).reshape(-1, 3)
        if self.tube_radius > 0.0 and not len(self.triangles) and len(self.particles) >= 2:
            # The path tracer's rule (``rtx._build_deformable_cache``), so the
            # page and a render agree about which deformables are curves.
            self.is_tube = True
            self.triangles = tube_triangles(len(self.particles), TUBE_SIDES).astype(np.int64)
            return
        if not self.is_slab and self.thickness >= MIN_VISUAL_THICKNESS:
            # A surface that cannot be shelled (a non-manifold edge survived
            # into a membrane) falls back to the film rather than to a crash:
            # the studio must load every scene its solver can run.
            try:
                self.shell_triangles = shell_faces(self.triangles, len(self.particles))
            except ValueError as exc:
                print(f"[studio] {self.name}: no visual shell ({exc}); drawing the membrane flat")
        self._setup_bands()

    @property
    def has_body(self) -> bool:
        """Whether there is a thickness to be fluffy about."""
        return self.is_slab or self.shell_triangles is not None

    @property
    def band_color(self) -> tuple[float, float, float]:
        """The bands' RGB: the garment's own yarn, woven flat."""
        return tuple(float(c) * BAND_TONE for c in self.color)

    def _setup_bands(self) -> None:
        """Resolve ``decor`` into face masks, or quietly into nothing.

        Quietly, because decoration must never be able to take a scene down:
        a spec whose ``cells`` do not match the particles it actually got --
        an entity that resampled, a future kind this arithmetic has never
        seen -- draws the plain garment it always drew.
        """
        if self.decor != "towel" or self.grid_cells is None:
            return
        nx, ny = (int(c) for c in self.grid_cells)
        n_mid = len(self.particles) // 2 if self.is_slab else len(self.particles)
        if nx < 1 or ny < 4 or (nx + 1) * (ny + 1) != n_mid:
            return
        # Banded ACROSS the fold, so the two bands land on each other when it
        # lands. Which way that is comes from the garment's own `panel`
        # declaration rather than from an assumption about grid layout.
        # Vertex k sits at row (k mod n_mid) // (nx + 1) and column k % (nx + 1)
        # by grid_surface's layout.
        across, n_across = (
            (np.arange(n_mid) // (nx + 1), ny) if self.crease_axis == 1 else (np.arange(n_mid) % (nx + 1), nx)
        )
        row, ny = across, n_across
        r0 = max(1, int(round(BAND_START * ny)))
        r1 = max(r0 + 1, int(round(BAND_END * ny)))
        on_rows = ((row >= r0) & (row <= r1)) | ((row >= ny - r1) & (row <= ny - r0))
        both_faces = np.concatenate([on_rows, on_rows])
        if self.is_slab:
            # The skin's triangles index bottom-then-top directly; a rim face
            # between band rows has all corners on band rows and wraps the
            # band around the edge, as a woven border does.
            self.band_mid = both_faces[self.triangles].all(axis=1)
            self.band_drawn = self.band_mid
        else:
            self.band_mid = on_rows[self.triangles].all(axis=1)
            self.band_drawn = (
                both_faces[self.shell_triangles].all(axis=1) if self.shell_triangles is not None else self.band_mid
            )
        if self.band_drawn is not None and not self.band_drawn.any():
            self.band_drawn = self.band_mid = None

    def weave_offsets(self) -> np.ndarray:
        """Fixed per-vertex relief along the live normals [m]. Zero on a film.

        White noise smoothed twice over the surface graph: raw per-vertex
        noise glitters, and two passes push it to about a two-cell wavelength,
        which at the towel's 10 mm cell is the scale terry actually undulates
        at. Seeded by name and computed once, like the fuzz -- the physics is
        what moves, the weave is what the fabric is.
        """
        if self._weave is not None:
            return self._weave
        n = len(self.particles)
        if not self.has_body:
            self._weave = np.zeros(n)
            return self._weave
        amplitude = WEAVE_MIN
        if self.thickness > 0.0:
            amplitude = float(np.clip(WEAVE_RATIO * self.thickness, WEAVE_MIN, WEAVE_MAX))
        rng = np.random.default_rng(zlib.crc32(f"{self.name}/weave".encode()))
        noise = rng.uniform(-1.0, 1.0, n)
        tris = self.triangles
        for _ in range(2):
            acc = np.zeros(n)
            count = np.zeros(n)
            for a, b in ((0, 1), (1, 2), (2, 0)):
                np.add.at(acc, tris[:, a], noise[tris[:, b]])
                np.add.at(count, tris[:, a], 1.0)
                np.add.at(acc, tris[:, b], noise[tris[:, a]])
                np.add.at(count, tris[:, b], 1.0)
            count[count == 0.0] = 1.0
            noise = 0.5 * noise + 0.5 * acc / count
        peak = float(np.abs(noise).max())
        self._weave = noise * (amplitude / peak) if peak > 1e-12 else np.zeros(n)
        return self._weave

    @property
    def pile(self) -> float:
        """Pile depth beyond the core surface [m]. Zero on a film."""
        if not self.has_body:
            return 0.0
        return float(np.clip(0.4 * self.thickness, 0.0008, 0.0025)) if self.thickness > 0.0 else 0.0012

    # ------------------------------------------------------------------ frames
    def surface(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Live vertices and outward unit normals of the drawn surface.

        The mid-surface for a membrane, the closed skin for a slab -- in both
        cases the normals every layer of the look is grown along, computed
        once per frame and shared. For a centerline, the tube swept along its
        nodes.
        """
        if self.is_tube:
            return tube_points(positions[self.particles], self.tube_radius, TUBE_SIDES)
        verts = positions[self.particles]
        return verts, _display_normals(verts, self.triangles)

    def core_geometry(self, verts: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The solid body: ``(vertices, faces)`` for one frame."""
        if self.shell_triangles is None:
            return verts, self.triangles
        half = 0.5 * self.thickness
        return np.concatenate([verts - normals * half, verts + normals * half]), self.shell_triangles

    def fur_geometry(self, verts: np.ndarray, normals: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
        """The core regrown ``fraction`` of the pile outward: one fur shell."""
        grow = self.pile * float(fraction)
        if self.shell_triangles is None:
            return verts + normals * grow, self.triangles
        offset = normals * (0.5 * self.thickness + grow)
        return np.concatenate([verts - offset, verts + offset]), self.shell_triangles

    # -------------------------------------------------------------------- fuzz
    def fuzz_arrays(
        self, verts: np.ndarray, normals: np.ndarray, light: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The pile sprinkle for one frame: points, u8 colors, and a tuft mask.

        The sampling -- which faces, where on them, how high in the pile, what
        tone, which points are the taller tufts -- is fixed at first call from
        rest geometry and seeded by name, so the nap is the same towel on
        every reload and only the physics moves it. Per frame the fixed
        samples are just carried by their faces. The mask is what lets the
        caller draw the undercoat fine and the tufts big: one point size was
        the polka-dot look.
        """
        state = self._fuzz
        if state is None:
            state = self._fuzz_setup(verts)
        faces = self.triangles[state["face"]]
        w = state["bary"]  # [n, 3]
        # einsum rather than the broadcast-multiply-and-sum it replaces: the
        # same barycentric blend without materialising the [n, 3, 3] product,
        # measured 3x faster at the 48,000-point cap. Same values -- the sum
        # runs over the same three corners.
        base = np.einsum("ijk,ij->ik", verts[faces], w)
        n_pt = np.einsum("ijk,ij->ik", normals[faces], w)
        lengths = np.linalg.norm(n_pt, axis=1)
        lengths[lengths < 1e-12] = 1.0
        n_pt /= lengths[:, None]
        # A membrane sprouts from both faces of its shell; sign carries which.
        n_pt *= state["side"][:, None]
        lift = state["root"] + state["height"] * self.pile
        points = base + n_pt * lift[:, None]

        # Unlit points, so the lighting is baked: see the module docstring.
        toward = n_pt @ np.asarray(light, dtype=np.float64)
        shade = _SHADE_AMBIENT + _SHADE_KEY * np.maximum(toward, 0.0) + _SHADE_BOUNCE * np.maximum(-toward, 0.0)
        rgb = np.asarray(self.color)[None, :] * (shade * state["tone"])[:, None]
        rgb += (1.0 - rgb) * (FUR_TIP_LIGHTEN * state["height"])[:, None]
        return points, np.clip(rgb * 255.0, 0, 255).astype(np.uint8), state["tuft"]

    def _fuzz_setup(self, rest_verts: np.ndarray) -> dict[str, Any]:
        tris = self.triangles
        cross = np.cross(
            rest_verts[tris[:, 1]] - rest_verts[tris[:, 0]],
            rest_verts[tris[:, 2]] - rest_verts[tris[:, 0]],
        )
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        if self.band_mid is not None:
            # A woven border has no pile: bald bands are what make them read
            # as bands rather than as a colour glitch under the nap.
            areas = np.where(self.band_mid, 0.0, areas)
        total = float(areas.sum())
        count = int(min(total * FUZZ_PER_M2, FUZZ_CAP))
        rng = np.random.default_rng(zlib.crc32(self.name.encode()))
        face = rng.choice(len(tris), size=count, p=areas / total) if total > 0.0 else np.zeros(0, dtype=np.int64)
        # Uniform barycentric: fold the unit square's far triangle back in.
        u, v = rng.uniform(size=(2, count))
        flip = u + v > 1.0
        u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
        # Real pile is root-heavy: most of what you see into a towel is short
        # fibre near the backing, with sparser loops standing through it. The
        # undercoat squares a uniform draw towards the root; the tufts get the
        # upper half so they genuinely stand proud.
        tuft = rng.uniform(size=count) < FUZZ_TUFT_FRACTION
        depth = rng.uniform(size=count)
        height = np.where(tuft, 0.35 + 0.65 * rng.uniform(size=count), 0.06 + 0.94 * depth * depth)
        state = {
            "face": face,
            "bary": np.stack([1.0 - u - v, u, v], axis=1),
            # A slab's skin already faces both ways; a membrane needs half its
            # tufts on the underside.
            "side": np.ones(count) if self.is_slab else rng.choice((-1.0, 1.0), size=count),
            "root": np.full(count, 0.0 if self.is_slab else 0.5 * self.thickness),
            "height": height,
            "tuft": tuft,
            "tone": 1.0 + _TONE_JITTER * rng.uniform(-1.0, 1.0, count),
        }
        self._fuzz = state
        return state


def _declared_crease_axis(doc, name: str) -> int:
    """Which axis a garment folds along, from the asset that declares it.

    Off the document's type for this instance, so it works the way every other
    declaration read in the studio does. ``1`` when there is no document, no
    record, or no ``panel`` feature, so a garment nobody has declared a crease
    for draws exactly what it drew.
    """
    if doc is None:
        return 1
    from robosimstudio.domains.cloth import crease_of

    from robosimstudio.assetlib.registry import ASSETS

    types = {o.name: o.type for o in doc.objects}
    try:
        record = ASSETS.get(types[name])
    except KeyError:
        return 1
    panels = record.features_of_kind("panel")
    return crease_of(panels[0].payload)[0] if panels else 1


def garment_visuals(scene, doc=None) -> list[GarmentVisual]:
    """One :class:`GarmentVisual` per declared cloth of a compiled scene.

    Reads the same spec fields the offline visual backends read --
    ``ClothSpec.color``, the material's shell kind and thickness -- so the
    studio and an exported video disagree about resolution, never about what
    the towel is.

    ``doc`` is the task document, when there is one, and supplies the one fact a
    spec cannot: which axis a garment creases along, which the border bands have
    to run across. Optional, because a caller holding only a compiled scene is a
    legitimate caller and gets the behaviour this had before it asked.
    """
    visuals: list[GarmentVisual] = []
    for name, handle in scene.handles.deformables.items():
        entity = scene.by_name.get(name)
        spec = getattr(entity, "spec", None)
        material = getattr(spec, "material", None)
        cells = getattr(spec, "cells", None) if getattr(spec, "mesh", "") == "grid" else None
        visuals.append(
            GarmentVisual(
                name=name,
                particles=handle.particles,
                triangles=handle.triangles,
                color=tuple(float(v) for v in getattr(spec, "color", (0.85, 0.85, 0.88))),
                opacity=float(getattr(spec, "opacity", 1.0)),
                # The compiler's resolved table rather than `spec.finish`: it
                # fills one in for every deformable, stated or not, by role.
                finish=getattr(scene.handles, "deformable_finishes", {}).get(name),
                thickness=float(getattr(material, "thickness", 0.0)),
                is_slab=bool(getattr(material, "is_slab", False)),
                decor=str(getattr(spec, "decor", "")),
                grid_cells=tuple(int(c) for c in cells) if cells is not None else None,
                crease_axis=_declared_crease_axis(doc, name),
                tube_radius=float(getattr(handle, "tube_radius", 0.0) or 0.0),
            )
        )
    return visuals


def scene_repaints_live(scene) -> bool:
    """Whether any entity of a compiled scene writes ``shape_color`` as it runs.

    Read the same way the garments are: off the compiled scene, at the moment
    the model is handed to the viewer. What it answers is
    :meth:`ViewerStudio.set_live_colors`, and the reason it is asked here rather
    than inside the viewer is that the viewer is handed a ``Model``, which has
    lost every trace of which entity wrote what into it.

    Tolerant of a caller holding no scene -- a bare ``Sim`` assembled by hand, a
    fleet mid-swap -- because the answer for a scene nobody can inspect is the
    behaviour this had before it was asked.
    """
    entities = getattr(scene, "entities", None) or ()
    return any(bool(getattr(entity, "recolors_live", False)) for entity in entities)


class ViewerStudio(newton.viewer.ViewerViser):
    """``ViewerViser`` that draws declared garments as garments.

    Everything else about the viewer is inherited. The coupling to Newton is
    one private method, ``_log_triangles`` -- if it changes shape upstream this
    override simply stops being called and the cloth goes back to grey rather
    than breaking, which for a display concern is the right failure.

    The garment scene nodes are owned here, not in the base class's handle
    table, so ``set_model`` clearing the model's nodes cannot half-remove
    them: :meth:`clear_model` retires the garments wholesale and the session
    re-declares them for the scene it loads next.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._garments: list[GarmentVisual] = []
        self._sheets: list[GarmentVisual] = []
        self._tubes: list[GarmentVisual] = []
        self._garment_style: str = "fluffy"
        self._garment_nodes: dict[str, Any] = {}
        self._garment_hidden: dict[str, bool] = {}
        self._leftover_faces: np.ndarray | None = None
        self._garment_light = np.array([2.2, -1.7, 1.9]) / np.linalg.norm([2.2, -1.7, 1.9])
        self.profile = NULL_PROFILE
        """Set by the session while it is measuring; disabled otherwise.

        Assigned before each draw rather than at construction, because the
        viewer outlives the session's decision to profile -- a page can be
        measured for a window and then left running.
        """
        self._shape_colors_dirty = False
        self._live_colors = False
        self._particles_were_shown = False
        self._com_was_shown = False
        self._static_batches: set[str] = set()
        # The scene's own materials, handed over by the session. Empty means the
        # page draws what it always drew: Newton's flat per-instance colour.
        self._finishes: dict[int, Any] = {}
        self._live_shapes: frozenset[int] = frozenset()
        self._pbr_handles: dict[str, dict[tuple, Any]] = {}
        self._pbr_groups: dict[str, tuple] = {}
        self._host_shape_color: np.ndarray | None = None
        """``model.shape_color`` read back once per repack, for the PBR batches."""
        self._static_guard: tuple | None = None
        self._static_skip_ok = False
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------ what to draw
    def set_garments(self, garments: Sequence[GarmentVisual]) -> None:
        """Adopt the scene's declared deformables. Call after ``set_model``.

        Whatever deformable triangles the model carries beyond the declared
        garments -- the tape film, or everything when nothing was declared --
        keep the viewer's own single-mesh draw, so a scene this module has
        never heard of looks exactly as it always did. A centerline is drawn
        as its tube in every style: it has no triangles for the flat draw to
        fall back on.
        """
        self._retire_garment_nodes()
        self._garments = list(garments)
        self._sheets = [g for g in self._garments if not g.is_tube]
        self._tubes = [g for g in self._garments if g.is_tube]
        self._leftover_faces = None
        if self.model is None or not self.model.tri_count or not self._garments:
            return
        tris = self.model.tri_indices.numpy().reshape(-1, 3)
        owned = np.zeros(len(tris), dtype=bool)
        for garment in self._garments:
            owned |= np.isin(tris, garment.particles).all(axis=1)
        leftover = tris[~owned]
        self._leftover_faces = leftover if len(leftover) else np.zeros((0, 3), dtype=np.int64)

    def set_garment_style(self, style: str) -> None:
        """Switch how the garments are drawn. One of :data:`GARMENT_STYLES`."""
        if style not in GARMENT_STYLES:
            raise ValueError(f"garment style must be one of {GARMENT_STYLES}, got {style!r}")
        if style != self._garment_style:
            self._garment_style = style
            self._retire_garment_nodes()

    def set_garment_light(self, position: Sequence[float]) -> None:
        """Where the key light sits, for the shading baked into the fuzz.

        The fuzz points are unlit geometry, so when the lighting panel moves
        the key the baked term has to follow or the pile stays lit from a room
        that no longer exists.
        """
        direction = np.asarray(position, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            self._garment_light = direction / norm

    @property
    def garment_style(self) -> str:
        return self._garment_style

    # -------------------------------------------------------------- lifecycle
    def clear_model(self):
        """Reset model-dependent state; the garments are that state too."""
        self._retire_pbr_nodes()
        clear_glb_cache()
        self._retire_garment_nodes()
        self._garments, self._sheets, self._tubes = [], [], []
        self._leftover_faces = None
        self._host_shape_color = None
        super().clear_model()

    def _retire_garment_nodes(self) -> None:
        # Children first: viser removes a node's whole subtree, so taking the
        # garment's root down before its /band or /tufts leaves those handles
        # pointing at nodes that are already gone, and each .remove() warns.
        for name in sorted(self._garment_nodes, key=lambda n: n.count("/"), reverse=True):
            try:
                self._garment_nodes[name].remove()
            except Exception:
                pass
        self._garment_nodes = {}
        self._garment_hidden = {}

    def _retire_flat_node(self) -> None:
        """Take down the base viewer's single-mesh node when garments draw.

        ``log_mesh`` leaves its last frame in the scene; switching styles
        without this would freeze that frame under the garments forever.
        """
        handle = self._scene_handles.pop(self._qualify("/model/triangles"), None)
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass

    # -------------------------------------------------------- draw economies
    # Everything in this block trims the per-frame draw without changing what
    # a session can show. Measured before it was written: on a tool-sorting scene
    # (107 shape batches) `log_state` made 212 device readbacks per drawn frame
    # -- exactly two per visible batch -- and on a 12,872-particle scene the
    # particle path ran a mask, a scan and two readbacks per frame to build a
    # point cloud whose show flag is off and has no control in this page.

    def set_model(self, model) -> None:
        super().set_model(model)
        if model is not None:
            self._static_batches = {b.name for b in self._shape_instances.values() if b.static}
        else:
            self._static_batches = set()
        self._static_guard = self._layer_pose()
        self._host_shape_color = None
        # A new model has to re-declare it: inheriting the last scene's answer
        # would leave a page that once showed the polish strip repacking colours
        # for every scene it is asked for afterwards.
        self._live_colors = False

    def begin_frame(self, time: float) -> None:
        # The static-batch skip below assumes the layer transform has not moved
        # since a static batch was last pushed; checked once per frame here
        # rather than per batch. Nothing in this stack moves it today, so the
        # check is seven float compares buying insurance against the day
        # something does: that frame re-pushes everything, then skipping
        # resumes.
        pose = self._layer_pose()
        self._static_skip_ok = pose == self._static_guard
        self._static_guard = pose
        super().begin_frame(time)

    @staticmethod
    def _to_numpy(x) -> np.ndarray | None:
        """A Warp array, or anything array-like, as numpy; ``None`` stays ``None``.

        Newton before 1.6 had this on the viewer class and 1.6 moved it to a
        module function; the page draws under both, so it keeps its own.
        """
        if x is None:
            return None
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    def _layer_pose(self) -> tuple:
        return tuple(float(v) for v in self.layer.xform)

    def _sync_shape_colors_from_model(self) -> None:
        """Repack shape colours only when the model (or a caller) says they moved.

        The base method launches its repack kernel every frame and ends by
        flagging every batch's colours changed -- which makes the viser backend
        read the colour array back and convert it to uint8 per batch per frame,
        the second of the two readbacks counted above. Its own docstring calls
        the kernel cheaper than a host-side comparison every frame; in this
        backend the flag it sets *causes* a host copy per batch, so the framing
        is inverted. In this stack every writer of ``shape_color`` is a builder
        -- the palette, the static overrides, entity tinting -- all before
        ``finalize``, so repacking once per model draws the same picture. A
        control that ever recolours a live model must call
        :meth:`mark_shape_colors_dirty`; a *scene* that does must be declared
        through :meth:`set_live_colors`, since it repaints every frame and has
        nobody to call a one-shot.
        """
        if self.model_changed or self._live_colors or self._shape_colors_dirty:
            self._shape_colors_dirty = False
            self._host_shape_color = None
            super()._sync_shape_colors_from_model()

    def mark_shape_colors_dirty(self) -> None:
        """Make the next frame repack and re-push shape colours."""
        self._shape_colors_dirty = True
        self._host_shape_color = None

    def set_finishes(self, finishes: Mapping[int, Any], live_shapes: Sequence[int] = ()) -> None:
        """Adopt the scene's per-shape materials, which turns PBR drawing on.

        Without these the page has nothing to shade with and draws as it always
        did. ``live_shapes`` are the ones a task recolours every step -- a stain
        being wiped off, a seam parting -- and they stay on the flat per-instance
        path, because a GLB carries one baked colour and re-encoding one per
        frame is not a trade worth making. The path tracer keeps them separate
        for the same reason.
        """
        # `or ()` would truth-test the argument, and `live_shapes` arrives as a
        # numpy array -- "the truth value of an array with more than one element
        # is ambiguous", raised from inside a defaulting idiom that looks total.
        self._finishes = dict(finishes) if finishes is not None else {}
        self._live_shapes = frozenset(int(s) for s in ([] if live_shapes is None else live_shapes))
        self._retire_pbr_nodes()

    def _retire_pbr_nodes(self, name: str | None = None) -> None:
        """Take down the instanced GLB batches, all of them or one batch's."""
        names = list(self._pbr_handles) if name is None else [name]
        for key in names:
            for handle in self._pbr_handles.pop(key, {}).values():
                try:
                    handle.remove()
                except Exception:
                    pass
            self._pbr_groups.pop(key, None)

    def set_live_colors(self, live: bool) -> None:
        """Declare that the scene now loaded repaints itself as it runs.

        Reinstates the base class's every-frame repack for this model only. One
        bool for every scene that does not -- which is all but two of them --
        and it is answered once at model swap rather than per frame, because
        what it is really asking is whether any entity in the scene declared
        :attr:`~robosimstudio.scene.entity.Entity.recolors_live`, and that
        cannot change without a new model.

        Without it a polished mark or a parted tape seam fades in the offline
        renderers and on the GL page and stays frozen here, on the one page a
        person is actually watching it on.
        """
        self._live_colors = bool(live)

    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False, opacities=None):
        # A static batch's world transform cannot have changed since it was
        # pushed: `update_shape_xforms` composes a constant shape transform
        # with the layer pose, and the guard in `begin_frame` covers the layer.
        # Skipping the push saves the readback, the numpy conversion and the
        # websocket message. Only when the handle already exists -- a hidden
        # spell removes it, and the recreate must go through -- and only when
        # no colours, materials or opacities ride along, which is the base
        # class's own "something else changed" signal.
        if (
            not hidden
            and colors is None
            and materials is None
            and opacities is None
            and self._static_skip_ok
            and name in self._static_batches
            # Either path's nodes: the PBR one pops the flat handle when it draws.
            and (self._qualify(name) in self._scene_handles or self._pbr_handles.get(self._qualify(name)))
        ):
            return
        if self._finishes and not hidden and self._log_pbr_instances(name, mesh, xforms, scales):
            return
        # Whatever the PBR path could not draw, the flat one still can -- and if
        # it drew this batch last frame, its nodes have to come down first or
        # the page shows both.
        self._retire_pbr_nodes(self._qualify(name))
        # Newton 1.6 hands a batch its per-instance opacities; the viewer before
        # it takes no such argument, so it is passed on only when it was given.
        more = {} if opacities is None else {"opacities": opacities}
        super().log_instances(name, mesh, xforms, scales, colors, materials, hidden=hidden, **more)

    # ------------------------------------------------------------------- PBR
    def _log_pbr_instances(self, name, mesh, xforms, scales) -> bool:
        """Draw one batch as instanced GLBs, one per material in it.

        Newton batches by geometry, so a batch is one shape in however many
        colours and finishes the scene gave it; a GLB carries a single material,
        so the batch is cut again along the material key and each group is
        pushed with ``add_batched_glb``. Every step that could fail -- no
        prototype, no shape table, an encoder that will not encode -- returns
        ``False`` so the caller falls back to the flat path. A duller picture is
        a much better failure than an empty one.

        Returns:
            Whether the batch was drawn here.
        """
        qualified, prototype = self._qualify(name), self._qualify(mesh)
        data = self._meshes.get(prototype)
        shapes = self._batch_shapes(qualified)
        if data is None or shapes is None or xforms is None or prototype in self._plane_meshes:
            return False
        # A shape whose colour is rewritten every step cannot be baked into a
        # GLB, so the whole batch goes back to the flat path if it holds one.
        if self._live_shapes and any(int(s) in self._live_shapes for s in shapes):
            return False
        xforms_np = self._to_numpy(xforms)
        if xforms_np is None or len(xforms_np) != len(shapes):
            return False
        # Read back once per repack rather than once per batch per frame: the
        # colours only move when _sync_shape_colors_from_model repacks them.
        if self._host_shape_color is None and getattr(self.model, "shape_color", None) is not None:
            self._host_shape_color = self.model.shape_color.numpy()
        colors = self._host_shape_color
        groups = group_by_material(shapes, colors, self._finishes, DEFAULT_FINISH)

        positions = xforms_np[:, :3].astype(np.float32)
        quats = self._quats_xyzw_to_wxyz(xforms_np[:, 3:7])
        scales_np = self._to_numpy(scales)
        signature = tuple((key[0], tuple(rows)) for key, rows in groups.items())
        handles = self._pbr_handles.setdefault(qualified, {})
        reuse = self._pbr_groups.get(qualified) == signature and len(handles) == len(groups)

        for slot, ((_key, color, finish), rows) in enumerate(groups.items()):
            index = np.asarray(rows, dtype=np.int64)
            node = f"{qualified}/m{slot}"
            batched = (
                positions[index],
                quats[index],
                np.ones((len(index), 3), dtype=np.float32)
                if scales_np is None
                else scales_np[index].astype(np.float32),
            )
            if reuse and node in handles:
                handle = handles[node]
                try:
                    handle.batched_positions, handle.batched_wxyzs = batched[0], batched[1]
                    handle.batched_scales = batched[2]
                    continue
                except Exception:
                    try:
                        handle.remove()
                    except Exception:
                        pass
                    handles.pop(node, None)
            blob = self._glb(data, color, finish)
            if blob is None:
                self._retire_pbr_nodes(qualified)
                return False
            try:
                handles[node] = self._server.scene.add_batched_glb(
                    node,
                    glb_data=blob,
                    batched_positions=batched[0],
                    batched_wxyzs=batched[1],
                    batched_scales=batched[2],
                    lod="off",
                )
            except Exception:
                self._retire_pbr_nodes(qualified)
                return False
        # A batch that lost a material leaves a node behind; drop the strays.
        for stale in [n for n in handles if n not in {f"{qualified}/m{i}" for i in range(len(groups))}]:
            try:
                handles.pop(stale).remove()
            except Exception:
                handles.pop(stale, None)
        self._pbr_groups[qualified] = signature
        # The flat node for this batch, if it ever drew one, is now a second
        # copy of the same shapes.
        flat = self._scene_handles.pop(qualified, None)
        if flat is not None:
            try:
                flat.remove()
            except Exception:
                pass
        return True

    def _batch_shapes(self, qualified: str):
        """The model shape behind each instance of a batch, by its node name.

        ``ShapeInstances`` records it as ``model_shapes`` and nothing else
        carries it: ``log_instances`` is handed arrays, not indices, so without
        this the page cannot ask what any instance is made of.
        """
        for batch in self._shape_instances.values():
            if batch.name == qualified:
                return list(batch.model_shapes)
        return None

    def _glb(self, data: dict, color, finish) -> bytes | None:
        """This prototype, in this material, as GLB bytes.

        UVs are only paid for when the finish has a texture to put on them: the
        mesh has to be unwelded first, since a box corner needs three different
        projections and a vertex can carry one.

        ``smooth`` because glTF flat-shades a mesh that carries no normals, and
        this path was handing it none -- so every prop drew faceted, which is
        invisible on a room of boxes and wrong on anything turned. It is safe on
        the boxes too: they arrive unwelded (24 vertices to 12 triangles), where
        a derived vertex normal *is* the face normal and nothing changes.
        Measured against the tracer on all four of the wiping scene's cameras it is
        better every time -- 12.11 to 11.89, 11.00 to 10.84, 12.78 to 12.66,
        25.27 to 25.18 -- and it is free at runtime, because this path is cached
        per prototype and pays its ~0.45 ms per 640 vertices once at scene load.
        """
        points, faces = data["points"], data["indices"]
        uvs = None
        if finish.texture:
            points, faces = unweld(points, faces)
            uvs = box_projected_uvs(points, faces)
        return glb_for(points, faces, color, finish, uvs=uvs, smooth=True)

    def _log_particles(self, state) -> None:
        # The base method does its masking, scan, compaction and two readbacks
        # before the show flag is consulted -- the flag only decides `hidden`.
        # Skip all of it unless the points will be seen; if they ever were,
        # one more base call pushes the hide.
        if self.show_particles and not self._layer_force_hidden():
            self._particles_were_shown = True
            super()._log_particles(state)
        elif self._particles_were_shown:
            self._particles_were_shown = False
            super()._log_particles(state)

    def _log_com(self, state) -> None:
        # Same shape as _log_particles: a kernel over every body per frame for
        # a marker that defaults hidden and has no control in this page.
        if self.show_com and not self._layer_force_hidden():
            self._com_was_shown = True
            super()._log_com(state)
        elif self._com_was_shown:
            self._com_was_shown = False
            super()._log_com(state)

    # ------------------------------------------------------------------- draw
    def _log_triangles(self, state) -> None:
        model = self.model
        if model is None or not (model.tri_count or self._tubes):
            return
        hidden = not self.show_triangles or self._layer_force_hidden()
        # The sheets on the viewer's own single mesh: nothing declared, or asked for.
        flat = not self._sheets or self._garment_style == "flat"
        if flat and model.tri_count:
            self._log_triangles_flat(state, hidden)
        if flat and not self._tubes:
            return

        if not flat:
            self._retire_flat_node()
        with self.profile.phase("iter.draw.garments"):
            with self.profile.phase("iter.draw.garments.readback"):
                points = self._to_numpy(self._apply_layer_transform_to_points(state.particle_q)).astype(np.float64)
            for garment in self._tubes if flat else self._garments:
                self._draw_garment(garment, points, hidden)
            if not flat and self._leftover_faces is not None and len(self._leftover_faces):
                self._draw_mesh_node("/model/garments/other", points, self._leftover_faces, CLOTH_COLOR, hidden)

    def _log_triangles_flat(self, state, hidden: bool) -> None:
        """The viewer's own draw, tinted: one node over every deformable triangle."""
        self.log_mesh(
            self._qualify("/model/triangles"),
            self._apply_layer_transform_to_points(state.particle_q),
            self.model.tri_indices.flatten(),
            hidden=hidden,
            # Cloth is a surface seen from both sides -- and a slab's rim is a
            # surface seen edge-on, which is the thickness itself.
            backface_culling=False,
            color=CLOTH_COLOR,
        )

    def _draw_garment(self, garment: GarmentVisual, points: np.ndarray, hidden: bool) -> None:
        base = f"/model/garments/{garment.name}"
        fluffy = self._garment_style == "fluffy" and garment.has_body
        if hidden:
            for suffix in ("", "/band", "/fur0", "/fur1", "/fuzz", "/tufts"):
                key = self._qualify(base + suffix)
                self._set_hidden(key, self._garment_nodes.get(key), True)
            return

        prof = self.profile
        with prof.phase("iter.draw.garments.normals"):
            verts, normals = garment.surface(points)
        if fluffy:
            # The weave rides the live normals, so it follows the sheet
            # through folds; sub-millimetre, so the drawn cloth never visibly
            # leaves the solver's. Fluffy only -- "shell" stays the honest
            # geometry an export renders.
            with prof.phase("iter.draw.garments.weave"):
                verts = verts + normals * garment.weave_offsets()[:, None]
        color = tuple(int(round(255 * c)) for c in garment.color)
        translucent = None if garment.opacity >= 1.0 else float(garment.opacity)
        with prof.phase("iter.draw.garments.core"):
            core_verts, core_faces = garment.core_geometry(verts, normals)
            finish = garment.finish
            if garment.band_drawn is None:
                self._draw_mesh_node(base, core_verts, core_faces, color, hidden, opacity=translucent, finish=finish)
            else:
                band_color = tuple(int(round(255 * c)) for c in garment.band_color)
                self._draw_mesh_node(
                    base, core_verts, core_faces[~garment.band_drawn], color, hidden, opacity=translucent, finish=finish
                )
                self._draw_mesh_node(
                    f"{base}/band",
                    core_verts,
                    core_faces[garment.band_drawn],
                    band_color,
                    hidden,
                    opacity=translucent,
                    finish=finish,
                )
        if not fluffy:
            return

        rgb = np.asarray(garment.color, dtype=np.float64)
        with prof.phase("iter.draw.garments.fur"):
            for i, (fraction, opacity) in enumerate(FUR_LAYERS):
                fur_verts, fur_faces = garment.fur_geometry(verts, normals, fraction)
                tip = rgb + (1.0 - rgb) * (FUR_TIP_LIGHTEN * fraction)
                self._draw_mesh_node(
                    f"{base}/fur{i}",
                    fur_verts,
                    fur_faces,
                    tuple(int(round(255 * c)) for c in tip),
                    hidden,
                    opacity=float(opacity),
                )

        with prof.phase("iter.draw.garments.fuzz"):
            fuzz_points, fuzz_colors, tufts = garment.fuzz_arrays(verts, normals, self._garment_light)
        with prof.phase("iter.draw.garments.push"):
            self._draw_fuzz_node(f"{base}/fuzz", fuzz_points[~tufts], fuzz_colors[~tufts], hidden, size=FUZZ_POINT_SIZE)
            self._draw_fuzz_node(f"{base}/tufts", fuzz_points[tufts], fuzz_colors[tufts], hidden, size=FUZZ_TUFT_SIZE)

    def _draw_mesh_node(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: tuple[int, int, int],
        hidden: bool,
        *,
        opacity: float | None = None,
        finish: Any = None,
    ) -> None:
        """Draw one garment surface, on whichever of the two paths fits it.

        ``add_mesh_simple`` carries a colour and an opacity and drops everything
        else a :class:`Finish` states, so a cloth the scene calls glossy draws
        the same as one it calls matte. The GLB path carries the lot -- but a
        GLB handle has no vertex setter, so a deforming mesh has to be re-encoded
        and re-added every frame, where the flat path writes vertices in place.

        Measured on the sponge core: 0.05 ms in place against 0.44 ms encoded
        and pushed, 24 KiB a frame on the wire, and no flicker across repeated
        swaps (spread 0.03 against a 33.38 drawn-versus-hidden gap in the same
        mask). Affordable, but not free and not needed by a matte sheet, so
        :func:`~robosimstudio_studio.finish_gltf.wants_pbr` decides and a
        garment nobody dressed keeps the cheap path.

        The pile layers never come here with a finish: fur and fuzz are unlit
        geometry with a Lambert term already baked into their colours, and
        shading them a second time would light them twice.
        """
        name = self._qualify(name)
        if wants_pbr(finish):
            if self._draw_pbr_mesh(name, vertices, faces, color, finish, opacity, hidden):
                return
        handle = self._garment_nodes.get(name)
        if handle is not None:
            try:
                if not hidden:
                    handle.vertices = vertices.astype(np.float32)
                self._set_hidden(name, handle, hidden)
                return
            except Exception:
                handle = None
        handle = self._server.scene.add_mesh_simple(
            name,
            vertices=vertices.astype(np.float32),
            faces=np.asarray(faces, dtype=np.uint32),
            color=color,
            opacity=opacity,
            flat_shading=False,
            side="double",
            # The translucent fur neither casts nor receives: a shadow of the
            # pile would double the core's, and a shadowed transparency is a
            # grey veil over the very silhouette the fur exists to soften.
            cast_shadow=opacity is None,
            receive_shadow=opacity is None,
            visible=not hidden,
        )
        self._garment_nodes[name] = handle
        self._garment_hidden[name] = hidden

    def _draw_pbr_mesh(self, name, vertices, faces, color, finish, opacity, hidden: bool) -> bool:
        """Draw one surface as a GLB. ``False`` means the caller should fall back.

        Hidden is handled without encoding anything: a garment nobody is looking
        at should not cost a GLB a frame.
        """
        if hidden:
            handle = self._garment_nodes.get(name)
            if handle is None:
                return True
            self._set_hidden(name, handle, True)
            return True
        if opacity is not None:
            # The spec's own opacity wins: it is what the flat path draws and
            # what the peel suites tuned their films against.
            finish = replace(finish, opacity=float(opacity))
        blob = glb_for(
            np.asarray(vertices, dtype=np.float64),
            np.asarray(faces, dtype=np.int64).reshape(-1, 3),
            tuple(c / 255.0 for c in color),
            finish,
            uvs=None,
            smooth=True,
            # Deforming: these vertices are new every frame, and a cached blob
            # would be one more entry nothing reads until the model changes.
            cache=False,
        )
        if blob is None:
            return False
        try:
            handle = self._server.scene.add_glb(name, glb_data=blob, cast_shadow=True, receive_shadow=True)
        except Exception:
            return False
        self._garment_nodes[name] = handle
        self._garment_hidden[name] = False
        return True

    def _draw_fuzz_node(
        self, name: str, points: np.ndarray, colors: np.ndarray, hidden: bool, *, size: float = FUZZ_POINT_SIZE
    ) -> None:
        name = self._qualify(name)
        handle = self._garment_nodes.get(name)
        if handle is not None:
            try:
                if not hidden:
                    handle.points = points.astype(np.float32)
                    handle.colors = colors
                self._set_hidden(name, handle, hidden)
                return
            except Exception:
                handle = None
        handle = self._server.scene.add_point_cloud(
            name,
            points=points.astype(np.float32),
            colors=colors,
            point_size=float(size),
            point_shape="circle",
            # Center-to-edge falloff: the soft dot that reads as a tuft.
            point_shading="gradient",
            precision="float16",
            visible=not hidden,
        )
        self._garment_nodes[name] = handle
        self._garment_hidden[name] = hidden

    def _set_hidden(self, name: str, handle: Any, hidden: bool) -> None:
        if handle is None:
            return
        if self._garment_hidden.get(name) != hidden:
            handle.visible = not hidden
            self._garment_hidden[name] = hidden


def viewer_studio(port: int, share: bool) -> ViewerStudio:
    """The studio's viewer: viser, with the deformables drawn as garments, titled as the page it is."""
    viewer = ViewerStudio(port=port, share=share, label=PAGE_TITLE)
    title_page(viewer._server)
    return viewer
