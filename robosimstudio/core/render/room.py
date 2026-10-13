# SPDX-License-Identifier: Apache-2.0
"""The room a scene stands in when nobody drew one.

:mod:`~robosimstudio.core.render.scenery` holds the *table* -- a registry a task
may register a backdrop into -- and says "content registers; the toolkit only
holds the table". This module is the deliberate exception, for the same reason
:mod:`~robosimstudio.core.render.look` is one: a default that lives in a task
directory reaches that task and nothing else, and what was missing is not a room
for one suite but a room for every scene that never asked for one. Two of eleven
suites drew themselves a backdrop; the other nine rendered into a uniform grey
void, and on their oblique cameras that void is most of the frame -- measured,
between 50 and 89 per cent of it lands past two metres or above the horizon.

**Three layers, and only the first two are here.** A picture of a bench is a
near field (the work surface and whatever is dressed onto it), a room (a floor
that catches the contact shadow, and walls the light bounces off), and a far
field (whatever is past the room). :func:`house_room` draws the middle one from
the cell the task already declares; the near field stays with the suite, because
a grid mat measured off a reference clip is a statement about that bench and not
about rooms in general. **There is no far field**, and that is a measurement
rather than an omission -- see :data:`ROOMS`.

**Why the room is drawn rather than photographed.** Two upstream benchmarks were
read before this was written and both make the whole background one dome light
carrying a warehouse panorama, with no floor geometry at all. One of them then
spent three pull requests trying to get a contact shadow back -- a matte
shadow-catcher plane, removed a day later for "floor-line and temporal
black-trail artifacts in moving-camera RGB" -- and its tables now meet the floor
with no shadow and read as cutouts pasted onto a photograph.
:mod:`robosimstudio_studio.lighting` had already ruled the same way from the
other side: it lights with an HDRI and keeps ``background=False``, because "a
photographic backdrop behind a robot arm reads as a compositing error rather
than as a room". So the near field here is geometry, which shadows and bounces
light like geometry, and a panorama is only ever what is beyond it.

**One shape, measured against every camera the suites declare.** Two candidate rules
existed in the tree already. The tool-sorting suite's -- a floor at the cell centre
and three walls at 2.2 m -- contains all 106 cameras across all 47 of the
benchmark's variants. The lab suite's, whose wall sits at the bench edge, leaves
21 of them outside it. So the general builder is the first, and the second is expressed as
the first with two of its walls dropped (:func:`lab`).

**A room is camera-only, and that is a contract.** Nothing here collides, none
of it is a landmark a framing test projects, and the tiled tracer a policy is
evaluated on has no code path that reads scenery at all. Dressing a scene
cannot move a score.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from robosimstudio.registry import Registry

from .finish import MATTE, Finish, register_texture
from .look import STUDIO
from .scenery import register_scenery
from .spec import RenderSpec

__all__ = [
    "FLOOR_HALF",
    "HANGAR_CLOSED",
    "LAB",
    "LAB_FLOOR",
    "LAB_GRID_MAT",
    "LAB_MAT_GREY",
    "LAB_OAK",
    "LAB_OAK_COLOR",
    "LAB_WALL",
    "OPEN",
    "ROOMS",
    "ROOM_CONCRETE",
    "ROOM_FLOOR",
    "ROOM_PLASTER",
    "ROOM_WALL",
    "VOID",
    "WALL_HEIGHT",
    "WALL_SETBACK",
    "get_room",
    "house_room",
    "lab",
    "lab_bench",
    "lab_lounge",
    "lab_rig",
    "register_room",
]

Vec3 = tuple[float, float, float]


# ------------------------------------------------------------------ geometry

WALL_SETBACK = 2.2
"""How far from the cell centre a room's walls stand [m].

Transcribed from the tool-sorting suite's shop, where it was chosen so the arm, its
pedestal and the bench are nowhere near a wall while a 45-degree frame is still
filled by a lit room. Checked afterwards against every camera the benchmark's
suites declare: the furthest any of them sits from its own cell centre is 1.20 m, so
none is outside the room it is standing in.
"""

WALL_HEIGHT = 2.6
"""Ceiling height [m]. Above the ``lab`` preset's panel light at
``surface_z + 1.7``, so the panel hangs inside the room rather than outside it.
The tallest shipped camera is 2.05 m above its cell's ground."""

FLOOR_HALF = 4.0
"""Half-extent of the drawn floor [m]. Larger than the walls it meets, so no
band of untextured ground plane shows in the corner where two walls do not."""

FLOOR_LIFT = 0.0015
"""How far the drawn floor's mid-plane sits above the cell's ground [m].

The compiler puts a real ground plane at ``ground_z`` on every scene
(``scene/compiler.py``), and two coplanar surfaces z-fight. This is also the
box's half-thickness, so the floor's underside rests exactly on that plane."""


# ------------------------------------------------------------------ colours
#
# Linear RGB in [0, 1], as ``shape_color`` wants. These are room surfaces rather
# than palette roles: a wall is not a fixture and must not take a fixture's grey,
# which is the colour the thing standing in front of it already has.

ROOM_FLOOR: Vec3 = (0.6600, 0.6450, 0.6200)
"""Pale sealed concrete.

Started at the tool-sorting shop's measured floor tile, ``(0.5501, 0.4938, 0.4441)``, and
was lightened and cooled from there. Two reasons, both from renders. A drawn room
*occludes* light an infinite void did not, so the same albedo reads darker inside
four walls than it did on an open plane -- measured on a gear-meshing scene's front view,
the warm floor cost 6 per cent of frame mean against no room at all. And the
warm cast that suited a workcell photographed against its own export read as
under-exposure everywhere else. The tile's own mottling is +/-11 per cent, so the
peaks still clear 1.0 comfortably."""

ROOM_WALL: Vec3 = (0.8600, 0.8500, 0.8300)
"""Painted plaster, bright.

This is the number that matters most in a path tracer: the walls are the largest
diffuse surface in the room, so almost all of the bounce is theirs, and lifting
them lifts everything. Kept under pyrender's measured clipping albedo of 0.9121
so a room stays renderable by the backends that
are not path tracers, and short of white so a white robot shell still separates
from the wall behind it."""

LAB_FLOOR: Vec3 = (0.33, 0.33, 0.34)
"""The workshop floor past the bench, as the lab suite's low camera sees it."""

LAB_WALL: Vec3 = (0.62, 0.62, 0.63)
"""A plain painted wall a couple of metres off."""

LAB_MAT_GREY: Vec3 = (0.456, 0.456, 0.451)
"""The foam mat between its grid lines. Carried by the grid texture; stated for
the untextured fallback."""

LAB_OAK_COLOR: Vec3 = (0.888, 0.584, 0.356)
"""The ledge and the slats, lit."""

LAB_GLASS: Vec3 = (0.020, 0.022, 0.026)
"""The glazed wall the slats stand against: what is behind the slats in every
clip is not a black panel but a dark window onto a dim lounge, with the bench
lights reflected in it. Near-black albedo under a clear coat, see
:data:`LAB_GLASS_FINISH`."""

LAB_LOUNGE_WALL: Vec3 = (0.30, 0.24, 0.18)
"""The lounge's far wall, warm: lit only by the dome it reads as the dim brown
depth the clips have behind the glass, through a glass that passes about 70 %."""

LAB_LOUNGE_TABLE: Vec3 = (0.86, 0.82, 0.74)
"""The pale round table that shows through the glass in the decant, routing and
peel clips (sampled off ``box_decanting.mp4``: a near-white oval top)."""

LAB_RIG_WHITE: Vec3 = (0.84, 0.85, 0.86)
"""The camera rig's aluminium-extrusion uprights, white, one beside each arm
base. Visible at the left edge of ``box_decanting.mp4``, ``film peeling.mp4``
and the right edge of ``channel routing.mp4``."""

LAB_BENCH_EDGE: Vec3 = (0.030, 0.030, 0.033)
"""The bench's near edge, which every clip shows as a black strip along the
bottom of the frame: the table frame the mats lie on."""


# ------------------------------------------------------------------ textures


def _srgb(linear: Vec3) -> np.ndarray:
    c = np.asarray(linear, dtype=np.float64)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def _noise(shape: tuple[int, int], seed: int, octaves: int = 4) -> np.ndarray:
    """Value noise in [0, 1], a few octaves, for mottling and grain."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, dtype=np.float64)
    amplitude, total = 1.0, 0.0
    for o in range(octaves):
        cells = 4 * 2**o
        coarse = rng.random((cells + 1, cells + 1))
        ys = np.linspace(0, cells, shape[0], endpoint=False)
        xs = np.linspace(0, cells, shape[1], endpoint=False)
        y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
        fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
        fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        a = coarse[y0][:, x0]
        b = coarse[y0][:, x0 + 1]
        c = coarse[y0 + 1][:, x0]
        d = coarse[y0 + 1][:, x0 + 1]
        out += amplitude * ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy)
        total += amplitude
        amplitude *= 0.5
    return out / total


def _concrete(size: int = 1024, tile_m: float = 1.2) -> np.ndarray:
    """A 1.2 m slab of poured concrete: mottled, faintly aggregated, with a
    darker control joint at the tile edge so a floor reads as a floor and not as
    a fog. Procedural rather than a PNG for the reason
    :mod:`~robosimstudio.core.render.finish` gives: a generated texture can be
    diffed and a binary cannot."""
    base = _srgb(ROOM_FLOOR)
    mottle = (_noise((size, size), seed=101, octaves=5) - 0.5) * 0.11
    speck = (np.random.default_rng(103).random((size, size)) - 0.5) * 0.045
    img = base[None, None, :] * (1.0 + mottle + speck)[..., None]
    joint = int(round(size * 0.004))
    img[:joint, :] *= 0.80
    img[:, :joint] *= 0.80
    return np.clip(img, 0.0, 1.0)


def _plaster(size: int = 512) -> np.ndarray:
    """Painted plaster: almost flat, with the long-wavelength unevenness a
    rolled wall has. Nearly featureless on purpose -- a wall that patterns
    competes with the work in front of it."""
    base = _srgb(ROOM_WALL)
    wash = (_noise((size, size), seed=107, octaves=3) - 0.5) * 0.05
    tooth = (np.random.default_rng(109).random((size, size)) - 0.5) * 0.02
    img = base[None, None, :] * (1.0 + wash + tooth)[..., None]
    return np.clip(img, 0.0, 1.0)


def _grid_mat(size: int = 1024, cells: int = 12, line_mm: float = 2.0, tile_mm: float = 300.0) -> np.ndarray:
    """One 30 cm interlocking mat tile: grey foam, a pale 25 mm grid, a darker seam at the edge.

    **25 mm on 300 mm tiles, not 50 mm on 600.** Pixel-counted off the reference
    clips' overhead inset (2026-09-09): the grid pitch is ~27 px where the two
    arm-base brackets, 0.62 m apart, are 700 px apart, and a tile is twelve
    squares across. The 50 mm reading that shipped first doubled every object
    measured against it."""
    px_per_mm = size / tile_mm
    base = _srgb(LAB_MAT_GREY)
    img = np.empty((size, size, 3), dtype=np.float64)
    grain = (_noise((size, size), seed=11, octaves=5) - 0.5) * 0.06
    img[...] = base[None, None, :] + grain[..., None]
    coords = np.arange(size) / px_per_mm  # mm
    pitch = tile_mm / cells
    on_line = (np.abs((coords % pitch) - 0.0) <= line_mm / 2.0) | (np.abs((coords % pitch) - pitch) <= line_mm / 2.0)
    line = np.array([0.843, 0.863, 0.871])
    mask = on_line[:, None] | on_line[None, :]
    img[mask] = line
    # The interlocking edge: an EVA puzzle mat's seam is not a straight line but
    # a square wave of tabs and notches, 25 mm wide and about 6 mm deep, and
    # it is the one feature that says "foam mat" rather than "printed grid" in
    # every frame of the clips. Drawn on this tile's top and left edges only,
    # which is what makes it periodic: the tab that leaves this tile at +6 mm
    # is the neighbour's notch at -6, and a tiled texture joins them.
    tab_w = int(round(25.0 * px_per_mm))
    tab_d = int(round(6.0 * px_per_mm))
    seam_w = max(1, int(round(1.5 * px_per_mm)))
    u = np.arange(size)
    offset = np.where((u // tab_w) % 2 == 0, tab_d, -tab_d)
    for du in range(seam_w):
        rows = (offset + du) % size
        img[rows, u, :] *= 0.72  # the seam that runs along u (a horizontal edge)
        img[u, rows, :] *= 0.72  # and the one down the side
    for edge in range(0, size, tab_w):
        span = np.arange(-tab_d, tab_d + 1) % size
        cols = np.clip(np.arange(edge, edge + seam_w), 0, size - 1)
        for c in cols:
            img[span, c, :] *= 0.72
            img[c, span, :] *= 0.72
    return np.clip(img, 0.0, 1.0)


def _oak(size: int = 512) -> np.ndarray:
    """Oak: warm, with vertical grain streaks."""
    base = _srgb(LAB_OAK_COLOR)
    x = np.arange(size) / size
    grain = (
        0.06 * np.sin(x * 2 * np.pi * 18 + 3.0 * _noise((1, size), seed=31, octaves=3)[0])
        + (_noise((size, size), seed=37, octaves=3) - 0.5) * 0.08
    )
    img = (
        base[None, None, :] * (1.0 + grain)[..., None]
        if grain.ndim == 2
        else base[None, None, :] * (1.0 + grain[None, :])[..., None]
    )
    return np.clip(img, 0.0, 1.0)


# Version 2: the generators are unchanged but ROOM_FLOOR and ROOM_WALL were
# lightened, and the cache is keyed by name and version rather than by content --
# so a stale v1 rendered under the old colours would be served forever.
register_texture("room_concrete", _concrete, version=2)
register_texture("room_plaster", _plaster, version=2)
# Version 2: the grid went from 50 mm on a 60 cm tile to the measured 25 mm on a
# 30 cm one; the cache is keyed by name and version, so v1 must not be served.
# Version 3: the tile seams became the puzzle mat's interlocking tabs.
register_texture("room_grid_mat", _grid_mat, version=3)
register_texture("room_oak", _oak, version=1)


# ------------------------------------------------------------------ finishes

ROOM_CONCRETE = Finish(roughness=0.88, texture="room_concrete", uv_scale=(1.0 / 1.2, 1.0 / 1.2))
"""A floor. ``uv_scale`` is repeats per metre, so the 1.2 m slab lands at 1.2 m
whichever box carries it."""

ROOM_PLASTER = Finish(roughness=0.94, texture="room_plaster", uv_scale=(0.8, 0.8))
"""A wall: matte, and coarse enough that the bounce is diffuse."""

LAB_GRID_MAT = Finish(roughness=0.85, texture="room_grid_mat", uv_scale=(1.0 / 0.3, 1.0 / 0.3))
"""The gridded foam mat: one 30 cm tile repeated, so the lines land on world
multiples of 25 mm on any box that carries it."""

LAB_OAK = Finish(roughness=0.55, texture="room_oak", uv_scale=(1.0, 1.0))
"""The ledge and the slats."""

LAB_GLASS_FINISH = Finish(roughness=0.04, opacity=0.30, ior=1.5)
"""Window glass: a clear coat over the near-black albedo. Part-transparent, so
the dim lounge behind it is seen through the slat gaps the way the clips show
it, and glossy enough to carry a reflection of the arms and the panel light."""


# ------------------------------------------------------------------ builders


def _cell_frame(kit) -> tuple[float, float, float, float, float]:
    """``(cx, cy, ground_z, hx, hy)`` for whatever cell this is, or a default.

    ``kit.cell`` is ``None`` for a scene built straight from a ``BuildRequest``
    with nothing bound -- which the bare-scene tests do, and which any caller
    using the package as a library does -- so a room that indexed it would be a
    default that works only for the eleven suites and crashes for everybody
    else. Falling back to the world origin puts the room where a scene with no
    declared cell actually is.
    """
    cell = getattr(kit, "cell", None)
    center = tuple(float(v) for v in (getattr(cell, "center", None) or (0.0, 0.0, 0.0)))
    half = tuple(float(v) for v in (getattr(cell, "half", None) or (0.3, 0.4, 0.375)))
    ground_z = float(getattr(cell, "ground_z", 0.0) or 0.0)
    return center[0], center[1], ground_z, half[0], half[1]


def house_room(
    kit,
    *,
    walls: tuple[str, ...] = ("-x", "+y", "-y"),
    floor_half: float = FLOOR_HALF,
    setback: float = WALL_SETBACK,
    wall_span: float | None = None,
    wall_height: float = WALL_HEIGHT,
    floor_lift: float = FLOOR_LIFT,
    floor_color: Vec3 = ROOM_FLOOR,
    wall_color: Vec3 = ROOM_WALL,
    floor_finish: Finish = ROOM_CONCRETE,
    wall_finish: Finish = ROOM_PLASTER,
    prefix: str = "room",
) -> None:
    """Draw a floor and walls around whatever cell ``kit`` is dressing.

    Everything is placed against the cell rather than at literals, so one rule
    serves eleven suites and anything a host stands in the same bench.

    Args:
        walls: Which sides to build, from ``"+x" "-x" "+y" "-y"``. The default
            leaves ``+x`` open because that is the side every oblique camera in
            the benchmark's suites look *from*; a wall there would be between the camera
            and the work. Passing fewer is how :func:`lab` reproduces a backdrop
            that was calibrated with only two.
        floor_half: Half-extent of the floor slab [m].
        setback: Distance from the cell centre to each wall [m].
        wall_span: Half-length of a wall along its own long axis [m]. Defaults
            to ``setback + 0.5``, which closes the corners.
        wall_height: Full height of a wall [m]; it is drawn from the ground up.
        floor_lift: Half-thickness of the floor slab [m], which is also how far
            its mid-plane sits above the cell's ground -- so its underside rests
            on the compiler's ground plane rather than z-fighting with it.
        floor_color, wall_color, floor_finish, wall_finish: What the surfaces
            are. Stated rather than fixed because a room measured against
            reference footage is still this shape with other materials.
        prefix: Prim-name prefix, so two builders may draw two rooms on one
            stage without colliding.
    """
    cx, cy, ground_z, _hx, _hy = _cell_frame(kit)
    span = setback + 0.5 if wall_span is None else float(wall_span)
    half_h = float(wall_height) / 2.0

    kit.box(
        f"{prefix}_floor",
        (cx, cy, ground_z + floor_lift),
        (floor_half, floor_half, floor_lift),
        color=floor_color,
        finish=floor_finish,
    )
    plan = {
        "-x": ((cx - setback, cy, ground_z + half_h), (0.02, span, half_h)),
        "+x": ((cx + setback, cy, ground_z + half_h), (0.02, span, half_h)),
        "+y": ((cx, cy + setback, ground_z + half_h), (span, 0.02, half_h)),
        "-y": ((cx, cy - setback, ground_z + half_h), (span, 0.02, half_h)),
    }
    for side in walls:
        if side not in plan:
            raise ValueError(f"a room wall is one of {sorted(plan)}, got {side!r}")
        center, half = plan[side]
        kit.box(
            f"{prefix}_wall_{side[1]}{'p' if side[0] == '+' else 'n'}",
            center,
            half,
            color=wall_color,
            finish=wall_finish,
        )


@register_scenery("house_room_closed")
def house_room_closed(kit) -> None:
    """The room every scene gets when it states nothing: concrete and plaster,
    all four walls closed rather than leaving ``+x`` open.

    A three-walled room only holds a camera that looks *from* roughly the open
    side, the way most suites' own agentviews do. A camera authored to look
    the other way (into rather than out of that side) puts the gap in frame
    instead of behind itself -- a miss ray that exits the room there and
    finds nothing, keyed to a flat fallback colour that reads as a patch
    rather than a wall. Closing all four sides holds every camera, not just
    the ones authored toward the open side."""
    house_room(kit, walls=("-x", "+x", "+y", "-y"))


@register_scenery("house_floor")
def house_floor(kit) -> None:
    """The floor alone, for a look whose walls come from a visible panorama.

    Kept separate rather than reached by an argument because it is a different
    *statement*: this one says the far field is photographed, and it only makes
    sense where the dome is drawn in the primary ray.
    """
    house_room(kit, walls=())


LAB_SLAT_WIDTH = 0.06
LAB_SLAT_GAP = 0.045
LAB_WALL_SETBACK = 0.06
"""How far beyond the bench edge the slat wall stands [m].

Small on purpose. The overhead's top edge lands metres past the bench, so any
real setback opens a band of bare floor between the ledge and the wall -- and in
the reference clips there is none: the mats run to the ledge and the slats start
immediately above it."""


def lab_bench(kit) -> None:
    """The near field of a photographic lab: a gridded mat over the work
    surface, an oak ledge at its ``+x`` edge, and the slat wall above it.

    Transcribed from the lab suite, which measured every number here off
    reference footage of the real cell. It is here rather than there because a
    room ought to be selectable by any suite, and this is the half of that room
    that is not a floor and two walls.

    The prim names are the ones that suite already wrote, unprefixed, because
    they are what a stage a test reads is made of and there is nothing to gain
    from renaming them.
    """
    cx, cy, ground_z, hx, hy = _cell_frame(kit)
    z = float(kit.surface_z)

    kit.box("grid_mat", (cx, cy, z + 0.0005), (hx, hy, 0.00025), color=LAB_MAT_GREY, finish=LAB_GRID_MAT)

    edge = cx + hx
    ledge_top = z + 0.06
    kit.box(
        "ledge",
        (edge + 0.02, cy, (ground_z + ledge_top) / 2.0),
        (0.02, hy + 0.06, (ledge_top - ground_z) / 2.0),
        color=LAB_OAK_COLOR,
        finish=LAB_OAK,
    )

    wall = edge + LAB_WALL_SETBACK
    # The glazing the slats stand against. It was a near-black matte panel,
    # which read as a wall; the clips show a window, dark because the room
    # beyond is dim, and the slat gaps carry both a reflection of the bench
    # lights and a glimpse of the lounge behind. Drawn as glass, with the
    # lounge behind it, so both of those happen in the render rather than in a
    # texture.
    kit.box(
        "wall_backing",
        (wall + 0.03, cy, ground_z + 0.95),
        (0.004, hy + 0.9, 0.95),
        color=LAB_GLASS,
        finish=LAB_GLASS_FINISH,
    )
    pitch = LAB_SLAT_WIDTH + LAB_SLAT_GAP
    for i, y in enumerate(np.arange(cy - hy - 0.9, cy + hy + 0.9, pitch)):
        kit.box(
            f"slat_{i}",
            (wall + 0.01, float(y) + LAB_SLAT_WIDTH / 2.0, ground_z + 0.95),
            (0.01, LAB_SLAT_WIDTH / 2.0, 0.95),
            color=LAB_OAK_COLOR,
            finish=LAB_OAK,
        )
    lab_lounge(kit, x0=wall + 0.05)
    lab_rig(kit)


LAB_LOUNGE_DEPTH = 2.6
"""How far behind the glass the lounge's far wall stands [m]. Deep enough that
the overhead's top edge, which lands metres past the bench, still sees a room
rather than the back of a wall."""


def lab_lounge(kit, *, x0: float) -> None:
    """The dim room beyond the glass: a far wall, a pale round table on a
    pedestal, two chairs and a low bench. Nothing here is measured -- the clips
    show it out of focus through the slats -- but the light is: lit by the dome
    alone at the lab's 0.3 ambient, the table top comes back at roughly a
    quarter of the bench's exposure, which is the relationship the clips have."""
    _cx, cy, ground_z, _hx, _hy = _cell_frame(kit)
    span = 3.4
    kit.box(
        "lounge_wall",
        (x0 + LAB_LOUNGE_DEPTH, cy, ground_z + 1.5),
        (0.02, span, 1.5),
        color=LAB_LOUNGE_WALL,
        finish=MATTE,
    )
    for i, y in enumerate((cy - span, cy + span)):
        kit.box(
            f"lounge_side_{i}",
            (x0 + LAB_LOUNGE_DEPTH / 2.0, float(y), ground_z + 1.5),
            (LAB_LOUNGE_DEPTH / 2.0, 0.02, 1.5),
            color=LAB_LOUNGE_WALL,
            finish=MATTE,
        )
    tx, ty = x0 + 1.25, cy + 0.15
    points, faces = _cylinder_mesh((tx, ty, ground_z + 0.735), 0.55, 0.015, segments=48)
    kit.mesh("lounge_table", points, faces, color=LAB_LOUNGE_TABLE, finish=MATTE)
    kit.box("lounge_pedestal", (tx, ty, ground_z + 0.36), (0.05, 0.05, 0.36), color=(0.08, 0.08, 0.085), finish=MATTE)
    for i, (dx, dy) in enumerate(((-0.75, 0.55), (0.05, -0.85))):
        kit.box(
            f"lounge_seat_{i}",
            (tx + dx, ty + dy, ground_z + 0.44),
            (0.22, 0.22, 0.03),
            color=(0.10, 0.09, 0.08),
            finish=MATTE,
        )
        kit.box(
            f"lounge_back_{i}",
            (tx + dx, ty + dy + (0.2 if dy > 0 else -0.2), ground_z + 0.66),
            (0.22, 0.02, 0.2),
            color=(0.10, 0.09, 0.08),
            finish=MATTE,
        )
    kit.box(
        "lounge_bench", (x0 + 2.2, cy - 1.4, ground_z + 0.22), (0.25, 0.9, 0.22), color=(0.30, 0.22, 0.15), finish=MATTE
    )


def lab_rig(kit) -> None:
    """The camera rig and the bench's own edge: two white extrusion uprights
    standing on the ledge at the bench's far corners, in front of the slats --
    the white bars at the left edge of the decant and peel clips and the right
    edge of the routing clip -- and the black apron along the near edge of the
    table that every clip's bottom edge shows."""
    cx, cy, _ground_z, hx, hy = _cell_frame(kit)
    z = float(kit.surface_z)
    top = z + 1.6
    for i, y in enumerate((cy - hy - 0.01, cy + hy + 0.01)):
        kit.box(
            f"rig_post_{i}",
            (cx + hx + 0.02, float(y), (z + 0.06 + top) / 2.0),
            (0.015, 0.015, (top - z - 0.06) / 2.0),
            color=LAB_RIG_WHITE,
            finish=MATTE,
        )
    kit.box("bench_apron", (cx - hx - 0.006, cy, z - 0.025), (0.006, hy, 0.025), color=LAB_BENCH_EDGE, finish=MATTE)


def _cylinder_mesh(center, radius: float, half_height: float, *, segments: int = 32):
    """A closed cylinder about z as ``(points, faces)`` for :meth:`SceneryKit.mesh`."""
    cx, cy, cz = (float(v) for v in center)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)
    top = np.column_stack([ring, np.full(segments, cz + half_height)])
    bottom = np.column_stack([ring, np.full(segments, cz - half_height)])
    points = np.vstack([top, bottom, [[cx, cy, cz + half_height]], [[cx, cy, cz - half_height]]])
    ct, cb = 2 * segments, 2 * segments + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([ct, i, j])
        faces.append([cb, segments + j, segments + i])
        faces.append([i, segments + i, segments + j])
        faces.append([i, segments + j, j])
    return points.astype(np.float32), np.asarray(faces, dtype=np.int32)


@register_scenery("lab")
def lab(kit) -> None:
    """The photographic lab: the mat, the ledge and the slat wall, standing in a
    pale room.

    The room half states two walls rather than three and puts them at the cell's
    own half-depth plus 1.6 m, which is where the lab suite's look measured them.
    There is deliberately no ``-x`` wall, and the ``+x`` one is the glazing the
    slats stand against; what is beyond it is :func:`lab_lounge`, the dim room
    the clips show through the glass, closed by its own far wall.
    """
    _cx, _cy, _gz, _hx, hy = _cell_frame(kit)
    house_room(
        kit,
        walls=("+y", "-y"),
        floor_half=3.0,
        setback=hy + 1.6,
        wall_span=3.0,
        floor_lift=0.0005,
        floor_color=LAB_FLOOR,
        wall_color=LAB_WALL,
        floor_finish=MATTE,
        wall_finish=MATTE,
        prefix="lab",
    )
    lab_bench(kit)


# ------------------------------------------------------------------ the rooms


def _same_room(a: RenderSpec, b: RenderSpec) -> bool:
    """A room registered twice under one name is fine when it is the same room.

    Modules are imported twice here as a matter of course -- a task package is
    loaded once by the registry and once more under the environment's own copy
    -- and a :class:`RenderSpec` is a frozen dataclass, so two copies of the same
    statement compare equal without being identical.
    """
    return a == b


ROOMS: Registry[RenderSpec] = Registry(
    "room",
    same=_same_room,
    hint="a room registers when the module naming it is imported; a document naming one by dotted path imports it",
)
"""``name -> RenderSpec``. The library's rooms are below; anybody else's arrive
when the module that states them is imported, which a document's
``(:render "their_pkg.mod:ROOM")`` does by itself.

**Why none of them carries an environment map.** The obvious way to give a scene
a far field is an HDR panorama on the dome, which is what both NVIDIA benchmarks
this was modelled on do. It does not work here, and the negative result is worth
keeping because it is not obvious and it costs a GPU to find. The stage RSS hands
the path tracer is correct -- ``inputs:texture:file`` resolves to a real file on
disk, ``inputs:texture:format`` is ``latlong``, ``visibleInPrimaryRay`` is set,
and the dome carries the Z-up rotation a lat-long map needs; all four survive the
export and were read back off the ``.usd`` the renderer opens. The renderer
ignores them anyway. Measured: two panoramas of quite different radiance
(``small_hangar_01``, upper-hemisphere mean 0.99 and warm, against
``brown_photostudio_01``, 1.05 and flat) rendered the same scene to a frame mean
of 86.77 and 86.95 -- identical to a fifth of one grey level, on every band. The
map is not being used as light *or* as background. This is the same family as the
already-recorded finding that this renderer ignores the path-tracing attributes
on a render product, and the same conclusion follows: what reaches the picture is
geometry.

:attr:`~robosimstudio.core.render.spec.RenderSpec.environment_map` is left wired
and now resolves properly (``rtx.py``), so the day a backend honours it, a room
states one and works. Until then a room is a floor and some walls."""


def register_room(name: str, spec: RenderSpec, *, origin: str = "") -> RenderSpec:
    """Register ``spec`` under ``name`` and hand it back, so a module can say
    ``WAREHOUSE = register_room("warehouse", replace(...))`` in one statement.

    The library's own rooms go through this door rather than around it. A seam
    only stays working if its owner uses it.
    """
    return ROOMS.register(name, spec, origin=origin or __name__)


def get_room(name: str) -> RenderSpec:
    """The room registered under ``name``. Raises naming what is registered."""
    return ROOMS.get(name)


HANGAR_CLOSED: RenderSpec = register_room(
    "hangar_closed",
    replace(STUDIO, scenery="house_room_closed"),
)
"""The default room: a concrete floor and four plaster walls around the cell.

It is ``STUDIO`` plus a room and nothing else stated here -- ``environment``,
``key_intensity`` and ``ambient`` all come from :data:`STUDIO` itself, which
now carries ``environment="lab"`` (see its own docstring for why: OVRTX's
baked ``"default"`` key lit a room's own corner unevenly, one wall lit and its
neighbour in the key's own shadow, in a way a single tabletop rig was never
measured against).

*No environment map.* See :data:`ROOMS` -- the RTX backend only makes a dome's
texture visible in the primary ray under the ``lab`` panel-light path, and
this room states no ``environment_map`` to feed it, so its dome stays a
uniform colour rather than an HDR photo.

It is the default because it is the only room that holds every camera the
suites declare, not because it flatters all of them: a close, steeply-angled
view gains a dark corner from the walls and is better served by :data:`OPEN`.
Measured on a USB-C insertion scene's ``overview``, 94.3 here against 125.9 there.

All four walls closed, rather than three around an open side: a three-walled
room only holds a camera that looks *from* roughly the open side, the way most
suites' own agentviews do, so the gap sits behind the camera, out of frame. A
camera that does not hold that assumption (a tool-hanging scene's hero
shot looks the other way) would put the gap itself in frame -- a miss ray that
exits the room and is keyed to whatever flat ``--background`` was passed, a
patch of colour standing where the other walls are a lit, textured surface.
Closing the fourth wall means the room surrounds the shot instead of
surrounding the camera, for every camera, not just the ones authored toward
the open side."""

OPEN: RenderSpec = register_room("open", replace(STUDIO, scenery="house_floor"))
"""The floor without the walls, for a camera low enough that a wall at 2.2 m
would be between it and the work.

It was called ``warehouse`` for about an hour, which was a promise it could not
keep: a floor is not a warehouse, and the walls that would have made it one were
going to come from a panorama on the dome, which this renderer ignores.

**Whether it beats a full room depends entirely on the camera**, and the two
measurements bracket it. On a gear-meshing scene's ``chain`` view -- 80 mm above the
bench, so the floor is below the frame -- it renders at mean 112.4, which is
:data:`VOID` exactly: nothing it draws is visible. On a USB-C insertion scene's
``overview`` it renders at **125.9**, against 107.8 for :data:`VOID` and 94.3 for
:data:`HANGAR_CLOSED`, and it is the best of the three: that camera sees plenty
of floor while a wall at 2.2 m does nothing but put a dark corner behind the
arm. So this is the room for a camera that looks *down* at a bench from close
in, and :data:`HANGAR_CLOSED` is the one for a camera that looks *across* it."""

LAB: RenderSpec = register_room("lab", replace(STUDIO, environment="lab", scenery="lab", quality="preview"))
"""The photographic lab one suite is calibrated in: a gridded mat, an
oak ledge and a slat wall, standing in a pale room, under the soft ``lab`` panel.

This one *does* state the preset, because this is the look that preset was tuned
for and the two were measured together against footage of the real cell.

**It does not suit every camera, and it fails loudly rather than gracefully.**
Its slat wall stands 60 mm beyond the bench edge -- close on purpose, so no band
of bare floor opens under a low camera -- which means a camera further out than
``cell.center.x + cell.half.x + 0.06`` is *outside the room*, looking at the
unlit back of a wall. Of the 106 cameras the benchmark's suites declare, 21 are:
every tool-sorting agentview, all five cable-routing corners,
and both cloth-folding fronts. Measured on the towel fold, whose
``front`` camera sits at x = 1.55 against a wall at x = 0.96 -- frame mean 9.0,
maximum pixel 11, a black picture.

So this room is for a camera close in and low, which is what it was built for:
the lab suite's own ``close`` sits 0.32 m out at 11.6 degrees. :data:`HANGAR_CLOSED`
is the one that holds every camera, and it is the default for that reason."""

VOID: RenderSpec = register_room("void", STUDIO)
"""The house look with no room at all -- what every scene rendered as before
this module existed. Registered so that "no backdrop" is a thing a document can
*say*, rather than something it gets by being silent."""
