# SPDX-License-Identifier: Apache-2.0
"""The colours a scene is built from, named once, and chosen against a renderer.

Not a theme and not decoration. ``ModelBuilder`` gives any shape nobody coloured
a colour indexed by its position in the build order, so "unstated" does not mean
"neutral" -- it means an arbitrary hue that *changes* when anything upstream
adds a shape. Naming the colours is what makes a scene look the same twice, and
naming them **here** is what stops four suites from each inventing their own
near-black table.

The scheme is the one the port suites arrived at independently: a dark ground, a
mid-grey fixture on top of it, exactly one saturated accent for the thing the
task is *about*, and near-white for anything deformable. It reads because it is
mostly achromatic -- the one saturated object in frame is the one a policy has to
find.

**What changed, and why it is arithmetic rather than taste.** The values used to
be chosen by eye and were roughly four stops too dark for the renderer that
actually draws them. The tiled tracer -- the default, and the one a policy is
scored through -- shades a hit as::

    shaded = albedo_linear · (ambient · 0.5) + albedo_linear · Σ (n·l · visibility)

with ``ambient`` a hemispheric blend of sky ``(0.4, 0.4, 0.45)`` and ground
``(0.1, 0.1, 0.12)``, a single light along ``(-1, 1, -1)``, and
``visibility = 0.3`` in shadow. All of it is compile-time constant inside the
Warp megakernel: ``RenderConfig`` exposes ``enable_ambient_lighting`` as a *bool*
and nothing else. So for an up-facing surface the whole transfer function is
``albedo · 0.777`` lit and ``albedo · 0.373`` shadowed, **and albedo is the only
exposure control there is.** No lighting change can substitute for it.

Run the old values through it and the frame explains itself:

===============  ========  ==========  ============
role             albedo    lit (8-bit)  in shadow
===============  ========  ==========  ============
``GROUND``       0.07      **15**       **7**
``FIXTURE``      0.28      63           43
``DEFORMABLE``   0.92      210          151
===============  ========  ==========  ============

A table at pixel 15 is black, which is what a polishing task's recorded front
camera shows, and it is why the arms' shadows in the overhead view read as holes
rather than as shadows. Measured over a whole frame, 76.4 per cent of a
cloth-folding scene rendered below pixel 32.

**The floor is raised, and no further.** That share falls to 1.6 per cent at
``GROUND`` 0.24 and does not fall again above it -- the crush is a step function
that saturates, and every value past it buys brightness by spending contrast. It
is worth naming what overshooting costs, because the obvious fix is to put
everything in one comfortable mid band and it is wrong: at ``GROUND`` 0.62 the
darkest object-against-table contrast in this palette is **1.02:1** and an
aluminium panel and the stage floor become the same colour. At 0.28 it is 2.22:1.
So the answer is a *ladder*, not a band -- lift the achromatic scenery off the
floor and spend the rest of the range on separation.

**The three accents did not move.** They were chosen against a near-black stage
and the worry was that they would stop reading against a lighter one. Against
this ground they sit at 2.2 to 3.7:1, which is what the ladder is for: the
scenery moved up into the 45-90 band and the accents were already above it.

A textured asset states no colour at all: see
``clear_texture_tints`` (the scene tier's build helpers) for why a colour on a
textured shape is actively harmful under the tiled renderer.
"""

from __future__ import annotations


def srgb_to_linear(rgb) -> tuple[float, float, float]:
    """The sRGB transfer function, inverted: display values 0..1 to linear albedo.

    Every colour in this module is a **display** value -- what the picture should
    look like -- and every renderer that shades physically wants linear albedo.
    The two are not the same number and the gap is large in the midtones, so a
    backend that skips this step draws a bench four stops too bright.

    Both backends need it, which is why it is here rather than private to one:
    the path tracer decodes into ``UsdPreviewSurface.diffuseColor``, and the
    browser page decodes into a glTF ``baseColorFactor``. Both of those are
    defined as linear; a viser ``color=`` triple is not, and must NOT come
    through here -- three.js converts that one itself on ingest.
    """
    out = []
    for value in rgb:
        v = min(max(float(value), 0.0), 1.0)
        out.append(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
    return (out[0], out[1], out[2])


__all__ = [
    "ACCENT_PEG",
    "ACCENT_PLUG",
    "ACCENT_PORT",
    "ACCENT_PROP",
    "BENCH_LIGHT",
    "BUILD_ORDER",
    "DEFORMABLE",
    "FIXTURE",
    "FLOOR",
    "GEAR_BRASS",
    "GEAR_BRONZE",
    "GEAR_STEEL",
    "GROUND",
    "MACHINE",
    "MAT_LIGHT",
    "METAL",
    "OBJECT",
    "ROBOT_DARK",
    "ROLE_DEFAULTS",
    "SORT_BENCH",
    "SORT_SOURCE",
    "SORT_TARGET",
    "SORT_TRAY",
    "SUBSTRATE",
    "build_order_color",
    "default_for",
    "is_build_order_color",
    "srgb_to_linear",
]


def _hex(code: str) -> tuple[float, float, float]:
    """An sRGB hex triple as the 0-1 floats ``shape_color`` wants.

    Written as hex because that is how the four below were chosen and how they
    would be checked against the source. Transcribing them to three decimals by
    hand is a step that can go wrong silently and cannot be diffed against the
    palette they came from.
    """
    value = code.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


# ------------------------------------------------------------------- scenery
#
# One cool achromatic family, laid out as a ladder rather than a band. The
# rendered 8-bit values under the tiled tracer are given because they are the
# point of the numbers, and because a value that looks reasonable in source can
# be black on the only renderer that matters.

FLOOR: tuple[float, float, float] = (0.20, 0.20, 0.21)
"""The ground plane every scene stands on. Renders 45 lit, 29 in shadow.

Separate from :data:`GROUND`, and a step below it, because the two are different
surfaces and the role vocabulary already separates them: ``world`` is the floor,
``surface`` is the table. Painted at the same value the table vanishes
into it. This is also the one surface nobody was ever stating -- the compiler
called ``add_ground_plane`` without a colour, so every scene shipped with
Newton's ``(0.125, 0.125, 0.15)``, pixel 27, whatever its table was doing.
"""

GROUND: tuple[float, float, float] = (0.28, 0.28, 0.29)
"""Table tops and anything else the task merely happens on top of. 63 lit, 43 shadowed.

Was ``(0.07, 0.07, 0.08)``, which rendered at 15 and 7 -- a table indistinguishable
from the void behind it. This is the value the crush measurement above settles
on: high enough that nothing is crushed, low enough that everything a task is
about still reads against it.
"""

MACHINE: tuple[float, float, float] = (0.22, 0.22, 0.23)
"""A powered tool, a stand, an enclosure -- equipment that is not the robot. 49 lit.

Exists because three suites needed it and none of them had it: ``polish``'s
``TOOL_COLOR`` was ``(0.10, 0.10, 0.11)``, which renders at **22** and is the
black cylinder silhouette in its recorded video. Dark on purpose -- machinery
should be the heaviest thing in frame -- but not a hole.
"""

BENCH_LIGHT: tuple[float, float, float] = (0.55, 0.55, 0.57)
"""A light-grey table top, for a cell whose subject is small metal parts. 140 lit.
:data:`GROUND` renders at 72 on the gear bench's overhead camera and the path
tracer, calibrated to the tiled frame, puts it there too -- a dark grey slab
under dark grey fixtures, the frame a person asked to have lightened. Two
stops up and still cool, so a warm part still reads as not scenery."""

MAT_LIGHT: tuple[float, float, float] = (0.62, 0.60, 0.56)
"""A pale, faintly warm work mat. 158 lit. Lighter than :data:`FIXTURE` by the
same two stops as :data:`BENCH_LIGHT`, and warmer than the steel gear
(173/178/189, cool) that lies on it, so the two do not merge."""

FIXTURE: tuple[float, float, float] = (0.40, 0.40, 0.43)
"""Benches, pads, mounts -- structure that is present but is not the subject. 90 lit.

Faintly cool, which puts it in the same family as :data:`GROUND` and
:data:`FLOOR`. That is deliberate: the scenery reads as one thing, so that
anything warm in frame reads as *not* scenery.
"""

METAL: tuple[float, float, float] = (0.68, 0.70, 0.74)
"""Machined surfaces: an aluminium panel, a stainless fixture. 155 lit.

Light and faintly cool, which is what aluminium reads as under a neutral rig.
Exists beside :data:`SUBSTRATE` because the two answer different scenes: a tape
is peeled off a cardboard box (warm), a protective film off a metal panel --
and a film whose whole point is a blue tint needs a substrate that is not
fighting it for warmth.
"""

# ---------------------------------------------------------------- manipulands
#
# Warm, against the cool scenery above, so that a thing on a table separates
# from the table by hue as well as by value.

OBJECT: tuple[float, float, float] = (0.60, 0.54, 0.47)
"""A free rigid body nobody stated a colour for. 137 lit.

Deliberately *not* an accent. A sorting cell has ten manipulands; painting them
all saturated is the opposite of the rule at the top of this module, which is
that the one saturated thing in frame is the one that has to be found. This is
the "you did not say" colour, and the appearance test is what keeps it rare.
"""

SUBSTRATE: tuple[float, float, float] = (0.74, 0.58, 0.40)
"""What a deformable is stuck to or draped over. Warm, so the sheet reads off it.

Tuned to shipping-carton kraft: the scene that uses it is the tape suite's box,
and a taped carton is the picture that box is standing in for.
"""

DEFORMABLE: tuple[float, float, float] = (0.88, 0.88, 0.87)
"""Cloth, cable, tape -- the things whose *shape* is the observation. 201 lit.

Was 0.92, which is a bug rather than a preference: **pyrender clips to 255 at
albedo 0.9121**, so every deformable it drew came out pure white and had done
since the value was chosen. 0.88 renders 248 there and 201 on tiled.

Only a chain of capsules actually receives this. A cloth is not a shape, so the tiled tracer
never looks up a colour for one and draws it white regardless, whatever a cloth
spec states. One small consolation of the
new value: at 0.88 the two backends disagree about a deformable by 0.12 rather
than by 0.08 of nothing, since tiled was going to draw 1.0 either way.
"""

ACCENT_PORT: tuple[float, float, float] = (0.30, 0.55, 0.82)
"""The receptacle half of a mating pair: the fixed thing being inserted *into*."""

ACCENT_PLUG: tuple[float, float, float] = (0.90, 0.62, 0.20)
"""The moving half of a mating pair.

Warm against :data:`ACCENT_PORT`'s cool, because on the insertion suites the
whole question a frame has to answer is where the plug is relative to the bore,
and two blues a centimetre apart do not answer it.
"""

# ------------------------------------------------------------------- alloys
#
# Three gears of one module that a policy must tell apart by SIZE, and a person
# by eye. They were tried in three saturated hues and that read as toys; they
# were tried all in METAL and the segmenter stopped finding them -- three
# identical steel discs on a grey mat, and a two-gear board would be read as a
# one-gear board (measured: 5 of 8 gear trials against 8 of 8). So they are
# three ALLOYS, which is what a real gear set is: the same brushed-metal finish
# under the path tracer, told apart by the colour the metal actually is.

GEAR_STEEL: tuple[float, float, float] = (0.68, 0.70, 0.74)
"""The 16-tooth gear: plain steel. METAL's own value, and the finish table keys brushed metal to it."""

GEAR_BRASS: tuple[float, float, float] = (0.78, 0.66, 0.32)
"""The 20-tooth gear: yellow brass. Warm and clearly not the steel, and far enough off the pinion's
saturated orange that the drive still reads as the one thing on the board nobody handles."""

GEAR_BRONZE: tuple[float, float, float] = (0.72, 0.50, 0.38)
"""The 24-tooth gear: phosphor bronze. The reddest of the three and the largest, so size and colour
say the same thing."""


ACCENT_PEG: tuple[float, float, float] = (0.86, 0.82, 0.68)
"""A peg a part is lowered onto: warm ivory, the brightest thing on its plate.

A peg is a 11 mm shaft on a near-black plate, and from every working camera
most of what shows is a face the key light does not reach -- so it renders at
the 0.2 ambient factor alone, and at METAL that is pixel 82 against a plate
at 41: a dark post on black. At this albedo the unlit face is 104 and the lit
one 163. Warm rather than DEFORMABLE's neutral white so a peg and a cloth are
not the same colour, and ivory rather than the pinion's orange so the target
and the drive do not read as one thing."""


ACCENT_PROP: tuple[float, float, float] = (0.85, 0.35, 0.12)
"""Free-standing props a task routes around -- pegs, posts, obstacles."""

# --------------------------------------------------------------- sorting cells
#
# One palette rather than four independent choices, and warm rather than the
# achromatic scheme above, because a sorting cell is the one case the rule at the
# top of this file does not serve. "Mostly achromatic with one saturated accent"
# works when there is one thing to find; a bin-to-bin sort has ten, they are all
# saturated already -- printed grocery cartons and painted hand tools -- and the
# scenery has to recede behind them rather than compete.
#
# The four are https://colorhunt.co/palette/622b14995f2f978f66e4d6a9, taken as a
# set. They are ordered light-to-dark in a way that does the work: the deepest is
# the bench everything sits on, the palest is the floor an object is read
# against, and the two mid tones separate the bin things come *out of* from the
# bin they go *into*.
#
# **These did not move with the rest of the palette**, and that is not an
# oversight. They were chosen as mid-tones and never had the crush problem: they
# render at 87 to 204 as they stand. Lifting them would break the hex round trip
# this module's test exists to protect, in exchange for nothing.

SORT_BENCH: tuple[float, float, float] = _hex("#622B14")
"""The workbench a sorting cell stands on. The darkest of the four.

Dark for the reason :data:`GROUND` is: a surface that is merely underneath
everything should be the thing your eye leaves first.
"""

SORT_SOURCE: tuple[float, float, float] = _hex("#995F2F")
"""The bin objects come out of."""

SORT_TARGET: tuple[float, float, float] = _hex("#978F66")
"""The bin objects go into.

A different tone from :data:`SORT_SOURCE` on purpose, and it gives nothing away:
which *compartment* a thing belongs in is what the printed plates say and what
the task is about, and telling the two bins apart at a glance is something the
geometry already does. What it buys is a person driving the page by hand.
"""

SORT_TRAY: tuple[float, float, float] = _hex("#E4D6A9")
"""The floor inside a bin, and the palest of the four.

Both compartment plates and settled objects are read against this, so it is the
one surface in the cell that wants to be light: a dark tool on a dark floor is
the picture a policy is asked to sort from.
"""


ROBOT_DARK: tuple[float, float, float] = (0.30, 0.30, 0.30)
"""A robot whose asset states no usable colour of its own.

Matched to the ``black`` material the well-authored YAM MJCF gives its links, so
an embodiment that needs a stated colour lands on the same grey as one that does
not. **Deliberately left where it was** when the scenery moved: 30 of the 38
visible shapes in a cloth-folding frame are this robot, so it is tempting to
treat it as the reason the frame is dark -- but lifting it moves the frame mean
by 1.8 levels and costs the match to the reference rig the embodiment's own
appearance pass exists to hold. The scenery was the problem; the robot is
supposed to be the heavy thing in the picture.
"""


# ------------------------------------------------------- what "unstated" means

ROLE_DEFAULTS: dict[str, tuple[float, float, float]] = {
    "surface": GROUND,
    "fixtures": FIXTURE,
    "objects": OBJECT,
    "deformables": DEFORMABLE,
    "hands": ROBOT_DARK,
    "robot": ROBOT_DARK,
    "self": FIXTURE,
    "world": FLOOR,
}
"""One default per declared scene role.

Keyed on the role rather than matched on a name, for the reason the roles are
declared at all: a domain nobody anticipated falls through every branch of a
name match and nothing says so. The role vocabulary is already declared, already
validated and already closed, so a table keyed on it cannot silently miss a case.
"""


def default_for(role: str) -> tuple[float, float, float]:
    """The colour a shape of this role takes when nobody stated one.

    An unknown role is not an error here. Appearance is the last thing that
    should stop a scene building, and a domain this table has not heard of is
    better drawn in the object colour than not drawn at all.
    """
    return ROLE_DEFAULTS.get(role, OBJECT)


# ------------------------------------------------- detecting the colour nobody chose

BUILD_ORDER: tuple[tuple[int, int, int], ...] = (
    (68, 119, 170),
    (102, 204, 238),
    (34, 136, 51),
    (204, 187, 68),
    (238, 102, 119),
    (170, 51, 119),
    (238, 153, 51),
    (0, 153, 136),
)
"""``ModelBuilder._SHAPE_COLOR_PALETTE``, copied so this module imports nothing.

Paul Tol's Bright 8, which is a *categorical data-viz* palette: it is built to
make N adjacent categories maximally distinguishable, which is the right job for
a bar chart and precisely the wrong one for a photograph. Used as a scene
palette it renders a five-part hammer as five unrelated hues.

Copied rather than imported because this module is the one place in the package
with no dependencies, and because a private name is not an import contract. The
copy is a liability -- a newton release that reorders this turns every check
below into a no-op that passes -- so the test suite asserts the two are still
equal (``test_build_order_is_newtons_palette``). That test is what makes the
copy safe.
"""


def build_order_color(index: int) -> tuple[float, float, float]:
    """The colour ``ModelBuilder`` would have given the shape at ``index``."""
    r, g, b = BUILD_ORDER[index % len(BUILD_ORDER)]
    return (r / 255.0, g / 255.0, b / 255.0)


def is_build_order_color(color: object, index: int) -> bool:
    """Whether this shape's colour is the one the builder guessed for it.

    The only reliable way to tell "an importer stated this" from "nobody did".
    After ``build()`` there is nothing else in the builder that records which
    happened, and testing membership in :data:`BUILD_ORDER` is the wrong question
    -- a task may legitimately want one of those eight colours. Equality against
    *this index's* entry cannot be hit by accident, and if it were, the render
    would be identical anyway.
    """
    want = build_order_color(int(index))
    try:
        have = tuple(float(channel) for channel in color)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return False
    return len(have) >= 3 and all(abs(a - b) <= 1.0e-6 for a, b in zip(have[:3], want, strict=True))
