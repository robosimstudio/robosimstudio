# SPDX-License-Identifier: Apache-2.0
"""The generic benches the library ships: a flat table at three heights.

**The only cells shipped with the library, and the roster is deliberately
short.** Somebody using ``robosimstudio`` to evaluate their own robot on their
own scenes wants ``CellSpec`` and ``register_cell``; they do not want a
particular room somebody measured. These three are here because they are not
particular: a rectangular table at a stated height with a stated work surface,
which is what a task needs when it has no opinion about the room. A cell that
*is* particular -- an imported arena, a pegboard workcell -- ships with the task
that measured it.

All three are the same shape -- a rectangular table centred at ``(0.6, 0)`` with
a pair of arms mounted behind its near edge -- and two of them differ in exactly
one number: the height of the top. That 22 mm is why they have to be two scenes
and not one, and why the height belongs here at all.

Both now declare their table. The other thing they declare is the
**work-surface frame**, which is what the embodiments' mounts are derived
against.
"""

from __future__ import annotations

from robosimstudio.core.render.palette import BENCH_LIGHT, GROUND
from robosimstudio.scene.registry import register_cell
from robosimstudio.specs import CellSpec, StaticSpec


def _table(top_z: float, half_xy, *, solid: bool, color=GROUND) -> StaticSpec:
    """The cell's table, as a static box whose top is the work surface.

    ``solid`` is the only real choice here. The three suites that shared this
    cell each drew their own slab and disagreed about it -- 30 mm, a full-height
    block, and 20 mm of non-colliding decor 5 mm low -- which is what a table
    being nobody's responsibility looks like. One table, and a solid one: a
    30 mm slab floating at knee height reads as an error, and nothing in any of
    the three touches below the top.

    ``color`` because a cell may repaint the one table it declares -- the
    tool-hanging bench matches its table to its holders -- without that being a
    second table.
    """
    half_z = top_z / 2.0 if solid else 0.015
    return StaticSpec(
        name="table",
        half_extents=(float(half_xy[0]), float(half_xy[1]), half_z),
        xform=(0.6, 0.0, top_z - half_z, 0.0, 0.0, 0.0, 1.0),
        color=color,
    )


BENCH_075 = register_cell(
    CellSpec(
        name="bench_075",
        ground_z=0.0,
        surface_z=0.75,
        center=(0.6, 0.0, 0.0),
        half=(0.2975, 0.65, 0.375),
        statics=(_table(0.75, (0.2975, 0.65), solid=True),),
        provenance="the original reference workcell, shared by the USB-C, cable-routing and (retired) RJ45 reference sims",
    )
)

BENCH_075_LIGHT = register_cell(
    CellSpec(
        name="bench_075_light",
        ground_z=0.0,
        surface_z=0.75,
        center=(0.6, 0.0, 0.0),
        half=(0.2975, 0.65, 0.375),
        # The same bench with a light top, for small metal parts that the
        # palette's GROUND slab put on a frame a person asked to have
        # lightened. A cell of its own so the suites that measured their
        # vision against the dark top keep it.
        statics=(_table(0.75, (0.2975, 0.65), solid=True, color=BENCH_LIGHT),),
        provenance="bench_075 with a light-grey top, for small metal parts a dark palette hides",
    )
)

BENCH_0728 = register_cell(
    CellSpec(
        name="bench_0728",
        ground_z=0.0,
        surface_z=0.728,
        center=(0.6, 0.0, 0.0),
        half=(0.30, 0.40, 0.364),
        statics=(_table(0.728, (0.30, 0.40), solid=True),),
        provenance="the YAM arena's play table: tape_peel",
    )
)

BENCH_0728_WIDE = register_cell(
    CellSpec(
        name="bench_0728_wide",
        ground_z=0.0,
        surface_z=0.728,
        center=(0.6, 0.0, 0.0),
        # 0.80 x 0.80 m of top, against ``bench_0728``'s 0.60 x 0.80. Only the
        # depth moved, and it moved to a measured number rather than a
        # comfortable one: Newton's ``example_cloth_franka`` -- the example that
        # ships ``unisex_shirt.usd`` and the parameters ``SHIRT_JERSEY`` is
        # converted from -- stands the same garment on a table of
        # ``hx = hy = 40 cm``. Matching it is what lets the shirt be simulated at
        # the size it was authored.
        #
        # The garment is why. Laid out as authored it settles to 0.65 x 0.70 m,
        # and on the 0.60 m top it overhung the near edge: the hem drew itself
        # over the lip and hung below the surface, where no crease can be struck
        # through it. (0.57 x 0.69 was the figure for the subdivision cage the
        # asset was re-sourced from; the remesh settles a little flatter and so a
        # little longer, and the verdict is unchanged.) That is not a cloth problem and no amount of
        # tuning reaches it -- the cloth-folding suite's shirt measurement reports
        # the margin.
        half=(0.40, 0.40, 0.364),
        statics=(_table(0.728, (0.40, 0.40), solid=True),),
        provenance="the YAM arena's play table, deepened to Newton example_cloth_franka's 80x80 cm",
    )
)
