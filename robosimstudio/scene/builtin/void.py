# SPDX-License-Identifier: Apache-2.0
"""``void`` -- the ground plane and nothing else.

The degenerate scene, and the one that proves the axis is real. Its work surface
*is* the floor (``surface_z == ground_z == 0``), so a task staged here has its
fixtures and objects sitting directly on the ground rather than on a table.

Two things it is for. It is the **null hypothesis** for a portability bug: a
task that compiles here and not on a real cell has a problem with the cell, and
a task that fails here has stated a coordinate in world terms that it claimed
was surface-relative. And it is the honest home for anything genuinely
floor-level, which the shipped suites are not but a mobile-base task would be.

The ground plane itself comes from the compiler, which adds one to every scene;
this declares no statics because there is nothing else here.
"""

from __future__ import annotations

from robosimstudio.scene.registry import register_cell
from robosimstudio.specs import CellSpec

VOID = register_cell(
    CellSpec(
        name="void",
        ground_z=0.0,
        surface_z=0.0,
        center=(0.0, 0.0, 0.0),
        # Unbounded rather than large: a half-extent is a claim about where a
        # task fits, and the floor makes no such claim.
        half=None,
        provenance="none -- the empty scene",
    )
)
