# SPDX-License-Identifier: Apache-2.0
"""What a cell is: the place.

A :class:`CellSpec` is the room -- its work surface, its mounts, the scenery it
draws. The arrangement that *steps* a cell is an
:class:`~robosimstudio.specs.physics.ArrangementSpec`, which lives beside it in
:mod:`~robosimstudio.specs.physics`; both are properties of the cell rather than
of the task standing in it, and a document may name either only by a
registered, versioned name.

**A document may never state a solver number.** ``(:staging (:physics ...))``
resolves a name against the arrangement registry, and the values live with the
content that measured them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .asset import AssetSpec, StaticSpec
from .base import Vec3, Xform

__all__ = ["CellSpec"]


@dataclass(frozen=True, kw_only=True)
class CellSpec:
    """Where a task happens: the ground, the work surface, and the furniture.

    The third registered axis, beside the embodiment and the task. A scene owns
    everything a task merely *happens on top of* -- the floor, the table, later
    the backdrop -- and owns none of what the task is about. That split is what
    lets one task be staged in several places and one place host several tasks.

    **The work-surface frame is the contract.** Its origin is :attr:`center` in
    x and y with z at :attr:`surface_z`, world-axis-aligned. A task states its
    fixtures and object poses in that frame and an embodiment states its mount
    in it, so moving a task to a taller table is a change of one number here
    rather than an edit to every coordinate in the suite. A scene with no table
    leaves ``surface_z`` at the ground, which is the "objects sit on the floor"
    configuration rather than a special case of anything.

    Attributes:
        name: Registry key.
        ground_z: World z of the floor plane [m].
        surface_z: World z of the work surface's top [m]. Equal to
            :attr:`ground_z` when the scene has no table, and that equality is
            the whole of what "on the ground" means here.
        center: Origin of the surface frame in x and y [m]. The z component is
            ignored -- :attr:`surface_z` is the height, stated once.
        half: Half-extent of the work surface [m], or ``None`` if unbounded. Read
            by feasibility checks (does this task's layout fit?) and by table
            generation; never by physics directly.
        statics: The furniture, in **world** coordinates. A scene is authored as
            a place, so its own pieces are not expressed relative to itself.
        asset: An imported room or arena, instead of (or beside) ``statics``.
        mounts: Named robot mount points ``(name, xform)``, in the surface
            frame. A cell with a fixed pedestal names it here and a binding
            picks it, which is more honest than every task restating the same
            pose.
        provenance: Where this place came from.
    """

    name: str
    ground_z: float = 0.0
    surface_z: float = 0.0
    center: Vec3 = (0.0, 0.0, 0.0)
    half: Vec3 | None = None
    statics: tuple[StaticSpec, ...] = ()
    asset: AssetSpec | None = None
    mounts: tuple[tuple[str, Xform], ...] = ()
    provenance: str = ""

    @property
    def surface_frame(self) -> Xform:
        """The work-surface frame as a world transform.

        Axis-aligned by construction: a rotated work surface would make every
        task coordinate depend on the scene's yaw, and no scene has wanted one.
        """
        return (float(self.center[0]), float(self.center[1]), float(self.surface_z), 0.0, 0.0, 0.0, 1.0)

    @property
    def has_surface(self) -> bool:
        """Whether there is a raised work surface at all, rather than bare ground."""
        return abs(float(self.surface_z) - float(self.ground_z)) > 1.0e-9

    def mount(self, name: str) -> Xform:
        for key, xform in self.mounts:
            if key == name:
                return xform
        raise KeyError(f"scene {self.name!r} has no mount {name!r}; has {[k for k, _ in self.mounts]}")
