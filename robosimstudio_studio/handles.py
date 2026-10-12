# SPDX-License-Identifier: Apache-2.0
"""Draggable handles: more than one at a time, and a way to move them as one.

A handle is a 6-DOF gizmo in the browser with something in the scene tied to it
-- a rigid body under a PD wrench, or a patch of a deformable under a pin. The
panels in :mod:`~robosimstudio_studio.panels` own the physics; what
lives here is the part that is the same either way: where the gizmo is, which
slot it occupies, and what happens to the *other* handles while one is dragged.

Two things it exists for.

**A handle is live the moment it is placed.** Placement and grabbing are the
same act, and a gizmo that is visible is one that is holding something.

**Several handles at once.** One grip can only translate a sheet; two can fold
it, stretch it, hold one corner while the other is carried, which is why
PhysTwin's playground exposes ``--n_ctrl_parts 2`` and why a cloth page wants
the same. Slots are fixed at construction because each one owns a dragger whose
kernel launch has to be in the captured graph -- see
:class:`~robosimstudio.core.ParticleDragger` and the note in
:mod:`~robosimstudio.core.pipeline` about hooks added after capture. An unused
slot is an inactive dragger, which launches over nothing.

**Moving them together** is one gesture, not two: with :attr:`HandleSet.linked`
set, the rigid transform that carries the dragged handle from where the gesture
started to where it is now is applied to every other live handle. So two grips
on a towel translate and rotate as a pair -- the sheet is carried rather than
pinched -- and one mouse is enough. Unlinked, the others stay where they were
and the drag is a single-point pull.

The viser threading contract from :mod:`~robosimstudio_studio.panels`
holds: the gizmo callback assigns plain attributes only (it does capture the
poses a linked drag is measured from, which is numpy and nothing else), and
every write *back* to the browser happens in :meth:`HandleSet.drain` on the main
thread. Writing a handle the client is dragging would fight it for the pose.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from robosimstudio.core.math import quat_inverse, quat_multiply, quat_to_mat, quat_wxyz_to_xyzw, quat_xyzw_to_wxyz

__all__ = ["HANDLE_COLORS", "GrabHandle", "HandleSet"]

HANDLE_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 176, 46),
    (64, 208, 255),
    (168, 120, 255),
    (120, 230, 140),
)
"""Marker colour per slot, so two handles on one garment are told apart.

The gizmo itself is red/green/blue by axis and says nothing about which handle
it is; the bead at its origin is the only thing that can.
"""

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
"""xyzw identity, copied on use -- these are mutable and handed out."""


@dataclass(frozen=True)
class _Drag:
    """Where every live handle was when the current gesture started.

    Snapshotted rather than integrated, so a linked drag cannot drift: each
    update recomputes the others' poses from the anchor and the total motion of
    the dragged one, instead of accumulating per-update deltas that a dropped or
    coalesced message would leave short.
    """

    index: int
    position: np.ndarray
    quat: np.ndarray
    anchors: tuple[tuple[int, np.ndarray, np.ndarray], ...]


class GrabHandle:
    """One gizmo, its pose, and what it is holding.

    The pose is authoritative on the host: the gizmo callback writes it, the
    panel reads it to steer a dragger, and :meth:`sync` pushes it back to the
    browser when something other than the browser moved it. The handle knows
    *what* it holds only as an opaque :attr:`key` and a display :attr:`label` --
    a body index here, a cloth name there -- because it is the panel that knows
    how to let go of one.

    Args:
        scene: ``server.scene``.
        name: Scene-graph path for the gizmo. The marker bead is added *under*
            it, so it is carried by the gizmo without a per-step pose write.
        index: Which slot this is, for the readout and the colour.
        scale: Gizmo size [m].
        line_width: Gizmo line width.
        color: Marker colour.
        marker_radius: Marker radius [m]. Small: it marks the grab point on a
            surface, and a bead that swallows the point it marks is worse than
            none.
    """

    def __init__(
        self,
        scene: Any,
        name: str,
        *,
        index: int = 0,
        scale: float = 0.10,
        line_width: float = 2.5,
        color: tuple[int, int, int] = (255, 255, 255),
        marker_radius: float = 0.008,
    ) -> None:
        self.index = int(index)
        self.position = np.zeros(3, dtype=np.float64)
        self.quat = IDENTITY.copy()
        self.live = False
        """Whether something is tied to this handle. A live handle is visible."""
        self.dragging = False
        """Whether the browser is mid-gesture on it, so the host leaves it alone."""
        self.seq = -1
        """Placement order, so a full set recycles the handle placed longest ago."""
        self.key: Any = None
        """What the owning panel tied to this slot; ``None`` when free."""
        self.label = ""
        """How the readout names it."""

        # depth_test=False, unlike the arm cursor: a handle sits *on* the thing
        # it holds -- a body's origin is inside its mesh, a grab point is on a
        # sheet -- so a depth-tested gizmo is half-buried in whatever it is
        # holding. Drawing it over the scene is what every DCC gizmo does, and
        # it is the difference between a handle you can see and one you hunt for.
        self.gizmo = scene.add_transform_controls(
            name, scale=scale, line_width=line_width, depth_test=False, visible=False
        )
        self.marker = scene.add_icosphere(f"{name}/marker", radius=marker_radius, color=color, visible=False)

    # ------------------------------------------------------------------- state
    def place(self, position, quat=None, *, key: Any = None, label: str = "") -> None:
        """Put the handle here and make it live. Main thread.

        Placement order is :meth:`HandleSet.acquire`'s to set, not this one's --
        a slot is ordered when it is taken, whether or not the grab it was taken
        for turns out to land.
        """
        self.position = np.asarray(position, dtype=np.float64).copy()
        if quat is not None:
            self.quat = np.asarray(quat, dtype=np.float64).copy()
        self.key = key
        self.label = label
        self.live = True
        self.dragging = False
        self.sync()

    def clear(self) -> None:
        """Let go: hide the gizmo and free the slot. Main thread."""
        self.live = False
        self.dragging = False
        self.key = None
        self.label = ""
        self.gizmo.visible = False
        self.marker.visible = False

    def sync(self) -> None:
        """Push the host's pose to the browser. Main thread, never mid-gesture."""
        self.gizmo.position = tuple(float(v) for v in self.position)
        self.gizmo.wxyz = tuple(float(v) for v in quat_xyzw_to_wxyz(self.quat))
        self.gizmo.visible = self.live
        self.marker.visible = self.live

    def remove(self) -> None:
        """Give the scene nodes back. Idempotent."""
        for node in (self.marker, self.gizmo):
            try:
                node.remove()
            except Exception:
                pass


class HandleSet:
    """A fixed number of handles, allocated by clicking and dragged as one.

    Args:
        scene: ``server.scene``.
        prefix: Scene-graph path prefix. Slots are siblings under it, never
            nested -- a gizmo parented to another gizmo would inherit its
            transform and the two would move together whether or not they were
            linked.
        count: How many handles. Fixed for the life of the set, because a slot
            costs a dragger and a dragger costs a hook the captured graph has to
            already hold.
        scale: Gizmo size [m].
        line_width: Gizmo line width.
        linked: Start with the handles moving as one rigid group.
    """

    def __init__(
        self,
        scene: Any,
        prefix: str,
        *,
        count: int = 2,
        scale: float = 0.10,
        line_width: float = 2.5,
        linked: bool = False,
    ) -> None:
        self.handles = [
            GrabHandle(
                scene,
                f"{prefix}_{i}",
                index=i,
                scale=scale,
                line_width=line_width,
                color=HANDLE_COLORS[i % len(HANDLE_COLORS)],
            )
            for i in range(max(1, int(count)))
        ]
        self.linked = bool(linked)
        """Whether a drag on one handle carries the others with it."""
        self._drag: _Drag | None = None
        for handle in self.handles:
            self._listen(handle)

    def _listen(self, handle: GrabHandle) -> None:
        """Bind the gizmo callback. viser thread -- assignments only."""

        @handle.gizmo.on_update
        def _(event) -> None:
            # The pose *before* this update, which is where a gesture that is
            # only now being recognised must be measured from. viser reports
            # phase="start" at the pose the grab began, so on that phase this is
            # the same value; keeping it also makes the set work if a client
            # ever sends updates without a start.
            previous = handle.position
            previous_quat = handle.quat
            handle.position = np.asarray(event.target.position, dtype=np.float64)
            handle.quat = quat_wxyz_to_xyzw(event.target.wxyz)
            phase = str(getattr(event, "phase", "update"))
            handle.dragging = phase != "end"
            if phase == "end":
                self._drag = None
                return
            drag = self._drag
            if phase == "start" or drag is None or drag.index != handle.index:
                # One snapshot object, assigned whole: drain() reads it on the
                # main thread and must never see half of a new gesture.
                self._drag = _Drag(
                    index=handle.index,
                    position=np.asarray(previous, dtype=np.float64).copy(),
                    quat=np.asarray(previous_quat, dtype=np.float64).copy(),
                    anchors=tuple((h.index, h.position.copy(), h.quat.copy()) for h in self.handles if h.live),
                )

    # ------------------------------------------------------------------ access
    def __iter__(self) -> Iterator[GrabHandle]:
        return iter(self.handles)

    def __len__(self) -> int:
        return len(self.handles)

    def __getitem__(self, index: int) -> GrabHandle:
        return self.handles[index]

    @property
    def live(self) -> list[GrabHandle]:
        """The handles currently holding something, in slot order."""
        return [h for h in self.handles if h.live]

    @property
    def dragging(self) -> bool:
        """Whether the browser is mid-gesture on any handle."""
        return any(h.dragging for h in self.handles)

    def find(self, key: Any) -> GrabHandle | None:
        """The live handle holding ``key``, or ``None``.

        What stops one body being held by two handles at once, which would put
        two wrenches on it and let the page fight itself.
        """
        return next((h for h in self.handles if h.live and h.key == key), None)

    def acquire(self, key: Any = None) -> GrabHandle:
        """Take the slot a new grab should use, and mark it taken.

        A free slot if there is one, the handle holding ``key`` if that is what
        is being re-seated, and otherwise the one placed longest ago. Recycling
        rather than refusing is deliberate: a page whose handles are all taken
        would otherwise answer a click by doing nothing, and the fix -- press
        Release, then click again -- is exactly the button dance direct
        manipulation is here to remove. The caller still has to let go of
        whatever the recycled slot was holding.

        The slot is reserved on the way out, so two acquisitions in a row cannot
        hand back the same handle. It is not *shown* until :meth:`GrabHandle.place`
        pushes a pose to it -- a reserved slot whose grab turns out to miss is
        cleared with nothing having appeared -- which is why reserving costs the
        caller only the duty to place it or clear it.
        """
        held = self.find(key) if key is not None else None
        chosen = held or next((h for h in self.handles if not h.live), None) or min(self.handles, key=lambda h: h.seq)
        HandleSet._seq += 1
        chosen.seq = HandleSet._seq
        chosen.live = True
        chosen.key = key
        return chosen

    _seq = 0
    """Placement order, counted across *every* set rather than per set.

    Within one set only the ordering matters, and either would do. Across two it
    has to be the same clock: the page's own ``Release last`` has to let go of
    the last thing *you* took hold of, and on a scene with a cloth and a box
    that is a choice between two sets. Two counters both starting at zero made
    "handle 3" on the cloth look exactly as recent as "handle 3" on the objects.
    """

    def newest(self) -> GrabHandle | None:
        """The live handle placed most recently, or ``None``."""
        live = self.live
        return max(live, key=lambda h: h.seq) if live else None

    # -------------------------------------------------------------------- loop
    def drain(self) -> None:
        """Carry the un-dragged handles with the dragged one. Main thread.

        Nothing happens unless a gesture is in flight and the set is linked. The
        dragged handle is not written back -- the browser owns its pose until it
        says the drag has ended.
        """
        drag = self._drag
        if drag is None or not self.linked:
            return
        dragged = self.handles[drag.index]
        # The rigid motion the gesture has applied so far, as a rotation about
        # the dragged handle's start pose plus a translation. Applying it to the
        # others makes the whole set behave like one body, so two grips on a
        # sheet swing around each other rather than each spinning in place.
        rotation = quat_multiply(dragged.quat, quat_inverse(drag.quat))
        basis = quat_to_mat(rotation)
        for index, position, quat in drag.anchors:
            if index == drag.index:
                continue
            other = self.handles[index]
            if not other.live or other.dragging:
                continue
            other.position = dragged.position + basis @ (position - drag.position)
            other.quat = quat_multiply(rotation, quat)
            other.sync()

    def release_all(self) -> None:
        """Let every handle go. Main thread. The caller detaches the physics."""
        self._drag = None
        for handle in self.handles:
            handle.clear()

    def close(self) -> None:
        """Give every scene node back. Main thread, idempotent."""
        self._drag = None
        for handle in self.handles:
            handle.remove()
        self.handles = []

    # ----------------------------------------------------------------- display
    def summary(self, empty: str = "nothing") -> str:
        """One line naming what is held, for a panel's readout."""
        live = self.live
        if not live:
            return empty
        return "  |  ".join(f"{h.index + 1}: {h.label}" for h in live)
