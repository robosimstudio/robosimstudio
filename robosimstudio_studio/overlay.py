# SPDX-License-Identifier: Apache-2.0
"""Drawing what an asset declares, where it actually is.

An asset's features are the geometry its measures are taken against -- a bore's
mouth and axis, a peg's shaft, a hanger's loop, the strip of particles a hand is
supposed to take.

**Nothing here computes a transform.** Every payload goes through the goal
machinery's own feature reader -- the benchmark tier's ``goals.bind.read`` with
``form="feature"``, which is the reader every *measure* resolves its operands
through -- so what is drawn is literally what is measured, in the frame it is
measured in. That reader is reached through :mod:`~robosimstudio_studio.seams`
rather than imported, because it needs the context a scored step leaves behind:
a scene with no task has neither, and its body-borne features are not drawn
(the same as before the reader moved behind the seam). That is the same
discipline :mod:`~robosimstudio_studio.regions` follows for goal regions
by calling the containment atom's own ``region_bounds``, and for the same reason:
a picture computed a second way is free to disagree with the thing it depicts,
and a *disagreeing* picture is worse than none because it is believed.

The kind decides the shape, because the kind is what says which payload fields
are geometry at all:

======================  =================================================
``cavity``              the mouth, and a line down the bore along its axis
``rod``                 a segment between its two endpoints
``loop``                a ring about its normal
``axis``                a line through its origin
``grasp_box``           a box on the body, at the declared centre
``anchor``              a marker where the scene holds it
``set`` / ``panel``     the resolved particles, as a cloud
======================  =================================================

**Regions are not here**, and that is not an omission. ``PanelRegions`` already
draws every entry in ``handles.regions`` -- which includes the ones an *asset*
declares, because :func:`~robosimstudio.assetlib.lower.region_specs` lowers a
``region`` feature into exactly the same ``RegionSpec`` a document's does. It
also tints them by their goal clause, which no other kind has. Drawing them a
second time here would be two pictures of one box, which is the failure this
module's opening paragraph is about.

A cloth's sets are the one kind not read through ``bind``: a garment's features
are stated against the rest layout and resolved *once* into particle indices at
scene-ready, so the live thing to draw is where those particles are now --
``DeformableHandle.sets`` holds the resolved indices and the probe holds the positions.
Going through the payload instead would draw the recipe rather than the selection.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.core.math import quat_xyzw_to_wxyz
from robosimstudio.specs.physics import reads_as

from . import seams
from .panels import Panel

__all__ = ["PanelOverlay"]

ROOT = "/studio/features"
"""Where the overlay's nodes live, so a task swap can drop the lot."""

COLOR = (255, 190, 90)
"""Amber. Distinct from the region boxes' grey-green and from anything a scene
is likely to be, and readable against both the light and dark stages."""

HELD_COLOR = (120, 200, 140)
"""For a facet a satisfied goal clause is about -- the same green
``PanelRegions`` tints with, because it means the same thing."""

POINT_SIZE = 0.004
"""Radius of a drawn particle [m]. Under a towel's 10 mm cell so a set reads as
a band of points rather than as a solid sheet hiding the cloth beneath it."""

RING_SEGMENTS = 48
"""Points around a drawn loop. Enough that a 13 mm ring has no visible corners."""

MARKER = 0.006
"""Radius of a point marker [m]."""


class PanelOverlay(Panel):
    """Draws the features the inspector asks it to.

    Holds no state of its own beyond which are showing: the inspector's
    checkboxes append to ``session.overlay_request`` from viser's thread, and
    this drains it on the main thread -- the same contract every other panel
    keeps.

    Args:
        session: The page.
    """

    def __init__(self, session) -> None:
        super().__init__(session)
        self.showing: dict[tuple[str, str], list[Any]] = {}
        """``(subject key, feature name)`` -> the scene nodes drawn for it."""
        self._synced = -1
        """Step the drawing was last brought up to date at."""

    def on_reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        # A reset puts step_count back to 0, under ``_synced``: without this the
        # gate in drain() holds the features where they were until the new
        # episode catches up with the old one. -1 makes the next drain redraw.
        self._synced = -1

    # ------------------------------------------------------------------- drain
    def drain(self) -> None:
        """Service the checkboxes, then re-place what is showing. Main thread.

        **Not every step.** ``drain`` runs once per control step, and re-placing
        a feature costs viser messages -- so redrawing here at step rate would
        send frames the browser is never asked to render. ``render_every`` is
        the interval the page is refreshed at
        (:meth:`~robosimstudio_studio.session.StudioSession.refresh_view`),
        and drawing faster than that is waste by construction. A request from a
        checkbox is serviced immediately whatever the gate says: that one is a
        person waiting for something to appear.
        """
        # Popped one at a time rather than copied and cleared: a checkbox can
        # append from viser's thread between a copy and a clear, and that
        # request would be lost. ``list.pop`` is atomic, and this is the one
        # thread that takes from the list.
        requests = self.session.overlay_request
        while requests:
            try:
                key, feature, show = requests.pop(0)
            except IndexError:
                break
            if show:
                self._show(key, feature)
            else:
                self._hide((key, feature))
        if not self.showing:
            return
        step = int(getattr(self.env, "step_count", 0))
        # ``_synced`` below zero is "never drawn": draw now, whatever the step.
        if self._synced >= 0 and step - self._synced < max(1, int(getattr(self.session, "render_every", 1))):
            return
        self._synced = step
        self._replace()

    def _show(self, key: str, feature: str) -> None:
        self._hide((key, feature))
        subject = self.session.index.by_key(key) if self.session.index else None
        if subject is None:
            return
        nodes = self._draw(subject, feature)
        if nodes:
            self.showing[(key, feature)] = nodes

    def _hide(self, key: tuple[str, str]) -> None:
        for node in self.showing.pop(key, ()):
            self._drop(node)

    @staticmethod
    def _drop(node: Any) -> None:
        """Take one node out of the scene, tolerating one that is already gone.

        A task swap retires the whole subtree from under this panel, so removing
        twice is normal rather than exceptional and is not worth a traceback on a
        page somebody is looking at.
        """
        try:
            node.remove()
        except Exception:
            pass

    def _replace(self) -> None:
        """Bring everything showing up to where it now is.

        A bore on a receptacle being inserted into moves every step and a set of
        particles moves continuously, so most of this has to be re-derived rather
        than re-posed -- a ring about a rotated normal is not the old ring at a
        new pose.

        **Re-added by name, never removed first.** viser's scene graph is
        addressed by name, so adding a node that already exists updates it: one
        message. Removing and re-adding is a delete *and* a create, which for a
        drawn particle set is the whole cloud twice, and this panel is the one
        that would be doing it while a policy runs. ``PanelRegions`` re-adds its
        boxes the same way and for the same reason.

        A redraw that yields fewer nodes than last time -- a payload that has
        stopped being placeable -- would leave the extras behind, so anything not
        re-drawn is removed explicitly.
        """
        for (key, feature), previous in list(self.showing.items()):
            subject = self.session.index.by_key(key) if self.session.index else None
            drawn = self._draw(subject, feature) if subject is not None else []
            live = {node.name for node in drawn}
            for node in previous:
                if node.name not in live:
                    self._drop(node)
            if drawn:
                self.showing[(key, feature)] = drawn
            else:
                self.showing.pop((key, feature), None)

    # -------------------------------------------------------------------- draw
    def _draw(self, subject, feature: str) -> list[Any]:
        """Every node for one feature, or an empty list if it cannot be placed."""
        record = self._record(subject)
        if record is None:
            return []
        try:
            spec = record.feature(feature)
        except KeyError:
            return []

        prefix = f"{ROOT}/{subject.kind}_{subject.name}_{feature}"
        colour = self._colour(subject, feature)
        if reads_as(record.substrate, "cloth"):
            return self._draw_particles(subject, feature, prefix, colour)

        payload = self._placed(subject, record, spec)
        if payload is None:
            return []
        return self._draw_kind(spec.kind, payload, subject, prefix, colour)

    def _placed(self, subject, record, spec) -> dict[str, Any] | None:
        """One feature's payload in world coordinates, through the goal's reader.

        Returns ``None`` rather than raising for anything unplaceable -- a
        feature on a body that has no pose, a snapshot not taken yet. A drawing
        that cannot be made is a drawing that is not made; it is not an error on
        a page somebody is looking at.
        """
        context = getattr(self.env, "last_context", None)
        if context is None:
            return None
        # A context is what a scored step leaves behind, so wherever there is
        # one the benchmark half is there to read it; ``None`` is only the
        # answer for something that merely looks like one.
        bench = seams.benchmark_or_none()
        return bench.feature_payload(context, subject, record, spec) if bench is not None else None

    def _draw_kind(self, kind: str, payload: dict, subject, prefix: str, colour) -> list[Any]:
        scene = self.server.scene
        if kind == "cavity":
            return self._draw_cavity(scene, payload, prefix, colour)
        if kind == "rod":
            return self._draw_rod(scene, payload, prefix, colour)
        if kind == "loop":
            return self._draw_loop(scene, payload, prefix, colour)
        if kind == "axis":
            return self._draw_axis(scene, payload, prefix, colour)
        if kind == "grasp_box":
            return self._draw_grasp_box(scene, payload, subject, prefix, colour)
        if kind == "anchor":
            return self._draw_anchor(scene, payload, prefix, colour)
        return []

    def _draw_cavity(self, scene, payload, prefix, colour) -> list[Any]:
        """The mouth, and the bore running from it along the insertion axis.

        Drawn as a line rather than as a tube because the useful question about a
        bore is which way it points and how deep it goes, and a translucent tube
        over a receptacle mesh hides the mesh it is describing.
        """
        mouth = _vec(payload.get("mouth"))
        axis = _unit(payload.get("axis"))
        if mouth is None or axis is None:
            return []
        depth = float(payload.get("depth", 0.02) or 0.02)
        nodes = [scene.add_icosphere(f"{prefix}/mouth", radius=MARKER, color=colour, position=tuple(mouth))]
        nodes.append(_segment(scene, f"{prefix}/bore", mouth, mouth + axis * depth, colour))
        return nodes

    def _draw_rod(self, scene, payload, prefix, colour) -> list[Any]:
        """A segment between the two endpoints, which is what a rod is.

        Both ends travel with the body -- ``rod``'s ``axis`` is declared as a
        pair of points for exactly that reason, and hanging past the end of one
        is not hanging.
        """
        axis = payload.get("axis")
        points = np.asarray(axis, dtype=np.float64) if axis is not None else None
        if points is None or points.ndim != 2 or len(points) < 2:
            return []
        return [_segment(scene, f"{prefix}/shaft", points[0], points[-1], colour)]

    def _draw_loop(self, scene, payload, prefix, colour) -> list[Any]:
        """A ring at the declared radius, in the plane the normal defines."""
        centre = _vec(payload.get("center"))
        normal = _unit(payload.get("normal"))
        radius = float(payload.get("inner_r", 0.0) or 0.0)
        if centre is None or normal is None or radius <= 0.0:
            return []
        u, v = _basis(normal)
        angles = np.linspace(0.0, 2.0 * np.pi, RING_SEGMENTS, endpoint=True)
        ring = centre + radius * (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v))
        return [scene.add_spline_catmull_rom(f"{prefix}/ring", positions=ring.astype(np.float32), color=colour)]

    def _draw_axis(self, scene, payload, prefix, colour) -> list[Any]:
        """A line through the origin, long enough to read as an axis.

        An ``axis`` declares no extent -- it is a directed line, not a segment --
        so the drawn length is a presentation choice and is stated here rather
        than implied: 12 cm each way, which spans a peg and reads at bench scale.
        """
        origin = _vec(payload.get("origin"))
        direction = _unit(payload.get("direction"))
        if origin is None or direction is None:
            return []
        span = 0.12
        return [_segment(scene, f"{prefix}/axis", origin - direction * span, origin + direction * span, colour)]

    def _draw_grasp_box(self, scene, payload, subject, prefix, colour) -> list[Any]:
        """The box the jaws are supposed to close on, oriented with the body.

        ``half`` is passed through by the feature reader untouched -- extents are
        not geometry to rotate -- so the box's *orientation* comes from the body
        it hangs off, which is the one thing here that is not in the payload.
        """
        half = _vec(payload.get("half"))
        centre = _vec(payload.get("center_local"))
        if half is None:
            return []
        pose = self._pose(subject)
        if pose is None:
            return []
        position = tuple(centre if centre is not None else pose[:3])
        return [
            scene.add_box(
                f"{prefix}/box",
                dimensions=tuple(2.0 * half),
                position=position,
                wxyz=_wxyz(pose[3:7]),
                color=colour,
                opacity=0.25,
            )
        ]

    def _draw_anchor(self, scene, payload, prefix, colour) -> list[Any]:
        """Where the scene holds this thing at episode start, when it says."""
        at = _vec(payload.get("at"))
        if at is None:
            return []
        return [scene.add_icosphere(f"{prefix}/anchor", radius=MARKER * 1.5, color=colour, position=tuple(at))]

    def _draw_particles(self, subject, feature: str, prefix: str, colour) -> list[Any]:
        """A garment's resolved set, where its particles are right now.

        The resolved indices, not the recipe. A cloth feature is a selector
        evaluated *once* against the rest layout, so the payload says
        ``beyond(axis: y, 0.06)`` and the thing worth drawing is the band of
        particles that produced -- which is what ``DeformableHandle.sets`` holds.
        """
        handle = subject.handle
        indices = None
        for key in (feature, f"{subject.name}/{feature}"):
            found = getattr(handle, "sets", {}).get(key)
            if found is not None:
                indices = np.asarray(found, dtype=np.int64)
                break
        if indices is None or not len(indices):
            return []
        state = getattr(self.env, "state", None)
        if state is None or getattr(state, "particle_q", None) is None:
            return []
        points = np.asarray(state.particle_q.numpy(), dtype=np.float32)[indices]
        colours = np.tile(np.asarray(colour, dtype=np.uint8), (len(points), 1))
        return [
            self.server.scene.add_point_cloud(f"{prefix}/set", points=points, colors=colours, point_size=POINT_SIZE)
        ]

    # ----------------------------------------------------------------- helpers
    def _record(self, subject):
        """This subject's declaration, or ``None`` when it has no asset."""
        from robosimstudio.assetlib.registry import find_asset

        return find_asset(subject.asset)

    def _pose(self, subject) -> np.ndarray | None:
        """The body pose a feature hangs off, however it is held.

        A free body's is in ``body_q``; a welded static's is on the handle,
        because a static has no ``body_q`` row at all. The feature reader makes
        the same distinction for the same reason.
        """
        handle = subject.handle
        body = int(getattr(handle, "body", -1))
        state = getattr(self.env, "state", None)
        if body >= 0 and state is not None and getattr(state, "body_q", None) is not None:
            return np.asarray(state.body_q.numpy()[body], dtype=np.float64)
        xform = getattr(handle, "xform", None)
        return np.asarray(xform, dtype=np.float64) if xform is not None else None

    def _colour(self, subject, feature: str):
        """Green while a goal clause about this feature holds, amber otherwise.

        Free, and the same convention the region boxes already use: the verdict
        is in ``metrics`` under the clause's label, so tinting costs a lookup
        rather than an evaluation.
        """
        goal = getattr(self.env, "goal", None)
        bench = seams.benchmark_or_none() if goal is not None else None
        if bench is None:
            return COLOR
        held = bench.feature_holds(goal, self.env.last_result.metrics, f"{subject.name}.{feature}")
        return HELD_COLOR if held else COLOR

    def close(self) -> None:
        super().close()
        for key in list(self.showing):
            self._hide(key)


def _segment(scene, name: str, start, end, colour):
    """One straight line. ``add_line_segments`` wants ``[n, 2, 3]``."""
    points = np.asarray([[start, end]], dtype=np.float32)
    return scene.add_line_segments(name, points=points, colors=colour, line_width=3.0)


def _vec(value) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return array[:3] if len(array) >= 3 else None


def _unit(value) -> np.ndarray | None:
    array = _vec(value)
    if array is None:
        return None
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 1e-12 else None


def _basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane ``normal`` is perpendicular to.

    The seed is chosen away from the normal so the cross product cannot come out
    degenerate, which it would for a loop whose normal happens to be +x.
    """
    seed = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, seed)
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


def _wxyz(quat) -> tuple[float, float, float, float]:
    """xyzw as Newton stores it, to wxyz as viser wants it."""
    return tuple(quat_xyzw_to_wxyz(np.asarray(quat, dtype=np.float64).reshape(-1)[:4]))
