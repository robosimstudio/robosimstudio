# SPDX-License-Identifier: Apache-2.0
"""Watching a whole wave at once, in a browser.

A fleet is one model holding every world's geometry, so drawing all of them is
already what the viewer does -- ``log_state`` walks the model, and every world's
shapes are in it. The only thing missing is that the worlds are *superimposed*:
replication puts them all at the origin, deliberately, because that is what is
numerically best for the physics.

So this is almost entirely a matter of asking the viewer for its own grid.
``set_world_offsets`` moves each world's drawing by a per-world translation at
draw time, leaving the simulation where it is. The result is a wall of robots,
each running its own episode, from one process and one step loop.

Two things this adds beyond the offsets, and both are about reading it rather
than rendering it:

* **A per-world label**, saying which episode a world is running and what its
  graph is doing. Watching thirty-two arms move is not the same as knowing
  which of them has already succeeded, and the difference matters most exactly
  when something is going wrong in one world and not the others.
* **A wall-clock draw budget.** The fleet's step loop is the thing being
  watched, so drawing is paced on the clock rather than on steps. Measured on
  ``tool_sort/level1_6`` at two frames a second, steady-state drawing costs
  essentially nothing up to 64 worlds (80.5 -> 82.9 steps/s at 16, 64.1 -> 65.5
  at 64, both inside the noise) and about 7% at 256. What is *not* free is the
  first frame: the initial scene upload to the browser is ~3.4 s whatever the
  world count, which over a short run is easy to mistake for a per-frame cost.

  So the binding constraint on a watchable grid is not the drawing, it is the
  per-world realtime factor -- ``tool_sort`` runs 256 worlds at 1.01x wall clock
  and 512 at 0.78x, while one towel is already 0.58x at a single world.

At large widths the escape hatch is ``max_worlds``, which restricts the viewer to
a window -- five hundred robots is not a picture anybody reads, and the viewer
draws every world it is shown.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from robosimstudio.core.math import quat_to_mat
from robosimstudio.core.streams import reading

__all__ = ["FleetView"]


def _place(pose: np.ndarray, centre: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Carry a box through one transform, staying axis-aligned.

    ``pose`` is a Newton transform, ``(position, quaternion)`` with the
    quaternion stored xyzw. The rotated box's own axis-aligned half-extent is
    ``abs(R) @ half`` -- the furthest corner along each axis -- which is exact
    rather than a bounding-sphere over-estimate, and matters because the shapes
    that decide a workcell's footprint are flat panels standing on edge.
    """
    rotation = quat_to_mat(np.asarray(pose[3:7], dtype=np.float64))
    return np.asarray(pose[:3], dtype=np.float64) + rotation @ centre, np.abs(rotation) @ half


class FleetView:
    """A live grid of every world in a wave.

    Args:
        fleet: The batched environment being stepped.
        spacing: Grid spacing between worlds [m]. ``None`` measures the scene
            and applies ``margin``.
        margin: How much wider than a world's own measured extent to space the
            grid, when ``spacing`` is not given. The extent is the box its
            *shapes* fill (see :meth:`_tight_spacing`), so 1.15 is a real gap of
            fifteen percent of a cell on every side of it -- about 110 mm on the
            0.76 x 0.80 m ``tool_hanging`` bench. The viewer's own default is 1.5
            *and* rounds up to a whole metre, which on a two-metre workcell puts
            the robots four metres apart -- a grid of distant islands rather than
            a wall of robots. Just over 1.0 puts them shoulder to shoulder, which
            is what makes a fleet legible as one picture.
        port: Where to serve the page.
        share: Whether to open a public tunnel.
        max_worlds: Draw at most this many. A wall of five hundred robots is not
            a picture anybody reads, and the viewer renders every world it is
            shown -- so this is a legibility limit and a cost one at once.
        viewer: An existing viewer to draw into. ``None`` opens one, which is
            what a standalone fleet page does; the studio passes its own so a
            world-count change reloads in place.
        fps: Draw rate ceiling, paced on the wall clock rather than on steps.
            Two is close to free below 64 worlds and about 7% at 256 (measured);
            lower it if a run's numbers matter more than watching it.
    """

    def __init__(
        self,
        fleet,
        *,
        spacing: tuple[float, float, float] | None = None,
        margin: float = 1.15,
        port: int = 8080,
        share: bool = False,
        max_worlds: int = 64,
        fps: float = 2.0,
        viewer: Any = None,
    ) -> None:
        from .garments import scene_repaints_live, viewer_studio

        self.fleet = fleet
        # An existing viewer is *borrowed*, not owned: the studio hands its own
        # over so that switching world count reloads the scene in the tab the
        # browser already has, rather than dropping the page and serving a new
        # one on a new port.
        self.owns_viewer = viewer is None
        self.viewer = viewer if viewer is not None else viewer_studio(port=port, share=share)
        self.viewer.set_model(fleet.model)
        # A grid of a scene that repaints itself as it runs draws it in every
        # world it shows, or in none of them: the colour array is one model's.
        setter = getattr(self.viewer, "set_live_colors", None)
        if setter is not None:
            setter(scene_repaints_live(getattr(fleet, "scene", None)))
        self._mesh_bounds: dict[int, np.ndarray] = {}
        """Vertex bounds per mesh source, so a CoACD hull set is walked once."""
        self.shown = list(range(min(int(max_worlds), int(fleet.num_worlds))))
        if len(self.shown) < fleet.num_worlds:
            self.viewer.set_visible_worlds(self.shown)
        grid = tuple(float(v) for v in spacing) if spacing is not None else self._tight_spacing(margin)
        if grid is not None:
            self.viewer.set_world_offsets(grid)
        self.interval = 1.0 / max(0.1, float(fps))
        self._last = 0.0
        self._labels: dict[int, Any] = {}
        self._captions: dict[int, str] = {}
        self.pending_spacing: float | None = None
        """A pitch the page asked for, applied by the next draw on the main thread."""
        # Only claim a port when this opened one. Borrowing the studio's viewer
        # leaves ``port`` unused, and printing it says the page is somewhere it
        # is not.
        where = f" on port {port}" if self.owns_viewer else ""
        print(f"[fleet-view] showing {len(self.shown)} of {fleet.num_worlds} worlds{where}")

    def _tight_spacing(self, margin: float) -> tuple[float, float, float] | None:
        """Grid spacing from what a world actually *occupies*.

        Measured from the **shapes**, not from the body origins and not from the
        viewer's own extents.

        Not the viewer's, because its ``_get_world_extents`` includes the
        **ground plane**, which is a 20-metre slab -- so it reports 20 m for a
        two-metre workcell and spaces the grid thirty metres apart, which is a
        field of distant islands rather than a wall of robots.

        Not the body origins, because they are *points*. A body origin says
        nothing about how far its geometry reaches from it, and every static
        shape -- the bench, the pegboard, the walls, which is most of what a
        workcell looks like -- hangs off no body at all and so was invisible to
        the measurement entirely. On ``tool_hanging/wrenches_level2`` at 256 worlds
        that under-measured the cell by about a metre and the pegboard panels of
        neighbouring worlds visibly interpenetrated.

        So: one world's shapes, each grown to its own half-extent, in world
        frame, per axis. The pitch is that extent times ``margin``, which is what
        leaves a visible gap between one cell and the next.

        Falls back to the viewer's own choice when there is nothing to measure.
        """
        extent = self._occupied_extent()
        if extent is None:
            return None
        spacing = [max(float(extent[axis]), 0.5) * float(margin) for axis in range(3)]
        spacing[int(self.viewer.model.up_axis)] = 0.0
        return tuple(spacing)

    def _occupied_extent(self) -> np.ndarray | None:
        """One world's world-frame bounding box, as a per-axis size [m].

        The worlds are superimposed at the origin, so "one world" is the point:
        measuring every shape in the model would measure the same box, only
        slower. This takes world 0's shapes -- both the ones carried by a body
        and the ones welded to the arena -- and unions their boxes.

        ``None`` when the model cannot be read or has nothing but ground.
        """
        try:
            model = self.fleet.model
            shapes = self._template_shapes(model)
            if shapes.size == 0:
                return None
            body_q = np.asarray(self.fleet.snapshot().body(0), dtype=np.float64)
            boxes = self._shape_boxes(model, shapes, body_q)
        except Exception:
            return None
        if not boxes:
            return None
        lower = np.min([box[0] for box in boxes], axis=0)
        upper = np.max([box[1] for box in boxes], axis=0)
        return np.asarray(upper - lower, dtype=np.float64)

    def _template_shapes(self, model) -> np.ndarray:
        """World 0's shapes. Global-world shapes -- the ground -- are excluded.

        ``shape_world`` is the model's own answer and the one to trust; the
        wave's own stride map is the fallback for a model that does not carry
        it, and is right for the same reason every other index in ``WorldMap``
        is: replication lays the copies out world-major.
        """
        world_of = getattr(model, "shape_world", None)
        if world_of is not None:
            values = world_of.numpy() if hasattr(world_of, "numpy") else np.asarray(world_of)
            return np.flatnonzero(np.asarray(values).reshape(-1) == 0).astype(np.int64)
        span = self.fleet.worlds.shapes(0)
        return np.arange(span.start, min(span.stop, int(model.shape_count)), dtype=np.int64)

    def _shape_boxes(self, model, shapes: np.ndarray, body_q: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Every shape as a world-frame axis-aligned box.

        Two frames to walk: a shape sits at ``shape_transform`` in its body's
        frame, and the body sits at ``body_q`` in the world's -- except a static
        shape, whose ``shape_body`` is -1 and whose transform is already world
        frame. Rotating a box gives a box: the axis-aligned half-extent of a
        rotated one is ``abs(R) @ half``, which is exact for the corner that
        sticks out furthest and is what "how much room does this need" means.

        Planes and heightfields are skipped -- the ground plane is the 20-metre
        slab this measurement exists to avoid, and a floor every world shares is
        not part of what a world occupies.
        """
        import newton

        types = model.shape_type.numpy()
        scale = model.shape_scale.numpy()
        xform = model.shape_transform.numpy()
        owner = model.shape_body.numpy()
        radius = model.shape_collision_radius.numpy()
        skipped = {int(newton.GeoType.PLANE), int(newton.GeoType.HFIELD), int(newton.GeoType.NONE)}
        boxes: list[tuple[np.ndarray, np.ndarray]] = []
        for shape in shapes.tolist():
            kind = int(types[shape])
            if kind in skipped:
                continue
            centre, half = self._local_box(model, shape, kind, scale[shape], float(radius[shape]))
            centre, half = _place(np.asarray(xform[shape], dtype=np.float64), centre, half)
            body = int(owner[shape])
            if 0 <= body < len(body_q):
                centre, half = _place(np.asarray(body_q[body], dtype=np.float64), centre, half)
            boxes.append((centre - half, centre + half))
        return boxes

    def _local_box(self, model, shape: int, kind: int, scale, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """One shape's own box, in its own frame, as ``(centre, half-extent)``.

        Newton's ``shape_scale`` means something different per type -- a radius,
        a radius and a half-height, three half-extents, a mesh scale factor --
        so each is spelled out rather than guessed at. A mesh is measured from
        its vertices, which is the only way to catch geometry that is not
        centred on its own origin: a pegboard panel modelled as a wall standing
        off to one side reads as a metre of nothing if you assume it is.
        """
        import newton

        size = np.asarray(scale, dtype=np.float64).reshape(-1)
        zero = np.zeros(3)
        if kind == int(newton.GeoType.SPHERE):
            return zero, np.full(3, size[0])
        if kind in (int(newton.GeoType.CAPSULE), int(newton.GeoType.CYLINDER), int(newton.GeoType.CONE)):
            # Radius and half-height about local +z; a cone is measured as the
            # cylinder that contains it.
            return zero, np.array(
                [size[0], size[0], size[1] + (size[0] if kind == int(newton.GeoType.CAPSULE) else 0.0)]
            )
        if kind in (int(newton.GeoType.BOX), int(newton.GeoType.ELLIPSOID)):
            return zero, np.abs(size[:3])
        mesh = self._mesh_box(model.shape_source[shape])
        if mesh is not None:
            lower, upper = mesh * np.abs(size[:3])
            return (lower + upper) * 0.5, (upper - lower) * 0.5
        # A source this cannot read -- an SDF, a Gaussian splat -- falls back to
        # the bounding sphere Newton already computed for the broad phase.
        return zero, np.full(3, float(radius))

    def _mesh_box(self, source) -> np.ndarray | None:
        """A mesh's unscaled vertex bounds, cached by mesh.

        A CoACD decomposition is dozens of shapes over one source and a fleet
        asks this once per shape, so the vertex pass is memoised on the mesh
        rather than repeated per hull.
        """
        vertices = getattr(source, "vertices", None)
        if vertices is None:
            return None
        cache = self._mesh_bounds
        key = id(source)
        bounds = cache.get(key)
        if bounds is None:
            points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
            if points.size == 0:
                return None
            bounds = np.stack([points.min(axis=0), points.max(axis=0)])
            cache[key] = bounds
        return bounds

    def set_spacing(self, spacing: float) -> None:
        """Re-space the grid, live. Draw-time only -- the physics does not move.

        Main thread only: the offsets are a new device array, allocated and
        copied on the process-global current stream. The page's slider assigns
        :attr:`pending_spacing` instead and the next draw applies it.
        """
        grid = [float(spacing)] * 3
        grid[int(self.viewer.model.up_axis)] = 0.0
        self.viewer.set_world_offsets(tuple(grid))
        # Labels sit above their world, so they follow the offsets.
        self._captions.clear()

    @property
    def spacing(self) -> float:
        """The grid pitch currently in use [m]."""
        offsets = self._offsets()
        if offsets is None or len(offsets) < 2:
            return 0.0
        return float(np.max(np.abs(offsets[1] - offsets[0])))

    def on_tick(self, fleet, slots, step: int) -> None:
        """Draw, if enough wall clock has passed. The runner's wave-tick hook."""
        now = time.perf_counter()
        if now - self._last < self.interval:
            return
        self._last = now
        self.viewer.begin_frame(step * fleet.control_dt)
        # Serialized against every other reader: a draw is hundreds of device
        # reads (one per shape batch), each of which moves the device's
        # process-global current stream, and a world's script thread may be
        # reading at the same moment. This is by far the largest source of
        # them -- measured at 3590 per thirty seconds against 640 for the next
        # one. See :mod:`robosimstudio.core.streams`.
        with reading():
            spacing, self.pending_spacing = self.pending_spacing, None
            if spacing is not None:
                self.set_spacing(spacing)
            self.viewer.log_state(fleet.pipeline.state_0)
            self.viewer.end_frame()
            # Inside the guard too: the labels read world_offsets, which is a
            # device array like any other.
            self._update_labels(slots)

    def _update_labels(self, slots) -> None:
        """One line per world: its episode, and what it is doing now.

        Rewritten only when the text changes. A label is a scene-graph node, and
        replacing thirty-two of them every draw is more traffic than the geometry
        that made the view worth having.
        """
        offsets = self._offsets()
        for index, world in enumerate(self.shown):
            slot = slots[world] if world < len(slots) else None
            caption = self._caption(world, slot)
            if self._captions.get(world) == caption:
                continue
            self._captions[world] = caption
            position = offsets[index] if offsets is not None and index < len(offsets) else np.zeros(3, np.float32)
            self._draw_label(world, caption, position)

    def _offsets(self) -> np.ndarray | None:
        """Where the viewer put each world, as host values.

        ``world_offsets`` is a device array -- it is what the draw kernels read
        -- so it has to be brought across before anything can be indexed out of
        it. Read once per draw rather than once per label.
        """
        offsets = getattr(self.viewer, "world_offsets", None)
        if offsets is None:
            return None
        values = offsets.numpy() if hasattr(offsets, "numpy") else np.asarray(offsets)
        return np.asarray(values, dtype=np.float32).reshape(-1, 3)

    def _caption(self, world: int, slot) -> str:
        if slot is None or getattr(slot, "parked", True):
            return f"w{world} idle"
        episode = getattr(slot.episode, "episode_id", "?")
        if not getattr(slot, "live", True):
            verdict = "SUCCESS" if slot.success else (slot.close_kind or "done")
            return f"w{world} ep{episode} {verdict}"
        return f"w{world} ep{episode} {self._status(world)}"

    def _status(self, world: int) -> str:
        """What the policy says this world is doing, if it says anything."""
        policy = getattr(self, "policy", None)
        try:
            return policy.status(world) if policy is not None else "running"
        except Exception:
            return "running"

    def _draw_label(self, world: int, caption: str, position: np.ndarray) -> None:
        """Put one world's caption above it, however this viewer does labels.

        Probed rather than assumed: the label API is a viser convenience the
        Newton viewer exposes indirectly, and a view that refuses to open
        because a helper was renamed is worse than one that draws the fleet with
        no captions. The geometry is the point; the text is the annotation.
        """
        above = (float(position[0]), float(position[1]), float(position[2]) + 1.4)
        handle = self._labels.get(world)
        if handle is not None:
            try:
                handle.text = caption
                # The grid can be re-spaced under a label; it follows its world.
                handle.position = above
                return
            except Exception:
                self._labels.pop(world, None)
        server = getattr(self.viewer, "server", None)
        scene = getattr(server, "scene", None)
        if scene is None or not hasattr(scene, "add_label"):
            return
        try:
            self._labels[world] = scene.add_label(f"/fleet/w{world}", text=caption, position=above)
        except Exception:
            pass

    def bind(self, policy) -> None:
        """Read per-world status from the policy driving the wave."""
        self.policy = policy

    def close(self) -> None:
        """Retire the labels, and the viewer if this view opened it.

        A borrowed viewer is left open for its owner. One this view opened is
        closed, or its server keeps the port bound until the process exits.
        """
        for handle in self._labels.values():
            try:
                handle.remove()
            except Exception:
                pass
        self._labels.clear()
        self._captions.clear()
        if self.owns_viewer:
            try:
                self.viewer.close()
            except Exception:
                pass
