# SPDX-License-Identifier: Apache-2.0
"""Pacing a live render against the loop it takes the time from.

A live view renders *inside* the loop that is stepping the solver, on the same
thread and the same CUDA stream, so every frame it draws is time the simulation
does not get. That makes "how often should I render?" a budget question rather
than a frame-rate one, and a target frame rate alone answers it wrongly in the
direction that hurts: ask for 15 fps of a path tracer that takes 300 ms and the
loop spends every wall-clock second in the renderer, the step rate collapses,
and the picture that arrives is of a scene that has stopped moving.

:class:`RenderBudget` is that arithmetic, and nothing else -- a clock, two
limits and a division:

* **target_fps** -- the most frames you want, when they are cheap.
* **max_load** -- the largest share of wall clock you will spend rendering,
  *measured* rather than assumed.

The interval is whichever limit is longer, so a cheap backend runs at the rate
you asked for and an expensive one self-limits instead of eating the loop: at a
measured 38 ms a frame and a 0.3 cap, one frame every 127 ms (7.9 fps), with no
knob touched. Gating is on wall clock rather than on step counts, so a loop
paced to a control rate and one running flat out both get the same picture rate
-- and a scene that slows down does not silently drag the view down with it.

The cost is a measured EMA, and the first measurement is *replaced* rather than
blended: the first frame of any backend is unrepresentative -- a tiled sensor
compiles its megakernel, and an OVRTX renderer does not exist until the first
frame asks for one, which is a USD export and a scene load, measured at six
seconds against the eighteen milliseconds every frame after it costs. A one-off
six-second frame folded into an average would throttle the view to a crawl for
the next minute. Until two frames have been measured the load cap is
not applied at all, so the warm-up frames are paced by the frame-rate floor and
the cap starts from a number it can trust.

What this deliberately does not do is decide *when* the frame is worth drawing.
A caller that renders one state twice has wasted the whole budget on a duplicate
image, so the caller checks that (a step counter is enough) -- this only answers
"may I, yet?".
"""

from __future__ import annotations

__all__ = ["RenderBudget"]


class RenderBudget:
    """How often a live render may run, given a target rate and a share of wall clock.

    Args:
        target_fps: Frames per second to aim for when they are affordable. 0
            means "as often as the load cap allows", which is the right setting
            for a view that should simply take its share and no more.
        max_load: Largest fraction of wall-clock time to spend rendering, in
            (0, 1]. This is the limit that actually protects the loop.
        smoothing: EMA weight on the newest cost measurement.
        window: Seconds between refreshes of the reported :attr:`fps` and
            :attr:`load`.

    Attributes:
        cost: Smoothed seconds per rendered frame, 0 before the first frame.
        fps: Frames per second achieved over the last closed window.
        load: Share of wall clock spent rendering over the last closed window.
            This is the number to read: it is what the loop actually paid, so a
            view configured to cost 30% and measured at 55% is saying the
            caller is rendering somewhere the gate does not know about.
        frames: Frames recorded since the last :meth:`reset`.
    """

    def __init__(
        self,
        *,
        target_fps: float = 15.0,
        max_load: float = 0.3,
        smoothing: float = 0.3,
        window: float = 1.0,
    ) -> None:
        self.target_fps = float(target_fps)
        self.max_load = float(max_load)
        self.smoothing = min(max(float(smoothing), 0.01), 1.0)
        self.window = max(float(window), 0.05)
        self.reset()

    def reset(self) -> None:
        """Forget every measurement. A different backend costs what it costs.

        Called when the thing being measured changes -- another backend, another
        resolution, another camera count -- because carrying a tiled sensor's
        4 ms over to a path tracer's 18, let alone to the six seconds its first
        frame spends building an OVRTX scene, means the first path-traced frame
        arrives before the budget has any idea what it just authorised.
        """
        self.cost = 0.0
        self.fps = 0.0
        self.load = 0.0
        self.frames = 0
        self._started: float | None = None
        self._window_start: float | None = None
        self._window_frames = 0
        self._window_spent = 0.0

    # ------------------------------------------------------------------ gating
    @property
    def interval(self) -> float:
        """Seconds to wait between the *starts* of two renders.

        Measured start to start rather than end to start, which is what makes
        the load cap exact: over one interval the loop spends :attr:`cost`
        rendering, so ``cost / interval`` is the load, and solving that for the
        cap gives the second term below.
        """
        fps = max(float(self.target_fps), 0.0)
        floor = 1.0 / fps if fps > 0.0 else 0.0
        # The cap needs a cost it can trust; until then the floor paces things.
        if self.frames < 2:
            return floor
        return max(floor, self.cost / min(max(float(self.max_load), 1.0e-3), 1.0))

    def due(self, now: float) -> bool:
        """Whether a frame may be rendered at wall-clock time ``now``."""
        return self._started is None or (now - self._started) >= self.interval

    def record(self, *, started: float, cost: float) -> None:
        """Book a frame that began at ``started`` and took ``cost`` seconds."""
        cost = max(float(cost), 0.0)
        self._started = float(started)
        # Replace, do not blend, while the measurement is still warm-up: see the
        # module docstring. Two samples in and the EMA takes over.
        self.cost = cost if self.frames < 2 else (1.0 - self.smoothing) * self.cost + self.smoothing * cost
        self.frames += 1

        finished = float(started) + cost
        if self._window_start is None:
            self._window_start = float(started)
        self._window_frames += 1
        self._window_spent += cost
        span = finished - self._window_start
        if span >= self.window:
            self.fps = self._window_frames / span
            self.load = self._window_spent / span
            self._window_start, self._window_frames, self._window_spent = finished, 0, 0.0
