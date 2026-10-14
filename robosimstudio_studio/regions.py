# SPDX-License-Identifier: Apache-2.0
"""Drawing the boxes a task is scored against.

A goal region is the geometry a placement clause measures, and until this the
page never showed one. That is a strange gap for a page whose whole job is
watching a task: you could drag a hammer toward a bin, watch the score stay zero,
and have no way to see that the region ends four centimetres short of where the
bin's rim suggests it does.

**Axis-aligned, always, and that is a fact about regions rather than a shortcut.**
A ``RegionHandle`` translates with its frame and never rotates -- ``containment``
compares a world point against world-axis-aligned bounds offset by the frame's
current origin, so a region riding a drawer slides out with it and stays square
to the world. A box drawn with the frame's orientation would look more correct
and would be lying about what the goal measures.

The box is drawn through the same ``region_bounds`` the atom uses, so the picture
and the verdict cannot disagree.

**Verdict tinting is free.** Every clause writes its verdict into ``metrics``
under its own label, so colouring a region by whether its clause holds costs a
dict lookup rather than an evaluation, and the page never runs the goal an extra
time to draw itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .panels import Panel

__all__ = ["PanelRegions"]

ROOT = "/studio/regions"

IDLE_COLOR = (150, 160, 175)
"""A region nothing is currently satisfying. Cool grey-blue: present, not loud."""

HELD_COLOR = (120, 200, 140)
"""A region whose clause holds right now. Green, and nothing latches -- a tool
knocked back out stops being green, which is the honest reading of a goal stated
over the world's current state."""

FILL_OPACITY = 0.16
"""Enough to read as a volume, little enough to see the object inside it. A
region is a place you are trying to put something, so the something has to stay
visible through it."""

LABEL_RISE = 0.02
"""How far above a region's top face its name floats [m]. Clear of the lid, so a
label does not z-fight with its own box."""


class PanelRegions(Panel):
    """Every goal region, drawn where it currently is.

    Rebuilt on load and re-posed as frames move; toggled off by default for a
    scene with none, which is most rigid tasks that score on something other
    than placement.
    """

    def __init__(self, session) -> None:
        super().__init__(session)
        self.show = True
        self.show_labels = True
        self.tint = True
        self._nodes: dict[str, Any] = {}
        self._synced = -1
        self._tinted: dict[str, tuple[int, int, int]] = {}
        # Which clause, if any, is about each region -- found once, by walking the
        # goal for leaves whose measure is a containment and whose second operand
        # names a region. Duck-typed and defensive: a hand-written evaluator has
        # no tree, and a goal that scores something other than placement has no
        # such leaves, both of which are ordinary rather than exceptional.
        self._clause: dict[str, str] = {}
        goal = getattr(self.env, "goal", None)
        if goal is not None:
            for leaf in _leaves(goal):
                atom, operands = getattr(leaf, "about", ("", ()))
                if atom == "containment" and len(operands) >= 2:
                    self._clause.setdefault(operands[1], leaf.label)

    @property
    def regions(self) -> dict:
        return dict(getattr(self.env.handles, "regions", {}) or {})

    def build(self, gui) -> None:
        if not self.regions:
            return
        with gui.add_folder("Regions", expand_by_default=False):
            show = gui.add_checkbox("Show", initial_value=self.show)
            show.on_update(lambda _: setattr(self, "show", bool(show.value)))
            labels = gui.add_checkbox("Names", initial_value=self.show_labels)
            labels.on_update(lambda _: self._set_labels(bool(labels.value)))
            tint = gui.add_checkbox("Tint when satisfied", initial_value=self.tint)
            tint.on_update(lambda _: setattr(self, "tint", bool(tint.value)))
            self._readout = gui.add_text("Regions", initial_value=f"{len(self.regions)}", disabled=True)

    def _set_labels(self, on: bool) -> None:
        # Forget the last sync so the next drain re-poses even on a paused
        # page, which is where labels are put up or taken down.
        self.show_labels = on
        self._synced = -1

    def bounds(self, name: str, body_q) -> tuple[np.ndarray, np.ndarray]:
        """One region's world corners, through the atom's own arithmetic.

        Not a second implementation: `containment` decides what is inside, and a
        box drawn from anything else would be a picture of a different region.
        """
        from robosimstudio.assetlib.vocabulary.regions import region_bounds

        handle = self.regions[name]
        region = {"xy_ranges": handle.xy_ranges, "z_range": handle.z_range}
        return region_bounds(region, handle.origin_now(body_q))

    def _color(self, name: str) -> tuple[int, int, int]:
        if not self.tint:
            return IDLE_COLOR
        label = self._clause.get(name)
        if not label:
            # No clause is about this region -- always so on a scene with no
            # task, which has no evaluator and so no ``last_result`` to ask.
            return IDLE_COLOR
        metrics = getattr(getattr(self.env, "last_result", None), "metrics", {}) or {}
        return HELD_COLOR if metrics.get(label) else IDLE_COLOR

    def _clear(self) -> None:
        for node in self._nodes.values():
            try:
                node.remove()
            except Exception:
                pass
        self._nodes.clear()
        self._tinted.clear()
        self._synced = -1

    def drain(self) -> None:
        """Draw, re-pose and re-tint. Main thread, once per step."""
        if not self.show or not self.regions:
            if self._nodes:
                self._clear()
            return
        # One readback for every region, and only when something has moved.
        # A region on a static frame never needs one at all, but a page with a
        # drawer in it does, and asking per region would pay for the array once
        # per box.
        step = int(getattr(self.env, "step_count", 0))
        if step == self._synced and self._nodes:
            self._retint()
            return
        self._synced = step
        body_q = self.env.state.body_q.numpy() if self.env.model.body_count else None

        for name in self.regions:
            lo, hi = self.bounds(name, body_q)
            centre = (lo + hi) / 2.0
            size = np.maximum(hi - lo, 1e-4)
            box = self._nodes.get(f"{name}/box")
            if box is None:
                self._nodes[f"{name}/box"] = self.server.scene.add_box(
                    f"{ROOT}/{name}/box",
                    color=self._color(name),
                    dimensions=tuple(float(v) for v in size),
                    position=tuple(float(v) for v in centre),
                    opacity=FILL_OPACITY,
                    cast_shadow=False,
                    receive_shadow=False,
                    side="double",
                )
                self._tinted[name] = self._color(name)
            else:
                box.position = tuple(float(v) for v in centre)
            if self.show_labels:
                label = self._nodes.get(f"{name}/label")
                where = (float(centre[0]), float(centre[1]), float(hi[2]) + LABEL_RISE)
                if label is None:
                    self._nodes[f"{name}/label"] = self.server.scene.add_label(
                        f"{ROOT}/{name}/label", name, position=where
                    )
                else:
                    label.position = where
            else:
                # Names unticked: take the label down rather than leave it
                # frozen where the region last was.
                label = self._nodes.pop(f"{name}/label", None)
                if label is not None:
                    label.remove()
        self._retint()

    def _retint(self) -> None:
        """Recolour only what changed, since a colour write is a viser message.

        Runs with tint off too: :meth:`_color` is then ``IDLE_COLOR``, which is
        what repaints a box left green when the box was unticked.
        """
        for name in self.regions:
            box = self._nodes.get(f"{name}/box")
            colour = self._color(name)
            if box is not None and self._tinted.get(name) != colour:
                box.color = colour
                self._tinted[name] = colour

    def refresh(self) -> None:
        readout = getattr(self, "_readout", None)
        if readout is None:
            return
        held = sum(1 for name in self.regions if self._color(name) == HELD_COLOR)
        scored = len(self._clause)
        readout.value = f"{len(self.regions)} drawn, {scored} scored, {held} satisfied"

    def close(self) -> None:
        self._clear()


def _leaves(node) -> tuple:
    """Every measuring leaf under a goal, without importing the asset layer.

    A leaf is a node with no children; walking for that rather than asking
    `assetlib.goal.leaves` keeps this panel's construction free of an import that
    would otherwise happen at module scope, in a layer that must not have one.
    """
    children = getattr(node, "children", None)
    kids = children() if callable(children) else ()
    if not kids:
        return (node,)
    return tuple(leaf for child in kids for leaf in _leaves(child))
