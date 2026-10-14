# SPDX-License-Identifier: Apache-2.0
"""Frozen dataclass in, viser widgets out.

The existing studio hand-writes every widget, which is right for panels whose
controls are all different from each other -- a grasp button is not a slider
with different bounds. A material is the opposite case: sixteen numbers that
differ only in range, unit and what changing one costs, and all of that is now
recorded on the field itself by :mod:`robosimstudio.specs.params`. So the
panel can be generated, and a field added to a spec grows a control without
anybody editing a studio file.

Two conventions worth stating.

**Stiffnesses get a logarithmic control.** The useful range of ``edge_ke`` spans
six decades; on a linear slider every value below the top decade is in the first
pixel. Log fields are dragged in log10 and read back in their own units, which
is what the readout beside each one is for -- a slider showing ``0.82`` when the
value is ``6.6`` would be worse than no slider.

**Callbacks only record.** Viser fires them on its own thread, and Warp work on
another thread is how a session acquires a heisenbug. So a callback stores the
edit and :meth:`SpecBinder.drain` hands it over on the main thread, which is the
same contract the existing studio's panels keep.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import fields, replace
from typing import Any

from robosimstudio.specs.params import TIERS, ParamUI, iter_params

__all__ = ["TIER_MARK", "SpecBinder", "add_param_widget"]

TIER_MARK = {"live": "", "recapture": " ~", "rebuild": " *"}
"""Suffix marking what a field costs: nothing live, ``~`` recapture, ``*`` rebuild.

Terse on purpose -- it sits in a widget label, and the panel spells the legend
out once rather than every row repeating "requires rebuild".
"""

_LOG_STEP = 0.01
"""Slider granularity in log10, i.e. about 2.3% per notch."""


def _fmt(value: Any, ui: ParamUI) -> str:
    """One field's value, for a readout."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{value}"
    if value == 0:
        text = "0"
    elif abs(value) >= 1e4 or abs(value) < 1e-3:
        text = f"{value:.4g}"
    else:
        text = f"{value:.6g}"
    return f"{text} {ui.unit}".strip()


def add_param_widget(
    gui: Any, name: str, value: Any, ui: ParamUI, on_value: Callable[[Any], None]
) -> tuple[Any, Any | None]:
    """One control for one described parameter, and its readout if it has one.

    ``on_value`` is called with the edited value in the parameter's own units
    and type; it should only record (see the module docstring). A log field
    gets a readout beside it, returned second, ``None`` otherwise.
    """
    label = f"{name}{TIER_MARK.get(ui.tier, '')}"
    readout = None
    if isinstance(value, bool):
        widget = gui.add_checkbox(label, initial_value=value)
        widget.on_update(lambda _, w=widget: on_value(bool(w.value)))
    elif isinstance(value, str):
        if ui.choices:
            initial = value if value in ui.choices else ui.choices[0]
            widget = gui.add_dropdown(label, options=tuple(ui.choices), initial_value=initial)
        else:
            widget = gui.add_text(label, initial_value=value)
        widget.on_update(lambda _, w=widget: on_value(str(w.value)))
    elif ui.log and ui.range is not None:
        low, high = ui.range
        start = min(max(float(value), low), high)
        widget = gui.add_slider(
            label,
            min=math.log10(low),
            max=math.log10(high),
            step=_LOG_STEP,
            initial_value=math.log10(start) if start > 0 else math.log10(low),
        )
        widget.on_update(lambda _, w=widget: on_value(10.0 ** float(w.value)))
        readout = gui.add_text(f"  = {name}", initial_value=_fmt(value, ui), disabled=True)
    elif ui.range is not None:
        low, high = ui.range
        integral = isinstance(value, int)
        step = ui.step if ui.step is not None else (1.0 if integral else (high - low) / 200.0)
        widget = gui.add_slider(
            label,
            min=low,
            max=high,
            step=step,
            initial_value=min(max(float(value), low), high),
        )
        cast = int if integral else float
        widget.on_update(lambda _, w=widget, c=cast: on_value(c(w.value)))
    else:
        widget = gui.add_number(label, initial_value=value)
        cast = int if isinstance(value, int) else float
        widget.on_update(lambda _, w=widget, c=cast: on_value(c(w.value)))
    return widget, readout


class SpecBinder:
    """Widgets for one described dataclass, and the edits they produce.

    Args:
        spec: The instance to bind. Its values seed the controls.
        tiers: Which tiers to show, cheapest first by default.
        skip: Field names to leave out.
    """

    def __init__(self, spec: Any, *, tiers: Iterable[str] = TIERS, skip: Iterable[str] = ()) -> None:
        self.spec = spec
        self.tiers = tuple(tiers)
        self.skip = frozenset(skip)
        self.pending: dict[str, Any] = {}
        """Edits made since the last :meth:`drain`, field name -> new value."""
        self._widgets: dict[str, Any] = {}
        self._readouts: dict[str, Any] = {}
        self._ui: dict[str, ParamUI] = {}
        self._unset: tuple[str, ...] = ()
        self._syncing = False

    # ------------------------------------------------------------------ build
    def groups(self) -> list[tuple[str, list[tuple[Any, ParamUI]]]]:
        """The bindable fields, bucketed by group in declaration order."""
        buckets: dict[str, list[tuple[Any, ParamUI]]] = {}
        for f, ui in iter_params(self.spec):
            if ui.tier not in self.tiers or f.name in self.skip:
                continue
            buckets.setdefault(ui.group or "other", []).append((f, ui))
        return list(buckets.items())

    def build(self, gui: Any) -> None:
        """A control per field, each group headed by its name.

        There are a dozen groups and a spec has dozens of fields, and the group
        names are what make one findable -- so they are kept. Opening the section
        that contains this is already the request for what is in it, so a heading
        is enough; it is the names that index the fields, not the triangles
        beside them.
        """
        unset: list[str] = []
        for group, entries in self.groups():
            gui.add_markdown(f"**{group}**")
            for f, ui in entries:
                value = getattr(self.spec, f.name)
                if value is None:
                    # A None means "let the solver pick", which no slider can
                    # express. Showing it read-only is honest; silently
                    # substituting a number would invent a setting.
                    gui.add_text(f.name, initial_value="unset (solver default)", disabled=True)
                    unset.append(f.name)
                    continue
                self._ui[f.name] = ui
                self._add(gui, f.name, value, ui)
        self._unset = tuple(unset)

    def _add(self, gui: Any, name: str, value: Any, ui: ParamUI) -> None:
        widget, readout = add_param_widget(gui, name, value, ui, lambda v, n=name: self._record(n, v))
        if readout is not None:
            self._readouts[name] = readout
        self._widgets[name] = widget

    def _record(self, name: str, value: Any) -> None:
        # Viser runs on_update synchronously on a ``.value`` assignment, so
        # set_spec moving a widget would otherwise record its own (clamped,
        # log-rounded) value as an edit.
        if not self._syncing:
            self.pending[name] = value

    # ------------------------------------------------------------------ drain
    def drain(self) -> dict[str, Any]:
        """Take the pending edits, main thread. Empty when nothing moved."""
        if not self.pending:
            return {}
        taken, self.pending = self.pending, {}
        self.spec = replace(self.spec, **taken)
        self._refresh_readouts(taken)
        return taken

    def _refresh_readouts(self, changed: dict[str, Any]) -> None:
        for name, value in changed.items():
            readout = self._readouts.get(name)
            if readout is not None:
                readout.value = _fmt(value, self._ui[name])

    # ------------------------------------------------------------------ reset
    def set_spec(self, spec: Any) -> None:
        """Point every control at a new spec, without firing edits.

        For a preset load: the widgets have to follow the values rather than the
        other way round. Any edit still pending is dropped, since it described
        the spec that was just replaced.
        """
        self.spec = spec
        self.pending = {}
        self._syncing = True
        try:
            self._sync_widgets(spec)
        finally:
            self._syncing = False

    def _sync_widgets(self, spec: Any) -> None:
        for name, widget in self._widgets.items():
            value = getattr(spec, name, None)
            if value is None:
                continue
            ui = self._ui[name]
            if ui.log and ui.range is not None and value > 0:
                widget.value = math.log10(min(max(float(value), ui.range[0]), ui.range[1]))
            elif ui.range is not None and not isinstance(value, (bool, str)):
                widget.value = min(max(float(value), ui.range[0]), ui.range[1])
            else:
                widget.value = value
            readout = self._readouts.get(name)
            if readout is not None:
                readout.value = _fmt(value, ui)

    @property
    def unset(self) -> tuple[str, ...]:
        """Fields shown read-only because their value is ``None``."""
        return self._unset

    def describe(self) -> str:
        """A one-line summary of what got bound, for a status readout."""
        counts = dict.fromkeys(TIERS, 0)
        for _, ui in iter_params(self.spec):
            if ui.tier in self.tiers:
                counts[ui.tier] += 1
        total = len(fields(self.spec))
        return f"{sum(counts.values())}/{total} bound: " + ", ".join(f"{n} {t}" for t, n in counts.items() if n)
