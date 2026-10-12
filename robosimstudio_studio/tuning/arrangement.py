# SPDX-License-Identifier: Apache-2.0
"""The arrangement as a panel: rates, every backend's parameters, the coupler's.

An :class:`~robosimstudio.specs.physics.ArrangementSpec` is not one flat
dataclass any more -- it is rates plus a backend per domain plus a coupler, and
a backend's parameters are a mapping the solver's constructor takes verbatim.
What makes them bindable is the *schema* registered beside the alias: per
parameter, a :class:`~robosimstudio.specs.params.ParamUI` and how a change
lands on the running program. This binder walks the arrangement, builds a
:class:`~robosimstudio_studio.tuning.binder.SpecBinder`-style control for every
described parameter, and on drain applies the recapture-tier ones through
:meth:`PhysicsProgram.apply` and accumulates the rest into a rebuilt spec.

A backend with no schema -- a dotted path nobody described -- still shows: its
parameters render as plain rebuild-tier inputs, which is the extensibility
contract's promise that an undescribed solver is usable, just less informed.
"""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any

from robosimstudio.specs.params import REBUILD, RECAPTURE, ParamUI, iter_params
from robosimstudio.specs.physics import ArrangementSpec, ParamRule, get_backend, get_coupler

from .binder import TIER_MARK, _fmt

__all__ = ["ArrangementBinder"]

_LOG_STEP = 0.01


class ArrangementBinder:
    """Controls for an arrangement: rates, backends, coupler, collision.

    Args:
        spec: The arrangement the scene was compiled with.
        program: The running :class:`~robosimstudio.scene.physics.PhysicsProgram`,
            for applying recapture-tier edits live. ``None`` accumulates only.
    """

    def __init__(self, spec: ArrangementSpec, program: Any = None) -> None:
        self.built = spec
        self.spec = spec
        self.program = program
        self.pending: dict[str, Any] = {}
        """Edits since the last drain: ``path -> value`` with paths
        ``rates:<field>``, ``collision:<field>``, ``backend:<domain>:<param>``,
        ``coupling:<param>``."""
        self._readouts: dict[str, tuple[Any, ParamUI]] = {}
        self.applied: dict[str, str] = {}
        """Path -> tier, for every edit applied live or left pending rebuild."""

    # ------------------------------------------------------------------ build
    def build(self, gui: Any) -> None:
        gui.add_markdown("**rates**")
        for f, ui in iter_params(self.spec):
            if ui.tier != REBUILD or f.name == "use_cuda_graph":
                continue
            self._add(gui, f"rates:{f.name}", f.name, getattr(self.spec, f.name), ui)
        for domain, backend in self.spec.backends:
            info = get_backend(backend.solver)
            gui.add_markdown(f"**{domain}: {backend.solver}**")
            effective = info.effective(backend, has_particles=True)
            schema = dict(info.schema or {})
            for name, value in effective.params:
                rule = schema.get(name, ParamRule(ui=ParamUI(tier=REBUILD)))
                if isinstance(value, (tuple, list, dict)) or value is None:
                    continue
                self._add(gui, f"backend:{domain}:{name}", name, value, rule.ui)
        try:
            coupler_info = get_coupler(self.spec.coupling.method)
            schema = dict(coupler_info.schema or {})
        except KeyError:
            schema = {}
        if self.spec.coupling.params:
            gui.add_markdown(f"**coupling: {self.spec.coupling.method}**")
            for name, value in self.spec.coupling.params:
                rule = schema.get(name, ParamRule(ui=ParamUI(tier=REBUILD)))
                if isinstance(value, (tuple, list, dict)) or value is None:
                    continue
                self._add(gui, f"coupling:{name}", name, value, rule.ui)
        gui.add_markdown("**collision**")
        for f, ui in iter_params(self.spec.collision):
            value = getattr(self.spec.collision, f.name)
            if value is None:
                gui.add_text(f.name, initial_value="unset (solver default)", disabled=True)
                continue
            self._add(gui, f"collision:{f.name}", f.name, value, ui)

    def _add(self, gui: Any, path: str, name: str, value: Any, ui: ParamUI) -> None:
        label = f"{name}{TIER_MARK.get(ui.tier, '')}"
        if isinstance(value, bool):
            widget = gui.add_checkbox(label, initial_value=value)
            widget.on_update(lambda _, p=path, w=widget: self.pending.__setitem__(p, bool(w.value)))
        elif isinstance(value, str):
            if ui.choices:
                widget = gui.add_dropdown(
                    label, options=tuple(ui.choices), initial_value=value if value in ui.choices else ui.choices[0]
                )
            else:
                widget = gui.add_text(label, initial_value=value)
            widget.on_update(lambda _, p=path, w=widget: self.pending.__setitem__(p, str(w.value)))
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
            widget.on_update(lambda _, p=path, w=widget: self.pending.__setitem__(p, 10.0 ** float(w.value)))
            self._readouts[path] = (gui.add_text(f"  = {name}", initial_value=_fmt(value, ui), disabled=True), ui)
        elif ui.range is not None:
            low, high = ui.range
            integral = isinstance(value, int)
            step = ui.step if ui.step is not None else (1.0 if integral else (high - low) / 200.0)
            widget = gui.add_slider(
                label, min=low, max=high, step=step, initial_value=min(max(float(value), low), high)
            )
            cast = int if integral else float
            widget.on_update(lambda _, p=path, w=widget, c=cast: self.pending.__setitem__(p, c(w.value)))
        else:
            widget = gui.add_number(label, initial_value=value)
            cast = int if isinstance(value, int) else float
            widget.on_update(lambda _, p=path, w=widget, c=cast: self.pending.__setitem__(p, c(w.value)))

    # ------------------------------------------------------------------ drain
    def drain(self) -> dict[str, Any]:
        """Take the pending edits on the main thread, applying the live ones.

        Returns the edits taken. Every one is folded into :attr:`spec`, so a
        rebuild carries them; the recapture-tier ones are also applied to the
        running program through :meth:`PhysicsProgram.apply`, which reports the
        tier it actually cost.
        """
        if not self.pending:
            return {}
        taken, self.pending = self.pending, {}
        for path, value in taken.items():
            self._fold(path, value)
            readout = self._readouts.get(path)
            if readout is not None:
                readout[0].value = _fmt(value, readout[1])
            if self.program is not None and path.split(":")[0] in ("backend", "coupling"):
                try:
                    self.applied[path] = self.program.apply(path, value)
                except Exception as exc:  # a private-API applier on a newer newton; the rebuild still carries it
                    print(f"[studio] could not apply {path} live ({type(exc).__name__}: {exc}); pending rebuild")
                    self.applied[path] = REBUILD
                if self.applied[path] == RECAPTURE and getattr(self.program, "_pipeline", None) is not None:
                    self.program._pipeline.invalidate_graph()
            else:
                self.applied[path] = REBUILD
        return taken

    def _fold(self, path: str, value: Any) -> None:
        kind, _, rest = path.partition(":")
        if kind == "rates":
            self.spec = self.spec.with_(**{rest: value})
        elif kind == "collision":
            from dataclasses import replace

            self.spec = self.spec.with_(collision=replace(self.spec.collision, **{rest: value}))
        elif kind == "coupling":
            self.spec = self.spec.with_(coupling=self.spec.coupling.with_(**{rest: value}))
        elif kind == "backend":
            domain, _, param = rest.partition(":")
            current = self.spec.backend_for(domain)
            if current is not None:
                self.spec = self.spec.with_backend(domain, current.with_(**{param: value}))

    # ---------------------------------------------------------------- status
    def pending_rebuild(self) -> tuple[str, ...]:
        """Paths whose edits the built scene cannot honour without a rebuild."""
        return tuple(sorted(path for path, tier in self.applied.items() if tier == REBUILD))

    def describe(self) -> str:
        backends = ", ".join(f"{d}={b.solver}" for d, b in self.spec.backends)
        return f"{backends}; coupling {self.spec.coupling.method}; {len(fields(self.spec.collision))} collision fields"
