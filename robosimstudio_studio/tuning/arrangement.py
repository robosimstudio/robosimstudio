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

from .binder import _LOG_STEP, _fmt, add_param_widget

__all__ = ["ArrangementBinder"]


def _same(value: Any, built: Any, ui: ParamUI | None = None) -> bool:
    """Whether an edit is back at its built value, as near as its control can say.

    A slider starts at the built value, generally off its own grid, and dragged
    back it lands on the grid point nearest it -- within half a step, in the
    slider's own space: log10 for a log field, the field's units for a linear
    one, and the built value clamped to the range in either, since that is
    where the slider started. Booleans, strings and number fields compare
    exactly; nothing rounds what they emit.
    """
    if isinstance(value, bool) or isinstance(built, bool) or not isinstance(value, (int, float)):
        return value == built
    if not isinstance(built, (int, float)):
        return False
    if ui is None or ui.range is None:
        return value == built
    low, high = ui.range
    start = min(max(float(built), low), high)
    if ui.log:
        if value <= 0:
            return False
        start = start if start > 0 else low
        return abs(math.log10(float(value)) - math.log10(start)) <= _LOG_STEP / 2 + 1e-9
    integral = isinstance(built, int)
    step = ui.step if ui.step is not None else (1.0 if integral else (high - low) / 200.0)
    return abs(float(value) - start) <= step / 2 + 1e-9


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
        self._built_values: dict[str, Any] = {}
        """Path -> the value its control was built with, i.e. what the scene runs."""
        self._ui: dict[str, ParamUI] = {}
        """Path -> how its control was built, for telling a slider set back."""

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
        self._built_values[path] = value
        self._ui[path] = ui
        _, readout = add_param_widget(gui, name, value, ui, lambda v, p=path: self.pending.__setitem__(p, v))
        if readout is not None:
            self._readouts[path] = (readout, ui)

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
        for path, edited in taken.items():
            built = self._built_values.get(path)
            back = _same(edited, built, self._ui.get(path))
            # Set back to where its control started: the scene's own value is
            # what a rebuild should carry, not the grid point next to it.
            value = taken[path] = built if back else edited
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
            # Set back to what the scene was built with: nothing to rebuild for.
            if self.applied[path] == REBUILD and back:
                del self.applied[path]
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
