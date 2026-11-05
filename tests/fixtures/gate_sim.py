# SPDX-License-Identifier: Apache-2.0
"""A sim as :func:`robosimstudio_real2sim_agent.checks.gate` reads it, and the server it hangs off.

The gate asks a sim for its cameras, its names, a position by name, whether its
state is finite, and to run a few steps left alone; it reads particles and
bodies off ``state`` and ``model`` where a test gives them. :class:`GateSim`
answers each from what the test hands it, and its ``run`` does whatever the
test says a few steps do -- a rope node sinking, a lid swinging -- so the gate's
lines can be held to a drift the test wrote down. Five tests each had grown a
``Sim`` class of their own for this, differing only in those answers.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np

__all__ = ["GateSim", "gate_server"]


class GateSim:
    """Cameras, names, positions by name, a finite flag, and a ``run`` that does what the test says.

    ``positions`` maps a name to where it is, or to a function returning where
    it is now; a name not in it raises ``KeyError``, as a thing with no
    position does. ``finite`` is a flag or a function of nothing. ``on_run``
    is called with the step count each time the gate runs the scene.
    ``scene``, ``state``, ``model``, ``contacts`` and ``handles`` are handed
    through as they are, for the gate's particle, body and contact reads.
    """

    def __init__(
        self,
        *,
        cameras: tuple = (),
        names: tuple[str, ...] | list[str] = (),
        positions: dict[str, Any] | None = None,
        finite: bool | Callable[[], bool] = True,
        on_run: Callable[[int], None] | None = None,
        scene: Any = None,
        state: Any = None,
        model: Any = None,
        contacts: Any = None,
        handles: Any = None,
    ) -> None:
        self.cameras = tuple(cameras)
        self._names = list(names)
        self.positions = dict(positions or {})
        self.finite = finite
        self.on_run = on_run
        self.scene = scene
        self.state = state
        self.model = model
        if contacts is not None:
            self.contacts = contacts
        if handles is not None:
            self.handles = handles
        self.runs: list[int] = []

    def state_is_finite(self) -> bool:
        return bool(self.finite() if callable(self.finite) else self.finite)

    def names(self) -> list[str]:
        return list(self._names)

    def run(self, steps: int) -> GateSim:
        self.runs.append(int(steps))
        if self.on_run is not None:
            self.on_run(int(steps))
        return self

    def sync(self) -> GateSim:
        return self

    def __getitem__(self, name: str) -> SimpleNamespace:
        if name not in self.positions:
            raise KeyError(name)
        where = self.positions[name]
        return SimpleNamespace(position=np.asarray(where() if callable(where) else where, dtype=np.float64))


def gate_server(sim: Any, *, build_error: str = "", last_error: str = "") -> SimpleNamespace:
    """What the gate reads off the live server: the sim, and the last build's and last anything's error."""
    return SimpleNamespace(sim=sim, build_error=build_error, last_error=last_error)
