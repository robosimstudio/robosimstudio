# SPDX-License-Identifier: Apache-2.0
"""One batched read, sliced into per-world views scripts already understand.

Every ``sim.get_object_pose``, every ``robot.get_ee_pose``, every
``check_success`` a graph performs is a read of state as of some recent step.
The single-world bus answers those from a snapshot the owner publishes each step
-- deliberately, since a pose lookup that waited for a step boundary would make a
node that reads three poses three steps slow.

That design carries over unchanged, with one substitution: the fleet publishes
*one* snapshot covering every world, and each world's view is a slice of it. The
view is a :class:`~robosimstudio.core.bus.SimSnapshot` -- the same type
the single-world path publishes -- so everything above it, including every
shipped suite tool, reads it without knowing worlds exist.

Two properties are worth stating because things depend on them:

* **Views are built once per publish and cached.** A wave of five hundred worlds
  whose nodes each read a pose would otherwise re-slice per call.
* **A view is replaced, never mutated.** A script thread reading concurrently
  with a publish sees either the old snapshot or the new one, never a mixture --
  the same guarantee the single-world bus gives, and for the same reason.

Staleness is bounded by the fleet's snapshot cadence: a view is at most
``eval_every`` steps behind. That is a real difference from the single-world
path, where a snapshot is published every step, and it is the reason the servo
does not read from here -- it measures on device, every step, inside the fleet's
control stage.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from robosimstudio.core.bus import SimSnapshot

__all__ = ["SnapshotTable"]


class SnapshotTable:
    """The fleet's latest state, presented per world.

    Args:
        num_worlds: Worlds in the fleet.
        handles: The scene's handles, for naming objects in the per-world
            observation. Template handles, which index a world's view exactly as
            they index a single-world model's arrays.
    """

    def __init__(self, num_worlds: int, *, handles: Any = None) -> None:
        self.num_worlds = int(num_worlds)
        self.handles = handles
        self._lock = threading.Lock()
        self._views: list[SimSnapshot] = [SimSnapshot() for _ in range(self.num_worlds)]
        self._infos: list[dict[str, Any]] = [{} for _ in range(self.num_worlds)]
        self._step = 0

    # ------------------------------------------------------------- fleet side
    def publish(
        self,
        snapshot,
        *,
        infos: list[dict[str, Any]] | None = None,
        obs: list[dict] | None = None,
        frames: list[dict] | None = None,
    ) -> None:
        """Slice one fleet snapshot into per-world views. Fleet thread.

        Args:
            snapshot: The fleet's snapshot, with per-world views over one read.
            infos: Per world, the evaluator's info for this step -- what
                ``sim.check_success`` answers from.
            obs: Per world, the observation dict a graph's ``get_observation``
                should see. Built by the caller because what belongs in one is a
                property of the connector, not of the state.
            frames: Per world, that world's camera frames, or ``None`` for none.

                Published rather than rendered on demand, and empty is the
                ordinary case: a privileged graph never asks for a picture, and
                a rig that rendered every step whether or not anybody looked
                would be the most expensive thing in the wave. A node that wants
                a *fresh* frame asks the fleet's rendezvous for one, which is
                where every world that wants one is answered together.
        """
        views = []
        for world in range(self.num_worlds):
            info = (infos[world] if infos else None) or {}
            views.append(
                SimSnapshot(
                    step=int(snapshot.step),
                    sim_time=float(snapshot.sim_time),
                    obs=(obs[world] if obs else None) or {},
                    body_q=snapshot.body(world),
                    info=info,
                    frames=(frames[world] if frames and world < len(frames) else None) or {},
                )
            )
        with self._lock:
            self._views = views
            self._infos = [view.info for view in views]
            self._step = int(snapshot.step)

    # ------------------------------------------------------------ worker side
    def view(self, world: int) -> SimSnapshot:
        """One world's snapshot. Safe to read from any thread."""
        with self._lock:
            return self._views[int(world)]

    @property
    def step(self) -> int:
        with self._lock:
            return self._step

    def steps(self) -> np.ndarray:
        """Every world's step count -- the same number today, since one clock
        drives the fleet. Kept per world so nothing above assumes that."""
        return np.full(self.num_worlds, self.step, dtype=np.int64)
