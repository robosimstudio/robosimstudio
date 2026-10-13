# SPDX-License-Identifier: Apache-2.0
"""The host-engine solver contract, written down once.

A Newton solver does its work inside ``step``, which runs inside the captured
CUDA graph. A foreign engine cannot: it is Python and pybind and its own device
work, none of which a graph capture will record. So a host-engine solver inverts
the usual shape -- ``step`` is deliberately nothing, and the real work hangs off
``host_tick``, which the composer schedules into the eager ``after`` phase.

The class exists so the ordinary vocabulary needs no new case: an arrangement
names ``backend("mine", ...)``, an island resolves a solver, a manifest records
one, and ``host_engine=True`` at registration is the whole of what tells the
composer to treat it differently.

**What a subclass writes is one method.** :meth:`SolverHost.open` builds the
engine's bridge from the model and the recipe; everything else here --- the
no-op step contract, the wrench feed, the pipeline attach, the rest capture and
the reset --- is the same for every engine and is inherited.

The lifecycle is the part worth reading before adding an engine, because it is
the part that was wrong for a year. A foreign engine's rest state is **its own**,
not something the entity owning the particles can snapshot on its behalf:

* :meth:`attach_pipeline` runs **before anything settles**. An entity's settle
  steps the pipeline, so by then the engine must already know both states it
  writes, or the double-buffer swap resurrects a stale copy.
* :meth:`capture_rest` runs **after everything has settled**, because the drape
  a deformable lands in is what a reset returns to.
* :meth:`reset` restores it. ``Sim.reset_scene`` resets the solvers first and
  the entities after: the engine returns its island to the captured rest, then
  an entity places this episode's state on top. The other order put a
  deformable an episode moves straight back to the nominal (measured at 56 mm
  between Newton's particles and the engine's nodes).

The composer drives all three off :attr:`PhysicsProgram.hosts`. An entity does
none of it, and that is load bearing rather than tidy --- while it was the soft
*entity's* job, a cloth routed to the same engine silently got none of it, reset
its particle arrays correctly, and had them overwritten by the still-deformed
engine on the very next tick.
"""

from __future__ import annotations

from typing import Any

from .feedback import WrenchFeed

__all__ = ["SolverHost"]


class SolverHost:
    """One host-engine island, as a solver. Subclass and implement :meth:`open`.

    Args:
        model: The finalized Newton model. **Positional** --- this is how the
            composer constructs a host solver, against ``model=`` for a Newton
            one, and the difference is deliberate: a host solver takes the whole
            model, never a compacted view.
        recipe: The manifest's host-island record: ``island``, ``gravity``,
            ``control_hz``, ``actors`` and ``mirrors``, all plain JSON. It is
            the *only* thing besides the model that describes the scene, which
            is what lets a repro bundle rebuild the island from a manifest.
        substeps: Engine steps per control step.
    """

    host_engine = True
    """Class-level echo of the registration flag, for a reader with only the class."""

    def __init__(self, model: Any, *, recipe: dict, substeps: int = 1, **params: Any) -> None:
        self.model = model
        self.recipe = dict(recipe)
        self.substeps = max(1, int(substeps))
        self.device = getattr(model, "device", None)
        self.bridge = self.open(model, self.recipe, substeps=self.substeps, **params)
        if getattr(self.bridge, "feedback_enabled", False):
            self._feed = WrenchFeed(model)
            self.bridge.apply_wrenches = self._feed.assign
            # An attribute rather than a method so the composer's duck-typed
            # check registers the stage only when there are wrenches to add.
            self.rigid_force_stage = self._feed.stage

    # ------------------------------------------------------------ the subclass
    def open(self, model: Any, recipe: dict, *, substeps: int, **params: Any) -> Any:
        """Build this engine's bridge. The one method a subclass owes.

        Must return an object with ``tick(state, dt)``, ``capture_rest()``,
        ``restore_rest()``, ``attach_pipeline(pipeline)``, a ``feedback_enabled``
        flag and, when that is true, an ``apply_wrenches`` slot this class fills.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement open()")

    # ------------------------------------------------------- solver contract
    def step(self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float) -> None:
        """Nothing, on purpose.

        The bridge already wrote this island's particles into *both* states from
        the ``after`` phase, so the double-buffer swap has no stale copy to
        resurrect and there is no device work to record into the graph.
        """
        del state_in, state_out, control, contacts, dt

    def notify_model_changed(self, flags: Any) -> None:
        """Model edits land on the next compile for a host backend (REBUILD tier)."""
        del flags

    def update_contacts(self, contacts: Any, state: Any = None) -> None:
        del contacts, state

    def reset(self, state: Any, world_mask: Any = None, flags: Any = None) -> None:
        """Return the engine to the rest it captured."""
        del state, world_mask, flags
        self.bridge.restore_rest()

    # -------------------------------------------------- host engine lifecycle
    def attach_pipeline(self, pipeline: Any) -> None:
        """Before anything settles: the engine learns which states it writes."""
        self.bridge.attach_pipeline(pipeline)

    def capture_rest(self) -> None:
        """After everything has settled: the engine records what a reset returns to."""
        self.bridge.capture_rest()

    # ---------------------------------------------------------- host surface
    def host_tick(self, state: Any, dt: float) -> None:
        """The per-control-step co-simulation tick; scheduled by the composer."""
        self.bridge.tick(state, dt)
