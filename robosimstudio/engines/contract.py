# SPDX-License-Identifier: Apache-2.0
"""What a host engine's bridge must be, written down.

:class:`~robosimstudio.engines.host.SolverHost` satisfies the composer's solver
contract for every engine; what differs is the object its ``open`` returns. This
is that shape as a :class:`typing.Protocol`.

**Structural, and deliberately so.** Nothing inherits from this and nothing is
type-checked against it at runtime by the composer -- the seam stays duck-typed,
exactly as ``scene/physics`` finds a host solver by ``getattr``. What the
Protocol buys is that the contract is now *checkable*: ``tests/test_cosim_contract.py``
asserts each shipped bridge satisfies it without an engine installed, so a method
renamed on one engine and not the other fails in the fast tier rather than on a
GPU box with a wheel.

**There is deliberately no capability record here**, and the omission is
argued rather than overlooked. The branches it would replace do not exist:
nothing outside an engine's own bundle asks what that engine can do,
``pin_nodes`` and ``contact_points`` are each reached on the SuperDex bridge by
the one suite that uses it, and no call site anywhere tests an alias.
``scene/entity.py`` makes this argument about ``observe`` and it holds here: a
contract no call site matches is worse than no contract, because it invites an
implementation that will silently never be read. Add it with the first branch
that needs it.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["CosimBridge"]


@runtime_checkable
class CosimBridge(Protocol):
    """One host island's engine, as the rest of this package needs it.

    Everything engine-neutral is elsewhere --- :class:`~robosimstudio.engines.cosim.CosimRig`
    owns the tick's state machine, :class:`~robosimstudio.engines.feedback.WrenchAccumulator`
    the shaping, ``geometry`` the reading. A bridge is what is left: the engine's
    own API, plus the five entry points below.
    """

    feedback_enabled: bool
    """Whether this island harvests contact wrenches back onto Newton bodies.

    False when the arrangement turned feedback off, when no mirror is both
    two-way and body-carried, or when the model has no bodies at all. The owning
    :class:`~robosimstudio.engines.host.SolverHost` reads it once, at
    construction, to decide whether to register a ``rigid_force`` stage --- so a
    bridge that sets it later sets it too late.
    """

    apply_wrenches: Any
    """``(wrench[n_bodies, 6] float32) -> None``, filled in by the solver.

    ``None`` until then. A bridge writes through it and never creates it: the
    device-side array belongs to the solver, which is the half that knows about
    warp.
    """

    def tick(self, state: Any, dt: float) -> None:
        """Advance one control period and write the results back.

        Scheduled into the eager ``after`` phase, outside the captured graph.
        Reads this step's rigid poses, drives the mirrors across the engine's
        substeps, harvests, and writes particle positions and velocities into
        **both** states of the double buffer.
        """

    def attach_pipeline(self, pipeline: Any) -> None:
        """Learn which states to write. Called before anything settles."""

    def capture_rest(self) -> None:
        """Record the settled state as what a reset returns to.

        Called after every entity's ready phase, because the drape a deformable
        lands in is the start state --- not the pose it was built at.
        """

    def restore_rest(self) -> None:
        """Return the engine to the captured rest.

        Called from the solver's ``reset``, which ``Sim.reset_scene`` runs
        *before* the entities' own resets: the engine is returned to its rest
        first, so an entity that places this episode's state (a deformable drawn
        somewhere new) is not restored straight back to the nominal. Must also
        arrange for the next tick to teleport its mirrors rather than sweep to
        them, and zero any harvested wrench.
        """
