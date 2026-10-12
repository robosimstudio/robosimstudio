# SPDX-License-Identifier: Apache-2.0
"""What drives a fleet, and how a job names one.

The batched twin of :mod:`robosimstudio_bench.solutions.base`, with the same
division of authority: a policy may *request* that a world stop and may not
declare it successful. That split is why this is a base class rather than a
callable, and it is what keeps a batched success rate comparable with a serial
one.

Two things differ from the single-world contract, and both are consequences of
there being many worlds behind one step:

* :meth:`FleetSolution.act` returns nothing. A single-world policy hands back an
  action and the environment applies it; at five hundred worlds that array would
  be assembled on the host every step to say what device state already says. So a
  fleet policy writes its intent into the batched primitives -- the IK rig, the
  segment drivers, the gripper banks -- and the fleet's captured control stage
  turns that into targets.
* Everything is indexed by world. A policy is handed the wave's
  :class:`~robosimstudio_bench.run.fleet.WorldSlot` list once at reset and is
  expected to keep its own per-world state keyed the same way.

A world the policy is not driving needs no action at all: nothing writes its
targets, so it holds its last pose exactly. "Idle" is therefore the absence of
work rather than a command, which is why :class:`FleetSolutionIdle` has an empty
:meth:`act`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio_bench.taskbuild.artifacts import resolve_dotted

from .base import SolutionCommon

__all__ = [
    "FLEET_EQUIVALENTS",
    "FleetSolution",
    "FleetSolutionIdle",
    "load_fleet_solution",
]


FLEET_EQUIVALENTS = {
    # A job names a policy the same way whichever runner picks it up, so the
    # ported ones are translated here rather than at every call site. A name
    # missing from this table and not already a FleetSolution is a policy nobody
    # has batched yet, and saying so is more useful than running something else.
    "robosimstudio_bench.solutions.base:SolutionIdle": "robosimstudio_bench.solutions.fleet:FleetSolutionIdle",
    "robosimstudio_bench.solutions.base.SolutionIdle": "robosimstudio_bench.solutions.fleet:FleetSolutionIdle",
    "robosimstudio_policy_agent.graph_solution:SolutionGraph": "robosimstudio_policy_agent.fleet.solution:SolutionGraphBatched",
    "robosimstudio_policy_agent.graph_solution.SolutionGraph": "robosimstudio_policy_agent.fleet.solution:SolutionGraphBatched",
    # A module may add its own pair when it is imported -- the scripted routines
    # do -- so load_fleet_solution reads this table again after the import.
}


class FleetSolution(SolutionCommon):
    """Base class for anything that drives a fleet.

    The batched twin of :class:`~robosimstudio_bench.solutions.base.SolutionBase`, and
    the two share :class:`~robosimstudio_bench.solutions.base.SolutionCommon`: holding
    what they drive, releasing it, and reporting a name. What they do not share
    is how they drive -- ``get_action`` returns one action, ``act`` writes the
    live worlds' intent in place -- because that is the difference between the
    two, not an accident of having been written twice.

    ``self.fleet`` keeps its name rather than becoming ``self.fleet``: ``fleet.``
    is driver scope and ``env.`` is world scope, and ``test_fleet.py`` holds the
    line by requiring every ``env.`` read to be one a ``WorldView`` can answer.

    Args:
        fleet: The batched environment. Held so a policy can read handles, plan
            through the IK rig, or drive the segment drivers; it must not step
            the fleet itself.
    """

    def __init__(self, fleet, **kwargs: Any) -> None:
        self.fleet = fleet
        self.kwargs = kwargs
        self.slots: list = []

    def reset_wave(self, slots: list) -> None:
        """Called after each wave reset, before the first step.

        Args:
            slots: One per world, in world order. A slot whose ``episode`` is
                ``None`` is parked and must not be driven.
        """
        self.slots = slots

    def act(self, live: np.ndarray) -> None:
        """Write this step's intent for the live worlds.

        Args:
            live: Per world, whether it is still being worked on. A world that is
                not live has already been scored; writing its targets would move
                a scene whose episode is over.
        """

    def on_step(self, snapshot) -> None:
        """Called after each step. ``snapshot`` is ``None`` off cadence ticks.

        The only place a policy sees state, and deliberately so: a policy that
        read the device itself would put a synchronize back in the hot loop that
        the fleet exists to remove.
        """

    def termination_requests(self) -> dict[int, str]:
        """Worlds this policy is finished with, by index, and why.

        Read at cadence ticks. A request ends the world's episode; it does not
        make it a success. The evaluator decides that, on the same step.
        """
        return {}

    def provenance(self) -> dict[str, Any]:
        """What this policy wants recorded about how the run was produced.

        Merged into the run's meta under ``policy_provenance``. The seam exists
        because some of what decides a batched number belongs to the *driver*
        rather than to the environment -- how long the fleet spent unable to
        step, how many calls were answered as a group, which worlds were given
        up on -- and a number whose conditions are only in somebody's memory is
        the thing this repository's provenance keys exist to prevent.
        """
        return {}

    def row_extras(self, world: int) -> dict[str, Any]:
        """Extra keys for one world's results row, read as it finishes.

        The seam through which per-episode telemetry lands in
        ``episode_results.jsonl`` without the run being recorded. Keys land at the
        row's top level; keep them JSON-able and distinctively named.
        """
        del world
        return {}

    def status(self, world: int) -> str:
        """One short line about what this world is doing, for a live view."""
        del world
        return ""


class FleetSolutionIdle(FleetSolution):
    """Drives nothing. Every world holds where the wave reset put it.

    The baseline every task should fail, and the fleet's own smoke test: a wave
    of idle worlds exercises the reset, the step, the snapshot and the scoring
    with nothing in between them to explain a difference.
    """

    def act(self, live: np.ndarray) -> None:
        del live


def load_fleet_solution(path: str, fleet, **kwargs: Any):
    """Instantiate a fleet policy from a dotted path.

    Args:
        path: ``"package.module:ClassName"`` or ``"package.module.ClassName"``.
            Empty gives :class:`FleetSolutionIdle`.
        fleet: The fleet to bind it to.
        **kwargs: Constructor arguments.

    Raises:
        TypeError: If the named class is a single-world policy. Loading one here
            would fail much later, in the middle of a wave, with an error about
            an unexpected argument -- and the two contracts differ in what
            ``act`` returns, which no signature check would catch.
    """
    if not path:
        return FleetSolutionIdle(fleet, **kwargs)
    path = FLEET_EQUIVALENTS.get(path, path)
    cls = resolve_dotted(path)
    if not issubclass(cls, FleetSolution) and path in FLEET_EQUIVALENTS:
        # Importing the policy's module registered its batched twin.
        path = FLEET_EQUIVALENTS[path]
        cls = resolve_dotted(path)
    if not issubclass(cls, FleetSolution):
        raise TypeError(
            f"{path!r} is not a fleet policy: batched runs need a FleetSolution subclass, and "
            f"{cls.__name__} drives one world at a time. Run it without --num-worlds, or wrap it."
        )
    return cls(fleet, **kwargs)
