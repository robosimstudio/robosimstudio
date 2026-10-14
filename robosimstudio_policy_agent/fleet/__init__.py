# SPDX-License-Identifier: Apache-2.0
"""Running many graph instances against one batched simulation.

The single-world integration gives one graph one environment and lets the two
take turns: the runtime's worker threads post requests, the rollout services them
between steps, and one control step is exchanged per round trip. That is
:mod:`robosimstudio.core.bus`, and it stays exactly as it is -- it is the
path a real robot and the bridge both use, and nothing here replaces it.

What this package adds is a second implementation of the *same surfaces*, sized
for a fleet. The constraint that shapes all of it: the graph, its scripts and its
tool interface are the portable artifact, so a batched run must not require them
to change. Worlds therefore stop at the tool boundary and go no further -- a
world index is bound when a world's connector is built, and everything above it
sees one robot in one scene.

Two substitutions do the work:

* :mod:`.intents` -- a node commands a *motion* rather than a step. The fleet
  runs it on device to completion and wakes the node's thread when it is done, so
  the host does no work per world per step.
* :mod:`.snapshots` and :mod:`.world_bus` -- reads come from one batched
  snapshot sliced per world, in the very type the single-world bus publishes, so
  every shipped suite tool runs against a fleet unmodified.
"""

from __future__ import annotations

from .barrier import Ask, ParkTable, Rendezvous, StallTimeout, StallWatch, park_if
from .intents import (
    GripperMove,
    Intent,
    IntentHandle,
    IntentTable,
    JointTrajectory,
    MoveToJoints,
    ServoToPose,
    Superseded,
)
from .obs_format import format_runtime_obs, world_observation, world_observations
from .snapshots import SnapshotTable
from .solution import SolutionGraphBatched
from .stepper import StepService
from .world_bus import ServiceQueue, StepWaiter, WorldBus
from .world_facade import WorldFacade

__all__ = [
    "Ask",
    "GripperMove",
    "Intent",
    "IntentHandle",
    "IntentTable",
    "JointTrajectory",
    "MoveToJoints",
    "ParkTable",
    "Rendezvous",
    "ServiceQueue",
    "ServoToPose",
    "SnapshotTable",
    "SolutionGraphBatched",
    "StallTimeout",
    "StallWatch",
    "StepService",
    "StepWaiter",
    "Superseded",
    "WorldBus",
    "WorldFacade",
    "format_runtime_obs",
    "park_if",
    "world_observation",
    "world_observations",
]
