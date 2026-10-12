# SPDX-License-Identifier: Apache-2.0
"""Composing solvers and assets at the end of a compile: declare, compose, schedule.

The one place solvers and assets meet. Every entity has been built and bound;
every asset has said what it *is* (a physics domain) and what it *touches*
(contact rules by role); the task has staged an
:class:`~robosimstudio.specs.physics.ArrangementSpec` naming a backend per
domain and a coupler. :func:`compose` turns the three into a
:class:`PhysicsProgram`: solver *islands* (one Newton solver each, over a
compacted view of the shared model), coupling *channels* between them, the
collision pipelines with exactly the shape pairs each needs, a
:class:`~robosimstudio.core.schedule.Schedule` for the substep loop, and a
:class:`~robosimstudio.specs.physics.PhysicsManifest` recording all of it.

* :mod:`.interactions` -- the interaction graph: nodes are built entities with
  their owned state and roles; edges are the declared or default contact rules.
* :mod:`.islands` -- partition entities into solver islands by domain and
  backend; equal backends share one island.
* :mod:`.routing` -- which shape pairs go to which pipeline, so a pair is
  detected once and every declared contact is routed somewhere.
* :mod:`.backends` -- resolve a backend alias or dotted path to a solver class,
  register its custom attributes before the build, construct it over a view.
* :mod:`.couplers` -- build the coupled solver for the channels: Newton's proxy
  coupler, its ADMM coupler, or a kind a task registered.
* :mod:`.kinematic` -- a solver for bodies that follow their targets exactly.
* :mod:`.world` -- index expansion for replicated models.
* :mod:`.compose` -- the orchestration, and the program it returns.

Nothing below this package names a solver; nothing above it needs to.
"""

from __future__ import annotations

from .backends import build_solver, register_custom_attributes, resolve_backend_class
from .compose import Channel, CompositionError, Island, PhysicsProgram, compose
from .couplers import COUPLER_BUILDERS, register_coupler_builder
from .interactions import Edge, InteractionGraph, Node, build_graph
from .routing import filtered_pairs
from .world import expand_indices

__all__ = [
    "COUPLER_BUILDERS",
    "Channel",
    "CompositionError",
    "Edge",
    "InteractionGraph",
    "Island",
    "Node",
    "PhysicsProgram",
    "build_graph",
    "build_solver",
    "compose",
    "expand_indices",
    "filtered_pairs",
    "register_coupler_builder",
    "register_custom_attributes",
    "resolve_backend_class",
]
