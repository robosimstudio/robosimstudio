# SPDX-License-Identifier: Apache-2.0
"""What a physics *arrangement* is, as data: backends by domain, a coupler, a collision policy.

A document stages physics by a versioned **name** -- ``(:staging (:physics
cable_coupled_960hz@v1))`` -- and never by a number. What the name stands for is
an :class:`ArrangementSpec`, and this module is where that is defined. It is
deliberately **data over classes**: a backend is a solver named by alias or
dotted path plus the keyword arguments its constructor takes, verbatim; a
coupler is a kind plus its parameters; a domain is a string with the kinds of
state it carries. Every extension point is a registry keyed by a string, so a
solver Newton ships tomorrow is usable from a task directory with no edit under
``robosimstudio/``::

    register_arrangement("cloth_xpbd@v1", ArrangementSpec(
        control_hz=60.0, substeps=10,
        backends=(("articulated", backend("mujoco", njmax=1024)),
                  ("cloth", backend("newton.solvers.SolverXPBD", iterations=8))),
        coupling=coupler("proxy", mode="staggered", iterations=2)))

**Schemas are an overlay, not a gate.** A :class:`ParamSchema` registered beside
an alias tells the studio what each parameter is -- its unit, its range, and the
:mod:`~robosimstudio.specs.params` tier a change costs -- and how to apply it to
a running solver. An alias with no schema still composes and still runs; its
parameters render as plain rebuild-tier inputs.

**Who declares what.** An asset declares what it *is* and what it *touches*
(:class:`SimRequirements`: a domain and contact rules by role); an
arrangement declares which backend serves each domain and how the islands
couple; the composer in :mod:`robosimstudio.scene.physics` derives everything
else and records what it built in a :class:`PhysicsManifest`. Nothing here
imports a solver: the registries hold dotted paths, and resolving one is the
composer's job.

**This module is a facade.** The definitions live in four modules, split by
subsystem, and every public name is re-exported here so that
``from robosimstudio.specs.physics import X`` keeps meaning what it meant:

* :mod:`~robosimstudio.specs.registries` -- the registries and their record
  types: domains and state layouts, engine bundles, backends, couplers,
  parameter schemas, what an engine consumes; and the three coupler
  declarations the library ships.
* :mod:`~robosimstudio.specs.declare` -- what an asset declares it needs:
  :class:`SimRequirements`, :class:`ContactRule`, :class:`HostActor`.
* :mod:`~robosimstudio.specs.arrangement` -- the data an arrangement is made
  of, :class:`ArrangementSpec` itself, :data:`GENERIC` and :data:`MUJOCO_500HZ`.
* :mod:`~robosimstudio.specs.manifest` -- :class:`PhysicsManifest` and its
  records.

The Newton solver registrations that used to close this file are a bundle now,
``robosimstudio.engines.newton``, found by the same scan that finds the foreign
engines: nothing is registered by importing this module.
"""

from __future__ import annotations

from .arrangement import (
    GENERIC,
    MUJOCO_500HZ,
    ArrangementSpec,
    BackendSpec,
    CollisionSpec,
    CouplerSpec,
    backend,
    coupler,
)
from .declare import COUPLINGS, ContactRule, HostActor, SimRequirements, contacts_with, sim_to_dict
from .manifest import ChannelRecord, IslandRecord, PhysicsManifest
from .registries import (  # noqa: F401 -- the names outside __all__ are public and imported by name across the tree
    ACTOR_CONSUMES,
    ADMM_SCHEMA,
    BACKENDS,
    COUPLERS,
    DOMAINS,
    ENGINES,
    PROXY_SCHEMA,
    ROLES,
    STATE_KINDS,
    SUBSTRATES,
    BackendInfo,
    CouplerInfo,
    DomainInfo,
    ParamRule,
    ParamSchema,
    actor_consumes,
    declare_consumes,
    engine_consumes,
    get_backend,
    get_coupler,
    get_domain,
    layout_chain,
    list_domains,
    reads_as,
    register_backend,
    register_coupler,
    register_domain,
    register_substrate,
    rule,
)

__all__ = [
    "ACTOR_CONSUMES",
    "BACKENDS",
    "COUPLERS",
    "COUPLINGS",
    "DOMAINS",
    "GENERIC",
    "MUJOCO_500HZ",
    "ROLES",
    "STATE_KINDS",
    "ArrangementSpec",
    "BackendInfo",
    "BackendSpec",
    "ChannelRecord",
    "CollisionSpec",
    "ContactRule",
    "CouplerInfo",
    "CouplerSpec",
    "DomainInfo",
    "HostActor",
    "IslandRecord",
    "ParamRule",
    "ParamSchema",
    "PhysicsManifest",
    "SimRequirements",
    "actor_consumes",
    "backend",
    "contacts_with",
    "coupler",
    "declare_consumes",
    "engine_consumes",
    "get_backend",
    "get_coupler",
    "get_domain",
    "list_domains",
    "register_backend",
    "register_coupler",
    "register_domain",
    "rule",
    "sim_to_dict",
]
