# SPDX-License-Identifier: Apache-2.0
"""Frozen dataclass specs: the whole configuration surface, as Python.

Python-first, no YAML. A spec is ``@dataclass(frozen=True, kw_only=True)``,
holds tuples rather than lists, and is therefore hashable -- which is what lets
a scene compiler cache on one and an episode sampler key on one. Defaults are
``None`` or a tuple, never a mutable object.

Three axes are registered independently and composed at compile time: a
:class:`CellSpec` is the place, an :class:`EmbodimentSpec` is the robot, and a
:class:`TaskSpec` is the objective and the things it acts on. A :class:`Binding`
is one admissible pairing of the three, and the work-surface frame
(:attr:`CellSpec.surface_frame`) is the coordinate system they agree in.

**These specs are the IR, not the authoring surface.** Nothing writes a
``TaskSpec`` by hand: a task is a document, the task language lowers it to one
of these, and the compiler, the ``Env``, the studio and the runner consume it
exactly as they always did.

**The split is by what a spec describes, and it is a DAG.** Every edge below is
a real field annotation rather than a filing decision, which is why the modules
import in one direction only::

    base ── asset ─┬─ robot ───┐
                   ├─ cell ────┼─ task          run (depends on neither)
    params ─ registries ─ arrangement ─┬─ declare ─┘
                                       └─ manifest

* :mod:`~robosimstudio.specs.base` -- the tuple aliases everything is written in
* :mod:`~robosimstudio.specs.asset` -- what a thing is: geometry, material, region
* :mod:`~robosimstudio.specs.robot` -- what a robot is: arms, hands, actuation
* :mod:`~robosimstudio.specs.cell` -- what a place is
* :mod:`~robosimstudio.specs.registries` -- the registries physics composition
  is keyed by: domains and state layouts, engine bundles, backends, couplers,
  parameter schemas, what an engine consumes
* :mod:`~robosimstudio.specs.declare` -- what an asset declares it needs: a
  domain, contact rules by role, what it is to a foreign engine
* :mod:`~robosimstudio.specs.arrangement` -- the arrangement that steps it:
  backends by domain, a coupler, a collision policy, and ``GENERIC``
* :mod:`~robosimstudio.specs.manifest` -- what the composer built, as data
* :mod:`~robosimstudio.specs.physics` -- a facade over those four, so
  ``from robosimstudio.specs.physics import X`` keeps working; ``asset`` reads
  ``SimRequirements`` through it and ``task`` reads ``ArrangementSpec``
* :mod:`~robosimstudio.specs.task` -- what to do, and what counts as done
* :mod:`~robosimstudio.specs.run` -- what a run records and a job asks for

Every name is re-exported here, flat, so a caller imports from
``robosimstudio.specs`` and never has to know which submodule a spec lives in.
Splitting them was for the reader and the blast radius, not for the import line.

**Checked, because it was not true.** Twenty-eight of seventy-two names were
reachable only through their submodule -- ``ShapeSpec``, ``HostActor`` and the
whole :mod:`~robosimstudio.specs.params` vocabulary among them, which is exactly
what a person annotating a new spec's fields with change tiers reaches for first.
``tests/test_specs_facade.py`` compares this ``__all__`` against every submodule's and
fails on a gap in either direction.

Task-specific parameters live on task-specific specs, not in string-keyed
dicts. An asset that needs adhesion constants subclasses :class:`ClothSpec` and
declares them as fields, beside the physics that reads them -- which keeps the
generic specs generic and puts every parameter where its code is.
"""

from __future__ import annotations

from dataclasses import replace

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.core.render.finish import Finish
from robosimstudio.core.render.spec import RenderSpec

from .asset import (
    ArticulatedSpec,
    AssetSpec,
    BodySpec,
    ContactMaterialSpec,
    JointDriveSpec,
    JointSpec,
    ObjectSpec,
    RegionSpec,
    ShapeSpec,
    StaticSpec,
)
from .base import IDENTITY, Quat, Vec3, Xform
from .cell import CellSpec
from .params import (
    LIVE,
    REBUILD,
    RECAPTURE,
    TIERS,
    ParamUI,
    iter_params,
    param_ui,
    ui,
    undescribed,
)
from .physics import (
    ACTOR_CONSUMES,
    BACKENDS,
    COUPLERS,
    COUPLINGS,
    CPU,
    DOMAINS,
    GENERIC,
    MUJOCO_500HZ,
    ROLES,
    STATE_KINDS,
    ArrangementSpec,
    BackendInfo,
    BackendSpec,
    ChannelRecord,
    CollisionSpec,
    ContactRule,
    CouplerInfo,
    CouplerSpec,
    DomainInfo,
    HostActor,
    IslandRecord,
    ParamRule,
    ParamSchema,
    PhysicsManifest,
    SimRequirements,
    actor_consumes,
    backend,
    contacts_with,
    coupler,
    declare_consumes,
    engine_consumes,
    get_backend,
    get_coupler,
    get_domain,
    list_domains,
    register_backend,
    register_coupler,
    register_domain,
    rule,
    sim_to_dict,
)
from .robot import ActuationSpec, ArmSpec, EmbodimentSpec, GripperSpec, jaw_gap_from
from .run import CaptureSpec, JobSpec, RecordSpec
from .task import (
    PARK_HEIGHT,
    PARK_PITCH,
    ActionSpec,
    Binding,
    Episode,
    EpisodeSpec,
    ReproHooksSpec,
    TaskRequirements,
    TaskSpec,
    TaskSuite,
    park_lattice,
    park_poses,
    restage,
)

__all__ = [
    "ACTOR_CONSUMES",
    "BACKENDS",
    "COUPLERS",
    "COUPLINGS",
    "CPU",
    "DOMAINS",
    "GENERIC",
    "IDENTITY",
    "LIVE",
    "MUJOCO_500HZ",
    "PARK_HEIGHT",
    "PARK_PITCH",
    "REBUILD",
    "RECAPTURE",
    "ROLES",
    "STATE_KINDS",
    "TIERS",
    "ActionSpec",
    "ActuationSpec",
    "ArmSpec",
    "ArrangementSpec",
    "ArticulatedSpec",
    "AssetSpec",
    "BackendInfo",
    "BackendSpec",
    "Binding",
    "BodySpec",
    "CameraSpec",
    "CaptureSpec",
    "CellSpec",
    "ChannelRecord",
    "CollisionSpec",
    "ContactMaterialSpec",
    "ContactRule",
    "CouplerInfo",
    "CouplerSpec",
    "DomainInfo",
    "EmbodimentSpec",
    "Episode",
    "EpisodeSpec",
    "Finish",
    "GripperSpec",
    "HostActor",
    "IslandRecord",
    "JobSpec",
    "JointDriveSpec",
    "JointSpec",
    "ObjectSpec",
    "ParamRule",
    "ParamSchema",
    "ParamUI",
    "PhysicsManifest",
    "Quat",
    "RecordSpec",
    "RegionSpec",
    "RenderSpec",
    "ReproHooksSpec",
    "ShapeSpec",
    "SimRequirements",
    "StaticSpec",
    "TaskRequirements",
    "TaskSpec",
    "TaskSuite",
    "Vec3",
    "Xform",
    "actor_consumes",
    "backend",
    "contacts_with",
    "coupler",
    "declare_consumes",
    "engine_consumes",
    "get_backend",
    "get_coupler",
    "get_domain",
    "iter_params",
    "jaw_gap_from",
    "list_domains",
    "param_ui",
    "park_lattice",
    "park_poses",
    "register_backend",
    "register_coupler",
    "register_domain",
    "replace",
    "restage",
    "rule",
    "sim_to_dict",
    "ui",
    "undescribed",
]
