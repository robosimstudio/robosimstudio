# SPDX-License-Identifier: Apache-2.0
"""The ``rod`` kind -- the rope: what it is, what builds it, what reads it.

One bundle per physical kind. Importing this registers the domain and everything
that comes with it -- the entity that builds one, and any measure that only makes
sense over it -- so a scene and a goal cannot disagree about whether this kind of
thing exists.

Reached through the ``DOMAINS`` registry's package scan rather than by anyone
importing it: a domain has to be registered before a goal binds, and the layer
below cannot name the layer that holds it.

**A pack: this directory is the whole of the rod, and a tree may leave it
out.** Nothing in ``robosimstudio/`` names it, so everything that is the rod's is
here and reaches the library by registering, through seams any other kind of
deformable could use the same way:

* ``entity`` / ``superdex`` -- the two entities, by ``register_entity("rod", engine=...)``;
* ``centerline`` -- the ``Centerline`` capability, by ``register_capability``;
* ``actor`` -- how a ``"rod"`` actor is read off the model and how SuperDex builds
  one, by ``register_actor_kind`` and ``register_actor_builder``;
* ``features`` -- the arc-length ``point`` feature kind, by ``register_kind``;
* ``measures`` -- ``winding_turns``, ``straightness`` and the layout's own
  ``height`` and ``speed``, by ``register_atom`` and ``handler``;
* ``materials`` -- the rod material kind and its preset;
* ``latch`` -- the grasp latch's kernel and device state. No registry: the entity
  uses it, and a repro bundle copies it.

The ``"rod"`` state layout itself is minted by ``register_domain(reads="rod")``
below. What the pack takes *from* the library rather than giving it is the post:
:class:`EntityPeg` is the static domain's ``EntityPost`` under the cable suites'
name. ``tests/test_rod_is_a_pack.py`` holds all of this, in both directions.
"""

from __future__ import annotations

import sys as _sys

from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.physics import register_domain

from . import actor as _actor  # noqa: F401 -- registers how a rod actor is read, and how SuperDex builds one
from . import latch as _latch
from . import materials as _materials  # noqa: F401 -- registers the rod material kind and its preset
from . import measures as _measures  # noqa: F401 -- registers winding_turns and straightness
from .centerline import Centerline
from .declare import Rod
from .entity import (
    CableDriveSpec,
    CableLatchSpec,
    EntityCable,
    EntityPeg,
    EntityPegRound,
    EntityRod,
    SpecCable,
    fixture_shape_cfg,
)
from .features import KIND_NAMES, RodParam  # registers the arc-length ``point`` feature kind
from .make import rod
from .spec import RodMaterialSpec, RodSpec

__all__ = [
    "KIND_NAMES",
    "CableDriveSpec",
    "CableLatchSpec",
    "Centerline",
    "EntityCable",
    "EntityPeg",
    "EntityPegRound",
    "EntityRod",
    "EntityRodSuperdex",
    "Rod",
    "RodMaterialSpec",
    "RodParam",
    "RodSpec",
    "SpecCable",
    "fixture_shape_cfg",
    "rod",
]

_sys.modules.setdefault("robosimstudio.solvers.grasp.latch", _latch)
"""The latch, under the dotted name content still declares it by.

The module is :mod:`robosimstudio.domains.rod.latch`. It was
``robosimstudio/solvers/grasp/latch.py`` while the rod's code was spread through
the library, and the cable suite's repro hooks name it by that path as a
*string* -- ``ReproHooksSpec(module="robosimstudio.solvers.grasp.latch")`` --
which the bundle exporter resolves with ``importlib.import_module`` to find the
file it copies verbatim. A string is not an import anything can rewrite, so the
old name is kept resolving to this module for as long as content spells it:
installed by the pack, on import, so the library holds no rod file to make it
work and a checkout without the pack has no such module.

``setdefault``, so a real module of that name -- should the library ever grow
one -- wins over this. Drop the line when the suite names the new path.
"""

DOMAIN = register_domain(
    "rod",
    # Bodies **and** particles, because a rod is not one representation. Newton
    # solves one as a chain of capsules on cable joints (bodies, joints,
    # shapes); SuperDex solves the same cable as a Cosserat centerline
    # (particles). The composer checks a backend against what an entity
    # *actually owns* rather than against everything the domain may own, so
    # listing both is what lets each implementation be refused by a backend that
    # cannot carry it -- a capsule chain on a particles-only solver still fails,
    # and says which state it could not own.
    state=("bodies", "joints", "particles", "shapes"),
    reads="rod",
    role="deformables",
    contacts=("surface", "fixtures", "objects", "hands", "deformables"),
    summary="A cable: a capsule chain on Newton, a Cosserat centerline on SuperDex.",
)

# The generic names for this kind of thing, so a spec can ask for one without a
# task directory having bound it first.
#
# **This reverses a doctrine, and the reversal is deliberate.**
# The cable suite's ``rod.py`` argues that "every entity name in this tree
# is declared by the task that stages it, and none by the library". That was
# right about *content vocabulary* and cannot survive a library whose rod is
# buildable. The line held instead: the library registers only the **domain's
# own generic name** -- exactly as ``register_domain("rod", ...)`` above already
# puts ``"rod"`` in the library -- and every descriptive name stays content's.
# ``cable_rod``, ``hang_line`` and ``hang_post`` are unaffected: the registry
# holds them as separate keys onto the same classes.
#
# ``"post"`` is not registered here. ``EntityPeg`` is the static domain's
# ``EntityPost`` and that domain registers it, so ``rss.post`` builds whether or
# not this pack is installed.
register_entity("rod")(EntityRod)

# The same kind, the other engine. Which one a scene gets is the arrangement's
# ``("rod", backend(...))`` line and nothing else; the spec is the same either
# way. Imported here rather than at the top because it is a *second*
# implementation and the module ordering should say so.
from .superdex import EntityRodSuperdex

register_entity("rod", engine="superdex")(EntityRodSuperdex)
