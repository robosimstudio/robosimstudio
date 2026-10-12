# SPDX-License-Identifier: Apache-2.0
"""Specs and an episode to a steppable, indexed scene.

Where the IR becomes something Newton can run. Everything here is about
*assembly*: which entities to build, in what order, bound to what model, indexed
how -- and nothing about what the scene is for, which is the task's business.

* :mod:`~robosimstudio.scene.compiler` -- the ten fixed steps, in the one order
  that works; every one of them is there because doing it later fails silently.
* :mod:`~robosimstudio.scene.build` -- what every import needs on the way into a
  builder: the collapse re-key applied to MJCF and USD, the textures the importers
  drop, and collider substitution that keeps a body's material and mass.
* :mod:`~robosimstudio.scene.entity` -- one class per kind of thing that can be
  built: rigid bodies, cloth, scenery, and the robot itself.
* :mod:`~robosimstudio.scene.handles` -- where everything landed, by name.
* :mod:`~robosimstudio.scene.registry` -- which cells and which entity classes
  exist.
* :mod:`~robosimstudio.scene.frames` -- the work-surface frame the three axes
  agree in, and the only place surface-relative becomes world.
* :mod:`~robosimstudio.scene.builtin` -- the one scene the library ships:
  ``void``, which declares no geometry and is the null hypothesis every measured
  cell is checked against.

The embodiment is built here rather than in the robot library, and that is
deliberate: welding a hand onto a wrist is an operation on a *builder*, so it
belongs with the other things that build. The robot library supplies what it
needs -- the specs, the IK, the gripper law -- and never learns that a scene
exists.
"""

from __future__ import annotations

from .assets import Added, load_asset
from .compiler import CannotReplicate, CompiledScene, compile_scene
from .frames import in_surface, surface_frame
from .handles import ArmHandle, DeformableHandle, EntityHandle, RegionHandle, SceneHandles
from .registry import CELLS, ENTITIES, get_cell, get_entity, list_cells, register_cell, register_entity
from .request import BuildRequest, SceneRequest

__all__ = [
    "CELLS",
    "ENTITIES",
    "Added",
    "ArmHandle",
    "BuildRequest",
    "CannotReplicate",
    "CompiledScene",
    "DeformableHandle",
    "EntityHandle",
    "RegionHandle",
    "SceneHandles",
    "SceneRequest",
    "compile_scene",
    "get_cell",
    "get_entity",
    "in_surface",
    "list_cells",
    "load_asset",
    "register_cell",
    "register_entity",
    "surface_frame",
]
