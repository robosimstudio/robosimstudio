# SPDX-License-Identifier: Apache-2.0
"""What every import needs on the way into a builder, and nothing a task knows.

Build-time only: each function here runs once, on a :class:`newton.ModelBuilder`,
before ``finalize``. They lived in ``core/builder.py`` until it was clear that
nothing in core called them -- core keeps the custom-attribute re-key and the
substep guard, which its own schedule needs, and the scene tier keeps the fixups
its entities apply:

* :mod:`~robosimstudio.scene.build.imports` -- an MJCF or a USD stage into a
  builder with what the importers drop restored: the collapse re-key applied to
  both, the colliders an Isaac export hides, the textures ``add_mjcf`` binds to
  meshes and to nothing else, and the loose props an arena has to lose.
* :mod:`~robosimstudio.scene.build.colliders` -- a substitute collider that is a
  substitute for the material as well as the shape, self-collision filtering for
  a mechanism, a slab decomposition, and the SDF a hydroelastic shape needs.
* :mod:`~robosimstudio.scene.build.bodies` -- a body's mass held across a
  collider swap, and a stated mass restated after one.
"""

from __future__ import annotations

from .bodies import preserve_body_inertia, set_body_mass
from .colliders import (
    SDF_MAX_RESOLUTION,
    SDF_NARROW_BAND_RANGE,
    build_hydroelastic_sdf,
    collider_vertices,
    copy_shape_material,
    filter_self_collisions,
    slab_decomposition,
)
from .imports import (
    add_mjcf_collapsed,
    add_usd_stage,
    clear_texture_tints,
    expose_declared_colliders,
    is_textured,
    strip_free_bodies,
    texture_box_geoms,
)

__all__ = [
    "SDF_MAX_RESOLUTION",
    "SDF_NARROW_BAND_RANGE",
    "add_mjcf_collapsed",
    "add_usd_stage",
    "build_hydroelastic_sdf",
    "clear_texture_tints",
    "collider_vertices",
    "copy_shape_material",
    "expose_declared_colliders",
    "filter_self_collisions",
    "is_textured",
    "preserve_body_inertia",
    "set_body_mass",
    "slab_decomposition",
    "strip_free_bodies",
    "texture_box_geoms",
]
