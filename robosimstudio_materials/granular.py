# SPDX-License-Identifier: Apache-2.0
"""The ``granular`` kind: a heap of grains -- rice, sand, beans, flour -- on Newton's implicit MPM.

This file is what ``rss.heap`` builds, imported as ``robosimstudio_materials.granular``.
Run ``examples/03_materials/11_granular.py`` first. The implementation is shared
with the liquid and dough kinds in :mod:`robosimstudio_materials.mpm`; this module
is the domain, the entity that names it, the constructor and the presets.

A heap is a continuum of particles, not grains: the MPM smears a bean into a
material with a bean's friction and density, and a tool thinner than about two
voxels (8 mm at the registered 4 mm) leaves grains inside itself as it pushes
(``probe_b``: a 3 mm blade kept ~300 particles inside it and left 10-16% of the
rice it swept behind it). A paddle or a scoop is 8 mm thick or more, and a
tool that slides along the bench names itself in the heap's ``containers`` so
the grid sees it one voxel fatter than the hand does: otherwise the bottom
voxel of grains passes under it (``probe_sweep``, GPU 3: 40% of a heap left
behind a 160 mm pusher with no margin, 10% with one). What a tool pushes
pushes back, and a parallel gripper feels it along its closing axis: 16,000
grains of rice ahead of that pusher opened the YAM's jaws from the pusher's
20 mm to 42 mm, with the fingers still either side of it -- measured on the
YAM's old hand, a 200 N/m spring, not re-measured on the 50 N-a-jaw clamp it
has now.
"""

from __future__ import annotations

from typing import Any, ClassVar

from robosimstudio.scene.registry import register_entity
from robosimstudio.specs.base import Vec3
from robosimstudio.specs.physics import register_domain

from .mpm import (
    BEANS,
    FLOUR,
    GRANULAR_60HZ,
    RICE,
    SAND,
    EntityParticleBody,
    Heap,
    ParticleBodySpec,
    ParticleMaterialSpec,
    resolve_material,
)

__all__ = [
    "BEANS",
    "FLOUR",
    "GRANULAR_60HZ",
    "RICE",
    "SAND",
    "EntityHeap",
    "Heap",
    "ParticleBodySpec",
    "ParticleMaterialSpec",
    "heap",
]


class EntityHeap(EntityParticleBody):
    """A granular body: :class:`~robosimstudio_materials.mpm.EntityParticleBody` on the ``granular`` domain."""

    domain: ClassVar[str] = "granular"


def heap(
    *,
    name: str = "heap",
    size: Vec3 = (0.12, 0.12, 0.03),
    at: Vec3 = (0.0, 0.0, 0.0),
    material: Any = "rice",
    **kwargs,
) -> ParticleBodySpec:
    """A block of grains, sampled on a lattice, that falls into a heap when the scene starts.

    Args:
        name: Key in handles, probes, measures and particle-set names.
        size: Metres, ``(x, y, z)`` of the block as built; it slumps to its
            angle of repose in the settle. The default is 27,648 particles at
            the default 2.5 mm spacing.
        at: Where the **bottom-face centre** goes, in the cell's work-surface
            frame: ``at=(0, 0, 0)`` stands the block on the middle of the bench.
        material: A preset name (``"rice"``, ``"sand"``, ``"beans"``,
            ``"flour"``), its record, or a
            :class:`~robosimstudio_materials.mpm.ParticleMaterialSpec`.
        **kwargs: Any other :class:`~robosimstudio_materials.mpm.ParticleBodySpec`
            field: ``shape="cylinder"``, ``particle_spacing``, ``particle_sets``,
            ``render="surface"``, ``containers``.

    A heap needs the ``mpm`` backend: :data:`~robosimstudio.specs.GENERIC` routes
    ``granular`` to it, and :data:`GRANULAR_60HZ` adds a rigid tool to the
    robot's island so the hand can hold one.
    """
    return ParticleBodySpec(entity="heap", name=name, size=size, at=at, material=resolve_material(material), **kwargs)


DOMAIN = register_domain(
    "granular",
    state=("particles",),
    reads=("granular", "cloth"),
    role="deformables",
    contacts=("surface", "fixtures", "objects", "hands", "deformables"),
    summary="A heap of grains: particles on Newton's implicit MPM.",
    engine="mpm",
    staged_by="GRANULAR_60HZ",
)

register_entity("heap")(EntityHeap)
