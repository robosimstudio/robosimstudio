# SPDX-License-Identifier: Apache-2.0
"""A rod's matter, in the registry that holds materials.

Cloth and foam could be named, saved, loaded and offered in the studio's
preset menu; a cable's drive gains could not, because nothing had declared
what kind of matter a :class:`~robosimstudio.domains.rod.spec.RodMaterialSpec`
is. One ``register_material_kind`` call is that declaration. The one record
is the reference cable's drive as ``RodSpec`` defaults it -- a read-back of
the built model, never a build-time statement, which is why a rod carries no
``material`` field (see ``RodMaterialSpec``'s docstring).
"""

from __future__ import annotations

from robosimstudio.assetlib.materials import MaterialRecord, register_material, register_material_kind

from .spec import RodMaterialSpec

__all__ = ["REFERENCE_CABLE_DRIVE"]

register_material_kind("rod", RodMaterialSpec)

REFERENCE_CABLE_DRIVE = register_material(
    MaterialRecord(
        name="reference_cable_drive",
        spec=RodMaterialSpec(),
        notes="The shipped capsule-chain cable's drive gains, as RodSpec defaults them; a starting point, not a measurement.",
    )
)
