# SPDX-License-Identifier: Apache-2.0
"""Writing a tuned material into a running simulation.

Three tiers, described per field in :mod:`robosimstudio.specs.params`. This
module is the whole surface a panel binds against; one of the three is
implemented a layer down and re-exported here.

**Live** is the interesting one, and the reason this studio can exist at all --
assigning into a cloth's constitutive Warp arrays takes effect on the very next
substep, and does so **even inside a captured CUDA graph**. It lives in
:mod:`robosimstudio.scene.tuning`, which says why at length: the entity that
owns those elements has to be able to answer for its own tunables, and ``scene``
is the lowest layer that can express the write. :class:`LiveMaterial` and
:func:`locate_cloth` are that module's, named here so a caller of this one sees
a single surface.

**Recapture** is the soft-contact trio. Those are plain Python floats on the
model, passed into kernels by value, so a capture bakes the *value*. Writing
them is legal at any time but a captured graph goes on using the old ones until
it is recaptured -- which is why this studio runs eager by default and the whole
tier becomes free.

**Rebuild** is everything consumed at build time: geometry, densities, solver
buffer sizes. Nothing here can help; the session builds a fresh model.

One asymmetry worth knowing before trusting a number: the live tier is written
**per cloth**, because those arrays are indexed per element and
:func:`locate_cloth` selects the elements belonging to one entity. The recapture
tier is written **globally**, because ``soft_contact_ke`` and friends are single
scalars on the model with no per-cloth dimension to select on. In a rig with one cloth that
distinction does not arise; in a side-by-side comparison it does, and contact
settings are the one thing the two sheets cannot disagree about.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
from robosimstudio.domains.cloth import ClothMaterialSpec

from robosimstudio.scene.tuning import ClothTarget, LiveMaterial, locate_cloth
from robosimstudio.specs.params import LIVE, RECAPTURE, param_ui

__all__ = ["ClothTarget", "LiveMaterial", "apply_contact", "contact_differs", "locate_cloth", "rebuild_fields"]


# ------------------------------------------------------------- recapture tier
_CONTACT = {"contact_ke": "soft_contact_ke", "contact_kd": "soft_contact_kd", "friction": "soft_contact_mu"}
"""Material field -> the model scalar it sets. Global, not per cloth."""


def apply_contact(model: Any, spec: ClothMaterialSpec) -> tuple[str, ...]:
    """Write the recapture-tier contact scalars onto the model.

    These are model-wide. The caller is responsible for the graph: either the
    pipeline is running eager, or it needs invalidating afterwards.
    """
    for field_name, attr in _CONTACT.items():
        setattr(model, attr, float(getattr(spec, field_name)))
    return tuple(_CONTACT)


def contact_differs(model: Any, spec: ClothMaterialSpec) -> bool:
    """Whether the model's contact scalars already match ``spec``.

    Lets a caller skip a graph invalidation nobody needs.
    """
    return any(
        not np.isclose(float(getattr(model, attr, np.nan)), float(getattr(spec, field_name)))
        for field_name, attr in _CONTACT.items()
    )


def rebuild_fields(current: Any, wanted: Any) -> tuple[str, ...]:
    """Which rebuild-tier fields differ between two specs.

    What a panel shows as "pending", and what makes the rebuild button worth
    pressing rather than a no-op.
    """
    out: list[str] = []
    for f in fields(wanted):
        described = param_ui(wanted, f.name)
        if described is None or described.tier in (LIVE, RECAPTURE):
            continue
        if getattr(current, f.name) != getattr(wanted, f.name):
            out.append(f.name)
    return tuple(out)
