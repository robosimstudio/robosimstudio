# SPDX-License-Identifier: Apache-2.0
"""Probes: the measurements a success test or an observation is built from.

:class:`ContactProbe` is imported lazily by name rather than eagerly, because it
needs ``mujoco_warp``; a scene that never scores contacts should not pay for it.
"""

from .clearance import ClearanceProbe
from .particle import ParticleProbe

__all__ = ["ClearanceProbe", "ContactProbe", "ParticleProbe"]


def __getattr__(name: str):
    if name == "ContactProbe":
        from .contact import ContactProbe

        return ContactProbe
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
