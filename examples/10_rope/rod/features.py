# SPDX-License-Identifier: Apache-2.0
"""What only a chain can carry: the arc-length ``point``, and the protocol a chain answers.

A feature kind is general when its geometry is: a ``loop``, an ``axis``, a
``region`` can be annotated on anything, and those are registered by the asset
library. ``point`` is the other sort. It is "a position along a 1-D body, by
normalised arc length", it is stated on the ``rod`` layout and nothing else can
carry it -- annotating one on a rigid body is a mistake the validator refuses.
So it is registered here, by the domain that mints that layout, and a checkout
without this domain has neither the layout nor a kind that could only be stated
on it.

:data:`KIND_NAMES` is this pack's half of the deliberate list the asset library
keeps for its own kinds (``assetlib.semantics.library.KIND_NAMES``): what the
feature-kind registry holds is that tuple plus every pack's.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from robosimstudio.assetlib.semantics.kinds import FeatureKind, register_kind

__all__ = ["KIND_NAMES", "POINT", "RodParam"]


POINT = register_kind(
    FeatureKind(
        name="point",
        summary="A position along a 1-D body, by normalised arc length.",
        payload={"s": "arc length in [0, 1]"},
        contracts=(
            "arc length is normalised, so a reference survives a change of segment count",
            "material identity, not world position: the cable moves, its midpoint does not stop being its midpoint",
        ),
        substrates=("rod",),
        provenance="Free from the rod substrate's own parameterisation.",
    )
)

KIND_NAMES: tuple[str, ...] = ("point",)
"""Every feature kind this domain registers, sorted. Pinned by the same test as the library's."""


@runtime_checkable
class RodParam(Protocol):
    """A 1-D body with an arc-length parameterisation.

    The reason a cable needs **no** feature authoring at all: the substrate ships
    its own coordinate system, so ``cable.point(0.5)`` is meaningful the moment
    the thing exists. Arc-length is normalised to ``[0, 1]`` so a reference
    survives a change of segment count.
    """

    def point(self, s: float) -> np.ndarray:
        """World position at normalised arc length ``s`` in ``[0, 1]`` [m]."""
        ...

    def segment(self, s: float) -> int:
        """Index of the segment containing normalised arc length ``s``."""
        ...

    def positions(self, body_q: np.ndarray) -> np.ndarray:
        """Every segment's world position ``[n, 3]``."""
        ...
