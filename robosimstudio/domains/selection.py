# SPDX-License-Identifier: Apache-2.0
"""Particle selectors: how a spec names a set of its own particles.

A sheet names its hem and a sponge its top face before either is built, so
neither can say which of the scene's particle indices those are. A spec carries
a :data:`Selector` instead: a function of the body's own rest positions that
returns a mask or an index array. :func:`resolve_selector` maps the answer back
onto the scene's particle array once the body exists, and the three
``selector_*`` constructors cover the selections that keep being written.

Kept in the library, beside the kinds rather than inside one, because every
kind that carries particles states its sets the same way: ``ClothSpec.particle_sets``,
and ``SoftSpec.particle_sets``, ``anchored`` and ``rigid``. The cloth and the
soft implementations both import these from here and neither imports the
other, so each one can be read, copied or left out on its own.

This module imports numpy and nothing of the library.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

__all__ = ["Selector", "resolve_selector", "selector_beyond", "selector_extreme", "selector_indices"]

Selector = Callable[[np.ndarray], np.ndarray]
"""``rest_positions [n, 3] -> boolean mask or index array``."""


def resolve_selector(selector: Selector, rest: np.ndarray, particles: np.ndarray) -> np.ndarray:
    """Turn a particle-set selector into global particle indices.

    A selector sees only *this* cloth's rest positions, so it can be written in
    the sheet's own terms ("everything past the box edge") without knowing where
    the cloth landed in the scene's particle array. The mapping back is here.
    """
    picked = np.asarray(selector(rest))
    if picked.dtype == bool:
        if len(picked) != len(rest):
            raise ValueError(f"selector returned a mask of {len(picked)} for {len(rest)} particles")
        return particles[np.flatnonzero(picked)].astype(np.int32)
    return particles[np.asarray(picked, dtype=np.int64)].astype(np.int32)


def selector_beyond(axis: int, threshold: float, *, above: bool = True) -> Selector:
    """Particles past a plane, in the sheet's own coordinates.

    The workhorse selector: "the tab" is everything past the box edge, "the
    hem" is everything below a height. Written as a function rather than a rule
    string because that is all it takes.
    """

    def select(rest: np.ndarray) -> np.ndarray:
        return rest[:, axis] > threshold if above else rest[:, axis] < threshold

    return select


def selector_extreme(axis: int, *, band: float, high: bool = True) -> Selector:
    """Particles within ``band`` [m] of a sheet's extreme along an axis.

    What picks out an edge row -- a handle, a corner, a graspable rim -- without
    the caller having to know where the sheet ended up in world coordinates.
    """

    def select(rest: np.ndarray) -> np.ndarray:
        edge = rest[:, axis].max() if high else rest[:, axis].min()
        return rest[:, axis] > edge - band if high else rest[:, axis] < edge + band

    return select


def selector_indices(indices: Sequence[int]) -> Selector:
    """A literal set of local indices, for a selection with no geometric rule."""
    idx = np.asarray(indices, dtype=np.int64)

    def select(rest: np.ndarray) -> np.ndarray:
        del rest
        return idx

    return select
