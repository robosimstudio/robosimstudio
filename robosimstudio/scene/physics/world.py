# SPDX-License-Identifier: Apache-2.0
"""Index expansion for replicated models.

``ModelBuilder.replicate`` lays world *w*'s entities at ``w * per_world`` offsets
(:class:`~robosimstudio.core.worlds.WorldMap`), so an island's ownership lists,
resolved once against the template, become every world's by adding the stride.
Shapes added after replication -- the ground plane -- live in the global world
and are kept as they are.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

__all__ = ["REPLICATED_HOSTS", "expand_indices", "expand_shapes"]

REPLICATED_HOSTS: frozenset[str] = frozenset({"superdex"})
"""Foreign engines whose bridge lays a replicated model's worlds out itself.

An allow-list rather than a flag on the backend, and here rather than beside
either refusal it narrows, because both of them read it -- the composer, which
decides whether an island may be built, and the compiler, which decides whether
a scene may be replicated at all.

What it asserts is a property of the *bridge* and not of the engine: that the
bridge translates each world somewhere of its own on the way in and takes the
translation off on the way out. A replicated model superimposes its worlds at
the origin, which Newton pairs contacts within and a host scene cannot, so an
engine added to this set without that has replaced a refusal with a pile of
superimposed actors colliding with each other -- which runs, and reports contact
forces for it.
"""


def expand_indices(indices: Iterable[int], per_world: int, num_worlds: int) -> np.ndarray:
    """Template indices, repeated for every world at the world's stride."""
    base = np.asarray(list(indices), dtype=np.int64).reshape(-1)
    if num_worlds <= 1 or base.size == 0:
        return base.astype(np.int32)
    offsets = (np.arange(int(num_worlds), dtype=np.int64) * int(per_world)).reshape(-1, 1)
    return (base.reshape(1, -1) + offsets).reshape(-1).astype(np.int32)


def expand_shapes(indices: Iterable[int], per_world: int, num_worlds: int, total: int) -> np.ndarray:
    """Template shape indices for every world, keeping global shapes (index >= worlds * per_world) once."""
    base = np.asarray(list(indices), dtype=np.int64).reshape(-1)
    if num_worlds <= 1 or base.size == 0:
        return base.astype(np.int32)
    local = base[base < per_world]
    expanded = expand_indices(local, per_world, num_worlds)
    global_shapes = np.arange(int(per_world) * int(num_worlds), int(total), dtype=np.int32)
    keep = np.asarray([s for s in global_shapes if s in set(base.tolist()) or s >= per_world], dtype=np.int32)
    return np.unique(np.concatenate([expanded, keep])).astype(np.int32)
