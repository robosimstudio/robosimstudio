# SPDX-License-Identifier: Apache-2.0
"""Turning a declared feature into something a handler can read.

Resolution happens **once**, at scene-ready, against the rest layout -- the same
moment and the same input the cloth entity already uses for its particle sets.
Everything after that is indexing, so a feature costs nothing per step no matter
which tier it was declared in.

The three tiers converge here:

* ``literal`` -- the constants are the answer. A loop's centre and normal are
  returned as they were written, transformed into the body frame if they were not
  already in it.
* ``recipe`` -- the closed selector language is compiled and evaluated against
  the rest positions, yielding an index set.
* ``resolver`` -- registered, versioned code is looked up by key and called.

:data:`RESOLVERS` is a registry like every other in this stack, which buys three
things worth having: a duplicate name is refused with both origins named, the
version suffix (``collar_curve@v1``) makes two generations of a resolver coexist
as distinct keys rather than one silently shadowing the other, and a sidecar can
name code it does not contain. That last point is the whole reason the tier
exists: a resolver reference is *data*, so the hard rule holds -- nothing
executable is ever embedded in a sidecar.

A resolver is expected to be pure and deterministic over its inputs. Nothing here
enforces that, and nothing can; the mint gate is where a new resolver earns its
place, exactly as an atom does.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from robosimstudio.registry import Registry

__all__ = [
    "RESOLVERS",
    "get_resolver",
    "list_resolvers",
    "register_resolver",
]

Resolver = Callable[..., np.ndarray]
"""Registered code resolving a selection: ``(rest[n, 3], **payload) -> mask | indices``."""

RESOLVERS: Registry[Resolver] = Registry("feature resolver")
"""Every registered resolver. Loads no packages -- resolvers register from the
asset modules that need them, which the asset registry already imports."""


def register_resolver(name: str) -> Callable[[Resolver], Resolver]:
    """Decorator registering a resolver under a versioned key.

    The key carries its version (``collar_curve@v1``). Bumping it is how a
    resolver changes behaviour without invalidating every sidecar that referenced
    the old one -- the two coexist, and an asset moves when somebody re-bakes it
    and says so.
    """

    def decorate(fn: Resolver) -> Resolver:
        RESOLVERS.register(name, fn, origin=fn.__module__)
        return fn

    return decorate


def get_resolver(name: str) -> Resolver:
    """Look one up by versioned key."""
    return RESOLVERS.get(name)


def list_resolvers() -> list[str]:
    """Every registered resolver key, sorted."""
    return RESOLVERS.names()
