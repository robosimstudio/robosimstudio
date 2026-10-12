# SPDX-License-Identifier: Apache-2.0
"""A handle on one thing in a built scene: what you got back when you added it.

``scene.add(spec)`` returns one of these, and it is the answer to "I put a thing
in a scene, now where is it". The alternative -- and what every caller in this
tree did before there was a :class:`Ref` -- is to keep the spec, remember the
name you gave it, and index ``sim.handles.objects[name].body`` into
``state.body_q``. That works, and it is four facts a caller has to hold that the
scene already knows.

**The accessors a thing offers are a fact about the thing**, so the class is
declared on the entity: ``Entity.ref`` is a :class:`Ref` subclass, and an entity
with something else to offer -- a rod's arc-length points, a cloth's corner --
says so there and gets it here without the compiler learning what a rod is. The
base class is what the compiler knows about *everything*: a name, a spec, the
indices it owns, and where those indices are right now.

**Nothing here is cached.** A ``Ref`` is a pair of pointers and a name; every
read goes to the live state. That is deliberate -- the alternative is a handle
that is silently one step stale, which is the hardest kind of wrong to see in a
number that is nearly right.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["Ref", "owned_indices"]


class Ref:
    """One named thing in a built scene.

    Args:
        scene: The :class:`~robosimstudio.scene.compiler.CompiledScene` it lives
            in. Held rather than copied out of, so a ``Ref`` survives a reset and
            reads the state that exists now.
        name: The spec's name, which is the key everything else uses.
    """

    __slots__ = ("_scene", "name")

    def __init__(self, scene: Any, name: str) -> None:
        self._scene = scene
        self.name = name

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    # ------------------------------------------------------------ what it is
    @property
    def entity(self) -> Any:
        """The built entity, or ``None`` for something the scene's own import declared."""
        return self._scene.by_name.get(self.name)

    @property
    def spec(self) -> Any:
        """The spec this was built from."""
        entity = self.entity
        return getattr(entity, "spec", None)

    @property
    def handle(self) -> Any:
        """The scene's handle for it -- an ``EntityHandle``, a ``DeformableHandle``, a ``RegionHandle``, or ``None``.

        Deformables and rigid bodies get different handle types because they
        answer different questions: one has a body and a grasp box, the other has
        particles and named particle sets. Both are reachable from here, and
        which one you got is a fact about what you added.
        """
        return self._scene.handles.find(self.name)

    # ---------------------------------------------------------- what it owns
    @property
    def bodies(self) -> np.ndarray:
        entity = self.entity
        return owned_indices(entity, "owned_bodies", "bodies")

    @property
    def shapes(self) -> np.ndarray:
        entity = self.entity
        handle = self.handle
        owned = owned_indices(entity, "owned_shapes", "shapes")
        if owned.size or handle is None:
            return owned
        return np.asarray(getattr(handle, "shapes", ()), dtype=np.int32)

    @property
    def particles(self) -> np.ndarray:
        entity = self.entity
        owned = owned_indices(entity, "owned_particles", "particles")
        if owned.size:
            return owned
        handle = self.handle
        return np.asarray(getattr(handle, "particles", ()), dtype=np.int32)

    # --------------------------------------------------------------- where it is
    @property
    def pose(self) -> np.ndarray:
        """``(x, y, z, qx, qy, qz, qw)`` of its first body, live.

        Raises:
            ValueError: If it owns no body. A cloth has no pose -- it has
                particles -- and a static welded to the world has no ``body_q``
                row at all. Both are answered by :attr:`handle`'s own fields
                rather than by a pose that would have to be invented.
        """
        bodies = self.bodies
        if bodies.size == 0:
            raise ValueError(
                f"{self.name!r} owns no body, so it has no pose. A deformable has .particle_q; "
                f"a welded static has .handle.xform."
            )
        return np.asarray(self._scene.pipeline.state_0.body_q.numpy()[int(bodies[0])], dtype=np.float64)

    @property
    def position(self) -> np.ndarray:
        """Just the translation of :attr:`pose`."""
        return self.pose[:3]

    @property
    def velocity(self) -> np.ndarray:
        """The spatial velocity of its first body, live: ``(vx, vy, vz, wx, wy, wz)``."""
        bodies = self.bodies
        if bodies.size == 0:
            raise ValueError(f"{self.name!r} owns no body, so it has no velocity")
        return np.asarray(self._scene.pipeline.state_0.body_qd.numpy()[int(bodies[0])], dtype=np.float64)

    @property
    def particle_q(self) -> np.ndarray:
        """Live positions of the particles it owns, ``(n, 3)``."""
        particles = self.particles
        if particles.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(self._scene.pipeline.state_0.particle_q.numpy()[particles], dtype=np.float32)

    # ------------------------------------------------------------------ extras
    def tunables(self) -> tuple:
        """What the studio would offer for this thing: its live-editable parameters."""
        entity = self.entity
        asked = getattr(entity, "tunables", None)
        return tuple(asked(self._scene.pipeline)) if callable(asked) else ()

    def world(self, index: int) -> Ref:
        """The same thing in another world of a replicated scene.

        Returns ``self`` for world 0 and for a scene that was never replicated,
        so a caller written for one world reads correctly in both.
        """
        worlds = getattr(self._scene, "worlds", None)
        if index == 0 or worlds is None:
            return self
        return _WorldRef(self._scene, self.name, index)


class _WorldRef(Ref):
    """One thing in world ``n`` of a replicated scene. See :meth:`Ref.world`."""

    __slots__ = ("_world",)

    def __init__(self, scene: Any, name: str, world: int) -> None:
        super().__init__(scene, name)
        self._world = world

    def _shift(self, indices: np.ndarray, kind: str) -> np.ndarray:
        """World ``n``'s block of ``kind``, which is world 0's plus one stride per world.

        The stride comes off :class:`~robosimstudio.core.worlds.WorldMap` rather
        than being counted here: "world ``w`` is a contiguous block starting at
        ``w * per_world``" is that module's stated invariant and everything
        downstream leans on it.
        """
        offset = getattr(self._scene.worlds, f"{kind}_per_world", 0)
        return indices + int(offset) * self._world if indices.size else indices

    @property
    def bodies(self) -> np.ndarray:
        return self._shift(Ref.bodies.fget(self), "bodies")

    @property
    def shapes(self) -> np.ndarray:
        return self._shift(Ref.shapes.fget(self), "shapes")

    @property
    def particles(self) -> np.ndarray:
        return self._shift(Ref.particles.fget(self), "particles")

    def world(self, index: int) -> Ref:
        return Ref(self._scene, self.name) if index == 0 else _WorldRef(self._scene, self.name, index)


def owned_indices(entity: Any, method: str, attribute: str) -> np.ndarray:
    """What an entity owns, asked the way the interaction graph asks it.

    ``owned_bodies()`` first, then the plain attribute, because an entity that
    overrides the method is correcting the attribute -- the rod's bracketed range
    covers the fingertip hulls its own tip decomposition added to the *robot*,
    and reading the attribute there would hand a caller shapes belonging to a
    hand.
    """
    if entity is None:
        return np.zeros(0, dtype=np.int32)
    asked = getattr(entity, method, None)
    if callable(asked):
        return np.asarray(list(asked()), dtype=np.int32).reshape(-1)
    return np.asarray(list(getattr(entity, attribute, ())), dtype=np.int32).reshape(-1)
