# SPDX-License-Identifier: Apache-2.0
"""Where world *w*'s entities live in a replicated model.

``ModelBuilder.replicate`` copies one builder ``W`` times, and it does so
uniformly: world *w*'s bodies, shapes, joints and coordinates each occupy one
contiguous block starting at ``w * per_world_count``. That single fact is what
this module turns into an interface, and it is worth stating out loud because
everything downstream leans on it:

    **A world's slice of a model array indexes exactly like the single-world
    model it was replicated from.**

So an index computed against the unreplicated model -- a body index in a handle,
a joint coordinate in an arm's index set, a shape in a grasp check -- stays
correct for every world, applied to that world's slice. Nothing that resolves
indices has to learn about worlds; it resolves them once against the template and
reads :meth:`WorldMap.view` slices thereafter.

The corollary is the trap: labels are *not* made unique by replication (every
world carries the same label strings), so anything resolving a name to an index
must do it against the template model. Resolving against the replicated one
raises on ambiguity, which is the good failure -- but only if nobody catches it.

:meth:`WorldMap.validate` exists to check the uniformity claim rather than assume
it. It is cheap, it runs once at compile, and the alternative is a stride that is
wrong by one entity and produces plausible motion in the wrong world.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import newton
import numpy as np

__all__ = ["WorldMap"]


@dataclass(frozen=True)
class WorldMap:
    """Per-world strides for a replicated model.

    Attributes:
        num_worlds: How many copies the model holds.
        bodies_per_world: Rigid bodies in one world.
        shapes_per_world: Collision/visual shapes in one world.
        joints_per_world: Joints in one world.
        coords_per_world: Joint coordinates in one world. This is the stride for
            ``joint_q`` and -- with ``use_coord_layout_targets`` on, which the
            compiler sets -- for ``joint_target_q`` too.
        dofs_per_world: Joint degrees of freedom in one world; the stride for
            ``joint_qd``.
        particles_per_world: Deformable particles in one world.
    """

    num_worlds: int
    bodies_per_world: int
    shapes_per_world: int
    joints_per_world: int
    coords_per_world: int
    dofs_per_world: int
    particles_per_world: int

    # ------------------------------------------------------------- construction
    @classmethod
    def of(cls, template: newton.Model, model: newton.Model, num_worlds: int) -> WorldMap:
        """Derive the strides from the template a model was replicated from.

        Args:
            template: The single-world model, finalized from the base builder.
            model: The replicated model actually being stepped.
            num_worlds: How many copies were made.

        Returns:
            The map, already validated against ``model``.
        """
        world_map = cls(
            num_worlds=int(num_worlds),
            bodies_per_world=int(template.body_count),
            shapes_per_world=int(template.shape_count),
            joints_per_world=int(template.joint_count),
            coords_per_world=int(template.joint_coord_count),
            dofs_per_world=int(template.joint_dof_count),
            particles_per_world=int(template.particle_count),
        )
        world_map.validate(model)
        return world_map

    def validate(self, model: newton.Model) -> None:
        """Check that ``model`` really is this map's template, ``num_worlds`` times.

        Raises:
            ValueError: If any count is not an exact multiple, or the model
                disagrees about how many worlds it holds. Both mean the strides
                would silently address the wrong world.
        """
        expected = {
            "body_count": self.bodies_per_world,
            "shape_count": self.shapes_per_world,
            "joint_count": self.joints_per_world,
            "joint_coord_count": self.coords_per_world,
            "joint_dof_count": self.dofs_per_world,
            "particle_count": self.particles_per_world,
        }
        wrong = {
            name: (int(getattr(model, name)), per_world * self.num_worlds)
            for name, per_world in expected.items()
            if int(getattr(model, name)) != per_world * self.num_worlds
        }
        if wrong:
            detail = ", ".join(
                f"{name}: model has {got}, {self.num_worlds} worlds need {want}" for name, (got, want) in wrong.items()
            )
            raise ValueError(f"replicated model is not {self.num_worlds} uniform copies of its template -- {detail}")
        # ``add_ground_plane`` and anything else added outside a world context
        # lands in the global world, which the broadphase collides with every
        # world; that is a shape count the strides above already excluded, so the
        # only thing left to check is the world count itself.
        if int(getattr(model, "world_count", self.num_worlds)) != self.num_worlds:
            raise ValueError(f"model reports {int(model.world_count)} worlds, replicated for {self.num_worlds}")

    # ------------------------------------------------------------------ indices
    def body(self, index: int, world: int) -> int:
        """World ``world``'s copy of template body ``index``."""
        return int(index) + int(world) * self.bodies_per_world

    def shape(self, index: int, world: int) -> int:
        return int(index) + int(world) * self.shapes_per_world

    def joint(self, index: int, world: int) -> int:
        return int(index) + int(world) * self.joints_per_world

    def coord(self, index: int, world: int) -> int:
        return int(index) + int(world) * self.coords_per_world

    def dof(self, index: int, world: int) -> int:
        return int(index) + int(world) * self.dofs_per_world

    def particle(self, index: int, world: int) -> int:
        return int(index) + int(world) * self.particles_per_world

    # ------------------------------------------------------------------- slices
    def bodies(self, world: int) -> slice:
        start = int(world) * self.bodies_per_world
        return slice(start, start + self.bodies_per_world)

    def shapes(self, world: int) -> slice:
        start = int(world) * self.shapes_per_world
        return slice(start, start + self.shapes_per_world)

    def coords(self, world: int) -> slice:
        start = int(world) * self.coords_per_world
        return slice(start, start + self.coords_per_world)

    def dofs(self, world: int) -> slice:
        start = int(world) * self.dofs_per_world
        return slice(start, start + self.dofs_per_world)

    def particles(self, world: int) -> slice:
        start = int(world) * self.particles_per_world
        return slice(start, start + self.particles_per_world)

    # -------------------------------------------------------------------- views
    def view(self, values: np.ndarray, world: int, *, per_world: int) -> np.ndarray:
        """One world's block of a host-side model array.

        A view, not a copy: the caller reads it and the next snapshot overwrites
        the buffer underneath. Anything that must outlive a snapshot -- a start
        pose held for the length of an episode -- copies it explicitly.

        Args:
            values: Host array whose leading axis runs over all worlds' entities.
            world: Which world.
            per_world: That array's per-world count, from this map.
        """
        start = int(world) * int(per_world)
        return values[start : start + int(per_world)]

    def body_view(self, body_q: np.ndarray, world: int) -> np.ndarray:
        """World ``world``'s body transforms, indexed like the template's."""
        return self.view(body_q, world, per_world=self.bodies_per_world)

    def coord_view(self, joint_q: np.ndarray, world: int) -> np.ndarray:
        """World ``world``'s joint coordinates, indexed like the template's."""
        return self.view(joint_q, world, per_world=self.coords_per_world)

    def dof_view(self, joint_qd: np.ndarray, world: int) -> np.ndarray:
        return self.view(joint_qd, world, per_world=self.dofs_per_world)

    def particle_view(self, particle_q: np.ndarray, world: int) -> np.ndarray:
        return self.view(particle_q, world, per_world=self.particles_per_world)

    # ---------------------------------------------------------------- spreading
    def spread(self, indices: Any, *, per_world: int) -> np.ndarray:
        """Template indices as their global index in **every** world. ``[W, n]``.

        The other direction from :meth:`view`, and the one an entity writing
        world-aware substep kernels needs. A kernel launched once for the whole
        fleet is handed index *arrays* rather than scalars, and building them is
        the same arithmetic in every entity that does it -- so it is here, once,
        rather than in each of them with its own off-by-a-world.

        Args:
            indices: One template index, or a sequence of them.
            per_world: That kind's per-world count, from this map.

        Returns:
            ``[num_worlds, len(indices)]`` of global indices, ``int32`` because
            that is what a Warp index array is. A negative index is passed
            through untouched: ``-1`` means "no such thing" throughout this
            stack, and offsetting it would turn a sentinel into somebody's body.
        """
        flat = np.atleast_1d(np.asarray(indices, dtype=np.int64)).reshape(-1)
        offsets = np.arange(self.num_worlds, dtype=np.int64)[:, None] * int(per_world)
        out = np.where(flat[None, :] < 0, flat[None, :], flat[None, :] + offsets)
        return out.astype(np.int32)

    def spread_bodies(self, indices: Any) -> np.ndarray:
        """Template body indices in every world. ``[W, n]``."""
        return self.spread(indices, per_world=self.bodies_per_world)

    def spread_shapes(self, indices: Any) -> np.ndarray:
        return self.spread(indices, per_world=self.shapes_per_world)

    def spread_joints(self, indices: Any) -> np.ndarray:
        return self.spread(indices, per_world=self.joints_per_world)

    def spread_coords(self, indices: Any) -> np.ndarray:
        return self.spread(indices, per_world=self.coords_per_world)

    def spread_dofs(self, indices: Any) -> np.ndarray:
        return self.spread(indices, per_world=self.dofs_per_world)

    def spread_particles(self, indices: Any) -> np.ndarray:
        return self.spread(indices, per_world=self.particles_per_world)

    # ------------------------------------------------------------------- tiling
    def tile_coords(self, per_world_values: np.ndarray) -> np.ndarray:
        """One world's coordinate vector, repeated for every world.

        The start of a wave reset: take the template's home posture and make the
        flat model-wide vector out of it, which one ``assign`` then writes.
        """
        values = np.asarray(per_world_values, dtype=np.float32).reshape(-1)
        if values.size != self.coords_per_world:
            raise ValueError(f"expected {self.coords_per_world} coordinates, got {values.size}")
        return np.tile(values, self.num_worlds)

    def stack_coords(self, rows: list[np.ndarray]) -> np.ndarray:
        """Per-world coordinate vectors, flattened into the model's layout.

        Args:
            rows: One vector per world, each the template's coordinate count.
        """
        if len(rows) != self.num_worlds:
            raise ValueError(f"expected {self.num_worlds} rows, got {len(rows)}")
        stacked = np.empty((self.num_worlds, self.coords_per_world), dtype=np.float32)
        for world, row in enumerate(rows):
            values = np.asarray(row, dtype=np.float32).reshape(-1)
            if values.size != self.coords_per_world:
                raise ValueError(f"world {world}: expected {self.coords_per_world} coordinates, got {values.size}")
            stacked[world] = values
        return stacked.reshape(-1)

    def __repr__(self) -> str:
        return (
            f"WorldMap(num_worlds={self.num_worlds}, bodies={self.bodies_per_world}, "
            f"shapes={self.shapes_per_world}, coords={self.coords_per_world}, dofs={self.dofs_per_world}, "
            f"particles={self.particles_per_world})"
        )
