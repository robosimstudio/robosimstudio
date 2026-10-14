# SPDX-License-Identifier: Apache-2.0
"""Parking: the objects an episode does not run, floated out of the way per world.

A superset model holds every object any of its instances could need -- the
tier's whole tool catalogue, eight syringes, three gears per class -- and one
episode runs a subset (:attr:`~robosimstudio.specs.Episode.active`). MuJoCo-Warp
runs ``W`` worlds as ``W`` copies of **one** model, so a body cannot be *removed*
from one world; it can only be put somewhere and made inert there. This module
is the one way that is done, for the fleet and for the serial environment
alike, so a superset base run serially behaves as it does in a wave.

**Float far away, hold against gravity, touch nothing.** A parked body is
written to its park pose (:func:`~robosimstudio.specs.park_poses`: six metres
up, on a lattice, outside every fixed camera frustum) with zero velocity, and
its ``body_gravcomp`` row in *that world* is set to 1 so it stays there. It then
contributes **no contact and no constraint row** to that world's solve -- it
cannot enter the other bodies' arithmetic at all -- and adds one body's forward
kinematics and a broadphase entry. Measured on a tool-hanging scene: 0.000 mm of
drift on the parked body over 300 control steps, and the sharing bodies within
the sub-micrometre floor two *identical* worlds differ by.

**Where the per-world gravcomp lives, and why two arrays.** MuJoCo-Warp tiles
``body_gravcomp`` per world (``mjw_model.body_gravcomp[world, mjc_body]``) and
the passive-force kernel reads that row; so does Newton's own body-force
kernel. That array is what a step reads, captured graph or not, so writing it
takes effect on the next step with no recapture. But Newton *regenerates* it:
``notify_model_changed(BODY_INERTIAL_PROPERTIES)`` -- a mass tunable, a
``set_const`` -- relaunches ``update_body_mass_ipos_kernel``, which copies
``model.mujoco.gravcomp[newton_body]`` (the per-body custom attribute on the
**replicated** Newton model, so already per world) over every row. A row
written to the tiled array alone would survive every step and vanish on the
first notify. So a park writes both, from one place, and a notify reproduces
exactly what was written. A model whose arrangement registered no ``gravcomp``
attribute has only the tiled array, and the regenerating kernel skips the
copy for it -- so that path survives a notify too, by omission.

The other mechanisms the study weighed -- a per-world weld, a negative contact
margin, sleeping, a shelf to rest on -- either cost constraint rows in the
parking world or have no per-world channel at all; see the study.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from robosimstudio.specs import park_lattice, park_poses

__all__ = ["GravcompParking", "active_objects", "parked_objects", "restrict_contacts", "scene_park_poses"]


def active_objects(scene: Any, active: Iterable[str] | None) -> frozenset[str] | None:
    """The object names an episode runs, or ``None`` for every one.

    Names the scene has no object for are dropped rather than refused here: a
    document's active set may name a static or a deformable (the sampler
    contract is "entities", and the benchmark tier's episode sampler
    has already checked every name against the task), and only *objects* --
    free bodies with a joint to write -- are what parking is about.
    """
    if active is None:
        return None
    return frozenset(name for name in active if name in scene.objects)


def parked_objects(scene: Any, active: Iterable[str] | None) -> list[str]:
    """Which of a scene's objects an episode parks: every free body not in ``active``, in build order.

    **The active set governs free bodies and nothing else.** An object with no
    free joint -- a fixture lowered as a static handle, a bin the arena
    declared -- is welded where it stands in every world and is shared, not
    parked, whether or not the set names it; so is every static, deformable
    and suite-owned extra. That is what lets a sampler name the tools it
    draws and nothing more, and it is the same rule
    :meth:`~robosimstudio.scene.handles.SceneHandles.restricted` filters by.
    """
    keep = active_objects(scene, active)
    if keep is None:
        return []
    return [
        name
        for name, entity in scene.objects.items()
        if name not in keep and int(getattr(entity, "coord_start", -1)) >= 0
    ]


def scene_park_poses(scene: Any) -> dict[str, np.ndarray]:
    """Every object's park pose, as a 7-vector in world coordinates.

    :func:`~robosimstudio.specs.park_poses` over the scene's request when it is
    a task, else the lattice over the scene's own object order -- a
    hand-assembled scene has no ``parking`` table and no ``objects`` view on
    its request, and still has objects an episode may park.
    """
    request = scene.request
    objects = getattr(request, "objects", None)
    if objects is not None and hasattr(request, "parking"):
        table = park_poses(request)
    else:
        table = {name: park_lattice(index) for index, name in enumerate(scene.objects)}
    return {name: np.asarray(xform, dtype=np.float64).reshape(-1)[:7] for name, xform in table.items()}


class GravcompParking:
    """Per-world gravity compensation on parked bodies, written where the solver reads it.

    One instance per compiled scene. :meth:`set` records which template bodies
    a world parks and :meth:`apply` writes the whole table -- the authored
    ``gravcomp`` everywhere, 1.0 on every parked row -- into both arrays the
    module docstring names, in two assigns. Idempotent: applying the same
    table twice is one write's worth of state.

    Args:
        solver: The :class:`newton.solvers.SolverMuJoCo` stepping the scene.
        model: The model it steps -- the replicated one for a fleet.
        worlds: The stride map for a replicated model, or ``None`` for one world.

    Raises:
        RuntimeError: When the solver's ``body_gravcomp`` is not tiled per
            world. Every MuJoCo-Warp build in this venv tiles it; a solver that
            did not would make a park in one world a park in every world, which
            is a different scene and not a degraded one.
    """

    def __init__(self, solver: Any, model: Any, worlds: Any = None) -> None:
        self.solver = solver
        self.model = model
        self.worlds = worlds
        table = solver.mjc_body_to_newton.numpy()
        self.num_worlds = int(table.shape[0])
        self._mjc_of: dict[int, tuple[int, int]] = {}
        for world in range(table.shape[0]):
            for mjc, newton_body in enumerate(table[world]):
                if newton_body >= 0:
                    self._mjc_of[int(newton_body)] = (world, int(mjc))
        gravcomp = solver.mjw_model.body_gravcomp
        self._base = np.asarray(gravcomp.numpy(), dtype=np.float32).copy()
        if self._base.ndim != 2 or self._base.shape[0] != self.num_worlds:
            raise RuntimeError(
                f"body_gravcomp has shape {self._base.shape}; expected ({self.num_worlds}, nbody) tiled per world, "
                "which is what lets one world park a body the others run"
            )
        attribute = getattr(getattr(model, "mujoco", None), "gravcomp", None)
        self._attribute = attribute
        self._attribute_base = np.asarray(attribute.numpy(), dtype=np.float32).copy() if attribute is not None else None
        self._parked: dict[int, set[int]] = {}

    # ---------------------------------------------------------------- table
    def _newton_body(self, body: int, world: int) -> int:
        return int(self.worlds.body(int(body), int(world))) if self.worlds is not None else int(body)

    def set(self, world: int, bodies: Iterable[int]) -> None:
        """Record the template bodies ``world`` parks, replacing its previous set."""
        self._parked[int(world)] = {int(b) for b in bodies}

    def add(self, world: int, body: int) -> None:
        self._parked.setdefault(int(world), set()).add(int(body))

    def discard(self, world: int, body: int) -> None:
        self._parked.get(int(world), set()).discard(int(body))

    def clear(self) -> None:
        self._parked.clear()

    def parked(self, world: int) -> frozenset[int]:
        return frozenset(self._parked.get(int(world), ()))

    def apply(self) -> None:
        """Write the table: authored values everywhere, 1.0 on every parked row, both arrays."""
        tiled = self._base.copy()
        flat = self._attribute_base.copy() if self._attribute_base is not None else None
        for world, bodies in self._parked.items():
            for body in bodies:
                newton_body = self._newton_body(body, world)
                where = self._mjc_of.get(newton_body)
                if where is None:
                    raise KeyError(f"template body {body} of world {world} has no MuJoCo body to hold")
                tiled[where] = 1.0
                if flat is not None:
                    flat[newton_body] = 1.0
        self.solver.mjw_model.body_gravcomp.assign(tiled)
        if self._attribute is not None and flat is not None:
            self._attribute.assign(flat)

    # ---------------------------------------------------------------- reads
    def row(self, world: int, body: int) -> float:
        """What the solver holds for one template body in one world, read back from the device."""
        where = self._mjc_of[self._newton_body(body, world)]
        return float(self.solver.mjw_model.body_gravcomp.numpy()[where])

    def attribute_row(self, world: int, body: int) -> float | None:
        """The Newton-side value a notify would regenerate the row from, or ``None`` without the attribute."""
        if self._attribute is None:
            return None
        return float(self._attribute.numpy()[self._newton_body(body, world)])


def restrict_contacts(contacts: Any, drop_shapes: Iterable[int]) -> Any:
    """A contact view with every row touching ``drop_shapes`` removed.

    Template shape indices, as a handle carries them. A parked body has no
    contacts by construction; this is what makes that a *promise* to whatever
    reads the view rather than a measurement it has to trust -- a grasp check
    or a placement test written against the restricted handles never sees a
    parked shape's row even on the step the park is written.

    Returns the view itself when there is nothing to drop, so the ordinary
    path costs nothing.
    """
    shapes = {int(s) for s in drop_shapes}
    if contacts is None or not shapes:
        return contacts
    geoms = getattr(contacts, "geoms", None)
    if geoms is None or not len(geoms):
        return contacts
    from robosimstudio.core.probes.contact import ContactView

    table = contacts.geom_to_shape
    first, second = table[geoms[:, 0]], table[geoms[:, 1]]
    wanted = np.fromiter(shapes, dtype=np.int64)
    drop = np.isin(first, wanted) | np.isin(second, wanted)
    if not drop.any():
        return contacts
    keep = ~drop
    return ContactView(
        geoms=geoms[keep],
        impulses=contacts.impulses[keep],
        depths=contacts.depths[keep],
        geom_to_shape=table,
        dt=float(contacts.dt),
    )
