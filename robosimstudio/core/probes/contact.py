# SPDX-License-Identifier: Apache-2.0
"""Pairwise contact impulses, read out of the MuJoCo-Warp solver.

Success tests transcribed from SAPIEN-based benchmarks walk
``scene.get_contacts()`` and sum per-point impulses between two actors. Newton
has no such list -- contacts live inside the MuJoCo-Warp solver's ``Data`` as
geom pairs plus constraint forces -- so :class:`ContactProbe` pulls those out
and re-forms them into the pairwise impulses those tests expect, mapping MuJoCo
geoms back to Newton shapes through the solver's own
``mjc_geom_to_newton_shape`` table.

The quantity is the same: force x timestep, in world coordinates, between two
sets of shapes. What the probe adds over the raw arrays is the *sign*
convention, which is the part that is easy to get backwards and impossible to
notice: MuJoCo applies its constraint force to the pair's **second** geom along
a normal pointing from the first to the second.
"""

from __future__ import annotations

from collections.abc import Sequence

import newton
import numpy as np
import warp as wp

from ..math import compute_angle_between, quat_to_mat
from ..streams import reading

__all__ = ["MIN_IMPULSE", "ContactProbe", "ContactQueries", "FleetContactProbe"]

MIN_IMPULSE = 1.0e-6
"""Impulse threshold [N·s] that counts as "touching" -- effectively any contact."""


class ContactQueries:
    """Reading a set of contacts as pairwise quantities between shape sets.

    Split out from :class:`ContactProbe` so that one world of a replicated
    scene can be asked the same questions, over its own rows, without a second
    implementation of the sign convention -- which is the part of this that is
    easy to get backwards and impossible to notice.

    A subclass supplies five things and inherits every query: ``geoms``
    (contacting geom pairs), ``impulses`` (impulse on each pair's second geom),
    ``depths`` (how far each pair overlaps), ``geom_to_shape`` (that world's
    table), and ``dt``.
    """

    geoms: np.ndarray
    impulses: np.ndarray
    depths: np.ndarray
    geom_to_shape: np.ndarray
    dt: float

    # ------------------------------------------------------------------ queries
    def _shape_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        return self.geom_to_shape[self.geoms[:, 0]], self.geom_to_shape[self.geoms[:, 1]]

    def penetrations(self, deeper_than: float = 0.0) -> np.ndarray:
        """Overlapping pairs, worst first: ``[[shape_a, shape_b, depth], ...]``.

        **Depth is positive metres of overlap**, which is the sign MuJoCo does
        *not* use: its ``contact.dist`` is a signed gap, negative when two geoms
        interpenetrate and positive for a pair merely inside the margin. Flipped
        here rather than at every call site because "penetration 0.3" reading as
        three tenths of a millimetre *apart* is exactly the kind of inversion
        this class exists to have in one place.

        What it is for: a scene that looks settled and is not. A resting contact
        under a soft model always overlaps a little -- that overlap is how the
        force is generated -- so the number is never zero and the question is
        always how big. On a correctly stiff contact these run to hundredths of
        a millimetre; the astro suite spent a week reading millimetres before
        anyone looked, because nothing surfaced the figure.

        Args:
            deeper_than: Ignore overlaps at or under this [m]. Zero keeps every
                touching pair and drops the near-misses the margin reports.

        Returns:
            ``float64[n, 3]``, sorted deepest first. Shape indices are Newton's.
        """
        if not len(self.geoms):
            return np.zeros((0, 3))
        depth = -np.asarray(self.depths, dtype=np.float64)
        rows = depth > float(deeper_than)
        if not rows.any():
            return np.zeros((0, 3))
        a, b = self._shape_pairs()
        out = np.column_stack([a[rows], b[rows], depth[rows]])
        return out[np.argsort(-out[:, 2])]

    def pairwise_impulse(self, shapes_a: Sequence[int], shapes_b: Sequence[int]) -> np.ndarray:
        """Total impulse applied to ``shapes_a`` by ``shapes_b`` [N·s].

        The sign follows the first argument, so a positive result is what pushes
        ``shapes_a`` away. The stored impulse acts on each pair's second geom,
        so a contact whose second geom is in ``shapes_a`` contributes with a
        plus and one whose first geom is contributes with a minus.
        """
        if not len(self.geoms):
            return np.zeros(3)
        a, b = self._shape_pairs()
        sa, sb = set(shapes_a), set(shapes_b)
        total = np.zeros(3)
        for k in range(len(a)):
            if a[k] in sb and b[k] in sa:
                total += self.impulses[k]
            elif a[k] in sa and b[k] in sb:
                total -= self.impulses[k]
        return total

    def pairwise_force(self, shapes_a: Sequence[int], shapes_b: Sequence[int]) -> np.ndarray:
        """Total force applied to ``shapes_a`` by ``shapes_b`` [N].

        The same quantity as :meth:`pairwise_impulse` divided by the substep, and
        it exists because published thresholds are stated in newtons: Arena's
        placement test fires above 0.1 N (1.0 N for its multi-object variant), and
        dividing at each call site is how one of them ends up in the wrong unit.
        """
        return self.pairwise_impulse(shapes_a, shapes_b) / self.dt

    def touching(self, shapes: Sequence[int], exclude_shapes: Sequence[int] = ()) -> bool:
        """Is anything outside ``exclude_shapes`` pushing on ``shapes``?

        The "completely on target" test: an object placed on a target must not
        also be resting against anything but that target and the robot.
        """
        if not len(self.geoms):
            return False
        a, b = self._shape_pairs()
        s, ex = set(shapes), set(exclude_shapes)
        mag = np.linalg.norm(self.impulses, axis=1)
        for k in range(len(a)):
            if a[k] in s:
                other = b[k]
            elif b[k] in s:
                other = a[k]
            else:
                continue
            if other in s or other in ex:
                continue
            if mag[k] > MIN_IMPULSE:
                return True
        return False

    def in_contact(self, shapes_a: Sequence[int], shapes_b: Sequence[int]) -> bool:
        """Whether any contact above threshold joins the two shape sets."""
        return bool(np.linalg.norm(self.pairwise_impulse(shapes_a, shapes_b)) >= MIN_IMPULSE)

    def check_grasp(
        self,
        finger_groups: Sequence[Sequence[tuple[int, Sequence[int]]]],
        object_shapes: Sequence[int],
        body_q: np.ndarray,
        *,
        open_axis_signs: Sequence[float] = (1.0, -1.0),
        max_angle: float = 85.0,
    ) -> bool:
        """Whether both sides of a hand are being pushed outwards by an object.

        A grasp is a contact impulse above threshold on each side whose
        direction is within ``max_angle`` of the direction that side opens in.
        The angle test is what stops "the object is resting on top of the
        closed gripper" from reading as a grasp.

        Args:
            finger_groups: One group per side; each group is a list of
                ``(body, shapes)`` pairs. A robot whose fingertip is a separate
                body lists both, and the side passes on the *larger* impulse
                and the *smaller* angle of the pair.
            object_shapes: The object's shapes.
            body_q: The step's body transforms, ``[body_count, 7]``.
            open_axis_signs: Per side, which way along the finger body's local
                +y the side opens. Different robots differ here: a symmetric
                hand tests ``(+1, -1)``, one whose finger frames are already
                mirrored tests ``(+1, +1)``.
            max_angle: Angle ceiling [deg].
        """
        flags = []
        for side, group in enumerate(finger_groups):
            sign = float(open_axis_signs[side])
            best_impulse, best_angle = 0.0, 180.0
            for body, shapes in group:
                impulse = self.pairwise_impulse(shapes, object_shapes)
                opens = sign * quat_to_mat(body_q[body][3:7])[:, 1]
                best_impulse = max(best_impulse, float(np.linalg.norm(impulse)))
                best_angle = min(best_angle, float(np.rad2deg(compute_angle_between(opens, impulse))))
            flags.append(best_impulse >= MIN_IMPULSE and best_angle <= max_angle)
        return bool(flags) and all(flags)


class ContactProbe(ContactQueries):
    """This step's contacts, keyed by Newton shape index.

    One instance per built scene. :meth:`update` copies the frame's contacts to
    the host, which is the only device-to-host traffic an evaluation costs;
    every query after it is numpy over a few dozen rows.

    Args:
        solver: The :class:`newton.solvers.SolverMuJoCo` stepping the scene. It
            must own its collision pipeline (``use_mujoco_contacts=True``);
            with contacts handed in from outside there is no geom table to map
            through.
        model: The finalized model.
        dt: The substep timestep [s], which converts constraint force to
            impulse.
        world: Which world's contacts to read. Scenes are single-world today
            and the table is per-world only because Newton supports replication.
    """

    def __init__(self, solver, model: newton.Model, dt: float, *, world: int = 0) -> None:
        import mujoco_warp as mjw

        self._mjw = mjw
        self.solver = solver
        self.model = model
        self.dt = float(dt)
        self.world = int(world)

        if getattr(solver, "mjc_geom_to_newton_shape", None) is None:
            raise RuntimeError(
                "SolverMuJoCo has no geom->shape table; contact-based success tests need the "
                "MuJoCo backend running its own collision pipeline (use_mujoco_contacts=True)."
            )
        self.geom_to_shape = solver.mjc_geom_to_newton_shape.numpy()[self.world]
        n = int(solver.mjw_data.contact.geom.shape[0])
        self._ids = wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device=model.device)
        self._force = wp.zeros(n, dtype=wp.spatial_vector, device=model.device)
        self.shape_body = model.shape_body.numpy()

        self.geoms = np.zeros((0, 2), dtype=np.int32)
        """This step's contacting MuJoCo geom pairs, shape ``[n, 2]``."""
        self.impulses = np.zeros((0, 3))
        """Impulse on each pair's second geom [N·s], shape ``[n, 3]``."""
        self.depths = np.zeros(0)
        """Signed gap per pair [m], MuJoCo's own sign: negative is overlapping.

        Kept as the solver reports it and flipped in :meth:`penetrations`, so a
        reader comparing this against ``mjw_data.contact.dist`` sees the same
        numbers.
        """
        self.step: int | None = None
        """Control step :attr:`geoms` was read at, or ``None`` if never."""

    def update(self, step: int | None = None) -> None:
        """Read this step's contacts. Call once per control step, after physics.

        Args:
            step: The control step these contacts belong to. Passing it makes
                the call idempotent *within* a step, which is what lets two
                callers ask for one frame's contacts without reading it twice:
                in teleop the studio's episode panel refreshes the probe every
                step and the arm panel's hold test asks again on the same
                frame, and the second read is three device-to-host copies for
                an answer already sitting on the host. ``None`` always reads,
                for a caller that knows physics has moved under it.
        """
        if step is not None and step == self.step:
            return
        self.step = step
        data = self.solver.mjw_data
        # Scoped, because mujoco-warp's helper launches on the *default* device
        # rather than the arrays'. On a scene built anywhere but ``cuda:0`` that
        # reads this world's contact buffers from the wrong context, which is an
        # illegal access that poisons the process rather than a wrong answer.
        # Active slots first, forces second — and only over the active slots.
        # The old order asked contact_force for every slot up to capacity;
        # slots past nacon hold stale efc addresses after heavy contact churn
        # (a tearing film), and dereferencing them is the CUDA 700 that
        # poisoned whole peel batches (bisect: contacts-on 1018 fault lines,
        # contacts-off zero, particle readback innocent).
        n = int(data.nacon.numpy()[0])
        if n == 0:
            self.geoms = np.zeros((0, 2), dtype=np.int32)
            self.impulses = np.zeros((0, 3))
            self.depths = np.zeros(0)
            return
        n = min(n, len(self._ids))
        with wp.ScopedDevice(self.model.device):
            self._mjw.contact_force(self.solver.mjw_model, data, self._ids[:n], True, self._force[:n])
        self.impulses = np.asarray(self._force.numpy()[:n, :3], dtype=np.float64) * self.dt
        self.geoms = np.asarray(data.contact.geom.numpy()[:n], dtype=np.int32)
        # A fourth copy of the same n rows. Cheap next to the three above, and
        # the one number that says whether a settled-looking scene is settled.
        self.depths = np.asarray(data.contact.dist.numpy()[:n], dtype=np.float64)


class FleetContactProbe:
    """Every world's contacts, from one read, asked per world.

    The replicated twin of :class:`ContactProbe`. MuJoCo-Warp packs every
    world's contacts into one array and tags each row with its ``worldid``, so
    the batched form is the same three device-to-host copies the single-world
    probe already costs -- once for the whole fleet rather than once per world.
    At five hundred worlds that difference is the difference between contact
    queries being free and being the run.

    :meth:`view` hands back one world's rows behind the same query surface, with
    its shape indices **rebased to the template**. That is what lets a success
    predicate written against a single-world model -- and holding template shape
    indices from its handles -- ask about world 300 without knowing it exists.

    Args:
        solver: The :class:`newton.solvers.SolverMuJoCo` stepping the fleet.
        model: The replicated model.
        dt: The substep timestep [s].
        worlds: The stride map, for rebasing shape indices.
    """

    def __init__(self, solver, model: newton.Model, dt: float, *, worlds) -> None:
        import mujoco_warp as mjw

        self._mjw = mjw
        self.solver = solver
        self.model = model
        self.dt = float(dt)
        self.worlds = worlds

        if getattr(solver, "mjc_geom_to_newton_shape", None) is None:
            raise RuntimeError(
                "SolverMuJoCo has no geom->shape table; contact-based success tests need the "
                "MuJoCo backend running its own collision pipeline (use_mujoco_contacts=True)."
            )
        table = solver.mjc_geom_to_newton_shape.numpy()
        # One table per world, rebased so a shape index means what it means on
        # the template. Done once here rather than per query, since it is the
        # same arithmetic every time and there are a lot of queries.
        stride = int(worlds.shapes_per_world)
        self.geom_to_shape = np.stack(
            [np.where(table[w] >= 0, table[w] - w * stride, -1) for w in range(int(worlds.num_worlds))]
        )
        n = int(solver.mjw_data.contact.geom.shape[0])
        self._ids = wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device=model.device)
        self._force = wp.zeros(n, dtype=wp.spatial_vector, device=model.device)

        self.geoms = np.zeros((0, 2), dtype=np.int32)
        self.impulses = np.zeros((0, 3))
        self.depths = np.zeros(0)
        self.worldids = np.zeros(0, dtype=np.int32)
        self.step: int | None = None
        self._views: dict[int, ContactView] = {}

    def update_all(self, step: int | None = None) -> None:
        """Read every world's contacts. Once per cadence tick, after physics.

        Idempotent within a step, for the same reason the single-world probe is:
        two callers wanting one frame's contacts should not read it twice.
        """
        if step is not None and step == self.step:
            return
        self.step = step
        self._views.clear()
        data = self.solver.mjw_data
        # Every read and the launch between them under the stream lock: a
        # world's graph thread reaches this through WorldView.make_context, and
        # an unlocked .numpy() or a launch on the current stream can land
        # inside another thread's capture. See :mod:`robosimstudio.core.streams`.
        with reading():
            # Active slots only — same stale-slot dereference the single-world
            # probe fixed; see ContactProbe.update.
            n = min(int(np.max(data.nacon.numpy())), len(self._ids))
            if n == 0:
                self.geoms = np.zeros((0, 2), dtype=np.int32)
                self.impulses = np.zeros((0, 3))
                self.depths = np.zeros(0)
                self.worldids = np.zeros(0, dtype=np.int32)
                return
            with wp.ScopedDevice(self.model.device):  # see ContactProbe.update
                self._mjw.contact_force(self.solver.mjw_model, data, self._ids[:n], True, self._force[:n])
            self.impulses = np.asarray(self._force.numpy()[:n, :3], dtype=np.float64) * self.dt
            self.geoms = np.asarray(data.contact.geom.numpy()[:n], dtype=np.int32)
            self.depths = np.asarray(data.contact.dist.numpy()[:n], dtype=np.float64)
            self.worldids = np.asarray(data.contact.worldid.numpy()[:n], dtype=np.int32)

    def view(self, world: int) -> ContactView:
        """One world's contacts, indexed as a single-world scene's would be."""
        index = int(world)
        cached = self._views.get(index)
        if cached is not None:
            return cached
        rows = self.worldids == index
        view = ContactView(
            geoms=self.geoms[rows],
            impulses=self.impulses[rows],
            depths=self.depths[rows],
            geom_to_shape=self.geom_to_shape[index],
            dt=self.dt,
        )
        self._views[index] = view
        return view


class ContactView(ContactQueries):
    """One world's rows of a fleet read, behind the single-world query surface."""

    def __init__(
        self,
        *,
        geoms: np.ndarray,
        impulses: np.ndarray,
        depths: np.ndarray,
        geom_to_shape: np.ndarray,
        dt: float,
    ) -> None:
        self.geoms = geoms
        self.impulses = impulses
        self.depths = depths
        self.geom_to_shape = geom_to_shape
        self.dt = float(dt)
