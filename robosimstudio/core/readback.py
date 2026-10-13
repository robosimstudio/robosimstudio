# SPDX-License-Identifier: Apache-2.0
"""One device-to-host read per control step, and a finite check that stays on the device.

Every ``.numpy()`` is a synchronize, and the single-world environment used to
pay ten to fifteen of them per control step -- the context's ``body_q``,
``body_qd`` and targets, the finite guard's three arrays, each cloth probe's
two, the contact probe's four. :class:`Readback` replaces them with one batch:
after the graph launches, it copies every array an observer might want into
pinned host buffers on the same stream and records an event; the first reader
waits on the event once, and everyone else reads the host copy. A caller that
asks twice in one step pays once.

The finite guard moves onto the device with it. A diverged solver keeps stepping
happily on NaN, at a very stable rate, so a run has to check rather than assume
-- but reading three arrays back to ask is the cost this module exists to not
pay. :func:`launch_finite_check` latches one sticky flag per world on the first
non-finite coordinate, and the flag rides along in the batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import newton
import numpy as np
import warp as wp

from .streams import reading

__all__ = ["HostSnapshot", "Readback", "launch_finite_check"]


@wp.kernel
def _flag_nonfinite_f32(values: wp.array[wp.float32], per_world: int, flags: wp.array[wp.int32]):
    """Latch ``flags[world]`` when a scalar is NaN, inf or absurdly large."""
    i = wp.tid()
    value = values[i]
    if not (value == value) or value > 1.0e18 or value < -1.0e18:
        world = i // per_world
        if world >= flags.shape[0]:
            world = flags.shape[0] - 1
        flags[world] = 1


@wp.kernel
def _flag_nonfinite_vec3(values: wp.array[wp.vec3], per_world: int, flags: wp.array[wp.int32]):
    i = wp.tid()
    v = values[i]
    bad = int(0)
    for k in range(3):
        c = v[k]
        if not (c == c) or c > 1.0e18 or c < -1.0e18:
            bad = 1
    if bad != 0:
        world = i // per_world
        if world >= flags.shape[0]:
            world = flags.shape[0] - 1
        flags[world] = 1


@wp.kernel
def _flag_nonfinite_transform(values: wp.array[wp.transform], per_world: int, flags: wp.array[wp.int32]):
    i = wp.tid()
    t = values[i]
    p = wp.transform_get_translation(t)
    bad = int(0)
    for k in range(3):
        c = p[k]
        if not (c == c) or c > 1.0e18 or c < -1.0e18:
            bad = 1
    if bad != 0:
        world = i // per_world
        if world >= flags.shape[0]:
            world = flags.shape[0] - 1
        flags[world] = 1


def launch_finite_check(
    state: newton.State,
    flags: wp.array,
    *,
    coords_per_world: int = 0,
    dofs_per_world: int = 0,
    particles_per_world: int = 0,
    bodies_per_world: int = 0,
) -> None:
    """Latch ``flags[world]`` for every world with a non-finite coordinate.

    Sticky: a world that produced a NaN once has already lost its episode, and
    a NaN can be transiently squashed by the next solve. Zero ``flags`` at reset.
    A per-world count of 0 means "one world, whatever the length".
    """
    device = flags.device

    def per(count: int, array) -> int:
        return int(count) if count > 0 else max(1, int(array.shape[0]))

    if state.joint_q is not None and state.joint_q.shape[0]:
        wp.launch(
            _flag_nonfinite_f32,
            dim=state.joint_q.shape[0],
            inputs=[state.joint_q, per(coords_per_world, state.joint_q), flags],
            device=device,
        )
    if state.joint_qd is not None and state.joint_qd.shape[0]:
        wp.launch(
            _flag_nonfinite_f32,
            dim=state.joint_qd.shape[0],
            inputs=[state.joint_qd, per(dofs_per_world, state.joint_qd), flags],
            device=device,
        )
    if state.particle_q is not None and state.particle_q.shape[0]:
        wp.launch(
            _flag_nonfinite_vec3,
            dim=state.particle_q.shape[0],
            inputs=[state.particle_q, per(particles_per_world, state.particle_q), flags],
            device=device,
        )
    if state.body_q is not None and state.body_q.shape[0]:
        wp.launch(
            _flag_nonfinite_transform,
            dim=state.body_q.shape[0],
            inputs=[state.body_q, per(bodies_per_world, state.body_q), flags],
            device=device,
        )


@dataclass
class HostSnapshot:
    """The host copy of one control step's observable state.

    The state arrays (``body_q`` through ``particle_qd``) are float64 copies
    made when the snapshot is built, so they keep their values. ``finite`` and
    ``extra`` are **views** into the read-back's pinned buffers: the next
    read-back overwrites them, so anything of theirs that must outlive a step
    copies.

    Attributes:
        step: The control step it was taken at.
        body_q: ``[bodies, 7]`` float64, or ``[0, 7]`` for a scene with no bodies.
        body_qd: ``[bodies, 6]`` float64.
        joint_q: Joint coordinates, float64.
        joint_qd: Joint velocities, float64.
        joint_target_q: Servo targets, float64 (empty for no articulation).
        particle_q: ``[particles, 3]`` float64, or ``[0, 3]``.
        particle_qd: ``[particles, 3]`` float64.
        finite: Per-world ``int32`` flags: 1 means that world produced a
            non-finite value since the flags were last zeroed.
    """

    step: int
    body_q: np.ndarray
    body_qd: np.ndarray
    joint_q: np.ndarray
    joint_qd: np.ndarray
    joint_target_q: np.ndarray
    particle_q: np.ndarray
    particle_qd: np.ndarray
    finite: np.ndarray
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def is_finite(self, world: int | None = None) -> bool:
        if self.finite.size == 0:
            return True
        if world is None:
            return not bool(self.finite.any())
        return not bool(self.finite[int(world)])


class Readback:
    """Batched device-to-host copies into pinned buffers, one event per step.

    Args:
        model: The model, for counts and the device.
        control: The control, for ``joint_target_q``.
        num_worlds: How many worlds the finite flags cover.
        coords_per_world, dofs_per_world, particles_per_world, bodies_per_world:
            Strides for the finite check; 0 means a single world.
    """

    ARRAYS = (
        ("body_q", wp.transform),
        ("body_qd", wp.spatial_vector),
        ("joint_q", wp.float32),
        ("joint_qd", wp.float32),
        ("particle_q", wp.vec3),
        ("particle_qd", wp.vec3),
    )

    def __init__(
        self,
        model: newton.Model,
        control: newton.Control,
        *,
        num_worlds: int = 1,
        coords_per_world: int = 0,
        dofs_per_world: int = 0,
        particles_per_world: int = 0,
        bodies_per_world: int = 0,
    ) -> None:
        self.model = model
        self.control = control
        self.device = model.device
        self.num_worlds = max(1, int(num_worlds))
        self._strides = {
            "coords_per_world": coords_per_world,
            "dofs_per_world": dofs_per_world,
            "particles_per_world": particles_per_world,
            "bodies_per_world": bodies_per_world,
        }
        pinned = bool(getattr(self.device, "is_cuda", False))
        self._host: dict[str, wp.array] = {}
        counts = {
            "body_q": model.body_count,
            "body_qd": model.body_count,
            "joint_q": model.joint_coord_count,
            "joint_qd": model.joint_dof_count,
            "particle_q": model.particle_count,
            "particle_qd": model.particle_count,
        }
        for name, dtype in self.ARRAYS:
            self._host[name] = wp.zeros(int(counts[name]), dtype=dtype, device="cpu", pinned=pinned)
        targets = getattr(control, "joint_target_q", None)
        self._host["joint_target_q"] = wp.zeros(
            int(targets.shape[0]) if targets is not None else 0, dtype=wp.float32, device="cpu", pinned=pinned
        )
        self.flags = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self._host_flags = wp.zeros(self.num_worlds, dtype=wp.int32, device="cpu", pinned=pinned)
        self.extra_sources: dict[str, wp.array] = {}
        self._host_extra: dict[str, wp.array] = {}
        self._event: wp.Event | None = wp.Event(self.device) if pinned else None
        self._pending = False
        self._snapshot: HostSnapshot | None = None
        self.step = -1
        self.reads = 0
        """How many device-to-host batches were issued."""
        self.syncs = 0
        """How many times a reader actually had to wait for one."""

    # ---------------------------------------------------------------- extras
    def watch(self, name: str, source: wp.array) -> None:
        """Add a device array to the batch under ``name``."""
        self.extra_sources[name] = source
        pinned = bool(getattr(self.device, "is_cuda", False))
        self._host_extra[name] = wp.zeros(source.shape, dtype=source.dtype, device="cpu", pinned=pinned)

    # ------------------------------------------------------------------ issue
    def reset_flags(self) -> None:
        self.flags.zero_()

    def enqueue(self, state: newton.State, step: int) -> None:
        """Issue this step's copies on the current stream and record the event.

        Call after the physics of ``step`` has been launched. Nothing here
        waits; the copies queue behind the graph.
        """
        launch_finite_check(state, self.flags, **self._strides)
        sources = {
            "body_q": state.body_q,
            "body_qd": state.body_qd,
            "joint_q": state.joint_q,
            "joint_qd": state.joint_qd,
            "particle_q": state.particle_q,
            "particle_qd": state.particle_qd,
            "joint_target_q": getattr(self.control, "joint_target_q", None),
        }
        for name, source in sources.items():
            host = self._host[name]
            if source is not None and host.shape[0] and source.shape[0] == host.shape[0]:
                wp.copy(host, source)
        for name, source in self.extra_sources.items():
            wp.copy(self._host_extra[name], source)
        wp.copy(self._host_flags, self.flags)
        if self._event is not None:
            # ON THIS READBACK'S OWN STREAM, not the process default's.
            # ``wp.record_event(e)`` is a convenience for
            # ``wp.get_stream().record_event(e)``, and ``wp.get_stream()`` with no
            # argument is the stream of warp's *default* device -- which is
            # cuda:0 in any process that did not set one. An event constructed
            # on cuda:1 (see ``self._event`` above) then cannot be recorded on
            # it, and warp raises
            #
            #   RuntimeError: Event from device cuda:1 cannot be recorded on
            #                 stream from device cuda:0
            #
            # at the first enqueue. Measured: every worker of a ``--device
            # cuda:1`` run died there during startup, the case reported 0/30 with
            # all thirty episodes unrun, and the second GPU sat at 0% holding
            # only the contexts warp opens on every visible device. A one-GPU
            # box never sees this, because there the default device IS the
            # readback's.
            wp.get_stream(self.device).record_event(self._event)
        self._pending = True
        self._snapshot = None
        self.step = int(step)
        self.reads += 1

    # ------------------------------------------------------------------- read
    def snapshot(self) -> HostSnapshot:
        """The latest batch, waiting for it once if it has not landed yet."""
        if self._snapshot is not None:
            return self._snapshot
        # Under ``reading()`` because both halves below reach the device from
        # whatever thread asked: a graph policy runs its workflow on its own
        # thread and reads state through the tool surface, while the owner
        # thread is inside ``capturing()``. An event synchronize *during* a
        # capture is not a race that corrupts a number, it is CUDA error 900,
        # "operation not permitted when stream is capturing" -- which surfaces
        # several frames away as the graph node that happened to ask.
        with reading():
            if self._pending:
                if self._event is not None:
                    wp.synchronize_event(self._event)
                else:
                    wp.synchronize_device(self.device)
                self.syncs += 1
                self._pending = False

            return self._build_snapshot()

    def _build_snapshot(self) -> HostSnapshot:
        """The host copies as one snapshot. Called with the stream lock held."""

        def view(name: str, width: int | None, dtype=np.float64) -> np.ndarray:
            host = self._host[name].numpy()
            if width is not None:
                host = host.reshape(-1, width) if host.size else np.zeros((0, width), dtype=host.dtype)
            return np.asarray(host, dtype=dtype)

        self._snapshot = HostSnapshot(
            step=self.step,
            body_q=view("body_q", 7),
            body_qd=view("body_qd", 6),
            joint_q=view("joint_q", None),
            joint_qd=view("joint_qd", None),
            joint_target_q=view("joint_target_q", None),
            particle_q=view("particle_q", 3),
            particle_qd=view("particle_qd", 3),
            finite=np.asarray(self._host_flags.numpy(), dtype=np.int32),
            extra={name: host.numpy() for name, host in self._host_extra.items()},
        )
        return self._snapshot

    def take(self, state: newton.State, step: int) -> HostSnapshot:
        """Issue and read in one call, for a caller outside the step loop."""
        self.enqueue(state, step)
        return self.snapshot()
