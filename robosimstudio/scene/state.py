# SPDX-License-Identifier: Apache-2.0
"""One control step of a running world, read back once.

The seam between the simulator and anything that measures it, and the reason it
is here rather than a layer up: **everything in it is a world fact.** Where the
bodies are, how fast, where they started, what was commanded, what is touching
what, what the entities published. No task, no episode, no goal, no reward -- so
a scene with none of those can produce one, and the goal machinery can consume
one without the scene having a task.

Gathered once because reading ``body_q`` back from device is the single largest
cost in scoring, and a naive reader does it once per object. This does it once
per step and hands the array around -- which is also why the environment keeps
the last one rather than only passing it on: a recorder, a studio panel or a
description reuses the read the step already paid for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import newton
import numpy as np

from robosimstudio.core.probes import ParticleProbe
from robosimstudio.scene.handles import SceneHandles

__all__ = ["WorldState"]


@dataclass
class WorldState:
    """What one control step of a world reads back, gathered once.

    Attributes:
        handles: Where everything is.
        state: The current state.
        step: Control steps since reset.
        sim_time: Simulated seconds since reset [s].
        body_q: ``State.body_q`` read back once, shape ``[body_count, 7]``.
        body_qd: ``State.body_qd`` read back once, shape ``[body_count, 6]``.
            Newton's spatial velocity is ``(linear, angular)``, linear first, and
            it is the *centre-of-mass* velocity in world coordinates. Gathered
            here for the same reason ``body_q`` is: a predicate that needs a
            velocity (Arena's placement test wants the object at rest) would
            otherwise pay a device-to-host copy per object per step.
        settled_body_q: The same array as of the episode's start, which is what
            "has it moved" is measured against.
        joint_target_q: The *commanded* joint targets, read back once. What a
            predicate comparing command against measurement needs -- a gripper
            stalled short of its target is holding something, and neither
            number alone can say so. Empty when the scene has no control
            vector.
        contacts: A :class:`~robosimstudio.core.probes.contact.ContactProbe`,
            already updated for this step, or ``None`` when the task declared no
            contact-based conditions.
        particles: Per-cloth :class:`~robosimstudio.core.probes.particle.ParticleProbe`,
            already updated.
        entities: Every built entity except the robot, by name. ``extras``
            remains as an alias.
        world: Which replicated world this is a reading of. ``0`` for a
            single-world scene, where it is also the only world. An entity
            that keeps per-world state (a latch, a windowed rate) reads this to
            answer for the right copy; ``body_q`` and the rest are already that
            world's rows.

            Here rather than only in the fleet, because the thing that needs it
            is on the far side of a duck type: an entity's
            ``published_states`` is handed this object and nothing else, and its
            own device arrays carry a world axis once the entity is world aware.
            Without a world on the context, a wave would score every world off
            row 0 -- the same verdict sixteen times, and the right one only by
            accident.
        active: Which entities this episode runs, or ``None`` for every one --
            :attr:`~robosimstudio.specs.Episode.active`, restated here so a
            measure can ask without reaching for the episode. When set,
            :attr:`handles` is already the restricted table
            (:meth:`SceneHandles.restricted`) and :attr:`contacts` carries no
            row of a parked body: a parked body is *in* ``body_q`` -- it is in
            the model -- but nothing that enumerates by name will find it.
    """

    handles: SceneHandles
    state: newton.State
    step: int = 0
    sim_time: float = 0.0
    body_q: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))
    body_qd: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    settled_body_q: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))
    joint_target_q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    contacts: Any | None = None
    particles: dict[str, ParticleProbe] = field(default_factory=dict)
    entities: dict[str, Any] = field(default_factory=dict)
    world: int = 0
    active: frozenset[str] | None = None

    @property
    def extras(self) -> dict[str, Any]:
        """Former name of :attr:`entities`. Kept for one release."""
        return self.entities

    def position(self, name: str) -> np.ndarray:
        """One entity's world position [m]."""
        return self.handles.entity(name).position(self.body_q)

    def pose(self, name: str) -> np.ndarray:
        """One entity's world pose as ``(px, py, pz, qx, qy, qz, qw)``."""
        return self.handles.entity(name).pose(self.body_q)

    def settled_position(self, name: str) -> np.ndarray:
        """Where an entity was when the episode started [m]."""
        return self.handles.entity(name).position(self.settled_body_q)

    def linear_speed(self, name: str) -> float:
        """One entity's linear speed [m/s].

        Newton's spatial velocity puts the *linear* part first (``State.body_qd``:
        "first three entries: linear velocity"), which is the opposite of the
        warp.sim convention this stack's ancestry came from -- read the wrong
        half and a spinning-but-stationary object looks like it is moving.
        """
        body = self.handles.entity(name).body
        if not len(self.body_qd) or body < 0:
            return 0.0
        return float(np.linalg.norm(self.body_qd[body][:3]))

    def probe(self, cloth: str) -> ParticleProbe:
        if cloth not in self.particles:
            raise KeyError(f"no particle probe for {cloth!r}; have {sorted(self.particles)}")
        return self.particles[cloth]
