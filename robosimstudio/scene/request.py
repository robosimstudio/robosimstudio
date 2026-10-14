# SPDX-License-Identifier: Apache-2.0
"""What the compiler asks of whatever it is building -- ten things, as a protocol.

:func:`~robosimstudio.scene.compiler.compile_scene` used to take a
:class:`~robosimstudio.specs.TaskSpec`, and it reads **ten** of that type's
twenty-six fields. The other sixteen are the benchmark: an evaluator, an
instruction, episode counts, shipped graphs, a scripted routine, what counts as
success. None of them reaches the compiler, and naming the whole type anyway is
what tied *building a scene* to *having a benchmark* -- so a rope on a bench
needed a task document, a suite registration and a goal nobody was going to
score.

So the contract is stated instead, and it is stated as a **protocol** for
exactly the reason :class:`~robosimstudio.core.env.ActionSpace` is one: the four
methods ``EnvBase`` actually calls are the whole contract, and naming the robot
library's class to get them would have made ``core`` depend on ``robots``. Here
the ten attributes the compiler actually reads are the whole contract, and
naming ``TaskSpec`` to get them makes a scene depend on a benchmark.

**``TaskSpec`` satisfies this structurally and is not edited.** It grew these
ten fields for its own reasons and keeps them; nothing about a task changes, no
task loader changes, and ``tests/test_scene_request.py`` asserts the conformance
so the day the two drift apart is the day it fails rather than the day somebody
compiles a suite.

What a caller who is *not* a benchmark hands in is a :class:`BuildRequest`,
which is the same ten fields with none of the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from robosimstudio.specs import (
    GENERIC,
    IDENTITY,
    ActionSpec,
    ActuationSpec,
    ArrangementSpec,
    Binding,
    CameraSpec,
    EmbodimentSpec,
    RegionSpec,
    RenderSpec,
    Xform,
)
from robosimstudio.specs.asset import BUILD_STAGES, build_stage

__all__ = ["BuildRequest", "SceneRequest"]


@runtime_checkable
class SceneRequest(Protocol):
    """The ten things :func:`compile_scene` reads off what it is building.

    Attributes:
        physics: The arrangement -- rates, a backend per domain, the coupler and
            the collision policy.
        embodiment: The robot -- a registry name, or an
            :class:`~robosimstudio.specs.EmbodimentSpec` carried inline -- or
            ``""`` for a scene with none. A robot-free scene is a legitimate
            scene: a material rig, a settle bench, a rope on a table. An inline
            spec is what lets a robot be loaded from a file without being
            registered first, which is pyroki's ``Robot.from_urdf`` shape.
        embodiment_pose: Where the robot's base goes, when neither the binding
            nor the embodiment states an offset to derive it from.
        arm_actuation: Servo gains overriding the embodiment's own, or ``None``.
        gripper_actuation: The same for the fingers.
        regions: Named axis-aligned volumes, each in some frame. Geometry here:
            the compiler resolves each one's origin and builds a handle for it,
            and who *reads* that handle -- a goal, a studio overlay, nobody -- is
            not this contract's business.
    """

    physics: ArrangementSpec
    embodiment: str | EmbodimentSpec
    embodiment_pose: Xform
    arm_actuation: ActuationSpec | None
    gripper_actuation: ActuationSpec | None
    regions: tuple[RegionSpec, ...]

    @property
    def full_name(self) -> str:
        """What to call this in an error message or a build line."""

    @property
    def statics(self) -> tuple[Any, ...]:
        """The scenery, as a view over whatever holds the entity specs."""

    def build_order(self) -> tuple[Any, ...]:
        """Every entity spec, in the order the compiler builds them."""

    def binding_for(self, scene: str = "", embodiment: str = "", gripper: str = "") -> Binding:
        """The pairing to build: which cell, which robot, which hand, mounted where."""


@dataclass(frozen=True, kw_only=True)
class BuildRequest:
    """A scene to build, stated directly. The non-benchmark :class:`SceneRequest`.

    Frozen and hashable for the same reason every spec in this tree is: a
    compiler may cache on one.

    The one field that is not simply a ``TaskSpec`` field under another name is
    :attr:`cell`. A task carries its cell on a :class:`~robosimstudio.specs.Binding`
    because *which* cell is a property of the pairing being measured, and a task
    declares several. A request has one, so it says so, and :meth:`binding_for`
    is where the two spellings meet.

    Attributes:
        cell: Registry name of the place, or ``""`` for bare ground.
        entities: Everything except the robot, in one list, each spec naming its
            own :data:`~robosimstudio.specs.asset.BUILD_STAGES` stage.
        cameras: The camera rig. Not read by the compiler -- it is read by
            whatever builds an observation over the compiled scene -- but it
            belongs to the request rather than to the runner, because a rig is a
            property of the scene it looks at.
        binding: An explicit pairing, for a caller that has one. ``None``
            synthesises it from :attr:`cell` and :attr:`embodiment`, which is
            what a scene assembled by hand wants.
        name: What to call this scene.
    """

    physics: ArrangementSpec = GENERIC
    cell: str = ""
    entities: tuple[Any, ...] = ()
    regions: tuple[RegionSpec, ...] = ()
    cameras: tuple[CameraSpec, ...] = ()
    embodiment: str | EmbodimentSpec = ""
    embodiment_pose: Xform = IDENTITY
    arm_actuation: ActuationSpec | None = None
    gripper_actuation: ActuationSpec | None = None
    render: RenderSpec | None = None
    action: ActionSpec = field(default_factory=ActionSpec)
    """How an action is interpreted -- scales, conventions, gripper range.

    Read by :class:`~robosimstudio.sim.Sim` when it builds control modes, not by
    the compiler, so it is absent from :class:`SceneRequest`: that protocol is
    exactly what ``compile_scene`` asks for and widening it to cover a *reader*
    of the request would make the smallest thing that can be compiled larger for
    no reason. ``ActionSpec``'s own defaults are, in its words, "this package's
    own rather than any benchmark's", which is what makes an unstated one right.
    """
    binding: Binding | None = None
    name: str = "scene"

    # ------------------------------------------------------------- the protocol
    @property
    def full_name(self) -> str:
        return self.name

    @property
    def statics(self) -> tuple[Any, ...]:
        return self.at_stage("scenery")

    @property
    def embodiment_name(self) -> str:
        """The robot's registry-shaped name whether it arrived as one or as a spec.

        A :class:`~robosimstudio.specs.Binding` names a robot by string, so the
        pairing this request synthesises has to be given the *name* of an inline
        spec rather than the spec. ``""`` for a scene with no robot.
        """
        return self.embodiment.name if isinstance(self.embodiment, EmbodimentSpec) else self.embodiment

    def at_stage(self, stage: str) -> tuple[Any, ...]:
        """Everything built at one stage, in declaration order."""
        return tuple(spec for spec in self.entities if build_stage(spec) == stage)

    @property
    def objects(self) -> tuple[Any, ...]:
        """Manipulable rigid bodies."""
        return self.at_stage("objects")

    @property
    def deformables(self) -> tuple[Any, ...]:
        """Deformables."""
        return self.at_stage("deformables")

    @property
    def extras(self) -> tuple[Any, ...]:
        """Entities the generic kinds cannot express.

        The four views exist here for the same reason they exist on a task: a
        reader legitimately wants one sort, and a view over one list cannot
        drift from it the way a second tuple can.
        """
        return self.at_stage("extras")

    def build_order(self) -> tuple[Any, ...]:
        """:attr:`entities`, in the order the compiler builds them."""
        return tuple(spec for stage in BUILD_STAGES for spec in self.at_stage(stage))

    def binding_for(self, scene: str = "", embodiment: str = "", gripper: str = "") -> Binding:
        """The pairing to build.

        A request has exactly one, so unlike a task's, this never refuses: the
        arguments exist to satisfy the protocol and to let a caller ask for the
        one pairing by name. Asking for a *different* one is a mistake worth
        naming rather than silently building the only one there is.
        """
        have = self.binding or Binding()
        # A binding passed for its gripper or collisions leaves the pairing to
        # the request: an empty field is the request's cell and robot, not none.
        if not have.scene or not have.embodiment:
            have = replace(have, scene=have.scene or self.cell, embodiment=have.embodiment or self.embodiment_name)
        want = (scene or have.scene, embodiment or have.embodiment, gripper or have.gripper)
        if want != (have.scene, have.embodiment, have.gripper):
            raise KeyError(
                f"scene {self.name!r} is one pairing -- cell={have.scene or '-'} "
                f"embodiment={have.embodiment or '-'} gripper={have.gripper or '-'} -- and was asked for "
                f"cell={want[0] or '-'} embodiment={want[1] or '-'} gripper={want[2] or '-'}. "
                "Build a second request rather than asking this one to be a different scene."
            )
        return have
