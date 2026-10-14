# SPDX-License-Identifier: Apache-2.0
"""Changing a built model while it runs, for anything that is in one.

Newton keeps a cloth's constitutive parameters as Warp arrays the VBD solver
re-reads on every colour group of every iteration of every substep, so assigning
into one takes effect on the next substep and does so even inside a captured CUDA
graph, because a capture bakes the array's *pointer* and not the numbers behind
it. A rigid body's mass and its shapes' friction are Warp arrays on the same
model, reached the same way.

The difference is who is reading them. VBD reads the cloth arrays directly, so a
write is the whole of the change. A solver that keeps its own model -- MuJoCo
converts Newton's into an ``mjm`` at construction -- reads a *copy*, and a write
that nothing tells it about lands in an array it has stopped consulting. That is
the failure this module exists to not have: a slider that moves, a number that
changes, and a simulation that goes on behaving exactly as before.

Newton's answer is :meth:`~newton.solvers.SolverBase.notify_model_changed`, and
every shipped solver implements it. The flags name the arrays exactly:

* ``SHAPE_PROPERTIES`` -- ``shape_material_mu``, ``shape_material_ke``,
  ``shape_material_kd``, transforms and collision radii;
* ``BODY_INERTIAL_PROPERTIES`` -- ``body_mass``, ``body_inv_mass``,
  ``body_inertia``, ``body_inv_inertia``, ``body_com``;
* ``JOINT_DOF_PROPERTIES`` -- ``joint_target_ke``/``kd``, armature, friction and
  both limits.

So a writer here does two things: assign into the model, and say which category
changed. What the solver does about it is the solver's business, which is the
point -- this file does not know that MuJoCo has an ``mjm``.

**Why these are ``RECAPTURE`` and not ``LIVE``.** The assignment is live by the argument above. The *notification* is
host work: MuJoCo's re-sync launches kernels and, for some flags, flips a Python
flag that changes what the next contact conversion does. A captured graph replays
neither. Under capture the honest thing is to drop the graph and take it again --
which is what :meth:`~robosimstudio.core.pipeline.ScenePipeline.invalidate_graph`
is for, and which is free while the studio runs eager, exactly as it already is
for the cloth contact trio.

**What is not here.** Cloth. ``studio.tuning.apply.LiveMaterial`` already writes
it, is calibrated and is tested, and its per-field mapping onto ``tri_materials``,
``edge_bending_properties`` and ``tet_materials`` is genuinely cloth-specific
rather than a special case of anything. It needs no notification, for the reason
its own docstring gives at length.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

__all__ = [
    "ArrayWriter",
    "BodyWriter",
    "JointWriter",
    "ShapeWriter",
    "Tunable",
    "notify",
]


@dataclass(frozen=True)
class Tunable:
    """One group of parameters an entity offers, and how a change to it lands.

    What an entity's ``tunables()`` returns, and the reason it is a protocol
    rather than a table: an asset's
    parameters are the asset's, the same way its published states are, and a
    consumer that carries its own list of what may be tuned on what is carrying a
    copy that can disagree.

    Attributes:
        name: Which group this is -- ``"contact"``, ``"mass"``, ``"drive"``,
            ``"material"``. Shown as the section heading.
        spec: A frozen dataclass whose fields carry ``ui()`` descriptions.
            **Read from the built model**, not from the spec the scene was compiled out of: those
            are different numbers whenever an import directive did not survive
            into the model, which is most of the time for a scene-declared body.
        write: ``(field, value) -> did it land``. ``False`` means the field has
            nothing in this entity to write to -- a bending stiffness on a mesh
            with no edges, a joint gain on a body with no joints -- which is
            information a panel can show rather than an error.
        summary: One line, for the section.
    """

    name: str
    spec: Any
    write: Any
    summary: str = ""


def notify(pipeline: Any, flags: int) -> None:
    """Tell the scene's solvers that part of the model changed.

    Every one of them: a coupled arrangement runs several solvers over one
    model, and a shape whose friction moved is a shape any of them may be
    resolving contacts for. The program fans the notification out.

    Drops a captured graph, because the notification is host work a replay does
    not perform -- see the module docstring. Free while eager, which is the
    studio's default and the reason this is affordable at slider rate.

    Args:
        pipeline: The scene's :class:`~robosimstudio.core.pipeline.ScenePipeline`,
            or anything carrying a ``program`` or a ``solver``. ``None`` is a
            no-op, so a writer can be exercised against a bare model in a test.
        flags: A bit-mask of :class:`newton.ModelFlags`.
    """
    if pipeline is None or not flags:
        return
    program = getattr(pipeline, "program", None)
    if program is not None:
        program.notify(int(flags))
    else:
        for solver in (
            getattr(pipeline, "solver", None),
            getattr(pipeline, "solver_rigid", None),
            getattr(pipeline, "solver_cloth", None),
        ):
            if solver is not None:
                solver.notify_model_changed(int(flags))
    if getattr(pipeline, "captured", False):
        pipeline.invalidate_graph()


class ArrayWriter:
    """Assigns into a built model's arrays, through a host copy read fresh per write.

    A write downloads the array, edits this entity's rows and uploads it. The copy
    is not kept between writes: these arrays are model-wide -- every entity's
    shapes share one ``shape_material_mu`` -- so a copy kept by one writer goes
    stale the moment another writer, a script or ``ctl eval`` touches the array,
    and uploading it would put the other rows back the way they were.

    Args:
        model: The finalized model.
        pipeline: The scene's pipeline, for the notification. May be ``None``.
    """

    flags: int = 0
    """Which :class:`newton.ModelFlags` this writer's arrays fall under."""

    def __init__(self, model: Any, pipeline: Any = None) -> None:
        self.model = model
        self.pipeline = pipeline

    # ------------------------------------------------------------------ arrays
    def _fetched(self, attr: str) -> np.ndarray | None:
        """The array as the model holds it now, as a host copy to edit."""
        array = getattr(self.model, attr, None)
        return None if array is None else np.array(array.numpy(), copy=True)

    def _flush(self, attr: str, values: np.ndarray) -> None:
        getattr(self.model, attr).assign(values)

    def _write(self, attr: str, rows: Sequence[int], value: Any, *, column: int | None = None) -> bool:
        """Write one value into ``rows`` of one array. ``False`` if there is nowhere to."""
        index = np.asarray(rows, dtype=np.int64)
        if not len(index):
            return False
        values = self._fetched(attr)
        if values is None or not len(values):
            return False
        if column is None:
            values[index] = value
        else:
            values[index, column] = value
        self._flush(attr, values)
        return True

    def _read(self, attr: str, rows: Sequence[int], *, column: int | None = None, default: float = 0.0) -> Any:
        """One value as the model currently holds it, or ``default``.

        The *first* row's, not a mean: these specs are a view of a set of shapes
        or bodies that a person is about to set to one value, and a mean would
        show a number none of them has. When they disagree the first one is at
        least a number something in the entity actually has.
        """
        index = np.asarray(rows, dtype=np.int64)
        array = getattr(self.model, attr, None)
        if array is None or not len(index):
            return default
        values = self._fetched(attr)
        if values is None or not len(values):
            return default
        row = values[index[0]]
        return float(row if column is None else row[column])

    # ------------------------------------------------------------------ notify
    def landed(self, wrote: bool) -> bool:
        """Notify the solvers if anything was written, and report what happened."""
        if wrote:
            notify(self.pipeline, self.flags)
        return wrote


class ShapeWriter(ArrayWriter):
    """One entity's contact material, over the shapes it built.

    Args:
        model: The finalized model.
        shapes: Shape indices belonging to this entity -- ``EntityRange.shape_indices``.
        pipeline: The scene's pipeline.
    """

    flags = 1 << 4
    """``ModelFlags.SHAPE_PROPERTIES``. Spelled as the bit rather than imported so
    this module costs no newton import at definition time; the value is pinned by
    a test against the enum."""

    FIELDS: ClassVar[dict[str, str]] = {
        "mu": "shape_material_mu",
        "mu_rolling": "shape_material_mu_rolling",
        "mu_torsional": "shape_material_mu_torsional",
        "restitution": "shape_material_restitution",
        "ke": "shape_material_ke",
        "kd": "shape_material_kd",
        "kf": "shape_material_kf",
        "ka": "shape_material_ka",
        "thickness": "shape_thickness",
    }
    """Spec field to model array. The same set ``_MATERIAL_FIELDS`` (``scene/build/colliders.py``)
    copies between shapes, which is the list that already had to be right."""

    def __init__(self, model: Any, shapes: Iterable[int], pipeline: Any = None) -> None:
        super().__init__(model, pipeline)
        self.shapes = np.asarray(list(shapes), dtype=np.int64)

    def read(self) -> dict[str, float]:
        """Every field as the model currently holds it."""
        return {name: self._read(attr, self.shapes) for name, attr in self.FIELDS.items()}

    def __call__(self, field: str, value: Any) -> bool:
        attr = self.FIELDS.get(field)
        if attr is None:
            return False
        return self.landed(self._write(attr, self.shapes, float(value)))


class BodyWriter(ArrayWriter):
    """One entity's mass properties, over the bodies it built.

    **Mass scales inertia with it.** Writing ``body_mass`` alone leaves a body
    that weighs a tenth of what it did and resists rotation exactly as much,
    which is not a lighter object -- it is an object made of a different
    substance in one axis and not the others, and it tumbles wrong. Scaling the
    tensor by the same factor is what "the same shape, less dense" means, and it
    is the assumption stated rather than hidden: a caller wanting a hollow shell
    wants a different tensor, not a smaller number here.

    A massless body -- a static's shell -- stays massless. Its inverse mass is
    what marks it fixed, and quietly giving it one would unweld it from the
    world, which is the failure ``preserve_body_inertia`` (``scene/build/bodies.py``) exists to
    avoid on the other side of the build.
    """

    flags = 1 << 3
    """``ModelFlags.BODY_INERTIAL_PROPERTIES``."""

    def __init__(self, model: Any, bodies: Iterable[int], pipeline: Any = None) -> None:
        super().__init__(model, pipeline)
        self.bodies = np.asarray(list(bodies), dtype=np.int64)

    def read(self) -> dict[str, float]:
        return {"mass": self._read("body_mass", self.bodies)}

    def __call__(self, field: str, value: Any) -> bool:
        if field != "mass":
            return False
        return self.landed(self._scale_mass(float(value)))

    def _scale_mass(self, mass: float) -> bool:
        if not np.isfinite(mass) or mass <= 0.0:
            # Zero would weld a dynamic body to the world -- and then the guard
            # below would refuse it its mass back -- and a negative one would
            # turn its inertia inside out. Neither is a mass.
            return False
        current = self._fetched("body_mass")
        if current is None or not len(self.bodies):
            return False
        wrote = False
        for body in self.bodies:
            was = float(current[body])
            if was <= 0.0:
                # Welded to the world. Leaving it alone is the whole point.
                continue
            factor = mass / was
            current[body] = mass
            self._write("body_inv_mass", [body], 1.0 / mass)
            self._scale_tensor("body_inertia", body, factor)
            self._scale_tensor("body_inv_inertia", body, 1.0 / factor)
            wrote = True
        if wrote:
            self._flush("body_mass", current)
        return wrote

    def _scale_tensor(self, attr: str, body: int, factor: float) -> None:
        values = self._fetched(attr)
        if values is None or body >= len(values):
            return
        values[body] = values[body] * factor
        self._flush(attr, values)


class JointWriter(ArrayWriter):
    """One entity's joint drive gains and limits, over the DOFs it built.

    An articulated asset -- a drawer, a cabinet -- is the case: how hard its
    joint is driven and how far it may travel are the two things somebody opening
    one by hand wants to change, and neither survives into anything a rigid
    body's contact material describes.
    """

    flags = 1 << 1
    """``ModelFlags.JOINT_DOF_PROPERTIES``."""

    FIELDS: ClassVar[dict[str, str]] = {
        "target_ke": "joint_target_ke",
        "target_kd": "joint_target_kd",
        "armature": "joint_armature",
        "friction": "joint_friction",
        "limit_lower": "joint_limit_lower",
        "limit_upper": "joint_limit_upper",
    }

    def __init__(self, model: Any, dofs: Iterable[int], pipeline: Any = None) -> None:
        super().__init__(model, pipeline)
        self.dofs = np.asarray(list(dofs), dtype=np.int64)

    def read(self) -> dict[str, float]:
        return {name: self._read(attr, self.dofs) for name, attr in self.FIELDS.items()}

    def __call__(self, field: str, value: Any) -> bool:
        attr = self.FIELDS.get(field)
        if attr is None:
            return False
        return self.landed(self._write(attr, self.dofs, float(value)))
