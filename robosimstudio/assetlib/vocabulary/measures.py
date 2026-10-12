# SPDX-License-Identifier: Apache-2.0
"""The shared measurement code the asset library's handlers are built on.

Plain functions taking an :class:`~robosimstudio.scene.state.WorldState`.
Not a composition framework: an atom's handler calls the one it needs and
the benchmark tier's goal combinators do the combining, because
the combining is the part that differs per goal and the measuring is the part
that does not.

Most of what remains is deformable and **new** -- coverage, fold extent, and
whether a grasp actually took hold of a sheet were designed here, since no
published benchmark defines them for a task shaped like these. Each reports a
continuous quantity rather than a boolean, because a threshold nobody has
calibrated is worth less than the number it thresholds, and the threshold now
belongs to whoever declares it: an asset's ``StateSpec`` or a document's clause.

:func:`cloth_fold_error` is the exception, and is deliberately **not** invented:
it is SoftGym's ``ClothFold`` measure, down to the 1.2 weight on the drag term
and the normalisation against the flat start. A folding benchmark that scores
folds its own way is not comparable to anything, and this one is the measure the
cloth-manipulation literature already reports.

:func:`resting_on` is likewise transcribed rather than designed -- see its own
docstring for which benchmark and why it is stronger than a region test.
"""

from __future__ import annotations

import numpy as np

from robosimstudio.scene.state import WorldState

__all__ = [
    "FOLD_DRAG_WEIGHT",
    "cloth_coverage",
    "cloth_fold",
    "cloth_fold_error",
    "cloth_lifted",
    "held_particles",
    "mirror_pairs",
    "resting_on",
]

FOLD_DRAG_WEIGHT = 1.2
"""Weight on the drag term of the fold score. SoftGym ``ClothFold``'s own."""


def resting_on(
    ctx: WorldState,
    obj: str,
    target: str,
    *,
    force_threshold: float = 0.1,
    velocity_threshold: float = 0.1,
) -> tuple[bool, dict[str, float]]:
    """Arena's placement test: ``obj`` pressing on ``target``, and at rest.

    Transcribed from ``isaaclab_arena.tasks.predicates.spatial.object_on_destination``,
    which is what ``PickAndPlaceTask`` and ``SortMultiObjectTask`` terminate on: a
    contact-sensor force between the object and its destination above
    ``force_threshold``, and the object's linear speed below
    ``velocity_threshold``. The thresholds are the benchmark's own and differ
    between its two shapes -- **0.1 N** for one object, **1.0 N** for the
    multi-object sort -- so a suite passes the one its variant is.

    This is a materially stronger statement than a region test, and worth saying
    why: BDDL's ``In`` is a box a *carried* object satisfies, so an episode can be
    scored on a can still between the fingers 15 cm up. A force from the
    destination that the object is not moving under means it was let go and it
    landed. There is no separate "released" conjunct to add, because a held object
    accelerating downwards fails the velocity test and one held *still* against
    the bin floor is, physically, resting on it.

    Nothing latches: an item knocked back out stops counting, which is the honest
    reading of a goal stated over the world's current state.

    Returns:
        ``(resting, {"force", "speed"})`` -- the verdict plus both measured
        quantities, so a run that scored zero says which half failed.
    """
    if ctx.contacts is None:
        return False, {"force": 0.0, "speed": 0.0}
    source = ctx.handles.entity(obj)
    destination = ctx.handles.entity(target)
    force = float(np.linalg.norm(ctx.contacts.pairwise_force(source.collision_shapes, destination.collision_shapes)))
    speed = ctx.linear_speed(obj)
    return bool(force > force_threshold and speed < velocity_threshold), {"force": force, "speed": speed}


# -------------------------------------------------------------- deformable
def held_particles(
    ctx: WorldState,
    cloth: str,
    *,
    arm: str = "",
    particle_set: str = "",
    radius: float = 0.02,
    dragger=None,
) -> tuple[bool, int]:
    """Whether a gripper has actually taken hold of a sheet, and of how much.

    Two ways to be held, and both count:

    * A **latch** -- a :class:`~robosimstudio.core.ParticleDragger`
      holding particles kinematically, which is how a scripted routine or a
      teleoperator carries cloth. If a dragger is passed, its grabbed set is
      the answer, because a latch is not a guess.
    * A **pinch** -- particles inside ``radius`` of the tool-centre point with
      the gripper closed. What a policy driving the fingers directly produces,
      and the only signal available without a dragger.

    Closing two rigid fingers on a sub-millimetre film does not reliably grip
    it, which is why the latch exists at all; this predicate reports either.
    """
    if dragger is not None and dragger.active:
        return True, int(dragger.n_grabbed)

    arm_handle = ctx.handles.arms.get(arm) if arm else next(iter(ctx.handles.arms.values()), None)
    if arm_handle is None:
        return False, 0
    from robosimstudio.core.math import transform_point

    tcp = transform_point(ctx.body_q[arm_handle.ee_body], arm_handle.ee_offset)
    probe = ctx.probe(cloth)
    name = f"{cloth}/{particle_set}" if particle_set else cloth
    near = probe.near(tcp, radius, name)
    return bool(len(near)), int(len(near))


def cloth_coverage(ctx: WorldState, cloth: str, *, particle_set: str = "") -> float:
    """How spread out a sheet is, as a fraction of its settled footprint.

    1.0 is "as spread as it settled"; folding halves it, balling it up drives it
    towards 0. The obvious continuous measure of "is the garment flat", and the
    one a flattening task should be scored on once its threshold is calibrated.
    """
    name = f"{cloth}/{particle_set}" if particle_set else cloth
    return ctx.probe(cloth).spread_ratio(name)


def cloth_fold(ctx: WorldState, cloth: str, axis: int = 0, *, particle_set: str = "") -> tuple[float, float]:
    """How folded a sheet is about an axis, and its residual extent [m].

    A fold halves the sheet's extent along the folding axis while leaving the
    other alone, so the ratio of current to settled extent along ``axis`` is
    the measure: 1.0 unfolded, about 0.5 folded once, about 0.25 twice.

    Returns:
        ``(fold_ratio, extent)`` -- the ratio, and the current extent [m], so an
        evaluator can threshold either.
    """
    probe = ctx.probe(cloth)
    name = f"{cloth}/{particle_set}" if particle_set else cloth
    lo, hi = probe.extent(name)
    extent = float(hi[axis] - lo[axis])
    if probe.baseline is None:
        return 1.0, extent
    base = probe.baseline[probe.indices(name)]
    base_extent = float(base[:, axis].max() - base[:, axis].min())
    return (extent / base_extent if base_extent > 1.0e-9 else 1.0), extent


def mirror_pairs(probe, moving: str, base: str, *, axis: int, origin: float) -> np.ndarray:
    """Pair each ``moving`` particle with the ``base`` particle it should land on.

    A fold about a line maps the moving half onto the standing half by
    reflection, so "where should this particle end up" has an exact answer on
    the *rest* layout: mirror its rest position through the fold line and take
    the nearest standing particle. On a grid that lands on a vertex exactly.

    Computed **once**, against the settled baseline, and reused every step --
    which is the only reason a brute-force nearest-neighbour is affordable here.
    Recomputing the correspondence from the live sheet would defeat the point:
    the measure is how far each particle is from where it *should* be, and a
    correspondence that follows the cloth around always reports zero.

    Args:
        probe: The cloth's probe, with a baseline already snapshotted.
        moving: Set name of the half that gets carried.
        base: Set name of the half that stays.
        axis: The axis the fold reflects across, 0 for x or 1 for y.
        origin: Where the fold line sits on that axis [m].

    Returns:
        Indices **into the base set**, one per moving particle.
    """
    if probe.baseline is None:
        raise RuntimeError("no settled baseline; call snapshot_baseline() when the scene is ready")
    rest_moving = probe.baseline[probe.indices(moving)]
    rest_base = probe.baseline[probe.indices(base)]
    if not len(rest_moving) or not len(rest_base):
        return np.zeros(0, dtype=np.int64)
    reflected = rest_moving.copy()
    reflected[:, axis] = 2.0 * float(origin) - reflected[:, axis]
    # [n_moving, n_base] once, at scene-ready. A few hundred thousand entries for
    # a sheet of a thousand particles, against scipy's KD-tree that this repo
    # deliberately does not depend on.
    d2 = ((reflected[:, None, :] - rest_base[None, :, :]) ** 2).sum(axis=2)
    return np.asarray(d2.argmin(axis=1), dtype=np.int64)


def cloth_fold_error(
    ctx: WorldState,
    cloth: str,
    pairs: np.ndarray,
    *,
    moving: str = "moving",
    base: str = "base",
    drag_weight: float = FOLD_DRAG_WEIGHT,
    layer_thickness: float = 0.0,
) -> tuple[float, float, float]:
    """How well a sheet has been folded, on SoftGym's ``ClothFold`` measure.

    Two distances, and the second is what makes the first mean anything:

    * ``group_dist`` -- how far each carried particle is from the standing
      particle it should have landed on. Small when the halves lie on top of
      each other in the right correspondence, and *not* small for a sheet that
      has merely been gathered into a pile.
    * ``fixation_dist`` -- how far the standing half has drifted from where it
      settled. This is the term that rejects the degenerate solution: an arm
      that shoves the whole sheet sideways halves its extent and stacks nothing,
      and every measure without a drag penalty scores that as a fold.

    Reported as a score normalised against the flat start, so 0 is "no better
    than the sheet you were given" and 1 is a geometrically perfect fold with no
    drift. Negative is possible and means worse than the start, which is a real
    outcome worth being able to report rather than clamping away.

    The normaliser is derived from the baseline rather than measured at reset:
    at the start the standing half has not drifted, so the initial performance
    is exactly the mean mirror separation of the flat sheet.

    Args:
        cloth: Which cloth.
        pairs: The correspondence from :func:`mirror_pairs`.
        moving: Set name of the half that gets carried.
        base: Set name of the half that stays.
        drag_weight: Weight on ``fixation_dist``. SoftGym's 1.2.
        layer_thickness: Separation to forgive between the stacked layers [m].
            0.0 is SoftGym's formula exactly, and the default for that reason.

            Why a task would set it: the correspondence is a *planar* mirror, so
            it asks each carried particle to arrive at its target's position --
            including its height. A real fold cannot do that. The carried layer
            comes to rest **on top of** the standing one, one sheet thickness
            up, so a geometrically perfect fold still reports a group distance
            of about that thickness and cannot score 1. On a sheet whose
            particles are 7 mm in radius that caps the achievable score near
            0.91, which is below where an unwary gate would sit. Setting this to
            the sheet's own thickness measures the fold in the plane it actually
            happens in. It cannot flatter a bad fold: it subtracts a fixed few
            millimetres from a quantity that runs to hundreds.

    Returns:
        ``(score, group_dist, fixation_dist)`` -- the score, and the two
        distances behind it [m], **unforgiven**, so what is reported is the
        measurement and only the score carries the correction.
    """
    probe = ctx.probe(cloth)
    moving_key = f"{cloth}/{moving}"
    base_key = f"{cloth}/{base}"
    now_moving = probe.points(moving_key)
    now_base = probe.points(base_key)
    if not len(pairs) or not len(now_moving) or not len(now_base):
        return 0.0, 0.0, 0.0

    rest_moving = probe.baseline[probe.indices(moving_key)]
    rest_base = probe.baseline[probe.indices(base_key)]

    group_dist = float(np.linalg.norm(now_moving - now_base[pairs], axis=1).mean())
    fixation_dist = float(np.linalg.norm(now_base - rest_base, axis=1).mean())
    # The flat sheet's own mirror separation: what a score of 0 means.
    group_init = float(np.linalg.norm(rest_moving - rest_base[pairs], axis=1).mean())
    # Forgiven on *both* sides of the ratio, or the flat sheet you were handed
    # already scores above zero -- measured at 0.09 on the towel and 0.11 on the
    # shirt, purely because its own thickness was being subtracted from the
    # numerator and not from what the numerator is measured against.
    forgiven = max(0.0, group_dist - float(layer_thickness))
    scale = group_init - float(layer_thickness)
    if scale <= 1.0e-9:
        return 0.0, group_dist, fixation_dist
    score = 1.0 - (forgiven + drag_weight * fixation_dist) / scale
    return float(score), group_dist, fixation_dist


def cloth_lifted(ctx: WorldState, cloth: str, *, height: float = 0.05, particle_set: str = "") -> float:
    """Fraction of a sheet lifted more than ``height`` above where it settled."""
    name = f"{cloth}/{particle_set}" if particle_set else cloth
    return ctx.probe(cloth).lifted_fraction(name, height=height)
