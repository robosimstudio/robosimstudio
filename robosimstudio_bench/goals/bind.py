# SPDX-License-Identifier: Apache-2.0
"""Binding a checked clause to the measure that answers it.

The last unbuilt seam. A goal tree knows how to compose verdicts; handlers know
how to compute one from arrays; the typechecker knows that ``cable.point(0.63)``
resolves. Nothing yet turns ``(>= (winding-turns cable peg0.axis) 0.5)`` into a
callable that reads a live scene, and this module is that.

Three steps, and keeping them separate is what makes the thing testable:

1. **Resolve** each reference into an :class:`Operand` -- what it names, which
   substrate it is, which feature kind it lands on. Pure catalog work, no scene.
2. **Dispatch** on those kinds and substrates to pick the handler. Also pure, and
   done **once at build time** rather than per step: a goal that cannot be
   measured should fail when the task is compiled, not 400 steps into episode 3.
3. **Read** the operands out of a snapshot at call time and hand the handler its
   arrays.

Step 3 is the only part that needs a world, and it is a small table: what a
reference *means numerically* depends on the operand's substrate and kind, and
those are exactly the two things the object model already dispatches on. A rigid
body means its position; a garment means its particles; a region means its box
and where that box currently is.

**A missing handler is a normal outcome, not a crash.** It means the object does
not afford what the goal asked of it -- the single most useful thing an authoring
agent can be told, and the reason :func:`bind` raises with the operands named
rather than returning something that quietly scores zero.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.semantics.atoms import ATOMS
from robosimstudio.assetlib.semantics.features import FeatureSpec
from robosimstudio.assetlib.semantics.handlers import dispatch
from robosimstudio.assetlib.semantics.kinds import get_kind, is_registered_kind
from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio.assetlib.semantics.states import COMPARISONS
from robosimstudio.assetlib.semantics.substrates import SUBSTRATES, is_substrate, layout_chain
from robosimstudio.core.math import quat_to_mat
from robosimstudio.scene.entity import entity_layouts
from robosimstudio_bench.taskdoc.ast import Condition, Ref, TaskDoc

from .combinators import Leaf
from .goal import about_key

__all__ = ["READERS", "BindError", "Operand", "bind", "reader", "resolve_operand"]


class BindError(Exception):
    """A clause that typechecks but has nothing to measure it.

    Separate from a check problem because the fix is different: the document is
    right and the *world* is short an implementation. Raised at build time, with
    the operands named, so the message reads as "nothing can measure this of
    these things" rather than as a stack trace at step 400.
    """


@dataclass(frozen=True)
class Operand:
    """One reference, resolved against the catalog.

    Attributes:
        ref: The reference as written.
        asset: Registry name of the asset it lands on, or empty for an untyped
            object or a document-declared region.
        substrate: Its physics family, or empty when it has none -- a region is
            geometry, not a body.
        kind: The feature kind the reference lands on, or empty when it names a
            whole object. This is what dispatch keys on, and the reason
            ``threaded(loop, rod)`` works for a mug and a pair of scissors
            without either declaring an interface.
        feature: The feature's declaration, when the reference names one.
    """

    ref: Ref
    asset: str = ""
    substrate: str = ""
    kind: str = ""
    feature: FeatureSpec | None = None

    @property
    def name(self) -> str:
        """The scene handle this operand reads from."""
        return self.ref.base


# ------------------------------------------------------------------- readers
Reader = Callable[[Snapshot, Operand], Any]
"""Turns one resolved operand into the value a handler takes."""

READERS: dict[tuple[str, str, str], Reader] = {}
"""``(substrate, kind, form) -> reader``. Open, like the kind set.

Keyed on the two things dispatch is keyed on plus the form the atom asked for,
so adding a substrate means adding readers rather than editing a chain of
conditionals. Lookup drops one constraint at a time -- exact triple, then any
kind, then any substrate, then neither -- which is what lets one region reader
serve every substrate and one ``feature`` reader serve every geometric kind.
"""


def reader(substrate: str = "", kind: str = "", form: str = "position") -> Callable[[Reader], Reader]:
    """Register how one ``(substrate, kind, form)`` triple is read."""

    def register(fn: Reader) -> Reader:
        key = (substrate, kind, form)
        if key in READERS:
            raise ValueError(f"a reader for {key} is already registered")
        READERS[key] = fn
        return fn

    return register


def read(snapshot: Snapshot, operand: Operand, form: str = "position") -> Any:
    """The value an operand denotes right now, in the form the atom asked for.

    Args:
        snapshot: The instant.
        operand: What to read.
        form: One of ``OPERAND_FORMS``. The atom declares it per operand, because
            what a handler needs of a rigid body depends on the measure and not
            on the body.

    Raises:
        BindError: When nothing knows how to read this. Named rather than
            defaulted, because every default here is a number, and a wrong number
            in a goal is worse than no goal.
    """
    # An unknown substrate is refused here rather than being allowed to fall
    # through. The distinction matters and is easy to lose: an operand with an
    # EMPTY substrate is one the document did not type, which is legal BDDL and
    # reads as a plain body; an operand typed as something nothing has heard of
    # is a mistake. Before this check the unconstrained fallback below absorbed
    # both, so a body declared `substrate="plasma"` quietly read as a position.
    if operand.substrate and not is_substrate(operand.substrate):
        raise BindError(
            f"{operand.ref} is declared on substrate {operand.substrate!r}, which is not one of "
            f"{SUBSTRATES.names()}. An unknown substrate reads as nothing, not as a default."
        )

    # Most specific first, then each constraint dropped in turn, then the
    # unconstrained slot. That last one is not a loophole: a form like ``handle``
    # or ``feature`` genuinely has one answer whatever it is pointed at, and
    # without it a reader registered for anything would be unreachable the moment
    # an operand happened to carry a kind.
    # The layout's own chain, then the unconstrained slot. ``dispatch`` walks the
    # same chain for the same operand, and the two must agree: a goal that binds
    # against one layout's reader and scores through another layout's handler is
    # wrong in a way that raises nothing and reports a number.
    # The live entity's own capabilities extend the chain -- after the layout's
    # own, and even when the operand has no layout at all (a thing added straight
    # to a scene, with no record). That is what lets a soft-domain cable be read
    # as a curve, and a document-less cable be read at all.
    extra = entity_layouts(snapshot.entities.get(operand.ref.base)) if hasattr(snapshot, "entities") else ()
    for layout in (*layout_chain(operand.substrate, extra=extra), ""):
        for key in ((layout, operand.kind, form), (layout, "", form)):
            if key in READERS:
                return READERS[key](snapshot, operand)
    raise BindError(
        f"nothing knows how to read {operand.ref} as {form!r} -- substrate "
        f"{operand.substrate or '(none)'!r}, kind {operand.kind or '(whole object)'!r}. "
        f"Registered readers: {sorted(READERS)}"
    )


@reader(kind="region", form="region")
def _region(snapshot: Snapshot, operand: Operand) -> dict:
    """A region's box, **in world coordinates**.

    The frame origin is applied here rather than passed alongside, because that
    is the contract the handlers were written to: ``regions.py`` says regions
    arrive in world coordinates and that applying the frame pose is the
    resolver's job. Handing over local ranges plus an origin the handler defaults
    to zero would put every region at the world origin -- silently, since a box
    at the wrong place is still a box.

    Re-read every step rather than baked at compile time, because a region
    anchored to a drawer rides the drawer. For a bin welded into an arena the two
    are identical and the cost is an array index.
    """
    handle = snapshot.handles.regions[operand.name]
    ox, oy, oz = (float(v) for v in handle.origin_now(snapshot.body_q))
    x0, y0, x1, y1 = handle.xy_ranges
    z0, z1 = handle.z_range
    return {"xy_ranges": (x0 + ox, y0 + oy, x1 + ox, y1 + oy), "z_range": (z0 + oz, z1 + oz)}


@reader(substrate="rigid")
def _rigid(snapshot: Snapshot, operand: Operand) -> np.ndarray:
    """A rigid body's position [m]."""
    return _body_position(snapshot, operand)


@reader(substrate="rigid", kind="landmark", form="position")
@reader(substrate="articulated", kind="landmark", form="position")
def _body_landmark(snapshot: Snapshot, operand: Operand) -> np.ndarray:
    """A declared body-frame landmark transformed into world coordinates."""
    feature = _feature(snapshot, operand)
    return np.asarray(feature["position"], dtype=np.float64)


@reader(substrate="rigid", form="velocity")
@reader(substrate="articulated", form="velocity")
@reader(substrate="", form="velocity")
def _body_velocity(snapshot: Snapshot, operand: Operand) -> np.ndarray:
    """A body's spatial velocity, ``[6]``, linear first.

    Both halves, not just the linear part: which one a measure wants is the
    measure's business, and truncating here would silently deny a future atom
    the spin.
    """
    handle = snapshot.handles.entity(operand.name)
    if 0 <= handle.body < len(snapshot.body_qd):
        return np.asarray(snapshot.body_qd[handle.body], dtype=np.float64)
    if handle.static:
        # A welded static does not move; see :func:`_static_or_body_pose`.
        return np.zeros(6, dtype=np.float64)
    raise BindError(f"{operand.ref} names {operand.name!r}, which has no body to read a velocity from")


@reader(substrate="rod", form="velocity")
def _rod_velocity(snapshot: Snapshot, operand: Operand) -> np.ndarray:
    """A chain's segment velocities, ``[n, 6]``, in order along its length.

    The counterpart to the chain position reader, and it was missing -- ``speed``
    registers a handler for every substrate, so ``(speed cable)`` typechecked,
    dispatched, and then fell through to the plain-body reader, which asked the
    scene for an entity called ``cable`` when a chain has no single body at all.
    A raw ``KeyError`` at step 400, which is the exact failure the build-time
    dispatch exists to prevent.

    The mean is not taken here. What "moving" means for a chain is the measure's
    business -- a cable whose ends counter-rotate is not at rest, and averaging
    it away here would decide that for every measure at once. See
    ``handlers.rod._rod_speed`` for what the cable suite decided.
    """
    return np.asarray(snapshot.body_qd[_rod_bodies(snapshot, operand)], dtype=np.float64)


@reader(substrate="cloth", form="velocity")
def _cloth_velocity(snapshot: Snapshot, operand: Operand) -> np.ndarray:
    """A garment's particle velocities, ``[n, 6]``-shaped by padding.

    Padded rather than returned as ``[n, 3]`` so that every velocity reader hands
    back the same column layout and a measure taking the first three columns
    means the same thing whatever it is pointed at.
    """
    probe = _own_probe(snapshot, operand)
    linear = np.asarray(probe.velocities[probe.indices(operand.name)], dtype=np.float64)
    return np.hstack([linear, np.zeros_like(linear)])


@reader(substrate="rigid", form="pose")
@reader(substrate="articulated", form="pose")
@reader(substrate="", form="pose")
def _body_pose(snapshot: Snapshot, operand: Operand) -> np.ndarray:
    """A body's full pose, ``[x, y, z, qx, qy, qz, qw]``.

    Separate from ``position`` because most measures do not want the orientation
    and the ones that do cannot do without it: a tilted plug's tab tip is not at
    its centre plus an offset along the world axis, and on a 13 mm bore that
    difference is 8% of the gate.
    """
    return _static_or_body_pose(snapshot, operand, "a pose")


def _static_or_body_pose(snapshot: Snapshot, operand: Operand, what: str) -> np.ndarray:
    """An entity's pose, ``[7]``, from its ``body_q`` row or, for a welded static, where it was put.

    A single-world build welds a fixture to the world as shapes on body -1, so
    it has no row to read; its pose is the compiler's ``xform``, as
    :func:`_feature` already reads it. Refusing that here crashed the first
    scored step of any goal pointing a position, pose or velocity at a fixture.
    """
    handle = snapshot.handles.entity(operand.name)
    if 0 <= handle.body < len(snapshot.body_q):
        return np.asarray(snapshot.body_q[handle.body], dtype=np.float64)
    if handle.static:
        return np.asarray(handle.xform, dtype=np.float64)[:7]
    raise BindError(f"{operand.ref} names {operand.name!r}, which has no body to read {what} from")


@reader(substrate="cloth", form="feature")
def _cloth_feature(snapshot: Snapshot, operand: Operand) -> dict[str, Any]:
    """A garment's declared feature, exactly as declared.

    No rotation and no translation, and the reason is that there is nothing to
    rotate *by*. A rigid feature is stated in its body's frame and the body has a
    pose; a garment has no pose -- it has particles, and every one of them moves
    independently. Its features are stated against the **rest layout**, resolved
    once when the sheet has settled, and that is the frame they stay in.

    Falling through to the general reader instead is not a smaller mistake than
    it looks: a cloth handle has no ``body_q`` row and is not a welded static, so
    the general reader raises -- which reads as "this garment cannot carry a
    feature" rather than as "a garment's frame is a different question".

    The feature's own **name** rides along with its payload, which the general
    reader has no need of and a garment's measures cannot do without: a particle
    set is realised on the cloth under the name the asset declared it by, so a
    handler holding only ``{"select": ...}`` knows the recipe and not which
    resolved set it produced.
    """
    del snapshot
    if operand.feature is None:
        raise BindError(f"{operand.ref} does not name a declared feature, so there is no payload to read")
    return {"name": operand.feature.name, **operand.feature.payload}


@reader(form="feature")
def _feature(snapshot: Snapshot, operand: Operand) -> dict[str, Any]:
    """A declared feature's payload, **in world coordinates**.

    One reader for every geometric kind, because the conversion is the same
    question each time and the *kind* already knows the answer: which payload
    fields are points (rotate and translate) and which are directions (rotate
    only). A handler that did this itself would have to get the same rotation
    right once per kind it dispatches on, and one that forgot to translate would
    put the feature at the world origin -- a number, plausible, and wrong by
    however far the body is from the origin.

    Fields the kind names as neither are passed through untouched: a radius, a
    depth and a bond count are not geometry and rotating them would be nonsense.
    """
    if operand.feature is None:
        raise BindError(f"{operand.ref} does not name a declared feature, so there is no payload to read")
    kind = get_kind(operand.kind) if is_registered_kind(operand.kind) else None
    handle = snapshot.handles.entity(operand.name)
    if 0 <= handle.body < len(snapshot.body_q):
        pose = np.asarray(snapshot.body_q[handle.body], dtype=np.float64)
    elif handle.static:
        # A welded static has no `body_q` row -- its pose is where the compiler
        # put it. Which is the common case for a feature: a bore is on a
        # receptacle and a receptacle that moved would not be a fixture.
        pose = np.asarray(handle.xform, dtype=np.float64)
    else:
        raise BindError(f"{operand.ref} hangs off {operand.name!r}, which has no body to place it against")
    origin, rotation = pose[:3], quat_to_mat(pose[3:7])
    points = set(kind.points if kind else ())
    directions = set(kind.directions if kind else ())

    out: dict[str, Any] = {}
    for name, value in operand.feature.payload.items():
        if name in points:
            out[name] = _place(np.asarray(value, dtype=np.float64), rotation, origin)
        elif name in directions:
            out[name] = _place(np.asarray(value, dtype=np.float64), rotation, np.zeros(3))
        else:
            out[name] = value
    return out


def _place(value: np.ndarray, rotation: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """One point or many, rotated into the world and offset.

    A payload field may hold a single ``[3]`` -- a bore's mouth, a loop's centre
    -- or a ``[n, 3]`` of them. A ``rod``'s ``axis`` is the second case: two
    endpoints, because a rod is a finite segment and hanging past its end is not
    hanging, so both ends have to travel with the body.
    """
    if value.ndim == 1:
        return origin + rotation @ value
    return (rotation @ value.T).T + origin


@reader(form="handle")
def _handle(snapshot: Snapshot, operand: Operand) -> str:
    """The scene name, for a measure that looks the entity up itself.

    Deliberately the *name* and not the handle object. A contact measure needs
    shapes, velocities and a body index together, and handing it one resolved
    handle would mean choosing which of those to resolve here -- so it gets the
    name and the snapshot, and does its own looking up. Registered for every
    substrate and kind, because "what is this thing called" has one answer.
    """
    del snapshot
    return operand.name


@reader(substrate="articulated")
def _articulated(snapshot: Snapshot, operand: Operand) -> np.ndarray:
    """A jointed body's own frame -- its root, not any of its parts."""
    return _body_position(snapshot, operand)


@reader(substrate="")
def _untyped(snapshot: Snapshot, operand: Operand) -> np.ndarray:
    """An object the document did not type.

    Legal BDDL and common in imported problems, which name ``hammer_0 - hammer``
    with no asset behind ``hammer``. It still has a body, so it still has a
    position, and reading it is better than refusing a whole corpus over a
    missing declaration.
    """
    return _body_position(snapshot, operand)


@reader(substrate="cloth")
def _cloth(snapshot: Snapshot, operand: Operand) -> np.ndarray:
    """A garment's particles, ``[n, 3]`` -- which is all a garment can answer."""
    probe = _own_probe(snapshot, operand)
    return np.asarray(probe.points(operand.name), dtype=np.float64)


@reader(substrate="rod")
def _chain(snapshot: Snapshot, operand: Operand) -> np.ndarray:
    """A chain's ordered points, ``[n, 3]``, along its length.

    Order is the whole content: winding turns and arc-length points are both
    statements about *sequence*, and a reader that returned the same points
    shuffled would compute a confident wrong answer rather than failing.

    **The entity says where its points are, and only then does this guess.**
    This read ``body_q`` unconditionally, which was right while a rod could only
    be a chain of capsules -- and a substrate is a *state layout*, not a place
    to find one. The same cable simulated as a Cosserat centerline owns no body
    at all, so it built, stepped, rested correctly, and bound to nothing. The
    body path stays as the fallback, so every rod that worked before still
    reads exactly as it did.
    """
    points = _entity_reads(snapshot, operand, "points")
    if points is not None:
        return np.asarray(points, dtype=np.float64)
    return np.asarray(snapshot.body_q[_rod_bodies(snapshot, operand)][:, :3], dtype=np.float64)


@reader(substrate="rod", kind="point", form="position")
def _rod_point(snapshot: Snapshot, operand: Operand) -> np.ndarray:
    """One world point along a chain, by normalised arc length.

    The ``point`` kind's whole content is an ``s`` in ``[0, 1]``, and the chain
    substrate answers it for free -- which is why the cable needed no authored
    features and a rigid mesh needs measured ones. ``s`` comes from the
    reference's own argument when it carries one (``cable.midpoint(0.63)``) and
    from the declaration otherwise, so a document may aim a named landmark
    somewhere else without the asset growing a feature per task.

    **Interpolated between segment centres, not snapped to the nearest one.**
    The alternative reads a different rod every time somebody changes the
    capsule count, which is exactly the property the kind's own contract
    promises against ("arc length is normalised, so a reference survives a
    change of segment count"). It also happens to be the more faithful reading
    of the measure this replaced: the shipped cable evaluator latched *the
    segment nearest a declared xy*, and with an even capsule count the midpoint
    falls between two of them -- ``argmin`` broke that tie toward the lower
    index and landed half a segment short, where this lands on the point itself.
    """
    positions = _chain(snapshot, operand)
    if len(positions) == 1:
        return positions[0]
    payload = operand.feature.payload if operand.feature is not None else {}
    # An `anchor` states where it holds under `at`; a `point` states `s`
    # directly. Both are one number and mean the same thing, so a reference to
    # either is readable as a place along the rod.
    at = payload.get("at", payload)
    declared = at.get("s", 0.5) if isinstance(at, dict) else 0.5
    s = float(operand.ref.arg if operand.ref.arg is not None else declared)
    span = float(np.clip(s, 0.0, 1.0)) * (len(positions) - 1)
    low = min(int(span), len(positions) - 2)
    return positions[low] + (span - low) * (positions[low + 1] - positions[low])


def _chain_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


@reader(substrate="rod", form="feature")
def _rod_feature(snapshot: Snapshot, operand: Operand) -> dict[str, Any]:
    """A chain as a feature: its ordered points in world, and their path length.

    The ``geometry`` atom reads its operand in this form. A rigid body answers
    it from a declared feature; a rod declares none and did not answer at all,
    so ``sim.query("geometry", "cable")`` -- the very example the tool's own
    docstring gives -- refused, and every agent that tried it concluded the
    cable was unreadable. A rod's whole geometry *is* its chain; that is the
    feature.
    """
    points = _chain(snapshot, operand)
    return {
        "kind": "chain",
        "points": [[float(v) for v in row] for row in np.atleast_2d(points)],
        "length_m": _chain_length(np.atleast_2d(points)),
        "n_points": int(len(np.atleast_2d(points))),
    }


@reader(substrate="rod", kind="point", form="feature")
def _rod_point_feature(snapshot: Snapshot, operand: Operand) -> dict[str, Any]:
    """One arc-length point of a chain, as a feature: ``{"at": [x, y, z], "s": s}``."""
    at = _rod_point(snapshot, operand)
    payload = operand.feature.payload if operand.feature is not None else {}
    declared = payload.get("at", payload)
    s_declared = declared.get("s", 0.5) if isinstance(declared, dict) else 0.5
    s = float(operand.ref.arg if operand.ref.arg is not None else s_declared)
    return {"kind": "point", "at": [float(v) for v in at], "s": float(np.clip(s, 0.0, 1.0))}


def _own_probe(snapshot: Snapshot, operand: Operand) -> Any:
    """This operand's probe, refusing one that has no set under its own name.

    A ``ParticleProbe`` holds the **whole model's** cloud -- one probe per
    entity, each sized ``model.particle_count`` -- and each entity registers its
    own range as a set under its own name when it binds. Reading ``positions``
    raw therefore hands a measure every particle in the scene, including other
    deformables': on a towel and a rag together, both read 124 points where they
    own 99 and 25. Not an exception and not a shape error -- a fold score
    computed over somebody else's cloth.

    Every shipped task carries one deformable, which is why this was latent and
    why fixing it moves no recorded number.

    A missing set is refused rather than defaulted back to the whole cloud --
    the default is what made the bug quiet.
    """
    probe = snapshot.particles.get(operand.name)
    if probe is None:
        raise BindError(f"{operand.ref} is a deformable, but the scene carries no particle probe for {operand.name!r}")
    if operand.name not in getattr(probe, "sets", {}):
        raise BindError(
            f"{operand.ref} names {operand.name!r}, which carries a particle probe that has no set of its "
            f"own under that name; it has {sorted(getattr(probe, 'sets', {}))}. An entity registers its own "
            "range when it binds, so this is a name that does not match rather than a thing without particles."
        )
    return probe


def _entity_reads(snapshot: Snapshot, operand: Operand, form: str) -> Any | None:
    """What the entity itself says, in one operand form, or ``None``.

    Duck-typed on purpose, and ``getattr`` rather than ``hasattr``: every
    ``Entity`` subclass has the method whether or not it answers anything, so
    ``hasattr`` is true for all of them and false for the one shape that matters
    -- a spec-and-builder written from outside without subclassing, which the
    extension surface promises works. Asking for the method and calling it
    treats both the same way, which is the whole point of a duck-typed seam.
    """
    reader = getattr(snapshot.entities.get(operand.name), "read", None)
    return reader(form, snapshot) if callable(reader) else None


def _rod_bodies(snapshot: Snapshot, operand: Operand) -> np.ndarray:
    """A chain's body indices, in order along its length.

    Order is the whole content: winding turns and arc-length points are both
    statements about *sequence*, and a reader that returned the same segments
    shuffled would compute a confident wrong answer rather than failing.

    **The entity's own array first, and that is not an optimisation.** A rod
    cable is built by ``ModelBuilder.add_rod`` as a suite-owned *extra*: its
    segments are bodies the scene never publishes as handles, the same way the
    RJ45 assembly's plug is not an object a document can name. So there is
    nothing called ``cable_0`` to find, and before this the chain readers found
    nothing at all. The entity brackets the rod's own body range when it binds
    and hands it over here in build order, which is chain order by construction.

    The name scan behind it stays for a chain some future scene assembles out of
    ordinary objects. **Its sort is lexical, which is not the same as numeric**
    -- ``cable_10`` sorts before ``cable_2`` -- so a chain that arrives that way
    owes zero-padded names. Recorded rather than fixed: a numeric key would be a
    guess about a naming convention no shipped scene has yet, and the one chain
    that ships does not come through this branch.
    """
    entity = snapshot.entities.get(operand.name)
    bodies = getattr(entity, "bodies", None)
    if bodies is not None and len(bodies):
        return np.asarray(bodies, dtype=np.int64)
    segments = [h for name, h in sorted(snapshot.handles.objects.items()) if name.startswith(f"{operand.name}_")]
    if not segments:
        raise BindError(
            f"{operand.ref} is a chain, but nothing in the scene answers for it: no entity named "
            f"{operand.name!r} publishing an ordered body range, and no objects named {operand.name}_*"
        )
    return np.asarray([h.body for h in segments], dtype=np.int64)


def _body_position(snapshot: Snapshot, operand: Operand) -> np.ndarray:
    return _static_or_body_pose(snapshot, operand, "a position")[:3]


# ------------------------------------------------------------------ resolving
def resolve_operand(ref: Ref, doc: TaskDoc, catalog: Callable[[str], AssetRecord | None]) -> Operand:
    """Resolve one reference against the document and the catalog.

    Assumes the document has typechecked: a reference to an object nothing
    declares is ``check``'s problem to report, with a message about the *file*,
    and repeating that here would give the same mistake two different voices.
    """
    if any(r.name == ref.base for r in doc.regions) and not ref.path:
        return Operand(ref=ref, kind="region")

    instance = next((o for o in doc.objects if o.name == ref.base), None)
    if instance is None or not instance.type:
        return Operand(ref=ref)

    record = catalog(instance.type)
    if record is None:
        return Operand(ref=ref, asset=instance.type)

    if not ref.path:
        return Operand(ref=ref, asset=record.name, substrate=record.substrate)

    spec = next((f for f in record.features if f.name == ref.path[0]), None)
    kind = spec.kind if spec else ""
    if spec is None and is_registered_kind(ref.path[0]):
        # A part named after a kind the substrate gives for free -- the
        # rod's ``point`` -- needs no declaration: ``cable.point(0.63)`` is
        # the grammar the query tool documents, and it resolved to an untyped
        # part that no handler admitted, so the documented example refused.
        free = get_kind(ref.path[0])
        if not free.substrates or record.substrate in free.substrates:
            kind = ref.path[0]
    return Operand(
        ref=ref,
        asset=record.name,
        substrate=record.substrate,
        kind=kind,
        feature=spec,
    )


# -------------------------------------------------------------------- binding
def bind(
    condition: Condition,
    doc: TaskDoc,
    catalog: Callable[[str], AssetRecord | None],
    *,
    task: str = "",
) -> Leaf:
    """Build the node that measures one leaf clause.

    Dispatch happens **here**, not per step. A goal naming a measure nothing
    implements for these operands is a task that should refuse to compile, and
    discovering it mid-episode would mean a run whose failure is indistinguishable
    from a policy failure.

    Args:
        condition: A leaf -- a bare predicate or a comparison. Combinators are
            :func:`~robosimstudio_bench.goals.goal.build_goal`'s business.
        doc: The document, for resolving references.
        catalog: Looks an asset up by name, returning ``None`` when it has none.
        task: The task's name, for task-scoped handler overrides.

    Raises:
        BindError: When no handler implements the measure for these operands.
    """
    operands = tuple(resolve_operand(ref, doc, catalog) for ref in condition.args)
    published = _published_reader(condition, operands, doc, catalog)
    if published is not None:
        return published

    # BDDL is a lisp and writes `contact-force`; Python identifiers cannot, so
    # atoms are registered with underscores. Normalising here rather than at the
    # parser keeps the document's own spelling in the AST -- which is what a
    # round-trip and an error message should show -- while everything downstream
    # sees one name. The published-state path does the same, and had to learn it
    # the same way: matching one spelling only reported `bond-fraction`, the
    # single quantity its task is scored on, as unresolved.
    # `atom:key` selects which of a multi-valued measure's returns is being
    # thresholded. `insertion` reports depth, lateral offset and mate gap from one
    # computation, and a goal needs to gate on two of them -- so rather than
    # minting three atoms that share an implementation and can drift apart, a
    # clause says which reading it means.
    #
    # Cost: two clauses over one atom call it twice per step. For `insertion`
    # that is pure arithmetic over seven floats. An atom where it would not be
    # should declare `cost` accordingly, which is what that field is for.
    measure, _, wanted_key = condition.head.replace("-", "_").partition(":")

    kinds = tuple(o.kind for o in operands)
    substrates = tuple(o.substrate for o in operands)
    owner = next((o.asset for o in operands if o.asset), "")
    try:
        spec = dispatch(measure, kinds, substrates, asset=owner, task=task)
    except LookupError as exc:
        raise BindError(
            f"{condition.head}({', '.join(str(o.ref) for o in operands)}) has no implementation: {exc}"
        ) from exc

    # The declaration, for what form each operand is read in and whether the
    # measure is smoothed. Separate from the handler on purpose: the atom is the
    # contract, the handler is one implementation of it, and a second handler
    # must not be free to want its operands in a different shape.
    # A handler with no declaration is refused as a BindError: every read below
    # (operand forms, ``reads_scene``, ``primary``, smoothing) is the
    # declaration's, and ``ATOMS.get`` raises ``KeyError`` for an undeclared name.
    try:
        atom = ATOMS.get(measure)
    except KeyError as exc:
        raise BindError(
            f"{condition.head}: a handler implements {measure!r}, but no atom declares it, so there "
            f"is no contract for how its operands are read or what it reports ({exc})"
        ) from None

    key = wanted_key or atom.primary
    if wanted_key and wanted_key not in atom.returns:
        raise BindError(
            f"atom {measure!r} does not report {wanted_key!r}; it reports {list(atom.returns)}. "
            "A clause selects one of a measure's declared returns, and a typo here would "
            "otherwise threshold whichever key happened to come first."
        )
    op, threshold = condition.op, condition.value

    # What this clause is about, as a metric key: the atom and every operand. A
    # goal tree already namespaces metrics by tree position, which is unambiguous
    # and unreadable -- `goal.2.fraction` says where a number sits and not what
    # it is about. Every operand, not just the first: `(In hammer_0 A)` and
    # `(In hammer_0 B)` are different clauses about one object, and a key naming
    # only the object would merge them into whichever ran last.
    #
    # Built through `about_key` rather than formatted here, so that whoever reads
    # this key builds it the same way. See that function for what it cost to
    # learn that the two drifting apart fails silently.
    # The full head -- selector included -- so `insertion:lat` and plain
    # `insertion` over one pair are two distinct clauses rather than one that
    # overwrites the other. This is what `about` carries too, so that
    # `about_key(*leaf.about)` reproduces exactly the key the leaf reports
    # under. They disagreed for one commit, and the symptom was two clauses
    # being indistinguishable to anything reading `about` while remaining
    # distinct in the metrics -- the two halves of one fact, differing.
    #
    # And the full REFERENCE per operand, not the scene handle it hangs off.
    # `port.bore` rather than `port`, which for a bore on a receptacle is only
    # more precise -- but for a garment it is the whole distinction. A shirt's
    # three creases are three clauses over one cloth, differing only in which
    # panel each names, so keyed by handle alone all three report under
    # `fold_score/shirt/shirt` and the last one written is the only one anybody
    # reads. Three creases collapsing into one number is not a reporting
    # blemish: `worst_of` exists to say which crease was the weak one.
    about = (condition.head.replace("-", "_"), tuple(str(o.ref) for o in operands))
    named = about_key(*about)

    # What form each operand is read in, declared by the atom. An operand the
    # atom did not name -- an optional one a call omitted -- falls back to a
    # position, which is what almost every measure wants.
    forms = [arg.reads or "position" for arg in atom.args] + ["position"] * len(operands)

    # A smoothing window is per *bound clause*, not per atom: two clauses reading
    # the same atom are two instruments, and sharing one buffer would let one
    # object's contact churn mask another's.
    window: list[float] = []

    def evaluate(ctx) -> Any:
        snapshot = ctx if isinstance(ctx, Snapshot) else Snapshot(ctx=ctx)
        values = [read(snapshot, o, f) for o, f in zip(operands, forms, strict=False)]
        reported = spec.fn(snapshot, *values) if atom.reads_scene else spec.fn(*values)
        held, score, metrics = _verdict(
            reported,
            key=key,
            op=op,
            threshold=threshold,
            head=measure,
            primary=atom.primary,
            window=window,
            smoothing=atom.smoothing,
        )
        metrics[named] = held
        return (held, score, metrics)

    leaf = Leaf(fn=evaluate, label="", about=about)
    if atom.smoothing:
        # Cleared between episodes, or the first steps of episode 2 are scored on
        # the last steps of episode 1 -- which reads as a task that starts
        # already half-satisfied and settles back down.
        leaf.on_reset = window.clear
    return leaf


def _published_reader(
    condition: Condition,
    operands: tuple[Operand, ...],
    doc: TaskDoc,
    catalog: Callable[[str], AssetRecord | None],
) -> Leaf | None:
    """A clause naming a state its operand *publishes*, rather than an atom.

    ``(<= (bond-fraction tape) 0.05)`` and ``(adhered tape)`` are both this: the
    quantity is computed by the entity as it steps and published, not measured
    from outside. Returning ``None`` when the head is not a published state lets
    :func:`bind` fall through to atom dispatch, which is the common case.
    """
    del doc
    if not operands:
        return None
    record = catalog(operands[0].asset) if operands[0].asset else None
    if record is None:
        return None

    wanted = condition.head.replace("-", "_")
    states = {s.name.replace("-", "_"): s for s in record.states}
    if wanted not in states and wanted not in {p.name.replace("-", "_") for p in record.published}:
        return None

    name = operands[0].name
    op, threshold = condition.op, condition.value
    declared = states.get(wanted)
    # Keyed per clause as the atom path keys it, so `about_key(*leaf.about)`
    # finds this leaf's verdict and `(bond-fraction tape_a)` beside
    # `(bond-fraction tape_b)` are two readings rather than one overwritten.
    about = (wanted, tuple(str(o.ref) for o in operands))
    named = about_key(*about)

    # A semantic state names the quantity it thresholds, and an entity may
    # publish that quantity rather than the word. `(adhered tape)` over
    # `StateSpec(name="adhered", measure="bond_fraction")` reads the bond
    # fraction and applies the asset's own >= 0.95 to it -- which is what
    # `measure` is *for*, and it was unreachable: the reader looked for the
    # state's name and nothing else, so an asset had to publish one number under
    # two names or its own words did not work. The state's name still wins where
    # an entity publishes it, because an entity that answers the word directly
    # has answered it.
    fallback = declared.measure.replace("-", "_") if declared is not None and declared.measure else ""

    def measure(ctx) -> Any:
        snapshot = ctx if isinstance(ctx, Snapshot) else Snapshot(ctx=ctx)
        owner = _published(snapshot, name)
        key = wanted if wanted in owner else fallback
        if key not in owner:
            raise BindError(
                f"{name!r} declares {condition.head!r} -- which its entity must publish as "
                f"{wanted!r}"
                + (f" or as its measure {fallback!r}" if fallback else "")
                + f" -- but published {sorted(owner)} this step. A declared state the "
                "entity never publishes scores every episode zero, silently."
            )
        value = owner[key]
        # A declared state carries its own comparison; a clause may state one
        # too, and the clause wins. Both being present is normal -- the asset
        # says when the tape counts as adhered, and a task may want a stricter
        # line for its own purposes without redefining the word.
        if op:
            held, score, metrics = _threshold(float(value), op, threshold, key=wanted)
        elif declared is not None and declared.op:
            held, score, metrics = _threshold(float(value), declared.op, declared.value, key=wanted)
        else:
            held, score, metrics = (bool(value), float(bool(value)), {wanted: value})
        metrics[named] = held
        return (held, score, metrics)

    return Leaf(fn=measure, label="", about=about)


def _published(snapshot: Snapshot, name: str) -> dict[str, Any]:
    """What one entity published this step, by name.

    ``extras`` holds the task's own entity *objects*, so this asks each for its
    states rather than indexing it. A plain mapping is accepted too, which is
    what the goal-semantics tests hand it -- the structural half of this module
    is meant to be testable without building an entity, and an entity is exactly
    the thing that needs a scene.
    """
    owner = snapshot.entities.get(name)
    if owner is None:
        return {}
    if isinstance(owner, Mapping):
        return dict(owner)
    reader = getattr(owner, "published_states", None)
    return dict(reader(snapshot)) if callable(reader) else {}


def _threshold(value: float, op: str, threshold: Any, *, key: str) -> tuple[bool, float, dict]:
    held = COMPARISONS[op](value, float(threshold))
    return (held, float(held), {key: value})


def _verdict(
    reported: Any,
    *,
    key: str,
    op: str,
    threshold: Any,
    head: str,
    primary: str = "",
    window: list[float] | None = None,
    smoothing: int = 0,
) -> Any:
    """Turn what a handler reported into a verdict and its metrics.

    ``smoothing`` takes the largest reading over the last N steps -- see
    ``AtomSpec.smoothing`` for the contact churn that forced it. Applied to the
    thresholded quantity only; the raw reading stays in ``metrics`` under its own
    name, so a run scored on a smoothed value still records the instantaneous
    one and the two can be compared.

    **A bare reading is keyed by the atom's declared** ``primary``, not by its
    name. Every handler in the corpus today returns a dict, so this branch is
    reached only by a newly minted atom -- and it is the one that fails quietly.
    The dict branch raises when the wanted key is absent, but keying a scalar by
    ``head`` puts it under ``winding_turns`` while every reader resolves the same
    clause to ``turns`` through
    :func:`~robosimstudio_bench.goals.reward.reading_key`, so ``Peak``,
    ``Integral`` and ``Progress`` would each raise on a key nobody reported.
    """
    if not isinstance(reported, dict):
        if key and primary and key != primary:
            raise BindError(f"atom {head!r} reports one value under {primary!r}, and {key!r} is not it")
        metrics = {primary or key or head: reported}
        value = reported
    else:
        metrics = dict(reported)
        chosen = key or next(iter(metrics), "")
        if chosen not in metrics:
            raise BindError(f"atom {head!r} reports {sorted(metrics)}, and {chosen!r} is not among them")
        value = metrics[chosen]
        key = chosen

    # `primary` last: a dict report has already resolved `key` to a real one of
    # its own, and only the bare-reading branch leaves it empty.
    named = key or primary or head
    if smoothing and window is not None:
        window.append(float(value))
        del window[:-smoothing]
        metrics[f"{named}.instant"] = value
        value = max(window)
        metrics[named] = value

    if op:
        held = COMPARISONS[op](float(value), float(threshold))
    else:
        held = bool(value)
    metrics.setdefault(named, value)
    return (held, float(held), metrics)
