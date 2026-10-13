# SPDX-License-Identifier: Apache-2.0
"""Placement: how hard is it pressing on that, and is it moving.

Arena's placement test, **decomposed into its two measures rather than taken as
one.** ``isaaclab_arena.tasks.predicates.spatial.object_on_destination`` answers
one boolean from two readings -- a contact force above a threshold and a linear
speed below another -- and a task that wants to say so writes two clauses.

That is deliberate and it is not pedantry. A single ``resting_on`` atom would
have to threshold one of its two readings and carry the other along, which makes
the name a lie about half of what it does; and when a run fails, "not resting on
it" is not a diagnosis where "0.4 N, wanted 1.0" and "moving at 0.3 m/s" are two
different faults with two different fixes. Composition is what the goal language
is *for*, so the measures stay measures.

**Both numbers are the calling task's, not these atoms'.** Arena uses 0.1 N for a
single-object place and 1.0 N for the multi-object sort, so a gate baked in here
would silently make one of its two task shapes measure the other's.

**The contact half is a scene measure, the speed half is not.** Whether two
things are pressing on each other is answered by the solver's contact set, which
belongs to the scene rather than to either object, so ``contact_force`` reads the
scene and its operands arrive as handles. A speed is a property of one body and
arrives as a velocity.

**Why this is stronger than a region test, and why neither replaces the other.**
BDDL's ``In`` is a box a *carried* object satisfies: measured on a passing
tool-sort episode, the environment terminated with the hammer 0.147 m up and
still in the fingers. A force from the destination that the object is not moving
under means it was let go and it landed -- there is no separate "released"
conjunct to add, because a held object accelerating downwards fails the speed
test and one held still against the floor is, physically, resting on it.

The converse hole is real too, and it is why the grocery goal wants both. The
contact test asks for a force *from* the destination, not for the object to be
*in* it. In that layout the milk carton starts 12 mm from the tote's outer wall,
so an arm that merely nudged it sideways left it leaning on the outside of the
bin, in contact and at rest, and the goal read satisfied -- reward 1.00 at step
240, before the graph had finished its grasp descent, with the carton still on
the table. Each half closes the other's hole.

**The force is smoothed, and that is a property of the instrument.** A settled
item's contact is not continuous: the solver's contact set churns, so the pair
drops out for up to six consecutive steps while nothing moves. See
``AtomSpec.smoothing`` for the measurement and for why this is declared on the
atom rather than written into a goal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.assetlib.semantics.atom import SpatialRelation
from robosimstudio.assetlib.semantics.atoms import ArgSpec, AtomCalibration, AtomSpec, register_atom
from robosimstudio.assetlib.semantics.handlers import handler
from robosimstudio.assetlib.vocabulary.measures import resting_on as _arena_resting_on

__all__ = ["FORCE_WINDOW", "SETTLED_SPEED", "UPRIGHT_SHARE", "Contact", "On"]

SETTLED_SPEED = 0.02
"""Below what speed a thing counts as having stopped [m/s].

The same number the cloth goals gate on, named once so a reading reported as
"at rest" and a relation saying "resting on" cannot mean two different things."""

FORCE_WINDOW = 10
"""Control steps the placement contact force is read over.

Comfortably above the longest measured dropout (6 steps) and short enough that an
item actually removed from its destination stops counting within 0.2 s at 50 Hz.
Lowering the threshold instead would not work: the dropouts are exactly 0.0, not
small.
"""


register_atom(
    AtomSpec(
        name="contact_force",
        summary="How hard one body is pressing on another [N].",
        args=(
            ArgSpec(name="object", reads="handle"),
            ArgSpec(name="against", reads="handle"),
        ),
        directions=("check", "observe"),
        returns=("force",),
        unit="N",
        cost="cheap",
        reads_scene=True,
        smoothing=FORCE_WINDOW,
        provenance=(
            "The contact half of isaaclab_arena.tasks.predicates.spatial."
            "object_on_destination, transcribed. The threshold is the calling task's."
        ),
        calibration=AtomCalibration(
            positives=("Five settled grocery items on a tote floor read a 2-5 N median over 300 steps."),
            negatives=(
                "An item on the table beside the tote reads no force from the tote at all. "
                "An item still descending in the fingers has not touched the floor yet."
            ),
            gate=None,
            bench="isaaclab_arena, grocery_packing/five_object",
            notes=(
                "No gate here on purpose. Arena uses 0.1 N for a single-object place and 1.0 N "
                "for the multi-object sort, so a gate baked in would silently make one of its "
                "two task shapes measure the other's."
            ),
        ),
    )
)


@handler(
    "contact_force",
    kinds=("", ""),
    substrates=("rigid", "rigid"),
    summary="The pairwise contact force between two rigid bodies' shapes.",
)
def _rigid_contact_force(snapshot, item: str, against: str) -> dict[str, Any]:
    """Arena's reading, with its thresholds turned off.

    ``resting_on`` is called for its measurement and not for its verdict, so both
    gates are opened: passing the task's numbers here would apply them twice,
    once invisibly, and the clause is where a threshold belongs.
    """
    _, measured = _arena_resting_on(
        snapshot.ctx,
        item,
        against,
        force_threshold=0.0,
        velocity_threshold=float("inf"),
    )
    return {"force": float(measured["force"])}


class Contact(SpatialRelation):
    """Whether two bodies are touching at all.

    **The boolean ``contact_force`` could not give.** Its reading is in newtons,
    and newtons are a fact about mass as much as about touching: the astro goals
    dropped a contact clause outright because no threshold worked across a scene
    holding a 30 g battery and a 900 g drill -- 0.2 N is a firm press for one and
    below the noise floor for the other. This asks the solver's contact set
    instead, which either has the pair in it or does not.

    The force rides along because it is free and because it is what a *diagnosis*
    wants: "touching, 0.03 N" and "touching, 6 N" are different situations, and a
    boolean alone would throw away the half that says which.

    Smoothed over :data:`FORCE_WINDOW`, for the reason every contact reading here
    is: the solver's contact set churns, dropping a settled pair for up to six
    consecutive steps while nothing moves.
    """

    name = "contact"
    summary = "Whether one body is touching another, and how hard [N]."
    args = (ArgSpec(name="object", reads="handle"), ArgSpec(name="against", reads="handle"))
    returns = ("touching", "force")
    directions = ("check", "observe")
    reads_scene = True
    smoothing = FORCE_WINDOW
    provenance = (
        "Minted here. The symbolic half of the placement pair contact_force already measures: "
        "same contact set, thresholded on impulse rather than on newtons so it means the same "
        "thing for a battery and for a drill."
    )
    calibration = AtomCalibration(
        positives="settled items in a tote read touching on every smoothed step over 300 steps at rest",
        negatives="an item carried above the tote, and one beside it on the bench, read 0.0 throughout",
        gate=None,
        bench="core/probes/contact.py MIN_IMPULSE (1e-6 N*s); a settle run over the astro bins",
        notes=(
            "The gate is structural rather than tuned: any contact the solver recorded counts, and "
            "MIN_IMPULSE exists to reject the numerical zero rather than to separate two populations."
        ),
    )


@Contact.implement(
    kinds=("", ""), substrates=("rigid", "rigid"), summary="Impulse-based touch test between two rigid bodies."
)
def _rigid_contact(snapshot, item: str, against: str) -> dict[str, Any]:
    if snapshot.ctx.contacts is None:
        return {"touching": 0.0, "force": 0.0}
    source = snapshot.ctx.handles.entity(item)
    destination = snapshot.ctx.handles.entity(against)
    contacts = snapshot.ctx.contacts
    touching = contacts.in_contact(source.collision_shapes, destination.collision_shapes)
    force = float(np.linalg.norm(contacts.pairwise_force(source.collision_shapes, destination.collision_shapes)))
    return {"touching": float(touching), "force": force}


class On(SpatialRelation):
    """Whether one body is resting on another: touching it, held up by it, still.

    **The composition placement.py argues against, packaged with the ingredient
    that makes it safe.** The objection to a monolithic ``resting_on`` was that
    thresholding one reading and hiding the other makes the name a lie and the
    failure undiagnosable -- so every ingredient is returned, and a false reading
    still says which half failed.

    What is new is ``up_frac``, and it closes the hole both existing halves
    share. ``contact_force`` asks for a force *from* the destination, not for the
    object to be held *up* by it: the milk carton starting 12 mm from the tote's
    outer wall could be nudged sideways, end up leaning on the outside of the
    bin, in contact and at rest, and read satisfied -- reward 1.00 at step 240,
    with the carton still on the table. A lean's impulse is horizontal. Support
    is the world-z share of it.

    ``implies = (Contact,)`` is a tested claim, not a comment: whatever is on
    something is touching it, and a bench where that failed would mean these two
    are measuring different contact sets.
    """

    name = "on"
    summary = "Whether a body rests on another: touching it, supported by it, and settled."
    args = (ArgSpec(name="object", reads="handle"), ArgSpec(name="support", reads="handle"))
    returns = ("on", "touching", "up_frac", "force", "speed")
    directions = ("check", "observe")
    reads_scene = True
    smoothing = FORCE_WINDOW
    implies = (Contact,)
    provenance = (
        "Minted here, composing contact_force + speed the way isaaclab_arena's "
        "object_on_destination does, plus the support test neither has. Arena's own thresholds "
        "stay the task's; what is baked in is the geometry of 'held up by', which is not a "
        "number a task should be choosing."
    )
    calibration = AtomCalibration(
        positives="items settled in a tote: up_frac 0.85-1.00, speed under 0.01 m/s",
        negatives=(
            "an item leaning on a tote's outer wall reads up_frac under 0.2 while contact and speed "
            "both pass -- the exact false positive that scored 1.00 at step 240"
        ),
        gate=0.5,
        bench="a settle run over the astro bins; the leaning case staged as a two-box fixture",
        notes="up_frac gates at 0.5: more of the support's push is upward than sideways.",
    )


UPRIGHT_SHARE = 0.5
"""How much of the support's push must be upward for ``on`` to hold.

Half, and the number is a shape rather than a tuning: above it the support is
carrying the object, below it the object is leaning against something. The
populations it separates are three-quarters apart (0.85-1.00 against under 0.2),
so nothing here is balanced on the exact value."""


@On.implement(
    kinds=("", ""), substrates=("rigid", "rigid"), summary="Touching, supported and settled, for two rigid bodies."
)
def _rigid_on(snapshot, item: str, support: str) -> dict[str, Any]:
    """Delegates the touching half rather than restating it.

    ``Contact.measure`` is called for the same reason the goal binder calls it:
    two implementations of one question is how the two come to disagree, and the
    ``implies`` claim above would then be untestable rather than merely false.
    """
    touch = Contact.measure(snapshot, item, support, substrates=("rigid", "rigid"))
    if snapshot.ctx.contacts is None:
        return {"on": 0.0, "touching": 0.0, "up_frac": 0.0, "force": 0.0, "speed": 0.0}
    source = snapshot.ctx.handles.entity(item)
    destination = snapshot.ctx.handles.entity(support)
    impulse = np.asarray(
        snapshot.ctx.contacts.pairwise_impulse(source.collision_shapes, destination.collision_shapes),
        dtype=np.float64,
    )
    magnitude = float(np.linalg.norm(impulse))
    up_frac = float(impulse[2] / magnitude) if magnitude > 0.0 else 0.0
    speed = float(snapshot.ctx.linear_speed(item))
    on = bool(touch["touching"] and up_frac > UPRIGHT_SHARE and speed < SETTLED_SPEED)
    return {
        "on": float(on),
        "touching": touch["touching"],
        "up_frac": up_frac,
        "force": touch["force"],
        "speed": speed,
    }
