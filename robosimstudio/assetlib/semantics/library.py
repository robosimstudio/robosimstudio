# SPDX-License-Identifier: Apache-2.0
"""The feature kinds this stack ships, and the contracts their handlers owe.

Every kind here was **earned**: something in the shipped corpus already computes
it, or a design fixture demonstrably needs it. None is here because an ontology
felt incomplete. That rule is the difference between a vocabulary and a taxonomy,
and it is worth being strict about -- the failure mode of the alternative is a
list of properties nobody can measure, asserted by committee.

Registration is what earns a kind its *generality*, not its existence. An asset
may annotate a feature of any kind it likes, and an unregistered kind resolves
and works; what it cannot do is claim a contract, because none is written. The
promotion path is the same as an atom's: a bespoke kind used twice is a candidate
for registration, with its contract and its tests.

The contracts are prose here and mechanical in the test suite: each string
becomes a check generated over the handlers that dispatch on the kind, so a
contract that is written is a contract that is enforced, and a new asset carrying
an existing kind inherits the checks for free.
"""

from __future__ import annotations

from .kinds import FeatureKind, register_kind

__all__ = ["KIND_NAMES"]


# --------------------------------------------------------------- rigid geometry
register_kind(
    FeatureKind(
        name="grasp_box",
        summary="The graspable part of a body, as an oriented box a gripper can close on.",
        payload={"half": "half-extents in the body frame [m]", "center_local": "box centre in the body frame [m]"},
        points=("center_local",),
        optional={"long_axis": "index of the box's long axis", "handle_frac": "slide along the long axis [0-1]"},
        contracts=(
            "reports the box AND whether a given gripper actually fits it -- an object wider than "
            "the jaws at its centroid cannot be grasped there, and saying so is the point",
            "derived from the mesh, never hand-authored: the compiler already computes it",
        ),
        substrates=("rigid", "articulated"),
        provenance="Already computed by the compiler and served by sim.get_object_obb.",
    )
)

register_kind(
    FeatureKind(
        name="region",
        summary="A volume something can be inside or resting on, anchored to a frame.",
        payload={"xy_ranges": "x0, y0, x1, y1 in the frame [m]"},
        optional={
            "z_range": "low and high above the frame origin [m]; BDDL's 0 to 0.15 when omitted",
            "frame": "part of this asset whose pose the region rides, empty for the asset itself",
        },
        contracts=(
            "containment returns a FRACTION, always -- a garment is never entirely inside a bin, "
            "and a rigid body's 0-or-1 is the easy case of that question, not a different one",
            "a region on a moving frame moves with it: a drawer's interior is only meaningful "
            "because it travels with the drawer",
        ),
        provenance="BDDL's region convention, matched to the digit by RegionSpec.",
    )
)

register_kind(
    FeatureKind(
        name="loop",
        summary="A closed opening something rod-shaped can pass through.",
        payload={
            "center": "disc centre in the body frame [m]",
            "normal": "disc normal in the body frame",
            "inner_r": "clear radius a rod must fit inside [m]",
        },
        points=("center",),
        directions=("normal",),
        optional={
            "thickness": "material thickness through the loop [m]",
            "enclosure_tolerance": "calibrated radial allowance added to inner_r by `encloses` [m]",
        },
        contracts=(
            "threading is tested GEOMETRICALLY -- the rod's axis segment crossing the loop's disc "
            "-- never by attributing contacts to a mesh subregion, which is brittle and can be "
            "faked by an object merely resting against the rod",
        ),
        substrates=("rigid", "articulated"),
        provenance="Designed for the hanging fixtures; generalises scissors, mug handles, keyrings.",
    )
)

register_kind(
    FeatureKind(
        name="rod",
        summary="A straight cylindrical span something can hang on or wind around.",
        payload={"axis": "two endpoints in the body frame [m]", "radius": "cylinder radius [m]"},
        # Two points, not a direction: a rod is a finite segment and hanging past
        # its end is not hanging, so both ends have to travel with the body.
        points=("axis",),
        contracts=("the axis is a finite segment, not an infinite line -- hanging past the end is not hanging",),
        substrates=("rigid", "articulated"),
        provenance="The cable pegs and the hanger fixture.",
    )
)

register_kind(
    FeatureKind(
        name="cavity",
        summary="A bore something is inserted into, with a mouth, an axis and a depth.",
        payload={
            "axis": "insertion direction in the body frame",
            "mouth": "mouth-plane point in the body frame [m]",
            "depth": "how deep the bore runs [m]",
        },
        optional={
            "half_extents": "mouth half-extents, for the lateral guard [m]",
            "mated": "fully seated pose",
            "tip_offset": "how far along the inserted part's own +z its tip reaches [m]",
        },
        points=("mouth",),
        directions=("axis",),
        contracts=(
            "depth counts only inside the mouth's own half-diagonal -- without that guard a plug "
            "pushed past the port BESIDE the bore reads as inserted",
            "tilt and roll are reported, never gated: nobody has calibrated a tilt threshold and "
            "the bore itself constrains both at depth",
        ),
        substrates=("rigid", "articulated"),
        provenance="Transcribed from the USB-C and RJ45 reference geometry.",
    )
)


# ----------------------------------------------------------------- deformables
register_kind(
    FeatureKind(
        name="set",
        summary="A named selection of particles, resolved on the rest layout.",
        payload={},
        optional={
            "select": "a selector recipe (tier 2)",
            "resolver": "registered code, by versioned key (tier 3)",
            "indices": "a baked local index list (tier 1) -- must pin the mesh hash",
        },
        contracts=(
            "resolved ONCE against the rest layout, never against the settled drape -- a set "
            "recorded from a drape follows whatever the drape did, and 'the tab' stops being the tab",
            "anything referencing discretisation pins the mesh hash, or it rots silently on a re-mesh",
        ),
        substrates=("cloth",),
        provenance="The cloth entity's existing particle-set mechanism, serialised.",
    )
)

register_kind(
    FeatureKind(
        name="panel",
        summary="A region of a sheet that one fold carries, paired with where it lands.",
        payload={},
        optional={
            "moving": "the selection carried",
            "base": "the strip it lands on",
            "crease": "the fold line",
            "forgiveness": "multiplier on the layer separation forgiven for THIS crease",
            "grasp": "the strip a hand takes to carry it",
            "grasp2": "the second hand's strip, on a bimanual crease",
        },
        contracts=(
            "a crease's base is the MIRROR of its own moving panel and nothing else -- declared as "
            "'everything on the far side' instead, a first fold counts the whole other half as its "
            "landing zone and a legitimate second fold then reads as drag against the first",
            "forgiveness is PER CREASE and multiplies a separation derived from the fabric, never a "
            "number written down: a crease landing its flap on an already-folded stack rests higher "
            "than a planar mirror expects, and a global dial would flatter every fold at once",
        ),
        substrates=("cloth",),
        provenance="The shirt's six panels, measured; the towel's two.",
    )
)

register_kind(
    FeatureKind(
        name="face",
        summary="A flat, oriented face of a body that a sheet is bonded to or worked against.",
        payload={
            "center": "face-box centre in the frame the asset states its layout in [m]",
            "half": "half-extents of that box [m], the face being its +z side",
            "quat": "the box's orientation, xyzw",
        },
        points=("center",),
        contracts=(
            "the ORIENTATION is the load-bearing field, not the centre: a bond opens along the "
            "face's own normal, so a face declared without its rotation is a face that reads as "
            "world z and breaks its bonds in the wrong direction the moment the part is tilted",
            "half-extents describe the box the face is the +z side of, so a reader can place a "
            "sheet on it without knowing how the part was modelled",
        ),
        substrates=("rigid", "static"),
        provenance=(
            "The phone the film suite peels from, whose glass leans back thirty degrees in its "
            "cradle; the flat panel it replaced needed no rotation and so never asked for one."
        ),
    )
)

register_kind(
    FeatureKind(
        name="bond_field",
        summary="A field of breakable adhesive bonds between a sheet and what it is stuck to.",
        payload={},
        optional={"stiffness": "penalty spring constant", "break_distance": "opening displacement that breaks [m]"},
        contracts=(
            "bonds break on OPENING displacement, not total: tape pressed along its surface does "
            "not unstick, and only being pulled off it peels",
            "a broken bond stays broken -- the peel front travels on its own, which is the entire "
            "reason to model adhesion this way rather than scripting it",
        ),
        substrates=("cloth",),
        provenance="AutoPeel's discrete cohesive-zone model; the tape and film suites.",
    )
)


register_kind(
    FeatureKind(
        name="blade",
        summary="A cutting edge: a straight segment on a rigid body that parts what it is drawn along.",
        payload={
            "origin": "the edge's start, in the body frame [m]",
            "direction": "unit direction the edge runs in",
            "length": "how long the edge is [m]",
        },
        contracts=(
            "the edge is a SEGMENT and not a ray: a law integrating along it must reject anything "
            "outside [0, length], or a blade cuts things the far side of the room it never touched",
            "sharpness is NOT geometry. A blade's collider is a box with a real thickness -- an "
            "infinitely sharp edge is a contact no solver resolves well -- so whether it parts what "
            "it touches is a force threshold the owning field applies, and the lateral tolerance "
            "around the edge is a fact about the technique rather than the width of a slit",
        ),
        substrates=("rigid",),
        points=("origin",),
        directions=("direction",),
        provenance="The box cutter's edge; the box_decant suite's severance field integrates along it.",
    )
)


# --------------------------------------------------------------------- lines
register_kind(
    FeatureKind(
        name="axis",
        summary="A directed line something is measured about -- a winding axis, a bore axis.",
        payload={"origin": "a point on the axis [m]", "direction": "unit direction"},
        points=("origin",),
        directions=("direction",),
        contracts=("a winding about an axis is SIGNED, so a figure-eight cancels and a hank beside it scores zero",),
        provenance="The cable pegs' vertical axes.",
    )
)

register_kind(
    FeatureKind(
        name="landmark",
        summary="A named point fixed in a rigid or articulated body's local frame.",
        payload={"position": "point in the body frame [m]"},
        points=("position",),
        contracts=("the point follows both the translation and rotation of the body that carries it",),
        substrates=("rigid", "articulated"),
        provenance="Introduced to score an inserted tool by its functional tip instead of its centre of mass.",
    )
)


# ------------------------------------------------------------------- staging
register_kind(
    FeatureKind(
        name="anchor",
        summary="A place the scene may hold at episode start, until something takes over.",
        payload={},
        optional={
            "at": "where to hold it",
            "grip": "finger command while holding [m]",
            "release_on": "what ends the hold",
        },
        contracts=(
            "an anchor is an INIT mechanism, not a goal condition: it shapes the state an episode "
            "begins in and must be verified to have taken hold before the episode counts as started",
        ),
        provenance="The cable's anchor grasp and the tape's pull-tab pin, generalised.",
    )
)

register_kind(
    FeatureKind(
        name="template",
        summary="A task-family geometric payload an asset carries: a fold sequence, a mating nominal.",
        payload={},
        optional={"steps": "ordered entries", "of": "what family this belongs to"},
        contracts=(
            "a template is DATA the goal references, never an order of operations -- which fold "
            "happens first is the policy's business, and a goal stated over a template is stated "
            "over the state it produces",
        ),
        provenance="SpeedFolding's three-crease t-shirt instruction; the insertion nominals.",
    )
)


# ------------------------------------------------------------- jointed objects
register_kind(
    FeatureKind(
        name="joint",
        summary="One named joint of a jointed object -- a drawer's slide, a door's hinge.",
        optional={"joint": "the joint's name in the entity's handle, when the feature is named otherwise"},
        contracts=(
            "read from where the joint's two bodies are, not from a joint coordinate a snapshot does not carry",
            "an opening is a fraction of the joint's own travel, so a drawer and a door answer the same question",
        ),
        substrates=("articulated",),
        provenance="SimReady's cabinets: ten named drawer and door joints per asset.",
    )
)


KIND_NAMES: tuple[str, ...] = (
    "anchor",
    "axis",
    "blade",
    "bond_field",
    "cavity",
    "face",
    "grasp_box",
    "joint",
    "landmark",
    "loop",
    "panel",
    "region",
    "rod",
    "set",
    "template",
)
"""Every kind this module registers, sorted.

Pinned by a test. The set is *open* -- an asset may annotate any kind it likes --
but the shipped ones are a deliberate list, and adding to it is an act with a
contract attached rather than a convenience.

A kind that only one kind of *thing* can carry is not here: it is registered by
that thing's domain pack, beside the layout it is stated on, and the pack keeps
its own ``KIND_NAMES`` for the same test to read. The registry scans the packs,
so ``list_kinds()`` is this tuple plus theirs.
"""
