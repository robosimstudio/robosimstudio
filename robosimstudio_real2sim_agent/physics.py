# SPDX-License-Identifier: Apache-2.0
"""Whether a twin's physics is real: its own routine replayed with the robot changed, and what moved read off it.

A twin's picture can match its clip while the physics does not hold: a thing
that starts inside a finger and is pushed out, a "grasp" that is the hand
wedging the thing at 700 N, a lift that works with the fingers held open, a
grip that only holds at a friction no rubber has. None of that shows in a
still or a take. So the routine is played again, five ways (:data:`MODES`):

- ``written``: as it is, every step logged (what moved, the force each hand
  puts on it, overlaps);
- ``frozen``: every arm and finger target held where it started. Nothing may
  move: what does is pushed by something other than the robot, or started
  inside something (a few millimetres is a note, more fails; an attachment
  is the contact line's FAIL);
- ``open``: the fingers held fully open, the arms as written. A pinch task
  fails; one that still succeeds is carried by something else (a support, a
  hook, two arms squeezing), and the report has to say what;
- ``built``: the robot as built -- every contact friction at most
  :data:`REAL_MU`, the arm's torque capped at the real arm's
  (:func:`real_arm_torque`). The routine still finishes;
- ``heavy``: as built, and every handled thing twice as heavy: the grasp's
  margin, a warning when it fails rather than a fault.

**One door.** Every routine moves the robot by writing ``joint_target_q``:
the arm panels' IK and the gripper write it just before the page steps, a
routine that steps the scene itself writes it through ``sim.step``. Both
step through ``EnvBase._step_once``, so :class:`Replay` holds the targets in
a pre-step hook there and logs in a post-step hook, and never needs to know
how a routine was written. Code that calls ``pipeline.step()`` itself goes
round that door and is not held: a routine is not written that way, and a
settle a material runs while the scene is built is not a routine's step.

**Free flight** is judged on nothing being within the contact margin, not on
the contact force: contacts are read after the last substep, and one that
comes and goes inside a step reads as none, while the step's velocity change
still carries it.

**Contacts are MuJoCo's, and a pass of their own.** In the default
arrangement free rigid things go to a VBD island coupled to the robot, and
there the fingers' contacts with them are resolved by VBD and never reach
MuJoCo's list. For those things a collision pass of the replay's own runs
after every step on the state the step left (read-only: nothing is solved
from it), so their free flight is judged and the fingers touching them is
seen; only the force is not read there.

**Two more replays.** ``drop`` holds the
fingers open from early in the written run's carry (where a thing it lifted first reached 90% of its height): what the fingers were
carrying falls, and what still rides along with nothing touching it is
carried by something that is not a contact. ``plain``, the one a routine
cannot tell from a play by its hooks (its stepping is capped by a wrapper on
the sim's own run and step), plays the routine with nothing hooked in and reads at the routine's own yields: a thing that
flies untouched there is carried another way when nobody is watching (a
different end is only a note: the sim is not deterministic).

**The model's own numbers.** Where the scene's settle (before the first
step) moved a thing from where it was put says what it was put inside; a
gravity that is not the Earth's, or a density no handled object has, is
read off the model.
"""

from __future__ import annotations

import functools
import math
import re
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import newton
import numpy as np
import warp as wp

from robosimstudio.core import tuning
from robosimstudio.engines.pose import compose_many

MODES = ("written", "frozen", "open", "built", "heavy", "drop", "plain")

REAL_MU = 1.0
"""The most friction a real contact here has: rubber pads on plastic, ceramic or metal run 0.5 to 1.0."""

OVERRUN = 0.1
"""How far past the written run's length a replay whose routine is still going may run, as a share of it."""

SAMPLE_EVERY = 5
"""Every how many steps a replay reads the jaws and the particles (contacts and poses it reads every step)."""

HEAVY = 2.0
"""How much heavier ``heavy`` makes every handled thing."""

STOPPED = "stopped before it ended (a reset, a save or Stop)"
"""The outcome of a replay the page stopped: not a reading, so ``./ctl physics`` plays it again."""

PLAIN_LONG = "still going, played with nothing watching, when the written replay had long ended"
"""plain's cut: a reading (the routine does not end unwatched as it does replayed), not a replay to play again."""

PLAIN_SLOW = "cut on the page's clock while still within the written run's steps"
"""plain stopped by ``./ctl physics``'s wall clock before it had taken more steps than the written run: a slow page (a
busy GPU), not a routine that does not end -- read as not read."""

FAILED = "the replay itself failed"
"""The start of the outcome of a replay that failed in its own code (not the routine's): not a reading either."""

SETTLE_MM = 2.0
"""How far a thing may move with the robot frozen: a resting contact settles by a fraction of a millimetre."""

DRAWN_ONLY_MM = (3.0, 10.0)
"""How far a handled thing's drawn shapes that do not collide may reach outside its colliders before it is a NOTE, and
before it FAILs [mm]: a leaf or a label drawn and never touched passes through whatever it meets."""

END_MM, END_DEG = 50.0, 30.0
"""How near a replay must end each handled thing to where the written run left it [mm, deg]."""

PUSHED_MM = 5.0
"""How far a rigid thing may move with the robot frozen and still be a wobble for the report (a key in a tight socket,
a bottle loose in a tray) rather than a push [mm], at the default :data:`SETTLE_MM`; ``--settle-mm`` scales it with
itself. A move under the settle bar is none at all. A thing tied to the robot or kinematic is the contact line's FAIL
whatever it moves; the frozen line only names it (a NOTE). Honest twins moved up to 3.1 mm, and a start inside something or a part driven by a motor
8.6 mm and more."""

# The real arms' joint torque limits [Nm], from their makers' data sheets, by a word in the robot's name or its own
# joints' names: a robot file's own numbers are a claim (the bimanual YAM's are right; a UR5e built for a twin gave
# its wrists 150). A Universal Robots arm whose size the names do not give is left uncapped rather than guessed.
ARM_TORQUE = {
    "ur3": (56.0, 56.0, 28.0, 12.0, 12.0, 12.0),
    "ur5": (150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
    "ur10": (330.0, 330.0, 150.0, 56.0, 56.0, 56.0),
    "ur16": (330.0, 330.0, 150.0, 56.0, 56.0, 56.0),
    "yam": (27.0, 27.0, 27.0, 10.0, 10.0, 10.0),
    "panda": (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
    "fr3": (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
}

GRAVITY = 9.81
"""Earth's [m/s²]: a scene whose gravity differs by more than 2% is read as a shortcut."""

FLIGHT_MM = 5.0
"""A thing is tested for free flight once it has moved this far [mm]."""

FLIGHT_STEPS = 10
"""Steps free of every contact and not falling at g that fail the run: a carrier that is not a contact. Fewer are a
note (a contact that comes and goes between reads)."""

DENSITY = (30.0, 12000.0)
"""The densities a handled thing is read against [kg/m³]: lighter than expanded polystyrene or heavier than lead is a
note for the report (and a heavy replay of a 2 g box proves nothing)."""

OVERLAP_MM = 1.0
"""How deep a thing may sit in something at the reset [mm]: a resting contact overlaps by a fraction of this; 1 mm is
the bar grasp datasets hold a valid grasp's penetration to."""

CARRIED_MM = 20.0
"""A thing lifted this far off its start [mm] was carried: what ``drop`` releases."""

HAND_FORCE = {
    "robotiq_2f85": ("a Robotiq 2F-85", 235.0),
    "panda_hand": ("a Franka hand", 140.0),
    "yam": ("a YAM's linear gripper", 50.0),
}
"""The most each real hand squeezes [N]: the 2F-85's datasheet 235 N; the Franka hand's 140 N peak (70 N continuous);
the YAM's 50 N, the clamp i2rt's driver holds a blocked gripper at. A finger force well past it is a hand no real one is."""

HAND_FORCE_MARGIN = 1.25
"""How far past the real hand's most the fingers may press before grip fails (a contact spike is not a squeeze)."""

REACH_MM = 5.0
"""The most the two shapes' contact margins widen "touching" [mm]: a solver acts on them that far apart, but a box
hovering a centimetre off a finger is not held by it."""

RELEASE_WINDOW = 25
"""Steps after drop's release that say what holds a thing: the robot must touch it in most of them to be bearing it."""

NEAR_MM = 2.0
"""Within this gap [mm] a pair the replay's own collision pass reports is touching: Newton reports every pair inside
the contact margin, which can be centimetres."""

ELASTIC_MM = 10.0
"""How far an elastic part (a stem, a spring) may bend or sag at the reset with the robot frozen before it reads as pushed
[mm]."""

SINK_MM = 5.0
"""A rigid thing that, with the robot frozen, only sinks into what it rests on (no more than this [mm], straight down) is
settling into a soft contact, a drift for the report rather than a push."""

_SHAPE_PROPERTIES = tuning.ShapeWriter.flags  # newton.ModelFlags, as the tuning writers name them
_BODY_INERTIAL = tuning.BodyWriter.flags
_JOINT_DOF = tuning.JointWriter.flags


# --------------------------------------------------------------------------- what the scene holds


def robot_name(sim: Any) -> str:
    spec = getattr(getattr(sim.scene, "embodiment", None), "spec", None)
    return str(getattr(spec, "name", "") or "")


def _robot_bodies(sim: Any) -> set[int]:
    """Every body of the robot, shapeless links (a flange, a tool frame) included."""
    m = sim.model
    owners = {
        int(b) for b in np.unique(m.shape_body.numpy()[np.asarray(sim.scene.handles.robot_shapes, dtype=np.int64)])
    }
    own = getattr(getattr(sim.scene, "embodiment", None), "bodies", None)
    return owners | ({int(b) for b in own} if own is not None else set())


def _robot_joint_labels(sim: Any) -> list[str]:
    """The labels of the robot's own joints: a thing's joint named after a robot must not name the arm."""
    m = sim.model
    labels = list(getattr(m, "joint_label", []) or [])
    if not labels:
        return []
    robot = _robot_bodies(sim)
    child = m.joint_child.numpy()
    return [str(labels[j]) for j in range(min(len(labels), len(child))) if int(child[j]) in robot]


def _match_arm(words: str) -> tuple[float, ...] | None:
    for key in sorted(ARM_TORQUE, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){key}e?(?![a-z0-9])", words.lower()):  # ur5e is a UR5; ur30, fur3, yaml are not
            return ARM_TORQUE[key]
    return None


def _hand_kind(words: str) -> str | None:
    if re.search(r"2f.?85|robotiq|driver_joint|follower_joint", words):
        return "robotiq_2f85"
    if re.search(r"panda|franka|fr3", words):
        return "panda_hand"
    if re.search(r"(?<![a-z0-9])yam", words):
        return "yam"
    return None


def real_hand(sim: Any, arm: Any, name: str = "") -> str | None:
    """Which real hand an arm's fingers are, by the names its model uses, or ``None``: a Robotiq 2F-85 by its
    driver/coupler/follower linkage or its name, a Franka hand, a YAM's linear gripper. The arm's own joints (those
    naming it) decide when there are any, so a pair with two different hands reads each."""
    every = [str(x).lower() for x in _robot_joint_labels(sim)]

    def names_(x: str, arm_name: str) -> bool:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(arm_name.lower())}(?![a-z0-9])", x))

    arms = [str(a) for a in getattr(getattr(sim.scene, "handles", None), "arms", {}) or {}]
    own = [x for x in every if name and names_(x, name)]
    if own:  # and the joints that name no arm at all (a gripper's driver joints)
        own += [x for x in every if not any(names_(x, a) for a in arms)]
    if not own:  # no joint names this arm: the robot's names decide only when they name a single kind of hand
        kinds = {_hand_kind(x) for x in every} - {None}
        if len(kinds) > 1:
            return None
    labels = " ".join(own or every) + " " + robot_name(sim).lower()
    hand = str(getattr(getattr(arm, "gripper_spec", None), "name", "") or getattr(arm, "gripper", "") or "").lower()
    return _hand_kind(hand + " " + labels)


def real_arm_torque(sim: Any, arm: Any = None) -> tuple[float, ...] | None:
    """The real arm's torque limits, per joint from the base, or ``None`` for a robot not in :data:`ARM_TORQUE`.

    Given an arm handle, read off that arm's own joints first, so two kinds of arm in one scene each keep their own;
    the robot's name and all its joints answer when the arm's joints name no model.
    """
    if arm is not None:
        labels = list(getattr(sim.model, "joint_label", []) or [])
        qd0 = sim.model.joint_qd_start.numpy()
        dofs = set(arm.arm_dofs)
        mine = [str(labels[j]) for j in range(min(len(labels), len(qd0))) if int(qd0[j]) in dofs]
        hit = _match_arm(" ".join(mine))
        if hit is not None:
            return hit
    return _match_arm(f"{robot_name(sim)} {' '.join(_robot_joint_labels(sim))}")


def _by_coord(sim: Any) -> bool:
    """Whether ``joint_target_q`` is laid out by coordinates (Newton's opt-in) rather than by DOFs.

    The two differ for any scene with a free body (7 coordinates, 6 DOFs).
    """
    m, n = sim.model, int(sim.control.joint_target_q.shape[0])
    return n == m.joint_coord_count and n != m.joint_dof_count


def layout(sim: Any) -> dict[str, np.ndarray]:
    """Where each arm's, each finger's and every robot joint's target sits in ``joint_target_q`` (:func:`_by_coord`),
    the matching DOF rows, and the fingers' open targets."""
    m, h = sim.model, sim.scene.handles
    by_coord = _by_coord(sim)
    arm, grip, opened = [], [], []
    for a in h.arms.values():
        arm += list(a.arm_coords if by_coord else a.arm_dofs)
        g = list(a.gripper_coords if by_coord else a.gripper_dofs)
        grip += g
        # a mirrored pair opens at (+w, -w), not at 0; a spec that names fewer open values than it drives fingers
        # repeats them, so every held row has its value
        opened += list(np.resize(a._finger_limits(len(g))[0], len(g)))
    # every target the robot has, a torso, a rail or a head too: `frozen` holds them all. Newton 1.7 says where each
    # joint's targets start (joint_target_q_start); its start arrays end in a sentinel, so [j + 1] always exists
    robot = _robot_bodies(sim)
    child, qd0 = m.joint_child.numpy(), m.joint_qd_start.numpy()
    starts = getattr(m, "joint_target_q_start", None)
    starts = starts.numpy() if starts is not None else (m.joint_q_start.numpy() if by_coord else qd0)
    rows: list[int] = []
    dofs: list[int] = []
    for j in range(len(child)):
        if int(child[j]) in robot:
            rows += list(range(int(starts[j]), int(starts[j + 1])))
            dofs += list(range(int(qd0[j]), int(qd0[j + 1])))
    grip_dofs = [d for a in h.arms.values() for d in a.gripper_dofs]
    return {
        "robot": np.asarray(sorted(set(rows) | set(arm) | set(grip)), dtype=np.int64),
        "robot_dofs": np.asarray(
            sorted(set(dofs) | {d for a in h.arms.values() for d in a.arm_dofs} | set(grip_dofs)), dtype=np.int64
        ),
        "grip_dofs": np.asarray(grip_dofs, dtype=np.int64),
        "arm": np.asarray(arm, dtype=np.int64),
        "grip": np.asarray(grip, dtype=np.int64),
        "open": np.asarray(opened, dtype=np.float64),
    }


def things(sim: Any) -> dict[str, dict[str, Any]]:
    """Every dynamic thing that is not the robot: its bodies and its shapes."""
    out = {}
    for name, e in sim.scene.handles.objects.items():
        bodies = [int(b) for b in (e.bodies or (e.body,)) if int(b) >= 0]
        if bodies and not e.static:
            out[name] = {"bodies": bodies, "shapes": np.asarray(e.shapes, dtype=np.int64)}
    return out


def fingers(sim: Any) -> dict[str, np.ndarray]:
    return {
        a: np.asarray(sorted({int(s) for g in arm.finger_groups for _, sh in g for s in sh}), dtype=np.int64)
        for a, arm in sim.scene.handles.arms.items()
    }


def unread_things(sim: Any, names: list[str]) -> dict[str, str]:
    """The things whose contacts the contact probe cannot see, each with why.

    The probe reads MuJoCo's contact list. In a coupled scene a thing another island owns (the VBD the default
    arrangement gives free rigid bodies) has its contacts resolved there, so it reads as touching nothing; with no
    probe at all, nothing is read. A thing whose island cannot be placed is counted unread too: a wrong "read"
    is what turns into a false accusation of free flight.
    """
    probe = getattr(sim, "contacts", None)
    if probe is None:
        return dict.fromkeys(names, "the scene has no MuJoCo contact list")
    program = getattr(getattr(sim, "pipeline", None), "program", None)
    if program is None or not program.coupled:
        return {}
    out = {}
    for name in names:
        try:
            solver = program.solver_for(name)
        except (KeyError, AttributeError):  # no island owns the name, or a coupler with no per-island lookup
            out[name] = "its solver could not be placed"
            continue
        if solver is not probe.solver:
            out[name] = f"stepped by {type(solver).__name__}"
    return out


def _hidden_colliders(sim: Any) -> dict[int, str]:
    """Static colliders nothing draws, by shape: a prop a camera cannot see that a thing can still rest on.

    A shape that collides and is not drawn, owned by no thing and not the robot's, in a group with no drawn shape: a
    static's group is its name (a label up to its first ``/``, less a ``_collision_NN`` proxy suffix), or its body when
    it was welded in place. A visible table whose collider is a separate invisible box is not hidden; a plane (the
    ground) is not a prop.
    """
    m = sim.model
    flags, sb = m.shape_flags.numpy(), m.shape_body.numpy()
    kind = m.shape_type.numpy()
    labels = list(getattr(m, "shape_label", None) or [""] * len(flags))
    visible, collide = int(newton.ShapeFlags.VISIBLE), int(newton.ShapeFlags.COLLIDE_SHAPES)
    plane = int(newton.GeoType.PLANE)
    owned = {int(x) for t in things(sim).values() for x in t["shapes"]}
    robot = {int(x) for x in np.asarray(sim.scene.handles.robot_shapes, dtype=np.int64)}
    robot_bodies = {int(b) for b in _robot_bodies(sim)}

    def group(i: int) -> str:
        if int(sb[i]) >= 0:
            return f"body {int(sb[i])}"
        return re.sub(r"_collision_\d+$", "", str(labels[i]).split("/")[0])

    drawn = {group(i) for i in range(len(flags)) if int(flags[i]) & visible}
    out = {}
    for i in range(len(flags)):
        if not int(flags[i]) & collide or int(flags[i]) & visible or int(kind[i]) == plane:
            continue
        if i in owned or i in robot or int(sb[i]) in robot_bodies:
            continue
        key = group(i)
        if key not in drawn:
            out[i] = (str(labels[i]).split("/")[0] or key) if int(sb[i]) < 0 else (str(labels[i]) or key)
    return out


def _shape_volume(m: Any, s: int, tables: tuple | None = None) -> float:
    """A collision shape's volume [m³]: Newton's own mass of it at a density of 1, a hollow shell's by its wall.
    ``tables`` is ``(shape_type, shape_scale, shape_source, shape_is_solid, shape_margin)`` already on the host, for a
    caller asking about many."""
    if tables is None:
        solid, thick = getattr(m, "shape_is_solid", None), getattr(m, "shape_margin", None)
        tables = (
            m.shape_type.numpy(),
            m.shape_scale.numpy(),
            getattr(m, "shape_source", None),
            solid.numpy() if hasattr(solid, "numpy") else solid,
            thick.numpy() if hasattr(thick, "numpy") else thick,
        )
    types_, scales, sources, solid, thick = tables
    src = (sources or [None] * (s + 1))[s]
    kw = {}
    if solid is not None:
        kw["is_solid"] = bool(np.asarray(solid)[s])
    if thick is not None:
        kw["thickness"] = float(np.asarray(thick)[s])
    try:
        mass, _com, _inertia = newton.geometry.compute_inertia_shape(
            int(types_[s]), wp.vec3(*(float(v) for v in scales[s][:3])), src, density=1.0, **kw
        )
        return max(float(mass), 0.0)
    except Exception:  # a shape Newton gives no mass to (a plane, a field): no volume to weigh it by
        return 0.0


def _round_axis(m: Any, shapes: list[int], tables: tuple | None = None) -> list[float] | None:
    """The body-frame axis a thing is round about -- every collider a cylinder, capsule or cone along one line, or a
    sphere centred on it -- or ``None``. A bottle spun about its own axis ends the same."""
    g = newton.GeoType
    kind, xf, flags = tables or (m.shape_type.numpy(), m.shape_transform.numpy(), m.shape_flags.numpy())
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    axis, point, spheres = None, None, []
    for shape in shapes:
        x = int(shape)
        if not int(flags[x]) & collide:
            continue
        k, p, q = int(kind[x]), np.asarray(xf[x, :3], dtype=np.float64), xf[x, 3:]
        if k == int(g.SPHERE):
            spheres.append(p)
            continue
        if k not in (int(g.CYLINDER), int(g.CAPSULE), int(g.CONE)):
            return None
        a = compose_many(np.array([[0, 0, 0, *q]]), np.array([[0, 0, 1.0, 0, 0, 0, 1.0]]))[0, :3]
        if axis is None:
            axis, point = a / np.linalg.norm(a), p
        elif abs(abs(float(np.dot(a, axis))) - 1.0) > 1e-3 or np.linalg.norm(np.cross(p - point, axis)) > 1e-3:
            return None
    if axis is None:  # spheres alone: round about every axis when they share a centre
        if spheres and all(np.linalg.norm(c - spheres[0]) < 1e-3 for c in spheres):
            return [0.0, 0.0, 0.0]
        return None
    if any(np.linalg.norm(np.cross(c - point, axis)) > 1e-3 for c in spheres):
        return None
    return [round(float(v), 6) for v in axis]


@wp.kernel
def _contact_gap(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    shape0: wp.array(dtype=wp.int32),
    shape1: wp.array(dtype=wp.int32),
    point0: wp.array(dtype=wp.vec3),
    point1: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    margin0: wp.array(dtype=float),
    margin1: wp.array(dtype=float),
    shape_margin: wp.array(dtype=float),
    gap: wp.array(dtype=float),
    reach: wp.array(dtype=float),
):
    """A contact's separation along its normal between the two surfaces [m], negative when they overlap: the thickness
    Newton reports (a capsule's radius plus the shape's contact margin) less the margin, which is padding, not
    surface."""
    i = wp.tid()
    a = shape_body[shape0[i]]
    b = shape_body[shape1[i]]
    p0 = point0[i]
    p1 = point1[i]
    if a >= 0:
        p0 = wp.transform_point(body_q[a], p0)
    if b >= 0:
        p1 = wp.transform_point(body_q[b], p1)
    s0 = shape0[i]
    s1 = shape1[i]
    gap[i] = wp.dot(p1 - p0, normal[i]) - (margin0[i] - shape_margin[s0]) - (margin1[i] - shape_margin[s1])
    reach[i] = shape_margin[s0] + shape_margin[s1]  # how far apart the solver still acts on them


class _SideContacts:
    """A collision pass of the replay's own over the whole model: the contacts a thing in another island has, which
    MuJoCo's list never holds. Read-only, after a step, on the state it left; nothing is solved from it. One per model,
    reused by every replay of it (:meth:`of`)."""

    @classmethod
    def of(cls, sim: Any) -> _SideContacts:
        """This sim's pass, built once and kept on the sim itself (a rebuilt scene's sim gets its own)."""
        side = getattr(sim, "_physics_side_contacts", None)
        if side is None or side.model is not sim.model:
            side = cls(sim)
            try:
                sim._physics_side_contacts = side
            except AttributeError:  # a sim that takes no new attribute: built per call
                pass
        return side

    def __init__(self, sim: Any) -> None:
        self.sim, self.model = sim, sim.model
        cap = int(getattr(sim.model, "rigid_contact_max", 0) or 0)
        self.pipe = newton.CollisionPipeline(sim.model, rigid_contact_max=max(cap, 16384), soft_contact_max=0)
        self.overflowed = False
        self._gap = self._reach = None
        self.reach = np.zeros(0)
        shape_margin = getattr(sim.model, "shape_margin", None)
        self.margin = (
            shape_margin
            if shape_margin is not None
            else wp.zeros(int(sim.model.shape_count), dtype=float, device=sim.model.device)
        )
        self.contacts = self.pipe.contacts()

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """This state's contacting shape pairs and their separation [m] (negative: overlapping)."""
        self.pipe.collide(self.sim.state, self.contacts)
        c = self.contacts
        count = int(c.rigid_contact_count.numpy()[0])
        n = min(count, len(c.rigid_contact_shape0))
        self.overflowed = count >= len(c.rigid_contact_shape0)  # full: Newton stops writing at the cap, pairs lost
        a, b = c.rigid_contact_shape0[:n].numpy(), c.rigid_contact_shape1[:n].numpy()
        if not n:
            self.reach = np.zeros(0)
            return a, b, np.zeros(0)
        if self._gap is None or len(self._gap) < n:
            self._gap = wp.zeros(max(n, 1024), dtype=float, device=self.model.device)
            self._reach = wp.zeros(max(n, 1024), dtype=float, device=self.model.device)
        wp.launch(
            _contact_gap,
            dim=n,
            inputs=[
                self.sim.state.body_q,
                self.model.shape_body,
                c.rigid_contact_shape0,
                c.rigid_contact_shape1,
                c.rigid_contact_point0,
                c.rigid_contact_point1,
                c.rigid_contact_normal,
                c.rigid_contact_margin0,
                c.rigid_contact_margin1,
                self.margin,
            ],
            outputs=[self._gap, self._reach],
            device=self.model.device,
        )
        self.reach = self._reach[:n].numpy().astype(np.float64)
        return a, b, self._gap[:n].numpy().astype(np.float64)


def drawn_only_reach(sim: Any) -> dict[str, float]:
    """Per handled thing, how far its drawn shapes that do not collide reach outside its colliders [mm], over each
    rigid part (bodies held together by fixed joints, in the frame of the first) by the box of each shape's own
    points there (the larger of the per-axis overhangs); things that reach no farther than a millimetre are left out.
    What is drawn and does not collide passes through what it meets. A part held to the world by fixed joints only
    (a lamp, a vase of flowers on the bench) is scenery, and a part with a collider whose shape cannot be measured
    makes no claim."""
    from scipy.spatial.transform import Rotation

    from robosimstudio.core.render.plate import static_bodies
    from robosimstudio.core.render.visual import shape_geometry

    m = sim.model
    fixed = static_bodies(m)
    flags, kind, scale, xf = (
        m.shape_flags.numpy(),
        m.shape_type.numpy(),
        m.shape_scale.numpy(),
        m.shape_transform.numpy(),
    )
    sb = m.shape_body.numpy()
    sources = getattr(m, "shape_source", None) or [None] * len(flags)
    visible, collide = int(newton.ShapeFlags.VISIBLE), int(newton.ShapeFlags.COLLIDE_SHAPES)
    welds: list[tuple[int, int]] = []
    if int(getattr(m, "joint_count", 0) or 0):
        jt, jp, jc = m.joint_type.numpy(), m.joint_parent.numpy(), m.joint_child.numpy()
        welds = [(int(p), int(c)) for t, p, c in zip(jt, jp, jc, strict=False) if int(t) == int(newton.JointType.FIXED)]
    body_q = None

    def corners(i: int) -> np.ndarray | None:  # the shape's own points in its body's frame (a round mesh stays round)
        geom = shape_geometry(sources[i], int(kind[i]), scale[i])
        if geom is None:
            return None
        v = np.asarray(geom[0], dtype=np.float64)
        return Rotation.from_quat(np.asarray(xf[i, 3:], dtype=np.float64)).apply(v) + xf[i, :3]

    def into(points: np.ndarray, body: int, root: int) -> np.ndarray:
        nonlocal body_q
        if body == root:
            return points
        if body_q is None:
            body_q = np.asarray(sim.state.body_q.numpy(), dtype=np.float64)
        rb, rr = Rotation.from_quat(body_q[body, 3:]), Rotation.from_quat(body_q[root, 3:])
        return rr.inv().apply(rb.apply(points) + body_q[body, :3] - body_q[root, :3])

    out = {}
    for name, t in things(sim).items():
        bodies = [b for b in t["bodies"] if b >= 0 and not (b < len(fixed) and fixed[b])]
        group = {b: b for b in bodies}

        def top(b: int, group: dict[int, int] = group) -> int:
            while group[b] != b:
                b = group[b]
            return b

        for p, c in welds:
            if p in group and c in group:
                group[max(top(p), top(c))] = min(top(p), top(c))
        parts: dict[int, list[int]] = {}
        for b in bodies:
            parts.setdefault(top(b), []).append(b)
        worst = 0.0
        for root, members in parts.items():
            hit, drawn, unmeasured = [], [], False
            for x in (int(x) for x in t["shapes"]):
                b, f = int(sb[x]), int(flags[x])
                if b not in members or not f & (visible | collide):
                    continue
                pts = corners(x)
                if f & collide:
                    if pts is None:
                        unmeasured = True
                        break
                    hit.append(into(pts, b, root))
                elif pts is not None:
                    drawn.append(into(pts, b, root))
            if unmeasured or not drawn:
                continue
            d = np.vstack(drawn)
            if not hit:  # a part drawn and never touched at all: all of it reaches past
                worst = max(worst, float(np.ptp(d, axis=0).max()) * 1000.0)
                continue
            c = np.vstack(hit)
            over = np.maximum(d.max(0) - c.max(0), c.min(0) - d.min(0)).max()
            worst = max(worst, float(over) * 1000.0)
        if worst > 1.0:
            out[name] = round(worst, 1)
    return out


ROBOT_FIELDS = (
    "gravcomp",
    "gravcomp_restate",
    "enable_self_collisions",
    "cross_arm_collisions",
    "hull_colliders",
    "decompose_colliders",
    "decompose_threshold",
    "monolithic",
)
"""The fields of a robot that change its physics besides its file, servos and hands, held to the registered one's
(its mount, look, planner and fixed-joint collapse are the twin's to choose)."""


@functools.lru_cache(maxsize=64)
def _body_names(path: str, stamp: float) -> frozenset[str]:
    """The body (MJCF) or link (URDF) names of a robot file (``stamp`` its mtime, so an edit is read again)."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return frozenset()
    return frozenset(re.findall(r"""<(?:body|link)\b[^>]*?\bname=["']([^"']+)["']""", text))


def _robot_file(spec: Any) -> Path | None:
    from robosimstudio.paths import resolve_asset

    try:
        return resolve_asset(spec.asset)
    except (FileNotFoundError, AttributeError, TypeError, OSError):
        return None


def _servo_words(a: Any) -> str:
    if a is None:
        return "the file's own"
    effort = f", {a.effort_limit:g} limit" if getattr(a, "effort_limit", 0) else ""
    return f"gains {a.target_ke:g}/{a.target_kd:g}{effort}"


def _hand_servo(spec: Any, arm: Any) -> Any:
    """The servo a finger DOF of ``arm`` is built with: the robot's stated gripper servo, else its hand's own."""
    from robosimstudio.robots.registry import resolve_gripper

    if spec.gripper_actuation is not None:
        return spec.gripper_actuation
    try:
        hand = resolve_gripper(spec, arm)
    except (KeyError, ValueError):
        return None
    return getattr(hand, "actuation", None)


def _registered_arm(reg: Any, arm: Any, i: int) -> Any:
    same = [a for a in reg.arms if a.name == arm.name]
    return same[0] if same else (reg.arms[i] if i < len(reg.arms) else None)


def shipped_robot(spec: Any) -> tuple[Any, str]:
    """The library's robot ``spec`` is, matched by its file -- or, for a file the library does not ship, by the bodies
    of a copy of one it does (a copy, edited or not, keeps them) -- and a few words when it is such a copy; ``(None,
    "")`` where it is no robot the library ships (a file of the twin's own, registered by it or not)."""
    from robosimstudio.paths import resolve_path
    from robosimstudio.robots.registry import get_embodiment, list_embodiments

    try:
        root = resolve_path("embodiments")
    except FileNotFoundError:
        return None, ""
    shipped = []
    for name in list_embodiments():
        reg = get_embodiment(name)
        f = _robot_file(reg)
        if f is not None and f.is_relative_to(root):
            shipped.append((reg, f))
    mine = _robot_file(spec)
    if mine is None:
        return None, ""
    same = [reg for reg, f in shipped if f == mine]
    if same:  # several robots share one file (a hand apart): the one of this name, else the one it departs least from
        return min(same, key=lambda reg: (reg.name != spec.name, sum(map(bool, _departures(spec, reg).values())))), ""
    names = _body_names(str(mine), mine.stat().st_mtime)
    best, share = None, 0.0
    for reg, f in shipped:
        theirs = _body_names(str(f), f.stat().st_mtime)
        if len(theirs) >= 6 and len(theirs & names) / len(theirs) > share:  # a few generic names match anything
            best, share = (reg, f), len(theirs & names) / len(theirs)
    if best is None or share < 0.8:
        return None, ""
    return best[0], f"its file {mine} is a copy of the registered {best[0].name}'s {best[1].name}"


def _departures(spec: Any, reg: Any, resident: bool = False) -> dict[str, str]:
    """How ``spec`` differs from the registered ``reg`` beyond its file: arm servo, each arm's hand and its servo,
    and :data:`ROBOT_FIELDS` (but cross-arm collisions on a scene-resident robot, which its scene's import decides)."""
    out = {"arm": "", "gripper": "", "hand": "", "other": ""}
    if spec.arm_actuation != reg.arm_actuation:
        out["arm"] = (
            f"its arm servo is the twin's own ({_servo_words(spec.arm_actuation)}; "
            f"registered: {_servo_words(reg.arm_actuation)})"
        )
    hands, servos = [], []
    for i, arm in enumerate(spec.arms):
        theirs = _registered_arm(reg, arm, i)
        if theirs is None:
            continue
        mine_hand, reg_hand = arm.gripper or spec.gripper, theirs.gripper or reg.gripper
        if mine_hand != reg_hand or arm.gripper_mount != theirs.gripper_mount:
            hands.append(f"{arm.name} {mine_hand or 'none'} (registered: {reg_hand or 'none'})")
        a, r = _hand_servo(spec, arm), _hand_servo(reg, theirs)
        if a != r:
            servos.append(f"{arm.name} {_servo_words(a)} (registered: {_servo_words(r)})")
    if hands:
        out["hand"] = "its hand is the twin's own: " + ", ".join(hands)
    if servos:
        out["gripper"] = "its gripper servo is the twin's own: " + ", ".join(dict.fromkeys(servos))
    other = [
        f"{k} {getattr(spec, k)!r} (registered: {getattr(reg, k)!r})"
        for k in ROBOT_FIELDS
        if getattr(spec, k, None) != getattr(reg, k, None) and not (resident and k == "cross_arm_collisions")
    ]
    if other:
        out["other"] = "its " + ", ".join(other)
    return out


def robot_departures(sim: Any) -> dict[str, Any]:
    """How the robot the scene was built with differs from the library's robot it is (:func:`shipped_robot`): its
    file, its arm servo, each arm's hand and gripper servo, its other physics (each a few words, empty when the
    same), or ``registered: False`` when it is no robot the library ships (a file of the twin's own, whose limits
    cannot be held to anything). A mount (``at=``, ``arms_at=``) is not a departure."""
    emb = getattr(sim.scene, "embodiment", None)
    spec = getattr(emb, "spec", None)
    if spec is None or getattr(spec, "asset", None) is None:
        return {}
    reg, copied = shipped_robot(spec)
    if reg is None:
        return {"name": spec.name, "registered": False, "file": str(getattr(spec.asset, "path", ""))}
    departed = _departures(spec, reg, resident=bool(getattr(emb, "scene_resident", False)))
    return {"name": spec.name, "registered": True, "as": reg.name, "file": copied, **departed}


def model_facts(sim: Any) -> dict[str, Any]:
    """What the model holds that the contact rule is about, and the numbers the robot runs on.

    A joint or equality from a thing's body to the robot is an attachment
    whatever it is called (``ties``); a thing welded to the world or to another
    thing is noted for the report to explain (``welds``), while a hinge or a
    slide to the world or to furniture is a mechanism (a lid, a drawer) and is
    not. A stiffness on a thing's own joint (``drives``) is its elasticity (a
    rod's bending) while its target stays put, and a motor once something moves
    the target: :class:`Replay` watches those targets (``drive_rows``) and the
    verdict reads the written run's. An infinite mass is a kinematic thing. The friction is the one
    MuJoCo mixes at a finger contact (the larger of the two surfaces'), which is
    not the number a twin wrote for its thing when the hand's own pads are
    grippier.
    """
    m = sim.model
    mu = m.shape_material_mu.numpy()
    mass = m.body_mass.numpy()
    th, fing = things(sim), fingers(sim)
    fixed_body = _fixed_bodies(m)
    owner = {b: name for name, t in th.items() for b in t["bodies"]}
    robot_bodies = {b for b in _robot_bodies(sim) if b >= 0}
    out: dict[str, Any] = {
        "robot": robot_name(sim),
        "real_arm_torque": real_arm_torque(sim),
        "arm_torque": {name: real_arm_torque(sim, arm) for name, arm in sim.scene.handles.arms.items()},
        "things": {
            name: {
                "mass_kg": round(float(sum(mass[b] for b in t["bodies"])), 4),
                "bodies": len(t["bodies"]),
                "mu": sorted({round(float(v), 3) for v in mu[t["shapes"]]}) if len(t["shapes"]) else [],
                "kinematic": bool(any(fixed_body[b] for b in t["bodies"])),
            }
            for name, t in th.items()
        },
        "finger_mu": {a: sorted({round(float(v), 3) for v in mu[s]}) for a, s in fing.items()},
        "hands": [real_hand(sim, arm, name) for name, arm in sim.scene.handles.arms.items()],
        "robot_spec": robot_departures(sim),
        "drawn_only_mm": drawn_only_reach(sim),
    }
    kinds = _joint_kinds()
    free, fixed_kind = kinds.get("free", -1), kinds.get("fixed", -2)
    kinds = {v: k for k, v in kinds.items()}
    jt, jp, jc = m.joint_type.numpy(), m.joint_parent.numpy(), m.joint_child.numpy()
    qd0 = m.joint_qd_start.numpy()
    ke = m.joint_target_ke.numpy()
    tstart = getattr(m, "joint_target_q_start", None)
    tstart = tstart.numpy() if tstart is not None else (m.joint_q_start.numpy() if _by_coord(sim) else qd0)
    ties, welds, drives, drive_rows = [], [], [], []
    tie_things, drive_things = [], []  # the things those name, for the frozen line
    enabled = getattr(m, "joint_enabled", None)
    live = enabled.numpy() if enabled is not None else np.ones(len(jt), dtype=bool)
    for j in range(len(jt)):
        child, parent = int(jc[j]), int(jp[j])
        if not live[j]:  # a weld built off (an MJCF weld with active="false") holds nothing until switched on
            continue
        if parent in owner and child in robot_bodies:  # a robot link hung off a thing: the same tie the other way
            ties.append(f"{owner[parent]} by a {kinds.get(int(jt[j]), 'joint')} joint to the robot")
            tie_things.append(owner[parent])
            continue
        if child not in owner or int(jt[j]) == free:
            continue
        thing, kind = owner[child], kinds.get(int(jt[j]), "joint")
        if parent in robot_bodies:
            ties.append(f"{thing} by a {kind} joint to the robot")
            tie_things.append(thing)
            continue
        if int(jt[j]) == fixed_kind and owner.get(parent) != thing:
            welds.append(f"{thing} welded to {'the world' if parent == -1 else owner.get(parent, f'body {parent}')}")
        end = int(qd0[j + 1]) if j + 1 < len(qd0) else int(m.joint_dof_count)
        stiff = [d for d in range(int(qd0[j]), end) if ke[d] > 0]
        if stiff:
            drives.append(f"{thing}'s {kind} joint {j} (ke {float(max(ke[d] for d in stiff)):g})")
            drive_things.append(thing)
            # where the target of each stiff dof sits in joint_target_q (a hinge's or a slide's one coordinate is its
            # dof's; a ball joint's four coordinates do not line up with its three dofs, and are read as they fall)
            drive_rows += [int(tstart[j]) + (d - int(qd0[j])) for d in stiff]
    rows = equality_rows(sim)
    enabled = getattr(getattr(m, "mujoco", None), "equality_constraint_enabled", None)
    on = enabled.numpy() if enabled is not None else None
    for k, (body1, body2, _label) in rows.items():
        pair = {body1, body2}
        hit = next((owner[b] for b in pair if b in owner), None)
        inside = owner.get(body1) is not None and owner.get(body1) == owner.get(body2)  # a loop closed within a thing
        if hit and not pair & robot_bodies and not inside and (on is None or bool(on[k])):
            welds.append(f"{hit} by equality constraint {k}")
    # a row between a thing and the robot ties it only while enabled (a suction cup's rows are built off and switch
    # on at a hold): the written replay watches them, and the verdict reads whether one was on
    out["robot_equalities"], out["robot_equality_things"] = {}, {}
    for k, (body1, body2, label) in rows.items():
        thing = next((owner[b] for b in (body1, body2) if b in owner and b not in robot_bodies), None)
        if {body1, body2} & robot_bodies and thing is not None:
            out["robot_equalities"][str(k)] = label
            out["robot_equality_things"][str(k)] = thing
    out["ties"] = ties
    out["tie_things"] = tie_things
    out["welds"] = welds
    out["drives"] = drives
    out["drive_things"] = drive_things
    out["drive_rows"] = drive_rows
    out["kinematic"] = [name for name, t in out["things"].items() if t["kinematic"]]
    out["gravity"] = round(float(np.linalg.norm(np.ravel(m.gravity.numpy())[:3])), 4)
    # a part whose joint is stiff and held at its target bends or sags under its own weight: its drift at the reset is
    # the thing's elasticity, read like a deformable's, not a rigid thing pushed
    out["elastic_parts"] = sorted(
        f"{owner[int(jc[j])]}#{th[owner[int(jc[j])]]['bodies'].index(int(jc[j]))}"
        for j in range(len(jt))
        if live[j]
        and int(jc[j]) in owner
        and th[owner[int(jc[j])]]["bodies"].index(int(jc[j])) > 0
        and any(ke[d] > 0 for d in range(int(qd0[j]), int(qd0[j + 1]) if j + 1 < len(qd0) else int(m.joint_dof_count)))
    )
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    flags = m.shape_flags.numpy()
    solid, thick = getattr(m, "shape_is_solid", None), getattr(m, "shape_margin", None)
    tables = (
        m.shape_type.numpy(),
        m.shape_scale.numpy(),
        getattr(m, "shape_source", None),
        solid.numpy() if hasattr(solid, "numpy") else solid,
        thick.numpy() if hasattr(thick, "numpy") else thick,
    )
    for name, t in th.items():
        vols = [_shape_volume(m, int(x), tables) for x in t["shapes"] if int(flags[int(x)]) & collide]
        vol, big = sum(vols), max(vols, default=0.0)
        mass_kg = out["things"][name]["mass_kg"]
        out["things"][name]["volume_l"] = round(vol * 1000.0, 4)
        out["things"][name]["density"] = round(mass_kg / vol, 1) if vol > 0 else None
        # overlapping colliders count twice in the sum: the largest alone bounds the density from above
        out["things"][name]["density_max"] = round(mass_kg / big, 1) if big > 0 else None
    # where the scene's settle (before the first step) moved each thing from where it was put: pushed up or aside, it
    # was put inside something; fallen, it was put in the air. Read off the state the replays start from
    authored, now = m.body_q.numpy(), sim.state.body_q.numpy()
    out["settle"] = {}  # filled below
    for name, t in th.items():
        b = t["bodies"][0]
        d = (now[b, :3] - authored[b, :3]) * 1000.0
        out["settle"][name] = {
            "rise_mm": round(float(d[2]), 2),
            "aside_mm": round(float(np.hypot(d[0], d[1])), 2),
            "turn_deg": round(_turn_deg(authored[b, 3:], now[b, 3:]), 2),
        }
    # and how deep each thing sits in something at the reset the replays start from: a library object is put back
    # where it was written (so an overlap is still there), a twin's own entity may keep where the warm-up left it (so
    # the settle above says it). The robot's fingers count: a hand that starts inside the thing is the same fault
    try:
        # kept on the sim where a replay reads with it (something is unread), else built for this one read
        side = _SideContacts.of(sim) if unread_things(sim, list(th)) else _SideContacts(sim)
        a, b, gap = side.read()
        if side.overflowed:
            out["settle_error"] = (
                "the start overlap read overflowed the collision pass's buffer: what it found is partial"
            )
        sowner = np.full(len(m.shape_body.numpy()), -1, dtype=np.int64)
        names = list(th)
        for i, name in enumerate(names):
            sowner[np.asarray(th[name]["shapes"], dtype=np.int64)] = i
        labels = list(getattr(m, "shape_label", None) or [""] * len(sowner))
        for name in names:
            out["settle"][name]["overlap_mm"] = 0.0
        for x, y, g in zip(a, b, gap, strict=True):
            for p, q in ((int(x), int(y)), (int(y), int(x))):
                i = sowner[p]
                if i >= 0 and sowner[q] != i and -g * 1000.0 > out["settle"][names[i]]["overlap_mm"]:
                    out["settle"][names[i]]["overlap_mm"] = round(float(-g * 1000.0), 2)
                    out["settle"][names[i]]["inside"] = str(labels[q]).split("/")[-1] or f"shape {q}"
    except (
        Exception
    ) as exc:  # a model no collision pass can be built over: the settle's own reading stands, and says so
        out["settle_error"] = f"the start overlap could not be read ({type(exc).__name__}: {exc})"[:300]
    axis_tables = (m.shape_type.numpy(), m.shape_transform.numpy(), m.shape_flags.numpy())
    for name, t in th.items():
        if len(t["bodies"]) == 1:
            axis = _round_axis(m, list(t["shapes"]), axis_tables)
            if axis is not None:
                out["things"][name]["round_axis"] = axis
    return out


# --------------------------------------------------------------------------- the replay


def equality_rows(sim: Any) -> dict[int, tuple[int, int, str]]:
    """The body-to-body equality rows (a weld, a connect) the scene has, by row: its two bodies and its label.

    Newton 1.7 keeps them on ``model.mujoco`` (``equality_constraint_*``), not on the model itself.
    """
    table = getattr(sim.model, "mujoco", None)
    count = int(getattr(table, "equality_constraint_count", 0) or 0)
    if not count:
        return {}
    b1, b2 = table.equality_constraint_body1.numpy(), table.equality_constraint_body2.numpy()
    objtype = getattr(table, "equality_constraint_objtype", None)
    objs = objtype.numpy() if objtype is not None else np.ones(count)
    kind = getattr(table, "equality_constraint_type", None)
    kinds = kind.numpy() if kind is not None else np.ones(count)
    labels = list(getattr(table, "equality_constraint_label", None) or [""] * count)
    # mjOBJ_BODY is 1; the default, -1, is read as a body by Newton's MuJoCo solver; a connect (0) or a weld (1)
    # is a body pair, a joint or tendon coupling is not
    keep = [k for k in range(count) if (int(objs[k]) == 1 or int(objs[k]) < 0) and int(kinds[k]) in (0, 1)]
    return {k: (int(b1[k]), int(b2[k]), str(labels[k]) if k < len(labels) else "") for k in keep}


def _fixed_bodies(model: Any) -> np.ndarray:
    """Per body, whether nothing but a script moves it: Newton's KINEMATIC flag (on a hinge or a slide too: a drawer
    flagged so opens only by script), or an infinite mass on a body free to move (a free joint, or none). A body
    welded in place (a cabinet's carcass, flagged or massless) is a weld, and a massless link on a joint is moved by
    the joint: neither is counted."""
    count = int(model.body_count)
    welded, jointed = np.zeros(count, dtype=bool), np.zeros(count, dtype=bool)
    kinds = _joint_kinds()
    jt, jc = model.joint_type.numpy(), model.joint_child.numpy()
    for t, c in zip(jt, jc, strict=False):
        if int(c) >= 0:
            welded[int(c)] |= int(t) == kinds.get("fixed", -2)
            jointed[int(c)] |= int(t) != kinds.get("free", -1)
    # an infinite mass where nothing but a script can move the body (free, or no joint): a massless link on a hinge
    # (a URDF's intermediate frame) is moved by its joint
    fixed = (model.body_inv_mass.numpy() == 0.0) & ~jointed
    flags = getattr(model, "body_flags", None)
    if flags is not None:
        try:
            import newton

            fixed |= (flags.numpy() & int(newton.BodyFlags.KINEMATIC)) != 0
        except Exception:  # pragma: no cover - a Newton without the flag
            pass
    return fixed & ~welded  # one welded in place (the weld_to_world pattern) is a weld, flagged or not


@functools.lru_cache(maxsize=1)
def _joint_kinds() -> dict[str, int]:
    """Newton's joint types by lower-case name (``free``, ``fixed``, ``revolute``, ...)."""
    try:
        import newton

        return {k.lower(): int(v) for k, v in newton.JointType.__members__.items()}
    except Exception:  # pragma: no cover - newton is always there on a page
        return {}


def _turn_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    d = abs(float(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1))))
    return math.degrees(2 * math.acos(min(1.0, d)))


def _own_points(probe: Any, name: str) -> np.ndarray:
    """Deformable ``name``'s own particles: its probe is over the whole model, with the thing's set by its name."""
    return np.asarray(probe.points(name if name in getattr(probe, "sets", {}) else None))


class Broken(BaseException):
    """The replay's own hold failed inside a step: a broken replay, not a reading of the routine."""


@wp.kernel
def _write_rows(rows: wp.array(dtype=wp.int32), values: wp.array(dtype=wp.float32), target: wp.array(dtype=wp.float32)):
    i = wp.tid()
    target[rows[i]] = values[i]


class Overrun(BaseException):
    """A replay's routine was still going well past the written run's length: raised from each step past it.

    A ``BaseException``, and raised again on every step until the replay
    clears it, so a routine that catches ``Exception`` around its own stepping
    cannot swallow it and loop on.
    """


class Replay:
    """Holds the robot's targets for ``frozen`` and ``open`` before each step (a pre-step hook), logs after (a post).

    Logged per step: each thing's centre of mass and turn (every body of a
    jointed thing), the force each hand's fingers and the robot as a whole put
    on it, whether anything is within the contact margin of it, the deepest
    overlap, each deformable's particles, the jaw widths.
    """

    def __init__(
        self,
        sim: Any,
        mode: str,
        drive_rows: list[int] | None = None,
        model_rows: dict | None = None,
        hold_from: int = 0,
    ) -> None:
        self.sim, self.mode, self.hold_from = sim, mode, int(hold_from)
        self.skip = max(10, int(0.5 * float(sim.request.physics.control_hz)))  # the fingers' own time to open
        self.lay, self.th, self.fing = layout(sim), things(sim), fingers(sim)
        self.dt = 1.0 / float(sim.request.physics.control_hz)
        n = int(sim.model.shape_count)
        self.names = list(self.th)
        self.owner = np.full(n, -1, dtype=np.int64)
        for i, name in enumerate(self.names):
            self.owner[self.th[name]["shapes"]] = i
        free_kind = _joint_kinds().get("free", -1)
        jt, jc = sim.model.joint_type.numpy(), sim.model.joint_child.numpy()
        free_bodies = {int(c) for t, c in zip(jt, jc, strict=False) if int(t) == free_kind}
        # a thing on a hinge, one a cloth or a soft body can push, or one another solver steps (their contacts are
        # not MuJoCo's) does not fall at g when nothing in MuJoCo's list touches it: free flight is read only where
        # it means something
        soft_scene = bool(sim.particle_probes())
        self.unread = unread_things(sim, self.names)
        self.side = None
        self.side_error = ""
        if self.unread and not soft_scene:  # contacts across islands read by a pass of the replay's own
            try:
                self.side = _SideContacts.of(sim)
            except Exception as exc:
                self.side = None
                self.side_error = f"the replay's collision pass could not be built ({type(exc).__name__})"
        self.unread_mask = np.array([name in self.unread for name in self.names] + [False], dtype=bool)
        self.single_free = {
            name
            for name in self.names
            if len(self.th[name]["bodies"]) == 1 and self.th[name]["bodies"][0] in free_bodies
        }
        read_here = mode in ("written", "plain")  # the replays whose every row the pass reads
        self.free = {
            name
            for name in self.names
            if not soft_scene
            and (name not in self.unread or (self.side is not None and read_here))
            and len(self.th[name]["bodies"]) == 1
            and self.th[name]["bodies"][0] in free_bodies
        }
        self.side_overflowed = 0
        try:
            self.hidden = _hidden_colliders(sim)
        except Exception:  # a scene the scan cannot read: no prop named, every other reading stands
            self.hidden = {}
        self.hidden_mask = np.zeros(n, dtype=bool)
        self.hidden_mask[list(self.hidden)] = True
        self.hidden_touch: dict[int, set[str]] = {}
        self.parts = [(f"{name}#{k}", b) for name in self.names for k, b in enumerate(self.th[name]["bodies"][1:], 1)]
        self.arms = list(self.fing)
        self.finger_of = np.full(n, -1, dtype=np.int64)
        for i, a in enumerate(self.arms):
            self.finger_of[self.fing[a]] = i
        self.robot = np.zeros(n, dtype=bool)
        self.robot[np.asarray(sim.scene.handles.robot_shapes, dtype=np.int64)] = True
        com = sim.model.body_com.numpy()
        self.pose_names = self.names + [p for p, _ in self.parts]
        self.pose_bodies = np.asarray(
            [self.th[n]["bodies"][0] for n in self.names] + [b for _, b in self.parts], dtype=np.int64
        )
        self.com_local = np.zeros((len(self.pose_bodies), 7))
        self.com_local[:, :3] = com[self.pose_bodies]
        self.com_local[:, 6] = 1.0
        self.rows: list[dict] = []
        self.stamps: list[int] = []
        self.soft0: dict[str, np.ndarray] = {}
        self.mu_seen: list[float] = []
        self.steps = 0
        self.error = ""
        self.start: np.ndarray | None = None
        self.held: np.ndarray | None = None
        """The target rows ``frozen`` or ``open`` hold, and what they are held at (``held_to``)."""
        self.held_to: np.ndarray | None = None
        self.hooks0: set[int] = set()
        self.kept: list = []
        self._hold_hook = self._after_hook = self._cap_hook = None
        self.cleanup_error = ""
        """Why the routine's own cleanup failed when the replay stopped it early, if it did (the reading stands)."""
        self.hooks_added = 0
        """Substep hooks found beyond those the scene had at the start: a dragger or a pin attached."""
        self.step_hooks_added = 0
        """Step-list hooks (pre-step, post-step, teleop) found beyond the start's: a follower, or only a logger."""
        self.limit = 0
        """Steps after which a routine that steps the scene itself is stopped, raised from the step; 0 never."""
        self.drive_rows = np.asarray(drive_rows if mode == "written" and drive_rows else [], dtype=np.int64)
        """The ``joint_target_q`` rows of the stiff joints on things (:func:`model_facts`'s ``drive_rows``), watched
        on the written replay, the one the verdict reads them off."""
        self.drive0: np.ndarray | None = None
        table = getattr(sim.model, "mujoco", None)
        robot_rows = (model_rows or {}) if mode == "written" else {}
        self.eq_rows = np.asarray(sorted(int(k) for k in robot_rows), dtype=np.int64)
        self.eq_enabled = getattr(table, "equality_constraint_enabled", None) if len(self.eq_rows) else None
        self.eq_on: set[int] = set()
        """The thing-to-robot equality rows found enabled at any step of the written run: a weld or a seal held it."""
        # holds a scene builds in rather than adds while it plays: a suction cup (its weld or its wrench tier) and a
        # rod's grasp latch, read each step of the written run
        self.cups: list[tuple[str, Any]] = []
        self.latches: list[tuple[str, Any]] = []
        if mode == "written":
            for name in sim.scene.handles.arms:
                try:
                    self.cups.append((name, sim.suction(name)))
                except Exception:
                    pass
            for name, entity in getattr(sim.scene, "by_name", {}).items():
                flag = getattr(entity, "_hold_flag", None)
                if flag is not None and getattr(getattr(entity, "spec", None), "allow_grasp_latch", False):
                    self.latches.append((name, flag))
        self.suction_held: set[str] = set()
        """The arms whose suction cup held a part at any step of the written run."""
        self.latched: set[str] = set()
        """The rods whose grasp latch was engaged at any step of the written run."""
        self.drive_moved = 0.0
        """The most any of those targets moved from where it started: a motor on a thing, if above zero."""
        self.broken = ""
        """Why the replay's own hold failed, if it did: the replay is then not a reading."""
        self.overrun_at = 0
        self.grace = 25
        self.inside = False
        """Whether the step now running was taken inside the routine's own ``next()`` (so may raise into it)."""

    def install(self) -> None:
        """Read the start (the first row: what a push-out in the first step moves is measured from it), then hook in."""
        self.start = self.sim.control.joint_target_q.numpy().copy()
        if self.mode == "frozen":
            self.held = self.lay["robot"]
            self.held_to = self.start[self.held]
        elif self.mode == "open" or (self.mode == "drop" and self.hold_from > 0):  # drop: from the top of the carry
            self.held, self.held_to = self.lay["grip"], self.lay["open"]
        self._hold_hook, self._after_hook = self._hold, self._after  # one object each, to find and remove
        self.kept = [fn for hooks in self.sim.hook_lists().values() for fn in hooks]  # held, so no id is reused
        self.hooks0 = {id(fn) for fn in self.kept}
        if len(self.drive_rows):
            self.drive0 = self.start[self.drive_rows].copy()
        self.log(contacts=False)  # the reset stepped nothing: the probe still holds the last replay's contacts
        if self.held is not None:
            dev = self.sim.model.device
            dofs = self.lay["robot_dofs"] if self.mode == "frozen" else self.lay["grip_dofs"]
            self._dof_rows = wp.array(np.asarray(dofs, dtype=np.int32), dtype=wp.int32, device=dev)
            control = self.sim.control
            self._dof_start = []  # (the array, what its held rows held at the start)
            for array in (getattr(control, "joint_target_qd", None), getattr(control, "joint_f", None)):
                if array is not None and len(dofs):
                    start = np.asarray(array.numpy(), dtype=np.float32)[np.asarray(dofs, dtype=np.int64)]
                    self._dof_start.append((array, wp.array(start, dtype=wp.float32, device=dev)))
            self._held_rows = wp.array(self.held.astype(np.int32), dtype=wp.int32, device=self.sim.model.device)
            self._held_values = wp.array(
                np.asarray(self.held_to, dtype=np.float32), dtype=wp.float32, device=self.sim.model.device
            )
        # on the env's last lists: after every other hook (the hold has the last word on the targets, the log reads
        # the step after any follower), and never retired, so a reset the routine itself takes keeps them on
        self._cap_hook = self._cap
        self.sim.last_pre_step_hooks.append(self._cap_hook)
        if self.held is not None:
            self.sim.last_pre_step_hooks.append(self._hold_hook)
        self.sim.last_post_step_hooks.append(self._after_hook)

    def _added(self) -> tuple[int, int]:
        """How many hooks are on the scene now that were not at the start: on substep phases, and on the step lists
        (this replay's own two left out)."""
        mine = {id(self._hold_hook), id(self._after_hook), id(self._cap_hook)}
        substep = steps = 0
        for name, hooks in self.sim.hook_lists().items():
            new = sum(1 for fn in hooks if id(fn) not in self.hooks0 and id(fn) not in mine)
            if name in ("pre_step", "post_step", "teleop", "last_pre_step", "last_post_step"):
                steps += new
            else:
                substep += new
        return substep, steps

    def remove(self) -> None:
        self.limit = 0
        for hooks, hook in (
            (self.sim.last_pre_step_hooks, self._cap_hook),
            (self.sim.last_pre_step_hooks, self._hold_hook),
            (self.sim.last_post_step_hooks, self._after_hook),
        ):
            if hook is not None and any(h is hook for h in hooks):
                hooks.remove(hook)

    def _hold(self, _env: Any = None) -> None:
        if self.steps < self.hold_from:  # drop: the fingers are the routine's until the top of the carry
            return
        try:
            control, dev = self.sim.control, self.sim.model.device
            wp.launch(
                _write_rows,
                dim=len(self.held),
                inputs=[self._held_rows, self._held_values],
                outputs=[control.joint_target_q],
                device=dev,
            )
            # a velocity target or a feed-forward force moves a held joint as surely as a position target does: both
            # are held where they started (zero for a position-driven arm; an OSC's start torque for a torque-driven
            # one, which keeps it up, where zero would drop it)
            for array, start in self._dof_start:
                wp.launch(
                    _write_rows, dim=len(self._dof_rows), inputs=[self._dof_rows, start], outputs=[array], device=dev
                )
        except Exception as exc:  # the replay's own fault: not a reading the routine's checks made
            self.broken = self.broken or f"holding the robot's targets: {type(exc).__name__}: {exc}"
            if self.inside:  # a step the routine took itself; one the page takes must not raise into the page
                raise Broken(self.broken) from exc

    def _cap(self, _env: Any = None) -> None:
        """Before a step the routine takes itself past the cap, stop it, raised into the routine before the physics, so
        no step is left half done. The routine's own cleanup (a finally that opens the gripper and settles) gets a
        grace of as many steps, and a routine that swallowed it and loops on is stopped before every step after."""
        if self.limit and self.inside and self.steps >= self.limit:
            first = self.overrun_at == 0
            self.overrun_at = self.overrun_at or self.steps
            if first or self.steps >= self.overrun_at + self.grace:
                raise Overrun(self.steps)

    def _after(self, _env: Any = None) -> None:
        self.steps += 1
        try:
            self.log()
        except Exception:
            self.error = self.error or traceback.format_exc()[-800:]

    def log(self, contacts: bool = True) -> None:
        sim = self.sim
        bq = sim.state.body_q.numpy()
        n = len(self.names)
        hand = np.zeros((n, len(self.arms)))
        touch = np.zeros((n, len(self.arms)))
        overflow = False
        robot_touch = np.zeros(n)
        robot, near, depth_robot = np.zeros(n), np.zeros(n), np.zeros(n)
        probe = getattr(sim, "contacts", None)
        if self.mode == "written":
            for name, cup in self.cups:
                try:
                    if cup.holding:
                        self.suction_held.add(name)
                except Exception:
                    pass
            for name, flag in self.latches:
                try:
                    if np.asarray(flag.numpy()).any():
                        self.latched.add(name)
                except Exception:
                    pass
        if self.eq_enabled is not None:  # every step: a seal switched on and off between samples still held
            on = np.asarray(self.eq_enabled.numpy())[self.eq_rows]
            self.eq_on |= {int(k) for k, v in zip(self.eq_rows, on, strict=True) if v}
        if contacts:  # every step and by identity: a hook swapped for another, or on for two steps, is seen
            substep, steps = self._added()
            self.hooks_added = max(self.hooks_added, substep)
            self.step_hooks_added = max(self.step_hooks_added, steps)
        if probe is not None and n and contacts:
            probe.update(sim.step_count)  # the step's read, shared with a panel reading the same frame
            if len(probe.geoms):
                a, b = probe._shape_pairs()
                force = np.linalg.norm(probe.impulses, axis=1) / probe.dt
                depth = np.maximum(-np.asarray(probe.depths, dtype=np.float64), 0.0)
                mapped = (a >= 0) & (b >= 0)  # a geom with no Newton shape reads -1, which numpy would wrap
                a, b, force, depth = a[mapped], b[mapped], force[mapped], depth[mapped]
                for x, y in ((a, b), (b, a)):
                    ox = self.owner[x]
                    real = (ox >= 0) & (self.owner[y] != ox)  # a thing's own shapes touching are not a contact
                    np.add.at(near, ox[real], 1)
                    on_robot = real & self.robot[y]
                    np.add.at(robot, ox[on_robot], force[on_robot])
                    np.add.at(robot_touch, ox[on_robot], 1)
                    np.maximum.at(depth_robot, ox[on_robot], depth[on_robot])
                    on_finger = real & (self.finger_of[y] >= 0)
                    np.add.at(hand, (ox[on_finger], self.finger_of[y][on_finger]), force[on_finger])
                    for i, sh in zip(ox[real & self.hidden_mask[y]], y[real & self.hidden_mask[y]], strict=True):
                        self.hidden_touch.setdefault(int(i), set()).add(self.hidden[int(sh)])
                if self.steps % 10 == 0:
                    self._mixed_friction(probe)
        in_window = self.mode != "drop" or (
            self.hold_from and self.hold_from + self.skip <= self.steps <= self.hold_from + self.skip + RELEASE_WINDOW
        )
        if self.side is not None and n and contacts and self.mode in ("written", "drop", "plain") and in_window:
            try:
                a, b, gap = self.side.read()
            except Exception as exc:  # flight goes unjudged for what only this pass reads; the row stands
                self.side_error = f"the replay's collision pass failed ({type(exc).__name__}: {exc})"[:200]
                self.side = None
                a = b = np.zeros(0, dtype=np.int32)
                gap = np.zeros(0)
            else:
                self.side_overflowed += int(self.side.overflowed)
                overflow = bool(self.side.overflowed)
            reach = self.side.reach if self.side is not None and len(self.side.reach) == len(gap) else 0.0
            close = gap < np.maximum(NEAR_MM / 1000.0, np.minimum(reach, REACH_MM / 1000.0))
            a, b, gap = a[close], b[close], gap[close]
            for x, y in ((a, b), (b, a)):
                ox = self.owner[x]
                keep = ox >= 0
                ys, ox, g = y[keep], ox[keep], gap[keep]
                other = self.owner[ys] != ox
                partner = self.owner[ys]
                across = other & ~self.unread_mask[ox] & (partner >= 0) & self.unread_mask[np.maximum(partner, 0)]
                np.add.at(near, ox[across], 1)  # a MuJoCo thing touching a thing another island steps
                real = self.unread_mask[ox] & other
                np.add.at(near, ox[real], 1)
                on_finger = real & (self.finger_of[ys] >= 0)
                np.add.at(touch, (ox[on_finger], self.finger_of[ys][on_finger]), 1)
                on_robot = real & self.robot[ys]
                np.add.at(robot_touch, ox[on_robot], 1)
                np.maximum.at(depth_robot, ox[on_robot], np.maximum(-g[on_robot], 0.0))
                for i, sh in zip(ox[real & self.hidden_mask[ys]], ys[real & self.hidden_mask[ys]], strict=True):
                    self.hidden_touch.setdefault(int(i), set()).add(self.hidden[int(sh)])
        rows = self.pose_bodies
        com = compose_many(bq[rows], self.com_local)[:, :3]  # each body's centre of mass, not its origin
        pose = {name: [*com[i], *bq[b, 3:]] for i, (name, b) in enumerate(zip(self.pose_names, rows, strict=True))}
        soft, jaw = {}, {}
        sampled = self.steps % SAMPLE_EVERY == 0 or self.mode == "plain"  # plain's reads are already sparse
        if self.drive0 is not None:  # every step: a target pulsed between samples is still a motor
            moved = np.abs(sim.control.joint_target_q.numpy()[self.drive_rows] - self.drive0)
            self.drive_moved = max(self.drive_moved, float(moved.max()))
        for name, p in (sim.particle_probes() or {}).items() if sampled else ():
            try:
                p.update(sim.state, None)
                pts = _own_points(p, name)
            except Exception:
                continue
            if len(pts):
                base = self.soft0.setdefault(name, pts.copy())
                if base.shape != pts.shape:  # particles added or cut away: measure from here on
                    base = self.soft0[name] = pts.copy()
                away = np.linalg.norm(pts - base, axis=1)
                soft[name] = [*pts.mean(0).tolist(), float(away.max()), float(np.median(away))]
        for a in self.arms if sampled else ():
            try:
                jaw[a] = float(sim.gripper(a).width(sim.state))
            except Exception:
                pass
        self.stamps.append(int(getattr(sim, "step_count", len(self.stamps)) or len(self.stamps)))
        self.rows.append(
            {
                "pose": pose,
                "hand": hand.tolist(),
                "touch": touch.tolist(),
                "robot_touch": robot_touch.tolist(),
                "overflow": overflow,
                "robot": robot.tolist(),
                "near": near.tolist(),
                "depth": depth_robot.tolist(),
                "soft": soft,
                "jaw": jaw,
            }
        )

    def _mixed_friction(self, probe: Any) -> None:
        """The friction MuJoCo used at this step's finger contacts: a sample every ten steps is plenty. Not on a CPU
        solver, whose Warp copy of the data is never stepped."""
        if getattr(probe, "_host", False):
            return
        try:
            data = probe.solver.mjw_data
            count = int(np.ravel(data.nacon.numpy())[0])
            if not count:
                return
            g = data.contact.geom.numpy()[:count]
            mu = data.contact.friction.numpy()[:count, 0]
            sa, sb = probe.geom_to_shape[g[:, 0]], probe.geom_to_shape[g[:, 1]]
            mapped = (sa >= 0) & (sb >= 0)
            sa, sb, mu = sa[mapped], sb[mapped], mu[mapped]
            hit = ((self.finger_of[sa] >= 0) & (self.owner[sb] >= 0)) | (
                (self.finger_of[sb] >= 0) & (self.owner[sa] >= 0)
            )
            if hit.any():
                self.mu_seen += [float(mu[hit].min()), float(mu[hit].max())]
        except Exception:
            pass

    def summary(self, outcome: str) -> dict[str, Any]:
        rows = self.rows
        out: dict[str, Any] = {
            "outcome": outcome,
            "steps": self.steps,
            "log_error": self.error,
            "held_rows": 0 if self.held is None else len(self.held),
            "cleanup_error": self.cleanup_error,
            "hooks_added": self.hooks_added,
            "step_hooks_added": self.step_hooks_added,
            "robot_equalities_on": sorted(self.eq_on),
            "suction_held": sorted(self.suction_held),
            "latched": sorted(self.latched),
            "thing_targets_moved": round(self.drive_moved, 6),
            "contacts_unread": dict(self.unread),
            "things": {},
        }
        if not rows:
            return out
        skip = self.skip  # the fingers' own time to open, for drop's reading
        for name in self.names + [p for p, _ in self.parts]:
            P = np.array([r["pose"][name] for r in rows])
            bad = ~np.isfinite(P).all(axis=1)
            if bad.any():  # a solver that blew up reads as nothing moving: say where it stopped being a number
                out.setdefault("nonfinite", {})[name] = int(np.argmax(bad))
                P = P[: int(np.argmax(bad))] if np.argmax(bad) > 0 else np.nan_to_num(P[:1])
            d = np.linalg.norm(P[:, :3] - P[0, :3], axis=1)
            t = {
                "moved_max_mm": round(float(d.max()) * 1000, 1),
                "end_mm": round(float(d[-1]) * 1000, 1),
                "start": [round(float(v), 4) for v in P[0, :3]],
                "end": [round(float(v), 4) for v in P[-1, :3]],
                "rise_max_mm": round(float((P[:, 2] - P[0, 2]).max()) * 1000, 1),
                "turn_end_deg": round(_turn_deg(P[0, 3:], P[-1, 3:]), 1),
                "end_quat": [round(float(v), 5) for v in P[-1, 3:]],
            }
            if name in self.names:
                i = self.names.index(name)
                hand = np.array([r["hand"][i] for r in rows])
                t["hand_force_p50_p95_N"] = [
                    [round(float(np.percentile(c[c > 0.5], q)), 1) for q in (50, 95)] if (c > 0.5).any() else [0.0, 0.0]
                    for c in hand.T
                ]
                t["overlap_with_robot_max_mm"] = round(float(max(r["depth"][i] for r in rows)) * 1000, 2)
                if name in self.unread:  # the fingers on it, in steps: its force is not read there
                    t["finger_touch_steps"] = [int(v) for v in np.array([r["touch"][i] for r in rows]).sum(axis=0)]
                if i in self.hidden_touch:
                    t["hidden_touch"] = sorted(self.hidden_touch[i])
                if self.mode == "drop" and self.hold_from:
                    after = rows[self.hold_from + skip : self.hold_from + skip + RELEASE_WINDOW]
                    t["robot_after_release_steps"] = int(sum(1 for r in after if r["robot_touch"][i]))
                    t["robot_touch_read"] = name not in self.unread or self.side is not None
                    t["robot_after_release_N"] = (
                        round(float(np.median([r["robot"][i] for r in after])), 2) if after else 0.0
                    )
                if name in self.unread and self.side_error:
                    t["flight_unread"] = self.side_error
                elif name in self.unread and self.side is not None and self.side_overflowed > 0.1 * len(rows):
                    t["flight_unread"] = (
                        f"the replay's collision pass overflowed its buffer on {self.side_overflowed} steps"
                    )
                elif name in self.free and t["moved_max_mm"] > FLIGHT_MM:
                    stepped = rows[1:]  # row 0 is the start, read without contacts
                    near = np.array([r["near"][i] == 0 and not r.get("overflow") for r in stepped])
                    span_all = np.maximum(np.diff(np.asarray(self.stamps, dtype=np.float64)), 1.0) * self.dt
                    pos = np.array([r["pose"][name][:3] for r in rows])
                    v = np.diff(pos, axis=0) / span_all[: len(pos) - 1, None]  # what the poses did, not what qd says
                    free = near[1:-1] & near[:-2] & near[2:] & (np.linalg.norm(v[1:-1], axis=1) > 0.01)
                    mid = 0.5 * (
                        span_all[1 : len(v) - 1] + span_all[2 : len(v)]
                    )  # between the two velocities' midpoints
                    acc = (v[2:] - v[1:-1]) / mid[:, None]
                    g = np.ravel(self.sim.model.gravity.numpy())[:3]
                    off = np.linalg.norm(acc - g, axis=1)
                    t["free_flight_steps"] = int(free.sum())
                    flying = free & (off > 3.0)
                    t["free_flight_not_falling"] = int(flying.sum())
                    # the longest run of flying reads, counted in steps (plain reads at yields that span several)
                    steps_of = np.maximum(np.diff(np.asarray(self.stamps, dtype=np.float64)), 1.0)[1 : 1 + len(flying)]
                    edges = np.diff(np.concatenate([[0], flying.astype(np.int8), [0]]))
                    t["free_flight_longest"] = int(
                        max(
                            (
                                steps_of[a:b].sum()
                                for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
                                if b - a >= 3  # three reads at least: one read across many steps misses a contact
                            ),
                            default=0,
                        )
                    )
            out["things"][name] = t
        # the carry: where each lifted body first reached 90% of its own height with the fingers on it; drop
        # releases at the earliest
        lifted = [k for k in self.names if k in self.single_free and out["things"][k]["rise_max_mm"] > CARRIED_MM]
        tops = {}  # thing -> the step it first reached 90% of its own height, with the fingers on it
        for k in lifted:
            i = self.names.index(k)
            z = np.array([r["pose"][k][2] for r in rows])
            good = np.isfinite(z)
            z = z[: int(np.argmin(good))] if not good.all() else z  # up to the first row that is not a number
            if len(z) < 2:
                continue
            rise = z - z[0]
            top = int(np.argmax(rise >= 0.9 * rise.max()))
            if sum(rows[top]["hand"][i]) > 0.5 or sum(rows[top]["touch"][i]) > 0:
                tops[k] = top
        if tops:
            # early in a carry (90% of its height), not over where it is set down: a thing dropped above its own goal
            # lands where it was going anyway; the earliest carry, where a weld or a write would start
            start = min(tops.values())
            at = rows[start]
            carried = [
                k
                for k in tops
                if sum(at["hand"][self.names.index(k)]) > 0.5 or sum(at["touch"][self.names.index(k)]) > 0
            ]
            out["apex_step"] = start
            out["carried"] = carried
            out["carried_later"] = sorted(k for k in tops if k not in carried)
            out["z_after"] = {
                k: [round(float(r["pose"][k][2]), 4) for r in rows[start : start + skip + 40]] for k in carried
            }
        out["release_skip"] = skip
        if self.mode == "drop" and self.hold_from:
            out["release_rows"] = max(0, len(rows) - 1 - self.hold_from)
            out["z_after"] = {
                k: [round(float(r["pose"][k][2]), 4) for r in rows[self.hold_from : self.hold_from + skip + 40]]
                for k in self.names
            }
        soft = {}
        for name in {k for r in rows for k in r["soft"]}:
            S = np.array([r["soft"][name] for r in rows if name in r["soft"]])
            c = S[:, :3]
            soft[name] = {
                "moved_max_mm": round(float(np.linalg.norm(c - c[0], axis=1).max()) * 1000, 1),
                "end_mm": round(float(np.linalg.norm(c[-1] - c[0])) * 1000, 1),
                "end": [round(float(v), 4) for v in c[-1]],
                "particle_moved_max_mm": round(float(S[:, 3].max()) * 1000, 1),
                # how far its particles end from where they started: a loop seated beside its start has its centre
                # near where it began, and its particles moved
                "particle_end_mm": round(float(S[-1, 4]) * 1000, 1),
            }
        out["deformables"] = soft
        out["jaw_mm"] = {
            a: [round(min(w) * 1000, 1), round(max(w) * 1000, 1)]
            for a in self.arms
            if (w := [r["jaw"][a] for r in rows[50:] if a in r["jaw"]] or [r["jaw"][a] for r in rows if a in r["jaw"]])
        }
        if self.mu_seen:
            out["finger_mu"] = [round(min(self.mu_seen), 3), round(max(self.mu_seen), 3)]
        return out


def _mimic_linkage(model: Any) -> dict[int, tuple[int, ...]]:
    """Each DOF an enabled mimic row ties to another, mapped to every DOF of its linkage (the robot entity's
    ``_geared_dofs``, read off the model): a finger an equality carries has the driven one's joint physics."""
    try:
        j0 = model.constraint_mimic_joint0.numpy()
        j1 = model.constraint_mimic_joint1.numpy()
        on = model.constraint_mimic_enabled.numpy() if hasattr(model, "constraint_mimic_enabled") else np.ones(len(j0))
        starts = model.joint_qd_start.numpy()
    except AttributeError:
        return {}
    parent: dict[int, int] = {}

    def root(j: int) -> int:
        parent.setdefault(j, j)
        while parent[j] != j:
            parent[j] = parent[parent[j]]
            j = parent[j]
        return j

    for a, b, e in zip(j0, j1, on, strict=False):
        if e and a >= 0 and b >= 0:
            parent[root(int(a))] = root(int(b))
    linkages: dict[int, list[int]] = {}
    for j in list(parent):
        linkages.setdefault(root(j), []).extend(range(int(starts[j]), int(starts[j + 1])))
    return {d: tuple(sorted(m)) for m in linkages.values() for d in m}


class AsBuilt:
    """The robot as built, for the length of a replay: friction at most :data:`REAL_MU`, the real arm's torque.

    ``heavier`` also scales every handled thing's mass and inertia; a thing with
    an infinite mass (kinematic, or welded) is left so. Only the entries changed
    are saved, and :meth:`restore` writes those back into the arrays as they are
    then, so a slider moved during the replay keeps what it set elsewhere; the
    solvers are told once each way, as :mod:`robosimstudio.core.tuning` tells
    them for a slider. :meth:`restore` puts back whatever :meth:`apply` got as
    far as changing.

    Not :class:`~robosimstudio.core.tuning.BodyWriter`: that sets one entity's
    mass to a value and holds only a massless body fixed, where this scales
    every handled thing at once, must also leave a body flagged ``KINEMATIC``
    (which keeps its mass) alone, and must put back exactly the rows it wrote
    rather than divide its way back.
    """

    MASS: ClassVar[tuple[str, ...]] = ("body_mass", "body_inv_mass", "body_inertia", "body_inv_inertia")
    FLAGS: ClassVar[dict[str, int]] = {
        "shape_material_mu": _SHAPE_PROPERTIES,
        "joint_effort_limit": _JOINT_DOF,
        "joint_target_ke": _JOINT_DOF,
        "joint_target_kd": _JOINT_DOF,
        "joint_damping": _JOINT_DOF,
        "joint_friction": _JOINT_DOF,
        "joint_armature": _JOINT_DOF,
    }

    def __init__(self, sim: Any, heavier: float = 1.0) -> None:
        self.sim, self.heavier = sim, float(heavier)
        self.saved: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        """Per array: each change's rows and what they held, in the order made (put back in reverse)."""
        self.lowered: dict[str, int] | None = None
        """How many shapes' friction and arm joints' torque :meth:`apply` lowered, once it has finished: none at all,
        and the scene was already as built."""

    def _change(self, name: str, rows: np.ndarray, new: np.ndarray) -> None:
        array = getattr(self.sim.model, name)
        values = array.numpy()
        self.saved.setdefault(name, []).append((rows, values[rows].copy()))
        values[rows] = new
        array.assign(values)
        tuning.reach_island_copies(self.sim.pipeline, name)  # an island that keeps its own copy sees it too

    def apply(self) -> None:
        m = self.sim.model
        mu = m.shape_material_mu.numpy()
        grippy = np.flatnonzero(mu > REAL_MU)
        flags = _SHAPE_PROPERTIES if len(grippy) else 0
        if len(grippy):
            self._change("shape_material_mu", grippy, np.full(len(grippy), REAL_MU, dtype=mu.dtype))
        lowered = {"friction": int(len(grippy)), "torque": 0}
        dofs, caps = [], []
        for arm in self.sim.scene.handles.arms.values():
            limits = real_arm_torque(self.sim, arm)
            if limits is not None:
                dofs += list(arm.arm_dofs[: len(limits)])
                caps += list(limits[: len(arm.arm_dofs)])
        if dofs:
            effort = m.joint_effort_limit.numpy()
            rows = np.asarray(dofs, dtype=np.int64)
            real = np.asarray(caps, dtype=effort.dtype)
            # what is written is what is counted, so "lowered nothing" is an unchanged replay; Newton hands this limit
            # to MuJoCo as the joint's actfrcrange, which clamps every actuator on it
            cut = real < effort[rows]
            lowered["torque"] = int(np.count_nonzero(cut))
            if cut.any():
                self._change("joint_effort_limit", rows[cut], real[cut])
                flags |= _JOINT_DOF
        # the registered hand's own servo, where the twin set its own (finger DOFs counted)
        lowered["servo"] = 0
        changed: set[int] = set()
        for name, (rows, values) in self._registered_gripper().items():
            now = getattr(m, name).numpy()[rows]
            differ = ~np.isclose(now, values.astype(now.dtype), rtol=1e-5, atol=1e-6)
            if differ.any():
                self._change(name, rows[differ], values[differ].astype(now.dtype))
                changed.update(int(r) for r in rows[differ])
                flags |= _JOINT_DOF
        lowered["servo"] = len(changed)
        if self.heavier != 1.0:
            fixed = _fixed_bodies(m)
            rows = np.asarray(
                [b for t in things(self.sim).values() for b in t["bodies"] if not fixed[b]], dtype=np.int64
            )
            if len(rows):  # a body of infinite mass stays so: giving it an inverse mass would set it free
                for name in self.MASS:
                    scale = self.heavier if name in ("body_mass", "body_inertia") else 1.0 / self.heavier
                    self._change(name, rows, getattr(m, name).numpy()[rows] * scale)
                flags |= _BODY_INERTIAL
        if flags:
            tuning.notify(self.sim.pipeline, flags)
        self.lowered = lowered

    def _registered_gripper(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Where a finger's servo is the twin's own rather than the registered hand's: per joint array, the rows and
        the registered servo's values -- gains and force limit on the driven DOFs, damping, friction and armature on
        their whole linkage, as the robot's own build writes them -- for the library's robot it is
        (:func:`shipped_robot`). Empty where it is the registered servo or no robot the library ships. What the
        registered servo leaves to the file (a limit it does not state) cannot be put back, nor limits the twin's
        servo widened; the ``robot`` line fails any of it."""
        emb = getattr(self.sim.scene, "embodiment", None)
        spec = getattr(emb, "spec", None)
        if spec is None or getattr(spec, "asset", None) is None:
            return {}
        reg, _ = shipped_robot(spec)
        if reg is None:
            return {}
        out: dict[str, tuple[list[int], list[float]]] = {}

        def put(name: str, rows: Any, value: float) -> None:
            got = out.setdefault(name, ([], []))
            for r in rows:
                got[0].append(int(r))
                got[1].append(float(value))

        linkage = _mimic_linkage(self.sim.model)
        for i, arm in enumerate(spec.arms):
            theirs = _registered_arm(reg, arm, i)
            handle = self.sim.scene.handles.arms.get(arm.name)
            if theirs is None or (arm.gripper or spec.gripper, arm.gripper_mount) != (
                theirs.gripper or reg.gripper,
                theirs.gripper_mount,
            ):
                continue  # another hand: the registered one's servo is not its own (the robot line fails the swap)
            act = _hand_servo(reg, theirs)
            if handle is None or act is None or _hand_servo(spec, arm) == act:
                continue
            dofs = [int(d) for d in handle.gripper_dofs]
            for k, dof in enumerate(dofs):
                put("joint_target_ke", [dof], act.ke_for(k))
                put("joint_target_kd", [dof], act.kd_for(k))
                if act.effort_for(k):
                    put("joint_effort_limit", [dof], act.effort_for(k))
            tied = sorted({t for d in dofs for t in linkage.get(d, (d,))})
            if act.joint_damping is not None:
                put("joint_damping", tied, act.joint_damping)
            if act.joint_friction is not None:
                put("joint_friction", tied, act.joint_friction)
            if act.armature:
                put("joint_armature", tied, act.armature)
            if act.armature_exact is not None:  # after armature, the driven DOFs only, as the build writes it
                put("joint_armature", dofs, act.armature_exact)
        return {k: (np.asarray(r, np.int64), np.asarray(v)) for k, (r, v) in out.items()}

    def restore(self) -> None:
        flags = 0
        for name, changes in self.saved.items():
            array = getattr(self.sim.model, name)
            values = array.numpy()
            for rows, old in reversed(changes):
                values[rows] = old
            array.assign(values)
            tuning.reach_island_copies(self.sim.pipeline, name)
            flags |= self.FLAGS.get(name, _BODY_INERTIAL)
        self.saved = {}
        if flags:
            tuning.notify(self.sim.pipeline, flags)


def cap_limit(steps: int) -> int:
    """How many steps a replay other than ``written`` may run: the written run's and a little past it."""
    return steps + max(25, int(OVERRUN * steps))


def replay(
    sim: Any,
    routine: Iterator,
    mode: str,
    store: dict,
    *,
    steps: int = 0,
    steps_itself: bool = False,
    drive_rows: list[int] | None = None,
    robot_equalities: dict | None = None,
    drop_at: int = 0,
) -> Iterator[None]:
    """The routine (the generator it returned) played under ``mode``, its summary in ``store[mode]`` however it ends.

    A replay other than ``written`` runs as long as the written run,
    ``steps``: on past the routine's end when it stops early (a thing that
    drifts late still shows), and when it is still going, a little past
    (:data:`OVERRUN`, so a routine whose last lines run on the step after the
    written one's did is not cut off there) and no further: a routine waiting
    for a pose the frozen arm never reaches is stopped and read as not done. One that ``steps_itself`` is stepped here once it has
    stopped; one driving the arm panels has the page step it, as it always
    does. A replay the page stops (a reset, a save, Stop) is stored as
    :data:`STOPPED`, one that fails itself as :data:`FAILED`, and the model is
    put back whatever ended it.
    """
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {', '.join(MODES)}")
    if mode == "plain":
        yield from _plain(sim, routine, store, steps)
        return
    capped = mode != "written"
    limit = cap_limit(steps)
    long = f"still going when the written replay had long ended ({steps} steps)"

    def stop() -> None:
        """The hold and the cap off first, then the routine's own cleanup, now and on this scene. A cleanup that
        fails (or yields in its finally) does not change why the replay stopped."""
        if tap is not None:
            tap.remove()
        try:
            routine.close()
        except Exception as exc:
            if tap is not None:
                tap.cleanup_error = f"{type(exc).__name__}: {exc}"

    tap: Replay | None = None
    built = AsBuilt(sim, HEAVY if mode == "heavy" else 1.0) if mode in ("built", "heavy") else None
    outcome = "finished"
    try:
        tap = Replay(sim, mode, drive_rows=drive_rows, model_rows=robot_equalities, hold_from=drop_at)
        if capped:
            # a routine may loop on `sim.run` until a pose is reached, never yielding: the cap is enforced at a step
            # the routine takes itself, raised into it; a step the page takes between yields is cut at the yield
            tap.limit = limit
            tap.grace = max(25, int(OVERRUN * steps))
        if built is not None:
            built.apply()
        tap.install()
        try:
            while True:
                tap.inside = True  # a step taken now is the routine's own: the cap and a broken hold raise into it
                try:
                    next(routine)
                except StopIteration:
                    break
                finally:
                    tap.inside = False
                yield
                if tap.broken:
                    raise Broken(tap.broken)
                if capped and tap.steps >= limit:
                    outcome = long
                    break
        except Overrun:
            outcome = long
        except Exception as exc:  # the routine's own checks stopping it is the reading, not a failure of this
            outcome = f"{type(exc).__name__}: {str(exc)[:300]}"
        if capped and tap.overrun_at and outcome == "finished":  # it caught the cap and returned
            outcome = long
        if outcome == long:
            stop()  # outside the try above: a failure here is the replay's, not the routine's
        tap.limit = 0
        while capped and tap.steps < steps:
            if steps_itself:
                sim.run(1)
            yield
            if tap.broken:
                raise Broken(tap.broken)
    except GeneratorExit:
        outcome = STOPPED
        stop()
        raise
    except Broken as exc:
        outcome = f"{FAILED}: {exc}"
        stop()
    except Exception as exc:
        outcome = f"{FAILED}: {type(exc).__name__}: {str(exc)[:300]}"
        stop()
        raise
    finally:
        if tap is not None:
            tap.remove()
        if built is not None:
            built.restore()
        store[mode] = tap.summary(outcome) if tap is not None else {"outcome": outcome, "steps": 0, "things": {}}
        if built is not None and built.lowered is not None:
            store[mode]["lowered"] = dict(built.lowered)


_ABSENT = object()


def _plain(sim: Any, routine: Iterator, store: dict, steps: int = 0) -> Iterator[None]:
    """The routine played with nothing hooked in -- no cap, no hold, no hook on any list -- and read only at its own
    yields (and once more when it returns): where the things are, what touches them.

    What a routine that looks for the replay's hooks cannot tell from a play. It is cut at a yield a little past the
    written run's length; one the page stops past that length (``./ctl physics``'s wait running out) is read as not
    ending unwatched, one stopped before it is played again. A routine that steps without ever yielding is read at its
    start and its end; its stepping is capped where it calls ``sim.run`` or ``sim.step``, but one call for many steps
    runs to its end.
    """
    start = int(getattr(sim, "step_count", 0) or 0)
    limit = cap_limit(steps) if steps else 0
    outcome = "finished"
    reader = None

    def past_limit() -> bool:
        return bool(limit) and int(getattr(sim, "step_count", 0) or 0) - start > limit

    def read() -> None:
        reader.steps += 1
        try:
            reader.log()
        except Exception:
            reader.error = reader.error or traceback.format_exc()[-800:]

    wrapped: dict[str, Any] = {}  # attribute -> what the sim's own __dict__ held there before (or _ABSENT)
    grace = max(25, int(OVERRUN * steps))
    cut_at = [0]  # the step plain was first cut at: the routine's own cleanup then has `grace` steps

    def cap(fn: Any) -> Any:
        def capped(*a: Any, **kw: Any) -> Any:
            now = int(getattr(sim, "step_count", 0) or 0)
            if past_limit() and (not cut_at[0] or now >= cut_at[0] + grace):
                cut_at[0] = cut_at[0] or now
                raise Overrun(now - start)
            return fn(*a, **kw)

        return capped

    if limit:
        for attr in ("run", "step"):  # instance attributes, gone in the finally: the class's own methods are untouched
            fn = getattr(sim, attr, None)
            if fn is None:
                continue
            try:
                before = vars(sim).get(attr, _ABSENT)
                setattr(sim, attr, cap(fn))
                wrapped[attr] = before
            except (AttributeError, TypeError):  # a sim that takes no attribute: cut at yields only
                pass
    try:
        try:
            reader = Replay(sim, "plain")
            reader.kept = [fn for hooks in sim.hook_lists().values() for fn in hooks]
            reader.hooks0 = {id(fn) for fn in reader.kept}
            reader.log(contacts=False)
        except Exception as exc:  # the replay's own fault: not the routine's reading
            raise Broken(f"setting up plain: {type(exc).__name__}: {exc}") from exc
        while True:
            try:
                next(routine)
            except StopIteration:
                if int(getattr(sim, "step_count", 0) or 0) != (reader.stamps[-1] if reader.stamps else start):
                    read()  # the steps it took after its last yield
                break
            yield
            read()
            if past_limit():  # read at its yields, cut there
                outcome = PLAIN_LONG
                try:
                    routine.close()
                except Exception:
                    pass
                break
    except Broken as exc:
        outcome = f"{FAILED}: {exc}"[:300]
    except Overrun:  # the cap on its own stepping
        outcome = PLAIN_LONG
        try:
            routine.close()
        except BaseException:
            pass
    except GeneratorExit:
        outcome = PLAIN_LONG if past_limit() else STOPPED
        try:
            routine.close()
        except Exception:
            pass
        raise
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {str(exc)[:300]}" if reader is not None else f"{FAILED}: {exc}"[:300]
    finally:
        for attr, before in wrapped.items():
            try:
                if before is _ABSENT:
                    delattr(sim, attr)
                else:
                    setattr(sim, attr, before)
            except AttributeError:
                pass
        store["plain"] = (
            reader.summary(outcome) if reader is not None else {"outcome": outcome, "steps": 0, "things": {}}
        )
        store["plain"]["sim_steps"] = int(getattr(sim, "step_count", 0) or 0) - start


# --------------------------------------------------------------------------- the verdict


def _moved(run: dict, floor: float = 20.0) -> list[str]:
    """What the run moved: things, a jointed thing's parts (a lid, a door, a key), deformables."""
    hit = [k for k, t in run.get("things", {}).items() if t["moved_max_mm"] > floor]
    return hit + [k for k, t in run.get("deformables", {}).items() if t["moved_max_mm"] > floor]


class _Miss(NamedTuple):
    """One way a run did not end where the written run left things (:func:`_misses`)."""

    thing: str
    kind: str
    """``knocked``, ``lost`` or ``end``."""
    far: dict[str, float]
    """For an ``end`` miss: ``position``, ``shape`` and ``turn``, each in tolerances."""
    amount: dict[str, float]
    """The same in mm, mm and degrees."""
    words: str
    tol: dict[str, float]
    """The tolerances it was read with, by measure (empty for ``knocked`` and ``lost``)."""


def _misses(
    run: dict,
    written: dict,
    names: list[str],
    mm: float = END_MM,
    deg: float = END_DEG,
    round_axes: dict | None = None,
    bystanders: bool = True,
    excursion: bool = True,
) -> list[_Miss]:
    """Each way ``run`` did not end where the written run left things, as :class:`_Miss`:
    ``knocked``, a thing the written run left alone moved more than ``mm``; ``lost``, a handled thing carried less than
    a third as far as written (a mug lifted and set back near its start ends there too); ``end``, a handled thing ended
    more than ``mm`` off, its shape ``mm / 2`` unlike, or turned more than ``deg``. ``far`` is, for ``end``, each of
    ``position``, ``shape`` and ``turn`` in tolerances, and empty for the others. One entry a thing."""
    out = []
    for k in {**written.get("things", {}), **written.get("deformables", {})} if bystanders else ():
        a = _entry(run, k)
        if k not in names and a and a.get("moved_max_mm", 0.0) > mm:  # left alone as written, knocked away here
            moved = a["moved_max_mm"]
            out.append(
                _Miss(k, "knocked", {}, {}, f"{k}, which the written run left alone, was moved {moved:.0f} mm", {})
            )
    for k in names:
        a = _entry(run, k)
        b = _entry(written, k)
        if not a or not b:
            continue
        if excursion and not _carried(run, written, k):
            words = (
                f"{k} moved {a.get('moved_max_mm', 0.0):.0f} mm where the written run moved it {b['moved_max_mm']:.0f}"
            )
            out.append(_Miss(k, "lost", {}, {}, words, {}))
            continue
        amount = _end_offset(a, b, (round_axes or {}).get(k))
        tol = {"position": mm, "shape": mm / 2, "turn": deg}
        far = {d: amount[d] / tol[d] for d in tol}
        first = next((d for d in ("position", "shape", "turn") if far[d] > 1.0), None)
        if first:
            out.append(
                _Miss(
                    k,
                    "end",
                    far,
                    amount,
                    _miss_words(k, first, amount[first]),
                    tol,
                )
            )
    return out


def _miss_words(k: str, measure: str, amount: float) -> str:
    """A thing's end missed by one measure, in a few words: ``position`` [mm], ``shape`` or ``turn`` [deg]."""
    return {
        "position": f"{k} ended {amount:.0f} mm from where the written run left it",
        "shape": f"{k}'s shape ended unlike the written run's",
        "turn": f"{k} ended turned {amount:.0f}° from how the written run left it",
    }[measure]


def _ended_elsewhere(
    run: dict,
    written: dict,
    names: list[str],
    mm: float = END_MM,
    deg: float = END_DEG,
    round_axes: dict | None = None,
    bystanders: bool = True,
    excursion: bool = True,
) -> str:
    """What did not end where the written run left it, said in a few words (the first of :func:`_misses`); ``""``
    when everything did."""
    misses = _misses(run, written, names, mm, deg, round_axes, bystanders, excursion)
    return misses[0].words if misses else ""


def _entry(run: dict, k: str) -> dict | None:
    """A run's record of thing ``k``, rigid or deformable."""
    return run.get("things", {}).get(k) or run.get("deformables", {}).get(k)


def cut_short(run: dict) -> str:
    """Why a replay did not run to a reading, ``""`` when it did: the page cut it (a reset, a save, Stop), its own
    machinery broke (``FAILED``), or its log failed (rows are missing, and it would read as nothing having moved). The
    server reports nothing while any replay is cut short, and says this."""
    if run.get("log_error"):
        lines = str(run["log_error"]).strip().splitlines()
        return f"its log failed: {lines[-1] if lines else 'no message'}"
    out = str(run.get("outcome", ""))
    return out if out.startswith((STOPPED, FAILED)) else ""


def _unread(run: dict) -> str:
    """Why a replay was not read, ``""`` when it was: the solver blew up in it, or it was :func:`cut_short` (which a
    caller of :func:`verdict` other than the server may still hand in)."""
    if run.get("nonfinite"):
        return "not read: the solver blew up in it"
    why = cut_short(run)
    return f"not read: {why}" if why else ""


def _carried(run: dict, written: dict, k: str, sampled: bool = False) -> bool:
    """Whether ``run`` carried thing ``k`` at least a third as far as the written run did: a mug lifted and set back
    near its start is carried too; one a grasp let go of, or never took, is not. A ``sampled`` run (plain, read only
    at the routine's yields, so its largest move can fall short) also counts as carrying a thing it left at least a
    third of the way from its start toward where the written run left it."""
    a, b = _entry(run, k) or {}, _entry(written, k) or {}
    if a.get("moved_max_mm", 0.0) >= b.get("moved_max_mm", 0.0) / 3:
        return True
    if sampled and "start" not in a:  # a deformable records no start: how far from it plain left it, against written
        return b.get("end_mm", 0.0) > 0.0 and a.get("end_mm", 0.0) >= b.get("end_mm", 0.0) / 3
    if not (sampled and "end" in a and "end" in b):
        return False
    # it left the thing a third of the way from its start toward the written end: shoved aside is not carried
    far = float(np.linalg.norm(np.subtract(a["start"], b["end"])))
    return far > 0.0 and far - float(np.linalg.norm(np.subtract(a["end"], b["end"]))) >= far / 3


def _end_offset(a: dict, b: dict, axis: list[float] | None = None) -> dict[str, float]:
    """How far one run's end of a thing is from another's, by measure: ``position`` [mm], ``shape`` (a deformable's
    particles) [mm] and ``turn`` [deg]; a thing round about ``axis`` counts only a tilt of that axis, and a ball
    (``[0, 0, 0]``) no turn."""
    off = float(np.linalg.norm(np.subtract(a["end"], b["end"])) * 1000)
    shape = abs(a.get("particle_end_mm", 0.0) - b.get("particle_end_mm", 0.0))
    if axis is not None and not any(axis):  # a ball: no turn of it is a different end
        turn = 0.0
    elif axis is not None and "end_quat" in a and "end_quat" in b:  # round about it: only a tilt of the axis counts
        ua = compose_many(np.array([[0, 0, 0, *a["end_quat"]]]), np.array([[*axis, 0, 0, 0, 1.0]]))[0, :3]
        ub = compose_many(np.array([[0, 0, 0, *b["end_quat"]]]), np.array([[*axis, 0, 0, 0, 1.0]]))[0, :3]
        turn = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(ua, ub))))))
    elif "end_quat" in a and "end_quat" in b:  # the turn between the two ends, not between the sizes of two turns
        turn = _turn_deg(np.asarray(a["end_quat"]), np.asarray(b["end_quat"]))
    else:
        turn = abs(a.get("turn_end_deg", 0.0) - b.get("turn_end_deg", 0.0))
    return {"position": off, "shape": shape, "turn": turn}


def _within_spread(
    misses: list[_Miss], run: dict, plain: dict, seen: list[_Miss], written: dict, axes: dict | None = None
) -> str:
    """Plain's evidence, in a few words, that a run's ``misses`` are no more than the scene's own spread; ``""`` when
    they are more. Plain changed nothing, so it and the written run are two samples of the scene, and ``seen`` is
    plain's own misses. The run's misses are spread when each is an end (not a thing lost on the way or a bystander
    knocked: a grip that let go), and for each thing and each of position, shape and turn the run missed by, plain
    carried that same thing (:func:`_carried`) and missed it by that same measure too, the run missed it by no more
    than plain did plus one tolerance, and it ended no farther from plain's end than plain's end is from the written
    one (or within the tolerance of it): a slip the other way from plain's spread is not spread, nor is one twice as
    far, nor one where plain's grasp failed."""
    if not misses or any(m.kind != "end" for m in misses):
        return ""
    by = {m.thing: m for m in seen if m.kind == "end"}
    for m in misses:
        p = by.get(m.thing)
        # plain is a sample of where the thing lands only if it carried it (a grasp that failed there is no spread)
        if p is None or not _carried(plain, written, m.thing, sampled=True):
            return ""
        tol = m.tol
        apart = _end_offset(_entry(run, m.thing), _entry(plain, m.thing), (axes or {}).get(m.thing))
        for dim, v in m.far.items():
            if v > 1.0 and (
                p.far[dim] <= 1.0
                or m.amount[dim] > p.amount[dim] + tol[dim]  # about as far as plain missed, not twice
                or apart[dim] > max(tol[dim], p.amount[dim])
            ):
                return ""
    worst = max(misses, key=lambda m: max(m.far.values()))
    dim = max(worst.far, key=worst.far.get)  # the measure the run missed most by, as plain missed it
    return _miss_words(worst.thing, dim, by[worst.thing].amount[dim])


def _robot_line(model: dict, line: Any) -> None:
    """``robot``: the robot is the registered one -- its own file, hands, servos and physics -- placed with ``at=``
    and ``arms_at=``."""
    spec = model.get("robot_spec")
    if not spec:
        return
    if not spec.get("registered"):
        line(
            None,
            "robot",
            f"{spec.get('name') or 'the robot'} is built from {spec.get('file') or 'a file'}, which "
            "the library does not ship: its servo gains and limits are the file's own and are not checked",
        )
        return
    gone = [spec[k] for k in ("file", "arm", "hand", "gripper", "other") if spec.get(k)]
    known = spec.get("as") or spec["name"]
    if gone:
        line(
            False,
            "robot",
            f"{spec['name']} is not the registered {known}: "
            + "; ".join(gone)
            + ": use the registered robot, placed with at= (and arms_at= for each arm)",
        )
    else:
        line(True, "robot", f"the registered {known}, its own file, hands and servos")


def _model_lines(
    model: dict,
    written: dict,
    names: list[str],
    line: Any,
    settle_mm: float = SETTLE_MM,
    frozen: dict | None = None,
) -> None:
    """The ``model`` line (gravity, densities, invisible props) and the ``settle`` line (how the scene's own settle,
    before the first step, moved each thing from where it was put)."""
    bad, notes = [], []
    g = model.get("gravity")
    if g is not None and abs(g - GRAVITY) > 0.02 * GRAVITY:
        bad.append(f"gravity is {g:g} m/s², not {GRAVITY:g}")
    lo, hi = DENSITY
    for k in dict.fromkeys(n.split("#")[0] for n in names):
        t = model.get("things", {}).get(k, {})
        rho, rho_max = t.get("density"), t.get("density_max", t.get("density"))
        light = rho_max is not None and rho_max < lo  # light even by its largest collider alone
        heavy = rho is not None and rho > hi  # heavy even over every collider's volume
        if light or heavy:
            rho = rho_max if light else rho
            notes.append(
                f"{k} weighs {t['mass_kg'] * 1000:g} g in {t['volume_l']:g} l ({rho:g} kg/m³, "
                + ("lighter than expanded polystyrene" if light else "heavier than lead")
                + "): say what it is"
            )
    note_mm, fail_mm = DRAWN_ONLY_MM
    for k, reach in sorted((model.get("drawn_only_mm") or {}).items(), key=lambda kv: -kv[1]):
        if reach > fail_mm:
            bad.append(
                f"{k}: drawn parts that do not collide reach {reach:g} mm outside its colliders, and pass through "
                "what they meet: give them colliders or draw them inside"
            )
        elif reach > note_mm:
            notes.append(
                f"{k}: drawn parts that do not collide reach {reach:g} mm outside its colliders: say what they are"
            )
    touched = {h for t in written.get("things", {}).values() for h in t.get("hidden_touch", [])}
    if touched:
        notes.append(
            f"{', '.join(sorted(touched))}: a collider nothing draws touched a handled thing: say what it is in the clip"
        )
    if bad or notes:
        line(False if bad else None, "model", "; ".join(bad + notes))
    elif model.get("gravity") is not None:
        line(True, "model", f"gravity {model['gravity']:g} m/s²; the handled things' densities are ones objects have")
    settle = model.get("settle") or {}
    if settle:
        proof = frozen if frozen and not _unread(frozen) else {}  # a frozen run that was not read proves nothing
        risen = {k: v for k, v in settle.items() if v["rise_mm"] > settle_mm}
        pushed_out = {k for k, t in (proof.get("things") or {}).items() if t.get("moved_max_mm", 0.0) > settle_mm}
        resting = {
            k: v for k, v in settle.items() if OVERLAP_MM < v.get("overlap_mm", 0.0) <= SINK_MM and k not in pushed_out
        }
        inside = {k: v for k, v in settle.items() if v.get("overlap_mm", 0.0) > OVERLAP_MM and k not in resting}
        rolled = {k: v for k, v in settle.items() if v["aside_mm"] > 1.5 * settle_mm and abs(v["rise_mm"]) <= settle_mm}
        if inside:
            k = max(inside, key=lambda n: inside[n]["overlap_mm"])
            line(
                False,
                "settle",
                f"{k} starts {inside[k]['overlap_mm']:.1f} mm inside {inside[k].get('inside', 'something')}",
            )
        elif risen:  # with nothing inside anything now: pushed out by the warm-up, or put there by the reset
            k = max(risen, key=lambda n: risen[n]["rise_mm"])
            v = risen[k]
            line(
                None,
                "settle",
                f"{k} starts {v['rise_mm']:.1f} mm higher than it was put: pushed out of something before the first "
                "step, or put back higher by the reset: say which",
            )
        elif resting:
            k = max(resting, key=lambda n: resting[n]["overlap_mm"])
            line(
                None,
                "settle",
                f"{k} rests {resting[k]['overlap_mm']:.1f} mm into {resting[k].get('inside', 'its support')}"
                + (
                    ": play frozen to see whether it is pushed out"
                    if frozen is None
                    else " and stays there with the robot frozen: a contact that soft, say why"
                    if not _unread(frozen)
                    else ": the frozen replay was not read whole, so whether it is pushed out is not known"
                ),
            )
        elif rolled:
            k = max(rolled, key=lambda n: rolled[n]["aside_mm"])
            line(
                None,
                "settle",
                f"the scene's settle moved {k} {rolled[k]['aside_mm']:.1f} mm aside before the first step: put inside "
                "something it slid out of, or round and rolled: say which",
            )
        elif model.get("settle_error"):
            line(None, "settle", f"not read: {model['settle_error']}")
        else:
            worst = max(abs(v["rise_mm"]) + v["aside_mm"] for v in settle.values())
            deep = max(v.get("overlap_mm", 0.0) for v in settle.values())
            line(
                True,
                "settle",
                f"nothing starts inside anything (deepest {deep:.1f} mm); the scene's settle moved nothing more than "
                f"{worst:.1f} mm",
            )


def _drop_line(model: dict, runs: dict, written: dict, line: Any, axes: dict | None = None) -> None:
    """``drop``: the fingers opened at the top of the written run's carry; what they carried should fall."""
    drop = runs.get("drop")
    if not drop:
        return
    carried = written.get("carried") or []
    if written.get("outcome") != "finished":
        line(None, "drop", "not read: the written replay did not finish")
        return
    if not carried:
        if soft := [k for k in _moved(written) if k in written.get("deformables", {})]:
            line(
                None,
                "drop",
                f"not read: what the routine moved is a deformable ({soft[0]}), and drop follows a rigid thing",
            )
        else:
            line(True, "drop", "nothing was lifted off its support, so nothing to release")
        return
    if drop.get("held_rows") == 0:
        line(None, "drop", "not read: this hand has no fingers to open")
        return
    if why := _unread(drop):
        line(None, "drop", why)
        return
    if drop.get("release_rows", 0) < int(drop.get("release_skip", 10)) + RELEASE_WINDOW // 2:
        line(None, "drop", "not read: the carry's top comes too near the end of the run to watch a fall")
        return

    def fell(k: str) -> bool:
        """Lower than the written run at the same moment after the release (a carried thing rides along; a dropped one
        falls at g), or ending somewhere else than the written run left it."""
        a, b = (drop.get("z_after") or {}).get(k, []), (written.get("z_after") or {}).get(k, [])
        n = min(len(a), len(b))
        skip = int(drop.get("release_skip", 10))
        start = (written.get("things", {}).get(k) or {}).get("start")
        height = (b[0] - start[2]) if (b and start) else 0.015
        if n > skip + 5 and min(np.subtract(a[skip:n], b[skip:n])) < -min(0.015, 0.5 * max(height, 0.0)):
            return True
        return bool(_ended_elsewhere(drop, written, [k], round_axes=axes, bystanders=False))

    stayed = [k for k in carried if not fell(k)]
    later = written.get("carried_later") or []
    if later and not stayed:
        line(
            None,
            "drop",
            f"with the fingers opened at the top of the first carry (step {written.get('apex_step', 0)}) "
            f"{', '.join(carried)} fell; {', '.join(later)}, carried later, was not released: one drop tests the "
            "carries under way at its release",
        )
        return
    if not stayed:
        line(
            True,
            "drop",
            f"with the fingers opened at the top of the carry (step {written.get('apex_step', 0)}) {', '.join(carried)} fell",
        )
        return
    unread = written.get("contacts_unread") or {}

    unknown = [k for k in stayed if not drop["things"].get(k, {}).get("robot_touch_read", True)]
    stayed = [k for k in stayed if k not in unknown]  # judged below; the unknown ones are said after a fail

    def held_by_robot(k: str) -> bool:
        """Touched, and (where the force is read) pressed hard enough to bear the thing's weight between them."""
        t = drop["things"].get(k, {})
        if t.get("robot_after_release_steps", 0) < RELEASE_WINDOW // 2:  # touched for most of the moments after it
            return False
        weight = float((model.get("things", {}).get(k) or {}).get("mass_kg", 0.0)) * GRAVITY
        return k in unread or t.get("robot_after_release_N", 0.0) >= 0.5 * weight

    touched = [k for k in stayed if held_by_robot(k)]
    free = [k for k in stayed if k not in touched]
    unread_note = (
        f"; {', '.join(unknown)} also stayed, and what touches it after the release is not read (its contacts are "
        "another solver's)"
        if unknown
        else ""
    )
    if not free and not touched and unknown:
        line(
            None,
            "drop",
            f"with the fingers opened at the top of the carry, {', '.join(unknown)} still ended where the written run "
            "left it, and what touches it after the release is not read (its contacts are another solver's): say what "
            "holds it",
        )
        return
    if free:
        line(
            False,
            "drop",
            f"with the fingers opened at the top of the carry, {', '.join(free)} still ended where the written run left it, touching nothing of the robot: something that is not a contact carries it"
            + unread_note,
        )
    else:
        line(
            None,
            "drop",
            f"with the fingers opened at the top of the carry, {', '.join(touched)} was still carried by the robot (two hands squeezing, a hook): say what holds it"
            + unread_note,
        )


def _held_things(model: dict, run: dict) -> tuple[set[str], bool]:
    """What held things to the robot or moved them in ``run``: a tie or a kinematic flag always, a grasp latch, an
    equality row or a motor on a thing's joint as ``run`` recorded them (only the written replay records those; pass
    ``{}`` for the first two alone); and whether a substep hook, which names no thing, was added there. From the names
    the model facts carry."""
    labels = model.get("robot_equalities", {})
    held = set(model.get("tie_things", []))
    held |= set(model.get("kinematic", [])) | set(run.get("latched", []))
    if run.get("thing_targets_moved", 0.0) > 1e-6:
        held |= set(model.get("drive_things", []))
    rows = [k for k in run.get("robot_equalities_on", []) if "suction" not in labels.get(str(k), "")]
    held |= {model.get("robot_equality_things", {}).get(str(k), "") for k in rows} - {""}
    return held, bool(run.get("hooks_added", 0) or run.get("step_hooks_added", 0))


def _plain_gate(runs: dict, written: dict) -> tuple[dict | None, str]:
    """The plain replay, and why its end cannot be read (the plain line's NOTE), ``""`` when it can: both it and the
    written run finished and nothing blew up. ``(None, "")`` when there is no plain replay."""
    plain = runs.get("plain")
    if not plain:
        return None, ""
    if written.get("outcome") != "finished":
        return plain, "not read: the written replay did not finish"
    if plain.get("nonfinite"):
        return plain, _unread(plain)
    if plain["outcome"] == PLAIN_SLOW:
        return plain, f"not read: {PLAIN_SLOW}"
    if why := _unread(plain):
        return plain, why
    if plain["outcome"] != "finished":  # a routine flaky run to run stops too: a note, not proof of anything
        return plain, (
            f"played with nothing watching, the routine did not finish as the replay did ({plain['outcome']}): it is "
            "not repeatable, or it does something else when it is replayed: say which"
        )
    return plain, ""


def _flew(run: dict) -> dict[str, int]:
    """The things a run carried free of every contact, not falling at g, for :data:`FLIGHT_STEPS` steps or more."""
    return {
        k: t.get("free_flight_longest", 0)
        for k, t in run.get("things", {}).items()
        if t.get("free_flight_longest", 0) >= FLIGHT_STEPS
    }


class _PlainReading(NamedTuple):
    """Plain read once (:func:`_plain_reading`)."""

    run: dict | None
    why: str
    flew: dict[str, int]
    misses: list[_Miss]

    @property
    def readable(self) -> bool:
        """Its end can be read as a sample of the scene: it finished as written did and carried nothing without
        contact."""
        return self.run is not None and not self.why and not self.flew


def _plain_reading(runs: dict, written: dict, names: list[str], axes: dict | None = None) -> _PlainReading:
    """Plain read once for the plain line and for built's spread: the run (``None`` without one), why its end cannot be
    read (:func:`_plain_gate`), what it carried without contact (:func:`_flew`), and its misses (handled things only,
    none when it flew)."""
    plain, why = _plain_gate(runs, written)
    reading = _PlainReading(plain, why, _flew(plain) if plain is not None and not why else {}, [])
    if not reading.readable:
        return reading
    return reading._replace(misses=_misses(plain, written, names, round_axes=axes, bystanders=False, excursion=False))


def _plain_line(plain: _PlainReading, line: Any) -> None:
    """``plain``: the routine played with nothing hooked in; the things should end where the written replay left them."""
    if plain.run is None:
        return
    if plain.why:
        line(None, "plain", plain.why)
        return
    if plain.misses:  # the sim is not deterministic: loose things land elsewhere run to run
        line(
            None,
            "plain",
            f"played with nothing watching, {plain.misses[0].words}: the outcome is not repeatable, or the routine "
            "does something else when it is replayed: say which",
        )
        return
    if plain.flew:
        line(
            False,
            "plain",
            f"played with nothing watching, free of every contact but not falling at g, steps: {plain.flew}: the "
            "routine carries it another way when it is not replayed",
        )
        return
    line(True, "plain", "played with nothing watching, the things ended where the replay left them")


def verdict(model: dict, runs: dict[str, dict], settle_mm: float = SETTLE_MM) -> dict[str, Any]:
    """The lines ``./ctl physics`` prints: PASS and FAIL as ``check`` says them, NOTE for what the report must explain."""
    lines: list[dict[str, Any]] = []

    def line(ok: bool | None, name: str, detail: str) -> None:
        lines.append({"ok": ok, "name": name, "detail": detail})

    axes = {k: t["round_axis"] for k, t in model.get("things", {}).items() if t.get("round_axis")}
    driven = (runs.get("written") or {}).get("thing_targets_moved", 0.0) > 1e-6
    found = model.get("ties", []) + (model.get("drives", []) if driven else [])
    found += [f"{k} is kinematic" for k in model.get("kinematic", [])]
    if driven:
        found.append("a target on a thing's own joint moved while the routine played: a motor")
    labels = model.get("robot_equalities", {})
    for k in (runs.get("written") or {}).get("robot_equalities_on", []):
        label = labels.get(str(k), "")
        if "suction" not in label:
            found.append(f"equality row {k} ({label or 'a weld'}) held a thing to the robot while the routine played")
    for name in (runs.get("written") or {}).get("latched", []):
        found.append(f"{name}'s grasp latch held it to a hand while the routine played")
    added = (runs.get("written") or {}).get("hooks_added", 0)
    if added:
        found.append(f"{added} substep hook(s) attached while the routine played (a dragger or a pin)")
    welds = model.get("welds", [])
    w = runs.get("written") or {}
    notes = []
    if w.get("suction_held") or any("suction" in labels.get(str(k), "") for k in w.get("robot_equalities_on", [])):
        notes.append(
            "a suction cup's seal held a thing while the routine played: right only if the clip's gripper is one"
        )
    if w.get("step_hooks_added", 0):
        notes.append(
            f"{w['step_hooks_added']} step hook(s) added while the routine played: say in the report what each does "
            "(one that moves a thing is an attachment)"
        )
    if welds:
        notes.append("; ".join(welds) + ": a welded thing cannot move, so say in the report why it is welded")
    if found:
        line(False, "contact", "; ".join(found + notes))
    elif notes:
        line(None, "contact", "; ".join(notes))
    else:
        springs = (
            f"; {len(model['drives'])} stiff joint(s) on things held at their targets (their elasticity)"
            if model.get("drives")
            else ""
        )
        line(
            True, "contact", "nothing ties a thing to the robot, no motor on a thing's joints, nothing welded" + springs
        )
    written = runs.get("written") or {}
    names = _moved(written)
    frozen = runs.get("frozen")
    _robot_line(model, line)
    _model_lines(model, written, names, line, settle_mm, frozen)
    if written:
        flew = {
            k: t.get("free_flight_not_falling", 0)
            for k, t in written["things"].items()
            if t.get("free_flight_not_falling")
        }
        if written.get("nonfinite"):
            k, at = min(written["nonfinite"].items(), key=lambda kv: kv[1])
            line(
                False,
                "finite",
                f"{k}'s pose stopped being a number at step {at}: the solver blew up (a pose or a speed written into "
                "a solver that integrates it, or a contact it cannot hold)",
            )
        else:
            blew = {
                mode: min(run["nonfinite"].items(), key=lambda kv: kv[1])
                for mode, run in runs.items()
                if mode != "written" and run.get("nonfinite")
            }
            if blew:
                line(
                    None,
                    "finite",
                    "the solver blew up in "
                    + "; ".join(f"{mode} ({k} at step {at})" for mode, (k, at) in blew.items())
                    + ": that replay's reading stops there",
                )
        short = cut_short(written)
        line(
            None if short else written["outcome"] == "finished",
            "written",
            f"not read: {short}"
            if short
            else f"finished; moved {', '.join(names) or 'nothing'}"
            if written["outcome"] == "finished"
            else f"stopped: {written['outcome']}",
        )
        unjudged = {k: t["flight_unread"] for k, t in written["things"].items() if t.get("flight_unread")}
        unjudged_note = "not judged for " + "; ".join(f"{k} ({why})" for k, why in unjudged.items()) if unjudged else ""
        if unjudged and not flew:
            line(None, "flight", unjudged_note)
        if flew:
            longest = {k: t.get("free_flight_longest", 0) for k, t in written["things"].items()}
            long = max(longest.values(), default=0) >= FLIGHT_STEPS
            line(
                False if long else None,
                "flight",
                f"free of every contact but not falling at g, steps: {flew}: "
                + ("something that is not a contact carries it" if long else "say in the report what holds it")
                + (f"; {unjudged_note}" if unjudged_note else ""),
            )
    if frozen and (why := _unread(frozen)):
        line(None, "frozen", why)
    elif frozen:
        # a rigid thing has nothing to settle into; a heap, a liquid or a draped cloth can still be relaxing at the
        # reset, so its drift is a note for the report, not a fault
        unread_things_ = set((runs.get("written") or {}).get("contacts_unread") or {})
        sunk = {
            k
            for k, t in frozen.get("things", {}).items()
            if "start" in t
            and 0.0 < t["start"][2] - t["end"][2] <= SINK_MM / 1000.0
            and float(np.hypot(t["end"][0] - t["start"][0], t["end"][1] - t["start"][1])) < 0.001
            and t.get("rise_max_mm", 0.0) < 0.5
        }
        chains: dict[str, int] = {}
        for k in model.get("elastic_parts", []):
            chains[k.split("#")[0]] = chains.get(k.split("#")[0], 0) + 1
        bendy = {k for k, n in chains.items() if n >= 3}  # a stem (a chain of bending joints) drifts with its base
        elastic = (
            {
                k
                for k in [*model.get("elastic_parts", []), *bendy]
                if frozen.get("things", {}).get(k, {}).get("moved_max_mm", 0.0) <= ELASTIC_MM
            }
            | sunk
            | {
                k
                for k, t in frozen.get("things", {}).items()
                if "#" in k and k.split("#")[0] in unread_things_ and t.get("moved_max_mm", 0.0) <= SINK_MM
            }
        )
        rigid = {k: t["moved_max_mm"] for k, t in frozen.get("things", {}).items() if k not in elastic}
        soft = {k: t["moved_max_mm"] for k, t in frozen.get("deformables", {}).items()}
        soft |= {k: t["moved_max_mm"] for k, t in frozen.get("things", {}).items() if k in elastic}
        drift = {k: v for k, v in soft.items() if v > settle_mm}
        # a tie or a kinematic flag holds a thing in every replay: the contact line fails it, and frozen only names it.
        # A latch, an equality row or a motor is seen only in the written run (what the contact line reads), so a thing
        # held that way is read like any other here, with that hold named as what may be moving it
        always = _held_things(model, {})[0]
        in_written, unknown_hold = _held_things(model, written)
        in_written -= always  # the holds only the written run saw: a latch, an equality row, a motor
        named_moves = {k: v for k, v in rigid.items() if k in always}  # a whole thing: a part of it may still swing
        free_moves = {k: v for k, v in rigid.items() if k not in named_moves}
        worst = max(free_moves, key=free_moves.get) if free_moves else ""
        moved = free_moves.get(worst, 0.0)
        held_worst = max(named_moves, key=named_moves.get) if named_moves else ""
        # a move up to PUSHED_MM (scaled with --settle-mm) is a wobble for the report
        pushed = PUSHED_MM * settle_mm / SETTLE_MM
        also = (
            f"; {held_worst} moved {named_moves[held_worst]:.1f} mm, held by what the contact line names (a tie or a "
            "kinematic flag)"
            if held_worst and named_moves[held_worst] > settle_mm
            else ""
        )
        k = max(drift, key=drift.get) if drift else ""
        settling = f"; and {k} drifted {drift[k]:.1f} mm: still settling" if k else ""  # a deformable is said too
        if moved > settle_mm:
            pushed_far = moved > pushed
            if "#" in worst:  # a lid, a rotor, a door: a part on a joint that swings or settles under its weight
                cause = (
                    "a part on a joint swung or settled, so the mechanism is not at rest at the start"
                    if pushed_far
                    else "a part on a joint swung or settled a few millimetres: say in the report why the mechanism "
                    "is not at rest"
                )
            elif pushed_far:
                cause = "it starts inside something, or something other than the robot pushes it"
            else:
                cause = "a few millimetres, but a thing at rest stays put: say in the report what moves it"
            if "#" in worst and worst.split("#")[0] in always:  # a part of a thing tied to the robot or kinematic
                cause += "; the thing it is part of is held (the contact line names it): say if that moves this part"
            elif worst.split("#")[0] in in_written:  # a latch, an equality row or a motor on it in the written run
                cause += (
                    "; the contact line names a hold on it (a latch, an equality row, a motor): say if that moves it"
                )
            elif unknown_hold:  # a hook the routine added names no thing: it may be what moves it
                cause += "; the contact line names a hook the routine added, which names no thing: say if it holds this"
            line(
                False if pushed_far else None,
                "frozen",
                f"{worst} moved {moved:.1f} mm with the robot frozen: {cause}{settling}{also}",
            )
        elif also:  # only held things moved: the contact line fails them, frozen says so
            line(None, "frozen", f"with the robot frozen nothing free moved more than {moved:.1f} mm{also}{settling}")
        elif drift:
            line(
                None,
                "frozen",
                f"with the robot frozen no rigid thing moved, but {k} drifted {drift[k]:.1f} mm: still settling",
            )
        else:
            line(
                True,
                "frozen",
                f"with the robot frozen nothing moved more than {max([0.0, *rigid.values(), *soft.values()]):.1f} mm",
            )
    opened = runs.get("open")
    if opened and opened.get("held_rows") == 0:
        line(None, "open", "not read: this hand has no fingers to hold open")
    elif opened and (why := _unread(opened)):
        line(None, "open", why)
    elif opened and (written.get("outcome") != "finished" or not names):
        line(
            None,
            "open",
            "not read: the written replay "
            + ("did not finish" if written.get("outcome") != "finished" else "moved nothing"),
        )
    elif opened:
        did = opened["outcome"] == "finished" and not _ended_elsewhere(opened, written, names, round_axes=axes)
        line(
            None if did else True,
            "open",
            "the routine finished with the fingers held open: say in the report what carries "
            f"{', '.join(names) or 'the thing'} (a support, a hook, two arms squeezing)"
            if did
            else f"with the fingers held open the task did not happen ({opened['outcome'] if opened['outcome'] != 'finished' else 'the things ended elsewhere'})",
        )
    per_arm = {k: tuple(v) for k, v in (model.get("arm_torque") or {}).items() if v}
    torque = model.get("real_arm_torque")

    def nm(limits) -> str:
        return "/".join(f"{v:g}" for v in sorted(set(limits), reverse=True))

    every_arm = bool(per_arm) and all((model.get("arm_torque") or {}).values())  # each arm handle's real torque known
    if len(set(per_arm.values())) > 1:
        caps = "; ".join(f"{k} {nm(v)} Nm" for k, v in per_arm.items())
        robot = f"every friction at most {REAL_MU:g} and each arm's torque at its own ({caps})"
    elif per_arm or torque:
        robot = (
            f"every friction at most {REAL_MU:g} and the arm's torque at {nm(next(iter(per_arm.values()), torque))} Nm"
        )
    else:
        robot = f"every friction at most {REAL_MU:g} (no torque limits known for this robot)"

    def held(run: dict, what: str) -> tuple[bool | None, str, list]:
        """Finished, and the things ended where the written run left them: a grip that slipped fails here. Also the
        misses (:func:`_misses`)."""
        if run.get("nonfinite"):
            k, at = min(run["nonfinite"].items(), key=lambda kv: kv[1])
            return None, f"not read: with {what} the solver blew up ({k} at step {at})", []
        if written.get("outcome") != "finished":
            return None, "not read: the written replay did not finish, so there is no end to hold this one to", []
        if why := _unread(run):  # cut by the page, or its own machinery broke
            return None, why, []
        if run["outcome"] != "finished":
            return False, f"with {what} the routine stopped: {run['outcome']}", []
        misses = _misses(run, written, names, round_axes=axes)
        if misses:
            return False, f"with {what} the routine finished, but {misses[0].words}", misses
        return True, f"finished with {what}, the things ending where the written run left them", []

    plain = _plain_reading(runs, written, names, axes)
    built = runs.get("built")
    lowered = (built or {}).get("lowered")
    if (
        built
        and lowered is not None
        and not any(lowered.values())
        and not written.get("deformables")
        and not written.get("contacts_unread")
    ):
        # (a deformable's own contact friction, or a co-sim island's, is not one built caps: with either, read below)
        # nothing was above the real robot, so built only played the written run again: a different end, or a stop,
        # there is the scene's own spread from run to run, not a grip that needs more than the real hand
        ok, detail, misses = held(built, robot)
        same = f"built changed nothing: no rigid shape's friction here is above {REAL_MU:g}" + (
            " and no arm joint the data sheet covers is above the real arm's"
            if every_arm
            else " (no torque limits known for every arm)"
        )
        if ok:
            line(True, "built", same)
        elif ok is None:
            line(None, "built", detail)
        else:  # the same model played again: whatever went otherwise is the scene's spread, not friction or torque
            lost = [m for m in misses if m.kind == "lost"]
            miss = (lost or misses)[0].words if misses else f"it stopped ({built['outcome']})"
            why = (
                "the grasp is that marginal, or the outcome differs from run to run: say which in the report"
                if lost
                else "the outcome differs from run to run, so say so in the report"
            )
            line(None, "built", f"{same}, so it only played the same model again, and {miss}: {why}")
    elif built:
        servo = " and the registered hand's own servo" if (lowered or {}).get("servo") else ""
        ok, detail, misses = held(built, robot + servo)
        seen = (
            _within_spread(misses, built, plain.run, plain.misses, written, axes)
            if ok is False and plain.readable
            else ""
        )
        spread = f"{seen} too" if seen else ""
        if spread:
            # plain, which changes nothing, ended the same things about as far off: built cannot tell a slip from that
            line(
                None,
                "built",
                f"{detail}; but played with nothing changed, {spread}: this scene's outcome is not repeatable, so "
                "built cannot tell a slip from that: say which in the report",
            )
        else:
            failed = [  # handled things plain, which changes nothing, did not carry either: a grasp failing at any friction
                m.thing
                for m in misses
                if plain.readable
                and m.kind in ("lost", "end")
                and _entry(plain.run, m.thing)  # plain recorded it
                and not _carried(plain.run, written, m.thing, sampled=True)
            ]
            if failed:
                w, p = (_entry(r, failed[0]) or {} for r in (written, plain.run))
                detail += (
                    f"; played with nothing changed, {failed[0]} ended {p.get('end_mm', 0.0):.0f} mm from its start "
                    f"where the written run took it {w.get('end_mm', 0.0):.0f}: the grasp is marginal at any friction, "
                    "so say so in the report"
                )
            line(ok, "built", detail)
    heavy = runs.get("heavy")
    if heavy:
        servo = " and the registered hand's own servo" if (heavy.get("lowered") or {}).get("servo") else ""
        soft = [k for k in names if k in written.get("deformables", {})]  # heavy scales only a body's mass
        what = f"the rigid handled things {HEAVY:g} times as heavy ({', '.join(soft)}: not)" if soft else ""
        ok, detail, _ = held(heavy, (what or f"the handled things {HEAVY:g} times as heavy") + servo)
        if ok and soft and set(soft) == set(names):
            line(
                None,
                "heavy",
                f"not read: what the routine moved is a deformable ({soft[0]}), which heavy makes no heavier",
            )
        else:
            line(True if ok else None, "heavy", detail if ok is not False else detail + " (the grasp's margin)")
    if written:
        grip = []
        for k in names:
            for i, (p50, p95) in enumerate(written["things"].get(k, {}).get("hand_force_p50_p95_N", [])):
                if p50:
                    grip.append(f"{k} {p50:g} N (p95 {p95:g}) by hand {i + 1}")
        mu = written.get("finger_mu")
        # named when it, or a part of it (``box#1``), moved past the 5 mm the flight test would have read it at
        unread_of = written.get("contacts_unread") or {}
        stirred = [k for k, t in written.get("things", {}).items() if t.get("moved_max_mm", 0.0) > FLIGHT_MM]
        unread = {k.split("#")[0]: unread_of[k.split("#")[0]] for k in stirred if k.split("#")[0] in unread_of}
        touched = {
            k: [i + 1 for i, n in enumerate(written["things"].get(k, {}).get("finger_touch_steps", [])) if n]
            for k in unread
        }
        hands = model.get("hands") or []
        crushed = []
        for k in names:
            for i, (p50, _p95) in enumerate(written["things"].get(k, {}).get("hand_force_p50_p95_N", [])):
                real = HAND_FORCE.get(hands[i]) if i < len(hands) and hands[i] else None
                # the log adds both fingers' contact forces (and the friction that bears the weight): two jaws' worth;
                # the median while touching, not the p95, which an impact spike sets as often as a squeeze does
                if real and p50 > HAND_FORCE_MARGIN * 2.0 * real[1]:
                    crushed.append(
                        f"{k} {p50:g} N (median) by hand {i + 1}'s two fingers, {p50 / (2.0 * real[1]):.1f}x the most "
                        f"{real[0]} squeezes ({real[1]:g} N a jaw)"
                    )
        if grip:
            detail = "the fingers pressed " + "; ".join(grip)
        elif unread:
            detail = "no finger force read"
        else:
            detail = "no finger force on a rigid thing (a deformable's is not read)"
        if unread:
            detail += "; " + "; ".join(
                f"{k} ({v}): "
                + (
                    f"touched by hand {', '.join(map(str, touched[k]))}"
                    if touched.get(k)
                    else "not touched by the fingers"
                )
                + ", its force not read (its contacts are not in MuJoCo's list)"
                for k, v in unread.items()
            )
        if mu:
            detail += f"; the solver used friction {mu[0]:g} to {mu[1]:g} at the finger contacts"
        if crushed:
            detail = "the fingers pressed " + "; ".join(crushed) + ": a squeeze no real hand gives; " + detail
        line(
            False if crushed else None if (unread or (mu and mu[1] > REAL_MU)) else True,
            "grip",
            detail + (" (above any real pad: the built line is the one to trust)" if mu and mu[1] > REAL_MU else ""),
        )
    _drop_line(model, runs, written, line, axes)
    _plain_line(plain, line)
    word = {True: "PASS", False: "FAIL", None: "NOTE"}
    return {
        "passed": all(c["ok"] is not False for c in lines),
        "checks": lines,
        "report": "\n".join(f"{word[c['ok']]}  {c['name']:<8} {c['detail']}" for c in lines),
    }
