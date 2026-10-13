# SPDX-License-Identifier: Apache-2.0
"""``:staging`` -- which cell, which robot, which physics, how many episodes.

Standard BDDL says what objects exist and what should become true of them. It
says nothing about which workcell they sit in, which arm reaches for them, at
what rate the physics runs, or how many episodes the task defines. LIBERO fills
those in by **harness convention**: the file is silent, and the runner knows.

That works until there are two runners. It also means a task cannot be read --
you can hold the whole problem file and still not know what it ran on, which
makes "why did this score differently" an archaeology problem. So this stack
records staging in the document, and this module is the typed shape of it.

**Two ways a field gets filled, and the difference is recorded.** A document may
state a field outright, or name an :class:`Profile` that supplies it -- which is
how an imported LIBERO or VAB problem gets a cell and a robot without anybody
inventing numbers for it. :attr:`Staged.filled` says which fields came from where,
one row per field, and that ledger is the honest version of what a convention
does silently. A task claiming a Panda in a robosuite cell because
``vab_industrial@v1`` says so is a different claim from one that named the Panda
itself, and a reader should be able to tell them apart.

**Profiles are content, not library.** A profile is a statement about how *some
benchmark* stages its tasks; the registry that holds them is machinery. So the
mechanism is here and the profiles live in the corpus, the same split the assets
and task documents already have.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, fields
from typing import Any

from robosimstudio.registry import Registry
from robosimstudio_bench.taskdoc.ast import TaskDoc
from robosimstudio_bench.taskdoc.bindings import BindingTable, apply_bindings

__all__ = [
    "PROFILES",
    "Profile",
    "Staged",
    "Staging",
    "bind_profile",
    "get_profile",
    "parse_staging",
    "register_profile",
    "stage",
]


@dataclass(frozen=True)
class Staging:
    """What a task runs on, as data.

    Every field maps onto a ``TaskSpec`` field of the same meaning; nothing is
    computed here. Empty means "unstated", which is what lets a profile fill it
    without having to distinguish a default from a decision -- ``max_steps=0`` is
    "nobody said", where ``max_steps=600`` is a choice somebody made.

    Attributes:
        scene: Registry name of the cell.
        export: A dumped scene this variant builds from, as a path under the
            asset root. Distinct from ``scene``: a scene is a registered cell
            this stack composes, an export is a file some other benchmark's
            environment wrote and this one imports whole. Naming it in the
            document rather than deriving it from the variant name is what lets
            two variants share one export -- ``hammer`` is a second goal over
            ``sort_all``'s scene, and deriving would have silently dumped a
            second copy.
        embodiment: Registry name of the robot.
        gripper: Registry name of the hand, when it is not the robot's own.
        work_arm: Which arm manipulates, on a bimanual robot.
        support_arm: Which arm holds, on a bimanual robot.
        scene_resident: Whether the robot is already inside the scene's asset.
            True for any cell imported whole from another benchmark, where the
            arm is welded into the same file as the workbench. Stating it is not
            optional: without it the compiler imports the standalone robot as
            well and the scene holds two arms, which surfaces as a duplicate
            joint label rather than as anything resembling its cause.
        physics: A named *arrangement*, never a solver configuration. A document
            that could set solver iterations would be a document that could
            silently change what the physics means; naming a calibrated
            arrangement keeps that decision where its calibration lives.
        cameras: A named rig, for the same reason.
        episodes: ``count`` / ``seed`` / ``sampler`` / ``model-key-fields``,
            by name.
        max_steps: Control steps before truncation. 0 is unstated.
        control_mode: What this variant's action vectors mean.
        objects_from: Where the manipulable objects' geometry and physics come
            from, when it is not the export's own bodies. Empty -- the default,
            and what every document meant before this existed -- takes them from
            the export, which is the reference in every sense: it is the file
            the poses were settled in.

            Naming something else states the *same problem* staged through a
            different importer: same objects, same names, same poses, same
            goals, so a difference measured afterwards is attributable to the
            import and to nothing else. That is why this is staging rather than
            a property of the objects. What the values mean is the suite's to
            say, since it is the suite that knows which files exist --
            ``tool_sort`` reads ``usd`` and ``usd_geom`` off its assets'
            declared files and refuses anything else.
    """

    scene: str = ""
    export: str = ""
    embodiment: str = ""
    gripper: str = ""
    work_arm: str = ""
    support_arm: str = ""
    scene_resident: bool | None = None
    """``None`` is unstated (read as false), so a document's explicit
    ``(:scene-resident false)`` survives :meth:`merged_over` a profile that says true."""
    physics: str = ""
    cameras: str = ""
    episodes: tuple[tuple[str, Any], ...] = ()
    max_steps: int = 0
    control_mode: str = ""
    objects_from: str = ""
    arms: int = 0
    """How many arms the task needs. 0 is unstated -- ``TaskRequirements``' default."""
    needs_surface: bool | None = None
    """Whether the task needs a raised work surface. ``None`` is unstated."""
    monolithic: bool | None = None
    """Scene and robot arrive in one import (``TaskRequirements.monolithic``)."""
    start_poses: str = ""
    """``(:start-poses poses/bench.json)``: a recorded ``{"objects": {name: {pose, quat, ...}}}``
    file, relative to the task directory. Unstated with an export takes the sidecar's poses."""
    collider: str = ""
    """``(:collider drawn_hull)``: the free objects' collider, where a pose entry states none."""
    export_file: str = ""
    """``(:export X (:file scene_norobot.xml))``: which file of the export is the workcell. ``scene.xml``."""
    export_entity: str = ""
    """``(:export X (:entity vab_workcell))``: the entity the workcell static is built by."""
    strip_free_bodies: bool | None = None
    """``(:export X (:strip-free-bodies true))``: drop the export's free bodies on import.
    Unstated strips exactly when ``objects_from`` re-imports them."""
    arms_solid: bool | None = None
    """``(:embodiment X (:arms-solid true))``: the native pairing's ``cross_arm_collisions``."""
    arm_actuation: tuple[tuple[str, float], ...] = ()
    """``(:arm-actuation (:ke 1600) (:kd 70))``: this task's arm servo gains."""
    gripper_actuation: tuple[tuple[str, float], ...] = ()
    """``(:gripper-actuation (:ke 20000) (:kd 20))``: likewise for the fingers."""
    terminate_on_success: bool | None = None
    """``(:terminate-on-success false)``: whether success ends the episode. Unstated is true."""
    servo_speed_mps: float | None = None
    """``(:servo-speed-mps 0.12)``: the Cartesian speed [m/s] the task's graphs servo the tool at.

    Unstated is the setpoint law's own cruise. A suite states it when its graphs
    were written and measured against a slower tool than the law's nominal
    cruise, so that speed is a fact recorded on the documents rather than one
    the host used to supply (``TaskSpec.servo_speed_mps``)."""
    fleet_base: str = ""
    """``(:fleet-base tool_sort/level1_superset)``: the document whose MODEL a fleet builds for this one.

    What an instance of a superset states. Its objects are a subset of the
    base's and a wave runs it as one world of the base's model, everything it
    does not name parked (``Episode.active``); the serial environment ignores
    it and builds the document's own model, which stays the reference. Read by
    ``robosimstudio_bench.run.fleet.fleet_model_key``."""

    def merged_over(self, base: Staging) -> Staging:
        """This staging, with anything it leaves unstated taken from ``base``.

        The document always wins. A profile supplies what the file did not say;
        it never overrides what it did -- otherwise importing a problem could
        change its meaning, which is the one thing an import must not do.
        """
        out: dict[str, Any] = {}
        for spec in fields(self):
            mine = getattr(self, spec.name)
            # Unstated is the field's default, not falsiness: a stated
            # ``False`` is a decision and must survive the merge.
            out[spec.name] = mine if mine != spec.default else getattr(base, spec.name)
        return Staging(**out)


@dataclass(frozen=True)
class Profile:
    """A named bundle filling what a dialect leaves to convention.

    Attributes:
        name: Versioned, always -- ``vab_industrial@v1``. A profile is a claim
            about how a body of tasks was staged, and revising that claim without
            a new version silently restages every task that named the old one.
        staging: The defaults it supplies.
        summary: What body of work it describes, for whoever reads the ledger.
        bindings: The dialect's predicate table, when the body of work was
            written in one -- how an imported ``In`` reads here. A profile is
            the one name a document states for "what this convention leaves
            unsaid", and its predicates are part of that convention.
    """

    name: str
    staging: Staging
    summary: str = ""
    bindings: BindingTable | None = None


PROFILES: Registry[Profile] = Registry("import profile", same=operator.eq)
"""Registered import profiles, by versioned name."""


def register_profile(profile: Profile) -> Profile:
    """Register a profile.

    Raises:
        ValueError: On an unversioned name, or a duplicate. Both refusals exist
            for the same reason: a profile that can be replaced in place is one
            whose tasks cannot be reproduced.
    """
    if "@" not in profile.name:
        raise ValueError(
            f"profile {profile.name!r} carries no version. Write {profile.name}@v1 -- "
            "a profile revised in place restages every task that named it, silently."
        )
    return PROFILES.register(profile.name, profile)


def get_profile(name: str) -> Profile:
    return PROFILES.get(name)


def bind_profile(doc: TaskDoc) -> TaskDoc:
    """A document read under its profile's predicate bindings, when it names a profile that carries any.

    A document that states its own predicates in this stack's vocabulary needs
    no dialect, and most do; one that names a profile whose body of work was
    written in an upstream dialect reads ``In`` the way that dialect means it,
    without restating anything. Silently unbound when the profile is not
    registered -- lowering reports that, with the profile's name.
    """
    name = str(dict(doc.staging).get("profile") or "")
    profile = PROFILES[name] if name in PROFILES else None
    if profile is None or profile.bindings is None:
        return doc
    return apply_bindings(doc, profile.bindings)


@dataclass(frozen=True)
class Staged:
    """Resolved staging, and the record of how each field got its value."""

    staging: Staging
    profile: str = ""
    filled: tuple[tuple[str, str], ...] = ()
    """``(field, source)`` for every field with a value. Source is ``"document"``
    or the profile's versioned name.

    Kept rather than discarded because the two are different claims. A task that
    names its own Panda has decided; one whose Panda arrived from
    ``vab_industrial@v1`` has inherited a convention, and if that convention was
    wrong the second is worth re-reading and the first is not."""

    def source_of(self, name: str) -> str:
        """Where one field came from, or empty if it is unstated."""
        return next((source for key, source in self.filled if key == name), "")


# ------------------------------------------------------------------ parsing
_ARM_KEYS = {
    "work-arm": "work_arm",
    "work_arm": "work_arm",
    "support-arm": "support_arm",
    "support_arm": "support_arm",
    "arms-solid": "arms_solid",
    "arms_solid": "arms_solid",
}
_NAMES = (
    "scene",
    "physics",
    "cameras",
    "gripper",
    "control_mode",
    "objects_from",
    "start_poses",
    "collider",
    "fleet_base",
)
_FLAGS = ("needs_surface", "monolithic", "terminate_on_success")
_GAINS = ("arm_actuation", "gripper_actuation")


def _flag(value: Any) -> bool:
    """``(:x)``, ``(:x true)`` and ``(:x yes)`` are true; anything else stated is false."""
    return value is True or str(value[0] if isinstance(value, list) else value).lower() in ("true", "yes")


def parse_staging(section: dict[str, Any]) -> Staging:
    """Read a parsed ``:staging`` section into a :class:`Staging`.

    The section arrives in the parser's ``_plain`` shape: ``(:embodiment
    yam_bimanual (:work-arm left))`` is ``{"embodiment": ["yam_bimanual",
    [":work-arm", "left"]]}``.

    Raises:
        ValueError: On a key this shape does not model. Refused rather than
            ignored -- a staging line silently dropped is a task that ran on
            something other than what it says.
    """
    out: dict[str, Any] = {}
    for raw, value in section.items():
        key = raw.replace("-", "_")
        if key == "embodiment":
            head, *rest = value if isinstance(value, list) else [value]
            out["embodiment"] = str(head)
            out.update(_arms(rest))
        elif key == "export":
            head, *rest = value if isinstance(value, list) else [value]
            out["export"] = str(head)
            out.update(_export(rest))
        elif key == "scene_resident":
            # A flag, so `(:scene-resident)` with no value means true -- which is
            # how a flag reads out loud, and `_plain` already yields True for a
            # valueless clause.
            out["scene_resident"] = _flag(value)
        elif key in _NAMES:
            out[key] = str(value[0] if isinstance(value, list) else value)
        elif key in ("max_steps", "arms"):
            out[key] = int(value[0] if isinstance(value, list) else value)
        elif key == "servo_speed_mps":
            out[key] = _speed(value[0] if isinstance(value, list) else value, what=raw)
        elif key in _FLAGS:
            out[key] = _flag(value)
        elif key in _GAINS:
            out[key] = _pairs(_clauses(value), what=raw, keys=("ke", "kd", "armature"))
        elif key == "episodes":
            out["episodes"] = _episodes(_clauses(value))
        else:
            raise ValueError(
                f":staging names {raw!r}, which is not one of "
                "scene, export, embodiment, gripper, physics, cameras, episodes, max-steps, control-mode, "
                "scene-resident, objects-from, arms, needs-surface, monolithic, start-poses, "
                "collider, arm-actuation, gripper-actuation, terminate-on-success, fleet-base, "
                "servo-speed-mps"
            )
    return Staging(**out)


def _speed(value: Any, *, what: str) -> float:
    """A positive speed [m/s], refused otherwise: a clause a servo cannot honour is an error, not a default."""
    try:
        speed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f":{what} takes a speed in m/s, got {value!r}") from None
    if not speed > 0.0:
        raise ValueError(f":{what} must be positive, got {value!r}")
    return speed


def _export(clauses: list[Any]) -> dict[str, Any]:
    """``(:file F) (:entity E) (:strip-free-bodies B)``, hanging off the export."""
    out: dict[str, Any] = {}
    for clause in clauses:
        if not isinstance(clause, list) or len(clause) != 2:
            raise ValueError(f":export takes (:file F), (:entity E) and (:strip-free-bodies B) clauses, got {clause!r}")
        key = str(clause[0]).lstrip(":").replace("-", "_")
        if key == "file":
            out["export_file"] = str(clause[1])
        elif key == "entity":
            out["export_entity"] = str(clause[1])
        elif key == "strip_free_bodies":
            out["strip_free_bodies"] = _flag(clause[1])
        else:
            raise ValueError(f":export names {key!r}; it takes file, entity and strip-free-bodies")
    return out


def _arms(clauses: list[Any]) -> dict[str, Any]:
    """``(:work-arm left)``, ``(:support-arm right)`` and ``(:arms-solid true)``, hanging off the embodiment."""
    out: dict[str, Any] = {}
    for clause in clauses:
        if not isinstance(clause, list) or len(clause) != 2:
            raise ValueError(f":embodiment takes (:work-arm SIDE) clauses, got {clause!r}")
        key = str(clause[0]).lstrip(":")
        if key not in _ARM_KEYS:
            raise ValueError(f":embodiment names {key!r}; it takes work-arm, support-arm and arms-solid")
        field = _ARM_KEYS[key]
        out[field] = _flag(clause[1]) if field == "arms_solid" else str(clause[1])
    return out


def _clauses(value: Any) -> list[Any]:
    """A sub-section's value as a list of clauses, however many it holds.

    ``_plain`` unwraps a single value, so ``(:episodes (:count 50))`` arrives as
    the one clause ``[":count", 50]`` while ``(:episodes (:count 50) (:seed 0))``
    arrives as a list of two. A list whose first element is not itself a list
    is that one clause, not a list of clauses.
    """
    if not isinstance(value, list):
        return [value]
    if value and not isinstance(value[0], list):
        return [value]
    return value


def _pairs(clauses: list[Any], *, what: str, keys: tuple[str, ...]) -> tuple[tuple[str, Any], ...]:
    """``(:count 1) (:seed 0)`` -- a clause of keyed pairs, keys checked against ``keys``."""
    out: list[tuple[str, Any]] = []
    for clause in clauses:
        if not isinstance(clause, list) or len(clause) != 2:
            raise ValueError(f":{what} takes {' '.join(f'(:{k} ..)' for k in keys)} clauses, got {clause!r}")
        key = str(clause[0]).lstrip(":").replace("-", "_")
        if key not in keys:
            raise ValueError(f":{what} names {key!r}; it takes {', '.join(keys)}")
        out.append((key, clause[1]))
    return tuple(out)


def _episodes(clauses: list[Any]) -> tuple[tuple[str, Any], ...]:
    """``(:count 1) (:seed 0) (:sampler fixed) (:model-key-fields layout)``, as pairs.

    Kept as names rather than resolved into a sampler callable, because
    resolution needs a registry and this module is deliberately registry-free --
    same reason the parser is. Lowering resolves them.

    ``model-key-fields`` names the sampled keys that change the *model* rather
    than only its state, and a task whose sampler moves anything built into the
    model has to say so: an environment re-poses between episodes and rebuilds
    only when one of these differs, so a sampler that moves a welded fixture or
    a soft body's lattice without declaring it varies nothing after episode
    zero -- silently, which is the shape of wrong this corpus keeps finding.
    One name or a list of them: ``(:model-key-fields layout)``,
    ``(:model-key-fields (layout instruction))``.
    """
    return _pairs(clauses, what="episodes", keys=("count", "seed", "sampler", "model_key_fields"))


def stage(section: dict[str, Any], *, profile: str = "") -> Staged:
    """Resolve a document's staging against an import profile.

    Args:
        section: The parsed ``:staging``.
        profile: A registered profile's versioned name, or empty for none. A
            document may also name one itself with ``(:profile NAME)``, which is
            the usual case for an imported problem; the argument wins, so a
            harness can restage an import without editing the file it imported.

    Returns:
        The merged staging and the per-field ledger.
    """
    stated_profile = str(section.get("profile", "") or "")
    chosen = profile or stated_profile
    document = parse_staging({k: v for k, v in section.items() if k != "profile"})

    if not chosen:
        return Staged(staging=document, filled=_ledger(document, Staging(), "document", ""))

    base = get_profile(chosen).staging
    return Staged(
        staging=document.merged_over(base),
        profile=chosen,
        filled=_ledger(document, base, "document", chosen),
    )


def _ledger(document: Staging, base: Staging, own: str, other: str) -> tuple[tuple[str, str], ...]:
    """One row per field that ended up with a value, saying which side gave it."""
    rows: list[tuple[str, str]] = []
    for spec in fields(document):
        if getattr(document, spec.name) != spec.default:
            rows.append((spec.name, own))
        elif other and getattr(base, spec.name) != spec.default:
            rows.append((spec.name, other))
    return tuple(rows)
