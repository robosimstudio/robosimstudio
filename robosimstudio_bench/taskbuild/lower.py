# SPDX-License-Identifier: Apache-2.0
"""Turning a checked task document into the ``TaskSpec`` the runtime consumes.

The step that makes a document *runnable*. Everything before this is text and
structure; after it there is a frozen spec indistinguishable from a hand-written
one, which is the whole claim of the refactor -- the compiler, the runner and the
policy layer never learn that task documents exist.

**The evaluator is injected.** Building one needs atoms, handlers and feature
resolution, and those are the asset layer's, which sits *above* the toolkit. So
this module takes a maker rather than importing one, the same way the typechecker
takes a catalog. The payoff is the same too: lowering is a pure function of a
document, so its tests need no assets registered, no scene and no GPU -- and the
structural half of a task, which is where a mis-lowered region silently moves a
goal, is exhaustively testable in milliseconds.

**Artifacts are resolved, not embedded.**
``(:routine "cloth_fold.routine:FoldRoutine")`` becomes the class, through the
resolver ``load_solution`` has always used. A solution is a thing that runs, so it
stays code; the task is data and references it. That is also what keeps a
document from growing a scripting language.

What this module does *not* do is decide what a scene contains. Object poses,
mesh imports and the five-phase lifecycle stay with the compiler. This turns
declarations into a spec and stops.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.specs import (
    ActuationSpec,
    ArrangementSpec,
    AssetSpec,
    Binding,
    EpisodeSpec,
    ObjectSpec,
    RegionSpec,
    StaticSpec,
    TaskRequirements,
    TaskSpec,
)
from robosimstudio_bench.taskbuild.episodes import fixed
from robosimstudio_bench.taskdoc.ast import TaskDoc

from .arrangements import CAMERA_RIGS, PHYSICS
from .staging import Staged, Staging, parse_staging, stage

__all__ = [
    "LowerError",
    "Lowered",
    "export_keeps_free_bodies",
    "lower",
    "object_specs",
    "region_specs",
    "register_sampler",
]

SAMPLERS: dict[str, Callable[..., Any]] = {"fixed": fixed}
"""Episode samplers a document may name by a bare name.

Small on purpose: a name here is code someone registered with
:func:`register_sampler`, not something a document can invent, so a document
cannot restage itself in a way nothing checks. A suite's own sampler is named
the other way, as an artifact (``"staging:draw"``).
"""


def register_sampler(name: str, sampler: Callable[..., Any] | None = None):
    """Name an episode sampler, so a document can say ``(:episodes (:sampler NAME))``.

    A sampler is ``sampler(task, episode_id, rng) -> dict``: the episode's
    overrides, where an empty dict is the task's own layout (see
    :func:`~robosimstudio_bench.taskbuild.episodes.fixed`). Usable as a decorator.

    Raises:
        ValueError: The name already names a different sampler. The same function
            defined again -- a script run twice in one process -- replaces it.
    """

    def add(function: Callable[..., Any]) -> Callable[..., Any]:
        first = SAMPLERS.get(name)
        if first is not None and first is not function:
            same = getattr(first, "__qualname__", None) == getattr(function, "__qualname__", "") and getattr(
                getattr(first, "__code__", None), "co_filename", None
            ) == getattr(getattr(function, "__code__", None), "co_filename", "")
            if not same:
                raise ValueError(f"sampler {name!r} is already registered to {first!r}")
        SAMPLERS[name] = function
        return function

    return add if sampler is None else add(sampler)


class LowerError(Exception):
    """A document that typechecks but cannot be built.

    Distinct from a parse error and from a check problem, because it is a
    different moment and a different fix: the file is well formed and everything
    it names exists, but something it asks for has no implementation to point at
    -- a routine whose module moved, a sampler nobody registered.
    """


@dataclass(frozen=True)
class Lowered:
    """A built spec, and the record of where its staging came from."""

    spec: TaskSpec
    staged: Staged

    @property
    def inherited(self) -> tuple[str, ...]:
        """Fields a profile supplied rather than the document.

        Worth surfacing rather than burying: these are the fields whose value
        nobody writing this task actually chose, and when a port scores
        differently they are the first place to look.
        """
        return tuple(name for name, source in self.staged.filled if source != "document")


def region_specs(doc: TaskDoc) -> tuple[RegionSpec, ...]:
    """The document's ``:regions``, as the toolkit's own.

    A BDDL region is stated in its target's frame and carries BDDL's height
    window, both of which ``RegionSpec`` already models exactly -- so this is a
    rename, and deliberately nothing more. Every number is the file's own,
    because a compiler that derived region geometry could not detect the file and
    the derivation disagreeing.
    """
    return tuple(RegionSpec(name=r.name, frame=r.target, xy_ranges=r.ranges, z_range=r.z_range) for r in doc.regions)


def rebase_regions(
    regions: tuple[RegionSpec, ...],
    hosts: dict[str, tuple[str, tuple[float, float, float]]],
) -> tuple[RegionSpec, ...]:
    """Restate any region whose frame is not a body, in the frame of what holds it.

    A document states a region in its target's own frame, which is right and is
    what :func:`region_specs` preserves. But a target is not always something the
    scene can hang a frame off: a container welded into a bin is *geometry*, so it
    has a position and no handle, and a region stated against it would resolve
    against nothing.

    The fix is arithmetic and it belongs here, while the two facts it needs are
    content: which frame actually holds the thing, and where it sits in that
    frame. Both come from whatever recorded the weld -- an export sidecar, for the
    imported suites -- so the caller supplies them and this applies them.

    Args:
        regions: As lowered from the document.
        hosts: ``{region frame: (host frame, (dx, dy, dz))}``. A frame absent
            from this maps to a region passed through untouched, which is the
            common case -- most regions name a bin that is a body.

    Returns:
        The same regions, with a hosted one's offset folded into its ranges.

    Every number that is not the offset stays the file's own, including the
    height window: a syringe resting on a sharps-box lid 0.17 m above the cavity
    floor is outside the region exactly as the document intends, and a rebase
    that also normalised the window would have quietly changed what the task
    scores.
    """
    out: list[RegionSpec] = []
    for region in regions:
        if region.frame not in hosts:
            out.append(region)
            continue
        host, (dx, dy, dz) = hosts[region.frame]
        x0, y0, x1, y1 = region.xy_ranges
        z0, z1 = region.z_range
        out.append(
            RegionSpec(
                name=region.name,
                frame=host,
                xy_ranges=(dx + x0, dy + y0, dx + x1, dy + y1),
                z_range=(dz + z0, dz + z1),
            )
        )
    return tuple(out)


def object_specs(
    doc: TaskDoc, poses: dict[str, dict[str, Any]] | None = None, *, collider: str = ""
) -> tuple[ObjectSpec, ...]:
    """The document's ``:objects`` and ``:fixtures``, as object handles.

    Args:
        doc: The task.
        poses: Where each object starts, by name -- ``{"pose": (x, y, z),
            "quat": (x, y, z, w), "body_name": ..., "joint_name": ...}``. BDDL
            has no coordinates: an object's *placement* is the harness's, which
            for an imported problem means the export sidecar that recorded the
            arrangement it was dumped from.
        collider: The free objects' collider where an entry states none --
            ``(:collider drawn_hull)``. A fixture keeps the compiler's.

    An object with no pose entry is left at the spec default rather than being
    invented a position, which is the right failure: a scene whose pile is at the
    origin under the table is obvious, where a plausible-looking guess is not.

    A fixture lowers to a **static** handle. It is scenery that regions hang off
    -- a bin, a bench, a wall -- and giving it a free joint would let a goal
    region drift, which is the one thing a region must not do.
    """
    supplied = poses or {}
    out: list[ObjectSpec] = []
    for item in doc.objects:
        entry = supplied.get(item.name, {})
        out.append(
            ObjectSpec(
                name=item.name,
                category=item.type or item.name.rpartition("_")[0] or item.name,
                # Defaulting to the object's own name rather than to empty. The
                # entity layer already falls back that way, so the two are
                # equivalent at runtime -- but a spec that *states* its body name
                # can be read, and a fixture whose body is named something else
                # supplies it through `poses` like any other.
                body_name=str(entry.get("body_name", "") or item.name),
                joint_name=str(entry.get("joint_name", "")),
                pose=tuple(entry["pose"]) if "pose" in entry else (0.0, 0.0, 0.0),  # type: ignore[arg-type]
                quat=tuple(entry["quat"]) if "quat" in entry else (0.0, 0.0, 0.0, 1.0),  # type: ignore[arg-type]
                static=item.static,
                # A fixture is not grasped, so an OBB for it is cost for nothing;
                # a free object keeps the default, which is the compiler's.
                obb_grasp=not item.static,
                collider=str(entry.get("collider", "") or ("" if item.static else collider)),
            )
        )
        # A recorded entry may state any further field the handle has -- a spare
        # imported from a bare mesh names its ``asset``, ``mass``, ``friction``,
        # ``color``, ``world_xform`` and ``obb_grasp``; a goal object its grip --
        # the way a poses entry states any field of a built spec
        # (``place.build_entities``). The placement keys above are already read.
        out[-1] = _with_recorded_fields(out[-1], entry)
    return tuple(out)


_POSE_KEYS = frozenset({"pose", "quat", "body_name", "joint_name", "collider"})


def _with_recorded_fields(spec: ObjectSpec, entry: dict[str, Any]) -> ObjectSpec:
    """The handle with every further ``ObjectSpec`` field a recorded poses entry states.

    ``asset`` becomes an :class:`AssetSpec` through :func:`recorded_asset`;
    sequences become tuples, since a JSON file has only lists. A key the handle has no
    field for is left for the build step: a fixture's record may build a
    different spec altogether -- a welded tray's ``xform`` -- and
    ``place.build_entities`` applies those and refuses what neither has.
    """
    known = {f.name for f in fields(ObjectSpec)}
    changes: dict[str, Any] = {}
    for key, value in entry.items():
        if key in _POSE_KEYS or key not in known:
            continue
        if key == "asset":
            if isinstance(value, dict) and getattr(spec, "asset", None) is None:
                # Fields to merge onto an asset the handle does not have yet:
                # the build step owns it, the way a fixture's ``xform`` is owned
                # there. Left rather than guessed at a ``path`` nobody stated.
                continue
            changes[key] = recorded_asset(value, getattr(spec, "asset", None))
        elif isinstance(value, list):
            changes[key] = tuple(value)
        else:
            changes[key] = value
    return replace(spec, **changes) if changes else spec


def recorded_asset(value: Any, base: AssetSpec | None = None) -> AssetSpec:
    """What a recorded ``asset`` key means, in the one place both readers ask.

    A **string** is the whole answer: the bare mesh to import, which is how a
    spare states the geometry it draws. A **mapping** states :class:`AssetSpec`
    fields and is merged onto ``base`` where there is one -- so an entry that
    differs from its record's factory in a single field says that field and not
    the rest. The goal syringe is the case that asked for it: it collides on the
    same sixty-one hulls every spare does, and differs only in carrying the
    export's diffuse map, which no ``ObjectSpec`` field can express because a
    texture is a property of the geometry rather than of this placement of it.
    Restating the hull list per entry would put two thousand identical paths
    into the recorded layouts, where one changed copy would be a scene nobody
    could tell had drifted.
    """
    if isinstance(value, AssetSpec):
        return value
    if isinstance(value, dict):
        stated = {k: tuple(v) if isinstance(v, list) else v for k, v in value.items()}
        if base is None and "path" not in stated:
            raise ValueError(
                f"recorded asset states {sorted(stated)} with no geometry to merge onto; "
                "state a path, or record it against an asset whose record builds one"
            )
        return replace(base, **stated) if base is not None else AssetSpec(**stated)
    return AssetSpec(path=str(value))


def _requirements(staging: Staging) -> TaskRequirements:
    """``(:arms N)`` and ``(:needs-surface ...)`` to the task's requirements, defaults where unstated."""
    kwargs = {}
    if staging.arms:
        kwargs["arms"] = int(staging.arms)
    if staging.needs_surface is not None:
        kwargs["needs_surface"] = bool(staging.needs_surface)
    if staging.monolithic is not None:
        kwargs["monolithic"] = bool(staging.monolithic)
    return TaskRequirements(**kwargs)


def _actuation(pairs: tuple[tuple[str, Any], ...]) -> ActuationSpec | None:
    """``(:arm-actuation (:ke 1600) (:kd 70))`` as the gains, or ``None`` where unstated."""
    if not pairs:
        return None
    names = {"ke": "target_ke", "kd": "target_kd", "armature": "armature"}
    return ActuationSpec(**{names[key]: float(value) for key, value in pairs})


def _prose(value: Any) -> str:
    return " ".join(_prose(v) for v in value) if isinstance(value, list) else str(value)


def _under_export(path: str, staging: Staging) -> str:
    """A bare file name in a static override is a file of this variant's export."""
    return path if "/" in path or not staging.export else f"{staging.export}/{path}"


def _host_bindings(hosts: tuple[dict[str, Any], ...], staging: Staging) -> tuple[Binding, ...]:
    """``(:hosts (:host ...))`` as the pairings that stand a foreign robot in this scene.

    A row says how to stand the robot here -- which file of the export, which
    arm works -- and never that the pairing *works*: that is a fact about the
    matrix, discovered by running it. The scene defaults to the staging's.

    A row naming the variant's *own* robot is skipped: a suite-level host is
    how a foreign robot gets in, and a document whose staging already stands
    that robot natively (``tool_sort``'s level1 tier under its Robotiq profile,
    beside the suite's Robotiq host row) has nothing for the row to add. Kept,
    it would be a second binding on the native key, which
    :func:`~robosimstudio.specs.task._check_bindings` refuses for the whole
    suite at import.
    """
    out: list[Binding] = []
    native = (staging.scene, staging.embodiment, staging.gripper)
    for host in hosts:
        clauses = dict(host)
        notes = clauses.pop("notes", "")
        graphs = clauses.pop("graphs", None)
        overrides = clauses.pop("static-override", clauses.pop("static_override", None))
        own = parse_staging(clauses)
        if (own.scene or staging.scene, own.embodiment, own.gripper) == native:
            continue
        rows = [] if overrides is None else (overrides if isinstance(overrides[0], list) else [overrides])
        out.append(
            Binding(
                scene=own.scene or staging.scene,
                embodiment=own.embodiment,
                gripper=own.gripper,
                work_arm=own.work_arm,
                support_arm=own.support_arm,
                scene_resident=bool(own.scene_resident),
                cross_arm_collisions=own.arms_solid,
                static_overrides=tuple((str(name), _under_export(str(path), staging)) for name, path in rows),
                graphs=()
                if graphs is None
                else tuple(str(g) for g in (graphs if isinstance(graphs, list) else [graphs])),
                notes=_prose(notes),
            )
        )
    return tuple(out)


def export_keeps_free_bodies(staging: Staging) -> bool:
    """Whether the export's own free bodies *are* the task's objects.

    They are unless the workcell strips them: stated outright with
    ``(:strip-free-bodies true)``, or implied by ``(:objects-from ...)``, which
    re-imports every object from its record's own stage -- importing both
    would build the pile twice. A stripped export's objects are the records'
    to build; a kept export's are placed by the sidecar and want no asset.
    """
    if not staging.export:
        return False
    strip = bool(staging.objects_from) if staging.strip_free_bodies is None else staging.strip_free_bodies
    return not strip


def _export_statics(staging: Staging) -> tuple[StaticSpec, ...]:
    """``(:export X)`` as the workcell: one imported asset with its sidecar, first in the build order."""
    if not staging.export:
        return ()
    strip = not export_keeps_free_bodies(staging)
    entity = {"entity": staging.export_entity} if staging.export_entity else {}
    return (
        StaticSpec(
            name="workcell",
            asset=AssetSpec(path=f"{staging.export}/{staging.export_file or 'scene.xml'}"),
            meta="scene_meta.json",
            strip_free_bodies=strip,
            world_xform=True,  # a room is not placed relative to itself
            visible=True,
            **entity,
        ),
    )


def _native_binding(staging: Staging, graphs: tuple[tuple[str, str], ...]) -> tuple[Binding, ...]:
    """The task's own scene/robot pairing, as the one binding a document states.

    ``TaskSpec`` carries ``scene`` and ``embodiment`` directly, but *which arm
    does the work* lives on a ``Binding`` -- because it is a property of the
    pairing rather than of the task, and the same task on a one-armed robot has
    no such choice to make. Every hand-written suite therefore states its native
    pairing twice, once as fields and once as its first binding, and lowering
    does the same.

    Only the native row is produced. A document does not enumerate which foreign
    robots its scene also accepts: that is a claim about the *matrix*, discovered
    by running it, and a task file asserting it would be a task file that goes
    stale without anything noticing.
    """
    if not staging.scene and not staging.embodiment:
        return ()
    return (
        Binding(
            scene=staging.scene,
            embodiment=staging.embodiment,
            gripper=staging.gripper,
            work_arm=staging.work_arm,
            support_arm=staging.support_arm,
            scene_resident=bool(staging.scene_resident),
            cross_arm_collisions=staging.arms_solid,
            graphs=tuple(name for name, _ in graphs),
        ),
    )


def _episode_spec(staging: Staging, resolve: Callable[[str], Any] | None) -> EpisodeSpec:
    """``(:episodes (:count 1) (:seed 0) (:sampler fixed))`` as a spec.

    A sampler with a colon in its name is an artifact -- ``"layout:scatter"``,
    the suite's own -- and resolves the way a routine does.

    **A sampler that changes the model declares which keys do it.** An override
    that only re-poses a body is free: the env keeps the compiled model and
    writes the new pose. An override that draws a *board* -- where the fixtures
    stand, how far a rod is bowed at spawn -- changes what has to be built, and
    an env that does not know that re-poses into episode 0's model and runs a
    layout no document describes.

    Two things can know which of a sampler's keys are which, and both say it
    here. A document declares ``(:model-key-fields ...)``, one name or a list of
    them -- see :func:`~robosimstudio_bench.taskbuild.staging._episodes` -- which
    is how a variant speaks for a sampler it shares with variants that vary
    nothing. A sampler declares a ``model_key_fields`` attribute on itself, which
    is how a sampler speaks for every document that names it without each of them
    repeating the list. Both reach :attr:`EpisodeSpec.model_key_fields` and the
    two are **unioned**: a sampler that sets nothing keeps the old behaviour
    exactly, a document that declares nothing keeps its sampler's, and neither
    can quietly narrow the other back to the stale-model bug they both exist to
    prevent.
    """
    values = dict(staging.episodes)
    name = str(values.get("sampler", "fixed"))
    if ":" in name:
        sampler = _artifact(name, what="episode sampler", resolve=resolve)
    elif name in SAMPLERS:
        sampler = SAMPLERS[name]
    else:
        raise LowerError(f"episodes name sampler {name!r}; the registered ones are {sorted(SAMPLERS)}")
    # Written the way the sampler's own dict keys are: a document says
    # `model-key-fields`, a sampler returns `object_xforms`, and the two have to
    # be the same string for `model_key_for` to find it.
    declared = values.get("model_key_fields", ())
    named = (declared,) if isinstance(declared, str) else tuple(declared)
    fields = [str(field).replace("-", "_") for field in named]
    fields += [field for field in getattr(sampler, "model_key_fields", ()) if field not in fields]
    return EpisodeSpec(
        count=int(values.get("count", 1)),
        seed=int(values.get("seed", 0)),
        sampler=sampler,
        model_key_fields=tuple(fields),
    )


def _artifact(reference: Any, *, what: str, resolve: Callable[[str], Any] | None) -> Any:
    """One ``"module:Name"`` reference, resolved to the thing it names."""
    if reference is None:
        return None
    text = str(reference)
    if resolve is None:
        raise LowerError(f"the document references {what} {text!r}, but nothing was given to resolve artifacts with")
    try:
        return resolve(text)
    except Exception as exc:
        raise LowerError(
            f"{what} {text!r} does not resolve: {exc}. "
            "An artifact reference names code that must already exist; check the dotted path."
        ) from exc


def _routine(reference: Any, *, resolve: Callable[[str], Any] | None) -> Any:
    """The scripted routine a document names, refused where routines are not installed."""
    if reference is None:
        return None
    from robosimstudio_bench.solutions.declared import ROUTINES_ABSENT, routines_installed

    if not routines_installed():
        raise LowerError(f"the document names a scripted routine, {str(reference)!r}. {ROUTINES_ABSENT}")
    return _artifact(reference, what="routine", resolve=resolve)


def _resolve(name: str, supplied: dict | None, registered: dict, *, what: str, default):
    """One staged name, from the caller's table or the registry.

    Supplied wins, and that ordering is the whole design. A loader passes a table
    for the things a registration cannot express -- a camera rig aimed at a
    layout the document has no coordinates for -- and the registry answers for
    everything else, so a document naming a shipped arrangement needs no loader
    at all.

    An unstaged name is not an error: a task may legitimately want the default
    physics and no cameras.

    Raises:
        LowerError: When the document stages a name neither table has, listing
            what is registered -- because the two failures it could be are a
            loader that forgot to supply one and a corpus that was never
            imported, and the registered list distinguishes them at a glance.
    """
    if not name:
        return default
    table = supplied or {}
    if name in table:
        return table[name]
    if name in registered:
        return registered[name]
    raise LowerError(
        f"the document stages {what} {name!r}, which was neither supplied nor registered. "
        f"Registered: {(registered.names() if hasattr(registered, 'names') else sorted(registered)) or 'nothing -- no corpus arrangements have been imported'}"
    )


def lower(
    doc: TaskDoc,
    *,
    suite: str = "",
    name: str = "",
    make_evaluator: Callable[[TaskDoc], Any] | None = None,
    resolve_artifact: Callable[[str], Any] | None = None,
    poses: dict[str, dict[str, Any]] | None = None,
    profile: str = "",
    physics: dict[str, ArrangementSpec] | None = None,
    cameras: dict[str, tuple[CameraSpec, ...]] | None = None,
    statics: tuple[StaticSpec, ...] = (),
) -> Lowered:
    """Build a ``TaskSpec`` from a checked document.

    Args:
        doc: A document that has already typechecked. Lowering assumes it: a
            document whose goal names a missing asset should be refused by
            ``check`` with a catalog gap, not here with an attribute error.
        suite: Suite name. Empty derives nothing -- a task's suite is a curatorial
            decision, not a property of its file.
        name: Variant name. Empty uses the document's own, lowercased.
        make_evaluator: Builds the scorer. Injected because it needs the asset
            layer; ``None`` leaves ``evaluator`` unset, which is what the
            structural tests use.
        resolve_artifact: Resolves ``"module:Name"``. ``None`` refuses any
            document that references one, rather than silently dropping a
            routine and reporting the task as having no baseline.
        poses: Where objects start; see :func:`object_specs`.
        profile: An import profile to fill unstated staging.
        physics: Named arrangements, overriding the registry. A document names
            one; this is how a caller answers for a name nothing has registered,
            or overrides one that is. ``None`` leaves the registry to answer.
        cameras: Named rigs, likewise -- and the reason this parameter stays
            rather than being replaced by the registry: a rig *aimed* at a
            layout is a fact about the loader's own arithmetic, not something a
            versioned registration can hold.
        statics: Scenery handles, from whatever supplies the cell.

    Raises:
        LowerError: When something the document names has no implementation.
    """
    try:
        staged = stage(doc.staging, profile=profile)
    except KeyError as exc:
        # Surfaced as a build failure rather than a bare lookup error, because
        # that is what it is: the document is fine and names a convention this
        # process has not loaded. The fix is an import, and the message should
        # say which name went looking.
        raise LowerError(f"the document stages an import profile that is not registered: {exc}") from exc
    staging = staged.staging

    def named(reference: str, supplied: dict | None, registered: dict, *, what: str, default: Any) -> Any:
        # A reference with a colon is an artifact -- a rig aimed at this
        # variant's own layout, an arrangement only this suite stages -- and
        # resolves the way a routine does. A bare name is looked up: supplied
        # first (the structural tests' registry-free seam), registered second.
        if ":" in reference:
            return _artifact(reference, what=what, resolve=resolve_artifact)
        return _resolve(reference, supplied, registered, what=what, default=default)

    rig = named(staging.cameras, cameras, CAMERA_RIGS, what="camera rig", default=())
    # An arrangement is all data -- rates, a backend per domain, the coupler --
    # and the composer builds the solvers from it, so a name means exactly what
    # its registration says and nothing is handed in beside the document.
    arrangement = named(staging.physics, physics, PHYSICS, what="physics arrangement", default=ArrangementSpec())

    artifacts = doc.artifacts
    graphs = artifacts.get("graph")
    graph_rows: tuple[tuple[str, str], ...] = ()
    if graphs is not None:
        pairs = graphs if isinstance(graphs[0], list) else [graphs]
        graph_rows = tuple((str(p[0]), str(p[1])) for p in pairs)

    spec = TaskSpec(
        bindings=(*_native_binding(staging, graph_rows), *_host_bindings(doc.hosts, staging)),
        name=name or doc.name.lower(),
        suite=suite,
        instruction=doc.language,
        scene=staging.scene,
        embodiment=staging.embodiment,
        gripper=staging.gripper,
        # The one place a TaskSpec is constructed. Scenery and objects go into
        # one list; each spec's own stage is what orders them at build time. A
        # supplied static wins over the one the export derives, by name -- the
        # same rule a supplied rig has over a registered one.
        entities=(
            *(cell for cell in _export_statics(staging) if cell.name not in {s.name for s in statics}),
            *statics,
            *object_specs(doc, poses, collider=staging.collider),
        ),
        regions=region_specs(doc),
        cameras=rig,
        evaluator=make_evaluator(doc) if make_evaluator is not None else None,
        routine=_routine(artifacts.get("routine"), resolve=resolve_artifact),
        tools=_artifact(artifacts.get("tools"), what="tools", resolve=resolve_artifact),
        repro_hooks=_artifact(artifacts.get("repro-hooks"), what="repro hooks", resolve=resolve_artifact),
        render=_artifact(artifacts.get("render"), what="render look", resolve=resolve_artifact),
        graphs=graph_rows,
        physics=arrangement,
        episodes=_episode_spec(staging, resolve_artifact),
        control_mode=staging.control_mode or "joint_position",
        requires=_requirements(staging),
        max_steps=staging.max_steps or 600,
        arm_actuation=_actuation(staging.arm_actuation),
        gripper_actuation=_actuation(staging.gripper_actuation),
        terminate_on_success=True if staging.terminate_on_success is None else bool(staging.terminate_on_success),
        servo_speed_mps=staging.servo_speed_mps,
        # Carried as the name the document wrote. Resolving it to a spec here
        # would import the base while its own suite is still being loaded;
        # ``run.fleet.fleet_build_task`` resolves it when a wave is planned.
        fleet_base=staging.fleet_base,
    )
    return Lowered(spec=spec, staged=staged)
