# SPDX-License-Identifier: Apache-2.0
"""An MJCF or a USD stage into a builder, with what the importers drop restored.

:func:`add_mjcf_collapsed` and :func:`add_usd_stage` are the two doors. Both
apply :func:`~robosimstudio.core.builder.rekey_custom_attributes` to their
fixed-joint collapse, which is the fixup every scene with a collapsed arm
needs and the reason a caller should not reach for ``add_mjcf`` directly.

:func:`add_usd_stage` is the USD half
of the same job. An Isaac-exported stage arrives with a *geared* gripper whose
coupling ``add_usd`` does not import, and with a scene graph in which the same
mesh appears twice -- once as render geometry and once as a collider -- so both
of those need fixing before the model is steppable and photographable.

:func:`texture_box_geoms` is the same genre one importer over: ``add_mjcf``
binds a material's texture to mesh geoms and silently drops it from every
primitive, so an arena that *labels* anything with a textured box imports unable
to state what the label said.
"""

from __future__ import annotations

import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import newton

from robosimstudio.core.builder import rekey_custom_attributes

__all__ = [
    "add_mjcf_collapsed",
    "add_usd_stage",
    "all_joints_fixed",
    "articulate",
    "clear_texture_tints",
    "collapse_own_fixed_joints",
    "collapsed_body",
    "draw_material_subsets",
    "expose_declared_colliders",
    "is_textured",
    "is_transparent",
    "open_session_stage",
    "select_engine_variant",
    "split_articulations",
    "strip_free_bodies",
    "texture_box_geoms",
]


def open_session_stage(source: str | Path) -> Any:
    """Open a stage whose every edit lands on a throwaway session layer.

    Opened with an explicit anonymous session layer, and that is not a detail.
    ``Usd.Stage.Open(path)`` gives a stage with **no** session layer, so
    ``SetEditTarget(GetSessionLayer())`` falls back to the *root* layer -- and
    root layers are shared process-wide by Sdf's layer cache. Editing there
    mutates the checked-in asset for every stage opened afterwards (measured:
    a second call found 0 instanceable prims because the first had already
    cleared them, and the root layer came back ``dirty``). Nothing writes it to
    disk, but a scene built later in the same process would silently get
    different collision geometry.

    Raises:
        ValueError: If the stage cannot be opened, or USD will not take the
            session layer as the edit target.
    """
    from pxr import Sdf, Usd

    source = str(source)
    root = Sdf.Layer.FindOrOpen(source)
    if not root:
        raise ValueError(f"could not open USD stage {source!r}")
    session = Sdf.Layer.CreateAnonymous("robosimstudio_session")
    stage = Usd.Stage.Open(root, session, Usd.Stage.LoadAll)
    if not stage:
        raise ValueError(f"could not open USD stage {source!r}")
    stage.SetEditTarget(session)
    if stage.GetEditTarget().GetLayer() != session:
        raise ValueError(
            f"USD refused the anonymous session layer for {source!r}; refusing to edit the root layer, "
            "which is shared process-wide and would change every scene built after this one"
        )
    return stage


#: The physics-engine variant sets a SimReady asset carries on its default prim.
#: Each holds ``Enabled``/``Disabled``, and the asset ships with PhysX enabled.
ENGINE_VARIANT_SETS = ("PhysX", "Newton", "MuJoCo")


def select_engine_variant(stage: Any, engine: str = "Newton", *, verbose: bool = False) -> list[str]:
    """Enable the stage's own variant for ``engine`` and disable the other engines'.

    NVIDIA's SimReady assets carry one variant set per physics engine on the
    default prim, and ship with ``PhysX = Enabled``. The PhysX variant asks for
    an ``sdf`` collider approximation on every mesh; the Newton variant asks for
    the authored mesh (``none``). A stage that declares a variant for the engine
    it is being built for is saying which claims hold there, so this takes it --
    the same rule as an object's USD import: the stage says which flavour it is.

    A stage with no variant set named ``engine`` is not touched. Edits go to the
    stage's edit target, which :func:`open_session_stage` makes a throwaway layer.

    Returns:
        ``"<set>=<selection>"`` for each selection changed.
    """
    prim = stage.GetDefaultPrim()
    if not prim:
        return []
    sets = prim.GetVariantSets()
    if engine not in sets.GetNames():
        return []
    changed = []
    for name in sets.GetNames():
        if name not in ENGINE_VARIANT_SETS:
            continue
        want = "Enabled" if name == engine else "Disabled"
        variant_set = sets.GetVariantSet(name)
        if want not in variant_set.GetVariantNames() or variant_set.GetVariantSelection() == want:
            continue
        variant_set.SetVariantSelection(want)
        changed.append(f"{name}={want}")
        if verbose:
            print(f"[usd] variant {name} = {want}")
    return changed


def articulate(stage: Any, *, verbose: bool = False) -> list[str]:
    """Give a stage whose joints belong to no articulation one root, on its default prim.

    ``add_usd`` gathers joints into articulations by ``PhysicsArticulationRootAPI``;
    a joint under no root is parsed and then left out of every articulation, and
    ``finalize`` refuses the model ("joint(s) not belonging to any
    articulation"). PhysX builds such a stage as loose maximal-coordinate joints,
    so the export is valid for Isaac and unbuildable here. 854 of the 969
    multi-body SimReady assets are in this state: a bottle and its cap held by a
    fixed joint, a file cabinet's ten drawers and doors.

    One root on the default prim makes the whole stage one articulation, which is
    what an object asset is. A stage that already declares a root anywhere is
    left alone -- it has said where its articulations are -- and so is one whose
    joints are all disabled or excluded from articulation.

    Returns:
        The path the root was applied to, or nothing.
    """
    from pxr import Usd, UsdPhysics

    prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    if any(prim.HasAPI(UsdPhysics.ArticulationRootAPI) for prim in prims):
        return []
    jointed = False
    for prim in prims:
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if joint.GetJointEnabledAttr().Get() is False or joint.GetExcludeFromArticulationAttr().Get():
            continue
        jointed = True
        break
    root = stage.GetDefaultPrim()
    if not jointed or not root:
        return []
    UsdPhysics.ArticulationRootAPI.Apply(root)
    if verbose:
        print(f"[usd] applied ArticulationRootAPI to {root.GetPath()}")
    return [str(root.GetPath())]


def split_articulations(stage: Any, *, fix_one_tree: bool = False, verbose: bool = False) -> dict[str, list[str]]:
    """Make each articulation the stage declares a tree Newton can build.

    Newton's importer builds an articulation as one tree hanging from one root,
    and refuses anything else ("Multiple roots found", "contains a cycle"). PhysX
    builds both, so SimReady ships both:

    ``a loop``
        A pair of pliers or a four-bar linkage. The joint that closes each loop
        is excluded from articulation; Newton then adds it after the tree as a
        joint of no articulation, which MuJoCo enforces as an equality.
    ``several trees under one root``
        A storage box whose lid has no hinge joint, a tray and its separate cover.
        The one root is replaced by a root on each tree's root body, so each tree
        is its own articulation.

    With ``fix_one_tree`` and several trees, none jointed to the world, the
    heaviest is named, for the caller to import fixed and the rest free (a jar
    bolted down, its lid loose): :func:`add_usd_stage` does that in two imports,
    since ``add_usd(floating=...)`` holds for every tree at once. A weld authored
    on the stage would not do: Newton leaves a fixed joint to a lone body out of
    every articulation.

    Every edit is on the stage's edit target; open it with
    :func:`open_session_stage`.

    Returns:
        The paths edited, under ``"loop_joints_excluded"`` and
        ``"articulation_roots_split"``, and the bodies of the tree to fix under
        ``"fixed_tree"``.
    """
    from pxr import Sdf, Usd, UsdPhysics

    changed: dict[str, list[str]] = {"loop_joints_excluded": [], "articulation_roots_split": [], "fixed_tree": []}
    prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    bodies = [str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not bodies:
        return changed
    is_body = set(bodies)

    def owner(targets) -> str:
        # A joint names a body or anything under it (SimReady names the mesh); no body is the world.
        path = Sdf.Path(targets[0]) if targets else Sdf.Path.emptyPath
        while not path.isEmpty and path != Sdf.Path.absoluteRootPath:
            if str(path) in is_body:
                return str(path)
            path = path.GetParentPath()
        return ""

    joints = []
    for prim in prims:
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if joint.GetJointEnabledAttr().Get() is False or joint.GetExcludeFromArticulationAttr().Get():
            continue
        parent, child = owner(joint.GetBody0Rel().GetTargets()), owner(joint.GetBody1Rel().GetTargets())
        if parent or child:
            joints.append((prim, parent, child))

    group = {b: b for b in bodies}
    anchored: set[str] = set()

    def find(b: str) -> str:
        while group[b] != b:
            group[b] = group[group[b]]
            b = group[b]
        return b

    # A body that two joints both call their child is where a loop most likely
    # closes, so those joints go last and, when one does close a loop, it is the
    # one excluded. Two joints onto one child from otherwise separate trees close
    # none and both stay: the importer sorts joints as an undirected graph and
    # turns one of them around.
    incoming: dict[str, int] = {}
    for _, _, child in joints:
        incoming[child] = incoming.get(child, 0) + 1
    tree_edges = []
    joined: set[tuple[str, str]] = set()
    for prim, parent, child in sorted(joints, key=lambda j: incoming.get(j[2], 0) > 1):
        # A second joint between two bodies already joined is no loop: an MJCF-style
        # humanoid states a shoulder as three hinges on one pair, which the
        # importer merges into one joint of three DOFs.
        if (parent, child) in joined or (child, parent) in joined:
            continue
        joined.add((parent, child))
        ends = [find(b) for b in (parent, child) if b]
        world = not (parent and child)
        closes = (ends[0] in anchored) if world else (ends[0] == ends[1] or {ends[0], ends[1]} <= anchored)
        if closes and not prim.IsInstanceProxy():
            UsdPhysics.Joint(prim).CreateExcludeFromArticulationAttr(True)
            changed["loop_joints_excluded"].append(str(prim.GetPath()))
            continue
        tree_edges.append((parent, child))
        if world:
            anchored.add(ends[0])
        else:
            group[ends[0]] = ends[1]
            if ends[0] in anchored:
                anchored.add(ends[1])

    trees: dict[str, list[str]] = {}
    for b in bodies:
        trees.setdefault(find(b), []).append(b)
    children = {child for parent, child in tree_edges if parent}

    def tree_root(tree: str) -> str:
        members = trees[tree]
        jointed_to_world = [child or parent for parent, child in tree_edges if not (parent and child)]
        return next(
            (b for b in members if b in jointed_to_world), next((b for b in members if b not in children), members[0])
        )

    def apply_roots(over: list[str]) -> None:
        for tree in over:
            root = stage.GetPrimAtPath(tree_root(tree))
            if not root.HasAPI(UsdPhysics.ArticulationRootAPI):
                UsdPhysics.ArticulationRootAPI.Apply(root)
                changed["articulation_roots_split"].append(str(root.GetPath()))

    for prim in prims:
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        under = prim.GetPath()
        covered = sorted({find(b) for b in bodies if Sdf.Path(b).HasPrefix(under)}, key=bodies.index)
        if len(covered) > 1 and not prim.IsInstanceProxy():
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            apply_roots(covered)

    free = [tree for tree in sorted(trees, key=bodies.index) if tree not in anchored]
    if fix_one_tree and len(trees) > 1 and len(free) == len(trees):

        def mass(tree: str) -> float:
            return sum(
                float(UsdPhysics.MassAPI(stage.GetPrimAtPath(b)).GetMassAttr().Get() or 0.0) for b in trees[tree]
            )

        apply_roots(free)
        changed["fixed_tree"] = list(trees[max(free, key=lambda tree: (mass(tree), len(trees[tree])))])
    if verbose:
        for kind, paths in changed.items():
            for path in paths:
                print(f"[usd] {kind}: {path}")
    return changed


def all_joints_fixed(stage: Any) -> bool:
    """Whether the stage has joints and every enabled one is a fixed joint.

    Such a stage is one object in several parts -- a bottle and its cap, a
    socket and its faceplate -- and collapses to a single body.
    """
    from pxr import Usd, UsdPhysics

    kinds = {
        prim.GetTypeName()
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if prim.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False
    }
    return bool(kinds) and kinds == {"PhysicsFixedJoint"}


def expose_declared_colliders(source: Any, *, verbose: bool = False) -> tuple[Any, dict[str, list[str]]]:
    """Open a stage with the colliders it declares actually reachable by ``add_usd``.

    **An Isaac export can declare a collider in two ways ``add_usd`` cannot
    follow, and a stage that mixes them imports an asymmetric robot.** Both are
    about *where* the ``UsdPhysics.CollisionAPI`` is relative to the mesh:

    ``CollisionAPI on an Xform, mesh underneath``
        The importer's collider pass keys on the ``Mesh`` prim, so an ``Xform``
        that says "I collide" over a ``Mesh`` that does not say so contributes
        nothing. PhysX walks the subtree and honours it, which is why the export
        is valid for Isaac and silently lossy here. This function copies the
        schema down onto the mesh the Xform is speaking for.
    ``the prim is instanceable``
        Then it has no composed children at all -- the subtree lives in a
        prototype reachable only through an instance proxy -- so there is no mesh
        to key on and nothing to copy the schema to either. Cleared first.

    On the grocery stage's DROID gripper, **eight of the eleven** gripper
    collision prims hit both problems at once -- instanceable *and* declaring the
    schema on the Xform -- so they import as nothing. The other three declare it
    on their mesh and import fine, and they do not divide evenly:
    ``left_inner_finger`` gets two of them (its pad and its pad backing) while
    ``right_inner_finger`` gets one (its pad alone). So the hand imports
    asymmetric, with two colliding shapes on the left against one on the right.

    Worth stating plainly, since the asymmetry is easy to over-read: on this stage
    it turned out to be **harmless**. It is the object *masses* that made grasps
    slip, and recovering these eight colliders measured worse than leaving them
    out -- the recovered hulls over-approximate the Robotiq's thin bars. See the
    grocery suite's ``.stage`` for the measured comparison. This function is what
    reproduces it.

    Only prims already carrying a ``CollisionAPI`` are touched, so this adds no
    collider the stage did not ask for; a mesh named ``*_visual`` is left alone,
    being the render half of a pair whose other half is the collider. Every edit
    goes to a throwaway session layer, so the checked-in asset is never modified,
    and render-only instances (the table's ``visuals``) keep their instancing.

    Args:
        source: Path to the USD stage, or a stage :func:`open_session_stage`
            opened, to edit further.
        verbose: Print each prim changed.

    Returns:
        ``(stage, changed)`` -- the opened stage to hand to ``add_usd``, and what
        was done, keyed ``"deinstanced"`` and ``"collision_applied"``.

    Raises:
        ValueError: If the stage cannot be opened.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = source if isinstance(source, Usd.Stage) else open_session_stage(source)

    # Collected before anything is edited: clearing `instanceable` restructures
    # the composed scene graph, and mutating it mid-traversal skips prims.
    declared = [
        prim.GetPath()
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsInstanceProxy()
    ]

    deinstanced: list[str] = []
    for path in declared:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsInstanceable():
            prim.SetInstanceable(False)
            deinstanced.append(str(path))
            if verbose:
                print(f"[usd] de-instanced collider {path}")

    applied: list[str] = []
    for path in declared:
        prim = stage.GetPrimAtPath(path)
        if not prim or prim.IsA(UsdGeom.Mesh):
            continue  # already where the importer looks
        meshes = [d for d in Usd.PrimRange(prim) if d.IsA(UsdGeom.Mesh) and not d.GetName().endswith("_visual")]
        if any(d.HasAPI(UsdPhysics.CollisionAPI) for d in meshes):
            continue  # the stage already says it where it counts
        for mesh in meshes:
            UsdPhysics.CollisionAPI.Apply(mesh)
            applied.append(str(mesh.GetPath()))
            if verbose:
                print(f"[usd] applied CollisionAPI to {mesh.GetPath()}")
    return stage, {"deinstanced": deinstanced, "collision_applied": applied}


#: Module-local aliases, because ``add_usd_stage`` takes keywords of the same names.
_expose_declared_colliders = expose_declared_colliders
_articulate = articulate


def add_usd_stage(
    builder: newton.ModelBuilder,
    source: str | Path,
    *,
    collapse_fixed_joints: bool | str = False,
    expose_declared_colliders: bool = False,
    engine_variant: str | None = "Newton",
    articulate: bool = True,
    material_subsets: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """``add_usd`` with the custom-attribute re-key applied to its collapse.

    The USD sibling of :func:`add_mjcf_collapsed`. ``parse_usd`` *does* hand back
    the collapse tables (under ``"collapse_results"``), so unlike the MJCF path
    there is nothing to work around -- but the re-key still has to happen, and a
    caller should not have to know that.

    The stage is opened on a session layer and given the fixups below before the
    import; when none of them changes anything the path itself is imported, as
    before they existed.

    Args:
        builder: Builder to import into.
        source: Path to the USD stage.
        collapse_fixed_joints: Fold the stage's fixed joints. Worth doing on an
            Isaac export, whose robot carries a wrist flange and a gripper base
            welded on: keeping them hands the solver zero-DOF articulations to
            maintain for nothing. ``"if_all_fixed"`` folds them only when every
            joint is fixed, so a multi-part prop becomes one body and a jointed
            one keeps its parts.
        expose_declared_colliders: Make the stage's declared colliders reachable
            by the importer -- colliders inside instance prototypes, and
            ``CollisionAPI`` declared on an Xform rather than on the mesh under
            it, are both otherwise dropped silently. See
            :func:`expose_declared_colliders` for what that costs on a gripper.
        engine_variant: The physics-engine variant to select where the stage
            offers one (:func:`select_engine_variant`); ``None`` keeps the
            stage's own selection.
        articulate: Give joints that belong to no articulation one root
            (:func:`articulate`), and make each articulation a tree
            (:func:`split_articulations`); with ``floating=False`` and an asset
            in several separate trees, the heaviest is the one fixed.
        material_subsets: Draw a mesh collider bound to several materials in
            one visual shape per material (:func:`draw_material_subsets`).
        verbose: Forwarded to the import and the re-key.
        **kwargs: Forwarded to :meth:`newton.ModelBuilder.add_usd`.

    Returns:
        ``parse_usd``'s own result dict.

    Raises:
        ValueError: If the stage's jointed bodies form more than one tree under
            one articulation root, or close a loop, naming the stage.
    """
    stage = open_session_stage(source)
    changed: dict[str, list[str]] = {}
    if engine_variant:
        changed["variant"] = select_engine_variant(stage, engine_variant, verbose=verbose)
    if articulate:
        changed["articulation_root"] = _articulate(stage, verbose=verbose)
        changed.update(split_articulations(stage, fix_one_tree=kwargs.get("floating") is False, verbose=verbose))
    fixed_tree = changed.pop("fixed_tree", [])
    # Folded after the import, keeping every joint already in the builder:
    # ``add_usd(collapse_fixed_joints=True)`` collapses the WHOLE builder, so a
    # prop imported after a fixed-base cabinet folded the cabinet into the world
    # and renumbered every body built before it.
    if collapse_fixed_joints == "if_all_fixed":
        fold_own = all_joints_fixed(stage)
    else:
        fold_own = bool(collapse_fixed_joints)
    joints_before, shapes_before = builder.joint_count, builder.shape_count
    if expose_declared_colliders:
        _, exposed = _expose_declared_colliders(stage, verbose=verbose)
        changed.update(exposed)
        if verbose:
            counts = ", ".join(f"{len(v)} {k}" for k, v in exposed.items())
            print(f"[usd] exposed declared colliders: {counts}")
    target = stage if any(changed.values()) else str(source)
    try:
        if fixed_tree:
            results = _add_usd_one_tree_fixed(builder, stage, fixed_tree, verbose=verbose, **kwargs)
        else:
            results = builder.add_usd(target, collapse_fixed_joints=False, verbose=verbose, **kwargs)
    except ValueError as error:
        if "Multiple roots" in str(error):
            raise ValueError(
                f"{source}: the stage's jointed bodies form more than one tree under one articulation root -- "
                "some part is joined to nothing (an authoring gap in the asset, e.g. a lid with no hinge joint)"
            ) from error
        if "contains a cycle" in str(error):
            raise ValueError(
                f"{source}: the stage's joints close a loop (a pair of pliers, a four-bar linkage), and an "
                "articulation is a tree; Newton builds no loop-closing joint"
            ) from error
        raise
    results = dict(results or {})
    if fold_own:
        collapse = collapse_own_fixed_joints(builder, joints_before, verbose=verbose)
        results["path_body_map"] = {
            path: collapsed_body(collapse, int(body)) for path, body in (results.get("path_body_map") or {}).items()
        }
        # A folded joint is gone; every other is renumbered past the ones folded before it.
        remap = collapse.get("joint_remap") or {}
        results["path_joint_map"] = {
            path: int(remap[int(joint)])
            for path, joint in (results.get("path_joint_map") or {}).items()
            if int(joint) in remap
        }
        results["collapse_results"] = collapse
    if material_subsets:
        results["material_subset_shapes"] = draw_material_subsets(
            builder, stage, range(shapes_before, builder.shape_count)
        )
    return results


def _add_usd_one_tree_fixed(builder: newton.ModelBuilder, stage: Any, fixed: list[str], **kwargs: Any) -> dict:
    """Import ``stage`` with the bodies in ``fixed`` on a fixed base and every other body free.

    Two imports of the one stage, each ignoring what the other takes; geometry on
    no body comes with the first.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    bodies = [str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    unbodied = [
        str(p.GetPath())
        for p in prims
        if p.IsA(UsdGeom.Gprim) and not any(p.GetPath().HasPrefix(Sdf.Path(b)) for b in bodies)
    ]

    def under(paths: list[str]) -> list[str]:
        # With a ".*", so the importer reads it as a pattern rather than an Sdf path.
        return [re.escape(path) + "(/.*)?$" for path in paths]

    ignored = list(kwargs.pop("ignore_paths", None) or [])
    kwargs.pop("floating", None)
    results = dict(
        builder.add_usd(
            stage, floating=False, ignore_paths=ignored + under([b for b in bodies if b not in fixed]), **kwargs
        )
        or {}
    )
    rest = builder.add_usd(stage, floating=True, ignore_paths=ignored + under(fixed + unbodied), **kwargs) or {}
    for key, value in rest.items():
        if isinstance(value, dict) and isinstance(results.get(key), dict):
            results[key] = {**results[key], **value}
    return results


def draw_material_subsets(builder: newton.ModelBuilder, stage: Any, shapes: Iterable[int]) -> list[int]:
    """Draw each mesh bound to several materials in one visual shape per material.

    Newton splits a *visual* mesh into one shape per bound ``GeomSubset``, but a
    mesh that is also a collider is drawn as one shape with one material -- the
    collider itself when it collides as the mesh, or the ``_visual`` copy kept
    beside an approximated one -- so a kettle with a metal body and a wooden
    handle in one mesh renders all metal. SimReady makes every mesh its own
    collider. Here each such drawn shape is replaced, on its body, by the visual
    path's per-subset meshes; hidden, it still collides if it did.

    Args:
        builder: The builder, after the import.
        stage: The stage the import read.
        shapes: The shapes the import added.

    Returns:
        The shapes added.
    """
    from newton._src.usd._visuals import _UsdVisuals
    from pxr import UsdGeom

    from robosimstudio.scene.build.bodies import preserve_body_inertia

    visuals = _UsdVisuals(stage)
    visible = int(newton.ShapeFlags.VISIBLE)
    cfg = builder.default_shape_cfg.copy()
    cfg.density = 0.0
    cfg.has_shape_collision = False
    cfg.has_particle_collision = False
    added: list[int] = []
    for shape in list(shapes):
        if builder.shape_type[shape] != newton.GeoType.MESH or not builder.shape_flags[shape] & visible:
            continue
        label = str(builder.shape_label[shape])
        path = label.removesuffix("_visual")
        prim = stage.GetPrimAtPath(path) if path.startswith("/") else None
        submeshes = visuals.get_visual_material_subset_meshes(prim) if prim and prim.IsA(UsdGeom.Mesh) else []
        if len(submeshes) < 2:
            continue
        body = int(builder.shape_body[shape])
        with preserve_body_inertia(builder, body):
            for subset, mesh in submeshes:
                added.append(
                    builder.add_shape_mesh(
                        body,
                        xform=builder.shape_transform[shape],
                        scale=builder.shape_scale[shape],
                        mesh=mesh,
                        cfg=cfg,
                        color=None,
                        label=subset,
                    )
                )
        builder.shape_flags[shape] = int(builder.shape_flags[shape]) & ~visible
    return added


def collapse_own_fixed_joints(builder: newton.ModelBuilder, joints_before: int, *, verbose: bool = False) -> dict:
    """Fold the fixed joints added since ``joints_before``, and re-key the custom attributes.

    ``builder.collapse_fixed_joints()`` with no ``joints_to_keep`` folds every
    fixed joint in the builder, including the world weld of anything imported
    earlier: its body is merged into the world and every later body renumbered,
    while the entity that built it keeps the old index. Each importer's own
    ``collapse_fixed_joints=True`` is that whole-builder call.

    Args:
        builder: The builder, right after an import.
        joints_before: ``builder.joint_count`` before that import.
        verbose: Forwarded to the collapse and the re-key.

    Returns:
        The collapse's own result dict.
    """
    results = builder.collapse_fixed_joints(verbose=verbose, joints_to_keep=list(range(joints_before)))
    rekey_custom_attributes(builder, results, verbose=verbose)
    return results


def collapsed_body(results: dict, body: int) -> int:
    """Where ``body`` went in a collapse: its own new index, or the one it merged into."""
    merged = results.get("body_merged_parent") or {}
    remap = results.get("body_remap") or {}
    while body in merged:
        body = int(merged[body])
    return int(remap.get(body, body))


def clear_texture_tints(builder: newton.ModelBuilder, shapes=None) -> int:
    """Set every textured shape's colour to white, so its texture renders as authored.

    Newton's tiled ray tracer multiplies a sampled texel by the shape's own
    ``shape_color``; the visual-matching backends let the texture win outright.
    That difference only shows when the two disagree about what the colour is,
    and for a textured shape they nearly always do -- an importer that binds a
    texture usually states no separate colour, so the shape falls to the
    build-order palette and the tiled render comes out tinted by an arbitrary
    hue while pyrender shows the texture clean.

    The grocery tote is the measured case: a grey plastic albedo (mean RGB
    91, 95, 90) times a palette red (0.96, 0.37, 0.37) renders a dark red bin
    in a scene whose provenance calls it grey.

    White is the identity of that multiply, so this makes the two backends agree
    on the one class of shape where the texture, not the colour, is the claim.

    Args:
        builder: The builder, after every import.
        shapes: Shape indices to consider; ``None`` is every shape.

    Returns:
        How many shapes were whitened.
    """
    cleared = 0
    for s in range(builder.shape_count) if shapes is None else shapes:
        if not is_textured(builder, s):
            continue
        if tuple(float(v) for v in builder.shape_color[s]) == (1.0, 1.0, 1.0):
            continue
        builder.shape_color[s] = (1.0, 1.0, 1.0)
        cleared += 1
    return cleared


def is_textured(builder: newton.ModelBuilder, shape: int) -> bool:
    """Whether this shape carries a texture, and so states no colour of its own.

    Split out of :func:`clear_texture_tints` because two passes need the same
    question and a second copy of it is a second thing to get wrong. The other
    caller is the per-entity display pass, which must not paint a role default
    onto a textured shape for the reason :func:`clear_texture_tints` exists:
    under the tiled tracer a colour on a texture is a multiply, not a choice.
    """
    src = builder.shape_source[shape]
    return src is not None and getattr(src, "texture", None) is not None


def is_transparent(builder: newton.ModelBuilder, shape: int) -> bool:
    """Whether the asset this shape came from asked for it not to be drawn.

    An MJCF ``rgba`` whose alpha is 0 means *not rendered* to MuJoCo, and Newton's
    importer keeps that alpha as ``shape_opacity``. The studio's materials and the
    path tracer's finishes take a shape's colour and not its alpha, so a shape its
    own file hid drew as a solid block. The YAM's two wrist-camera meshes are the
    case: ``rgba="0 0 0 0"``, drawn as a black box beside each wrist.

    Only a stated 0 counts; a fraction is a translucency, which is drawing.
    """
    opacity = getattr(builder, "shape_opacity", None)
    return opacity is not None and float(opacity[shape]) <= 0.0


def texture_box_geoms(builder: newton.ModelBuilder, source: str, *, verbose: bool = False) -> int:
    """Draw the textures Newton's MJCF importer binds to meshes and to nothing else.

    **The gap this fills.** ``import_mjcf`` resolves a geom's ``material`` to a
    texture file and threads it through as ``override_texture`` -- but only on
    its ``geom_type == "mesh"`` branch. Every primitive branch calls
    ``add_shape_box`` (and its siblings) with the material's *colour* and drops
    the image on the floor, silently: no warning, because from the importer's
    side nothing failed. The shape arrives with ``shape_source is None``, and
    with no source there is nothing for any backend to sample, so it falls to
    the build-order palette and renders as a flat coloured slab.

    The measured case is the tool-sort workcell, whose bin2 compartments are
    labelled by four textured box geoms 2 mm thick -- ``BATTERY``,
    ``MULTIMETER``, ``FLASHLIGHT``, ``SAFETY GLASSES``. In MuJoCo they are
    readable plates; imported, they are a cyan, a green, a yellow and a red
    rectangle. That is not a cosmetic loss on a task whose whole instruction is
    *"into the compartment whose label names each one"*: the scene stops stating
    the thing the policy is asked to read, and a VLM shown the render is being
    asked a question the picture no longer contains.

    What this adds is **visual only** -- a textured mesh per affected geom, with
    ``has_shape_collision=False`` -- and the original box is hidden rather than
    removed, so shape indices, contact behaviour and every handle that already
    points at one are untouched.

    **Cylinders too, for the same reason boxes needed it.** The tool-sort
    workbench's own four legs are ``type="cylinder"`` wearing a brushed-steel
    material, dropped exactly as the compartment plates were -- a flat grey
    pole where MuJoCo shows chrome. See :func:`_textured_cylinder_mesh`.

    **Placement is read back from the builder, not recomputed.** Only the
    ``geom name -> texture file`` mapping comes from the XML; the body, the
    transform and the half-extents come from the shape the importer already
    placed. So a scaled import, an ``xform`` argument and a fixed-joint collapse
    all land correctly by construction rather than by re-deriving them here and
    hoping the two agree.

    **UVs follow MuJoCo's own box convention**, which was read off a render
    rather than assumed: on the ``+z`` face, viewed down the axis, image-right
    is local ``+x`` and image-down is local ``-y``, with the full ``[0, 1]^2``
    across each face (``texuniform="false"``). Each of the six faces gets the
    whole image, which is what MuJoCo draws.

    Args:
        builder: The builder, immediately after the MJCF import.
        source: The same MJCF path.
        verbose: Print one line per geom re-textured.

    Returns:
        How many textured meshes were added.
    """
    import os
    import xml.etree.ElementTree as ET

    import numpy as np
    import warp as wp

    root = ET.parse(source).getroot()
    base = os.path.dirname(os.path.abspath(source))
    compiler = root.find("compiler")
    texdir = (
        (compiler.attrib.get("texturedir") or compiler.attrib.get("assetdir") or ".") if compiler is not None else "."
    )

    face_attrs = ("fileright", "fileleft", "filetop", "filebottom", "filefront", "fileback")
    textures: dict[str, str] = {}
    for tex in root.iter("texture"):
        name, file = tex.attrib.get("name"), tex.attrib.get("file")
        # A builtin (the skybox) has no file. A cube map is normally six
        # images this single-image path has no business guessing at -- but
        # MuJoCo also lets ``type="cube"`` name just one ``file`` and paints
        # it on all six faces (no per-face attribute), which is exactly a 2D
        # texture for any purpose that doesn't care which face is which, and
        # is how this file's own brushed-steel table legs are declared.
        if not (name and file):
            continue
        kind = tex.attrib.get("type", "2d")
        if kind == "2d" or (kind == "cube" and not any(a in tex.attrib for a in face_attrs)):
            textures[name] = os.path.normpath(os.path.join(base, texdir, file))

    materials: dict[str, tuple[str, tuple[float, float], bool]] = {}
    for m in root.iter("material"):
        name, tex = m.attrib.get("name"), m.attrib.get("texture")
        if not name or tex not in textures:
            continue
        # texuniform="true" asks for repeats per unit *length* -- a tiling in
        # world space rather than across the face -- so it is carried through
        # as a flag and the per-face extents scale the repeats at mesh build.
        # It used to be skipped outright, which left the one geom this whole
        # function was measured on a sibling of -- the workbench top, steel
        # plate at texrepeat "2 4" texuniform -- as a flat slate slab.
        uniform = m.attrib.get("texuniform", "false").lower() in ("true", "1")
        repeat = [float(v) for v in m.attrib.get("texrepeat", "1 1").split()]
        materials[name] = (textures[tex], (repeat[0], repeat[-1]), uniform)

    wanted = {
        g.attrib["name"]: materials[g.attrib["material"]]
        for g in root.iter("geom")
        if g.attrib.get("name")
        and g.attrib.get("type") in ("box", "cylinder")
        and g.attrib.get("material") in materials
    }
    if not wanted:
        return 0

    visible = int(newton.ShapeFlags.VISIBLE)
    box_type, cylinder_type = int(newton.GeoType.BOX), int(newton.GeoType.CYLINDER)
    added = 0
    # Snapshotted: add_shape_mesh appends, and iterating a growing range would
    # walk into the meshes this loop is producing.
    for s in range(builder.shape_count):
        shape_type = int(builder.shape_type[s])
        if shape_type not in (box_type, cylinder_type):
            continue
        # The importer's labels are slash-joined paths; the geom name is the leaf.
        found = wanted.get(str(builder.shape_label[s]).rpartition("/")[2])
        # A source means the importer already bound something -- leave it alone.
        if found is None or builder.shape_source[s] is not None:
            continue
        if not (int(builder.shape_flags[s]) & visible):
            continue
        texture, repeat, uniform = found
        if not os.path.exists(texture):
            raise ValueError(f"{source}: geom {builder.shape_label[s]!r} names a texture that is not there: {texture}")

        if shape_type == box_type:
            hx, hy, hz = (float(v) for v in builder.shape_scale[s])
            mesh = _textured_box_mesh(np, hx, hy, hz, texture, repeat, uniform=uniform)
        else:
            radius, half_height, _ = (float(v) for v in builder.shape_scale[s])
            mesh = _textured_cylinder_mesh(np, radius, half_height, texture, repeat, uniform=uniform)
        cfg = builder.default_shape_cfg.copy()
        cfg.has_shape_collision = False
        cfg.has_particle_collision = False
        cfg.density = 0.0
        xf = builder.shape_transform[s]
        builder.add_shape_mesh(
            body=int(builder.shape_body[s]),
            xform=wp.transform(wp.vec3(*(float(v) for v in xf[:3])), wp.quat(*(float(v) for v in xf[3:7]))),
            mesh=mesh,
            cfg=cfg,
            # White, so the texture is what shows -- the same identity
            # clear_texture_tints restores for every other textured shape.
            color=(1.0, 1.0, 1.0),
            label=f"{builder.shape_label[s]}_textured",
        )
        # Hidden, not deleted: the box keeps its index, and anything already
        # holding one keeps pointing at the same shape.
        builder.shape_flags[s] = int(builder.shape_flags[s]) & ~visible
        added += 1
        if verbose:
            print(f"re-textured {builder.shape_label[s]} from {os.path.basename(texture)}")
    return added


def _textured_box_mesh(
    np, hx: float, hy: float, hz: float, texture: str, repeat: tuple[float, float], *, uniform: bool = False
):
    """A box carrying the image on each of its six faces, ``texrepeat`` times.

    Split into 24 vertices rather than 8 because a UV is per-vertex: a corner
    shared by three faces would have to hold three different texture
    coordinates, and a cube with 8 vertices can state only one.

    Args:
        uniform: MuJoCo's ``texuniform``: ``repeat`` counts repeats per metre
            rather than per face, so each face multiplies it by its own extent
            along that face's u and v.
    """
    # (axis, sign, u axis, v axis) -- v runs against the image's own row order,
    # which is why each v axis is negated when the frame is built below.
    faces = (
        (2, +1, (1, 0, 0), (0, -1, 0)),  # +z: image-right is +x, image-down is -y
        (2, -1, (-1, 0, 0), (0, -1, 0)),
        (1, +1, (1, 0, 0), (0, 0, -1)),
        (1, -1, (-1, 0, 0), (0, 0, -1)),
        (0, +1, (0, -1, 0), (0, 0, -1)),
        (0, -1, (0, 1, 0), (0, 0, -1)),
    )
    half = np.array([hx, hy, hz], dtype=np.float32)
    vertices, normals, uvs, indices = [], [], [], []
    for axis, sign, u_axis, v_axis in faces:
        normal = np.zeros(3, dtype=np.float32)
        normal[axis] = sign
        u = np.array(u_axis, dtype=np.float32)
        v = np.array(v_axis, dtype=np.float32)
        centre = normal * half
        base = len(vertices)
        face_repeat = repeat
        if uniform:
            # Repeats per metre: scale by this face's full extent along u and v.
            face_repeat = (
                repeat[0] * 2.0 * float(np.abs(u * half).sum()),
                repeat[1] * 2.0 * float(np.abs(v * half).sum()),
            )
        # Corners in image order: (u=0,v=0) top-left, then across, then down.
        for du, dv, tu, tv in ((-1, 1, 0.0, 0.0), (1, 1, 1.0, 0.0), (1, -1, 1.0, 1.0), (-1, -1, 0.0, 1.0)):
            vertices.append(centre + u * half * du + v * half * dv)
            normals.append(normal)
            uvs.append((tu * face_repeat[0], tv * face_repeat[1]))
        indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))
    return newton.Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        normals=np.asarray(normals, dtype=np.float32),
        uvs=np.asarray(uvs, dtype=np.float32),
        texture=texture,
        color=(1.0, 1.0, 1.0),
        compute_inertia=False,
    )


def _textured_cylinder_mesh(
    np, radius: float, half_height: float, texture: str, repeat: tuple[float, float], *, uniform: bool = False
):
    """A cylinder's side wall carrying one image, wrapped once around like a label.

    MuJoCo states no UV convention for a textured cylinder primitive the way
    it does for a box's six faces (``_textured_box_mesh``'s docstring), so this
    picks the one every renderer uses for a can or a pole: image left-to-right
    is the circumference (angle 0 to 2*pi), image top-to-bottom is the height
    (``-half_height`` to ``+half_height``). ``texuniform`` scales the repeat by
    the true circumference and height, the same convention the box mesh's
    per-face scaling uses, so a material stating ``texrepeat="1 1"`` with no
    ``texuniform`` wraps the image exactly once around and once along --
    which is what makes a brushed-steel image read as a chrome pole rather
    than as a smear or an illegible tile.

    The end caps are triangle fans sampled at the image's own centre rather
    than genuinely unwrapped: a table leg's caps are butted against the floor
    and the underside of the top, so no camera this suite ships ever sees one,
    and a flat sample is what keeps a stray view reading as "the same
    material" instead of a seam of the raw palette colour.
    """
    segments = 32
    height = 2.0 * half_height
    circumference = 2.0 * float(np.pi) * radius
    u_repeat = repeat[0] * circumference if uniform else repeat[0]
    v_repeat = repeat[1] * height if uniform else repeat[1]

    vertices: list = []
    normals: list = []
    uvs: list = []
    indices: list = []

    # Side wall: one ring of quads, wound so the outward normal faces out.
    for i in range(segments):
        a0 = 2.0 * np.pi * i / segments
        a1 = 2.0 * np.pi * (i + 1) / segments
        n0 = np.array([np.cos(a0), np.sin(a0), 0.0], dtype=np.float32)
        n1 = np.array([np.cos(a1), np.sin(a1), 0.0], dtype=np.float32)
        lo = np.array([0.0, 0.0, -half_height], dtype=np.float32)
        hi = np.array([0.0, 0.0, half_height], dtype=np.float32)
        base = len(vertices)
        vertices += [n0 * radius + lo, n1 * radius + lo, n1 * radius + hi, n0 * radius + hi]
        normals += [n0, n1, n1, n0]
        u0, u1 = u_repeat * i / segments, u_repeat * (i + 1) / segments
        uvs += [(u0, 0.0), (u1, 0.0), (u1, v_repeat), (u0, v_repeat)]
        indices += [base, base + 1, base + 2, base, base + 2, base + 3]

    # End caps: a centre vertex plus the rim, both sampled at the image centre.
    centre_uv = (0.5 * u_repeat, 0.5 * v_repeat)
    for sign in (-1.0, 1.0):
        z = sign * half_height
        normal = np.array([0.0, 0.0, sign], dtype=np.float32)
        centre = len(vertices)
        vertices.append(np.array([0.0, 0.0, z], dtype=np.float32))
        normals.append(normal)
        uvs.append(centre_uv)
        ring_start = len(vertices)
        for i in range(segments):
            a = 2.0 * np.pi * i / segments
            vertices.append(np.array([radius * np.cos(a), radius * np.sin(a), z], dtype=np.float32))
            normals.append(normal)
            uvs.append(centre_uv)
        for i in range(segments):
            a_index, b_index = ring_start + i, ring_start + (i + 1) % segments
            # Wound the opposite way on the two caps, so both face outward.
            tri = (centre, a_index, b_index) if sign > 0 else (centre, b_index, a_index)
            indices.extend(tri)

    return newton.Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        normals=np.asarray(normals, dtype=np.float32),
        uvs=np.asarray(uvs, dtype=np.float32),
        texture=texture,
        color=(1.0, 1.0, 1.0),
        compute_inertia=False,
    )


def add_mjcf_collapsed(
    builder: newton.ModelBuilder,
    source: str,
    *,
    verbose: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """``add_mjcf`` with the fixed-joint collapse run afterwards, and re-keyed.

    ``add_mjcf(collapse_fixed_joints=True)`` throws away the remap tables the
    collapse returns, so the re-key cannot be applied to its result. Importing
    uncollapsed and collapsing explicitly keeps them, and folds only this
    import's joints (:func:`collapse_own_fixed_joints`).

    Args:
        builder: Builder to import into.
        source: Path to the MJCF, or its XML text.
        verbose: Forwarded to both the import and the collapse.
        **kwargs: Forwarded to :meth:`newton.ModelBuilder.add_mjcf`. Passing
            ``collapse_fixed_joints`` is an error -- that is what this function
            is for.

    Returns:
        The collapse results dict, so a caller can remap its own indices.
    """
    if "collapse_fixed_joints" in kwargs:
        raise ValueError("add_mjcf_collapsed collapses itself; drop the collapse_fixed_joints argument")
    joints_before = builder.joint_count
    builder.add_mjcf(source, verbose=verbose, collapse_fixed_joints=False, **kwargs)
    return collapse_own_fixed_joints(builder, joints_before, verbose=verbose)


_ASSET_DIRS = ("meshdir", "texturedir")


def _pin_source_paths(root: ET.Element, base: Path) -> None:
    """Make every path MuJoCo would resolve against ``base`` absolute.

    MuJoCo reads ``assetdir``, ``meshdir``, ``texturedir`` and the ``file`` of
    an ``<include>`` or ``<model>`` relative to the model file's directory, and
    a missing ``meshdir``/``texturedir`` falls back to ``assetdir`` and then to
    that directory. A copy written elsewhere keeps resolving to the same
    assets only if those paths are spelled out.
    """

    def absolute(value: str) -> str:
        return value if Path(value).is_absolute() else str(base / value)

    compilers = root.findall("compiler")
    if not compilers:
        compilers = [ET.Element("compiler")]
        root.insert(0, compilers[0])
    for compiler in compilers:
        if assetdir := compiler.get("assetdir"):
            compiler.set("assetdir", absolute(assetdir))
        for attr in _ASSET_DIRS:
            if value := compiler.get(attr):
                compiler.set(attr, absolute(value))
            elif not compiler.get("assetdir"):
                compiler.set(attr, str(base))
    for element in (*root.iter("include"), *root.iter("model")):
        if file := element.get("file"):
            element.set("file", absolute(file))


def strip_free_bodies(scene_xml: str | Path) -> str:
    """Write a copy of the MJCF with every free-jointed body and keyframe removed.

    The copy goes in the system temp directory, never beside the source, so a
    read-only install (a shared site-packages, a hermetic build) can still
    strip its scenes. Every path the source resolves against its own directory
    -- ``assetdir``, ``meshdir``, ``texturedir`` and ``<include>``/``<model>``
    files -- is written into the copy as an absolute path, so the copy loads
    the same assets. The caller owns the temp file.

    **Both spellings of a free joint count.** MJCF says the same thing two ways
    -- ``<freejoint/>`` and ``<joint type="free"/>`` -- and a check for one of
    them fails open on a file that uses the other: nothing raises, nothing is
    removed, and the caller gets back a copy of the scene it asked to have
    emptied. Every robosuite export in this repo uses the attribute form, so
    this read only the spelling they do not use.
    """
    src = Path(scene_xml)
    tree = ET.parse(src)
    root = tree.getroot()

    def free_jointed(body) -> bool:
        return body.find("freejoint") is not None or any(j.get("type") == "free" for j in body.findall("joint"))

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for body in list(worldbody.findall("body")):
            if free_jointed(body):
                worldbody.remove(body)
    # A keyframe indexes the qpos of the bodies just removed.
    for keyframe in root.findall("keyframe"):
        root.remove(keyframe)

    _pin_source_paths(root, src.resolve().parent)

    fd, path = tempfile.mkstemp(prefix="robosimstudio_stripped_", suffix=".xml")
    os.close(fd)
    tree.write(path, xml_declaration=False, encoding="unicode")
    return path
