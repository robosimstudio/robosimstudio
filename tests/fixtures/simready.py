# SPDX-License-Identifier: Apache-2.0
"""Small USD stages laid out the way NVIDIA's SimReady assets are, written on demand.

The real assets are tens of megabytes each, so the tests carry this instead: the
same structure, a few boxes big. What is reproduced is what the import has to
handle, and nothing else:

* a default prim ``/RootNode``, Z-up and in metres, carrying one variant set per
  physics engine (``PhysX``, ``Newton``, ``MuJoCo``, each ``Enabled``/``Disabled``)
  with PhysX enabled, as the dataset ships;
* each part an **instanceable** Xform ``Geometry/<part>_obj_00`` carrying
  ``PhysicsRigidBodyAPI``, its collider mesh inside the instance prototype;
* joints under ``/RootNode/Joints`` named ``joint_<part>_joint_00``, and **no**
  ``PhysicsArticulationRootAPI`` anywhere -- which is what 854 of the 969
  multi-body assets lack.

Mass is authored inside the engine variants, so a test can tell which one the
import selected: the PhysX variant states :data:`PHYSX_MASS` for every part and
the Newton variant :data:`NEWTON_MASS`, each times the part's own ``mass``.

A part given ``materials`` splits its box's six faces between them, one bound
``GeomSubset`` each, as a SimReady kettle's one mesh holds its metal and wood.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["NEWTON_MASS", "PHYSX_MASS", "Joint", "Part", "write_stage"]

PHYSX_MASS = 1.0
"""Each part's mass under the shipped ``PhysX`` variant [kg]."""

NEWTON_MASS = 2.0
"""Each part's mass under the ``Newton`` variant [kg]."""


@dataclass(frozen=True)
class Part:
    """One rigid part: a box ``half`` in half-extents [m], centred at ``at`` in the asset frame."""

    name: str
    at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half: tuple[float, float, float] = (0.05, 0.05, 0.05)
    mass: float = 1.0
    materials: tuple[tuple[float, float, float], ...] = ()
    """Diffuse colours; with two or more, the faces are dealt out between them."""


@dataclass(frozen=True)
class Joint:
    """One joint from ``parent`` to ``child``, at ``anchor`` in the asset frame.

    ``kind`` is ``"fixed"``, ``"prismatic"`` or ``"revolute"``; ``limits`` is in
    metres for a slide and degrees for a hinge, as USD states them. An empty
    ``parent`` is the world. A ``drive`` stiffness above zero authors a position
    drive aimed at 0, as an asset's own drive is.
    """

    name: str
    kind: str
    parent: str
    child: str
    anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: str = "X"
    limits: tuple[float, float] = (0.0, 0.3)
    drive: float = 0.0


def _box(stage, path: str, half, materials=()) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    hx, hy, hz = half
    mesh = UsdGeom.Mesh.Define(stage, path)
    corners = [(sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    mesh.CreatePointsAttr(corners)
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    for index, colour in enumerate(materials if len(materials) > 1 else ()):
        material = UsdShade.Material.Define(stage, f"{path}_looks/m_{index}")
        shader = UsdShade.Shader.Define(stage, f"{path}_looks/m_{index}/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        faces_of = [f for f in range(len(faces)) if f % len(materials) == index]
        subset = UsdGeom.Subset.CreateGeomSubset(mesh, f"m_{index}", "face", faces_of, "materialBind")
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)


def write_stage(path: str | Path, parts: list[Part], joints: list[Joint] = ()) -> Path:
    """Write a SimReady-shaped stage of ``parts`` and ``joints`` to ``path``; return it."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    path = Path(path)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/RootNode").GetPrim()
    stage.SetDefaultPrim(root)

    # The instance prototypes: one collider mesh per part, outside the default prim.
    for part in parts:
        _box(stage, f"/Prototypes/{part.name}/{part.name}_mesh_00", part.half, part.materials)
        stage.GetPrimAtPath(f"/Prototypes/{part.name}").SetSpecifier(Sdf.SpecifierClass)

    UsdGeom.Scope.Define(stage, "/RootNode/Geometry")
    for part in parts:
        obj = UsdGeom.Xform.Define(stage, f"/RootNode/Geometry/{part.name}_obj_00")
        obj.AddTranslateOp().Set(Gf.Vec3d(*part.at))
        prim = obj.GetPrim()
        prim.GetReferences().AddInternalReference(f"/Prototypes/{part.name}")
        prim.SetInstanceable(True)
        UsdPhysics.RigidBodyAPI.Apply(prim)

    UsdGeom.Scope.Define(stage, "/RootNode/Joints")
    for index, joint in enumerate(joints):
        # Numbered like SimReady's own: a second joint onto one part is ``_joint_01``.
        repeat = sum(1 for other in joints[:index] if other.name == joint.name)
        joint_path = f"/RootNode/Joints/joint_{joint.name}_joint_{repeat:02d}"
        if joint.kind == "fixed":
            usd_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        elif joint.kind == "prismatic":
            usd_joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        elif joint.kind == "revolute":
            usd_joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        else:
            raise ValueError(f"unknown joint kind {joint.kind!r}")
        if joint.kind != "fixed":
            usd_joint.CreateAxisAttr(joint.axis)
            usd_joint.CreateLowerLimitAttr(joint.limits[0])
            usd_joint.CreateUpperLimitAttr(joint.limits[1])
        if joint.drive > 0.0:
            drive = UsdPhysics.DriveAPI.Apply(usd_joint.GetPrim(), "linear" if joint.kind == "prismatic" else "angular")
            drive.CreateStiffnessAttr(joint.drive)
            drive.CreateDampingAttr(0.1 * joint.drive)
            drive.CreateTargetPositionAttr(0.0)
        by_name = {part.name: part for part in parts}
        for rel, attr, name in (
            (usd_joint.CreateBody0Rel(), usd_joint.CreateLocalPos0Attr(), joint.parent),
            (usd_joint.CreateBody1Rel(), usd_joint.CreateLocalPos1Attr(), joint.child),
        ):
            at = by_name[name].at if name else (0.0, 0.0, 0.0)
            if name:
                rel.SetTargets([f"/RootNode/Geometry/{name}_obj_00"])
            attr.Set(Gf.Vec3f(*(a - b for a, b in zip(joint.anchor, at, strict=True))))

    # One variant set per engine, PhysX enabled; mass is the opinion that differs.
    sets = root.GetVariantSets()
    for engine, mass in (("PhysX", PHYSX_MASS), ("Newton", NEWTON_MASS), ("MuJoCo", PHYSX_MASS)):
        variant_set = sets.AddVariantSet(engine)
        for option in ("Disabled", "Enabled"):
            variant_set.AddVariant(option)
        variant_set.SetVariantSelection("Enabled")
        with variant_set.GetVariantEditContext():
            for part in parts:
                prim = stage.GetPrimAtPath(f"/RootNode/Geometry/{part.name}_obj_00")
                mass_api = UsdPhysics.MassAPI.Apply(prim)
                mass_api.CreateMassAttr(mass * part.mass)
                mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.01, 0.01, 0.01))
        variant_set.SetVariantSelection("Enabled" if engine == "PhysX" else "Disabled")
    stage.GetRootLayer().Save()
    return path
