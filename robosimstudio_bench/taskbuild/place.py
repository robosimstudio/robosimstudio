# SPDX-License-Identifier: Apache-2.0
"""Placing what a document typed: the record's factory or mesh, and the export's sidecar.

A document names *types* and, through its staging, where their placement was
recorded. Two sources answer:

* **The catalog.** :func:`build_entities` asks each typed entity's record for
  its spec -- a ``build`` factory or a mesh -- under the document's own name,
  and at its recorded start pose when one was supplied.
* **The export.** :func:`sidecar_poses` reads where an imported scene's free
  bodies stand. :func:`from_export` folds a welded container's frame into the
  regions that hang off it and drops the container's handle (it is geometry,
  not a body), and for ``(:objects-from ...)`` re-imports the free objects
  from their own stages, so a difference measured afterwards is the
  importer's alone.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from robosimstudio.specs import ObjectSpec, TaskSpec, replace
from robosimstudio_bench.taskdoc.sidecar import load_sidecar, wxyz_to_xyzw

from .lower import rebase_regions, recorded_asset
from .staging import Staging

__all__ = ["build_entities", "export_masses", "from_export", "sidecar_hosts", "sidecar_poses"]

_PLACEMENT = ("pose", "quat", "body_name", "joint_name")


def build_entities(
    spec: TaskSpec, *, poses: dict[str, dict[str, Any]] | None = None, resident: bool = False
) -> TaskSpec:
    """Every entity the document typed, built the way its record says.

    A document names a *type* and nothing else -- ``(:objects towel - drape_towel)``
    -- and the type is an asset-registry key. What a towel *is* lives on the
    towel, once, and a task that wants one says so by name.

    Two ways a record answers. A record with a ``build`` factory is asked for
    its spec outright -- a ``ClothSpec``, a ``SoftSpec``, a welded ``StaticSpec``,
    whatever the kind is -- and that replaces the placeholder the document
    lowered to, under the document's own name and, where ``poses`` recorded
    one, at the document's placement (plus any further field of the built
    spec the entry states by name). Keyed on the record and not on a list of
    domains, so a kind nobody here anticipated builds the same way. A record
    with a mesh and no factory gets the mesh (a rigid object). A free body that
    is ``resident`` -- one the export already holds and the sidecar places --
    is left as the document lowered it, factory or no factory: the export is
    the geometry. A record the catalog does not know, or one that
    declares neither, is left alone: the typechecker already reported the
    first as a catalog gap, and the second is an asset that is placed rather
    than imported.
    """
    from robosimstudio.assetlib.lower import declared_asset
    from robosimstudio.assetlib.registry import get_asset

    supplied = poses or {}
    entities = []
    for item in spec.entities:
        category = getattr(item, "category", "") or ""
        entry = supplied.get(item.name, {})
        try:
            record = get_asset(category) if category else None
        except (KeyError, LookupError):
            record = None
        if record is None or (resident and isinstance(item, ObjectSpec) and not item.static):
            # A free body the export already holds is placed by the sidecar and
            # wants neither a factory's spec nor a mesh of its own.
            entities.append(_stated(item, entry))
            continue
        build = getattr(record, "build", None)
        if callable(build):
            built = build()
            changes = {"name": item.name} if getattr(built, "name", item.name) != item.name else {}
            changes.update({k: getattr(item, k) for k in _PLACEMENT if k in entry and hasattr(built, k)})
            # A poses entry may state any other field the built spec has -- a
            # gear board's ``bends``, a welded tray's ``xform``. Taken from the
            # entry itself, because the lowered placeholder only ever carried
            # the placement fields.
            stated = {k: v for k, v in entry.items() if k not in _PLACEMENT and hasattr(built, k)}
            if "asset" in stated:
                # Merged onto the factory's, not swapped for it: see
                # ``lower.recorded_asset``.
                stated["asset"] = recorded_asset(stated["asset"], getattr(built, "asset", None))
            changes.update(stated)
            entities.append(_stated(replace(built, **changes) if changes else built, entry))
            continue
        if getattr(item, "asset", None) is not None:
            entities.append(_stated(item, entry))
            continue
        try:
            entities.append(_stated(replace(item, asset=declared_asset(record)), entry))
        except (KeyError, LookupError):
            entities.append(_stated(item, entry))
    return replace(spec, entities=tuple(entities))


_RECORDED = ("collider",)
"""Poses keys the lowering reads that are not placement: stated on the handle, never on a built spec."""


def _stated(built: Any, entry: dict[str, Any]) -> Any:
    """``built``, once every key its poses entry states is a field it has.

    A recorded key that names no field of the spec the entity became is
    refused rather than dropped: a ``frictoin`` would otherwise be a typo that
    changed nothing and said nothing, and a tray's ``xform`` recorded against
    a record that builds no spec would be a placement that never happened.
    """
    unknown = sorted(k for k in entry if k not in _PLACEMENT and k not in _RECORDED and not hasattr(built, k))
    if unknown:
        name = getattr(built, "name", "?")
        raise ValueError(
            f"{name}: recorded start pose names {unknown}, which a {type(built).__name__} has no field for"
        )
    return built


def sidecar_poses(export: str, *, collider: str = "") -> dict[str, dict[str, Any]]:
    """Where an export's free bodies stand, keyed by the name a document uses.

    The sidecar records each body's root and free joint; a body that names
    neither follows the exporter's ``<name>_main`` / ``<name>_joint0``
    convention. ``collider`` is stated on every entry, replacing the single
    oriented box the export fitted beside its drawn mesh.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, entry in load_sidecar(export)["objects"].items():
        out[name] = {
            "pose": tuple(float(v) for v in entry["pos"]),
            "quat": wxyz_to_xyzw(entry["quat_wxyz"]),
            "body_name": entry.get("root_body") or f"{name}_main",
            "joint_name": entry.get("joint") or f"{name}_joint0",
        }
        if collider:
            out[name]["collider"] = collider
    return out


def sidecar_hosts(export: str) -> dict[str, tuple[str, tuple[float, float, float]]]:
    """Each welded container, as the frame that holds it and where in that frame.

    The sidecar writes ``containers[name].frame_in_<host>``; the host is read
    off the key, so a container welded into ``bin1`` and one welded into
    ``bin2`` both resolve.
    """
    out: dict[str, tuple[str, tuple[float, float, float]]] = {}
    for name, entry in (load_sidecar(export).get("containers") or {}).items():
        key = next((k for k in entry if k.startswith("frame_in_")), None)
        if key is None:
            raise ValueError(f"{export}: container {name!r} records no frame_in_<host>")
        dx, dy, dz = (float(v) for v in entry[key])
        out[name] = (key.removeprefix("frame_in_"), (dx, dy, dz))
    return out


def export_masses(export: str) -> dict[str, float]:
    """Each free body's mass [kg], off its one colliding geom in the export's ``scene.xml``.

    Not in the sidecar, which records frames and sizes and not masses. Read
    from the export rather than the per-asset MJCF because the export is the
    scene being reproduced.
    """
    from robosimstudio.paths import resolve_path

    root = ET.parse(resolve_path(f"{export}/scene.xml")).getroot()
    masses: dict[str, float] = {}
    for body in root.find("worldbody").findall("body"):
        colliders = [g for g in body.findall("geom") if g.get("contype") != "0" and g.get("mass")]
        if len(colliders) == 1:
            masses[(body.get("name") or "").removesuffix("_main")] = float(colliders[0].get("mass"))
    return masses


def from_export(spec: TaskSpec, staging: Staging) -> TaskSpec:
    """A lowered task, with what its export's sidecar knows applied.

    * A container welded into a bin is geometry with no body: its region is
      restated in the bin's frame (:func:`~.lower.rebase_regions`) and the
      handle the document lowered it to is dropped, since the compiler would
      otherwise go looking for a body that is not there.
    * ``(:objects-from usd|usd_geom)`` re-imports every free object from that
      stage of its record -- same names, same poses, same goals -- with the
      geometry-only flavour told its collider and its mass, which the
      physics-bearing one carries itself.
    """
    if not staging.export:
        return spec
    hosts = sidecar_hosts(staging.export)
    entities = [e for e in spec.entities if getattr(e, "name", "") not in hosts]
    if staging.objects_from:
        masses = export_masses(staging.export) if staging.objects_from == "usd_geom" else {}
        entities = [_restaged(e, staging, masses) for e in entities]
    return replace(spec, entities=tuple(entities), regions=rebase_regions(spec.regions, hosts))


def _restaged(item: Any, staging: Staging, masses: dict[str, float]) -> Any:
    from robosimstudio.assetlib.lower import declared_asset
    from robosimstudio.assetlib.registry import get_asset

    if not isinstance(item, ObjectSpec) or item.static:
        return item
    geometry_only = staging.objects_from == "usd_geom"
    return replace(
        item,
        asset=declared_asset(
            get_asset(item.category),
            staging.objects_from,
            approximate=(staging.collider or None) if geometry_only else None,
            keep_visual=True,
        ),
        world_xform=True,
        mass=masses.get(item.name) if geometry_only else None,
        collider="",
    )
