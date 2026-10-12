# SPDX-License-Identifier: Apache-2.0
"""Named materials, and how each one was measured.

A :class:`MaterialRecord` is a named material plus the measurement that
justifies it, so a tool can put a slider next to "real bath towelling is
2.0-3.0 cm of bending length, this sheet is 2.63" instead of next to a bare
float. That is the difference between tuning and guessing.

**Calibration is against cloth, not taste.** The drape bench clamps a
strip, lets it droop, and inverts the elastica for a *bending length* -- the
quantity textiles are actually characterised by (ASTM D1388) and tabulated for
real fabric. Every ``edge_ke`` and ``core_shear_modulus`` below cites a row of
that sweep, and :class:`CalibrationSpec` is where the row now lives.

A caveat that matters when reading the membrane presets: the hinge bending
model's rigidity scales with cell size, so a stiffness calibrated at one mesh
resolution is a *different sheet* at another. :attr:`CalibrationSpec.cell`
records the resolution each was measured at, and is not decoration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from robosimstudio.registry import Registry

__all__ = [
    "SCHEMA_VERSION",
    "CalibrationSpec",
    "MaterialRecord",
    "get_material",
    "list_materials",
    "material_from_dict",
    "material_to_dict",
    "read_material",
    "register_material",
    "write_material",
]

SCHEMA_VERSION = 1
"""Version written into every material file, so an old one can be recognised."""


@dataclass(frozen=True, kw_only=True)
class CalibrationSpec:
    """What measurement a material's numbers came from.

    Attributes:
        fabric: The real material being imitated, in words.
        bench: What measured it, e.g. ``"the cloth drape bench (ASTM D1388)"``.
        knob: Which field the sweep varied.
        cell: Mesh cell the sweep was run at [m], or 0 when it does not apply.
            Load bearing for a membrane: hinge bending rigidity scales with cell
            size, so the same stiffness at another resolution is another sheet.
        bending_length_cm: What this material measures [cm], or 0 if unmeasured.
        target_cm: The real fabric's band [cm], or ``None`` when the fabric is
            not one the reference table covers.
        sweep: The measured ``(knob value, bending length [cm])`` rows. What
            makes the chosen value legible as a choice rather than a constant --
            you can see whether it sits in the middle of the band or on an edge.
        notes: Anything the fields above cannot hold.
    """

    fabric: str = ""
    bench: str = ""
    knob: str = ""
    cell: float = 0.0
    bending_length_cm: float = 0.0
    target_cm: tuple[float, float] | None = None
    sweep: tuple[tuple[float, float], ...] = ()
    notes: str = ""

    @property
    def in_band(self) -> bool | None:
        """Whether the measurement lands inside the real fabric's band.

        ``None`` when there is nothing to compare -- an unmeasured material or a
        fabric with no tabulated target. A tool should show that as "unknown"
        rather than as a failure, because it is.
        """
        if self.target_cm is None or not self.bending_length_cm > 0.0:
            return None
        low, high = self.target_cm
        return low <= self.bending_length_cm <= high


MATERIAL_KINDS: Registry[type] = Registry("material kind", ("robosimstudio.domains",))
"""What kinds of matter a material may be, and the spec type each is.

**Open, and it was closed at one.** This registry is called ``MATERIALS`` and
held only cloth: ``MaterialRecord.spec`` was typed ``ClothMaterialSpec``, the
JSON writer wrote ``"kind": "cloth"`` as a literal, and the reader refused
anything else outright. The cost was not the branch -- it was that
``SoftMaterialSpec`` could not be registered *at all*, so ``SOFT_FOAM`` and
``FIRM_FOAM`` sat in ``domains/soft`` as module constants that nothing could
look up by name, save to a file, or offer in a preset menu. A registry named for
the general thing, serving one.

A kind registers the spec type it means; the record's kind follows from the type
of its spec, so there is one statement of what a material *is* rather than a
field that can disagree with the object beside it.

Scans the domains for the same reason ``MATERIALS`` does, and it is not
optional: without it, which kinds exist would depend on **what was asked
first** -- ``MATERIALS.names()`` loads the domains and fills this as a side
effect, so a caller that asked this one first would get an empty registry and a
caller that asked the other first would not."""


def register_material_kind(kind: str, spec_type: type) -> type:
    """Declare that ``kind`` means ``spec_type``. Call beside the domain's materials."""
    MATERIAL_KINDS.register(kind, spec_type, origin=getattr(spec_type, "__module__", ""))
    return spec_type


def kind_of(spec: Any) -> str:
    """Which registered kind a material spec is.

    Raises:
        LookupError: When nothing has registered its type, naming what has. A
            material whose kind is unknown cannot be written to a file that
            anything could read back, so guessing a default here would only move
            the failure to the load.
    """
    for kind, cls in MATERIAL_KINDS:
        if type(spec) is cls:
            return kind
    raise LookupError(
        f"no material kind is registered for {type(spec).__name__}; registered: {MATERIAL_KINDS.names()}. "
        "A domain registers its own with register_material_kind()."
    )


@dataclass(frozen=True, kw_only=True)
class MaterialRecord:
    """A named material and the measurement behind it.

    Attributes:
        name: Registry key, and the stem of the file it saves to.
        spec: The constitutive parameters themselves -- whichever spec type its
            kind registered. Deliberately untyped here: this module is the
            *registry*, and a registry that names one member's type is a
            registry for that member.
        notes: What this material is, in one or two sentences.
        calibration: How its numbers were arrived at, or ``None`` for one that
            has not been measured -- a tuned draft, most often.
    """

    name: str
    spec: Any
    notes: str = ""
    calibration: CalibrationSpec | None = None

    def tuned(self, name: str, **changes: Any) -> MaterialRecord:
        """A renamed copy with some fields changed, and no calibration.

        Dropping the calibration is the point rather than an oversight: the
        moment a number moves, the measurement that justified the old one no
        longer describes this material, and carrying it forward would make a
        tuned draft claim a provenance it does not have.
        """
        return MaterialRecord(
            name=name,
            spec=replace(self.spec, **changes),
            notes=f"Tuned from {self.name!r}." + (f" {self.notes}" if self.notes else ""),
            calibration=None,
        )


MATERIALS: Registry[MaterialRecord] = Registry("material", ("robosimstudio.domains",))
"""Every registered material, found by scanning the domain bundles.

The values used to sit in this module and register themselves on import. They
are facts about a *substrate* -- terry towelling is a fact about cloth -- so they
live in ``domains/cloth/materials.py`` now, and this scans for them exactly as
``DOMAINS`` scans for the domains. A domain that ships a material needs no entry
here."""


def register_material(record: MaterialRecord) -> MaterialRecord:
    """Register a material under its own name."""
    return MATERIALS.register(record.name, record, origin=__name__)


def get_material(name: str) -> MaterialRecord:
    """Look one up by name."""
    return MATERIALS.get(name)


def list_materials() -> list[str]:
    """Every registered name, sorted."""
    return MATERIALS.names()


# ------------------------------------------------------------------ json
def material_to_dict(record: MaterialRecord) -> dict[str, Any]:
    """A material as plain JSON-able data."""
    out: dict[str, Any] = {
        "version": SCHEMA_VERSION,
        "name": record.name,
        "kind": kind_of(record.spec),
        "notes": record.notes,
        "material": asdict(record.spec),
    }
    if record.calibration is not None:
        out["calibration"] = asdict(record.calibration)
    return out


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    """Construct ``cls`` from ``data``, refusing keys it does not have.

    Unknown keys are an error rather than a shrug: silently dropping one turns a
    typo'd ``edge_kd`` into a material that quietly kept the default, which is
    the kind of thing that gets blamed on the solver.
    """
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {unknown}; {cls.__name__} has {sorted(known)}")
    return cls(**data)


def material_from_dict(data: dict[str, Any], *, where: str = "material") -> MaterialRecord:
    """Rebuild a material from :func:`material_to_dict` output.

    Raises:
        ValueError: On a version this code does not know, a kind it cannot
            build, or a field the spec does not have.
    """
    version = data.get("version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"{where}: schema version {version!r}, expected {SCHEMA_VERSION}")
    kind = str(data.get("kind", "cloth"))
    if kind not in MATERIAL_KINDS:
        raise ValueError(
            f"{where}: unknown material kind {kind!r}; registered: {MATERIAL_KINDS.names()}. "
            "A kind arrives with the domain that declares it, so this usually means the domain "
            "that owns this material is not on any root."
        )
    if "name" not in data or "material" not in data:
        raise ValueError(f"{where}: needs both 'name' and 'material'")

    calibration = data.get("calibration")
    if calibration is not None:
        calibration = dict(calibration)
        for key in ("target_cm", "sweep"):
            if calibration.get(key) is not None:
                calibration[key] = (
                    tuple(tuple(row) for row in calibration[key]) if key == "sweep" else tuple(calibration[key])
                )
        calibration = _build(CalibrationSpec, calibration, f"{where}.calibration")

    return MaterialRecord(
        name=str(data["name"]),
        spec=_build(MATERIAL_KINDS.get(kind), dict(data["material"]), f"{where}.material"),
        notes=str(data.get("notes", "")),
        calibration=calibration,
    )


def write_material(record: MaterialRecord, path: str | Path) -> Path:
    """Save a material as JSON. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(material_to_dict(record), indent=2) + "\n")
    return path


def read_material(path: str | Path) -> MaterialRecord:
    """Load a material from JSON."""
    path = Path(path)
    return material_from_dict(json.loads(path.read_text()), where=str(path))
