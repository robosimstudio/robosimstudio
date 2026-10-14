# SPDX-License-Identifier: Apache-2.0
"""Where one built cloth's material lives in the model, and how to write it.

**Live tuning, and the reason a material studio can exist at all.** Newton keeps
a cloth's constitutive parameters as Warp arrays -- ``tri_materials``,
``edge_bending_properties``, ``tet_materials``, ``particle_radius`` -- and VBD
passes those arrays into its kernels on every colour group of every iteration of
every substep, indexing them per element. Nothing is cached solver-side, and
``ModelFlags`` has no entry for cloth material precisely because none is needed.
So assigning into one of those arrays takes effect on the very next substep, and
does so **even inside a captured CUDA graph**: a capture bakes the array's device
pointer, not the numbers behind it. Moving a stiffness slider is a few hundred
kilobytes of upload and nothing else.

The write is **per cloth**, because those arrays are indexed per element and
:func:`locate_cloth` selects the elements belonging to one entity. The
model-wide tiers -- the soft-contact scalars, and everything consumed at build
time -- are the studio's, in :mod:`robosimstudio_studio.tuning.apply`, which
re-exports what is here so a panel sees one surface.

**It sits in ``scene`` because that is the lowest layer that can express it.**
:func:`locate_cloth` takes a built :class:`~robosimstudio.domains.cloth`
entity and reads its build range; ``core`` may not name the IR these fields are
declared in, and ``studio`` is above the entity that has to answer for its own
tunables. Living here is what lets ``EntityCloth.tunables`` delegate rather than
carry a second copy of the field table -- which is the same reason
``EntityRigid`` reaches for ``core.tuning``'s writers instead of writing its own.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from robosimstudio.specs.params import LIVE, param_ui

__all__ = ["ClothTarget", "LiveMaterial", "locate_cloth"]


@dataclass(frozen=True)
class ClothTarget:
    """Which rows of the model's element arrays belong to one cloth.

    Particles and triangles come from the entity's own build range, which the
    builder's counts bracketed exactly. Edges and tetrahedra are not tracked
    there, so they are recovered by asking which elements reference a particle
    in the range -- sound because an element never spans two entities.

    Attributes:
        name: The cloth's name.
        particles: Particle indices.
        tris: Triangle indices.
        edges: Bending-edge indices, empty when the mesh has none.
        tets: Tetrahedron indices, empty for a membrane.
    """

    name: str
    particles: np.ndarray
    tris: np.ndarray
    edges: np.ndarray
    tets: np.ndarray

    def __repr__(self) -> str:
        return (
            f"ClothTarget({self.name!r}, particles={len(self.particles)}, tris={len(self.tris)}, "
            f"edges={len(self.edges)}, tets={len(self.tets)})"
        )


def _rows_referencing(flat: Any, span: tuple[int, int], stride: int, column: int) -> np.ndarray:
    """Indices of the ``stride``-wide rows whose ``column`` lands inside ``span``."""
    if flat is None:
        return np.zeros(0, dtype=np.int32)
    packed = np.asarray(flat.numpy(), dtype=np.int64).reshape(-1, stride)
    if not len(packed):
        return np.zeros(0, dtype=np.int32)
    lo, hi = span
    vertex = packed[:, column]
    return np.nonzero((vertex >= lo) & (vertex < hi))[0].astype(np.int32)


def locate_cloth(model: Any, entity: Any) -> ClothTarget:
    """Work out where one built cloth lives in the model's arrays.

    Args:
        model: The finalized model.
        entity: A built :class:`~robosimstudio.domains.cloth.EntityCloth`.
    """
    span = entity.range.particles
    tri_lo, tri_hi = entity.range.tris
    # Column 2 of an edge row is a vertex *on* the edge, so it is always a real
    # particle; columns 0 and 1 are the opposite vertices and are -1 on a
    # boundary edge.
    return ClothTarget(
        name=entity.spec.name,
        particles=np.arange(*span, dtype=np.int32),
        tris=np.arange(tri_lo, tri_hi, dtype=np.int32),
        edges=_rows_referencing(getattr(model, "edge_indices", None), span, 4, 2),
        tets=_rows_referencing(getattr(model, "tet_indices", None), span, 4, 0),
    )


class LiveMaterial:
    """Writes the live-tier fields of a material into one cloth's arrays.

    Every write downloads the array fresh, edits this cloth's rows and uploads
    the whole thing. No host copy is kept between writes: the arrays are shared
    by every cloth in the model, so a copy held here would put another cloth's
    rows -- or an outside write, a ``ctl eval`` or a preset -- back to whatever
    they were when it was taken.
    """

    def __init__(self, model: Any, target: ClothTarget) -> None:
        self.model = model
        self.target = target

    def _fetched(self, attr: str) -> np.ndarray | None:
        array = getattr(self.model, attr, None)
        return None if array is None else np.array(array.numpy(), copy=True)

    def _flush(self, attr: str, values: np.ndarray) -> None:
        getattr(self.model, attr).assign(values)

    def _write_rows(self, attr: str, rows: np.ndarray, column: int, value: float) -> bool:
        if not len(rows):
            return False
        values = self._fetched(attr)
        if values is None:
            return False
        values[rows, column] = value
        self._flush(attr, values)
        return True

    def _write_flat(self, attr: str, rows: np.ndarray, value: float) -> bool:
        if not len(rows):
            return False
        values = self._fetched(attr)
        if values is None:
            return False
        values[rows] = value
        self._flush(attr, values)
        return True

    # -------------------------------------------------------------- per field
    def write_field(self, name: str, value: Any) -> bool:
        """Write one live field. Returns whether anything was actually written.

        ``False`` means the field has no elements to write to in this cloth --
        a slab's core parameters on a membrane, bending on a mesh with no edges
        -- which is information a panel can show rather than an error.
        """
        t = self.target
        match name:
            case "tri_ke":
                return self._write_rows("tri_materials", t.tris, 0, float(value))
            case "tri_ka":
                return self._write_rows("tri_materials", t.tris, 1, float(value))
            case "tri_kd":
                return self._write_rows("tri_materials", t.tris, 2, float(value))
            case "edge_ke":
                return self._write_rows("edge_bending_properties", t.edges, 0, float(value))
            case "edge_kd":
                return self._write_rows("edge_bending_properties", t.edges, 1, float(value))
            case "core_shear_modulus":
                return self._write_rows("tet_materials", t.tets, 0, float(value))
            case "core_lame_lambda":
                return self._write_rows("tet_materials", t.tets, 1, float(value))
            case "core_damping":
                return self._write_rows("tet_materials", t.tets, 2, float(value))
            case "particle_radius":
                return self._write_flat("particle_radius", t.particles, float(value))
        return False

    def write(self, spec: Any, *, only: str = "") -> tuple[str, ...]:
        """Write every live field of ``spec``, or just ``only``.

        ``spec`` is deliberately untyped. It was annotated ``ClothMaterialSpec``,
        and that annotation was the *only* thing tying this module to a
        particular kind of matter -- the body walks ``fields(spec)`` and
        dispatches on each field's name, so it has always been structural. The
        annotation was decoration with a cost: ``scene`` sits below ``domains``,
        so naming a per-kind spec here is a cycle, and it was what kept
        ``ClothMaterialSpec`` in the shared IR long after nothing needed it
        there.

        Returns the names that landed. Writing the whole set is idempotent and
        costs a handful of small uploads, which is what a preset load wants;
        ``only`` is the single-slider path.
        """
        written: list[str] = []
        for f in fields(spec):
            if only and f.name != only:
                continue
            described = param_ui(spec, f.name)
            if described is None or described.tier != LIVE:
                continue
            if self.write_field(f.name, getattr(spec, f.name)):
                written.append(f.name)
        return tuple(written)
