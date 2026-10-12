# SPDX-License-Identifier: Apache-2.0
"""The selector language: material-frame selections, written as data.

A deformable feature cannot be a world-frame box. "The hem" is a fact about the
*fabric*, so it is stated against the sheet's rest layout and carried through
whatever the cloth does afterwards. That is not a new idea here -- it is what
:meth:`~robosimstudio.domains.cloth.EntityCloth.resolve_sets` already
does, and its reason is worth repeating: a set recorded from the settled drape
"would follow whatever the drape did, and 'the tab' would stop being the tab the
moment the cloth moved".

What is new is that the selection is **data** rather than a Python lambda. Today
the shirt's six panels are closures built in a suite module; a sidecar cannot
hold a closure, an agent cannot author one, and a schema cannot validate one. So
this module serialises the algebra that already exists:

===================================  ==========================================
Recipe                               What it serialises
===================================  ==========================================
``beyond(axis, threshold, above)``   ``entities.cloth.selector_beyond``
``extreme(axis, band, high)``        ``entities.cloth.selector_extreme``
``indices([...])``                   ``entities.cloth.selector_indices``
``between(axis, lo, hi)``            the garment-folding task's ``_between``
``rim(axis, along, band, bins)``     the garment-folding task's ``_rim``
``all(of=[...])``                    the garment-folding task's ``_all``
===================================  ==========================================

Names are transcribed rather than regularised. A rename is cheap and reversible;
proving that a serialised recipe selects the *same particles* as the lambda it
replaces is the expensive part, and doing both at once would make the golden
meaningless. ``half``, ``any`` and ``invert`` are additions with no counterpart:
``half`` because a fraction of a sheet is the most common thing anyone wants and
deriving it from the rest extent survives a re-mesh, the other two because a
selection language closed under intersection but not union or complement is a
language with a hole in it.

Every recipe is JSON-able, so a sidecar carries constants and structure and never
a callable. That is the hard rule this module exists to enforce: **nothing
executable is ever embedded in a sidecar.** A selection the closed language
genuinely cannot express is a named resolver -- registered, versioned code
referenced by key -- which is :mod:`~robosimstudio.assetlib.semantics.resolve`'s
business, not this module's.

Compiled recipes have the signature the cloth entity already consumes:
``(rest[n, 3]) -> bool mask | int indices``, evaluated **once** at scene-ready.
Runtime cost afterwards is indexing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

__all__ = [
    "OPS",
    "ROUND_PLACES",
    "Selector",
    "compile_selector",
    "round_floats",
    "selector_to_dict",
    "validate_selector",
]

Selector = Callable[[np.ndarray], np.ndarray]
"""A resolved selection: rest positions in, mask or indices out."""

AXES = {"x": 0, "y": 1, "z": 2}
"""Axis names a recipe may use instead of an index, since ``axis: y`` reads and
reviews better than ``axis: 1`` in a sidecar a human or a model is authoring."""


def _axis(value: Any, where: str) -> int:
    """An axis, given as ``0|1|2`` or ``"x"|"y"|"z"``."""
    if isinstance(value, str):
        if value not in AXES:
            raise ValueError(f"{where}: unknown axis {value!r}; use one of {sorted(AXES)} or 0/1/2")
        return AXES[value]
    axis = int(value)
    if axis not in (0, 1, 2):
        raise ValueError(f"{where}: axis must be 0, 1 or 2, got {axis}")
    return axis


# --------------------------------------------------------------------- ops
def _op_beyond(*, axis: Any, threshold: float, above: bool = True, where: str) -> Selector:
    """Particles past a plane, in the sheet's own coordinates.

    The workhorse: "the tab" is everything past the box edge, "the hem" is
    everything below a height.
    """
    index = _axis(axis, where)
    limit = float(threshold)
    high = bool(above)

    def select(rest: np.ndarray) -> np.ndarray:
        return rest[:, index] > limit if high else rest[:, index] < limit

    return select


def _op_between(*, axis: Any, lo: float, hi: float, where: str) -> Selector:
    """Particles in a band.

    A band and not a half-plane, and on a multi-fold garment that distinction is
    the whole difference between a score that means something and one that does
    not: a crease's landing strip is the mirror of its own moving panel and
    nothing else. Declared as "everything on the far side" instead, a shirt's
    first fold counts the entire other half as its landing zone, and the second
    fold legitimately moving that half then reads as drag against the first --
    measured at 0.44 where the geometry deserved 0.9.
    """
    index = _axis(axis, where)
    low, high = float(lo), float(hi)
    if low > high:
        raise ValueError(f"{where}: lo {low} is above hi {high}")

    def select(rest: np.ndarray) -> np.ndarray:
        return (rest[:, index] >= low) & (rest[:, index] <= high)

    return select


def _op_extreme(*, axis: Any, band: float, high: bool = True, where: str) -> Selector:
    """Particles within ``band`` [m] of the sheet's own extreme along an axis.

    What picks out an edge row -- a handle, a corner, a graspable rim -- without
    the caller having to know where the sheet ended up.
    """
    index = _axis(axis, where)
    width = float(band)
    top = bool(high)

    def select(rest: np.ndarray) -> np.ndarray:
        edge = rest[:, index].max() if top else rest[:, index].min()
        return rest[:, index] > edge - width if top else rest[:, index] < edge + width

    return select


def _op_half(*, axis: Any, side: int = -1, where: str) -> Selector:
    """One half of a sheet, split at the midpoint of its own rest extent.

    Derived rather than authored, which is what makes it survive a re-mesh: the
    towel's ``near_half`` means the same fraction of the sheet whether the grid is
    21x21 or 41x41, where a baked index list means nothing at all after a
    decimation.

    A row lying *exactly* on the midpoint -- which every odd-resolution grid has
    -- goes to ``side: +1``. Arbitrary, but deterministic and documented, and it
    is what makes the two halves an exact partition rather than overlapping or
    dropping a row between them. A fold's crease row is genuinely ambiguous, and
    the one thing that must not happen is for it to land differently on different
    runs and move a fold score with nothing else having changed.
    """
    index = _axis(axis, where)
    if int(side) not in (-1, 1):
        raise ValueError(f"{where}: side must be -1 or +1, got {side!r}")
    below = int(side) < 0

    def select(rest: np.ndarray) -> np.ndarray:
        mid = 0.5 * (float(rest[:, index].min()) + float(rest[:, index].max()))
        return rest[:, index] < mid if below else rest[:, index] >= mid

    return select


def _op_rim(*, axis: Any, along: Any, band: float = 0.012, bins: int = 24, high: bool = True, where: str) -> Selector:
    """A flap's outer *boundary* as a full-width strip.

    Per bin along the fold line, the particles within ``band`` of that bin's own
    extreme. The towel taught why a grasp is a strip; the shirt taught why the
    strip must follow the boundary. A shirt's side flap is not a rectangle -- the
    sleeve reaches 3 cm further out than the shoulder or the hem corner -- so a
    plain extreme band selects only the sleeve tip: a 4 cm patch on a 31 cm flap.
    Latched there the panel is carried by one point and gathers into a purse,
    scoring 0.15-0.20 where the same machinery folds the straight-edged towel at
    0.9. Binning along the crease traces the sleeve, the shoulder and the hem
    corner alike -- the line a person's fingers would run along.
    """
    index = _axis(axis, where)
    direction = _axis(along, where)
    width = float(band)
    count = int(bins)
    top = bool(high)
    if count < 1:
        raise ValueError(f"{where}: bins must be at least 1, got {count}")

    def select(rest: np.ndarray) -> np.ndarray:
        mask = np.zeros(len(rest), dtype=bool)
        lo, hi = float(rest[:, direction].min()), float(rest[:, direction].max())
        edges = np.linspace(lo, hi, count + 1)
        which = np.clip(np.digitize(rest[:, direction], edges) - 1, 0, count - 1)
        for b in range(count):
            in_bin = which == b
            if not in_bin.any():
                continue
            values = rest[in_bin, index]
            extreme = values.max() if top else values.min()
            near = values > extreme - width if top else values < extreme + width
            mask[np.flatnonzero(in_bin)[near]] = True
        return mask

    return select


def _op_indices(*, of: Sequence[int], where: str) -> Selector:
    """A literal set of local indices, for a selection with no geometric rule.

    Tier-1 content in a tier-2 wrapper, and the one recipe that is **not**
    re-mesh safe: a feature using it must pin the mesh hash of the layout the
    indices were taken from, or it rots silently the first time somebody
    decimates the asset.
    """
    idx = np.asarray(list(of), dtype=np.int64)

    def select(rest: np.ndarray) -> np.ndarray:
        del rest
        return idx

    return select


def _op_all(*, of: Sequence[Any], where: str) -> Selector:
    """Particles satisfying every sub-selection -- a rectangle, from bands.

    Composes boolean masks only; a literal ``indices`` recipe inside an ``all``
    is refused rather than coerced, because the intersection of a mask and an
    index list is a different operation wearing the same name.
    """
    parts = [compile_selector(item, where=f"{where}.of[{i}]") for i, item in enumerate(of)]
    if not parts:
        raise ValueError(f"{where}: 'all' needs at least one sub-selector")

    def select(rest: np.ndarray) -> np.ndarray:
        mask = np.ones(len(rest), dtype=bool)
        for part in parts:
            mask &= _as_mask(part(rest), len(rest), where)
        return mask

    return select


def _op_any(*, of: Sequence[Any], where: str) -> Selector:
    """Particles satisfying at least one sub-selection."""
    parts = [compile_selector(item, where=f"{where}.of[{i}]") for i, item in enumerate(of)]
    if not parts:
        raise ValueError(f"{where}: 'any' needs at least one sub-selector")

    def select(rest: np.ndarray) -> np.ndarray:
        mask = np.zeros(len(rest), dtype=bool)
        for part in parts:
            mask |= _as_mask(part(rest), len(rest), where)
        return mask

    return select


def _op_invert(*, of: Any, where: str) -> Selector:
    """Everything the sub-selection does not pick -- "the rest of the sheet"."""
    part = compile_selector(of, where=f"{where}.of")

    def select(rest: np.ndarray) -> np.ndarray:
        return ~_as_mask(part(rest), len(rest), where)

    return select


def _as_mask(picked: np.ndarray, count: int, where: str) -> np.ndarray:
    """A sub-selector's output as a boolean mask, refusing an index list."""
    array = np.asarray(picked)
    if array.dtype != np.bool_:
        raise ValueError(
            f"{where}: sub-selectors must be geometric (mask-valued); "
            "an 'indices' recipe cannot be composed -- select it directly"
        )
    if len(array) != count:
        raise ValueError(f"{where}: mask has {len(array)} entries for {count} particles")
    return array


OPS: dict[str, Any] = {
    "beyond": _op_beyond,
    "between": _op_between,
    "extreme": _op_extreme,
    "half": _op_half,
    "rim": _op_rim,
    "indices": _op_indices,
    "all": _op_all,
    "any": _op_any,
    "invert": _op_invert,
}
"""The closed set. Growing it is a deliberate act with a golden to update -- the
same firewall the combinator set has, for the same reason: a selection language
that grows on demand becomes a programming language nobody validated."""


# ----------------------------------------------------------------- compile
def compile_selector(recipe: Any, *, where: str = "selector") -> Selector:
    """Compile a recipe into the callable the cloth entity consumes.

    Args:
        recipe: A ``{"op": ..., **params}`` mapping. A bare list is read as
            ``all`` of its entries, since a feature is most often the
            intersection of a few bands and making the common case wordless is
            worth one special form.
        where: Path used in error messages, so a bad recipe names itself.

    Returns:
        ``(rest[n, 3]) -> mask | indices``, to be evaluated once on the rest
        layout at scene-ready.

    Raises:
        ValueError: On an unknown op, a missing parameter, or an unknown
            parameter. Unknown parameters are refused rather than ignored:
            silently dropping a typo'd ``bins`` yields a selection that quietly
            kept the default, and that gets blamed on the physics.
    """
    if isinstance(recipe, (list, tuple)):
        return _op_all(of=list(recipe), where=where)
    if not isinstance(recipe, dict):
        raise ValueError(f"{where}: expected a recipe mapping or a list, got {type(recipe).__name__}")

    params = dict(recipe)
    op = params.pop("op", "")
    if not op:
        raise ValueError(f"{where}: recipe has no 'op'; one of {sorted(OPS)}")
    if op not in OPS:
        raise ValueError(f"{where}: unknown selector op {op!r}; known ops are {sorted(OPS)}")

    try:
        return OPS[op](where=where, **params)
    except TypeError as exc:  # missing or unexpected parameter
        raise ValueError(f"{where}: bad parameters for op {op!r}: {exc}") from exc


def validate_selector(recipe: Any, *, where: str = "selector") -> list[str]:
    """Problems with a recipe, without needing a sheet to resolve it against.

    What the gauntlet's schema stage calls: an agent-authored sidecar is checked
    for structure long before anything is compiled or simulated.
    """
    try:
        compile_selector(recipe, where=where)
    except ValueError as exc:
        return [str(exc)]
    return []


ROUND_PLACES = 9
"""Decimal places numbers are rounded to on the way into canonical form.

Not cosmetic. ``WIDTH / 6`` is exactly ``0.06999999999999999`` in binary
floating point, and a catalog that serialises it that way is one a reviewer
mistrusts and a golden diff full of noise. Nine places is far below any
measurement in this stack -- a nanometre on a garment -- so nothing real is lost,
and two documents that mean the same crease now compare equal.
"""


def round_floats(value: Any) -> Any:
    """Round every float in a nested structure, leaving everything else alone."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, ROUND_PLACES)
    if isinstance(value, dict):
        return {k: round_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(v) for v in value]
    return value


def selector_to_dict(recipe: Any) -> dict[str, Any]:
    """Normalise a recipe to its canonical JSON-able form.

    Axis names become indices, a bare list becomes an explicit ``all``, and
    floats are rounded to :data:`ROUND_PLACES` -- so two sidecars that mean the
    same selection compare equal, which is what lets a golden diff be about
    content rather than spelling.
    """
    if isinstance(recipe, (list, tuple)):
        return {"op": "all", "of": [selector_to_dict(item) for item in recipe]}
    params = dict(recipe)
    op = params.pop("op", "")
    out: dict[str, Any] = {"op": op}
    for key, value in params.items():
        if key in ("axis", "along"):
            out[key] = _axis(value, f"selector.{op}")
        elif key == "of":
            out[key] = (
                [selector_to_dict(item) for item in value]
                if isinstance(value, (list, tuple))
                else selector_to_dict(value)
            )
        else:
            out[key] = round_floats(value)
    return out
