# SPDX-License-Identifier: Apache-2.0
"""Compiling an asset down into what the toolkit already consumes.

This module is the load-bearing claim of the whole layering, made executable. The
asset component sits *above* the toolkit, which sounds backwards -- surely the
compiler needs to know about features? -- and the reason it does not is here:

============================  ==================================================
Declared here                 Becomes, for the toolkit
============================  ==================================================
a ``set`` feature's recipe    a ``Selector`` callable in ``ClothSpec.particle_sets``
a ``region`` feature          a ``RegionSpec``
a goal tree                   an ``Evaluator`` in ``TaskSpec.evaluator``
sim requirements              a named arrangement profile
============================  ==================================================

Every one of those is a slot the toolkit **already has and already fills**.
``ClothSpec.particle_sets`` has always held callables; ``TaskSpec.evaluator`` has
always held an evaluator. So the toolkit never learns that features, recipes or
goal trees exist -- it keeps consuming exactly what it consumes today, and this
layer keeps the freedom to change without touching the runtime.

**What is not here.** Building entities, posing them, or deciding what a scene
contains -- all of that stays with the compiler. This module turns declarations
into specs and stops.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from robosimstudio.specs import AssetSpec, RegionSpec

from .record import AssetRecord
from .semantics.features import FeatureSpec
from .semantics.resolve import get_resolver
from .semantics.selectors import compile_selector

__all__ = ["declared_asset", "declared_file", "particle_sets", "region_specs", "selector_for"]


def declared_file(record: AssetRecord, key: str = "mesh") -> str:
    """The path an asset declares under one name.

    Args:
        record: The declaration.
        key: Which file. ``"mesh"`` by default, since that is what a scene needs
            of most assets; an imported cell declares ``"export"`` or ``"stage"``
            instead.

    Returns:
        A path ``resolve_path`` can resolve: absolute when the file ships beside
        the declaration that owns it, relative to an asset root otherwise. The
        record is normalised at registration, so this is just the lookup.

    Raises:
        KeyError: Naming what the record does declare. An asset that has not
            said where its geometry is cannot be guessed at, and a caller that
            asked for the wrong key wants to know which one exists.
    """
    try:
        declared = record.files[key]
    except KeyError:
        raise KeyError(
            f"asset {record.name!r} declares no file {key!r}; it declares {sorted(record.files) or 'none'}"
        ) from None
    return declared


def declared_asset(record: AssetRecord, key: str = "mesh", **kwargs: Any) -> AssetSpec:
    """An :class:`AssetSpec` pointing at what the record declares.

    The one-line form of "load the thing this asset is about". Everything else an
    ``AssetSpec`` carries -- scale, up-axis, whatever a particular import needs --
    stays the caller's, because those are facts about *this* use of the geometry
    rather than about the asset.
    """
    return AssetSpec(path=declared_file(record, key), **kwargs)


def selector_for(spec: FeatureSpec) -> Callable[[Any], Any]:
    """One feature's selection, as the callable ``ClothSpec`` has always held.

    The three tiers converge on one signature, which is the point -- a caller
    never learns which tier a feature happened to use:

    * ``recipe`` compiles the closed selector language;
    * ``resolver`` looks up registered code by its versioned key and binds the
      remaining payload as its parameters;
    * ``literal`` returns its baked index list.

    Raises:
        ValueError: For a literal feature carrying no index list -- it selects
            nothing, and a set that silently selects nothing is worse than one
            that refuses, because every measure over it reports a confident zero.
    """
    if spec.tier == "recipe":
        return compile_selector(spec.payload.get("select", spec.payload), where=f"feature {spec.name!r}")

    if spec.tier == "resolver":
        key = str(spec.payload.get("resolver", ""))
        if not key:
            raise ValueError(f"feature {spec.name!r}: tier 'resolver' names no resolver")
        params = {k: v for k, v in spec.payload.items() if k != "resolver"}

        # The lookup is deferred to call time, deliberately. An asset may name a
        # resolver nobody has written yet -- the shirt's collar does, and saying
        # so is exactly what tier 3 is *for*. Resolving eagerly would make that
        # honest declaration poison the whole asset: no shirt feature could
        # lower, so no shirt task could run, because of a collar no task in the
        # corpus references. Deferred, the failure surfaces at scene-ready in
        # the one task that actually asks for the collar, naming it.
        def select(rest, _key=key, _params=params):
            return get_resolver(_key)(rest, **_params)

        return select

    for key in ("indices", "particles", "vertices"):
        if key in spec.payload:
            baked = list(spec.payload[key])

            def select(rest, _baked=baked):
                del rest
                return _baked

            return select

    raise ValueError(
        f"feature {spec.name!r}: tier 'literal' carries no index list, so it selects nothing; "
        "give it an 'indices' payload or declare it as a recipe"
    )


def particle_sets(record: AssetRecord, *, kinds: tuple[str, ...] = ("set",)) -> tuple:
    """An asset's selections, in the shape ``ClothSpec.particle_sets`` takes.

    Returns ``((name, selector), ...)`` -- exactly what a cloth spec is
    constructed with today, so a declared shirt and a hand-written one are
    interchangeable to the compiler.

    Only selection-bearing kinds are lowered. A ``template`` is a payload the
    *goal* reads and not a set of particles, so it is skipped rather than
    failing: an asset is allowed to carry things the cloth spec has no slot for.

    **``panel`` is skipped for the same reason, and it did not used to be.** A
    panel reads like a selection and is not one: the kind's own summary says "a
    region of a sheet that one fold carries, *paired with where it lands*", and
    a thing that names two selections plus the line between them is a
    description of a crease rather than a set of particles.
    """
    return tuple((spec.name, selector_for(spec)) for spec in record.features if spec.kind in kinds)


def region_specs(record: AssetRecord, instance: str = "") -> tuple[RegionSpec, ...]:
    """An asset's regions, as the toolkit's own ``RegionSpec``.

    Args:
        record: The asset.
        instance: What this asset is called in the scene. Region names are
            qualified with it (``bin2.hammer_slot``) so two bins in one scene do
            not collide, and the frame defaults to the instance -- a region is
            stated in its target's own frame, which is what lets a drawer's
            interior travel with the drawer.

    The z window defaults to BDDL's literal 0 to 0.15 m above the frame origin,
    because that convention is what makes "in the bin" mean what people expect:
    an object held above an open bin is not in it, and one resting on its floor
    is.
    """
    name = instance or record.name
    out = []
    for spec in record.features:
        if spec.kind != "region":
            continue
        payload = spec.payload
        ranges = tuple(float(v) for v in payload["xy_ranges"])
        z_range = tuple(float(v) for v in payload.get("z_range", (0.0, 0.15)))
        # A region riding a jointed part names that part; one that does not rides
        # the instance itself.
        frame = payload.get("frame", "")
        out.append(
            RegionSpec(
                name=f"{name}.{spec.name}",
                frame=f"{name}.{frame}" if frame else name,
                xy_ranges=ranges,  # type: ignore[arg-type]
                z_range=z_range,  # type: ignore[arg-type]
            )
        )
    return tuple(out)
