# SPDX-License-Identifier: Apache-2.0
"""Whether a built task honours what its assets asked of the simulation.

``SimRequirements`` is an asset saying what it needs to behave like itself -- a
cable needing hands that can push it, a garment needing to collide with itself,
a film needing a solver that integrates a membrane at all. The composer in
:mod:`robosimstudio.scene.physics` *builds* from those declarations and records
what it built in a manifest; this module is the cheaper, static half that gates
every commit: **does the staged arrangement have a backend for every domain the
task's assets declare, and can that backend own the state the domain carries?**

Every check is static. No model, no scene, no GPU: an asset's requirement and a
``TaskSpec`` are both data. The dynamic half -- did the composer actually route
every declared contact -- is :meth:`robosimstudio.scene.physics.PhysicsProgram.problems`,
checked where a scene exists.
"""

from __future__ import annotations

from collections.abc import Callable

from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.specs import TaskSpec
from robosimstudio.specs.physics import DOMAINS, get_backend
from robosimstudio_bench.taskdoc.ast import TaskDoc

__all__ = ["sim_problems"]

Catalog = Callable[[str], AssetRecord | None]
"""Resolves an asset name to its declaration, or ``None``.

Injected rather than imported, the way :mod:`~robosimstudio_bench.goals.bind` takes
its catalog: it keeps this testable against a two-line fake, and it means a
caller checking a document against a *different* corpus can.
"""


def _declared(doc: TaskDoc, catalog: Catalog) -> dict[str, AssetRecord]:
    """Every instance's declaration, by instance name, skipping the uncatalogued.

    Uncatalogued rather than raising: a document naming an asset nothing declares
    is a ``catalog-gap`` and the typechecker's business, reported there against
    the whole document rather than here against one clause of it.
    """
    out: dict[str, AssetRecord] = {}
    for instance in doc.objects:
        record = catalog(instance.type) if instance.type else None
        if record is not None and record.sim is not None:
            out[instance.name] = record
    return out


def sim_problems(doc: TaskDoc, spec: TaskSpec, catalog: Catalog) -> list[str]:
    """Everything the staged arrangement fails to give the task's assets.

    Args:
        doc: The document, for which assets are actually in the scene.
        spec: What the loader built from it.
        catalog: Resolves an asset name to its declaration.

    Returns:
        One line per disagreement, naming the asset and what it asked for. Empty
        is the answer for every shipped variant, which is what makes a non-empty
        one worth reading.
    """
    records = _declared(doc, catalog)
    arrangement = spec.physics
    out: list[str] = []
    for name, record in sorted(records.items()):
        sim = record.sim
        out += [f"{name}: {problem}" for problem in sim.problems()]
        # Membership first: ``Registry.get`` raises where the plain dict this
        # used to be returned ``None``, and an unregistered domain is a problem
        # the *record* validator reports, not one to raise from here.
        if sim.domain not in DOMAINS:
            continue
        domain = DOMAINS.get(sim.domain)
        chosen = sim.backend or arrangement.backend_for(sim.domain)
        if chosen is None:
            if sim.domain == "static":
                continue
            out.append(
                f"{name} is a {sim.domain} and the staged arrangement names no backend for that domain "
                f"(it serves {list(arrangement.domains())}). The {sim.domain} would sit where it was built."
            )
            continue
        try:
            info = get_backend(chosen.solver)
        except KeyError as exc:
            out.append(f"{name}: {exc}")
            continue
        if info.supports:
            carries = _storage(sim.domain, chosen.solver, domain.state)
            missing = [kind for kind in carries if kind not in info.supports and kind != "shapes"]
            if missing:
                out.append(
                    f"{name} is a {sim.domain}, whose implementation on {chosen.solver!r} carries {list(carries)}, "
                    f"but that backend integrates only {list(info.supports)} -- it cannot own {missing}."
                )
    return out


def _storage(domain: str, solver: str, state: tuple[str, ...]) -> tuple[str, ...]:
    """What the implementation the arrangement selects would own.

    Storage is the implementation's, not the domain's: a rod is a chain of
    capsules on Newton and a Cosserat centerline of particles on SuperDex, so
    the domain lists both and neither implementation carries both. The
    composer resolves ``(kind, backend)`` to one class; so does this, and a
    particle implementation is one built on
    :class:`~robosimstudio.domains.capabilities.ParticleMixin`. A domain with
    no implementation of its own name is checked against everything it lists.
    """
    from robosimstudio.domains.capabilities import ParticleMixin
    from robosimstudio.scene.registry import get_entity

    try:
        cls = get_entity(domain, engine=solver)
    except (KeyError, LookupError):
        return state
    if issubclass(cls, ParticleMixin):
        return tuple(kind for kind in state if kind in ("particles", "shapes"))
    return tuple(kind for kind in state if kind != "particles")
