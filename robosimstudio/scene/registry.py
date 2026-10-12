# SPDX-License-Identifier: Apache-2.0
"""Which places a task can be staged in, and which entity classes exist.

Two registries that answer the same kind of question -- what kinds of thing can
the compiler build -- and are populated from opposite directions. Scenes come
from packages the registry imports; entity classes come from whatever registers
one while being imported for another reason.

The library keeps ``void`` and nothing else: every other cell is a *particular
room* somebody measured, so it is content.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

from robosimstudio.registry import Registry, same_source
from robosimstudio.specs import CellSpec

__all__ = [
    "ANY_ENGINE",
    "CELLS",
    "ENTITIES",
    "entity_engines",
    "entity_key",
    "get_cell",
    "get_entity",
    "list_cells",
    "register_cell",
    "register_entity",
    "resolve_entity_class",
]


CELLS: Registry[CellSpec] = Registry(
    "cell",
    ("robosimstudio.scene.builtin",),
    hint="A cell declared by a task directory registers when the benchmark tier is imported: where it is installed, "
    "import its taskbuild package (or ask for a task) and try again. Without it, the cells listed are all "
    "there are.",
    same=operator.eq,
)
"""Places a task can be staged in. **Two packages, and the split is the point.**

The library keeps ``void`` -- the ground plane and nothing else, which declares
no geometry and is the null hypothesis a portability bug is checked against. Every
other cell is a *particular room* somebody measured, so it is content and ships
in the task directory that measured it, beside its assets and its arrangement.

**This registry knows nothing about where content lives.** It scans the one
package the library ships and stops. A task directory's cells arrive because the
benchmark tier -- the package beside the library -- installs a
populator with :meth:`~robosimstudio.registry.Registry.add_discover` when its
``taskbuild`` is imported -- which is what keeps ``get_cell("bench_075")``, a
question about a library bench, from executing every task on disk to answer it."""
ENTITIES: Registry[type] = Registry("entity", ("robosimstudio.domains",), same=same_source)
"""Entity classes, by the name a spec asks for them by.

Scans ``robosimstudio.domains`` so the library's own -- ``static``, ``arena``,
``rigid``, ``cloth`` -- are there for a spec that names one, and is filled the
rest of the way by suites as they import: a task registering ``EntityTape`` does
so from its own module, which discovery has already executed.

The scan is what let the compiler stop keeping its own table. It held three
dotted paths, one per build stage, plus a fourth reached by sniffing the *shape*
of a spec -- if it was scenery and stripped its free bodies and collapsed its
fixed joints, it got ``EntityArena``. That branch selected nothing across all ten
shipped tasks, so the compiler carried a heuristic for a class nobody could
reach. A spec names what builds it now, always, and there is one rule instead of
two."""


def register_cell(spec: CellSpec) -> CellSpec:
    """Register a cell. Call at module scope in the cell's own module."""
    return CELLS.register(spec.name, spec, origin=getattr(spec, "__module__", ""))


ANY_ENGINE = "*"
"""The engine key of an implementation that serves every engine.

Spelled rather than left as ``""`` because it is the *answer* to a question --
which engine is this for -- and "all of them" is a real answer that reads badly
as an empty string in a registry listing."""


def entity_key(name: str, engine: str = ANY_ENGINE) -> str:
    """The registry key for one implementation of one kind.

    A **string** rather than a tuple, and the engine-neutral case keys on the
    bare name, so every registration and every error message that existed before
    the engine axis is byte-identical to what it was. A keyed one reads
    ``"strand@superdex"``, which is what a lookup miss should print.
    """
    return name if engine in ("", ANY_ENGINE) else f"{name}@{engine}"


def register_entity(name: str, *, engine: str = ANY_ENGINE) -> Callable[[type], type]:
    """Class decorator registering an entity implementation under ``name``.

    A suite that introduces a new *kind* of thing -- an adhesive tape, an
    articulated container -- registers its entity class here and names it from
    its specs, which is how a task contributes physics without core or the
    compiler learning about it.

    **A kind may have an implementation per engine.** ``engine`` is a backend
    alias (:attr:`~robosimstudio.specs.arrangement.BackendSpec.solver`), and the
    default serves every engine::

        register_entity("strand")(EntityStrand)                            # any engine
        register_entity("strand", engine="superdex")(EntityStrandSuperdex) # this one only

    What a kind *is* and how it is *realised* are different questions, and
    fusing them is what limited a cloth to one discretisation and one material
    vocabulary however many engines could simulate it. The compiler resolves the
    pair at compile time, from the arrangement that already says which backend
    serves the domain -- so choosing an implementation stays a statement about
    physics rather than a second thing to keep in step with it.

    Registering only the neutral form is the ordinary case and stays the
    ordinary case: an implementation that authors Newton state and lets a
    foreign engine integrate it is engine-neutral by construction, which is
    every entity in this tree today.

    Args:
        name: What a spec's ``entity`` field asks for.
        engine: A backend alias, or :data:`ANY_ENGINE` for every engine.
    """

    def decorate(cls: type) -> type:
        # Imported here rather than at module scope, which is this module's
        # whole arrangement: entity classes arrive by registering themselves
        # while being imported for some other reason, and a registry that
        # imported them would invert that.
        from robosimstudio.scene.entity import Entity

        if not (isinstance(cls, type) and issubclass(cls, Entity)):
            # Checked because the alternative is finding out during a scene
            # build, as a missing attribute on an object nobody expected to be
            # here. The way this happens is a helper class landing between the
            # decorator and the class it was written for, which registers the
            # helper and quietly leaves the entity unregistered -- and neither
            # half of that says anything until a scene is compiled.
            raise TypeError(
                f"register_entity({name!r}) was given {getattr(cls, '__qualname__', cls)!r}, which is "
                "not an Entity subclass. The compiler calls Entity's surface on whatever it finds "
                "here. If a helper class sits directly beneath this decorator, it was registered "
                "instead of the entity below it."
            )
        ENTITIES.register(entity_key(name, engine), cls, origin=cls.__module__)
        return cls

    return decorate


def get_cell(name: str) -> CellSpec:
    return CELLS.get(name)


def get_entity(name: str, *, engine: str = ANY_ENGINE) -> type:
    """The class realising ``name``, for ``engine`` when one is registered for it.

    Falls back to the engine-neutral implementation, which is what keeps every
    kind registered before the engine axis existed working under every engine.
    A miss reports the *kind*, never the pair: "there is no cloth" is the useful
    message, and "there is no cloth@genesis" would send a reader looking for the
    wrong thing when the neutral one is missing too.
    """
    keyed = entity_key(name, engine)
    if keyed != name and keyed in ENTITIES:
        return ENTITIES.get(keyed)
    return ENTITIES.get(name)


def resolve_entity_class(spec: Any, arrangement: Any = None) -> type:
    """The class that should build ``spec`` under ``arrangement``.

    Two questions, resolved in order, because the second needs the first's
    answer: *what kind of thing is this* (the spec's ``entity`` name, and the
    domain that name's default implementation declares), then *which engine will
    integrate it* (the arrangement's backend for that domain).

    Three rules, each of which was a way to get this wrong:

    * **The domain comes from the spec's declaration or from the class, never
      from ``or``.** :attr:`SimRequirements.domain` defaults to ``"rigid"`` and
      is therefore never falsy, so ``spec.sim.domain or cls.domain`` would read
      the class never -- and a spec stating ``sim=SimRequirements(contacts=...)``
      without a domain would silently resolve as rigid. This is the same rule
      :meth:`~robosimstudio.scene.entity.Entity.sim_requirements` applies, and
      it is applied here rather than restated.
    * **A missing backend is not an error here.** ``backend_for`` returns
      ``None`` when the arrangement serves neither the domain nor ``"*"``, and
      the message worth reading is
      :func:`~robosimstudio.scene.physics.islands.partition`'s, which names the
      domain and what the arrangement does serve. Raising at build time would
      pre-empt it with an ``AttributeError``.
    * **The engine key is the solver string verbatim**, not a normalised alias.
      An island's *name* reduces ``mypkg.SolverGenesis`` to ``"genesis"``, which
      is right for a label and wrong for a lookup: a third party's solver class
      that happened to end in ``SolverGenesis`` would select the Genesis cloth.
      So an implementation registers against the spelling the arrangement uses.

    A world-fixed domain is skipped outright: nothing integrates a static, so
    the arrangement never resolves a backend for one and the ``"*"`` fallback
    would key it on whatever happens to serve everything else.

    Args:
        spec: The build spec. Needs an ``entity`` name; ``sim`` is read when present.
        arrangement: The staged :class:`~robosimstudio.specs.ArrangementSpec`,
            or ``None`` to take the engine-neutral implementation.

    Returns:
        The entity class. Never ``None`` -- an unregistered name raises, naming
        the kind.
    """
    from robosimstudio.specs.physics import DOMAINS, SimRequirements, get_domain

    name = getattr(spec, "entity", "")
    if not name:
        raise ValueError(
            f"{getattr(spec, 'name', spec)!r} names no entity class. Every spec says what builds it: "
            "the generic kinds default the field (`rigid`, `cloth`, `static`) and anything else states one."
        )
    cls = ENTITIES.get(name)
    if arrangement is None:
        return cls

    declared = getattr(spec, "sim", None)
    domain = declared.domain if isinstance(declared, SimRequirements) else getattr(cls, "domain", "")
    if not domain or (domain in DOMAINS and get_domain(domain).world_fixed):
        return cls

    chosen = (declared.backend if isinstance(declared, SimRequirements) else None) or arrangement.backend_for(domain)
    if chosen is None:
        return cls
    return get_entity(name, engine=chosen.solver)


def entity_engines(name: str) -> list[str]:
    """Which engines ``name`` has an implementation for, neutral one included.

    What a reader -- or the studio's physics panel -- asks to find out whether a
    kind can be switched at all.
    """
    out = [ANY_ENGINE] if name in ENTITIES else []
    prefix = f"{name}@"
    out.extend(key[len(prefix) :] for key in ENTITIES.names() if key.startswith(prefix))
    return out


def list_cells() -> list[str]:
    """Every registered cell name, sorted."""
    return CELLS.names()
