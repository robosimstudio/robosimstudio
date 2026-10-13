# SPDX-License-Identifier: Apache-2.0
"""An asset: everything a thing knows about itself, in one declaration.

This is the record the whole component exists to hold. It carries:

* the **substrate** -- what kind of state the engine integrates for it, the only
  type it must have;
* **features** -- what it offers, per object, three-tiered, with kinds where the
  kinds recur;
* **states** -- named thresholds a document and a structured query can use;
* **published states** -- internal quantities it exposes, typed, with an
  observability tier;
* **sim requirements** -- what it needs of physics to behave like itself;
* **tags** -- perception labels, for grounding language, never for dispatch;
* **files** and **provenance** -- where it came from.

**Note what is not here: behaviour.** No methods for grasping, containing or
opening. What a record declares *is* the asset's interface, in the sense
:mod:`~robosimstudio.assetlib.semantics` states and this layer uses throughout:
the Python one -- duck-typed, per object, freely defined, discovered rather than
inherited. What does not exist is an interface *roster*. Nothing declares itself
``Graspable``; an object affords grasping because it *has* a grasp box, and the
free handler functions dispatch on what it has. A mug and a pair of scissors both
hang on a hanger because both annotate a ``loop``, not because either inherits
anything. Categories never generalise; geometry does.

**So every asset's interface is its own, and no two need resemble each other.**
Beyond the substrate, two records may share no field at all: the towel offers
three particle sets and a fold template, the receptacle offers a bore with an
axis and a depth, the tape offers ten published quantities and a bond field, and
none of those is a specialisation of the others. Anything reading a record --
a query, a checker, an inspector -- therefore has to take each declaration
on its own terms rather than against a shape it assumed: *this* asset's features,
parsed against the kinds *it* names, with the payload each of those kinds
declares. A reader that flattens them into one table is reporting the roster
this design does not have.

Records are frozen and JSON-able, and deliberately **not** hash keys: their
payloads are nested JSON, and deep-freezing them to buy a hash nothing needs
would be complexity for its own sake. The compiler keys on the task, as ever.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .semantics.features import FeatureSpec, feature_to_dict
from .semantics.kinds import get_kind, is_registered_kind
from .semantics.published import PublishedState, published_to_dict
from .semantics.states import StateSpec, state_to_dict
from .semantics.substrates import is_substrate, reads_as
from .sim import SimRequirements, sim_to_dict

__all__ = ["AssetRecord", "asset_to_dict"]


@dataclass(frozen=True, kw_only=True)
class AssetRecord:
    """One thing the simulator can contain, and what is known about it.

    Attributes:
        name: Registry key, and how a task document refers to it.
        substrate: One of the engine's state layouts.
        summary: What this is, in a sentence a person or a model can read.
        tags: Perception labels (``"scissors"``, ``"bowl"``). Ground language and
            connect to detectors; **never** dispatched on. Physics grounds types,
            perception grounds tags, and conflating the two is how an ontology
            starts asserting things no instrument can measure.
        features: What it offers.
        states: Named thresholds over its measures.
        published: Internal quantities it exposes, typed and tiered.
        sim: What it needs of physics.
        home: The directory this record was declared in, stamped at registration.

            An asset ships beside the declaration that owns it, so its geometry
            is found relative to *that* directory before any global root is
            searched. This is what lets a task directory hold its own meshes and
            be moved without editing a path -- and it is stamped rather than
            declared, because a record cannot know where somebody will put it.
        files: Named source files -- mesh, MJCF, USD -- resolved against the
            asset root by the toolkit, so this holds references and not paths.
        mesh_sha: Hash of the rest layout its discretisation-referencing features
            were authored against.
        provenance: Where it came from: which export, which vendor tool, which
            reference implementation, measured by whom.
    """

    name: str
    summary: str = ""
    tags: tuple[str, ...] = ()
    features: tuple[FeatureSpec, ...] = ()
    states: tuple[StateSpec, ...] = ()
    published: tuple[PublishedState, ...] = ()
    sim: SimRequirements | None = None
    files: dict[str, str] = field(default_factory=dict)
    home: Path | None = None
    mesh_sha: str = ""
    provenance: str = ""
    build: Callable[[], Any] | None = field(default=None, compare=False, repr=False)
    """How to put one in a scene: a factory returning the placed spec, or ``None``.

    The half of a declaration the document layer could not reach. Every suite
    already had this function -- ``drape_towel()``, ``press_sponge()`` -- and
    only its ``task.py`` could call it, which is why no cloth, rod or soft task
    could be a document alone. Named here, the generic loader asks the record
    and a ``(:objects towel - drape_towel)`` lowers to the ``ClothSpec`` the
    factory returns, under the document's own name.

    Excluded from equality and from the JSON form: it is construction code, not
    a fact about the asset, and two copies of one declaration are still one
    declaration whichever module's function each carries.
    """

    # ----------------------------------------------------------- derived
    @property
    def substrate(self) -> str:
        """Which state layout a measure sees for this asset.

        **Derived, never authored.** It used to be a field, and every asset
        stated it beside ``sim.domain`` -- two words for what turned out to be
        one decision plus a total mapping. Checked at the point it was removed:
        all 66 shipped assets derived exactly what they had declared by hand.

        The two are still different questions, which is why this is
        :attr:`~robosimstudio.specs.physics.DomainInfo.reads` and not identity:
        a welded peg integrates nothing (``domain="static"``) and still has a
        pose to measure against (``reads=("rigid",)``). Every fixture in the
        corpus is that pair.

        An unregistered domain reads as ``""``, which :meth:`problems` reports --
        a property that raised would make the validator unable to say what was
        wrong.
        """
        from robosimstudio.specs.physics import DOMAINS

        # ``Registry.get`` raises on an unknown name where a plain dict returned
        # ``None``; the membership test keeps this a *reading* rather than an
        # exception, so :meth:`problems` can still say which domain is wrong.
        # ``reads`` is a fallback *chain*; the head is the layout this asset
        # actually is. The tail is what a measure may settle for when nothing
        # implements it for the head, which is dispatch's business and not a
        # fact about the asset -- a sponge is a volume whether or not anybody
        # has written a volume handler yet.
        name = self.sim.domain if self.sim is not None else "rigid"
        return DOMAINS.get(name).reads[0] if name in DOMAINS else ""

    # -------------------------------------------------------------- lookup
    def feature(self, name: str) -> FeatureSpec:
        """One feature by name.

        Raises:
            KeyError: Naming what this asset *does* offer. A missing feature is
                the most common authoring outcome there is -- a goal asked for
                something the object does not afford -- so the message is written
                for whoever wrote the goal, agent or human.
        """
        for spec in self.features:
            if spec.name == name:
                return spec
        raise KeyError(f"asset {self.name!r} has no feature {name!r}; it offers {[f.name for f in self.features]}")

    def features_of_kind(self, kind: str) -> tuple[FeatureSpec, ...]:
        """Every feature of one kind -- what a handler's dispatch asks for."""
        return tuple(spec for spec in self.features if spec.kind == kind)

    def state(self, name: str) -> StateSpec:
        """One named state."""
        for spec in self.states:
            if spec.name == name:
                return spec
        raise KeyError(f"asset {self.name!r} has no state {name!r}; it declares {[s.name for s in self.states]}")

    @property
    def kinds(self) -> tuple[str, ...]:
        """Every feature kind this asset carries, sorted and deduplicated.

        The catalog's answer to "what can this thing take part in", and the thing
        an agent reads when deciding whether its instruction is expressible
        against the assets that exist.
        """
        return tuple(sorted({spec.kind for spec in self.features}))

    # ---------------------------------------------------------- validation
    def problems(self) -> list[str]:
        """Everything structurally wrong with this asset, or an empty list.

        Structural only -- no model, no GPU, no scene. An asset declaration is
        checked the moment it is written, which is what lets the asset library be
        a per-commit gate rather than something only a built scene can verify.
        """
        out: list[str] = []
        where = f"asset {self.name!r}"
        if not self.name:
            out.append("asset has no name")
        domain = self.sim.domain if self.sim is not None else "rigid"
        if not is_substrate(self.substrate):
            # The substrate is derived, so the thing that is actually wrong is
            # the domain -- report that, not the empty string it produced.
            out.append(f"{where}: unknown physics domain {domain!r}")
        if not self.summary:
            out.append(f"{where}: no summary, so the catalog cannot describe it")

        seen: set[str] = set()
        for spec in self.features:
            if spec.name in seen:
                out.append(f"{where}: two features named {spec.name!r}")
            seen.add(spec.name)
            # A feature may pin its own hash; failing that it inherits the
            # asset's, which is the common case since one mesh backs them all.
            inherited = spec if spec.mesh_sha or not self.mesh_sha else _with_sha(spec, self.mesh_sha)
            out += inherited.problems()
            # A kind may say which substrates can carry it, and until now nothing
            # checked. Writing the first real assets produced a ``point``
            # (arc-length along a chain) annotated on a rigid hook, which is
            # meaningless and would have resolved to nothing at scene-ready --
            # the quietest possible failure. The kind knew; nobody asked it.
            if is_registered_kind(spec.kind):
                # Chain membership, not equality: a kind that belongs on a
                # cloth belongs on anything that says it reads as one.
                allowed = get_kind(spec.kind).substrates
                if allowed and not any(reads_as(self.substrate, entry) for entry in allowed):
                    out.append(
                        f"{where}: feature {spec.name!r} is a {spec.kind!r}, which belongs on "
                        f"{list(allowed)} -- this asset is {self.substrate!r}"
                    )

        state_names: set[str] = set()
        for state in self.states:
            if state.name in state_names:
                out.append(f"{where}: two states named {state.name!r}")
            state_names.add(state.name)
            out += state.problems()
            if state.of and state.of not in seen:
                out.append(f"{where}: state {state.name!r} is about feature {state.of!r}, which it has no feature for")

        published_names: set[str] = set()
        for member in self.published:
            if member.name in published_names:
                out.append(f"{where}: two published states named {member.name!r}")
            published_names.add(member.name)
            out += member.problems()

        if self.sim is not None:
            out += [f"{where}: {problem}" for problem in self.sim.problems()]
        return out


def _with_sha(spec: FeatureSpec, mesh_sha: str) -> FeatureSpec:
    """A feature carrying the asset's mesh hash, for validation only."""
    from dataclasses import replace

    return replace(spec, mesh_sha=mesh_sha)


def _portable_path(record: AssetRecord, path: str) -> str:
    """``path`` relative to the record's home when it lies under it, else unchanged."""
    if record.home is None or not Path(path).is_absolute():
        return path
    try:
        return Path(path).relative_to(Path(record.home)).as_posix()
    except ValueError:
        return path


def asset_to_dict(record: AssetRecord) -> dict[str, Any]:
    """An asset as plain JSON-able data -- the catalog form an agent reads.

    ``files`` are written as declared, not as resolved: registration rewrites a
    file shipped beside its declaration to an absolute path so it loads from
    anywhere, and that path names the machine it was built on (a home
    directory, in a catalog that is pasted into a model's prompt). The catalog
    form is relative to the declaring module's directory instead.
    """
    out: dict[str, Any] = {
        "name": record.name,
        "substrate": record.substrate,
        "summary": record.summary,
        "kinds": list(record.kinds),
    }
    if record.tags:
        out["tags"] = list(record.tags)
    if record.features:
        out["features"] = [feature_to_dict(spec) for spec in record.features]
    if record.states:
        out["states"] = [state_to_dict(spec) for spec in record.states]
    if record.published:
        out["published"] = [published_to_dict(spec) for spec in record.published]
    if record.sim is not None:
        out["sim"] = sim_to_dict(record.sim)
    if record.files:
        out["files"] = {key: _portable_path(record, path) for key, path in record.files.items()}
    if record.mesh_sha:
        out["mesh_sha"] = record.mesh_sha
    if record.provenance:
        out["provenance"] = record.provenance
    return out
