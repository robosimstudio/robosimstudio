# SPDX-License-Identifier: Apache-2.0
"""robosimstudio: a Newton-native benchmark stack, as ten packages; the benchmark tier is a package beside them.

Read bottom-up; each names only what is below it.

**The mechanisms.**

* :mod:`robosimstudio.core` -- what makes *one hand-built Newton scene*
  steppable, controllable, observable and renderable. No registries, no
  benchmark semantics, no knowledge of tasks, and it names nothing above it --
  where it needs knowledge from higher up it takes explicit data or states a
  protocol, which is why ``EnvBase`` declares an ``ActionSpace`` protocol rather
  than importing one.
* :mod:`robosimstudio.specs` -- the frozen IR every layer speaks, split by what
  a spec describes: an asset, a robot, a scene, a task, a run.

**What a thing is.**

* :mod:`robosimstudio.robots` -- who acts: the robots and hands
  themselves, the action spaces, and the algorithmic parts each in their own
  folder (``ik/``, ``motion/``, ``grasp/``).
* :mod:`robosimstudio.scene` -- specs and an episode to a steppable, indexed
  scene: the compiler's fixed order, the entities, the handles.
* :mod:`robosimstudio.assetlib` -- what an asset *is*: its record, its
  substrates and feature kinds, the selector language, the atom registry, the
  handlers and the materials. It knows how an asset is described and nothing
  about any particular asset -- and nothing about any document.
* :mod:`robosimstudio.sim` -- the front door: ``Scene``, ``Sim``, ``Ref``.
  Nothing here knows what a task is.
* :mod:`robosimstudio.record` -- what a run recorded. The replay bundle, which
  replays a run outside this checkout, is not a layer of the library but a
  top-level package of its own, which ``Env`` reaches lazily and an install may
  lack.

**Beside the library.**

* The benchmark tier -- everything that needs a *document* to exist, a
  top-level package of its own rather than a layer of this one, and not in
  every release. ``taskdoc`` is the ``.task`` language (read, typecheck, write;
  it names nothing but the IR, so an authoring agent can lint a task with no
  scene, no robot and no asset registry). ``taskbuild`` is which tasks exist and
  how a document becomes one -- a sibling of the asset library, not a layer over
  it. ``goals`` is what a benchmark asks *of* an asset: binding, goal trees,
  evaluation, the gauntlet, the catalogue. ``envs`` is the spec-driven ``Env``
  and its fleet twin; ``policies`` everything that drives one; ``run`` running
  many. The core tier never imports it -- only ``studio`` does, and ``assetlib``
  reaches its discovery inside a function.
* :mod:`robosimstudio_studio` -- the browser page over all of it, a bare scene
  included.

**The content is not in the package at all.** A task is a *directory* -- its
documents, the assets they name, the cell they stand in, the arrangement they
stage and the solutions that answer them, all in one place -- and the registries
find them by scanning the **paths** :mod:`robosimstudio.roots` declares, which is
a lookup rather than a dependency. This used to name a content *package* of four
subpackages, which worked for exactly one body of content at a time and required
it to be installable; a root is a path, so somebody else's tasks run beside these
with no packaging step. A suite's
solutions -- its routine, its tools, its shipped graphs -- sit beside the
documents that reference them, because a routine that folds this shirt is no
more general than this shirt.

**This module imports nothing above ``_version``, and that is load-bearing.**
Importing :mod:`robosimstudio.core` executes this file first, so a module-scope
``from .sim import Scene`` here would make core un-importable without ``sim`` and
quietly invert the whole partition, which nothing would notice until somebody
imported a lower package without the layers above it.

The flat names below are resolved through :pep:`562` ``__getattr__`` instead,
which fires on *attribute access* and never during ``importlib.import_module``.
So ``import robosimstudio`` still costs nothing and still names nothing, and
``rss.Scene`` imports :mod:`robosimstudio.sim` at the moment you ask for it::

    import robosimstudio as rss

    scene = rss.Scene(cell="bench_075")
    cube = scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0, 0, 0.2)))
    sim = scene.build()
    sim.step(150)
    print(cube.position)

The flat surface is **core-tier only** -- assembling a scene, stepping it,
reading it, and registering a new kind of thing. There is no ``rss.Env`` and no
``rss.get_task``: a benchmark is a layer over this, and reaching it is an import
from the benchmark tier's own package, which says which layer you are in.
"""

from ._version import __version__

__all__ = ["__version__"]

_EXPORTS: dict[str, str] = {
    # Assembling and stepping.
    "Scene": "robosimstudio.sim",
    "Sim": "robosimstudio.sim",
    "Ref": "robosimstudio.sim",
    "Robot": "robosimstudio.sim",
    "robot": "robosimstudio.sim",
    "draw": "robosimstudio.sim",
    "view": "robosimstudio.sim",
    "viewer": "robosimstudio.sim",
    "Viewer": "robosimstudio.sim",
    "compile_scene": "robosimstudio.scene",
    "load_asset": "robosimstudio.scene",
    # Writing a new kind of thing.
    "Entity": "robosimstudio.scene.entity",
    "register_entity": "robosimstudio.scene",
    "register_cell": "robosimstudio.scene",
    "register_domain": "robosimstudio.specs.registries",
    "register_backend": "robosimstudio.specs.registries",
    "register_coupler": "robosimstudio.specs.registries",
    "get_cell": "robosimstudio.scene",
    "list_cells": "robosimstudio.scene",
    # Where it runs: the device, and the renderer that draws it.
    "platform": "robosimstudio.machine",
    "Platform": "robosimstudio.machine",
    # Saying how it is simulated.
    "GENERIC": "robosimstudio.specs",
    "MUJOCO_500HZ": "robosimstudio.specs",
    "CPU": "robosimstudio.specs",
    "ArrangementSpec": "robosimstudio.specs",
    "CollisionSpec": "robosimstudio.specs",
    "SimRequirements": "robosimstudio.specs",
    "backend": "robosimstudio.specs",
    "coupler": "robosimstudio.specs",
    "contacts_with": "robosimstudio.specs",
    "replace": "robosimstudio.specs",
    # Saying what it is.
    "ActuationSpec": "robosimstudio.specs",
    "ArmSpec": "robosimstudio.specs",
    "ArticulatedSpec": "robosimstudio.specs",
    "AssetSpec": "robosimstudio.specs",
    "CameraSpec": "robosimstudio.specs",
    "CellSpec": "robosimstudio.specs",
    "EmbodimentSpec": "robosimstudio.specs",
    "JointSpec": "robosimstudio.specs",
    "ObjectSpec": "robosimstudio.specs",
    "RegionSpec": "robosimstudio.specs",
    "ShapeSpec": "robosimstudio.specs.asset",
    "StaticSpec": "robosimstudio.specs",
    # Constructors, each beside the domain it describes.
    "articulated": "robosimstudio.domains.articulated",
    "body": "robosimstudio.domains.rigid",
    "box": "robosimstudio.domains.rigid",
    "capsule": "robosimstudio.domains.rigid",
    "cylinder": "robosimstudio.domains.rigid",
    "sphere": "robosimstudio.domains.rigid",
    "post": "robosimstudio.domains.static",
    "slab": "robosimstudio.domains.static",
}
"""Flat name -> the module that defines it. A table rather than a wildcard.

Wildcards were the alternative and they are worse in the two ways that matter
here: the set of flat names becomes whatever four packages happen to export
today, and a typo resolves to something plausible from the wrong layer instead of
raising. This is also the list a reader scans to answer "what is the core tier",
which is a question the tier split exists to have an answer to. The deformable
kinds' names -- ``sheet``, ``sponge``, ``rod``, ``heap``, ``liquid``, ``dough``
and their specs -- are not here but in :data:`_PACKS`: they come from
``robosimstudio_materials``, which a tree may leave out.

There are no aliases: a physics arrangement is ``rss.ArrangementSpec(control_hz=60,
substeps=16)``, under the one name its module gives it (see :data:`_ALIASES`).

``replace`` is ``dataclasses.replace``, re-exported for one reason: amending a
frozen spec is the *normal* way to use this tier -- ``replace(GENERIC,
substeps=4)`` appears in three docstrings and every example that stages its own
physics -- and a reader who has to work out that the name comes from the standard
library before the snippet runs is a reader the snippet failed.
"""

_PACKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "robosimstudio_materials.cloth": (
        "cloth (the `cloth` kind: sheets, towels, garments)",
        ("ClothMaterialSpec", "ClothSpec", "materials", "sheet"),
    ),
    "robosimstudio_materials.soft": (
        "soft bodies (the `soft` kind: sponges, tetrahedral meshes)",
        ("SoftMaterialSpec", "SoftSpec", "sponge"),
    ),
    "robosimstudio_materials.rod": (
        "the rope (the `rod` kind: cables, ropes, cords)",
        ("RodMaterialSpec", "RodSpec", "rod"),
    ),
    "robosimstudio_materials.granular": (
        "grains (the `granular` kind: rice, sand, beans, flour on the implicit MPM)",
        ("GRANULAR_60HZ", "ParticleBodySpec", "ParticleMaterialSpec", "heap"),
    ),
    "robosimstudio_materials.liquid": (
        "liquids (the `liquid` kind: water, honey, ketchup on the implicit MPM)",
        ("LIQUID_120HZ", "liquid"),
    ),
    "robosimstudio_materials.dough": (
        "plastic solids (the `dough` kind: dough, plasticine, clay, butter on the implicit MPM)",
        ("DOUGH_60HZ", "dough", "roller"),
    ),
    "robosimstudio_materials.paper": (
        "paper (the `paper` kind: sheets and card whose folds stay)",
        ("PaperMaterialSpec", "PaperSpec", "paper", "papers"),
    ),
    "robosimstudio_materials.bag": (
        "bags (the `bag` kind: two layers of cloth sewn into a pocket, on SuperDex)",
        ("BAG_60HZ", "BagSpec", "bag", "bags"),
    ),
    "robosimstudio_materials.brush": (
        "brushes (the `brush` kind: a handle with a field of short capsule-chain bristles)",
        ("BRUSH_60HZ", "BrushMaterialSpec", "BrushSpec", "brush"),
    ),
    "robosimstudio_materials.tape": (
        "tape and labels (the `tape` kind: a sheet stuck to a body by bonds that let go, with a pull tab)",
        ("TAPE_60HZ", "TapeMaterialSpec", "TapeSpec", "tape", "tapes"),
    ),
    "robosimstudio_materials.cutting": (
        "cutting (the `cutting` kind: a block a knife parts, on the walk-on-spheres cage cutter)",
        ("CUTTING_50HZ", "CuttableMaterialSpec", "CuttableSpec", "KnifeSpec", "cuttable", "cuttables", "knife"),
    ),
    "robosimstudio_materials.water": (
        "water (the `water` kind: a liquid that keeps its volume, on Genesis's SPH)",
        ("WATER_60HZ", "WaterMaterialSpec", "WaterSpec", "water"),
    ),
}
"""Kind -> what it is, in words, and the flat names it provides.

The kinds of thing in **a package a tree may leave out**: each is one module of
``robosimstudio_materials``, the sibling package the library never imports. Where
it is installed these names resolve like any in :data:`_EXPORTS`; where it is
not, they are not part of the flat namespace at all -- ``hasattr(rss, "rod")``
is false, ``dir(rss)`` does not list it, and asking for one raises an
``AttributeError`` that says the kind is not installed rather than a
``ModuleNotFoundError`` from three layers down.

Kept apart from :data:`_EXPORTS` for that reason: that table is the names every
tree has, and every test that walks it expects each to resolve. The words are
here, in the library, because a tree without the package is exactly the one
that has to be able to say what is missing.
"""

_PACK_EXPORTS: dict[str, str] = {name: pack for pack, (_, names) in _PACKS.items() for name in names}
"""Flat name -> the ``robosimstudio_materials`` module that provides it, when that package is installed."""

_ALIASES: dict[str, str] = {}
"""Flat name -> the name it resolves to in its module, where the two differ.

Empty. It held ``Arrangement -> ArrangementSpec``: a second spelling of one
concept, written in no example, no task, no tool, no test and no document. Two
names for one thing is the cheapest kind of concept to remove and the most
expensive to leave, because a reader who meets both has to find out they are the
same.
"""


def __getattr__(name: str):
    """Resolve one flat name, importing its module on first access (:pep:`562`)."""
    import importlib

    module = _EXPORTS.get(name)
    if module is None:
        pack = _PACK_EXPORTS.get(name)
        if pack is None:
            # Built from the table alone, and that is load-bearing: the import system asks for
            # ``robosimstudio.<submodule>`` as an *attribute* before it imports one (``from
            # robosimstudio import roots``), so this branch runs on ordinary imports and must
            # not touch the import system itself -- asking which kinds are installed here
            # would probe ``robosimstudio_materials`` from inside every such import.
            raise AttributeError(
                f"module 'robosimstudio' has no attribute {name!r}. The flat namespace is the core tier: "
                f"{sorted(_EXPORTS)}. A benchmark name (Env, get_task) is one layer up: "
                "import it from the benchmark tier's own package, where that tier is installed."
            )
        try:
            value = getattr(importlib.import_module(pack), name)
        except ModuleNotFoundError as error:
            if not _is_absent(pack, error):
                raise  # the pack is there and something *it* needs is not: that is a real error
            raise AttributeError(
                f"module 'robosimstudio' has no attribute {name!r}: {_PACKS[pack][0]} is not installed here. "
                f"Its module ({pack}) is not importable: the robosimstudio_materials package is not in this tree, "
                f"or ships without that kind. So neither is rss.{name}."
            ) from None
    else:
        value = getattr(importlib.import_module(module), _ALIASES.get(name, name))
    globals()[name] = value  # so the second access is a plain lookup
    return value


def _is_absent(pack: str, error: ModuleNotFoundError) -> bool:
    """Whether ``error`` says the kind's module (or its package) is missing, rather than something it imports."""
    missing = error.name or ""
    return missing == pack or pack.startswith(f"{missing}.")


def _installed(pack: str) -> bool:
    """Whether a kind's module is importable here. Asked of the import system, without importing it."""
    import importlib.util

    try:
        return importlib.util.find_spec(pack) is not None
    except (ImportError, ValueError):
        return False


def _flat_names() -> list[str]:
    """The flat namespace as it stands in this checkout: the core names, and each installed kind's."""
    packs = [name for name, pack in _PACK_EXPORTS.items() if _installed(pack)]
    return sorted([*_EXPORTS, *packs])


def __dir__() -> list[str]:
    return sorted([*_flat_names(), "__version__"])
