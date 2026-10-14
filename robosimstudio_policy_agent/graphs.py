# SPDX-License-Identifier: Apache-2.0
"""What a graph directory is, and where the scripts it names come from.

A GaP graph is a directory: ``workflow.json`` beside a ``scripts/`` directory,
and every ``type: script`` (or ``router``) node names its file relative to that
directory -- ``"script": "scripts/next_goal.py"``. The runtime
(``gap.runtime.nodes``) joins the two and imports the file, and that is the
whole of its resolution: a graph is readable top to bottom because everything it
runs is beside it.

**The rule this adds has four places, in order.** A script the workflow names
is looked for beside the graph first, joined exactly as the runtime joins it.
One that is *not there* is looked for in the bundle the enclosing subgraph's
``skill`` names -- ``<bundle>/scripts/<name>`` in open-robot-skills, the
registry this repository consumes and never ships (see
:mod:`robosimstudio_policy_agent.skills`); a label that names a bundle runs that
bundle's scripts, which is what makes the label honest. Then in the
collection's ``_shared/scripts/`` -- ``<root>/_shared/scripts/<name>`` for a
graph under ``<root>/<suite>/graphs/``, the rules more than one of that
collection's graphs run, kept once with the collection
(:func:`collection_shared`). Last, in :data:`NODES` --
``robosimstudio_policy_agent/nodes/`` -- by its file name: the rules graphs of *more than
one collection* run, which is what makes them the library's rather than a
collection's. A name found nowhere is refused before the graph starts, naming
all four places; a ``skill`` that names no bundle in any
resolved registry is refused the same way, naming the registries searched and
how to get one. A subgraph whose skill is ``generic`` -- its own choreography;
``generated`` is the same label as the self-learning harness's older policies
spell it (:data:`~robosimstudio_policy_agent.skills.GENERIC_LABELS`) -- consults no
bundle. Nothing else changes: ``workflow.json`` still says
``scripts/next_goal.py``, a graph that ships its own ``next_goal.py`` keeps
running its own (``tool_hanging/gt`` does), and the runtime knows none of this
-- it is handed a directory in which every script exists, which
:func:`stage_graph` builds when, and only when, a graph needs one.

Same name, different file stays normal and is not a violation --
``verify_grasp.py`` and ``apply_grasp_yaw.py`` differ between ``sharps_disposal``
and ``tool_sort`` on purpose, because what "verify" means is the suite's own
business -- which is exactly why the rule is *beside the graph first*. What a
graph may not do is carry a byte-identical copy of a bundle's script under the
bundle's name: that is drift waiting to happen.

**A bundle comes with its helpers.** A bundle's scripts import their siblings
by directory (``sys.path.insert(0, dirname(abspath(__file__)))``, then ``import
tipcommon``), and so may a graph's own script beside them. So when a graph
takes any script from a bundle, every ``*.py`` in that bundle's ``scripts/``
the workflow does not name is linked in beside it too. Two sources claiming one
staged path with different bytes is refused, not resolved: a graph carrying its
own unnamed ``tipcommon.py`` next to a bundle that also ships one would run the
bundle's scripts against the wrong helper.

The scripts under ``nodes/`` and under a collection's ``_shared/scripts/`` are
workflow content, not library code: they run inside the gap runtime with a
``ctx`` this package never constructs, and ruff is told to leave them alone for
the same reason it leaves ``*/tasks/*/graphs/**``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .skills import GENERIC_LABELS, BundleIndex, bundle_index, registry_hint

__all__ = [
    "COLLECTION_SHARED",
    "NODES",
    "check_graph",
    "collection_shared",
    "outside_links",
    "resolve_script",
    "script_references",
    "shared_nodes",
    "stage_graph",
    "subgraph_skills",
]

NODES = Path(__file__).resolve().parent / "nodes"
"""Where the shared node scripts are: one file per rule, resolved by file name."""


def shared_nodes() -> list[str]:
    """The file names :data:`NODES` offers."""
    return sorted(p.name for p in NODES.glob("*.py"))


COLLECTION_SHARED = Path("_shared") / "scripts"
"""Where a task root keeps the scripts more than one of its graphs run: ``<root>/_shared/scripts/``.

``_shared`` is the one name under a root that is not a task (``roots.is_task_dir``
skips it) and the one directory a task may reach across,
so a rule two suites of one collection share lives there, with the collection,
and travels with it.
"""


def collection_shared(graph_dir: str | Path) -> Path | None:
    """The ``_shared/scripts`` of the root a graph directory sits on, or ``None`` off a root.

    A shipped graph is ``<root>/<suite>/graphs/<label>``, so the root is three
    levels up; a scratch graph in a temporary directory has no root, and the
    directory that would be its ``_shared`` does not exist.
    """
    graph_dir = Path(graph_dir).resolve()
    if len(graph_dir.parents) < 3:
        return None
    candidate = graph_dir.parents[2] / COLLECTION_SHARED
    return candidate if candidate.is_dir() else None


def subgraph_skills(workflow: dict[str, Any]) -> dict[str, str]:
    """Each subgraph's declared ``skill`` (``agent`` is the alias gap accepts), ``""`` when absent."""
    out: dict[str, str] = {}
    for name, sg in (workflow.get("subgraphs") or {}).items():
        if not isinstance(sg, dict):
            continue
        skill = sg.get("skill")
        if skill is None:
            skill = sg.get("agent", "")
        out[str(name)] = str(skill or "")
    return out


def script_references(workflow: dict[str, Any]) -> list[tuple[str, str]]:
    """Every ``(script, skill)`` a workflow names -- top level and inside each subgraph.

    Any node carrying a ``script`` field, which is ``type: script`` and
    ``type: router`` in the runtime's schema. A top-level node carries no
    skill (``""``); a subgraph's nodes carry the subgraph's. Document order,
    each pair once -- one script under two skills is two pairs, because it is
    resolved twice.
    """
    seen: dict[tuple[str, str], None] = {}
    skills = subgraph_skills(workflow)
    scopes: list[tuple[Any, str]] = [(workflow.get("nodes") or {}, "")]
    for name, sg in (workflow.get("subgraphs") or {}).items():
        if isinstance(sg, dict):
            scopes.append((sg.get("nodes") or {}, skills.get(str(name), "")))
    for nodes, skill in scopes:
        entries = nodes.values() if isinstance(nodes, dict) else nodes
        for node in entries:
            if isinstance(node, dict) and node.get("script"):
                seen.setdefault((str(node["script"]), skill))
    return list(seen)


def resolve_script(
    graph_dir: str | Path,
    script: str,
    *,
    skill: str = "",
    registry: BundleIndex | None = None,
) -> Path:
    """The file a workflow's ``script`` names: beside the graph, else the bundle, else the collection's ``_shared``, else a shared node.

    Beside the graph first, joined as ``gap.runtime.nodes`` joins it, so a graph
    shipping a file of the same name as a bundle script or a shared node runs
    its own. Then, when *skill* names a bundle, ``<bundle>/scripts/<name>`` in
    the registries *registry* indexes (resolved from this checkout when
    ``None``). Then the collection's ``_shared/scripts`` (:func:`collection_shared`),
    the rules two suites of one collection share. Then :data:`NODES`, by file
    name. A name in none of the four is
    refused *here*, with every place in the message, rather than left for the
    runtime to hit on the node's first execution -- which it reports as a failed
    graph, and a failed graph scores zero and reads like a policy that tried
    the task and lost.

    Raises:
        LookupError: *skill* is neither generic (either spelling) nor a bundle
            any resolved registry holds. The message says which registries
            were searched and how to get open-robot-skills.
        FileNotFoundError: the script is in none of the places consulted.
    """
    graph_dir = Path(graph_dir)
    local = graph_dir / script
    if local.is_file():
        return local
    name = Path(script).name
    places = [str(graph_dir)]
    if skill and skill not in GENERIC_LABELS:
        index = registry if registry is not None else bundle_index()
        if skill not in index:
            raise LookupError(
                f"{graph_dir}: the workflow names skill {skill!r}, which no resolved registry supplies "
                f"(known: {index.names() or 'none'}). {registry_hint(index.roots)}"
            )
        canonical = index[skill] / "scripts" / name
        if canonical.is_file():
            return canonical
        places.append(str(index[skill] / "scripts"))
    else:
        places.append("no bundle consulted: the subgraph's skill is generic")
    shared_here = collection_shared(graph_dir)
    if shared_here is not None:
        if (shared_here / name).is_file():
            return shared_here / name
        places.append(str(shared_here))
    else:
        places.append(f"no {COLLECTION_SHARED} beside the graph's root")
    shared = NODES / name
    if shared.is_file():
        return shared
    places.append(f"{NODES} (which has {shared_nodes()})")
    raise FileNotFoundError(f"{graph_dir}: the workflow names {script!r}, which is in none of: " + "; ".join(places))


def stage_graph(graph_dir: str | Path, *, registries: Sequence[str | Path] | None = None) -> str:
    """The directory to hand ``gap.execute``.

    ``graph_dir`` itself when every script its workflow names is beside it --
    the common case for a graph of ``generic`` subgraphs, in which nothing here
    has happened. Otherwise a scratch directory holding the graph as the
    runtime needs to see it: every entry of the graph directory linked in, a
    directory that has to receive an outside script rebuilt file by file so the
    script can stand beside the graph's own, each missing script linked from
    its bundle or from :data:`NODES`, and each bundle's unnamed helpers linked
    beside the scripts that import them. Links rather than copies, so
    ``readlink`` says where each file came from and an edit to the original is
    what runs; a copy where the filesystem refuses a link.

    The runtime copies ``workflow.json`` and ``scripts/`` into its trace
    (following links), so the trace of a graph that used shared nodes or bundle
    scripts carries them: the trace is the complete graph, which a trace holding
    a dangling reference would not be.

    Args:
        graph_dir: The graph directory.
        registries: Registry roots to consult, in precedence order; ``None``
            resolves them from this checkout (:func:`~robosimstudio_policy_agent.skills.registry_roots`).
            The index is built only when a script is not beside the graph, so
            a graph of ``generic`` subgraphs never touches a registry.

    Raises:
        LookupError, FileNotFoundError: as :func:`resolve_script`.
        ValueError: one staged path claimed by two different sources.

    A directory with no readable ``workflow.json`` is passed through: what is
    wrong with it is the runtime's to say, in its own words.
    """
    graph_dir = Path(graph_dir)
    try:
        workflow = json.loads((graph_dir / "workflow.json").read_text())
    except (OSError, ValueError):
        return str(graph_dir)

    index: BundleIndex | None = None
    resolved: dict[str, Path] = {}
    for script, skill in script_references(workflow):
        local = graph_dir / script
        if local.is_file():
            resolved.setdefault(script, local)
            continue
        if index is None:
            index = bundle_index(registries)
        path = resolve_script(graph_dir, script, skill=skill, registry=index)
        previous = resolved.get(script)
        if previous is not None and previous != path:
            raise ValueError(
                f"{graph_dir}: {script!r} resolves to {previous} under one subgraph and to {path} under "
                f"another; one staged path cannot carry two files"
            )
        resolved[script] = path

    outside = {script: path for script, path in resolved.items() if path != graph_dir / script}
    if not outside:
        return str(graph_dir)

    # A bundle's helpers: every *.py in the scripts/ of a bundle that supplied
    # a script, that the workflow does not name, staged beside the script that
    # came from there. Shared nodes have no helpers.
    named = {Path(script).name for script in resolved}
    helpers: dict[str, Path] = {}
    for script, path in outside.items():
        if path.parent == NODES:
            continue
        for helper in sorted(path.parent.glob("*.py")):
            if helper.name in named:
                continue
            rel = str(Path(script).parent / helper.name)
            local = graph_dir / rel
            if local.is_file():
                if local.read_bytes() != helper.read_bytes():
                    raise ValueError(
                        f"{graph_dir}: ships {rel!r}, which the workflow does not name, beside a bundle "
                        f"that also supplies it ({helper}) with different bytes; the bundle's scripts would "
                        f"import the wrong one"
                    )
                continue
            other = helpers.get(rel)
            if other is not None and other != helper and other.read_bytes() != helper.read_bytes():
                raise ValueError(f"{graph_dir}: two bundles supply {rel!r} with different bytes ({other} and {helper})")
            helpers.setdefault(rel, helper)

    staged = Path(tempfile.mkdtemp(prefix="robosimstudio-gap-graph-"))
    # Every ancestor of a staged path is a real directory, not a link to the
    # graph's own: ``scripts/sub/x.py`` written through a linked ``scripts``
    # would land in the graph's source tree.
    rebuilt = {parent for rel in (*outside, *helpers) for parent in Path(rel).parents if parent != Path(".")}

    def mirror(source: Path, dest: Path) -> None:
        for entry in source.iterdir():
            if entry.name == "__pycache__":
                continue
            if entry.is_dir() and entry.relative_to(graph_dir) in rebuilt:
                (dest / entry.name).mkdir()
                mirror(entry, dest / entry.name)
            else:
                _link(entry, dest / entry.name)

    mirror(graph_dir, staged)
    for rel, path in (*outside.items(), *helpers.items()):
        dest = staged / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        _link(path, dest)
    return str(staged)


def _link(source: Path, dest: Path) -> None:
    """A link at *dest* to *source*, resolved first: the staged directory is
    under ``tempfile``, so a relative *source* -- a graph named as
    ``my_tasks/push/graphs/pick`` from the checkout root -- linked verbatim would
    be read relative to the scratch directory and dangle."""
    source = Path(source).resolve()
    try:
        os.symlink(source, dest)
    except OSError:
        if source.is_dir():
            shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(source, dest)


def _display(path: Path) -> str:
    """A path relative to this checkout when it is under it, else as given."""
    from .skills import RSS_ROOT

    return str(path.relative_to(RSS_ROOT)) if path.is_relative_to(RSS_ROOT) else str(path)


def outside_links(staged: str | Path, graph_dir: str | Path) -> list[str]:
    """``rel -> source`` for every staged file linked from outside the graph directory."""
    # Resolved, as ``_link`` resolves every target: a relative graph_dir would
    # otherwise report the graph's own files as linked from outside it.
    staged, graph_dir = Path(staged), Path(graph_dir).resolve()
    rows: list[str] = []
    for dirpath, _dirs, files in os.walk(staged, followlinks=True):
        for name in files:
            path = Path(dirpath) / name
            if not path.is_symlink():
                continue
            target = Path(os.readlink(path))
            if target.is_relative_to(graph_dir):
                continue
            rows.append(f"{path.relative_to(staged)} -> {_display(target)}")
    return sorted(rows)


def check_graph(
    graph_dir: str | Path, registries: Sequence[str | Path], skill_registry: object
) -> tuple[list[str], list[str], list[str]]:
    """``(errors, warnings, provenance)`` for one graph, validated as staged.

    A staging refusal or a workflow that does not load is one error: nothing
    downstream of it is a complete answer. The validator is gap's own, run on
    what would execute; ``skill_registry`` is what ``gap.skills.load_registry_set``
    returns for ``registries``, or ``None`` to validate without one.
    """
    from gap.runtime.validate import validate_workflow
    from gap.runtime.workflow import load_workflow

    graph_dir = Path(graph_dir)
    try:
        staged = Path(stage_graph(graph_dir, registries=registries))
    except (LookupError, FileNotFoundError, ValueError) as exc:
        return [f"STAGE FAIL ({type(exc).__name__}): {exc}"], [], []
    try:
        provenance = outside_links(staged, graph_dir) if staged != graph_dir else []
        try:
            workflow = load_workflow(staged / "workflow.json")
        except Exception as exc:
            return [f"LOAD FAIL ({type(exc).__name__}): {exc}"], [], provenance
        try:
            issues = validate_workflow(workflow, agent_registry=None, skill_registry=skill_registry, tool_registry=None)
        except Exception as exc:
            return [f"VALIDATOR CRASHED ({type(exc).__name__}): {exc}"], [], provenance
    finally:
        if staged != graph_dir:
            shutil.rmtree(staged, ignore_errors=True)
    errors = [str(issue) for issue in issues if issue.severity == "error"]
    warnings = [str(issue) for issue in issues if issue.severity != "error"]
    return errors, warnings, provenance
