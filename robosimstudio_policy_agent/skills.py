# SPDX-License-Identifier: Apache-2.0
"""Where the skill bundles are, and how a graph's ``skill`` finds one.

**This repository ships no skill bundles.** A subgraph's ``"skill": "<name>"``
names a bundle in open-robot-skills, the registry every shipped graph is
written against, and the runtime resolves that bundle's ``scripts/`` the way
:mod:`robosimstudio_policy_agent.graphs` states: beside the graph first, then the
named bundle, then the collection's ``_shared/scripts/``, then
``robosimstudio_policy_agent/nodes/``. Everything here is about *which*
registries that lookup consults, and the answer has to be the same whoever
asks -- the policy staging a graph, a test on a laptop with no GPU, a
``bench_run`` spawned from some other directory.

**The pin is a fact about the checkout, not about the cwd.** ``gap`` resolves
its registries from the process's working directory: an explicit ``skills=``
first, then ``$GAP_SKILLS_PATH``, then the nearest ``pyproject.toml`` with a
``[tool.gap]`` table walking *up from the cwd*, then user config, then a
side-by-side ``open-robot-skills`` walking up from the gap package and the cwd.
A run started from the wrong directory would resolve nothing and refuse every
graph that names a bundle. So this module reads the ``[tool.gap]`` table from
*this* repository's ``pyproject.toml`` and hands gap that directory as its
starting point (:func:`pinned_registries`), which makes the answer the same
from anywhere. An explicit ``skills=`` or ``$GAP_SKILLS_PATH`` still overrides
it wholesale -- that is gap's contract and the way to point a run at a
checkout that lives somewhere else.

**Stdlib only, on purpose.** The fast tier stages every shipped graph with no
``gap`` installed (nothing outside ``nodes/`` may ``import gap`` at module
scope), so the walk over ``<root>/{skills,tools}/*/SKILL.md``
is done here, mirroring what ``gap.skills.SkillsRegistry.discover`` does: a
bundle is a directory holding ``SKILL.md`` whose frontmatter ``name`` equals the
directory name; underscore-prefixed directories are skipped; a name claimed by
an earlier registry shadows a later one. Only :func:`pinned_registries` imports
gap, lazily, because the *merged* set (project config plus user config plus the
sibling walk) is gap's to compute and this module should not have a second
opinion about it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "GENERIC",
    "GENERIC_LABELS",
    "ORS_SIBLING",
    "RSS_ROOT",
    "BundleIndex",
    "bundle_index",
    "bundle_name",
    "pinned_registries",
    "project_registries",
    "registry_hint",
    "registry_roots",
]

RSS_ROOT = Path(__file__).resolve().parents[1]
"""This checkout's root: the directory holding ``pyproject.toml``."""

ORS_SIBLING = RSS_ROOT.parent / "open-robot-skills"
"""Where the documented side-by-side layout puts the skill registry.

The path ``[tool.gap]`` names, and the one every refusal below points at.
"""

GENERIC = "generic"
"""The ``skill`` a subgraph declares when it runs its own choreography.

Consults no bundle: the scripts are the graph's own, its collection's
``_shared/scripts/``, or shared rules under ``robosimstudio_policy_agent/nodes/``.
"""

GENERIC_LABELS = frozenset({GENERIC, "generated"})
"""Every spelling of that label. ``generated`` is what the self-learning
harness's older policies wrote -- gap's codegen vocabulary for a subgraph the
agent invented rather than took from the catalogue -- and those policies still
run here. Compare against this, not :data:`GENERIC`, wherever the question is
"does this subgraph consult a bundle"; the shipped graphs spell it ``generic``.
"""

KIND_DIRS = ("skills", "tools")
"""The bundle roots inside a registry, as gap's loader scans them."""

_NAME_LINE = re.compile(r"^name:\s*['\"]?([A-Za-z0-9_.-]+)['\"]?\s*$")


def bundle_name(skill_md: str | Path) -> str:
    """The ``name:`` a ``SKILL.md`` frontmatter declares, or ``""``.

    The frontmatter is the block between the first ``---`` line and the next;
    the name is read off its own line, without a YAML parser, because the fast
    tier has none to spare and the field is a bare identifier by contract.
    """
    try:
        lines = Path(skill_md).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    if not lines or lines[0].strip() != "---":
        return ""
    for line in lines[1:]:
        if line.strip() == "---":
            break
        match = _NAME_LINE.match(line.strip())
        if match:
            return match.group(1)
    return ""


def project_registries() -> list[Path]:
    """The registries this checkout's ``pyproject.toml`` names under ``[tool.gap]``.

    Resolved against the checkout root, not the cwd. ``tomllib`` where the
    interpreter has it (3.11+), ``tomli`` where that is installed, and a
    regular expression over the table otherwise -- the value is a list of
    string literals and nothing else, so the fallback is exact on the file this
    repository ships and merely lenient on one it does not.
    """
    pyproject = RSS_ROOT / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return []
    entries: list[str] | None = None
    toml: Any = None
    try:
        import tomllib as toml  # type: ignore[no-redef]
    except ImportError:
        try:
            import tomli as toml  # type: ignore[no-redef]
        except ImportError:
            toml = None
    if toml is not None:
        try:
            value = toml.loads(text).get("tool", {}).get("gap", {}).get("registries")
        except Exception:  # an unparseable file gets the regex below
            value = None
        if isinstance(value, list):
            entries = [str(v) for v in value]
    if entries is None:
        table = re.search(r"^\[tool\.gap\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
        if table is None:
            return []
        listed = re.search(r"registries\s*=\s*\[(.*?)\]", table.group(1), re.S)
        if listed is None:
            return []
        entries = re.findall(r"""["']([^"']+)["']""", listed.group(1))
    return [(RSS_ROOT / entry).resolve() if not Path(entry).is_absolute() else Path(entry) for entry in entries]


def registry_roots(explicit: str | Path | Sequence[str | Path] | None = None) -> list[Path]:
    """The registry roots to consult, in precedence order, without importing gap.

    The same first three layers :func:`pinned_registries` gets from gap --
    explicit argument, then ``$GAP_SKILLS_PATH``, then this checkout's
    ``[tool.gap]`` -- minus the two only gap can read (user config and the
    sibling walk). Roots that are not directories are dropped rather than
    refused: a checkout without the sibling is a supported state for every
    graph that names no bundle, and the graphs that do say so themselves.
    """
    if explicit is None:
        env = os.environ.get("GAP_SKILLS_PATH", "")
        if env.strip():
            candidates = [Path(p) for p in env.split(os.pathsep) if p.strip()]
        else:
            candidates = project_registries()
    elif isinstance(explicit, (str, Path)):
        candidates = [Path(explicit)]
    else:
        candidates = [Path(p) for p in explicit]
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.expanduser()
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(resolved)
    return roots


@dataclass(frozen=True)
class BundleIndex:
    """Every bundle the consulted registries hold, by name.

    ``roots`` are the registries that were searched, in precedence order, so a
    refusal can say where it looked; ``bundles`` maps a bundle name to its
    directory (the one holding ``SKILL.md``).
    """

    roots: tuple[Path, ...] = ()
    bundles: Mapping[str, Path] = field(default_factory=dict)

    def __contains__(self, name: object) -> bool:
        return name in self.bundles

    def __getitem__(self, name: str) -> Path:
        return self.bundles[name]

    def get(self, name: str) -> Path | None:
        return self.bundles.get(name)

    def names(self) -> list[str]:
        return sorted(self.bundles)


def bundle_index(registries: Sequence[str | Path] | None = None) -> BundleIndex:
    """Walk the registries and index their bundles, first root winning.

    Args:
        registries: Roots to walk, in precedence order. ``None`` consults
            :func:`registry_roots`.

    Raises:
        ValueError: a ``SKILL.md`` whose ``name`` disagrees with its directory,
            or one name twice inside one registry -- the two shapes gap's own
            loader refuses, caught here so that a bad bundle refuses one graph
            with its name rather than every graph at ``gap.execute``.
    """
    roots = registry_roots() if registries is None else registry_roots(list(registries))
    bundles: dict[str, Path] = {}
    for root in roots:
        claimed: set[str] = set()
        for kind in KIND_DIRS:
            kind_dir = root / kind
            if not kind_dir.is_dir():
                continue
            for entry in sorted(kind_dir.iterdir()):
                if not entry.is_dir() or entry.name.startswith("_"):
                    continue
                skill_md = entry / "SKILL.md"
                if not skill_md.is_file():
                    continue
                name = bundle_name(skill_md)
                if name and name != entry.name:
                    raise ValueError(
                        f"{skill_md}: frontmatter names the bundle {name!r} but the directory is "
                        f"{entry.name!r}; gap's loader refuses the pair, so it is refused here first"
                    )
                name = name or entry.name
                if name in claimed:
                    raise ValueError(f"{root}: the bundle name {name!r} appears twice under {root}")
                claimed.add(name)
                if name in bundles:
                    # A higher-precedence registry already claimed it: gap's
                    # shadow rule, first wins.
                    continue
                bundles[name] = entry
    return BundleIndex(roots=tuple(roots), bundles=bundles)


def registry_hint(roots: Sequence[Path]) -> str:
    """The sentence every refusal ends with: where the registries are, and how to get one."""
    if not roots:
        listed = ", ".join(str(p) for p in project_registries()) or "nothing"
        return (
            f"no skill registry resolved ($GAP_SKILLS_PATH is unset; [tool.gap] in {RSS_ROOT / 'pyproject.toml'} "
            f"names {listed}, which is not a checkout). RoboSimStudio ships no bundles: clone "
            f"open-robot-skills beside this checkout at {ORS_SIBLING}, or set GAP_SKILLS_PATH=/path/to/a/checkout "
            f"(scripts/bench_run.py --skills)."
        )
    where = ", ".join(str(p) for p in roots)
    return f"registries searched, in order: {where}. {_sibling_state()}"


def _sibling_state() -> str:
    """Which branch and commit the sibling checkout is at, when it is a git checkout."""
    if not ORS_SIBLING.is_dir():
        return f"The sibling {ORS_SIBLING} is absent."
    try:
        import subprocess

        def rev_parse(*args: str) -> str:
            out = subprocess.run(
                ["git", "-C", str(ORS_SIBLING), "rev-parse", *args, "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return out.stdout.strip() or "?"

        # Two calls: ``--abbrev-ref`` applies to every rev that follows it, so
        # one ``rev-parse --abbrev-ref HEAD HEAD`` prints the branch twice.
        branch, sha = rev_parse("--abbrev-ref"), rev_parse("--short")
        return f"The sibling {ORS_SIBLING} is at {branch}@{sha}; a graph may be written against a newer one."
    except Exception:  # provenance in a message, never a reason to fail
        return f"The sibling {ORS_SIBLING} is present."


def pinned_registries(skills: str | Path | Sequence[str | Path] | None = None) -> list[str]:
    """The registry set a run should execute with, resolved from this checkout.

    gap's own resolution, started from :data:`RSS_ROOT` rather than the cwd, so
    the project ``[tool.gap]`` layer is this repository's whatever directory
    the process was launched from; an explicit *skills* or ``$GAP_SKILLS_PATH``
    remains a full override, exactly as gap means them. Without gap importable
    the answer is :func:`registry_roots`, which is the same list minus user
    config and the sibling walk.

    Returns:
        Registry roots as strings, in precedence order -- possibly empty, which
        ``gap.execute(skills=None)`` reads as "resolve for yourself".
    """
    try:
        from gap.skills.registries import resolve_registries
    except ImportError:
        return [str(p) for p in registry_roots(skills)]
    resolved = resolve_registries(skills, cwd=RSS_ROOT, search_from=(RSS_ROOT,))
    return [str(p) for p in resolved.paths()]
