#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write each example folder's README sections from its scripts' docstrings.

Every ``examples/<NN_name>/README.md`` holds two kinds of text. Between the
markers ``<!-- examples:start ... -->`` and ``<!-- examples:end -->`` is one
section per numbered script, written here from the script's docstring: its
title, its clip (``media/<script>.webp``, when there is one), what it shows, and
the line that runs it, with the extras the docstring says it needs. Everything
outside the markers is written by hand and left alone. So a new example is a
script with a docstring, a clip in ``media/`` if it has one, and a run of this.

Usage, from the repository root::

    uv run python scripts/example_readmes.py           # rewrite the sections
    uv run python scripts/example_readmes.py --check   # exit 1 if a README is out of date

``--check`` also refuses a clip in a ``media/`` folder that no page under
``examples/`` shows, a page that shows a clip that is not there, and a numbered
folder with no README to write into.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"

START = "<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->"
END = "<!-- examples:end -->"
BLOCK = re.compile(r"<!-- examples:start[^>]*-->\n.*?<!-- examples:end -->", re.S)

FOLDER = re.compile(r"^\d\d_\w+$")
SCRIPT = re.compile(r"^\d\d_\w+\.py$")
REQUIRES = re.compile(r"(?:\b[Nn]eeds|\b[Rr]equires)\b[^.\n]*?\[(?P<extra>[A-Za-z][\w-]*)\]")
"""How a docstring declares an extra: ``Needs the ``[superdex]`` extra``."""
ROLE = re.compile(
    r":(?:func|class|meth|attr|data|mod|obj|exc):`(?P<tilde>~?)(?P<target>[^`<]+?)(?:\s*<(?P<link>[^`>]+)>)?`"
)
SHOWN = re.compile(r"\]\(((?:\.\./)*(?:[\w.-]+/)?media/[\w.-]+\.(?:webp|gif|png|jpg))\)")
"""A picture a page shows: ``](media/x.webp)`` or ``](../02_cloth/media/x.webp)``."""


def flat_names() -> set[str]:
    """The names ``rss.`` carries, read off the package's tables without importing it."""
    tree = ast.parse((REPO / "robosimstudio" / "__init__.py").read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else next(iter(getattr(node, "targets", [])), None)
        if getattr(target, "id", None) == "_EXPORTS":
            names |= set(ast.literal_eval(node.value))
        elif getattr(target, "id", None) == "_PACKS":
            for _, provided in ast.literal_eval(node.value).values():
                names |= set(provided)
    return names


def markdown(text: str, flat: set[str]) -> str:
    """Docstring prose as GitHub Markdown: roles become code spans, ``literals`` become `literals`."""

    def role(match: re.Match) -> str:
        target = match.group("target").strip()
        if match.group("link"):
            return f"`{target}`"
        if match.group("tilde"):
            return f"`{target.rsplit('.', 1)[-1]}`"
        parts = target.split(".")
        if parts[0] == "robosimstudio" and len(parts) > 1 and parts[1] in flat:
            return "`rss." + ".".join(parts[1:]) + "`"
        return f"`{target}`"

    text = ROLE.sub(role, text)
    return re.sub(r"``([^`]+)``", r"`\1`", text)


def section(folder: Path, script: Path, flat: set[str]) -> str:
    """One script's section: title, clip, what it shows, how to run it."""
    docstring = ast.get_docstring(ast.parse(script.read_text(encoding="utf-8"))) or ""
    lines = docstring.strip().splitlines()
    title = (lines[0].strip().rstrip(".") if lines else "") or script.stem.split("_", 1)[1].replace(
        "_", " "
    ).capitalize()
    body = markdown("\n".join(lines[1:]).strip("\n"), flat)
    extras: list[str] = []
    for match in REQUIRES.finditer(" ".join(docstring.split())):
        if match.group("extra") not in extras:
            extras.append(match.group("extra"))
    run = " ".join(["uv run", *(f"--extra {extra}" for extra in extras), f"examples/{folder.name}/{script.name}"])

    parts = [f"## {title}", ""]
    clip = folder / "media" / f"{script.stem}.webp"
    if clip.is_file():
        parts += [f"![{title}](media/{clip.name})", ""]
    if body:
        parts += [body, ""]
    parts += ["```bash", run, "```", "", f"Source: [`{script.name}`]({script.name})"]
    return "\n".join(parts)


def block(folder: Path, flat: set[str]) -> str:
    scripts = sorted(p for p in folder.iterdir() if SCRIPT.match(p.name))
    sections = "\n\n".join(section(folder, script, flat) for script in scripts)
    return f"{START}\n\n{sections}\n\n{END}"


def folders() -> list[Path]:
    return sorted(p for p in EXAMPLES.iterdir() if p.is_dir() and FOLDER.match(p.name))


def written(folder: Path, flat: set[str]) -> tuple[str, str]:
    """The README as it is and as it should be."""
    readme = folder / "README.md"
    if not readme.is_file():
        raise SystemExit(f"{readme.relative_to(REPO)} is missing: every numbered folder has one")
    text = readme.read_text(encoding="utf-8")
    if not BLOCK.search(text):
        raise SystemExit(f"{readme.relative_to(REPO)} has no {START[:21]}... / {END} markers")
    return text, BLOCK.sub(lambda _: block(folder, flat), text, count=1)


def unshown_media() -> list[str]:
    """Clips under ``examples/*/media/`` that no page under ``examples/`` shows."""
    shown = {
        (page.parent / ref).resolve()
        for page in EXAMPLES.rglob("*.md")
        for ref in SHOWN.findall(page.read_text(encoding="utf-8"))
    }
    clips = sorted(p for p in EXAMPLES.glob("*/media/*") if p.is_file())
    return [str(p.relative_to(REPO)) for p in clips if p.resolve() not in shown]


def dangling_media() -> list[str]:
    """Clips a page under ``examples/`` shows that are not on disk: ``page -> clip``.

    The other half of :func:`unshown_media`. A clip renamed or moved -- an
    extra clip a recorder take now writes under a new name, a demo moved to
    its own folder -- leaves the page pointing at nothing, and GitHub draws
    that as a broken image with the alt text beside it.
    """
    missing = []
    for page in sorted(EXAMPLES.rglob("*.md")):
        for ref in SHOWN.findall(page.read_text(encoding="utf-8")):
            if not (page.parent / ref).is_file():
                missing.append(f"{page.relative_to(REPO)} -> {ref}")
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a README is out of date; write nothing")
    args = parser.parse_args(argv)
    flat = flat_names()
    stale = []
    for folder in folders():
        now, should = written(folder, flat)
        if now != should:
            stale.append(str((folder / "README.md").relative_to(REPO)))
            if not args.check:
                (folder / "README.md").write_text(should, encoding="utf-8")
    if args.check:
        unshown, dangling = unshown_media(), dangling_media()
        for path in stale:
            print(f"{path} is out of date: run scripts/example_readmes.py", file=sys.stderr)
        for path in unshown:
            print(f"{path} is shown by no page under examples/", file=sys.stderr)
        for shown in dangling:
            print(f"{shown} shows a clip that is not there", file=sys.stderr)
        return 1 if stale or unshown or dangling else 0
    for path in stale:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
