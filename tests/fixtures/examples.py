# SPDX-License-Identifier: Apache-2.0
"""An example run from a test: which scripts there are, and ``main()`` against a studio that is not there.

Shared by :mod:`tests.test_examples` and, in a tree that has the older rope
scripts, the module that tests them, which split one module along the
600-line rule: the first is about the numbered examples every tree ships, the
second about what only those rope scripts claim. Both load a script by file path and run its
``main()`` with :func:`robosimstudio.view` replaced, and one copy of that is
more than tidiness: an example that registers a kind does so at import, so the
load is cached here, and a loader in each module would be two registrations in
any process that runs both.

**The layout is viser's**: numbered category folders holding numbered scripts
(``examples/02_cloth/00_towel.py``), each of which runs alone. A numbered folder
is not an importable package, so every example is loaded by file path -- which
is also how a reader runs one (``uv run examples/02_cloth/00_towel.py``). The
rope's numbered examples are in ``examples/10_rope/`` like every other kind's;
:data:`ROPE` is the four older flat scripts under ``examples/scenes/``, in a
tree that has them, and is empty in a tree that does not.

Importing this module imports nothing of the library, so reading its tables
needs no GPU; the library is loaded with the first example a test asks for.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import re
import sys
from pathlib import Path

__all__ = [
    "DEMOS",
    "EXAMPLES",
    "HAS_ROPE",
    "NO_ROPE",
    "NUMBERED",
    "ROOT",
    "ROPE",
    "SCRIPTS",
    "_load",
    "_run",
    "_stub_studio",
]

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"

NUMBERED = (
    "01_rigid/00_drag_things.py",
    "02_cloth/00_towel.py",
    "02_cloth/02_engines.py",
    "03_soft/01_tet_mesh.py",
    "04_extending/01_new_kind.py",
    "10_rope/00_cable.py",
    "10_rope/01_between_posts.py",
)
"""The numbered scripts that run on a base checkout: no flags, every parameter a
local, two arms, one call to the studio. ``02_engines`` is here because its one
local defaults to ``newton``; the engines it can be switched to are the subject
of ``TestTheMultiEngineDemos`` in :mod:`tests.test_examples`."""

ROPE = tuple(
    path
    for path in ("scenes/cable.py", "scenes/cables.py", "scenes/engines.py", "scenes/ported.py")
    if (EXAMPLES / path).is_file()
)
"""The older flat rope scripts, held to the same rules where a tree has them.
Empty in a tree that ships without them, and every test over :data:`SCRIPTS`
then covers the numbered scripts alone."""

SCRIPTS = (*NUMBERED, *ROPE)

DEMOS = ("03_soft/00_sponge.py", "02_cloth/03_genesis.py", "10_rope/02_superdex.py")
"""The multi-engine demos, which the ones above cannot stand in for: each needs a
foreign engine, and each is the *acceptance* scene for one kind on it. They are
not in :data:`SCRIPTS` because none satisfies its rules on a base checkout --
the sponge stands under a single-armed Panda where the scene scripts ship a
bimanual robot, the Genesis cloth cannot build at all without a wheel this venv
deliberately does not carry, and the SuperDex cable needs the ``[superdex]``
extra to build."""

HAS_ROPE = bool(ROPE)
NO_ROPE = "the older rope scripts (examples/scenes/) are not in this tree"
"""What a test about a rope example says when it skips for want of :data:`HAS_ROPE`."""


@contextlib.contextmanager
def _stub_studio():
    """``rss.view`` replaced for the duration; what it was handed is yielded."""
    import robosimstudio as rss

    opened: list[tuple] = []

    def view(sim, **kwargs):
        opened.append((sim, kwargs))

    real = rss.view
    rss.view = view
    try:
        yield opened
    finally:
        rss.view = real


@functools.cache
def _load(rel: str):
    """One example as a module, loaded by file path, once.

    ``examples/02_cloth`` is not a package and ``00_towel`` is not a module name,
    so ``importlib.import_module`` cannot reach these and is not how a reader
    does either. Once, because an example that registers a kind does so at
    import, and a second import would be a second registration. Entered into
    ``sys.modules`` before it runs, which is what a ``@wp.kernel`` or a
    dataclass defined in the file needs to find its own module.
    """
    name = "_rss_example_" + re.sub(r"\W", "_", rel.removesuffix(".py"))
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run(rel: str):
    """One example's ``main()``, returning the ``(sim, kwargs)`` it opened the studio with."""
    with _stub_studio() as opened:
        _load(rel).main()
    if len(opened) != 1:
        raise AssertionError(f"examples/{rel} opened {len(opened)} pages, not one")
    return opened[0]
