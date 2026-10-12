# SPDX-License-Identifier: Apache-2.0
"""Which tasks exist, and how a document becomes one.

The layer between the task *language* and the scene: it takes a checked
:class:`~robosimstudio_bench.taskdoc.ast.TaskDoc` and produces a
:class:`~robosimstudio.specs.TaskSpec`, resolving every name the document was
allowed to leave as a name -- the arrangement it stages, the camera rig it
names, the artifacts it references.

* :mod:`~robosimstudio_bench.taskbuild.lower` -- document to spec, the one function every
  suite loader ends in.
* :mod:`~robosimstudio_bench.taskbuild.staging` -- import profiles, and merging what a
  document says over what its profile supplies.
* :mod:`~robosimstudio_bench.taskbuild.arrangements` -- the versioned names a document
  may stage: physics bundles and camera rigs, registered by the content that
  measured them. **A document may never state a solver number**; it names one of
  these, and the name is what a recorded result means.
* :mod:`~robosimstudio_bench.taskbuild.registry` -- which suites exist.
* :mod:`~robosimstudio_bench.taskbuild.matrix` -- which (task, scene, robot, hand)
  combinations are admissible, and why a refused one is refused.
* :mod:`~robosimstudio_bench.taskbuild.episodes` -- what one episode of a task is.

It sits beside the asset library rather than above or below it: an asset never
names a task, and a task reaches the asset library only through the evaluator
factory its loader passes to :func:`~robosimstudio_bench.taskbuild.lower.lower`.
"""

from __future__ import annotations

from .arrangements import (
    CAMERA_RIGS,
    PHYSICS,
    get_arrangement,
    get_camera_rig,
    register_arrangement,
    register_camera_rig,
)
from .episodes import make_episode
from .lower import Lowered, LowerError, lower, register_sampler
from .matrix import admits, candidate_pairings, candidate_triples, list_pairs, pairing_key
from .place import build_entities, from_export, sidecar_poses
from .registry import SUITES, get_suite, get_task, list_tasks, load_all, register_suite
from .staging import Profile, Staged, Staging, bind_profile, get_profile, register_profile, stage
from .text import task

__all__ = [
    "CAMERA_RIGS",
    "PHYSICS",
    "SUITES",
    "LowerError",
    "Lowered",
    "Profile",
    "Staged",
    "Staging",
    "admits",
    "bind_profile",
    "build_entities",
    "candidate_pairings",
    "candidate_triples",
    "from_export",
    "get_arrangement",
    "get_camera_rig",
    "get_profile",
    "get_suite",
    "get_task",
    "list_pairs",
    "list_tasks",
    "load_all",
    "lower",
    "make_episode",
    "pairing_key",
    "register_arrangement",
    "register_camera_rig",
    "register_profile",
    "register_sampler",
    "register_suite",
    "sidecar_poses",
    "stage",
    "task",
]


def _install_content_discovery() -> None:
    """Teach the library's own registries where content lives.

    ``CELLS``, ``EMBODIMENTS`` and ``GRIPPERS`` hold things the **library**
    ships -- four benches, six robots, seven hands -- and a handful that a task
    directory adds.

    The hook is installed from here instead, because *where content lives* is
    this package's subject and not theirs. Nothing else changes: a task
    directory's cell or robot is found exactly as before, from the moment
    anything imports the task machinery -- and
    :meth:`~robosimstudio.registry.Registry.add_discover` re-arms a registry that
    had already been loaded, so a lookup made before this ran is not left with a
    stale answer.

    ``DOMAINS`` is here for a different reason from the other three. It was the
    registry whose docstring said *"A task registers its own the same way, from
    its assets/ package"* -- and that was **false as a contract**: it scanned
    ``robosimstudio.domains`` and had no populator at all, so a domain declared
    by a task registered only if something else happened to import that task
    first. Eight domains are declared in the library and none in content, which
    is what a promise nobody could keep looks like from the outside.

    ``ASSETS`` is deliberately not here: it keeps its own hook, because the
    library declares two assets and content declares thirty-six, so loading
    content to answer an asset lookup is the answer rather than a cost. Reaching
    it from this module would also be a module-scope ``taskbuild -> assetlib``
    import, which is the cycle ``LAZY_EDGES`` exists to keep inside a function.
    """
    from robosimstudio.robots.registry import EMBODIMENTS, GRIPPERS
    from robosimstudio.scene.registry import CELLS
    from robosimstudio.specs.physics import DOMAINS

    from .discovery import load_roots

    for registry in (CELLS, EMBODIMENTS, GRIPPERS, DOMAINS):
        registry.add_discover(load_roots)


_install_content_discovery()
