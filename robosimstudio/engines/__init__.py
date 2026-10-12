# SPDX-License-Identifier: Apache-2.0
"""Foreign physics engines, co-simulated beside the Newton islands.

An engine is not a domain, and this package exists to say so.

What lives here is the half of a co-simulation that is not about any particular
engine:

``contract``
    ``CosimBridge``: what an engine's bridge must be, as a Protocol.
``pose``
    The pose arithmetic a kinematic mirror needs, in the hot-path form.
``geometry``
    Newton shapes and particle ranges, read into a form any engine builds from.
``feedback``
    Contact wrenches from a foreign engine, shaped and fed into ``body_f``.
``cosim``
    The per-control-step tick, minus the engine.
``host``
    ``SolverHost``: the solver contract a host engine satisfies.

and, beside them, **one subpackage per engine, in parallel** -- ``superdex`` and
``genesis`` -- each holding only what that engine's own API forces and each
registering its own alias. Nothing is registered at this level: an engine that
does not register itself is an engine whose bundle nobody imported, and the
``ENGINES`` scan in ``specs/registries.py`` is what imports them. The seam they meet at is the host-engine contract the composer
already speaks -- ``scene/physics`` splits a host island out before the
couplers, ticks it once per control step in the eager ``after`` phase, and reads
a ``rigid_force_stage`` off the solver if it offers one. None of that changes
here, and none of it learns an engine's name.

**``newton`` is the third bundle, and it is not a foreign engine.** The five
aliases for the solvers that integrate Newton's own state on the device --
``mujoco``, ``vbd``, ``xpbd``, ``featherstone`` and ``kinematic`` -- register
from ``engines/newton`` through the same scan, so an alias has one kind of home
and ``get_backend`` has one path.
"""

from __future__ import annotations

__all__: list[str] = []


from robosimstudio import roots as _roots

__path__ = _roots.extend_path(__path__, _roots.engine_roots(), kind="engine")
"""This package's directories: the library's own, then every engine root.

**A foreign engine may live outside this checkout**, and reach the same import
name. ``ROBOSIMSTUDIO_ENGINE_ROOTS`` is the list, and
``robosimstudio.engines.<name>`` resolves against it exactly as against the
directory beside this file -- so the registries that scan this package
(``pkgutil.iter_modules(__path__)``) find in-tree and out-of-tree alike in one
sweep, with no registry change and no second discovery mechanism.

Extending ``__path__`` rather than executing each root under a synthetic parent,
which is what the *task* roots do: a task is a directory of documents that must
not become a top-level importable name, while this is a package that wants to
*be* one of ours. The cost is that two roots holding the same name is a real
clash, which :func:`~robosimstudio.roots.extend_path` refuses outright --
``iter_modules`` would silently keep the first and never import the second.
"""
