# SPDX-License-Identifier: Apache-2.0
"""What a thing *is*, one bundle per physical kind.

Each bundle -- a sub-package, or one module -- registers its domain and carries
what belongs to it: the entity that builds one, the measures that read one, its
materials. They are imported by the ``DOMAINS`` registry's package scan, not by
name from below -- the layer that defines what a domain *is* sits under the layer
that says which ones exist.

The directory beside this file holds the kinds the library itself is made of
(``rigid``, ``static``, ``articulated``) and what more than one kind shares
(``capabilities``, ``selection``). ``cloth``, ``soft`` and the rope are not
here: each lives in its own example folder -- ``examples/02_cloth/cloth.py``,
``examples/03_soft/soft.py``, a package in ``examples/10_rope`` -- written to be
read beside the numbered scripts that run it, and reached as
``robosimstudio.domains.cloth`` all the same -- see ``__path__`` below.

A domain may name another domain. A fixture one kind of thing bears against is
a static and subclasses one -- the post lives in ``static`` and the kinds that
wind round it reuse it -- which is a real relationship rather than a layering
slip; the rule that matters is the one between *tasks*, and it is unchanged.

**Not every kind is in every release.** A domain is a directory, here or on a
domain root, and a release may leave one out. Nothing in the library names a
domain package, so nothing breaks when one is absent: its registrations simply
are not there, and the flat names it would have provided say so
(``robosimstudio._PACKS``).
"""

from __future__ import annotations

__all__: list[str] = []


from robosimstudio import roots as _roots

__path__ = _roots.extend_path(__path__, _roots.domain_roots(), kind="domain")
"""This package's directories: the library's own, then every domain root.

**A kind of thing may live outside this package**, and reach the same import
name. A checkout's own roots come first -- ``examples/02_cloth``,
``examples/03_soft`` and ``examples/10_rope``, each only if the directory is there
(:data:`~robosimstudio.roots.DEFAULT_DOMAIN_ROOT_NAMES`); an installed wheel
carries the same packs in ``robosimstudio/_domain_packs`` instead. After them
``ROBOSIMSTUDIO_DOMAIN_ROOTS`` is the list for anybody else's, and
``robosimstudio.domains.<name>`` resolves against all of it exactly as against
the directory beside this file -- so the registries that scan this package
(``pkgutil.iter_modules(__path__)``) find in-tree and out-of-tree alike, package
or single module, in one sweep, with no registry change and no second discovery
mechanism.

Extending ``__path__`` rather than executing each root under a synthetic parent,
which is what the *task* roots do: a task is a directory of documents that must
not become a top-level importable name, while this is a package that wants to
*be* one of ours. The cost is that two roots holding the same name is a real
clash, which :func:`~robosimstudio.roots.extend_path` refuses outright --
``iter_modules`` would silently keep the first and never import the second.

**A root may be an example folder.** The numbered scripts beside ``cloth.py``
(``00_towel.py``) are on this ``__path__`` too, and are not packs: a name that
is not a Python identifier is skipped by the duplicate rule and by the
registries' scan, so nothing imports one as ``robosimstudio.domains.<name>``.
"""
