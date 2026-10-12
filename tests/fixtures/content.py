# SPDX-License-Identifier: Apache-2.0
"""Whether any content registered an asset record here, for a test that has to say "if there is any".

The library declares no :class:`~robosimstudio.assetlib.AssetRecord` of its own.
The records are content's: a task collection declares each beside its meshes,
and they are loaded when a root holds a collection. A release may ship the
library with no collection at all, and then the catalog is not small, it is
empty -- ``list_assets()`` is ``[]``, ``find_asset`` answers ``None`` for every
name, and ``get_asset`` raises a ``KeyError`` that lists nothing.

Most tests about an asset belong to the collection that declares it and are
withheld with it. A few are about the catalog the library keeps on both sides of
that line -- that it validates what it is given, that one reader survives every
record -- and those have to state both answers rather than assume a corpus. This
is the one question they ask.

Asked of the registry rather than of the filesystem, so a collection on a root
outside the checkout counts and a directory nothing loaded does not.
"""

from __future__ import annotations

import unittest

from robosimstudio.assetlib import list_assets

__all__ = ["HAS_CONTENT_ASSETS", "needs_content_assets"]

HAS_CONTENT_ASSETS = bool(list_assets())
"""Whether the asset registry holds anything in this checkout."""

needs_content_assets = unittest.skipUnless(HAS_CONTENT_ASSETS, "no content registered an asset record in this tree")
"""Decorator for a test or a class that sweeps the corpus or reaches for one asset by name."""
