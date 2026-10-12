# SPDX-License-Identifier: Apache-2.0
"""Which domain packs this checkout carries, for a test that has to say "if it is here".

A domain pack is a directory on a domain root, and a release may leave one out
(the rope, ``robosimstudio.domains.rod``). Most tests about a pack belong *to*
the pack and are withheld with it. A few are about a seam the library keeps on
both sides of that line -- which layouts exist, which actor kinds are
registered, which feature kinds have a contract -- and those have to state both
answers rather than assume the pack. This is the one question they ask.

Asked of the import system rather than of the filesystem, so a pack on a root
outside the checkout counts and a directory that is not on any root does not.
"""

from __future__ import annotations

import importlib.util
import unittest

__all__ = ["HAS_ROD_PACK", "has_pack", "needs_rod_pack"]


def has_pack(name: str) -> bool:
    """Whether ``robosimstudio.domains.<name>`` is installed here."""
    try:
        return importlib.util.find_spec(f"robosimstudio.domains.{name}") is not None
    except (ImportError, ValueError):
        return False


HAS_ROD_PACK = has_pack("rod")
"""Whether the rope -- the ``rod`` domain -- is in this checkout."""

needs_rod_pack = unittest.skipUnless(HAS_ROD_PACK, "the rod pack (robosimstudio.domains.rod) is not in this checkout")
"""Decorator for a test or a class that is about something only the rod pack provides."""
