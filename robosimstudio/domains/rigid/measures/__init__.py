# SPDX-License-Identifier: Apache-2.0
"""What may be measured of a rigid body, beside the kind that builds one.

A package rather than one module because the subject genuinely splits four ways,
and the split is the one the files already had: contact and support
(``placement``), a jaw's grip (``jaws``), a mated fit (``insertion``), and the
feature geometry a loop and a rod make between them (``geometry``). Collapsing
1,100 lines into a single ``measures.py`` for symmetry with ``domains/cloth``
-- whose measures really are one subject -- would be symmetry bought with
legibility.

Importing this registers every one of them, and ``rigid/__init__.py`` imports
this, which is how the ``HANDLERS`` and ``ATOMS`` scans over
``robosimstudio.domains`` reach them.
"""

from __future__ import annotations

from . import geometry, insertion, jaws, placement, regions

__all__ = ["geometry", "insertion", "jaws", "placement", "regions"]
