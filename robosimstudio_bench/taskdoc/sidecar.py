# SPDX-License-Identifier: Apache-2.0
"""Reading an export's sidecar, and the one convention it does not share with us.

BDDL has no coordinates, and neither does the exported scene file: MJCF and USD
say what the geometry *is*, and the arrangement a benchmark actually ran -- what
the pile settled into, which body carries which name, where a welded container's
cavity sits -- is recorded beside it, in ``scene_meta.json``.

The **quaternion convention** is the part worth naming rather than inlining.
MuJoCo and USD both write ``wxyz``; Newton reads ``xyzw``. A conversion that is
correct and unnamed reads as reordering for no reason, and one that is *missing*
produces a scene that builds, steps, and has every object rotated -- which is a
long way to find out from a tuple index.

What is deliberately **not** here is how a suite turns its sidecar into poses.
The two do that differently for measured reasons -- tool_sort forces bounding-box
colliders because a raw hull leaves a thin tool rocking in a limit cycle, and its
body-name fallbacks differ from the grocery stage's prim-path leaves -- so a
shared ``poses()`` would take a callback per difference and be longer than the two
functions it replaced, with the collider rationale moved somewhere it is no
longer next to its scene.
"""

from __future__ import annotations

import json
from typing import Any

from robosimstudio.paths import resolve_path

__all__ = ["load_sidecar", "wxyz_to_xyzw"]


def load_sidecar(export: str, name: str = "scene_meta.json") -> dict[str, Any]:
    """One export's metadata, by the export directory the document staged.

    Args:
        export: The export directory, as ``(:staging (:export ...))`` names it.
        name: The sidecar's filename, for an export that keeps more than one.

    Returns:
        Whatever the export recorded. Untyped on purpose: the file is the
        exporter's format rather than ours, different exporters record different
        things, and a schema here would be this package asserting something about
        somebody else's file.
    """
    with open(resolve_path(f"{export}/{name}")) as f:
        return json.load(f)


def wxyz_to_xyzw(q) -> tuple[float, float, float, float]:
    """A quaternion from an exporter's convention into this stack's.

    MuJoCo and USD write ``wxyz``; Newton reads ``xyzw``.
    """
    w, x, y, z = (float(v) for v in q)
    return (x, y, z, w)
