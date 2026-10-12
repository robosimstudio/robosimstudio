# SPDX-License-Identifier: Apache-2.0
"""The vocabulary: what may be asked, and the geometry several substrates share.

What is left is what genuinely belongs to no one kind: the **atom** declarations
(a goal may ask the height of a sheet, a hammer or a cable, so what "height"
means is nobody's substrate), and the geometry several substrates run --
occupancy-grid overlap, region bounds, the mean of a point cloud. Importing this
package registers the catalogue; the domains register the measurements, and the
``HANDLERS`` and ``ATOMS`` scans over ``robosimstudio.domains`` are what reach
them.

Every handler here is a **pure function of geometry already in world
coordinates**. That separation is deliberate and it is what makes this layer
testable at all: transforming a body-frame payload by a pose is one job, and
deciding whether a rod passes through a loop is another. Split that way, the
second can be checked against hand-derived numbers on synthetic inputs, in
microseconds, with no simulator -- and the first is a matrix multiply nobody
needs a test to believe.

The alternative, handlers that reach into an environment for their operands, is
how measures become untestable without a GPU. This stack has already paid that
price once: identical code has been measured scoring 0.84 and 0.90 on
consecutive folds, so anything only checkable through a rollout is only ever
checkable within a band. Pure functions of explicit inputs are checkable exactly.
"""

from __future__ import annotations

from . import coverage, motion, regions

__all__ = ["coverage", "motion", "regions"]
