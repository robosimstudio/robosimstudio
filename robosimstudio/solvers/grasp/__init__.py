# SPDX-License-Identifier: Apache-2.0
"""Where to put the jaws.

Two answers with different evidence. The oriented-box heuristic reads a bounding
box and is what a benchmark port needs to reproduce a number. The antipodal
sampler reads the real mesh, scores pairs by what the jaws would actually meet,
and is what a policy needs when the box is a centimetre of invisible shell.
"""

from __future__ import annotations

__all__: list[str] = []
