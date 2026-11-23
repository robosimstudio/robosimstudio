# SPDX-License-Identifier: Apache-2.0
"""i2rt's crank_4310, the gripper on the real bimanual YAM rig (flip-anything ``real/``).

Built from i2rt's own model files by flip-benchmark's ``robot/build/build_yam_crank.py``, which measures every number below on the
composed model and writes them to ``assets/embodiments/yam_crank_real/crank_4310.json``. This module reads
that file, so the spec cannot drift from the asset it describes.

Facts the measurement found (i2rt @ 120c3c81):

* ``joint7`` runs 0 (shut: the pads meet, the meshes overlap 4 mm) to 0.0396 (open); ``joint8`` follows by
  an equality. i2rt's normalized command is the same way round: 0 closed, 1 open.
* i2rt's model opens 75 mm; the slide range is extended to the real gripper's measured 100 mm.
* The fingertips are bent: the pads sit 45 mm off the wrist axis (gripper +y), 140 mm out along +z.
* The gripping pad is the last 14 mm of each finger (behind it the face is recessed 4 mm). The TCP is the
  pad's centre, 7 mm short of i2rt's ``grasp_site``, which sits at the very end of the pads.
  The pads run parallel to +z near the tip, so +z is the approach axis; the jaws close along gripper x.
* The tips are long, curved, open shells: their convex hulls would be solid wedges, so the embodiment
  decomposes them.
"""

from __future__ import annotations

import json

from robosimstudio.paths import resolve_path
from robosimstudio.specs import GripperSpec

from ..registry import register_gripper

__all__ = ["CRANK_4310", "MEASURED"]

# Written by robot/build/build_yam_crank.py; without it there is no robot to describe,
# and the other embodiments must still load.
try:
    MEASURED = json.loads(resolve_path("embodiments/yam_crank_real/crank_4310.json").read_text())
except FileNotFoundError:
    MEASURED = None

CRANK_4310 = None if MEASURED is None else register_gripper(
    GripperSpec(
        name="crank_4310",
        finger_names=("{arm}_joint7",),
        follower_names=("{arm}_joint8",),
        finger_open_q=(MEASURED["finger_open_q"],),
        finger_closed_q=(MEASURED["finger_closed_q"],),
        open_width=abs(MEASURED["finger_open_q"] - MEASURED["finger_closed_q"]),
        span=MEASURED["span_open"],
        tcp_body="{arm}_gripper",
        tcp_offset=tuple(MEASURED["tcp_offset"]),
        tcp_offset_rot=tuple(MEASURED["tcp_offset_rot"]),
        approach_axis=tuple(MEASURED["approach_axis"]),
        close_axis=tuple(MEASURED["close_axis"]),
        finger_bodies=(("{arm}_tip_left",), ("{arm}_tip_right",)),
        # The tip bodies slide along their local x, not y; ContactProbe.check_grasp reads local y, so its
        # verdict is not meaningful on this hand. The flip policy measures its own grasps.
        open_axis_signs=(1.0, -1.0),
        # Measured: the TCP is the centre of the 14 mm gripping pad, the fingertips end this far past it.
        # Not padded: the policy sets grasp heights from it (Primitives.floor), and an 8 mm margin put the
        # pads wholly above an 8 mm breadboard -- every pivot on it closed on air.
        tip_reach=MEASURED["tip_reach"],
        # i2rt get_robot()'s GripperForceLimiter on this gripper, with the motor's PD seen at the finger. All
        # derived from i2rt's crank_4310.yml and the measured 100 mm jaw by robot/build/build_yam_crank.py
        # (robot/PARAMETERS.md): 50 N asked through i2rt's 71 mm force map is 35.5 N on a 25.3 mm crank.
        **MEASURED["limiter"],
    )
)
