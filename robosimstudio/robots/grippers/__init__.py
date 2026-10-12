# SPDX-License-Identifier: Apache-2.0
"""Gripper specs: the hands, separately from the arms they bolt to.

Each module registers on import; the registry loads them all. One module per
hand, named for the hardware rather than for the robot that happens to carry it
-- ``robotiq_2f85``, not ``droid_hand`` -- because the whole point of the split
is that the next arm to carry one names the same entry.

**A descriptive spec is keyed to the file that carries the hand, not just to the
hardware.** The 2F-85 as the Isaac stage names it and a 2F-85 out of some other
export are two entries, because the joint and body labels a spec indexes by are
the exporter's, not the manufacturer's. Where that matters the registry key says
so.

The measurements live here rather than in the graph scripts that used to restate
them, and they come with the sentence that justifies them: a number like "the
housing's underside sits 40 mm above the fingertips" is worth exactly as much as
the account of how it was arrived at.
"""
