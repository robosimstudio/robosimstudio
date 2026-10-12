# SPDX-License-Identifier: Apache-2.0
"""Live input sources for teleop, alongside the studio's own gizmo drag.

* :mod:`~robosimstudio.robots.teleop.lerobot_controller` -- a Feetech leader
  arm read as a ``joint_position`` action: calibration, tick conversion, and a
  read-only controller.
* :mod:`~robosimstudio.robots.teleop.force_feedback` -- the same leader with
  the simulated arm's resistance rendered back onto its servos.
* :mod:`~robosimstudio.robots.teleop.follower` -- which shipped robot a YAM
  leader drives, and the servo gains that make it track.
* :mod:`~robosimstudio.robots.teleop.simulated` -- a servo bus over a table and
  a scripted hand, for running the above with no arm on the desk.

Nothing here is imported eagerly: a controller wraps a specific piece of
hardware, and opening one pulls in that hardware's driver library. Reach a
module by importing it directly, the way a job file names a policy by dotted
path.
"""
