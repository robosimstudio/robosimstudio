# SPDX-License-Identifier: Apache-2.0
"""The cells the library itself ships: ``void``, and three plain benches.

A scene owns the ground, the work surface and the furniture, and nothing a task
is *about*. The dividing line is whether removing it would leave the task still
meaningful: take away the table and a USB-C insertion is a different setup but
still an insertion; take away the receptacle and there is no task at all. So the
table is the scene's and the receptacle is the asset's.

**A cell that is a *particular room* is content and ships with the task that
measured it.** An imported arena, a pegboard workcell, a stage somebody
photographed -- each is a measurement of one table, which is exactly what a
library must not make its users carry. Those live in the task directory that
stages them.

What is left is two kinds of thing that are not particular. ``void`` is the
ground plane, no statics, work surface *at* the floor: it declares nothing about
any room, which makes it the null hypothesis a portability bug is checked
against. And :mod:`~robosimstudio.scene.builtin.benches` is a flat table at three
heights, for a task with no opinion about the room it stands in.

The roster is short on purpose, and a fourth entry should be argued for rather
than added: it is a deliberate edit.
"""

from __future__ import annotations
