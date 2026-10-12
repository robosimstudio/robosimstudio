# SPDX-License-Identifier: Apache-2.0
"""Tests for the semantic layer: features, selectors, atoms, handlers, goals.

**No GPU, and no scene.** Everything here runs against synthetic geometry and
scripted contexts, which is deliberate rather than convenient: a goal tree is a
pure function of a snapshot, so the semantics of ``once``, ``hold_for`` and
``after`` can be pinned exactly, in milliseconds, without asking a solver to
reproduce a situation. The pattern is the one ``TestFoldMetric`` already uses --
a hand-built layout, a real probe, the real production measure, and hand-derived
expected values.

That is also why these tests are the *strict* half of the equivalence discipline.
A live rollout can only ever be compared within a tolerance band, because
identical code has been measured scoring 0.84 and 0.90 on consecutive folds. A
scripted snapshot has no such problem, so what can be checked exactly is checked
exactly here, and the rollout gates are left to prove only what they alone can:
that the physics still behaves.
"""
