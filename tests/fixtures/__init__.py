# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures: helpers, not tests.

Nothing here is collected -- ``unittest`` discovers ``test*.py``, and these are
deliberately named otherwise. What lives here is the machinery several test
modules would otherwise each grow their own copy of, which is a real cost and
not a hypothetical one: ``_teleport`` existed twice in this repo, in
a physics test module and in a bench under ``scripts/``, and the two copies had
already drifted apart in their docstrings before the shared version landed.

Three families, and they correspond to the three ways a measure can be tested
without asking a solver to reproduce a situation:

:mod:`tests.fixtures.grids`
    Synthetic geometry -- flat sheets, folded sheets, ragged sheets, chains.
    Hand-built layouts with hand-derived expected values.
:mod:`tests.fixtures.snapshots`
    Scripted contexts and probes: what a measure reads, dictated by the test
    rather than produced by physics.
:mod:`tests.fixtures.teleport`
    Putting a real scene into a chosen state without stepping it, so a test of
    an *evaluator* is not also a test of a *controller*.

The reason to invest in these at all is that a goal tree is a pure function of a
snapshot. Anything provable about goal semantics is provable here, exactly, in
milliseconds -- and what is left for the slow rollout gates is only the claim
they alone can make, that the physics still behaves.
"""
