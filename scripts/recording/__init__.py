# SPDX-License-Identifier: Apache-2.0
"""The example-clip recorder behind ``scripts/record_examples.py``.

:mod:`.takes` is the table of clips and :mod:`.beats` its vocabulary; both import
nothing heavy, so a test can read them. :mod:`.lockstep` puts an example's page
on a film clock, :mod:`.page` drives it in a headless Chromium on the GPU, and
:mod:`.encode` writes the clip, its contact sheet and its timeline.
"""
