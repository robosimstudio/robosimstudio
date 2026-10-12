# SPDX-License-Identifier: Apache-2.0
"""Tests for the asset component: declarations, and what refuses them.

No GPU. An asset declaration is data, so whether it is well formed is a question
answerable the moment it is written -- which is the whole reason the asset layer
can be a per-commit gate rather than something only a built scene can verify.

What a built scene must still prove is separate and smaller: that a resolved
selector picks the same particles the old lambda picked, and that declared
physics produces the same arrangement. Those are the golden gates, and they are
not here.
"""
