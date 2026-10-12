# SPDX-License-Identifier: Apache-2.0
"""The one error composition raises."""

from __future__ import annotations

__all__ = ["CompositionError", "MaterialLossWarning"]


class CompositionError(ValueError):
    """The arrangement cannot honour what the assets declared, and the message says which.

    Raised at compile time, naming the asset, the rule and the arrangement --
    the same contract a catalog gap has: a scene that cannot be composed as
    declared is refused where it can still be named, never built wrong.
    """


class MaterialLossWarning(UserWarning):
    """A material statement that will not reach the engine it was routed to.

    A warning and not an error, on purpose: both shipped foreign engines drop
    something every cloth states, so an error would refuse every cloth scene
    the day it landed. What matters is that a *person* learns which knob is
    dead -- and that a test can assert it was said.
    """
