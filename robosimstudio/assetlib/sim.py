# SPDX-License-Identifier: Apache-2.0
"""What an asset needs of physics, declared rather than assembled.

The declaration half of physics composition. An asset says what it *is* (a
domain) and what it *touches* (contact rules by role), in its own
terms; the arrangement a task stages says which solver serves each domain; and
the composer in :mod:`robosimstudio.scene.physics` derives the solver islands,
the coupling channels and the contact routing from the two. A towel says *"I am
cloth, hands have to be able to push me, and my contacts are with the surface
and the fixtures"* -- it does not say ``SolverVBD(iterations=40)``.

The types live in :mod:`robosimstudio.specs.physics`, beside the arrangement,
because a declaration is data every layer may read -- the catalog serialises it,
the composer consumes it, the studio shows it. This module is the asset
library's name for them.
"""

from __future__ import annotations

from robosimstudio.specs.physics import ContactRule, SimRequirements, contacts_with, sim_to_dict

__all__ = ["ContactRule", "SimRequirements", "contacts_with", "sim_to_dict"]
