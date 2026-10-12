# SPDX-License-Identifier: Apache-2.0
"""What kind of state the engine integrates for a thing.

The substrate is the **only** type an object must have, and it is not a taxonomy:
it is a statement about physics. A rigid body's state is one pose; a drawer's is
a base pose plus joint coordinates; a shirt's is particles plus the named sets
resolved on its rest layout. Everything else about an object -- what it affords, what it is called,
what can be done to it -- lives in features and handlers, which are open sets
attached per object.

There is deliberately **no capability hierarchy** here. No ``Graspable``, no
``Containing``, no ``Openable``. An object affords grasping because it *has* a
grasp box, a chain segment or a pinchable particle set, and the ``holding``
handler finds its evidence in whichever it has. That is Python's model -- an
object is a bag of attributes and handlers, and shared structure is discovered
after the fact rather than declared in advance -- and it is the model this layer
implements. Categories like "scissors" never generalise; geometric feature kinds
like ``loop`` almost always do.

These protocols are :func:`~typing.runtime_checkable` so a handler can ask what a
thing offers rather than what it *is*, which is the same duck-typed question the
compiler already asks of entities. They describe **state access only** -- the
minimum a handler needs to read the world -- and none of them says anything about
meaning.

The layouts here are the ones the library's own handlers take. A layout that
belongs to one kind of thing -- a chain with an arc-length parameterisation, a
volume -- is minted by that kind's domain pack, and so is the protocol that
describes it.

Embodiments are **not** assets and appear nowhere here. A hand is its own axis:
arms and grippers are registered through ``ArmSpec``/``GripperSpec``, their models
and calibration belong to that layer, and they enter this one only as the *hand
side* of a handler -- finger contact groups, jaw stall evidence, reach bands.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "SUBSTRATES",
    "Jointed",
    "Particled",
    "Posed",
    "PublishesState",
    "is_substrate",
    "layout_chain",
    "reads_as",
]

# The one definition lives with the IR, beside the domains that name it: a
# substrate is what a *measure* sees, a domain is what physics integrates, and
# `DomainInfo.reads` is the arrow between them. Re-exported here because this is
# where handlers look for it.
from robosimstudio.specs.physics import DOMAINS, SUBSTRATES, layout_chain, reads_as


def is_substrate(name: str) -> bool:
    """Whether ``name`` is a state layout something has registered.

    The set is open now -- a domain mints its own layout by declaring it -- so
    this answers "has anybody declared this" rather than "is this one of four".
    That is the question every caller was really asking: the check exists to
    catch a *typo*, and a typo is still a name nothing registered.

    **The domains are loaded first**, because "nobody declared this" is only an
    answer once everybody has had the chance to: a layout that belongs to one
    kind is minted when that kind's domain registers, and without the scan the
    answer for it would depend on whether anything had imported its pack yet.
    """
    DOMAINS.load()
    return name in SUBSTRATES


@runtime_checkable
class Posed(Protocol):
    """Something with a world pose: one body, or the base of a composite."""

    def pose(self, body_q: np.ndarray) -> np.ndarray:
        """World pose as ``(px, py, pz, qx, qy, qz, qw)``."""
        ...

    def position(self, body_q: np.ndarray) -> np.ndarray:
        """World position [m]."""
        ...


@runtime_checkable
class Jointed(Protocol):
    """Something with joint coordinates -- a drawer, a door, a lid.

    ``coordinate`` is the raw generalised coordinate and ``limits`` its range, so
    an ``open_fraction`` handler can normalise without knowing what kind of joint
    it is. A prismatic drawer and a revolute door differ in units, not in the
    shape of the question.
    """

    def coordinate(self, joint_q: np.ndarray) -> float:
        """The joint's generalised coordinate now [m or rad]."""
        ...

    @property
    def limits(self) -> tuple[float, float]:
        """Lower and upper travel [m or rad]."""
        ...


@runtime_checkable
class Particled(Protocol):
    """Something made of particles, with named sets resolved on its rest layout.

    The rest-layout rule is the load-bearing part and it is not new: the cloth
    entity already resolves its selectors "once, on the flat as-built sheet",
    because "a set recorded from the settled drape would follow whatever the drape
    did, and 'the tab' would stop being the tab the moment the cloth moved". A
    deformable feature is a *material-frame* selection carried through the live
    configuration -- never a world-frame box.
    """

    def indices(self, name: str = "") -> np.ndarray:
        """Global particle indices of one named set, or of the whole body."""
        ...

    def points(self, name: str = "") -> np.ndarray:
        """Live world positions of one named set ``[n, 3]``."""
        ...

    @property
    def baseline(self) -> np.ndarray | None:
        """The settled rest layout every material-frame measure is stated against."""
        ...


@runtime_checkable
class PublishesState(Protocol):
    """Something exposing internal state under declared, typed names.

    The grown-up form of the ``extras`` dict a task entity publishes today. A tape
    knows its bond fraction and nothing else can compute it; the difference this
    protocol makes is that the name, the type and the *observability tier* are
    declared rather than discovered by reaching into an array.

    Tiers are the roadmap for de-privileging perception, member by member:

    ``internal``
        Sim-side only. Evaluators and ``describe`` may read it; the bridge may not
        carry it.
    ``measurable``
        A privileged tool may expose it -- a real rig could instrument this.
    ``perceivable``
        A camera could estimate it, so a non-privileged policy may legitimately
        be given it.

    Nothing here weakens the privilege boundary: the allowlist still decides what
    crosses the bridge. This only lets the allowlist be stated per member instead
    of per tool.
    """

    def published(self) -> dict[str, Any]:
        """Current values of every declared published state, by name."""
        ...
