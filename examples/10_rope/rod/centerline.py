# SPDX-License-Identifier: Apache-2.0
"""``Centerline``: **I am a curve.** The capability this domain earned, and publishes.

A domain answers one question and is quietly asked two. ``rod`` picks the solver
(a cable chain wants a soft solver, not MuJoCo) **and** decides what the thing is
to everybody else (which measures bind, which actor a foreign engine is handed,
whether the hold latch will touch it). Those are different questions, and
welding them is why a cable could only ever be a ``rod``:

* a cable simulated as a **tetrahedral body** is ``domain="soft"``, so it routes
  to SuperDex or Genesis correctly -- and is then invisible to ``winding_turns``,
  to ``straightness`` and to the grasp latch, because all three are bound to the
  rod domain;
* a cable simulated as a **rolled shell** is ``domain="cloth"``, with the same
  result.

:class:`Centerline` is the answer: which geometry a thing can show, and therefore
which measures and which affordances it gets, whatever its domain. One abstract
method, everything else derived from it. A thing may have more than one
capability, and that is the point: a tetrahedral cable would declare a
``Volume`` so a foreign engine builds it out of tetrahedra, and
:class:`Centerline` so the rod measures and the hold still reach it.

**It is here because this domain earned it.** It has two implementations that
disagree about everything internal --
:class:`~robosimstudio.domains.rod.entity.EntityRod` reads capsule poses out of
``body_q``, :class:`~robosimstudio.domains.rod.superdex.EntityRodSuperdex` reads
nodes out of ``particle_q`` -- and it derives real behaviour from the one method
they share. A ``Surface`` and a ``Volume`` are the obvious next two and are
deliberately *not* written: each would have one implementer and zero
implementers respectively, and would derive nothing.

**Published, not private.** :func:`~robosimstudio.domains.capabilities.register_capability`
makes it importable as ``robosimstudio.domains.capabilities.Centerline``, which
is the name an entity in another domain -- the tetrahedral cable above -- declares
it by, without that entity's module naming this pack.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from robosimstudio.domains.capabilities import register_capability

__all__ = ["Centerline"]


@register_capability
class Centerline:
    """**I am a curve.** One method; the rest follows.

    An implementation says where its nodes are, in order, at a given instant.
    Everything a curve can be asked -- its arc length, its sag, how sharply it
    bends, where it is for a measure, what a foreign engine's ``rod`` actor
    reader should cut -- is a statement about that sequence and about nothing
    else. Which is why the two shipped rod implementations already agreed on the
    accessor's *name* and shape before this class existed: ``EntityRod`` reads
    capsule centres out of ``body_q`` and ``EntityRodSuperdex`` reads nodes out
    of ``particle_q``, and neither of those is the substrate.
    """

    forms: ClassVar[tuple[str, ...]] = ("points",)
    """An ordered centerline, which is the whole content of this layout."""

    layouts: ClassVar[tuple[str, ...]] = ("rod",)
    """A curve reads as ``rod`` -- whatever its domain says. This is the line
    that lets a soft-domain cable meet a rod measure."""

    def centerline(self, snapshot: Any) -> np.ndarray:
        """``(n, 3)`` node positions, in order tip to tip, **at this instant**.

        The instant is passed rather than read off the model, and that is not
        decoration: the model's own array is the *build-time* layout and never
        moves, so a cable that has fallen half a metre reports itself still
        hanging where it was declared -- a plausible number and completely wrong.
        """
        raise NotImplementedError(f"{type(self).__name__} declares Centerline and must implement centerline()")

    def substrate_points(self, snapshot: Any) -> np.ndarray | None:
        """This curve's ordered points, for any measure over the ``rod`` layout.

        Derived, so an implementation states where its nodes are once and does
        not also have to know that a measure calls the accessor something else.
        """
        points = self.centerline(snapshot)
        if points is None or not len(points):
            return None
        return np.asarray(points, dtype=np.float64)

    # ------------------------------------------------------------- holding
    HOLD_REACH = 0.05
    """How near a hand's tool-centre point a node must be to be taken [m]."""

    def hold(
        self, on: bool, state: Any = None, *, arm: str = "right", slot: int = 0, at: int | None = None, world: int = 0
    ) -> int:
        """Take the curve in ``arm``'s jaws so it rides the hand, or let it go.

        **The affordance every implementation owes, dispatched per storage.** A
        capsule chain is held by the body latch -- a kinematic coupling in a
        ``rigid_force`` kernel; a particle rod on a foreign engine is held by
        that engine's own node constraint, carried on the hand through every
        engine substep. Neither is the substrate; both answer this.

        Args:
            on: Take, or let go.
            state: The instant, for the picker. Unused when letting go.
            arm: Which hand.
            slot: Which of two holds -- ``0`` the working hand, ``1`` an anchor.
            at: A node (or segment) index along the curve to take, instead of
                the one nearest the jaws. For a caller that already knows.

        Returns:
            The index taken along the curve, or ``-1`` -- let go, or nothing
            within :attr:`HOLD_REACH` of the jaws.
        """
        raise NotImplementedError(f"{type(self).__name__} declares Centerline and must implement hold()")

    def hand_tcp(self, arm: str, body_q: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        """``(grip body, hand position, hand quaternion xyzw)`` for ``arm``, off the embodiment.

        Read from the hand's own declaration -- ``tcp_offset`` on the gripper,
        ``ee_body`` on the arm -- so no implementation carries a copy of how
        a hand is found. Raises by name on a scene with no such arm.
        """
        embodiment = getattr(self, "embodiment", None)
        if embodiment is None:
            raise LookupError(
                f"{type(self).__name__} {getattr(self, 'name', '?')!r}: no robot in this scene to hold with"
            )
        arms = embodiment.arms
        if arm not in arms:
            raise LookupError(f"no arm {arm!r}; this robot has {sorted(arms)}")
        grip = int(arms[arm].ee_body)
        hand = embodiment.hands.get(arm)
        offset = np.asarray(hand.tcp_offset if hand is not None else (0.0, 0.0, 0.0), dtype=np.float64)
        from robosimstudio.core.math import quat_to_mat

        q = np.asarray(body_q[grip], dtype=np.float64)
        pos = q[:3] + quat_to_mat(q[3:7]) @ offset
        return grip, pos, q[3:7]

    def nearest_to(self, points: np.ndarray, target: np.ndarray, *, at: int | None = None) -> int:
        """Which point along the curve to take: ``at``, or the nearest within reach, or ``-1``."""
        if at is not None:
            return int(at) if 0 <= int(at) < len(points) else -1
        if not len(points):
            return -1
        distances = np.linalg.norm(np.asarray(points, dtype=np.float64) - np.asarray(target, dtype=np.float64), axis=1)
        index = int(np.argmin(distances))
        return index if float(distances[index]) <= self.HOLD_REACH else -1

    def curve_states(self, snapshot: Any) -> dict[str, float]:
        """Arc length, sag and the sharpest bend -- what a curve is, as numbers.

        Offered rather than published: an implementation folds these into its own
        :meth:`~robosimstudio.scene.entity.Entity.published_states` if it wants
        them, because what a *task* scores is the task's business.
        """
        points = self.centerline(snapshot)
        if points is None or len(points) < 2:
            return {}
        pts = np.asarray(points, dtype=np.float64)
        spans = np.diff(pts, axis=0)
        length = float(np.linalg.norm(spans, axis=1).sum())
        sag = float(pts[:, 2].max() - pts[:, 2].min())
        out = {"length": length, "sag": sag}
        if len(pts) > 2:
            # Menger curvature over each consecutive triple: the radius of the
            # circle through three points. The smallest radius on the curve is
            # the sharpest bend, which is what a kink is.
            a, b, c = pts[:-2], pts[1:-1], pts[2:]
            ab, bc, ca = b - a, c - b, a - c
            # |ab x bc| is *twice* the triangle's area, so the circumradius
            # abc/(4*area) is abc/(2*|ab x bc|). Writing the 4 here instead
            # reported every bend at half its real radius.
            cross = np.linalg.norm(np.cross(ab, bc), axis=1)
            denom = 2.0 * np.maximum(cross, 1.0e-12)
            radius = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ca, axis=1) / denom
            out["min_bend_radius"] = float(radius[cross > 1.0e-12].min()) if np.any(cross > 1.0e-12) else float("inf")
        return out
