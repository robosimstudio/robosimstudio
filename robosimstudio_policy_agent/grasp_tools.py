# SPDX-License-Identifier: Apache-2.0
"""``sim.grasp_candidates``: grasps found in the mesh, not derived from a box.

**The gap this closes.** A graph's only way to ask for a grasp was
``geometry.top_down_grasp_candidates``, which is a pure function of an
:class:`OrientedBoundingBox`. That is right for a soup can and wrong for
everything with a shape. A box knows an object's extent and nothing about where
it is *thin*, so a top-down grasp derived from one aims at the middle of the
bounding volume -- which for a wrench lying flat is its mid-thickness, roughly a
centimetre below where the metal is. ``tool_hanging`` paid for that directly:
the descend ended 8.2 mm short, the jaws sat at 92-95 mm through the entire
close, and the tool was only ever picked up when the lift happened to unload
them.

Meanwhile :class:`~robosimstudio.solvers.grasp.antipodal.AntipodalSampler`
already searches the actual triangles, and has since the studio's picker needed
it. It was simply unreachable from a graph. This is that seam, and nothing more:
the sampler is unchanged, the scoring is unchanged, and the numbers it was tuned
against still hold.

**What the mesh buys, in the sampler's own words.** ``GraspCandidate.width`` is
*"what the object actually is between the pads, not what its bounding box is"*.
From that follows ``grip_command``, which closes to just inside the measured
width rather than driving the jaws shut -- the failure recorded on
``sharps_disposal/tray_pair``, where a smooth 24 mm cylinder is extruded out of a fully
closed hand mid-carry. And ``finger_depth`` refuses the grasp a box cannot see is
impossible: *"a 100 mm tall soup can proposes a beautiful 65 mm antipodal pair
across its middle, and a top-down hand asked for it presses the can into the
table instead of lifting it."*

**Why ``sim.`` and not ``geometry.``.** The split is not about the mesh. Knowing
an object's *shape* is a CAD model, which a real cell has for the parts it was
built to handle; that is not privileged in the sense
:mod:`~robosimstudio_policy_agent.contract` means. What is privileged is that
this reads the object's live **pose** out of the simulator, and reads it by
name. So it sits with the other ground-truth tools and is withheld under
``--no-privileged``, and the honest non-privileged twin is the same sampler run
over a reconstructed cloud -- the same substitutability the perception
subgraphs already have, at a different layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.arms import arm_name
from robosimstudio.core.bus import Bus
from robosimstudio.core.math import quat_xyzw_to_wxyz
from robosimstudio.core.streams import reading

from .refusal import ToolRefusal

__all__ = ["GraspTools", "register_grasp_tools"]

DEFAULT_CANDIDATES = 8
"""How many grasps a call returns by default.

The sampler's own default. The pair search is quadratic in its seeds rather
than in this, so asking for more is cheap; asking for *fewer* does not make it
faster.
"""


def register_grasp_tools(registry, env: Any, bus: Bus) -> None:
    """Register ``sim.grasp_candidates``. Privileged: it reads a live pose."""
    tools = GraspTools(env, bus)
    for name, fn, summary in tools.entries():
        registry.register_callable(name, fn, summary=summary, tags=("privileged", "planning"))


class GraspTools:
    """The mesh-searching grasp proposer, as a tool."""

    def __init__(self, env: Any, bus: Bus) -> None:
        self.env = env
        self.bus = bus
        self._samplers: dict[str, Any] = {}

    def entries(self) -> list[tuple[str, Any, str]]:
        return [
            (
                "sim.grasp_candidates",
                self.grasp_candidates,
                "Antipodal grasps found by searching an object's mesh, best first.",
            )
        ]

    def _sampler(self, arm: str) -> Any:
        """One sampler per hand, built once.

        Per *hand*, not per scene: the sampler is parameterised by the span,
        the approach and closing axes and the grasp-frame rotation, all of which
        belong to whichever gripper is on this arm. A bimanual robot with two
        different hands would otherwise propose one hand's grasps to the other.

        **Why ``ignore_shapes=handle.shapes`` is passed.**
        :meth:`AntipodalSampler._obstacles` boxes everything that is not the
        body being grasped, and that includes the hand doing the grasping. Both
        of its obstacle tests then run into it: the stand-off cast runs
        *backwards* from the grasp, so for a top-down candidate it runs upwards
        into whatever the arm is hovering with, and the pad-volume test's column
        runs the same way. Measured on ``astro/wiring_sort``, the wire spool
        offers 3 passing candidates of 17 with the arm at home and **0 of 16**
        with the hand a few centimetres above it, refusal empty -- every
        vertical grasp lost, and not to anything in the scene. The hand that is
        going to fly the approach is the thing that moves, so it is not an
        obstacle to it.

        The exclusion was admitting candidates nothing was rejecting.
        :meth:`AntipodalSampler._pads_clear` is what rejects them. A side grasp
        on the spool puts a plate where the breadboard 6 mm away is, and it is
        now refused on the geometry rather than surviving to be drawn by
        lottery. Re-measured against the evaluator with the pad test in place,
        five episodes each and one keyword apart, everything else identical::

            reward     with:    0.50 0.00 0.50 0.25 0.25    mean 0.300
                    without:    0.25 0.25 0.00 0.25 0.50    mean 0.250
            contained  with: 6 of 20 goals      without: 5 of 20

        A tie, on a suite whose own spread over five identical runs is wider
        than that -- so this is no longer a *measured* choice and the honest
        reading is that the exclusion has stopped costing anything. It is wired
        because the reason for it stands on its own: a hand is not an obstacle
        to itself, and leaving it in the set throws away every vertical grasp on
        an object the arm happens to be hovering over. Unwiring it is one
        keyword and the numbers above are what it would cost.
        """
        cached = self._samplers.get(arm)
        if cached is not None:
            return cached

        from robosimstudio.solvers import AntipodalSampler

        handle = self.env.handles.arms[arm]
        gripper = self.env.scene.embodiment.grippers.get(arm)
        if gripper is None:
            raise ToolRefusal(f"arm {arm!r} carries no gripper, so it has nothing to grasp with")
        # The *measured* span off the live gripper, never a spec default: the
        # sampler's own docstring warns that a geared hand's span is not its
        # joint travel, and the span is what decides which pairs fit at all.
        sampler = AntipodalSampler(
            self.env.model,
            span=float(handle.gripper_span),
            approach_axis=tuple(handle.approach_axis),
            close_axis=tuple(handle.close_axis),
            ee_offset_rot=tuple(handle.ee_offset_rot),
            # The WHOLE robot, deliberately -- not just this arm. The measured
            # rationale above is about the sampler's own hand, but the arm
            # handle's `shapes` is now honestly per-arm (it used to be the
            # whole robot on every handle), and narrowing this exclusion with
            # it would silently turn the other arm into a grasp obstacle on
            # suites whose arms are collision-filtered against each other.
            ignore_shapes=tuple(int(s) for s in np.asarray(self.env.handles.robot_shapes).reshape(-1)),
            # Template width, so a world of a wave searches its own geometry.
            # `scene.worlds` is `None` on a single-world scene, which is the
            # sampler's "all of them" and its old behaviour exactly.
            shapes=int(getattr(getattr(self.env.scene, "worlds", None), "shapes_per_world", 0) or 0),
        )
        self._samplers[arm] = sampler
        return sampler

    def grasp_candidates(
        self,
        object_name: str,
        arm_id: int = 0,
        role: str = "",
        count: int = DEFAULT_CANDIDATES,
        near: list[float] | None = None,
    ) -> dict[str, Any]:
        """Grasps on one object, best first, found in its triangles.

        Args:
            object_name: The object to grasp, as the scene names it.
            arm_id: Which hand is going to take it -- the span and the finger
                geometry are that hand's, so a candidate is only valid for the
                arm it was proposed for.
            role: ``"work"`` or ``"support"``, which wins over *arm_id* when
                given.
            count: How many to return.
            near: Optional world point to search around, for an object whose
                far end is the one that matters. ``None`` searches all of it.

        Returns:
            ``{"candidates": [...], "refusal": str}``. An empty list is not an
            error -- it means no opposed pair fits this hand, and ``refusal``
            says which of the reasons it was in words, because "no grasp" and
            "no grasp *because the span is too narrow*" send a caller to
            different fixes.

            Each candidate carries ``position`` and ``rotation`` as the runtime's
            ``Se3Pose`` halves, plus ``width`` (the measured separation, which is
            what ``grip`` was computed from), ``cone_angle``, ``clearance`` and
            ``score``. ``grip`` is the 0..1 command that closes to just inside
            the object rather than driving the pads together.
        """
        name = self._arm(int(arm_id), str(role))
        handles = self.env.handles
        try:
            body = handles.entity(object_name).body
        except KeyError:
            raise ToolRefusal(
                f"no object {object_name!r}; objects {sorted(handles.objects)}, statics {sorted(handles.statics)}"
            ) from None
        if body < 0:
            raise ToolRefusal(f"{object_name!r} has no body to search; it may be a bodiless static")

        def run() -> dict[str, Any]:
            sampler = self._sampler(name)
            span = float(self.env.handles.arms[name].gripper_span)
            with reading():
                body_q = np.asarray(self.env.state.body_q.numpy(), dtype=np.float64)
            found = sampler.propose(body, body_q, near=None if near is None else list(near), k=int(count))
            return {
                "candidates": [_candidate(c, span) for c in found],
                # Carried even on success: the sampler sets it whenever it
                # rejected everything of some kind, and a caller taking a weak
                # grasp benefits from knowing what was thrown away.
                "refusal": str(getattr(sampler, "last_refusal", "") or ""),
            }

        return self.bus.call(run)

    def _arm(self, arm_id: int, role: str) -> str:
        """The arm a call means. ``role`` wins over ``arm_id`` when given."""
        handles = self.env.handles
        if role:
            if role not in ("work", "support"):
                raise ValueError(f"role must be 'work' or 'support', not {role!r}")
            chosen = handles.work_arm if role == "work" else handles.support_arm
            if not chosen:
                raise ToolRefusal(f"this scene has no {role!r} arm; arms are {sorted(handles.arms)}")
            return chosen
        return arm_name(self.env, arm_id)


def _candidate(candidate: Any, span: float) -> dict[str, Any]:
    """One ``GraspCandidate`` in the wire shape: plain floats, wxyz rotation.

    ``sim_tools.py``'s rule -- a tool that hands back a numpy array serialises
    in process and fails the moment the same graph is pointed at an RPC bundle.
    """
    position = np.asarray(candidate.position, dtype=np.float64).reshape(3)
    rotation = quat_xyzw_to_wxyz(np.asarray(candidate.quat, dtype=np.float64).reshape(4))
    return {
        "position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
        "rotation": {
            "w": float(rotation[0]),
            "x": float(rotation[1]),
            "y": float(rotation[2]),
            "z": float(rotation[3]),
        },
        "width": float(candidate.width),
        "grip": float(candidate.grip_command(span)),
        "cone_angle": float(candidate.cone_angle),
        "clearance": float(candidate.clearance),
        # What the object is across the closing axis *here*, so a caller can ask
        # whether this pair is on the outside of it or inside a recess without
        # reaching for a bounding box -- which answers a different question on
        # anything longer than it is wide. See GraspCandidate.local_width.
        "local_width": float(candidate.local_width),
        "score": float(candidate.score),
        "top_down": bool(candidate.top_down),
    }
