# SPDX-License-Identifier: Apache-2.0
"""Which model the fleet's batched IK should solve on.

Newton's LM solver is the fast path, and it refuses two kinds of model
(:meth:`~robosimstudio.solvers.ik.arm_ik.ArmIK._lm_hostility`): one holding
exotic joint types, whose analytic-Jacobian kernels take the better part of an
hour to specialize, and one with more than 64 joint coordinates, where the
Cholesky tile overflows shared memory. Both limits are about the *scene*, not the
arm -- across the benchmark's suites the offending joints are a cable, a D6-jointed
connector, and bins of loose tools whose free joints alone push the count past
the ceiling, while every arm is six or seven revolute joints.

So there are three tiers, tried in order, and which one a pairing lands in is
worth reporting rather than inferring from how slow the run turned out:

``template``
    The scene's own single-world model is already fine. Nothing is built, no
    index mapping exists, and the arm coordinates are the ones everything else
    in the run already uses. Most shipped variants are here.

``reduced``
    The scene is hostile but the *robot in its place* is not: rebuild through
    :func:`_build_base` with ``dynamics=False``, which drops the task's objects,
    deformables and extras and keeps the scenery a resident robot lives inside.
    Because those all build after the robot, the arm's indices are unchanged --
    but that is checked here, not assumed, and the coordinates are paired
    explicitly either way.

``host``
    Neither model can host LM. This happens when the exotic joint is inside the
    same asset as the robot, so no amount of dropping task content removes it.
    The fleet has no batched Cartesian solve for such a pairing; the caller falls
    back to the existing host solver, per world, and pays for it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import newton
import numpy as np

from robosimstudio.solvers.ik.arm_ik import ArmIK
from robosimstudio.solvers.ik.batched import BatchedArmIK, RobotIKRig
from robosimstudio.specs import Episode

from .compiler import CompiledScene, _build_base
from .request import SceneRequest

__all__ = ["IKPlan", "build_ik_rig", "plan_ik_model"]


@dataclass
class IKPlan:
    """Where the fleet's IK will be solved, and how its indices line up.

    Attributes:
        tier: ``"template"``, ``"reduced"`` or ``"host"``.
        model: The single-world model to solve on, or ``None`` for ``host``.
        coord_map: For each of ``model``'s coordinates, the template coordinate
            it is the same joint as, or -1. ``None`` when the two are the same
            model and the map would be the identity.
        arm_coords: Per arm, its ``joint_q`` indices *in* ``model``.
        arm_dofs: Per arm, its ``joint_qd`` indices in ``model``.
        reason: Why this tier and not the faster one, for the run's provenance.
    """

    tier: str
    model: newton.Model | None = None
    coord_map: np.ndarray | None = None
    arm_coords: dict[str, list[int]] = field(default_factory=dict)
    arm_dofs: dict[str, list[int]] = field(default_factory=dict)
    reason: str = ""

    @property
    def batched(self) -> bool:
        """Whether a batched Cartesian solve is available at all."""
        return self.tier != "host"


def plan_ik_model(request: SceneRequest, episode: Episode, scene: CompiledScene) -> IKPlan:
    """Choose the tier for one pairing, building the reduced model if needed.

    Args:
        request: What the scene was built from.
        episode: The episode the scene was compiled at. The reduced build takes
            the same one so a resident robot mounts where it actually mounts.
        scene: The compiled fleet scene, for its template model and arm indices.

    Returns:
        The plan. Never raises for a hostile scene -- ``tier="host"`` is a
        supported outcome and the caller decides what to do about it.
    """
    embodiment = scene.embodiment
    template = scene.index_model
    if embodiment is None or not embodiment.arms:
        return IKPlan(tier="host", reason="the scene has no robot")

    hostile = ArmIK._lm_hostility(template)
    if not hostile:
        return IKPlan(
            tier="template",
            model=template,
            coord_map=None,
            arm_coords={name: list(index.arm_coords) for name, index in embodiment.arms.items()},
            arm_dofs={name: list(index.arm_dofs) for name, index in embodiment.arms.items()},
        )

    reduced = _build_base(request, episode, dynamics=False)
    if reduced.embodiment is None:
        return IKPlan(tier="host", reason=f"{hostile}, and the reduced build has no robot")
    model = reduced.builder.finalize()
    for entity in reduced.entities:
        entity.bind(model)

    still = ArmIK._lm_hostility(model)
    if still:
        # The exotic joint is in the same asset as the robot -- a stage export
        # holding both. Nothing this function can drop removes it.
        return IKPlan(tier="host", reason=f"{hostile}; even without the task's own content, {still}")

    names = set(embodiment.arms) & set(reduced.embodiment.arms)
    missing = set(embodiment.arms) - names
    if missing:
        return IKPlan(tier="host", reason=f"{hostile}; the reduced build is missing arm(s) {sorted(missing)}")

    coord_map = _pair_coords(scene, reduced, model)
    return IKPlan(
        tier="reduced",
        model=model,
        coord_map=coord_map,
        arm_coords={name: list(reduced.embodiment.arms[name].arm_coords) for name in names},
        arm_dofs={name: list(reduced.embodiment.arms[name].arm_dofs) for name in names},
        reason=hostile,
    )


def _pair_coords(scene: CompiledScene, reduced, model: newton.Model) -> np.ndarray:
    """Map each reduced-model coordinate to the template coordinate it *is*.

    Paired by position within each arm's own index list rather than by label:
    both lists come from the same joint-matching pass over the same asset, so
    entry *i* is the same joint in both, and pairing by name would re-derive that
    less reliably.

    Coordinates the pairing does not reach -- the scenery's own joints, which the
    arm never moves -- stay -1 and seed from the reduced model's rest values.
    """
    mapping = np.full(int(model.joint_coord_count), -1, dtype=np.int32)
    for name, template_index in scene.embodiment.arms.items():
        reduced_index = reduced.embodiment.arms.get(name)
        if reduced_index is None:
            continue
        pairs = [
            (reduced_index.arm_coords, template_index.arm_coords),
            (reduced_index.gripper_coords, template_index.gripper_coords),
            (list(reduced_index.gripper_follower_coords), list(template_index.gripper_follower_coords)),
        ]
        for source, target in pairs:
            if len(source) != len(target):
                raise ValueError(
                    f"arm {name!r}: reduced build has {len(source)} coordinates where the scene has {len(target)}; "
                    "the two builds disagree about the robot, which would make every solve land on the wrong joint"
                )
            for reduced_coord, template_coord in zip(source, target, strict=True):
                mapping[int(reduced_coord)] = int(template_coord)
    return mapping


def build_ik_rig(
    scene: CompiledScene,
    plan: IKPlan,
    *,
    num_worlds: int,
    iterations: int = 8,
    arms: Sequence[str] | None = None,
) -> RobotIKRig | None:
    """Assemble the batched rig for a plan, or ``None`` when there is no batched tier.

    Every arm of the robot gets its own masked solver over ``num_worlds``
    problems, sharing one seed buffer -- see
    :class:`~robosimstudio.solvers.ik.batched.RobotIKRig` for why the arms
    are solved in sequence rather than jointly.

    ``arms`` restricts the rig to the named arms; ``None`` takes every arm the
    plan covers. The single-world Cartesian servo builds one rig per arm at
    ``num_worlds=1`` through this same function, so its per-turn solve is the
    fleet's per-step solve -- the same solver class, the same tool frame, the
    same iteration count and rotation weight -- rather than a second solver
    that has to be kept equal by hand.
    """
    if not plan.batched or plan.model is None or scene.embodiment is None:
        return None
    wanted = None if arms is None else {str(name) for name in arms}
    worlds = scene.worlds
    coords_per_world = worlds.coords_per_world if worlds is not None else int(scene.model.joint_coord_count)
    rig = RobotIKRig(
        plan.model,
        num_worlds=num_worlds,
        coords_per_world=coords_per_world,
        coord_map=plan.coord_map,
        iterations=iterations,
    )
    # The IK model's own home posture, for the resting pose an unsteered world
    # solves to. Read off this model rather than the scene's, since a reduced
    # build has its own body indices.
    home_state = plan.model.state()
    newton.eval_fk(plan.model, plan.model.joint_q, plan.model.joint_qd, home_state)
    body_q = home_state.body_q.numpy()

    for name, template_index in scene.embodiment.arms.items():
        if name not in plan.arm_coords or (wanted is not None and name not in wanted):
            continue
        hand = scene.embodiment.hands.get(name)
        ee_offset = hand.tcp_offset if hand else (0.0, 0.0, 0.0)
        ee_offset_rot = hand.tcp_offset_rot if hand else (0.0, 0.0, 0.0, 1.0)
        ee_body = _ee_body_in(plan, scene, name, template_index)
        rig.add_arm(
            name,
            BatchedArmIK(
                plan.model,
                num_worlds=num_worlds,
                arm_coords=plan.arm_coords[name],
                arm_dofs=plan.arm_dofs[name],
                scene_coords=template_index.arm_coords,
                ee_body=ee_body,
                home_pose=_tcp_pose(body_q[ee_body], ee_offset, ee_offset_rot),
                ee_offset=ee_offset,
                ee_offset_rot=ee_offset_rot,
                iterations=iterations,
            ),
        )
    return rig


def _ee_body_in(plan: IKPlan, scene: CompiledScene, name: str, template_index) -> int:
    """The tool-centre body, in the IK model's body numbering."""
    if plan.tier == "template":
        return int(template_index.ee_body)
    # Reduced: the robot and its scenery build before anything that was dropped,
    # so body indices up to that point are unchanged -- but say so out loud, since
    # a silent mismatch here aims every solve at the wrong link.
    ee_body = int(template_index.ee_body)
    if plan.model is not None and ee_body >= int(plan.model.body_count):
        raise ValueError(
            f"arm {name!r}: tool-centre body {ee_body} is outside the reduced IK model "
            f"({plan.model.body_count} bodies); the two builds disagree about body order"
        )
    if plan.model is not None:
        want = list(scene.index_model.body_label)[ee_body]
        got = list(plan.model.body_label)[ee_body]
        if want != got:
            raise ValueError(
                f"arm {name!r}: body {ee_body} is {got!r} in the reduced IK model but {want!r} in the scene; "
                "the two builds disagree about body order"
            )
    return ee_body


def _tcp_pose(body_xform: np.ndarray, offset, offset_rot) -> tuple[np.ndarray, np.ndarray]:
    """World tool-centre pose from a body transform, the way ArmIK reads it."""
    from robosimstudio.core.math import quat_multiply, quat_to_mat

    pos = np.asarray(body_xform[:3], dtype=np.float64)
    quat = np.asarray(body_xform[3:7], dtype=np.float64)
    tcp = pos + quat_to_mat(quat) @ np.asarray(offset, dtype=np.float64)
    return tcp.astype(np.float32), quat_multiply(quat, np.asarray(offset_rot, dtype=np.float64)).astype(np.float32)
