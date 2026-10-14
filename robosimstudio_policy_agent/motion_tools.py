# SPDX-License-Identifier: Apache-2.0
"""``motion.*``: asking for a *plan*, which ``robot.*`` cannot express.

Without these the planner is installed and unreachable. ``robot.solve_ik`` and
``robot.go_to_pose`` route to the per-step solver by design -- see
:class:`~robosimstudio_policy_agent.ik.IKBackend`,
which keeps ``solve_ik`` cheap because ``CartesianServo`` calls it up to 400
times a move -- so a graph that wants a collision-aware path, or a full-SE(3)
one, has no way to say so. These are that way.

**Unprivileged, and registered beside ``robot.*``.** Nothing here reads ground
truth: a plan is a question about the robot and a world the caller supplies.
That matters because the privilege boundary is what lets a graph cross to a real
rig, and a graph forced to become privileged-only in order to *plan* would be
privileged-only for a reason having nothing to do with knowing where the hammer
is. The one tool that would read the simulator -- building a collision world
from body poses -- is deliberately absent until it can be written with a
non-privileged twin, per the oracle/vision split the perception subgraphs use.

**What a graph gains that it did not have.** ``robot.solve_ik`` constrains
position and an approach axis, leaving wrist roll free, which
``tool_sort``'s ``post_upright.py`` records as the reason it cannot stand a tool
on end: *"a roll about the approach axis is exactly what standing a tool on end
is, so the component that matters here is the one the solver discards."*
``motion.plan_to_pose`` solves the full pose, so that goal becomes expressible
-- on an embodiment with a planner model, which today is the Panda alone. And
``orientation="lock"`` gives the same authority to a *leg*, on every robot,
which is what a pinch across a film's faces needs and what four suite-private
planners each implemented for themselves.

**Two primitives, because a robot has two kinds of move.** ``plan_linear`` pins
every point on the way and is what manipulation wants; ``plan_joint`` pins only
the endpoints and is what transit wants. The distinction is measured rather than
stylistic -- on ``tape_peel``'s YAM the leg from home to the tab's pre-grasp is
162 mm out of reach as a straight line and exact as a posture ease, while the
two legs at the tab are exact as lines. A surface offering only ``plan_linear``
is a surface every deformable suite has to route around.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["MotionTools", "register_motion_tools"]


def register_motion_tools(registry, backend: Any) -> None:
    """Register whichever of ``motion.*`` this *backend* can actually answer.

    Per tool, not all-or-nothing, and the split is the difference between a
    capability that needs a planner model and one that does not.

    ``motion.plan_to_pose`` is collision-aware free-space planning and needs
    cuRobo behind it, so it stays gated: a registry advertising it on a robot
    with no planner model would hand a graph a tool that always returns
    ``planned: false``, which reads as "this pose is unreachable" rather than
    "this robot has no planner".

    ``motion.plan_linear`` and ``motion.plan_joint`` need only an IK backend,
    which every embodiment has. Withholding them until a planner arrives is what
    left the corpus with four suite-private ``sim.plan_*`` tools computing legs
    the same two ways -- each privileged, each unportable, and each one a copy of
    arithmetic that belongs to the robot rather than to the task. What a caller
    is entitled to know is *which* it got, and ``motion.describe_planner``
    answers that with ``world_aware``.
    """
    tools = MotionTools(backend)
    world_aware = bool(getattr(backend, "supports_world_aware_plan", False))
    for name, fn, summary in tools.entries():
        if name == "motion.plan_to_pose" and not world_aware:
            continue
        required = _BACKEND_METHOD.get(name, "")
        if required and not hasattr(backend, required):
            continue
        registry.register_callable(name, fn, summary=summary, tags=("planning",))


_BACKEND_METHOD = {
    "motion.plan_support_disengagement": "plan_linear",
    "motion.plan_linear": "plan_linear",
    "motion.plan_joint": "plan_joint",
    "motion.plan_to_pose": "plan_to_pose",
    "motion.get_robot_collision_spheres": "robot_collision_spheres",
    "motion.build_world_tsdf": "build_world_tsdf",
    "motion.solve_ik_batch": "solve_ik_batch",
}
"""What each tool needs the backend to implement. ``describe_planner`` needs
nothing, which is why it is absent: a graph must always be able to ask."""


class MotionTools:
    """The planning surface, over whatever backend the connector installed."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    @staticmethod
    def _contact_kwargs(fn: Any, contact: dict[str, Any], kwargs: dict[str, Any], asked_by: Any = None) -> dict | None:
        """Fold the collision-aware arguments into *kwargs*, or refuse.

        These three are only meaningful to a collision-aware backend. They used
        to be passed unconditionally, so on the plain solver -- the default
        on every embodiment that declares no planner model -- the call died with
        ``IKBackend.plan_linear() got an unexpected keyword argument
        'allow_start_contact'``. A ``TypeError`` from inside a documented tool
        is the worst shape this can take: it is not a refusal a subgraph can
        route, it names an internal class rather than the caller's mistake, and
        it kills the episode after the arm has already moved. Twelve at a time,
        in the run that found it.

        Two cases, and they are genuinely different. If the caller left all
        three at their defaults it is asking for an ordinary straight line, so
        the arguments are simply not forwarded. If the caller *set* one, it
        asked for something this backend cannot do, and a plan that quietly
        ignored the request would report ``planned: true`` for a path that was
        never checked against anything -- so that is a refusal, in the shape
        :mod:`~robosimstudio_policy_agent.motion_tools` refuses in everywhere else.

        Only what was *asked* is forwarded, never the defaults, even to a
        backend whose signature would take them. That is not tidiness: the
        backend a planner run installs is
        :class:`~robosimstudio_policy_agent.ik.IKBackend`,
        whose ``plan_linear`` is ``(*args, **kwargs)`` -- it accepts anything
        and passes it on, to the planner first and to
        ``IKBackend.plan_linear`` when the planner finds no path. Handing
        that wrapper three defaults nobody asked for puts the TypeError above
        back, one layer down, on the fallback leg.
        """
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            params = {}
        accepted = any(p.kind is p.VAR_KEYWORD for p in params.values())
        # "The caller asked for it" means a value different from *this tool's own
        # default*, not different from a hardcoded zero. `contact_margin`
        # defaults to 0.005, so comparing against zero read every ordinary call
        # as a request for collision-aware planning and refused it -- which
        # would have broken every straight-line move on the default planner,
        # far worse than the TypeError this replaced. Found by the surface
        # smoke before it reached a run, which is the whole reason that smoke
        # exists (scripts/smoke_tool_surface.py in the study harness).
        defaults = {}
        if asked_by is not None:
            try:
                defaults = {name: param.default for name, param in inspect.signature(asked_by).parameters.items()}
            except (TypeError, ValueError):
                defaults = {}
        asked = {name: value for name, value in contact.items() if name not in defaults or value != defaults[name]}
        missing = [name for name in asked if not accepted and name not in params]
        if missing and defaults:
            return {
                "planned": False,
                "trajectory": None,
                "position_error_m": None,
                "reason": (
                    f"this run's planner is not collision-aware, so "
                    f"{', '.join(sorted(missing))} cannot be honoured. Ask "
                    f"motion.describe_planner what backend is installed; plan the move "
                    f"without contact arguments, or check the path yourself with "
                    f"sim.clearance before executing it."
                ),
            }
        # No signature to compare against, so "asked for" cannot be told from
        # "left alone" -- drop what the backend has no parameter for rather than
        # refuse a caller who may never have asked for it.
        asked = {name: value for name, value in asked.items() if name not in missing}
        kwargs.update(asked)
        return None

    def entries(self) -> list[tuple[str, Any, str]]:
        return [
            (
                "motion.plan_support_disengagement",
                self.plan_support_disengagement,
                "Plan a short grasp-preserving move away from expected support contact.",
            ),
            (
                "motion.plan_linear",
                self.plan_linear,
                "Plan a straight line to a pose, holding the axes the move is not along.",
            ),
            (
                "motion.plan_joint",
                self.plan_joint,
                "Plan an eased joint-space move to a pose, and say how near it got.",
            ),
            (
                "motion.plan_to_pose",
                self.plan_to_pose,
                "Plan a collision-aware free-space path to a full pose, wrist roll included.",
            ),
            (
                "motion.describe_planner",
                self.describe_planner,
                "What this robot's planner is, and what it was measured against.",
            ),
            (
                "motion.get_robot_collision_spheres",
                self.get_robot_collision_spheres,
                "Get live robot-model collision spheres for RGB-D self-removal.",
            ),
            (
                "motion.build_world_tsdf",
                self.build_world_tsdf,
                "Integrate RGB-D into an on-device ESDF voxel collision world.",
            ),
            (
                "motion.solve_ik_batch",
                self.solve_ik_batch,
                "IK feasibility for many candidate poses at once, from one shared seed.",
            ),
        ]

    def plan_support_disengagement(
        self,
        attached_object: dict,
        support_normal: dict | None = None,
        margin: float = 0.04,
        arm_id: int = 0,
        num_waypoints: int = 0,
    ) -> dict[str, Any]:
        """Plan the contact-transition immediately after grasping a supported object.

        The move preserves TCP orientation and travels along the supplied
        support normal. Its distance is derived from the observed attachment
        sphere thickness plus a general planning-clearance margin, so callers
        do not author an object-specific lift.  The margin leaves the planner
        a meaningful free-space start instead of merely breaking table contact.
        Collision-world checking begins on the following motion, after this
        expected initial contact has been cleared.
        """
        import math

        normal = support_normal or {"x": 0.0, "y": 0.0, "z": 1.0}
        direction = [float(normal[key]) for key in ("x", "y", "z")]
        length = math.sqrt(sum(value * value for value in direction))
        if length <= 1.0e-9:
            raise ValueError("support_normal must be non-zero")
        direction = [value / length for value in direction]
        spheres = list((attached_object or {}).get("spheres") or [])
        radius = max(
            (float(sphere.get("radius", 0.0)) for sphere in spheres),
            default=0.0,
        )
        distance = max(0.005, 2.0 * radius + float(margin))
        start = self._here(int(arm_id))
        end = {
            "position": dict(start["position"]),
            "rotation": dict(start["rotation"]),
        }
        for key, component in zip(("x", "y", "z"), direction, strict=True):
            end["position"][key] = float(end["position"][key]) + distance * component
        plan = self.backend.plan_linear(
            start,
            end,
            arm_id=int(arm_id),
            num_waypoints=int(num_waypoints),
            hold_rotation=True,
        )
        result = _result(plan, "support-disengagement trajectory was unreachable")
        result["target_pose"] = end
        result["distance"] = distance
        return result

    def get_robot_collision_spheres(self, arm_id: int = -1) -> dict[str, Any]:
        """Robot proprioceptive geometry in the shared world frame."""
        return {"spheres": list(self.backend.robot_collision_spheres(arm_id=int(arm_id)))}

    def solve_ik_batch(
        self,
        poses: list[dict],
        arm_id: int = 0,
        seed_joints: list | None = None,
        attached_object: dict | None = None,
    ) -> dict[str, Any]:
        """IK feasibility for many candidate poses at once, from one shared seed.

        A search over alternative goals -- a heading grid, a set of grasp
        candidates -- that only needs each one's reachability and resulting
        joint configuration (not an executable trajectory) gets all of them
        from one GPU call instead of one call per candidate. Measured 3.2x
        over 12 candidates against a freshly built single-goal planner, and
        the gap widens with more candidates -- see the cuRobo backend's
        ``solve_ik_batch`` for the benchmark and why this needs its own
        cached planner rather than reusing the single-goal one.

        Returns ``{"results": [...]}``, one entry per input pose in order:
        ``None`` where IK found no solution, else ``{"joint_config": [...]}``.
        """
        results = self.backend.solve_ik_batch(
            list(poses),
            arm_id=int(arm_id),
            **self._accepted(
                self.backend.solve_ik_batch,
                {"seed_joints": seed_joints, "attached_object": attached_object},
            ),
        )
        return {"results": list(results)}

    def build_world_tsdf(
        self,
        cameras: list,
        exclude_masks: list | None = None,
        voxel_size: float = 0.01,
        clear_cylinders: list | None = None,
        extent: list | None = None,
        center: list | None = None,
        arm_id: int = -1,
    ) -> dict[str, Any]:
        """Build the planner's collision world by TSDF integration on device.

        The grid stays on the backend; a later ``motion.plan_*`` call reaches
        it by passing ``world_config={"use_tsdf": true}``. See the cuRobo
        backend's ``build_world_tsdf`` for the frame and carving semantics.

        ``extent``/``center`` (world-frame metres, ``[x, y, z]``) override the
        backend's own default box. A caller whose task needs the grid to
        reach higher or wider than that generic default -- rather than
        colliding with its own outer wall, which reads exactly like an
        obstacle to the planner -- says so here instead of everyone inheriting
        a bigger default box.
        """
        kwargs: dict[str, Any] = {
            "cameras": list(cameras),
            "exclude_masks": list(exclude_masks or []),
            "voxel_size": float(voxel_size),
            "clear_cylinders": list(clear_cylinders or []),
            "arm_id": int(arm_id),
        }
        if extent is not None:
            kwargs["extent"] = tuple(float(v) for v in extent)
        if center is not None:
            kwargs["center"] = tuple(float(v) for v in center)
        return self.backend.build_world_tsdf(**self._accepted(self.backend.build_world_tsdf, kwargs))

    def plan_linear(
        self,
        end: dict,
        start: dict | None = None,
        arm_id: int = 0,
        num_waypoints: int = 0,
        axis: str = "",
        orientation: str = "axis",
        tolerance: float = 0.0,
        world_config: dict | None = None,
        attached_object: dict | None = None,
        allow_start_contact: bool = False,
        allow_goal_contact: bool = False,
        contact_margin: float = 0.005,
        seed_joints: list | None = None,
    ) -> dict[str, Any]:
        """A straight line, held straight by the planner rather than by
        interpolation.

        Args:
            start: Where the line begins. Omitted means *where the hand is*,
                read from the same solver that will fly the line -- which is the
                form to prefer, and not only for brevity. A caller reading the
                start pose off the runtime's observation and passing it in puts
                the two ends of one leg in two frames on every robot whose grasp
                site is offset from its wrist, and the line then begins
                somewhere the arm is not: 134 mm away on the i2rt YAM.
            axis: Which tool axis travel runs along. Empty derives it from the
                travel direction, which is the safe default -- naming an axis
                the move is not along constrains the two axes it needs, and the
                measured result is a path *worse* than not constraining at all
                (43.00 mm against 28.40 mm on a 20 cm Franka move).
            orientation: ``"axis"`` holds only the end pose's approach axis and
                lets the solver pick a wrist; ``"lock"`` holds its whole
                rotation. Lock it when the *roll* is the grasp -- a parallel jaw
                closing across a film's faces is a roll, and left free it lands
                wherever the seed's basin puts it, measured at 22 degrees skewed
                across a 140 mm sheet. Leave it free for a top-down reach, where
                constraining it is what makes the chain stall in a saddle. A
                robot that declares ``honour_roll`` locks either way.

            tolerance: How close the last waypoint has to land for the plan to
                be accepted [m]. 0 keeps the backend's own, which is **30 mm**
                -- wider than a hammer's 36 mm handle, so the default gate can
                pass a plan that misses the thing being grasped entirely. Ask
                for less when the target is small.

        Returns:
            ``{"trajectory": Trajectory, "planned": bool, "reason": str,
            "position_error_m": float | None}``. ``planned: false`` rather than
            an exception, so a subgraph can route an unreachable pose to its own
            exit instead of dying -- and ``position_error_m`` because
            ``planned: true`` only means "inside the gate", which is a different
            question from "will reach it".
        """
        if orientation not in ("axis", "lock"):
            raise ValueError(f"orientation must be 'axis' or 'lock', got {orientation!r}")
        kwargs: dict[str, Any] = {"arm_id": int(arm_id), "num_waypoints": int(num_waypoints)}
        if axis:
            kwargs["axis"] = str(axis)
        if orientation == "lock":
            kwargs["hold_rotation"] = True
        if tolerance and float(tolerance) > 0.0:
            kwargs["tolerance"] = float(tolerance)
        if world_config is not None:
            kwargs["world_config"] = world_config
        if attached_object is not None:
            kwargs["attached_object"] = attached_object
        # Forwarded by signature, not unconditionally, and refused rather than
        # dropped when this backend cannot honour one. _contact_kwargs states why.
        contact = {
            "allow_start_contact": bool(allow_start_contact),
            "allow_goal_contact": bool(allow_goal_contact),
            "contact_margin": float(contact_margin),
        }
        refusal = self._contact_kwargs(self.backend.plan_linear, contact, kwargs, asked_by=MotionTools.plan_linear)
        if refusal is not None:
            return refusal
        if seed_joints is not None:
            # Plan from a posture the arm is not in yet -- a probe of a leg
            # that follows other legs -- rather than from the live joints.
            # Give ``start`` too, or the line begins where the arm is now.
            #
            # This has to happen BEFORE the call, and it did not: the merge
            # that brought this argument across left its forwarding after the
            # ``return`` below, so every seed a graph sent was dropped on the
            # floor and cuRobo planned the line from wherever the arm stood.
            # Measured on ``sharps_disposal/tray_pair``: the wrist-turn heading
            # search probes each candidate's 50 mm lift from the post-turn
            # posture; planned from the pre-turn one instead, the "line" bowed
            # 100-170 mm, the IK fallback was refused for carrying an
            # attachment, and every heading failed -- 0/4 on trials the same
            # graph passes 4/4 with the seed honoured.
            kwargs["seed_joints"] = [float(v) for v in seed_joints]
        plan = self.backend.plan_linear(start or self._here(int(arm_id)), end, **kwargs)
        # The backend knows which waypoint missed, by how much, and whether it
        # was position or the held rotation that could not be met. A caller
        # given only the generic sentence has to guess between an unreachable
        # target, an impossible line to a reachable one, and a wrist held by
        # `orientation="lock"` -- three different fixes. Backends that do not
        # report keep the generic reason.
        result = _result(
            plan,
            getattr(self.backend, "last_refusal", "") or "no linear plan holds this move's constraint",
        )
        # How far the plan actually lands from the pose asked for. `planned` is
        # a gate at `tolerance`, which defaults to 30 mm -- wider than many of
        # the things this robot picks up -- so passing the gate and reaching
        # the object are different questions, and only this answers the second.
        result["position_error_m"] = getattr(self.backend, "last_position_error", None)
        return result

    @staticmethod
    def _accepted(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop kwargs the backend's planner does not accept.

        The collision kwargs describe constraints only a collision-aware
        backend can check; the Newton fallback plans without one, so on it
        they are not a request that can be honoured or refused -- dropping
        them *is* the fallback, already announced when the backend was chosen.
        """
        import inspect

        params = inspect.signature(fn).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in params}

    def _here(self, arm_id: int) -> dict:
        """Where the hand is, from the backend that will fly the line.

        Named rather than raised through an ``AttributeError``, because a backend
        without it is a legitimate state -- ``IKBackendPyroki`` solves and plans
        and does not report -- and the fix is one keyword at the call site.
        """
        reader = getattr(self.backend, "current_pose", None)
        if reader is None:
            raise RuntimeError(
                f"{type(self.backend).__name__} cannot say where the hand is, so a line with no "
                "start cannot be planned on it -- pass start= explicitly"
            )
        return reader(arm_id=arm_id)

    def plan_joint(
        self,
        pose: dict,
        arm_id: int = 0,
        num_waypoints: int = 0,
        orientation: str = "axis",
    ) -> dict[str, Any]:
        """Reach a pose as a **posture**: solve it once, ease there in joint space.

        The move to make when the space between here and there is not the
        constraint -- a transit, a clearance via, a retreat. Only the endpoints
        are pinned, so it goes where the arm can go rather than where a straight
        line would have to pass; and it eases at both ends, so no waypoint is a
        velocity step on a stiff position servo.

        **It reports instead of refusing.** ``position_error_m`` and
        ``rotation_error_rad`` come back with the path and the caller decides
        whether they disqualify it. That is deliberate: a clearance via is a
        direction to swing through rather than a pose to arrive at, and the one
        in ``tape_peel`` misses by 161 mm on both layouts while the peel it
        leads into solves to a millimetre. A tool that refused it would refuse a
        motion that works.

        Returns:
            ``{"trajectory", "planned", "reason", "position_error_m",
            "rotation_error_rad", "joint_config"}``. ``planned`` is false only
            when the backend has no joint-space primitive at all.
        """
        if orientation not in ("axis", "lock"):
            raise ValueError(f"orientation must be 'axis' or 'lock', got {orientation!r}")
        plan = self.backend.plan_joint(
            pose,
            arm_id=int(arm_id),
            num_waypoints=int(num_waypoints),
            hold_rotation=(orientation == "lock"),
        )
        return {
            # None, not an empty container, when there is no path: `if
            # plan.get("trajectory")` is the guard every graph writes, and an
            # empty-but-truthy dict makes it execute a refusal. See `_result`.
            "trajectory": {"waypoints": list(plan["waypoints"])} if plan["waypoints"] else None,
            "planned": bool(plan["waypoints"]),
            "reason": "" if plan["waypoints"] else "the backend produced no joint path",
            "position_error_m": float(plan["position_error_m"]),
            "rotation_error_rad": float(plan["rotation_error_rad"]),
            "joint_config": list(plan.get("joint_config", ())),
        }

    def plan_to_pose(
        self,
        pose: dict,
        arm_id: int = 0,
        num_waypoints: int = 0,
        world_config: dict | None = None,
        hold_position: bool = False,
        attached_object: dict | None = None,
        allow_start_contact: bool = False,
        allow_goal_contact: bool = False,
        contact_margin: float = 0.005,
        seed_joints: list | None = None,
    ) -> dict[str, Any]:
        """A collision-aware path to a **full** pose -- position and rotation.

        Deliberately not what ``robot.go_to_pose`` becomes. ``tool_sort``'s
        ``waypoint_move.py`` records the cost of letting a planner reshape a
        deliberately-shaped path: *"TrajOpt's shortest smooth path routinely
        dips LOW while translating"*, dragging a held object and shoving the
        container ~12 cm. A caller reaching for this is asking for the shortest
        safe path, which is a different request from "go in a straight line",
        and should have to say so.

        ``seed_joints``, when given, is where the solve starts its search
        rather than the robot's live joints. The same target pose has more
        than one joint-space solution, and the backend seeds from wherever
        the arm currently is by default -- fine when that is also wherever
        the *next* call wants to seed from, and not when it is not: two
        callers reaching this same pose from different live postures land on
        different elbow configurations even though the pose matches to the
        millimetre, and a caller downstream of that pose can find one
        configuration plans cleanly from and the other does not.
        """
        plan = self.backend.plan_to_pose(
            pose,
            arm_id=int(arm_id),
            **self._accepted(
                self.backend.plan_to_pose,
                {
                    "num_waypoints": int(num_waypoints),
                    "world_config": world_config,
                    "hold_position": bool(hold_position),
                    "attached_object": attached_object,
                    "allow_start_contact": bool(allow_start_contact),
                    "allow_goal_contact": bool(allow_goal_contact),
                    "contact_margin": float(contact_margin),
                    "seed_joints": seed_joints,
                },
            ),
        )
        return _result(plan, "no collision-free path to this pose was found")

    def describe_planner(self) -> dict[str, Any]:
        """What is planning, so a node can decline rather than discover.

        ``post_upright.py`` already declines on
        ``robot.describe_arm()['solver']['honours_roll']`` rather than
        attempting a stand-up its solver cannot express. This is the same
        courtesy for planning: a node needing collision awareness can ask
        whether it has any.
        """
        return {
            "backend": type(self.backend).__name__,
            "world_aware": bool(getattr(self.backend, "supports_world_aware_plan", False)),
            # The *planner's*, not the IK backend's. The IK backend reports its
            # solver's honours_roll because that is what the servo's arrival
            # test must describe; a graph asking what a *plan* can express wants
            # the other answer, and conflating them is how a node concludes it
            # cannot stand a tool on end when the planner could.
            "honours_roll": bool(
                getattr(getattr(self.backend, "_planner", None) or self.backend, "honours_roll", False)
            ),
        }


def _result(plan: Any, reason: str) -> dict[str, Any]:
    """A plan, or a routable refusal. Never an exception.

    A refusal carries ``trajectory: None``, not an empty waypoint container.
    The container was truthy, so the idiom every graph writes --
    ``if plan.get("trajectory"): execute(...)`` -- executed a refusal as if it
    were a plan and then failed downstream on ``trajectory required``, naming
    the executor rather than the refused plan. Measured on one study cell:
    every episode of every iteration died that way in a lift, and four rounds
    of improvement went to nodes that never ran. ``None`` makes the guard
    every author already writes correct.
    """
    if plan is None:
        return {"trajectory": None, "planned": False, "reason": reason}
    return {"trajectory": plan, "planned": True, "reason": ""}
