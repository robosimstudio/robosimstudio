# SPDX-License-Identifier: Apache-2.0
"""``robot.*`` introspection: what a graph may ask about the robot it was given.

**The gap this closes.** A graph node script may import ``gap``, ``gap_core``,
numpy, scipy and the standard library, and nothing else -- it cannot reach into
this package. Until now nothing in the tool surface answered a single question
about the *hand*: ``robot.get_gripper`` returns one travel fraction and
``sim.get_object_obb`` returns whether an object fits, but never what it fits
into. So every embodiment fact in a shipped graph is a restated literal --
``panda_hand_to_tcp = 0.1029``, ``OPEN_HALF_SPAN = 0.052``, ``"width_m":
0.0475`` -- and a graph is portable only as far as those literals happen to
travel, which is nowhere.

**Why ``robot.`` and not ``sim.``.** Two reasons that point the same way.

The privilege boundary's own test is *could a real driver answer this*. A
Robotiq driver knows its own stroke, its own tool-centre point and which way its
jaws close; nothing whatsoever knows where the hammer is. A hand's geometry is
proprioception in the same sense joint angles are, which
:mod:`~robosimstudio_policy_agent.contract` already admits and says why.

And the mechanical reason: ``register_sim_tools`` is called only under
``privileged``. A graph rewritten against a ``sim.`` gripper tool would become
privileged-only -- unable to cross the bridge for a reason having nothing to do
with ground truth -- which is the exact opposite of what this is for. These
register unconditionally.

Claiming ``robot.*`` against the pinned runtime is safe because ``gap_core``'s
registry raises on a duplicate name: if a future pin ships a tool called any of
these, the connector fails loudly at construction rather than one silently
shadowing the other.

**Facts and judgements are separate tools, deliberately.**
:meth:`RobotTools.describe_gripper` and :meth:`RobotTools.describe_arm` report
things that were measured. :meth:`RobotTools.describe_workspace` reports things
this package *decided* -- a transport height, a pregrasp standoff -- derived from
those facts plus the scene. A node reaching for the second is accepting a
judgement, and it should be able to see that it is doing so.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from robosimstudio.arms import arm_id_of, arm_name, arm_order, arm_role
from robosimstudio.core.bus import Bus
from robosimstudio.core.math import quat_xyzw_to_wxyz
from robosimstudio.core.streams import reading
from robosimstudio.solvers import grasp_quat

from .facade import EnvFacade
from .refusal import ToolRefusal
from .types import quat_wxyz, vec3

__all__ = ["RobotTools", "fingertip_reach", "register_robot_tools"]

#: Clearance left above whatever the hand has to pass over, when this package
#: derives a height rather than being told one [m]. A judgement, not a
#: measurement, which is why it lives here and not on a spec.
MARGIN = 0.02


def register_robot_tools(registry, env, bus: Bus, *, ik: Any = None, facade: EnvFacade | None = None) -> None:
    """Register the non-privileged ``robot.*`` introspection tools.

    Called before the ``privileged`` branch, so a graph built on these still runs
    under ``--no-privileged``.
    """
    tools = RobotTools(env, bus, ik=ik, facade=facade)
    for name, fn, summary in tools.entries():
        # The verbs that step the simulation are spans on the capture tape,
        # like every other stepping tool: a jaw ramp that ejects the part is
        # a motion the feedback has to be able to see, and until set_grip
        # carried the tag a policy's grasp attempts left no geometry at all.
        tags = ("robot", "sim_step") if name in _STEPPING else ("robot",)
        registry.register_callable(name, fn, summary=summary, tags=tags)


_STEPPING = frozenset({"robot.set_grip", "robot.wait_steps", "robot.stream_dual"})
"""The robot.* verbs that advance the simulation."""


class RobotTools:
    """The introspection implementations."""

    def __init__(self, env, bus: Bus, *, ik: Any = None, facade: EnvFacade | None = None) -> None:
        self.env = env
        self.bus = bus
        self.ik = ik
        self.facade = facade
        self._workspace: dict[str, dict[str, Any]] = {}

    def entries(self) -> list[tuple[str, Any, str]]:
        return [
            ("robot.describe_gripper", self.describe_gripper, "The hand's geometry: span, axes, pads, housing."),
            ("robot.describe_arm", self.describe_arm, "The arm's joints, home posture, tool-centre point and solver."),
            (
                "robot.describe_workspace",
                self.describe_workspace,
                "Derived heights and clearances for this robot in this cell.",
            ),
            ("robot.grasp_frame", self.grasp_frame, "A grasp orientation built from this hand's measured axes."),
            ("robot.set_grip", self.set_grip, "Drive the jaws to a width, an object's width, or a named preset."),
            ("robot.wait_steps", self.wait_steps, "Hold position while the scene runs on for a number of steps."),
            (
                "robot.stream_dual",
                self.stream_dual,
                "Advance several arms along their own dense joint trajectories together, in lockstep.",
            ),
            ("robot.instruction", self.instruction, "What this episode asked for, as language."),
            (
                "robot.forward_kinematics",
                self.forward_kinematics,
                "Tool-centre poses for hypothetical arm joint vectors, and the arm's current joints.",
            ),
        ]

    def instruction(self) -> dict[str, Any]:
        """The episode's own command, as language.

        **Here because the runtime drops it, and here rather than in a suite
        because every suite loses it the same way.** The facade publishes the
        instruction beside the camera blocks -- :data:`~robosimstudio_policy_agent.contract.ALLOWED_KEYS`
        names it as one of the things that may cross the privilege boundary,
        the task and the clock -- and the pinned runtime's
        ``Connector._build_observation`` then assembles a typed ``Observation``
        out of ``cameras`` and ``arms`` and drops everything else. A body that
        reads ``robot.get_observation()["instruction"]`` gets the empty string,
        always, and silently.

        The alternative is worse in a way that is easy to miss. A workflow can
        carry the instruction into a subgraph as a declared input fed from
        ``{"$ref": "in.instruction"}`` -- the executor binds it, because
        ``_lookup_cross_subgraph_output`` falls back to the run's own inputs --
        but ``gap.runtime.validate``'s W8 knows only two producers, an upstream
        subgraph's output and the observation stream, so it refuses the graph
        before its first node. ``gap.execute`` validates. A graph wired that way
        does not merely fail its checker; it does not start. A tool node inside
        the subgraph is a producer W8 understands.

        **Not privileged.** A cell is *told* what to do; being told is not
        seeing. A vision body may read this under ``--no-privileged`` and
        should: an instruction that names the tool is the difference between
        looking for a wrench and looking for whichever of four tools scores
        highest, which on a table that also holds three distractors is the
        difference between the right answer and a plausible one.
        """
        return {"instruction": str(getattr(self.env.episode, "instruction", "") or "")}

    def stream_dual(self, tracks: dict, tolerance: float = 0.005, settle_steps: int = 60) -> dict[str, Any]:
        """Drive two or more arms along one synchronised path.

        ``tracks`` maps ``arm_id`` to a dense list of joint vectors. Two
        hands carrying one rigid object have to move together; taking turns
        either strains the grasp or, done as a tool call per waypoint, costs
        a graph round-trip per waypoint.
        """
        from gap_core.errors import ToolError

        facade = self.facade
        if facade is None or not hasattr(facade, "stream_dual_trajectory"):
            raise ToolError("robot.stream_dual", "this facade cannot stream several arms together")
        parsed = {int(a): [list(map(float, w)) for w in wps] for a, wps in dict(tracks).items() if wps}
        if not parsed:
            raise ToolError("robot.stream_dual", "tracks required")
        facade.stream_dual_trajectory(parsed, settle_tolerance=float(tolerance), settle_max_steps=int(settle_steps))
        return {"arms": sorted(parsed), "waypoints": max(len(w) for w in parsed.values())}

    # ---------------------------------------------------------------- helpers
    def _arm(self, arm_id: int, role: str) -> str:
        """The arm a call means. ``role`` wins over ``arm_id`` when given."""
        handles = self.env.handles
        if role:
            if role not in ("work", "support"):
                raise ValueError(f"role must be 'work' or 'support', not {role!r}")
            name = handles.work_arm if role == "work" else handles.support_arm
            if not name:
                raise ToolRefusal(f"this scene has no {role!r} arm; arms are {sorted(handles.arms)}")
            return name
        return arm_name(self.env, arm_id)

    def _hand(self, name: str):
        hand = self.env.scene.embodiment.hands.get(name)
        if hand is None:
            raise ToolRefusal(f"arm {name!r} declares no gripper")
        return hand

    def _gripper(self, name: str):
        return self.env.scene.embodiment.grippers.get(name)

    # ------------------------------------------------------------------ facts
    def describe_gripper(self, arm_id: int = 0, role: str = "") -> dict[str, Any]:
        """Everything that changes when the hand changes, and nothing else.

        Frames matter and are stated once: ``reach`` and ``rise`` are measured
        along the gripper's own **approach** axis, ``half_width`` and
        ``clearance`` across its own **closing** axis. The grocery graph states
        the same quantities in *world* axes today, which is correct only because
        that hand is always driven top-down.

        ``span_m`` is the gap an object passes through. ``open_footprint_m`` is
        the separation of the pad *centres*, which is a different and larger
        number -- on the 2F-85, 0.085 against 0.104 -- and is the one a planner
        deciding whether a placement sweeps its neighbour wants.

        Every geometry block carries ``stated``. False means nobody measured it
        and the number below is this package's fallback, so a node can decline to
        make a decision on it rather than trusting a plausible zero.
        """
        name = self._arm(arm_id, role)
        hand, gripper = self._hand(name), self._gripper(name)
        handle = self.env.handles.arms[name]
        span = float(handle.gripper_span)
        clearance = float(hand.pad_clearance) or 0.004
        return {
            "arm": name,
            "arm_id": arm_id_of(self.env, name),
            "role": arm_role(self.env, name),
            "gripper": hand.name,
            # From the hand's own declaration, not inferred from how many joints
            # it drives or which signs they run on: two of these hands are
            # parallel jaws that satisfy neither of those heuristics.
            "kind": "geared" if hand.geared else "prismatic",
            "dof": len(handle.gripper_coords),
            "span_m": span,
            "open_footprint_m": 2.0 * float(hand.pad_half_span) if hand.pad_half_span else span,
            "max_grasp_width_m": max(0.0, span - 2.0 * clearance),
            "min_grasp_width_m": float(gripper.width_for_command(1.0)) if gripper is not None else 0.0,
            "command": {
                "open": 0.0,
                "closed": 1.0,
                # Stated because the two conventions in this stack are inverted
                # from each other, and a node that assumes one reading applies to
                # both concludes a hand closed on air is a hand holding
                # something.
                "convention": "0 = open, 1 = fully closed",
                "observed_convention": "robot.get_gripper reports 1.0 = OPEN",
            },
            "travel_is_linear": bool(hand.linear_travel or (gripper is not None and gripper.calibrated)),
            "width_fit": {
                "source": "measured" if (gripper is not None and gripper.calibrated) else "span",
                "at_open_m": float(gripper.width_for_command(0.0)) if gripper is not None else span,
                "at_closed_m": float(gripper.width_for_command(1.0)) if gripper is not None else 0.0,
            },
            "finger": {
                "stated": bool(hand.tip_reach or hand.pad_thickness or hand.pad_clearance),
                "reach_m": float(hand.tip_reach),
                "pad_thickness_m": float(hand.pad_thickness),
                "clearance_m": clearance,
            },
            "housing": {
                "stated": bool(hand.housing_half or hand.housing_rise),
                "half_width_m": float(hand.housing_half),
                "rise_m": float(hand.housing_rise),
            },
            "approach_axis": vec3(hand.approach_axis),
            "close_axis": vec3(hand.close_axis),
            "finger_bodies": [list(side) for side in hand.finger_bodies],
        }

    def describe_arm(self, arm_id: int = 0, role: str = "") -> dict[str, Any]:
        """The arm: its joints, its home, its tool-centre point, its solver.

        ``arms`` lists every arm with its id and role, so a bimanual node never
        has to guess which number means which hand.

        ``solver.honours_roll`` is worth its place on its own. The default IK
        aligns position and an approach axis with wrist roll left free, so it
        cannot express standing a tool on end -- ``tool_sort``'s upright node
        spends fourteen lines saying so and then discovers it at runtime by
        failing. With this a node can decline up front.
        """
        name = self._arm(arm_id, role)
        handle = self.env.handles.arms[name]
        hand = self.env.scene.embodiment.hands.get(name)
        model = self.env.model
        # A script thread reads here; see :mod:`robosimstudio.core.streams`.
        with reading():
            lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        # The home posture's pose, fixed when the solver was built -- not the
        # live TCP, which after a descent is wherever the hand went.
        position, quat = self.env.arm(name).home_pose
        wxyz = quat_xyzw_to_wxyz(quat)
        return {
            "arm": name,
            "arm_id": arm_id_of(self.env, name),
            "role": arm_role(self.env, name),
            "arms": [
                {"arm_id": i, "name": n, "role": arm_role(self.env, n)} for i, n in enumerate(arm_order(self.env))
            ],
            "dof": len(handle.arm_coords),
            "joint_limits": {
                "lower": [float(lower[d]) for d in handle.arm_dofs],
                "upper": [float(upper[d]) for d in handle.arm_dofs],
            },
            "home_q": [float(v) for v in handle.home_q],
            # Position *and* rotation. The home end-effector quaternion was
            # already computed for the grasp heuristic and its position thrown
            # away; a transport height wants exactly that position.
            "home_pose": {
                "position": vec3(position),
                "rotation": {"w": float(wxyz[0]), "x": float(wxyz[1]), "y": float(wxyz[2]), "z": float(wxyz[3])},
            },
            "tcp_offset": vec3(hand.tcp_offset if hand else (0.0, 0.0, 0.0)),
            "tcp_rotation": quat_wxyz(hand.tcp_offset_rot if hand else (0.0, 0.0, 0.0, 1.0)),
            "solver": {
                "name": type(self.ik).__name__ if self.ik is not None else "",
                "honours_roll": bool(getattr(self.ik, "honours_roll", False)),
            },
        }

    # ------------------------------------------------------------- judgements
    def describe_workspace(self, arm_id: int = 0, role: str = "") -> dict[str, Any]:
        """Heights and clearances this package derives for this robot here.

        Every number below is a *judgement* built from facts, which is why it is
        a separate tool from the two above.

        ``transport_z`` is the one that matters, and the rule is not invented:
        two authors on two robots wrote the same literal for it and justified it
        the same way. The grocery graph says "the DROID Franka's tool centre
        rests at 0.342 m, and asking the linear planner for much more puts the
        arm near its workspace boundary"; the tool-sort graph says "going higher
        pushes Franka near the workspace boundary and the linear plan refuses".
        Both landed on 0.353. So the rule is *the arm's own resting tool height*,
        floored by what the hand needs to clear the surface -- which reproduces
        both numbers and, unlike them, is a fact about whichever robot arrived.

        Published in two frames because tasks think in both: ``transport_z`` is
        world, ``transport_above_surface_m`` is relative to the work surface.
        """
        name = self._arm(arm_id, role)
        if name in self._workspace:
            return self._workspace[name]
        hand = self._hand(name)
        surface_z = float(self.env.handles.surface_z)
        # The resting height, not the live one: a first ask made after a descent
        # would otherwise cache grasp height as transport height for good.
        home_z = float(self.env.arm(name).home_pose[0][2])
        reach = float(hand.tip_reach)
        floor = surface_z + reach + MARGIN
        transport_z = max(home_z, floor)
        out = {
            "arm": name,
            "arm_id": arm_id_of(self.env, name),
            "surface_z": surface_z,
            "transport_z": transport_z,
            "transport_above_surface_m": transport_z - surface_z,
            "transport_z_source": "home_tcp" if home_z >= floor else "surface_clearance",
            # What a hover has to clear to get the whole hand above an object:
            # the pads, plus the housing standing above them, plus margin.
            "align_clearance_m": reach + float(hand.housing_rise) + float(hand.pad_thickness) + MARGIN,
            "pregrasp_standoff_m": max(2.0 * reach, 0.10) if reach else 0.10,
            "margin_m": MARGIN,
            # False means the hand's envelope was never measured, so every
            # clearance above is MARGIN and nothing else. A node planning around
            # a housing should say it cannot rather than trust that.
            "envelope_stated": bool(hand.tip_reach or hand.housing_rise),
        }
        self._workspace[name] = out
        return out

    # ------------------------------------------------------------------ pose
    def grasp_frame(
        self,
        approach: dict[str, float] | None = None,
        close_heading_deg: float | None = None,
        arm_id: int = 0,
        role: str = "",
    ) -> dict[str, Any]:
        """A grasp orientation built from *this* hand's measured axes.

        With no arguments: the canonical top-down grasp, jaws closing across
        world +x. That is what every ``_DOWN = {"w": 0, "x": 1, "y": 0, "z": 0}``
        literal in the shipped graphs is trying to be, and what it silently is
        not on a hand whose approach axis is not its tool-local +z.

        The arithmetic is :func:`~robosimstudio.solvers.grasp.antipodal.grasp_quat`, the
        same function the suite-side helper is written over, so a grasp frame is
        built one way in this package rather than two.

        Returns both wrist solutions. They are not interchangeable to the arm --
        they differ by a half turn on the last joint, and one of them is
        routinely against a hard stop -- so a caller with an IK picks.
        """
        name = self._arm(arm_id, role)
        hand = self._hand(name)
        down = np.asarray(
            [approach["x"], approach["y"], approach["z"]] if approach else [0.0, 0.0, -1.0], dtype=np.float64
        )
        norm = float(np.linalg.norm(down))
        if norm < 1.0e-9:
            raise ValueError("approach must be a non-zero direction")
        down = down / norm
        yaw = np.radians(float(close_heading_deg or 0.0))
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        # Orthogonalise against the approach, so a caller may ask for a heading
        # that is not square to it without getting a non-orthogonal basis back.
        heading = heading - np.dot(heading, down) * down
        if float(np.linalg.norm(heading)) < 1.0e-6:
            heading = np.cross(down, [1.0, 0.0, 0.0])
            if float(np.linalg.norm(heading)) < 1.0e-6:
                heading = np.cross(down, [0.0, 1.0, 0.0])
        heading = heading / float(np.linalg.norm(heading))
        q = grasp_quat(hand.approach_axis, hand.close_axis, hand.tcp_offset_rot, down, heading)
        flipped = grasp_quat(hand.approach_axis, hand.close_axis, hand.tcp_offset_rot, down, -heading)
        return {
            "arm": name,
            "rotation": quat_wxyz(q),
            "rotation_flipped": quat_wxyz(flipped),
            "approach": vec3(down),
            "close_axis": vec3(heading),
        }

    # ------------------------------------------------------------------ grip
    def set_grip(
        self,
        preset: str = "",
        width_m: float = -1.0,
        object_width_m: float = -1.0,
        squeeze_m: float = 0.004,
        fraction_closed: float = -1.0,
        arm_id: int = 0,
        role: str = "",
        ramp_steps: int = 24,
        settle_steps: int = 0,
    ) -> dict[str, Any]:
        """Drive the jaws, in whichever terms the caller actually has.

        ``object_width_m`` is the portable verb, and the one a graph should
        reach for: *grip the thing I measured*. A node saying that never states a
        number that depends on which hand arrived, which is the whole difference
        between ``"width_m": 0.003`` in a workflow and a graph that survives a
        gripper swap.

        Exactly one of ``preset`` / ``width_m`` / ``object_width_m`` /
        ``fraction_closed`` may be given, with one exception: the ``pinch`` and
        ``grip`` presets are relative to an object and so take
        ``object_width_m`` alongside them (and refuse without it). The presets
        are derived from the hand rather than tabulated: ``open`` is its span,
        ``close`` is its own minimum, ``pinch`` is the object plus the finger
        clearance either side, ``grip`` is the object less ``squeeze_m``.

        Writes through the facade's single gripper path, so the command has one
        holder -- a second writer here is what previously made a calibrated
        partial close report success while the jaws stayed at 94 mm.
        """
        relative = preset in ("pinch", "grip")
        if relative and object_width_m < 0.0:
            raise ValueError(f"grip preset {preset!r} is relative to an object: give object_width_m with it")
        given = [bool(preset), width_m >= 0.0, object_width_m >= 0.0 and not relative, fraction_closed >= 0.0]
        if sum(given) != 1:
            raise ValueError(
                "give exactly one of preset, width_m, object_width_m, fraction_closed"
                " (object_width_m may accompany preset 'pinch' or 'grip')"
            )
        name = self._arm(arm_id, role)
        gripper = self._gripper(name)
        if gripper is None:
            raise ToolRefusal(f"arm {name!r} has no gripper to drive")
        info = self.describe_gripper(arm_id=arm_id, role=role)
        span = float(info["span_m"])
        clearance = float(info["finger"]["clearance_m"])

        requested = float("nan")
        if fraction_closed >= 0.0:
            command = float(np.clip(fraction_closed, 0.0, 1.0))
            requested = gripper.width_for_command(command)
        else:
            if preset:
                table = {
                    "open": span,
                    "close": float(info["min_grasp_width_m"]),
                    "pinch": object_width_m + 2.0 * clearance,
                    "grip": object_width_m - squeeze_m,
                }
                if preset not in table:
                    raise ValueError(f"unknown grip preset {preset!r}; have {sorted(table)}")
                requested = table[preset]
            elif object_width_m >= 0.0:
                requested = object_width_m - squeeze_m
            else:
                requested = width_m
            command = gripper.command_for_width(requested)

        clamped_width = gripper.width_for_command(command)
        reason = ""
        if requested == requested and requested > span:
            reason = "wider than the span"
        elif requested == requested and requested < float(info["min_grasp_width_m"]):
            reason = "narrower than the jaws close"

        if self.facade is None:
            raise RuntimeError("robot.set_grip needs the connector's facade; it is attached at registration")
        # Ramped, not stepped: a step input on a stiff finger servo slams the
        # pads, which is how a part gets ejected rather than held.
        this_arm = arm_id_of(self.env, name)
        steps = max(1, int(ramp_steps))
        ramped = getattr(self.facade, "set_grip", None)
        if ramped is not None:
            # A facade that ramps for itself -- the fleet's -- gets one motion,
            # started from wherever *this world's* command is. The loop below
            # used to run here too, and it read its start off the template
            # gripper, which nothing in a fleet ever writes: every ramp began
            # from "open", so a re-grip on a held object let go of it first.
            ramped(1.0 - command, ramp_steps=steps, settle_steps=int(settle_steps), arm_id=this_arm)
            waited = int(settle_steps)
        else:
            # This world's command, not the template's, on an env that has one
            # per world; the single-world env's gripper *is* the template.
            holder = self.env.gripper(name) if hasattr(self.env, "gripper") else gripper
            start = float(self.bus.call(lambda: holder.command))
            for i in range(1, steps + 1):
                blend = start + (command - start) * (i / steps)
                # 1 - blend: the facade speaks GaP's convention, where 1.0 is open.
                self.facade._set_gripper(1.0 - blend, this_arm)
                self._wait(1)
            waited = self._wait(int(settle_steps)) if settle_steps > 0 else 0
        return {
            "arm": name,
            "arm_id": arm_id_of(self.env, name),
            "requested_width_m": float(requested),
            "command_width_m": float(clamped_width),
            "fraction_closed": float(command),
            "clamped": bool(reason),
            "reason": reason,
            "waited": waited,
        }

    def wait_steps(self, steps: int = 30, timeout_s: float = 0.0) -> dict[str, Any]:
        """Hold where you are while the scene runs on.

        **Why this is ``robot.`` and not ``sim.``.** Waiting on the clock names
        nothing. A real driver can hold position and count control cycles, which
        is the privilege test :mod:`~robosimstudio_policy_agent.contract` applies,
        so a graph that settles is not thereby a graph that needs ground truth.

        And it is worth having at all, rather than being a node that returns
        immediately, for the reason cloth_fold's version already gave: a fold is
        not scored until the sheet has been still, and cloth keeps relaxing long
        after the hand has left it. *"A graph that returns as soon as it has
        moved never sees its own result."*

        Args:
            steps: Control steps to wait for.
            timeout_s: Give up after this long. ``0`` derives a budget from
                *steps*, which is the right default because the two suites that
                hand-picked one disagreed by an order of magnitude -- cloth_fold
                allowed 0.3 s per step, cable_route 2.0 s -- and neither number
                is a property of waiting. The derived budget is generous against
                the slower of the two measured scenes; a caller who knows their
                scene is slower still should say so rather than discover it as a
                short wait.

        Returns:
            ``{"waited": <steps actually observed>}``. A short count is how a
            caller learns the budget ran out; this does not raise, because a
            settle that timed out has still settled *something* and the node
            above is better placed to decide whether that was enough.
        """
        # Two seconds a step and half a minute at least: at sixteen episodes on one box a step takes a
        # tenth of a second and more, and a settle cut short by the clock is a hand that has not settled.
        budget = float(timeout_s) if timeout_s > 0.0 else _wait_budget(steps)
        return {"waited": self._wait(int(steps), timeout_s=budget)}

    def forward_kinematics(self, joint_configs: list | None = None, arm_id: int = 0, role: str = "") -> dict[str, Any]:
        """Where the tool centre would be for each arm joint vector, and where the joints are now.

        Pure kinematics on the runtime's own chain -- the same FK the Cartesian
        controller trusts -- for a script that wants to *search* joint space
        rather than ask a planner: which wrist setting stands a held object
        upright, what arc the hand sweeps doing it. Batched, because one round
        trip per configuration is what makes a few thousand of them slow.

        Returns ``joint_config`` (the arm's current joints, in the order the
        vectors are given) and ``poses`` (one TCP pose per vector given).
        """
        name = self._arm(arm_id, role)
        handle = self.env.handles.arms[name]
        ik = self.env.arm(name)
        coords = list(handle.arm_coords)

        def run() -> tuple[list[float], list[dict[str, Any]]]:
            # On the owner thread: fk_ee evaluates into the solver's shared
            # scratch state, which the controller uses between steps.
            with reading():
                full = np.asarray(self.env.state.joint_q.numpy(), dtype=np.float64).copy()
            poses = []
            for config in joint_configs or []:
                q = full.copy()
                q[coords] = np.asarray(config, dtype=np.float64)[: len(coords)]
                position, quat = ik.fk_ee(np.asarray(q, dtype=np.float32))
                poses.append({"position": vec3(position), "rotation": quat_wxyz(quat)})
            return [float(v) for v in full[coords]], poses

        current, poses = self.bus.call(run)
        return {"arm": name, "joint_config": current, "poses": poses}

    def _wait(self, steps: int, timeout_s: float | None = None) -> int:
        """Block until the owner thread has stepped ``steps`` times.

        Through the bus's own step wait rather than by polling ``snapshot.step``,
        and in a wave that is the difference between waiting and deadlocking.

        A wave steps only when every live world's graph thread is parked in a
        fleet wait, and parking is something the three instrumented waits do --
        this was a fourth, polling in a sleep loop and parked nowhere. So a
        graph that ramped its gripper closed held the whole fleet: this world
        spun, the fleet could not step, so the step it was spinning for never
        came, so every ``_wait(1)`` in the ramp burned its full timeout while
        the other worlds sat parked on intents that needed that same step.
        Silent -- the ramp logs nothing -- and measured at 55 minutes before a
        runner's cap called it "wrote no results rows".

        ``wait_for_step`` is on both bus kinds: in a fleet it parks the world so
        the barrier can close, and in a single world it is the same blocking
        wait the loop was approximating. The step count is still returned rather
        than raised on, because a settle that timed out has settled *something*
        and the caller decides whether that was enough -- so a timeout here is a
        short count, exactly as before.
        """
        from robosimstudio.core.bus import BusTimeout

        # No poll fallback. Both shipped bus kinds carry ``wait_for_step``, and
        # a loop that watched the clock go by while sleeping is exactly the
        # thing this method was rewritten to stop being -- keeping one for a bus
        # that does not exist would leave the deadlock reachable and untested.
        # ``tests/test_fleet_waits.py`` scans for the shape.
        start = int(self.bus.snapshot.step)
        wait_for_step = self.bus.wait_for_step

        # ``timeout_s`` is one deadline for the whole wait, not a budget per
        # step: per step, a 30-step settle with a 60 s timeout could block for
        # half an hour.
        if timeout_s is None:
            timeout_s = _wait_budget(steps)
        target = start + int(steps)
        deadline = time.monotonic() + float(timeout_s)
        while int(self.bus.snapshot.step) < target:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                wait_for_step(after=int(self.bus.snapshot.step), timeout=remaining)
            except BusTimeout:
                break
        return int(self.bus.snapshot.step) - start


def _wait_budget(steps: int) -> float:
    """The derived total deadline for a wait of *steps* control steps [s]."""
    return max(30.0, float(steps) * 2.0)


# ---------------------------------------------------------------- wire types
# gap's types are TypedDicts of plain floats. A tool handing back a numpy array
# serialises in process and fails the moment the same graph is pointed at an RPC
# bundle, so the coercion happens here rather than being the thing under test.
def fingertip_reach(env, arm: str, body_q: np.ndarray | None = None) -> float | None:
    """How far the fingertips sit past the tool-centre point right now [m], along the approach axis.

    **Live, off the finger colliders, because on a geared hand it is not a
    constant.** A prismatic hand's pads ride a fixed distance below its TCP and
    :attr:`GripperSpec.tip_reach` says so once. The 2F-85's pads swing on a
    four-bar: measured on the Panda pairing, they sit 5.9 mm past the TCP at
    full open, 11.5 mm at a 67 mm pre-open and 15-16 mm closed on a 55 mm gear
    -- ten millimetres of travel along the approach axis over the stroke, in a
    task whose grasp clearance is 2.5 mm. The stated 4.4 mm is the open-hand
    number, and a graph that planned every height on it put the pads 10 mm
    lower than it thought at the moment it closed.

    The same measurement ``scripts/measure_gripper.py`` takes at the rest pose,
    taken instead at the current one: the furthest corner of any finger
    collider from the TCP along the grasp frame's +z. Proprioception -- a real
    driver reports its finger position and publishes its linkage -- so it
    belongs beside the travel fraction in ``robot.get_gripper`` rather than
    behind the privilege boundary. ``None`` when the hand names no finger body.
    """
    from robosimstudio.core.math import quat_multiply, quat_to_mat, transform_point

    handle = env.handles.arms[arm]
    if not handle.finger_groups:
        return None
    if body_q is None:
        body_q = np.asarray(env.state.body_q.numpy(), dtype=np.float64)
    # Cached on the arm's handle, not in a module dict keyed by ``id(env)``: the
    # handle is rebuilt with the scene and freed with it, so a later env that
    # reuses a freed one's id can never be handed the old hand's corners.
    corners = getattr(handle, "_finger_corners", None)
    if corners is None:
        from robosimstudio.solvers.grasp.obb import _shape_corners_local

        builder = getattr(env.scene, "builder", None)
        corners = []
        if builder is not None:
            import newton

            collides = int(newton.ShapeFlags.COLLIDE_SHAPES)
            for side in handle.finger_groups:
                for body, shapes in side:
                    for s in shapes:
                        if not (int(builder.shape_flags[int(s)]) & collides):
                            continue
                        local = _shape_corners_local(builder, int(s))
                        if len(local):
                            corners.append((int(body), np.asarray(local, dtype=np.float64)))
        handle._finger_corners = corners
    if not corners:
        return None
    ee = np.asarray(body_q[handle.ee_body], dtype=np.float64)
    tcp = transform_point(ee, handle.ee_offset)
    approach = quat_to_mat(quat_multiply(ee[3:7], handle.ee_offset_rot))[:, 2]
    reach = None
    for body, local in corners:
        pose = np.asarray(body_q[body], dtype=np.float64)
        world = np.asarray([transform_point(pose, c) for c in local])
        far = float(np.max((world - tcp) @ approach))
        reach = far if reach is None else max(reach, far)
    return reach
