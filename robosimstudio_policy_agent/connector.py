# SPDX-License-Identifier: Apache-2.0
"""``ConnectorNewton``: the pinned GaP runtime, over a toolkit-layer environment.

The runtime is **reused as a library**, not reimplemented. Its ``Connector``
base already ships the full ``robot.*`` tool registry -- move to joints, go to
pose, open and close the gripper, execute a trajectory, with the convergence
stepping and settle counts that make those behave like real motions -- and an IK
seam to hang a backend off. Reimplementing that would mean a second
implementation of the semantics real evaluations run on, which is the one thing
worth *not* having two of.

So this composes rather than rewrites::

    ConnectorNewton = Connector(GapEnvFacade(env, bus), config, ik=IKBackend)
                      + register_robot_tools(...)      # always
                      + register_geometry_tools(...)   # always
                      + register_sim_tools(...)        # privileged only
                      + whatever task tools the suite contributes, filtered the
                        same way (see NonPrivilegedRegistry)

If the runtime is not installed, importing this module raises with the extra to
install. A ~400-line minimal executor is the documented contingency, not a
silent fallback: a graph that runs against a stand-in and not against the real
runtime has been tested against the stand-in.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from robosimstudio.arms import arm_name as _arm_name
from robosimstudio.core.bus import Bus, SimBus
from robosimstudio.solvers.ik.solver import POS_TOLERANCE as _SERVO_POS_TOLERANCE
from robosimstudio.solvers.ik.solver import ROT_TOLERANCE_DEG as _SERVO_ROT_TOLERANCE_DEG
from robosimstudio_bench.envs.env import Env

from .facade import EnvFacade, GapEnvFacade
from .geometry_tools import register_geometry_tools
from .grasp_tools import register_grasp_tools
from .ik import IKBackend
from .motion_tools import register_motion_tools
from .perception_tools import register_perception_tools
from .robot_tools import register_robot_tools
from .sim_tools import register_sim_tools

_SERVO_ROT_TOLERANCE = math.radians(_SERVO_ROT_TOLERANCE_DEG)

_log = logging.getLogger(__name__)

__all__ = ["ConnectorNewton", "NewtonEnvConfig", "NonPrivilegedRegistry", "make_connector"]


def _routed_servo_move(move) -> dict:
    """Run one blocking servo motion and hand back an outcome dict, never a raise for an unreachable pose.

    A pose the solver cannot reach comes back as ``{"outcome": "unreachable"}``,
    a refusal the graph can route on. Raised, it went to the workflow's
    ``abort`` recovery, whose first act on every shipped graph is to open the
    gripper -- which dropped the tool at the pick site in three study cells and
    read, downstream, as "the grasp failed". Every servo-backed pose verb goes
    through here so they all answer the same way: ``reached`` plus the servo's
    own ``outcome`` (``reached``, ``timeout``, ``contact``, ``unreachable``).
    """
    try:
        result = move()
    except RuntimeError as exc:
        if "unreachable" not in str(exc):
            raise
        return {"reached": False, "outcome": "unreachable", "reason": str(exc)}
    result = dict(result or {})
    result["reached"] = result.get("outcome") == "reached"
    return result


MOTION_PLANNERS: dict[str, str] = {
    "curobo": "robosimstudio_policy_agent.planners.curobo.MotionBackendCuRobo",
}
"""Motion-planner alias -> dotted path.

A *peer* of the IK backend rather than a replacement: ``solve_ik`` stays on the
cheap solver because the servo calls it up to 400 times a move, and only plans go
to the planner. ``""``/``"plan"``/``"osc"`` name no planner at all -- they choose
whether a servo is built -- which is why they are not entries here.
"""

SERVO_MODES: tuple[str, ...] = ("", "osc")
"""Motion modes that build a :class:`CartesianServo` on their own.

``"plan"`` deliberately does not. A *planner* mode does, though it is not listed
here: a planner answers ``plan_to_pose`` and the servo still answers the linear
moves the graph asks for between plans, so naming one is a request for both."""


def _resolve(path: str) -> Any:
    """Import a dotted path and return what it names: ``module:Name`` or ``module.Name``."""
    from robosimstudio_bench.taskbuild.artifacts import resolve_dotted

    return resolve_dotted(path)


def _accepted(cls: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    """Whichever of ``candidates`` this constructor actually declares.

    A planner registered from outside cannot be expected to take cuRobo's
    ``config_path``/``urdf_path``/``lock_joints``/``time_scale``, nor PyRoKi's
    ``ik``, and handing them over would make an unrelated ``TypeError`` look like
    an unavailable planner. Asked of the signature, so a backend states its own
    needs -- which is also how one planner can want the cheap IK member and
    another want nothing but a URDF.
    """
    import inspect

    params = inspect.signature(cls).parameters
    return {k: v for k, v in candidates.items() if k in params}


def _finger_lock(env: Env) -> dict[str, float]:
    """Where the *scene* holds the work arm's fingers, for cuRobo to lock them.

    Not a nominal open. cuRobo plans with the gripper frozen, and freezing it
    somewhere the scene does not hold it moves the fingertip spheres: the Panda
    reads tens of millimetres of phantom disagreement when its finger is locked
    at 40 mm while the scene holds it at 5.6. The URDF's mimic joint means one
    number drives both fingers, so only the first is named -- by the robot
    (:attr:`~robosimstudio.specs.EmbodimentSpec.curobo_lock_joint`): a planner
    model that ends at the wrist, as the YAM's does, names none, and a joint
    injected there failed every first planner call whether or not a motion
    tool was ever asked for.
    """
    joint = str(getattr(getattr(env.scene.embodiment, "spec", None), "curobo_lock_joint", "") or "")
    if not joint:
        return {}
    name = env.handles.work_arm or next(iter(env.handles.arms), "")
    arm = env.handles.arms.get(name) if name else None
    coords = getattr(arm, "gripper_coords", ()) if arm is not None else ()
    if not coords:
        return {}
    import numpy as np

    return {joint: float(np.asarray(env.state.joint_q.numpy())[coords[0]])}


class NonPrivilegedRegistry:
    """A registry that drops privileged registrations instead of making them.

    ``register_sim_tools`` is called under ``if privileged``, but a suite's own
    registrar is not -- and a suite's tools are *more* privileged than the
    generic set, not less: ground-truth stage plans, particle counts, the
    evaluator's own metrics. So ``--no-privileged`` withheld
    ``sim.get_object_pose`` while handing the same graph ``sim.plan_fold``,
    which is the whole boundary defeated by the one tool that mattered.

    Fixing it here rather than in six ``tools.py`` files is deliberate. A suite
    registrar is content, written once per task and rarely reread; a rule that
    every one of them has to remember is a rule the seventh one breaks. This is
    the same argument :mod:`~robosimstudio_policy_agent.contract` makes for an
    allowlist over a convention, and it wants the same answer.

    **What counts as privileged**, either is enough:

    * a ``sim.`` name -- the prefix *is* the boundary, stated in
      :mod:`~robosimstudio_policy_agent.sim_tools`, and every one of the 26
      suite tools shipped today is one;
    * a ``privileged`` tag -- so a suite that reaches for ground truth under
      some other name is still caught.

    Dropped rather than refused, because the point of ``--no-privileged`` is to
    *run* the graph without those tools and see what it does. A node that calls
    one then fails on an unknown name, which is exactly what it does for the
    generic set today.
    """

    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self.dropped: list[str] = []
        """Names withheld, in registration order. Read by the contract test."""

    def _withheld(self, name: str, tags: Any) -> bool:
        if str(name).startswith("sim.") or "privileged" in tuple(tags or ()):
            self.dropped.append(str(name))
            return True
        return False

    def register_callable(self, name: str, fn: Any, **kwargs: Any) -> Any:
        if self._withheld(name, kwargs.get("tags")):
            return None
        return self._registry.register_callable(name, fn, **kwargs)

    def register_rpc(self, name: str, client: Any, **kwargs: Any) -> Any:
        if self._withheld(name, kwargs.get("tags")):
            return None
        return self._registry.register_rpc(name, client, **kwargs)

    def __contains__(self, name: object) -> bool:
        # Dunder lookups skip __getattr__, so ``in`` has to be spelled out.
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    _READERS = frozenset({"get", "invoke", "runtime_tools", "codegen_tools", "by_transport"})
    """The read-only part of the registry a suite registrar may reach through.

    An allowlist, not a denylist of registrar names: the registry has more
    doors than ``register*`` -- ``_discover_python``, the adapters' own
    ``register``, the ``_tools`` dict itself -- and a door this class does
    not know is a door it cannot check, so it is not forwarded at all."""

    def __getattr__(self, attr: str) -> Any:
        if attr not in NonPrivilegedRegistry._READERS:
            raise AttributeError(f"{attr!r} is not available to a registrar under --no-privileged")
        return getattr(self._registry, attr)


@dataclass
class NewtonEnvConfig:
    """The duck-typed config the runtime's ``Connector`` reads.

    Its fields are the runtime's, not this package's; they are filled from the
    task and the embodiment so a suite never states them twice.
    """

    arm_dof: int = 7
    num_arms: int = 1
    action_mode: str = "absolute_joints"
    control_freq: float = 50.0
    home_joints: list[float] | None = None
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tcp_rotation_z: float | None = None
    arm_bases: tuple = ()
    robot_urdf_path: str | None = None
    default_cameras: tuple[str, ...] = ()
    is_real: bool = False

    @classmethod
    def from_env(cls, env: Env) -> NewtonEnvConfig:
        embodiment = env.scene.embodiment
        handles = env.handles
        arms = list(handles.arms.values())
        # The *work* arm, not the first one. The two YAM cells list their arms
        # in opposite orders, so an index here hands the runtime the idle hand's
        # TCP offset and home posture -- which is a plausible-looking config that
        # aims every commanded grasp at the wrong end of the robot.
        name = handles.work_arm if handles.work_arm in handles.arms else next(iter(handles.arms), "")
        first = handles.arms.get(name) or (arms[0] if arms else None)
        home: list[float] = []
        if embodiment is not None and embodiment.spec.arms:
            spec = next((a for a in embodiment.spec.arms if a.name == name), embodiment.spec.arms[0])
            home = [float(v) for v in spec.home_q]
        return cls(
            arm_dof=len(first.arm_coords) if first else 7,
            num_arms=max(1, len(arms)),
            control_freq=env.task.physics.control_hz,
            home_joints=home or None,
            tcp_offset=first.ee_offset if first else (0.0, 0.0, 0.0),
            default_cameras=tuple(c.name for c in env.cameras),
        )


def _connector_base():
    """The pinned runtime's ``Connector``, or a message saying how to get it.

    The message distinguishes the two ways this fails, because they need
    different fixes and the generic one sent a reader looking in the wrong
    place. ``gap.runtime`` imports on gap_core + numpy + stdlib, so the runtime
    being *present* proves nothing about the connector: importing
    ``gap.connector.core`` runs ``gap/connector/__init__.py``, which pulls
    ``collector.py -> h5py`` and the runtime's pydantic models. Reporting that
    as "the runtime is not installed" is how you end up re-cloning a checkout
    that was never missing.
    """
    try:
        from gap.connector.core import Connector
    except ImportError as exc:  # pragma: no cover -- depends on the [gap] extra
        missing = getattr(exc, "name", "") or ""
        if missing.split(".")[0] not in ("gap", "gap_core", ""):
            raise ImportError(
                f"the graph-as-policy runtime is installed but {missing!r} is not, and "
                f"gap.connector.core needs it. The [gap] extra declares the runtime's own "
                f"dependencies (h5py, pydantic) for exactly this reason: install it with "
                f"'uv sync --extra gap', or put {missing!r} on PYTHONPATH."
            ) from exc
        raise ImportError(
            "the graph-as-policy runtime is not installed. Install the [gap] extra "
            "(uv sync --extra gap, or uv run --extra gap <script>), which pins gap and gap-core."
        ) from exc
    return Connector


class _ProprioceptiveFeedback:
    """Motion feedback reads encoders and FK; only an explicit observation renders.

    ``Connector.get_ee_pose`` / ``get_gripper_fraction`` / ``_current_joints``
    upstream each call ``get_observation``, and a trajectory asks for one per
    waypoint. On an RTX rig that path traces every camera in the scene to read a
    joint angle. These three read
    :meth:`~robosimstudio_policy_agent.facade.GapEnvFacade.get_proprioception` instead --
    the same encoders and FK, no render -- and a facade that does not offer one
    keeps the upstream behaviour.

    Prepended to ``ConnectorNewton``'s bases so it wins the lookup; it defines no
    ``_tool_get_gripper``, so the override below still reaches the base's through
    ``super()``.
    """

    def _robot_feedback(self):
        # Through ``self.facade``, which is what this attribute actually holds:
        # ``make_connector`` constructs ``ConnectorNewton(facade, ...)`` and gap's
        # Connector stores that as ``self.env``. Naming it ``facade`` here keeps
        # the fleet's world-view contract honest -- a WorldView answers neither
        # of these, and it never has to, because the object in hand is a facade.
        facade = getattr(self, "facade", None) or self.env
        reader = getattr(facade, "get_proprioception", None)
        # A facade without the narrower reader -- the fleet's WorldFacade -- keeps
        # the upstream behaviour rather than losing the feedback entirely.
        return reader() if reader is not None else facade.get_observation()

    def get_ee_pose(self, arm_id: int = 0):
        import numpy as np
        from gap_core.types import make_pose

        obs = self._robot_feedback()
        cart = np.asarray(obs.get(f"robot_cartesian_pos_{arm_id}", np.zeros(8)))
        position = self._base_frame_to_world(cart[:3].copy(), arm_id)
        rotation = cart[3:7] if len(cart) >= 7 else np.array([1.0, 0.0, 0.0, 0.0])
        return make_pose(position, rotation)

    def get_gripper_fraction(self, arm_id: int = 0) -> float:
        import numpy as np

        obs = self._robot_feedback()
        joints = np.asarray(obs.get(f"robot_joint_pos_{arm_id}", np.zeros(self._arm_dof + 1)))
        return float(joints[self._arm_dof]) if len(joints) > self._arm_dof else self._gripper_fraction

    def _current_joints(self, arm_id: int):
        import numpy as np

        if self.env is None:
            return None
        joints = self._robot_feedback().get(f"robot_joint_pos_{arm_id}")
        return None if joints is None else list(np.asarray(joints[: self._arm_dof], dtype=np.float64))


def make_connector(
    env: Env,
    *,
    bus: Bus | None = None,
    config: NewtonEnvConfig | None = None,
    privileged: bool = True,
    task_tools: Any = None,
    suite_tools: bool = True,
    roll: str = "auto",
    motion: str = "",
    facade: EnvFacade | None = None,
    ik: Any = None,
    servo: Any = None,
    capture_spans: Any = None,
    skills: Any = None,
) -> Any:
    """Build a connector for one environment.

    Args:
        env: The environment a graph will drive.
        skills: The registry roots the run executes with, as
            :func:`~robosimstudio_policy_agent.skills.pinned_registries` resolves them.
            Read by the in-process stand-ins for ``geometry.*`` and ``sam3.*``,
            which stand down when a registry in *this* list ships the real
            bundle -- the same list ``gap.execute`` is handed, so the decision
            and the run cannot disagree. ``None`` lets each stand-in resolve
            gap's registries for itself.
        bus: The bus mediating worker access. One is created if omitted.
        config: Overrides the config derived from the task.
        privileged: Register the ground-truth ``sim.*`` tools. The in-process
            mode's whole purpose; the remote bridge never sets it. False also
            filters the suite's own registrar -- see
            :class:`NonPrivilegedRegistry` for why that is not a separate switch.
        task_tools: ``(registry, env, bus) -> None``, for a suite's own tools.
            Handed a :class:`NonPrivilegedRegistry` when *privileged* is False,
            so a suite cannot hand a graph ground truth the generic set is
            withholding.
        suite_tools: Register the suite's own tools at all. False withholds
            them *while keeping the generic privileged set* -- the tier the
            generic oracle is an experimental treatment and a suite's scripted
            plan is not. The generic ``sim.*`` set is the same catalogue on
            every task (a pose, a bound, an OBB, a relation); a suite's own
            are written for one task and routinely hand over the answer:
            ``sim.plan_pick``/``sim.plan_place`` are the tool_hanging routine's
            own stations, ``sim.plan_insert`` the usbc schedule,
            ``sim.get_tab`` the peel's free end. A policy measured with those
            in reach is measuring the routine, not the policy.
        facade: The env surface the runtime drives, if one is already built. A
            batched run supplies one bound to a single world; omitted, the
            single-world facade over ``env`` is constructed here.
        ik: The IK backend, if one is already built.
        servo: The Cartesian move executor, if one is already built. Supplying it
            bypasses the ``motion`` argument entirely.

            These three exist so a batched run reuses *this* connector subclass
            rather than a second copy of it. Everything below -- the ``arm_id``
            forwarding the runtime drops, the servo overrides on both pose verbs,
            the selective ``go_home`` -- is behaviour a graph depends on, and two
            implementations of it would be two benchmarks.
        capture_spans: A :class:`~robosimstudio.record.capture.ToolSpanRecorder`,
            or ``None`` for no tape. Given one, every ``sim_step``-tagged call is
            wrapped so its span carries the request, the result and the exact
            step boundaries; ``begin`` returns ``None`` for anything else, so a
            read costs nothing.
        motion: ``"osc"`` (default) replaces both pose verbs with the reference
            benchmark's behaviour: a feedback servo that re-measures the end
            effector every control step and advances a setpoint toward the goal
            under ``OSC_POSE``'s own law, so the arm moves at a bounded Cartesian
            speed and tapers into the target instead of lunging at it. See
            :mod:`~robosimstudio_policy_agent.servo`. It is the default because a
            graph and a scripted routine that move differently are not two
            baselines on one benchmark, they are two benchmarks -- and the
            reference's motion is the one the tasks were authored against.

            ``"plan"`` keeps the runtime's own: one IK solve, then a joint-space
            move, which on a stiff position servo is a step input and moves fast
            and loose. Faster in wall clock and worth having for a graph being
            debugged for its *logic*, where how the arm gets there is noise.

            Either mode works with either IK backend. The servo tests arrival on
            whatever its backend actually solves -- an approach axis for the
            default DLS solver, the full rotation for pyroki -- so pairing is a
            question of whether a goal needs wrist-roll authority, not of which
            motion is selected.

    Returns:
        A ``ConnectorNewton``, which is the runtime's ``Connector`` over this
        environment.
    """
    base = _connector_base()
    bus = bus or SimBus()
    config = config or NewtonEnvConfig.from_env(env)
    facade = facade if facade is not None else GapEnvFacade(env, bus)
    # Published on the bus so a SUITE tool can reach it. The task-tool hook is
    # ``(registry, env, bus)``, which is enough for anything that reads state or
    # commands one arm -- and not enough for a tool that must drive two arms on
    # the same control period, because only the facade owns that write. The
    # cable weave needs exactly that (see ``cable.carry_with_feed``), and
    # widening the hook's signature would touch every suite registrar to serve
    # one of them.
    bus.facade = facade
    if ik is None:
        # The roll contract is the run's knob; the solver is the arm's, and the
        # backend falls back, saying so, where PyRoKi cannot serve it.
        ik = IKBackend(env, bus, roll=roll)
    planner_path = MOTION_PLANNERS.get(motion) or (motion if "." in motion else "")
    if not planner_path and motion not in (*SERVO_MODES, "plan"):
        raise ValueError(
            f"unknown motion {motion!r}: neither a servo mode ({[*SERVO_MODES, 'plan']}) nor a "
            f"registered planner ({list(MOTION_PLANNERS)}) nor a dotted path"
        )
    if planner_path:
        # A *peer* of the IK backend, not a replacement for it. `solve_ik` stays
        # on the cheap solver because CartesianServo calls it up to 400 times a
        # move; only the plans go to the planner. `ik.honours_roll` keeps
        # following the IK solver; describe_planner reports the planner's.
        from robosimstudio.paths import resolve_path

        spec = getattr(env.scene.embodiment, "spec", None)
        urdf = str(getattr(spec, "curobo_urdf", "") or "")
        # A config naming a directory is this checkout's, and only an absolute
        # path finds it; a bare filename is the planner's own -- cuRobo ships
        # `franka.yml` and resolves it against its content directory, which a
        # `resolve_path` here would turn into a file that does not exist.
        planner_config = str(getattr(spec, "curobo_config", "") or "")
        if planner_config and "/" in planner_config:
            planner_config = str(resolve_path(planner_config))
        planner_cls = _resolve(planner_path)
        try:
            planner = planner_cls(
                env,
                bus,
                **_accepted(
                    planner_cls,
                    {
                        "config_path": planner_config,
                        "urdf_path": str(resolve_path(urdf)) if urdf else "",
                        "lock_joints": _finger_lock(env),
                        "time_scale": float(getattr(spec, "curobo_time_scale", 1.0) or 1.0),
                        "ik": ik,
                    },
                ),
            )
            ik.planner = planner
        except NotImplementedError as exc:
            # Most embodiments have no planner model yet, and that is a routine
            # state rather than a broken install -- so it names the robot and
            # falls back, the same way the IK branch above does. Caught as
            # ``NotImplementedError`` rather than ``MotionUnavailable`` because a
            # planner named by dotted path cannot be expected to raise cuRobo's
            # private exception, and this layer must not import it to find out.
            print(f"[gap] {exc}; planning with {type(ik).__name__} instead")
    if servo is None and (motion in SERVO_MODES or planner_path):
        from .servo import CartesianServo

        servo = CartesianServo(env, facade, ik)

    class ConnectorNewton(_ProprioceptiveFeedback, base):  # type: ignore[valid-type, misc]
        """The runtime's connector, with Newton IK and optional ground truth."""

        @property
        def tool_registry(self):
            """The normal registry with an RSS-owned lifecycle observer."""
            if self._tool_registry is None:
                from gap_core.tools import ToolRegistry

                observer = capture_spans

                class ObservedToolRegistry(ToolRegistry):
                    def invoke(self, name, ctx=None, **kwargs):
                        descriptor = self.get(name)
                        token = observer.begin(name, descriptor.tags, ctx, kwargs)
                        try:
                            result = super().invoke(name, ctx, **kwargs)
                        except BaseException as exc:
                            observer.end(token, error=exc)
                            raise
                        observer.end(token, result=result)
                        return result

                # Wrapped only when there is a tape to write: an unrecorded run
                # gets the registry itself, not a subclass testing ``is None``
                # on every call.
                registry = ObservedToolRegistry() if observer is not None else ToolRegistry()
                self._register_robot_tools(registry)
                self._register_extra_tools(registry)
                self._tool_registry = registry
            return self._tool_registry

        def _register_extra_tools(self, registry) -> None:
            # Before the privileged branch, and unconditionally. A hand's own
            # geometry is proprioception -- a real driver knows its stroke and
            # its tool-centre point -- so a graph rewritten against these still
            # runs under --no-privileged and could still cross the bridge.
            # Registering them inside the branch would make every portable graph
            # privileged-only, which is the opposite of the point.
            register_robot_tools(registry, env, bus, ik=ik, facade=facade)
            # Unconditional for the same reason, and a stronger one: a geometry
            # tool is a pure function of the box it is handed, so there is
            # nothing in it to privilege. The grocery `gap` graph fits its box
            # from SAM3 masks and then calls straight into these.
            register_geometry_tools(registry, registries=skills)
            register_perception_tools(registry, registries=skills, bus=bus)
            # Unconditional. Registered here rather than under `privileged` for
            # the reason above: a plan is a question about the robot and a world
            # the caller supplies, so nothing in it is ground truth, and forcing
            # a planning graph to be privileged-only would make it unportable
            # for a reason having nothing to do with what it knows. Which of
            # motion.* actually appears depends on the backend -- free-space
            # planning needs a planner model, a straight line and a joint ease
            # need only IK.
            register_motion_tools(registry, ik)
            if servo is not None:
                registry.register_callable(
                    "robot.move_cartesian_until_contact",
                    self.move_cartesian_until_contact,
                    summary=(
                        "Move straight toward a pose and stop when the pose is reached or "
                        "proprioceptive progress indicates contact."
                    ),
                    tags=("sim_step",),
                )
            if privileged:
                register_sim_tools(registry, env, bus)
                # Privileged because it reads a live pose by name, not because
                # it reads the mesh: knowing a part's shape is a CAD model, and
                # a real cell has those for what it was built to handle.
                register_grasp_tools(registry, env, bus)
            if task_tools is not None and suite_tools:
                # Filtered under --no-privileged, for the reason
                # NonPrivilegedRegistry gives: a suite's own sim.* tools are
                # more privileged than the generic set, so withholding one and
                # not the other withheld nothing. Not registered at all under
                # --no-suite-tools, for the sharper half of the same reason:
                # the generic set is uniform across tasks, a suite's own is
                # this task's solution whatever its name or tags, and a study
                # of what a policy can learn has to withhold the second while
                # keeping the first.
                target = registry if privileged else NonPrivilegedRegistry(registry)
                task_tools(target, env, bus)
            # A skills checkout's geometry bundle exports names this connector
            # already serves in-process -- geometry_tools.py is that bundle's
            # stated stand-in -- and the pinned gap raises on the collision at
            # bundle boot, where newer gap builds let the first registration
            # stand. First wins here too: an RPC name an in-process tool
            # already claims is skipped, not fatal.
            first_registration = registry.register_rpc

            def _register_rpc_first_wins(name, *a, **k):
                try:
                    return first_registration(name, *a, **k)
                except ValueError as exc:
                    if "already registered" not in str(exc):
                        raise
                    return None

            registry.register_rpc = _register_rpc_first_wins

        def go_to_pose(
            self, pose, tolerance=0.0, max_steps=0, tcp_offset=None, arm_id=0, z_approach=0.0, nonblocking=False
        ):
            # The servo path replaces solve-once-then-lunge with the reference's
            # feedback loop. Nonblocking keeps the runtime's own semantics --
            # a servo is a loop, and "publish once and return" is not one.
            if servo is None or nonblocking:
                return super().go_to_pose(
                    pose,
                    tolerance=tolerance,
                    max_steps=max_steps,
                    tcp_offset=tcp_offset,
                    arm_id=arm_id,
                    z_approach=z_approach,
                    nonblocking=nonblocking,
                )

            # Unreachable comes back routed, not raised: see _routed_servo_move.
            def _move():
                if z_approach and z_approach > 0.0:
                    above = {"position": dict(pose["position"]), "rotation": dict(pose["rotation"])}
                    above["position"] = dict(above["position"])
                    above["position"]["z"] = float(above["position"]["z"]) + float(z_approach)
                    approach = servo.move(above, arm_id=arm_id)
                    # A pre-approach that timed out or touched something has not
                    # earned the descent: report it rather than go on down.
                    if (approach or {}).get("outcome", "reached") != "reached":
                        return {**approach, "phase": "approach"}
                # The runtime's own ``tolerance`` reaches the servo: 0 keeps its default ball.
                return servo.move(
                    pose,
                    arm_id=arm_id,
                    max_steps=max_steps or 0,
                    pos_tolerance=float(tolerance) if tolerance else None,
                )

            return _routed_servo_move(_move)

        def _tool_solve_ik(self, pose, tcp_offset=None, arm_id=0):
            """The runtime's solve, plus the numbers the verdict hides.

            The backend accepts a solve within its own 30 mm gate; the servo
            arrives within 4 mm. A graph that gated reachability on the
            ``success`` this used to return alone learned nothing about the
            26 mm between the two -- measured on one study cell, every
            reachability gate the improver wrote was blind to poses that
            ``plan_linear`` then called 160 mm out of reach. ``success`` keeps
            its meaning (a solution came back); ``position_error_m``,
            ``rotation_error_rad`` and ``within_servo_tolerance`` say how good.
            """
            out = super()._tool_solve_ik(pose, tcp_offset=tcp_offset, arm_id=arm_id)
            measured = getattr(ik, "last_solve_error", None)
            if measured is not None:
                pos_err, rot_err = measured
                out["position_error_m"] = float(pos_err)
                out["rotation_error_rad"] = None if rot_err is None else float(rot_err)
                out["within_servo_tolerance"] = bool(
                    pos_err <= _SERVO_POS_TOLERANCE and (rot_err is None or rot_err <= _SERVO_ROT_TOLERANCE)
                )
            return out

        def go_to_pose_cartesian(self, pose, arm_id=0, speed_scale=1.0):
            # The reference has no separate straight-line executor either: its
            # per-step clipped feedback *is* the line, near enough that the
            # held object never dips. Same here.
            #
            # ``speed_scale`` is checked HERE and refused as a routed result,
            # not left to the ValueError the servo raises for a direct caller.
            # A raise from inside a blocking motion primitive reaches the
            # workflow's ``abort`` recovery, whose first act on every shipped
            # graph is to open the gripper -- so a typo in a graph literal would
            # drop the held tool at the pick site and read, downstream, as "the
            # grasp failed". Same reasoning as the unreachable-pose refusal
            # above.
            try:
                scale = float(speed_scale)
            except (TypeError, ValueError):
                scale = float("nan")
            if not 0.0 < scale <= 1.0:
                return {
                    "reached": False,
                    "outcome": "bad_speed_scale",
                    "reason": f"speed_scale must be in (0, 1], got {speed_scale!r}",
                }
            if servo is None:
                return super().go_to_pose_cartesian(pose, arm_id=arm_id)
            return _routed_servo_move(lambda: servo.move(pose, arm_id=arm_id, speed_scale=scale))

        def move_cartesian_until_contact(
            self,
            pose,
            arm_id=0,
            max_steps=300,
            stall_steps=30,
            stall_epsilon=0.0002,
        ):
            """Straight contact transition using only commanded/observed motion.

            ``max_steps`` and ``stall_steps`` are CONTROL steps (1/60 s on every
            shipped scene): a 5 s budget and a 0.5 s no-progress window. They
            were 120 and 12 when a servo "step" was one loop turn, and a turn
            spanned however many simulator steps the free-running simulator took
            during the turn's solve and bus round-trip -- 2.3 on an idle GPU,
            3.0 at three processes per GPU -- so the graphs were written and
            measured against windows two to three times longer than the numbers
            said. The servo now counts every simulator step; the numbers state
            the durations the graphs always relied on.

            No fixture identity, object identity, force oracle, or simulator
            contact state is used, so the primitive transfers to real robot
            connectors that provide Cartesian feedback.
            """
            if servo is None:
                raise RuntimeError("contact-aware Cartesian motion requires feedback servo support")
            return _routed_servo_move(
                lambda: servo.move(
                    pose,
                    arm_id=int(arm_id),
                    max_steps=int(max_steps),
                    stop_on_stall=True,
                    stall_steps=int(stall_steps),
                    stall_epsilon=float(stall_epsilon),
                )
            )

        # ------------------------------------------------- the arm_id that was lost
        #
        # Three of the runtime's motion tools take no ``arm_id``, and the reason
        # this is worth overriding rather than living with is *how* it fails. The
        # tool adapter filters unknown keyword arguments against the signature
        # rather than raising, so ``arm_id=1`` passed by a node is silently
        # dropped and the call drives arm 0. Four shipped node scripts pass it
        # today. Nothing errors; the wrong hand simply moves, and on a bimanual
        # robot that is the *anchor* hand letting go of what it was holding.
        #
        # The underlying methods already take it -- only the tool-facing wrappers
        # do not -- so each of these forwards and nothing else.

        def _tool_execute_trajectory(self, trajectory, subsample=0, tolerance=0.0, max_steps_per_waypoint=0, arm_id=0):
            if not trajectory or not trajectory.get("waypoints"):
                from gap_core.errors import ToolError

                raise ToolError("robot.execute_trajectory", "trajectory required")
            self._execute_trajectory(
                trajectory,
                subsample if subsample > 0 else 1,
                tolerance if tolerance > 0 else 0.01,
                max_steps_per_waypoint if max_steps_per_waypoint > 0 else 120,
                arm_id=int(arm_id),
            )
            # The streamed path spends its per-waypoint budget and moves on, so
            # a physically blocked arm finishes the feed with no error raised
            # and nobody told: hang_v7b under-tracked a place by ~18 cm of hand
            # height this way, silently, for eight straight batches. Report the
            # final gap so the caller (and the feedback that reads returns) can
            # see a motion that completed its stream without arriving.
            #
            # The arm is resolved in role order, as every other arm_id is (see
            # robosimstudio.arms), and the joints are read on the owner thread
            # through the bus, not off live state from this graph thread.
            final = trajectory["waypoints"][-1]
            commanded = final.get("positions") if isinstance(final, dict) else final
            if commanded is None:
                return {"status": "ok"}
            toolkit = self.toolkit_env
            try:
                coords = list(toolkit.handles.arms[_arm_name(toolkit, int(arm_id))].arm_coords)
                live = bus.call(lambda: np.asarray(toolkit.state.joint_q.numpy(), dtype=np.float64)[coords])
                cmd = np.asarray(list(commanded), dtype=np.float64).reshape(-1)[: len(coords)]
            except Exception as exc:  # the motion ran; only the report of how well is missing
                _log.warning("execute_trajectory: final tracking error not measured: %s", exc)
                return {"status": "ok", "final_tracking_error_unavailable": f"{type(exc).__name__}: {exc}"}
            gap = float(np.max(np.abs(cmd - live[: len(cmd)]))) if len(cmd) else 0.0
            return {
                "status": "ok",
                "final_tracking_error_rad": round(gap, 4),
                "waypoints": len(trajectory["waypoints"]),
            }

        def _tool_move_to_joints(self, joint_config, tolerance=0.0, max_steps=0, arm_id=0):
            from gap.connector.core import _as_positions
            from gap_core.errors import ToolError

            positions = _as_positions(joint_config) if joint_config is not None else []
            if not positions:
                raise ToolError("robot.move_to_joints", "joint_config required")
            self.move_to_joints(
                positions,
                tolerance=tolerance if tolerance > 0 else 0.01,
                max_steps=max_steps if max_steps > 0 else 120,
                arm_id=int(arm_id),
            )

        def _tool_get_gripper(self, arm_id: int = 0):
            """The travel fraction, plus where the fingertips are right now.

            ``tip_reach_m`` is how far the pads sit past the tool-centre point
            along the approach axis at the current opening -- constant on a
            prismatic hand, ten millimetres of travel over the stroke on a
            2F-85 (:func:`~robosimstudio_policy_agent.robot_tools.fingertip_reach`). Off
            the published snapshot when there is one, so a graph polling it in
            a descent costs no bus round trip.
            """
            out = dict(super()._tool_get_gripper(arm_id=arm_id) or {})
            try:
                from .robot_tools import fingertip_reach

                toolkit = self.toolkit_env
                body_q = bus.snapshot.body_q
                if body_q is None:
                    body_q = bus.call(lambda: np.asarray(toolkit.state.body_q.numpy(), dtype=np.float64))
                reach = fingertip_reach(toolkit, _arm_name(toolkit, int(arm_id)), body_q=np.asarray(body_q))
                if reach is not None:
                    out["tip_reach_m"] = float(reach)
            except Exception:  # proprioception that cannot be read is left out, not invented
                pass
            return out

        def go_home(self, arm_id: int = -1):
            """Home one arm, or every arm when ``arm_id`` is negative.

            The runtime's own homes them all, which is right for a single-armed
            robot and wrong for the case this stack has: a support arm holding a
            cable against a peg is *parked on purpose*, and a node retracting the
            working hand has no business unparking it. Negative keeps the old
            behaviour, so nothing that did mean "all of them" changes.
            """
            if arm_id < 0:
                return super().go_home()
            # This arm's own home, not the work arm's: ``_home_joints`` is the
            # config's, which is the work arm's posture, and driving the support
            # arm to it sends it wherever the work arm's joints happen to point.
            toolkit = self.toolkit_env
            handle = toolkit.handles.arms[_arm_name(toolkit, int(arm_id))]
            home = [float(v) for v in getattr(handle, "home_q", ())]
            if not home:
                # No home of its own to go to: refuse to move it rather than
                # home every arm, which would unpark the one held on purpose.
                return {"homed": False, "reason": f"arm {int(arm_id)} declares no home posture"}
            self.move_to_joints(home, tolerance=0.01, max_steps=200, arm_id=int(arm_id))
            return None

    connector = ConnectorNewton(facade, config, camera_names=list(config.default_cameras), ik=ik)
    connector.bus = bus
    connector.facade = facade
    # NOT ``connector.env``: the runtime's Connector stores the facade there and
    # drives it (``env._step_once()``, ``env.compute_reward()``). Overwriting it
    # with the toolkit-layer Env swaps out the whole env surface, and the first
    # ``robot.*`` tool call fails on a missing private method.
    connector.toolkit_env = env
    return connector


# Exported under the prefix-first name the rest of this package refers to. The
# class itself is built inside make_connector because it must subclass a base
# that is only importable with the [gap] extra present.
ConnectorNewton = make_connector
