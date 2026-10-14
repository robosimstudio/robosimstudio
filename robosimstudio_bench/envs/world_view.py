# SPDX-License-Identifier: Apache-2.0
"""One world, presented as an environment to the tools built against one.

Every ``sim.*`` and suite tool is constructed with ``(registry, env, bus)`` and
reads a small, stable surface off that ``env``: ``task``, ``scene``, ``handles``,
``arm(name)``, ``gripper(name)``, ``state``, ``goal``, ``doc`` and
``make_context()``. Most of that is scene knowledge and is the same in every
world. Some of it is not, and the part that is not is exactly where a fleet
would otherwise answer every world's question with world 0's state.

**The surface is a duck, so it drifts silently.** A tool reaching a name that is
missing here does not degrade -- it takes the graph down.

So rather than hand the tools a fleet and hope they only ask world-independent
questions, each world's connector is built against a :class:`WorldView`. The
world-independent surface forwards; the world-dependent surface serves *this*
world's slice. Nothing above has to know which is which, and a tool that grows a
new state read gets the right world for free instead of silently getting world 0.

The state shim is the interesting part. A tool writes
``env.state.body_q.numpy()`` and expects a template-shaped array, because that is
what it has always been handed. :class:`_WorldState` gives it exactly that -- a
view into the fleet's array, offset to this world -- so the index a handle
resolved against the template lands on this world's body. That identity is
:class:`~robosimstudio.core.worlds.WorldMap`'s, and this is where the tools
collect on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.core.streams import reading

__all__ = ["WorldView"]


class _ArraySlice:
    """A device array's world slice, with the ``.numpy()`` a tool expects."""

    __slots__ = ("_array", "_per_world", "_world")

    def __init__(self, array, world: int, per_world: int) -> None:
        self._array = array
        self._world = int(world)
        self._per_world = int(per_world)

    @property
    def warp(self):
        """This world's slice as a ``wp.array``, for the tools that *copy* it.

        :meth:`__getattr__` forwards to the whole fleet's array, which is the
        right answer for ``dtype`` and ``shape`` and the wrong one for anything
        that reads the memory. A device-side op needs the array itself, and this
        wrapper is not one: ``wp.copy(dst, view)`` raises "Copy source and
        destination must be arrays". Warp slices the outer dimension into a view
        sharing the fleet's memory, so this costs nothing and stays live.

        Serially the same read yields a ``wp.array`` already, so a call site
        that wants either writes ``getattr(a, "warp", a)``.
        """
        start = self._world * self._per_world
        return self._array[start : start + self._per_world]

    def numpy(self) -> np.ndarray:
        # Serialized: this runs on a script thread, one per world, and a device
        # read moves the device's process-global current stream. See
        # :mod:`robosimstudio.core.streams`. Copies this world's slice only,
        # not the whole fleet's array.
        with reading():
            return self.warp.numpy()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._array, name)


class _WorldArm:
    """One world's arm solver: the fleet's template one, reached safely.

    A solver is kinematics and every world has the same ones, so there is one
    :class:`~robosimstudio.solvers.ik.arm_ik.ArmIK` per arm and every world's
    routine asks it. That was fine while a wave's routines ran one after another
    on the fleet's own thread. It stops being fine the moment they run on
    threads of their own, and not only because a device read moves the device's
    process-global current stream: **the solver is stateful**. It keeps a
    scratch ``newton.State`` that ``_eval_fk`` fills and ``ee_pose`` then reads,
    a host-FK depth counter that ``host_fk`` opens for whoever is inside it, and
    the last solve's error. Two worlds interleaved in there read each other's
    forward kinematics, and one world opening ``host_fk`` quietly moves every
    other world onto the host chain walk -- which
    :class:`~robosimstudio.solvers.ik.arm_ik._ChainFK` measured as a different
    set of solutions, not merely faster ones.

    So every call that touches the device goes through :attr:`_call`, and a
    driver decides what that means. With no driver it is the stream guard and
    nothing else, which is what a single-threaded caller has always effectively
    had. A driver that runs its routines on threads hands one in that puts the
    call somewhere only one thread runs -- the fleet's, where the solves of a
    whole wave can also be answered together.

    Everything that is *not* a device call -- ``arm_coords``, ``ee_body``,
    ``model``, ``iterations`` -- is delegated untouched, because it is a fact
    about the arm and reading it costs nothing and races with nothing.
    """

    OFFLOADED = ("solve_q_full", "solve_q", "solve_best", "pose_errors", "ee_pose", "fk_ee", "plan_cartesian")
    """What must not run on a routine's own thread.

    Every one of these evaluates forward kinematics on the shared solver. The
    list is explicit rather than "anything callable" so that a solver method
    added later is a decision somebody makes here, not a race nobody notices."""

    __slots__ = ("_arm", "_call", "_world")

    def __init__(self, arm: Any, world: int, call: Any = None) -> None:
        self._arm = arm
        self._world = int(world)
        self._call = call

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._arm, name)
        if name not in self.OFFLOADED:
            return attribute

        def offloaded(*args: Any, **kwargs: Any) -> Any:
            if self._call is not None:
                # The arm travels with the call: a pairing has more than one and
                # a driver answering for a wave has to know which it is being
                # asked about, not merely which world asked.
                return self._call(self._arm, self._world, name, args, kwargs)
            with reading():
                return attribute(*args, **kwargs)

        return offloaded

    def __repr__(self) -> str:
        return f"<world {self._world}'s {self._arm!r}>"


class _WorldState:
    """The fleet's state, sliced to one world and indexed like a template's."""

    __slots__ = ("_fleet", "_state", "_world")

    def __init__(self, fleet, state, world: int) -> None:
        self._fleet = fleet
        self._state = state
        self._world = int(world)

    @property
    def body_q(self):
        return _ArraySlice(self._state.body_q, self._world, self._fleet.worlds.bodies_per_world)

    @property
    def body_qd(self):
        return _ArraySlice(self._state.body_qd, self._world, self._fleet.worlds.bodies_per_world)

    @property
    def joint_q(self):
        return _ArraySlice(self._state.joint_q, self._world, self._fleet.worlds.coords_per_world)

    @property
    def joint_qd(self):
        return _ArraySlice(self._state.joint_qd, self._world, self._fleet.worlds.dofs_per_world)

    @property
    def particle_q(self):
        return _ArraySlice(self._state.particle_q, self._world, self._fleet.worlds.particles_per_world)

    @property
    def particle_qd(self):
        return _ArraySlice(self._state.particle_qd, self._world, self._fleet.worlds.particles_per_world)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._state, name)


class _WorldPipeline:
    """The fleet's pipeline, with the two states sliced to one world.

    ``WorldView.state`` has answered per world since the view existed; ``.pipeline``
    handed back the fleet's own, and a tool reaching through it read the whole
    array and indexed it with a *template* index -- which is world 0's row, in
    every world, silently. The scripted routine does exactly that twelve times:
    it plans the press off ``pipeline.state_0.body_q[bar_body]`` and the pick off
    ``pipeline.state_0.particle_q[sponge.particles]``. Every world planned
    against world 0's bench, so a wave of drawn layouts scored world 0 and
    eleven near-misses -- which is what 1 of 12 was.

    Invisible until the worlds differed: at one layout world 0's bench IS every
    world's bench, which is every replicated run this path had ever done.

    Everything else forwards, because everything else about a pipeline is the
    fleet's: one solver, one graph, one clock.
    """

    __slots__ = ("_fleet", "_pipeline", "_world")

    def __init__(self, fleet, pipeline, world: int) -> None:
        self._fleet = fleet
        self._pipeline = pipeline
        self._world = int(world)

    @property
    def state_0(self):
        return _WorldState(self._fleet, self._pipeline.state_0, self._world)

    @property
    def state_1(self):
        return _WorldState(self._fleet, self._pipeline.state_1, self._world)

    @property
    def state(self):
        return _WorldState(self._fleet, self._pipeline.state, self._world)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pipeline, name)


class _WorldGripper:
    """One world's hand, behind the ``.command`` a tool reads and writes.

    A tool treating a gripper as a scalar it can set is right in a single world
    and would be a second writer here -- the fleet's bank owns every world's
    command, and the last tool to write a shared object would decide what all of
    them were driven to.
    """

    __slots__ = ("_bank", "_template", "_world")

    def __init__(self, bank, template, world: int) -> None:
        self._bank = bank
        self._template = template
        self._world = int(world)

    @property
    def command(self) -> float:
        return float(self._bank.commands()[self._world]) if self._bank is not None else 0.0

    @command.setter
    def command(self, value: float) -> None:
        if self._bank is not None:
            self._bank.set(self._world, float(value))

    def apply(self, joint_target_q) -> None:
        """A no-op: the fleet's control stage writes every world's fingers.

        Present because the single-world gripper has it and tools call it after
        setting a command. Writing here would be a second writer on the same
        coordinates, one world at a time, outside the captured stage.
        """
        del joint_target_q

    def __getattr__(self, name: str) -> Any:
        return getattr(self._template, name)


class WorldView:
    """The environment surface of one world of a fleet.

    Args:
        fleet: The batched environment.
        world: Which world this view answers for.
    """

    def __init__(self, fleet, world: int) -> None:
        self.fleet = fleet
        self.world = int(world)
        self._evaluator: Any = None
        self._clearance: Any = None
        self._hooks: list = []
        self.arm_call: Any = None
        """Where this world's solver calls go, or ``None`` for "right here".

        Set by a driver that runs its routines on threads of their own: the arm
        solver is shared and stateful, so its device work has to happen
        somewhere only one thread is. Called as
        ``(arm, world, name, args, kwargs)``. See :class:`_WorldArm`."""

    # --------------------------------------------------------- this world's own
    @property
    def task(self):
        """The document **this world** runs, which in a mixed wave is its own.

        Was ``fleet.task`` -- the document the fleet happened to be opened from.
        That was right while a wave was ``W`` copies of one document and is wrong
        now that the worlds of one wave run different ones: see
        :meth:`~robosimstudio_bench.envs.fleet.FleetEnv.world_document`, which
        never answers ``None`` so this needs no branch.
        """
        document = getattr(self.fleet, "world_document", None)
        return document(self.world) if callable(document) else self.fleet.task

    @property
    def _task_evaluator(self):
        """An evaluator for the questions that are this world's *document's*.

        :attr:`doc` and :attr:`goal` are properties of a document, and in a mixed
        wave each world has its own -- so this is built from :attr:`task`, which
        is that document. Built lazily, on the first graph that asks, and only
        that graph pays for it.

        **The bug this replaced, because it is the shape of every mixed-document
        defect.** This used to build from ``fleet.task`` on the stated ground
        that "every world of a wave is running the same one". Once they are not,
        every world's graph reads world 0's goals, masked to its own active set
        (:func:`~robosimstudio_bench.goals.goal.mask_condition`) -- so a world
        whose document shares no object with document 0 gets a goal with **no
        leaves**, its ``sim.next_goal`` answers "done" on the first call, and the
        episode closes at ~61 control steps having done nothing, with no error
        anywhere. Measured on a four-document ``tool_sort/level1`` wave: world 0
        (the opened document) scored 0.75, world 2 scored exactly 0.25 because
        ``level1_13`` shares exactly one object with ``level1_1``, and the two
        worlds sharing none scored 0.00 in 61 steps. The per-world *score* was
        right the whole time -- the runner's own evaluator is built per world --
        which is what made it look like a policy failure rather than a wiring one.

        A fleet has no evaluator to forward to, which is the whole reason this
        exists: ``Env`` answers both off the instance it already holds, and a
        tool written against that surface used to die on a wave with
        ``'WorldView' object has no attribute 'evaluator'`` -- every world's
        graph, on its first goal.
        """
        # Keyed on the document as well as the active set: the document is what
        # the goal is read from, and the active set is what it is masked to. A
        # view that outlives the wave would otherwise hand the next episode's
        # cursor the previous episode's placements.
        document = self.task
        active = getattr(self.episode, "active", None)
        key = (getattr(document, "full_name", None), active)
        if self._evaluator is None or self._evaluator[0] != key:
            from robosimstudio_bench.goals.context import EvaluatorNone, evaluator_for

            built = evaluator_for(document, self.episode)
            self._evaluator = (key, None if isinstance(built, EvaluatorNone) else built)
        return self._evaluator[1]

    @property
    def clearance(self):
        """How close things are at any configuration, without stepping.

        World-independent, and not by luck: the probe is built on the scene's
        ``index_model``, which is the *template* even when ``model`` is a wave,
        and it is asked with template shape indices and a template-shaped
        ``body_q``. So one probe answers for every world, and it is built on
        the first ask because walking the collision filtering is not work to
        repeat per world. See :attr:`Env.clearance`.
        """
        if self._clearance is None:
            from robosimstudio.core.probes.clearance import ClearanceProbe

            self._clearance = ClearanceProbe(self.fleet.scene.index_model)
        return self._clearance

    @property
    def doc(self):
        """The document this task was defined by, or ``None``. See :attr:`Env.doc`."""
        return getattr(self._task_evaluator, "doc", None)

    @property
    def goal(self):
        """The compiled goal tree, or ``None``. See :attr:`Env.goal`."""
        return getattr(self._task_evaluator, "goal", None)

    @property
    def scene(self):
        return self.fleet.scene

    @property
    def handles(self):
        """This world's handles: the template's, restricted to what this world's episode runs.

        World-dependent after all, and only in what it *lists*: every handle
        is the template's own object, so an index resolved through it is right
        for this world's slice exactly as before. What a superset instance
        parks is absent from ``objects`` and from the regions that hang off
        it, which is what keeps ``sim.list_objects`` and the goal cursor from
        naming a body six metres in the air. See :meth:`FleetEnv.handles_for`.
        """
        return self.fleet.handles_for(self.world)

    @property
    def model(self):
        return self.fleet.model

    @property
    def pipeline(self):
        return _WorldPipeline(self.fleet, self.fleet.pipeline, self.world)

    @property
    def cameras(self):
        return self.fleet.cameras

    @property
    def max_steps(self) -> int:
        return int(self.fleet.max_steps)

    @property
    def step_count(self) -> int:
        return int(self.fleet.step_count)

    @property
    def sim_time(self) -> float:
        return self.fleet.step_count * self.fleet.control_dt

    def arm(self, name: str = ""):
        """One arm's solver, on the template model, as this world may use it.

        World-independent in the way that matters: a solver is kinematics, and
        every world has the same ones. What differs is the *seed* it is given,
        which its caller supplies.

        What comes back is a :class:`_WorldArm` rather than the solver itself,
        because "every world has the same one" is exactly the problem once a
        wave's routines run on threads of their own -- the same one is also the
        same scratch state. The proxy is transparent for everything that is a
        fact about the arm, and routes everything that touches the device.
        """
        iks = self.fleet.scene.embodiment.iks if self.fleet.scene.embodiment else {}
        if not iks:
            raise RuntimeError(f"task {self.task.full_name!r} has no arms")
        arm = iks[name] if name else next(iter(iks.values()))
        return _WorldArm(arm, self.world, self.arm_call)

    # --------------------------------------------------------- world-dependent
    @property
    def episode(self):
        return self.fleet.episodes[self.world] or self.fleet.episode

    @property
    def contacts(self):
        """This world's contacts, from the fleet's last batched read.

        The attribute ``sim.check_grasp`` reads off an environment. Absent, it
        answers "not held" for every grasp in every world -- which is not a
        degraded result but an impossible one: the verify node routes to give-up
        and no episode can succeed however well the hand actually closed.
        """
        probe = self.fleet.contacts
        return probe.view(self.world) if probe is not None else None

    @property
    def state(self):
        """This world's state, indexed exactly as a single-world model's is."""
        return _WorldState(self.fleet, self.fleet.pipeline.state_0, self.world)

    def gripper(self, name: str = ""):
        """This world's hand for one arm."""
        grippers = self.fleet.scene.embodiment.grippers if self.fleet.scene.embodiment else {}
        if not grippers:
            raise RuntimeError(f"task {self.task.full_name!r} has no grippers")
        key = name or next(iter(grippers))
        return _WorldGripper(self.fleet.grippers.get(key), grippers[key], self.world)

    def idle_action(self, mode: str = "") -> Any:
        """An action that holds *this world* where it is, in ``mode``'s layout.

        Off this world's slice of the control vector rather than the whole of
        it: a caller sizing its arm from ``len(idle_action(mode))`` would
        otherwise measure the fleet and conclude the robot has sixteen times as
        many joints as it has.

        The layout is :class:`~robosimstudio.solvers.control.action.ActionSpace`'s
        -- one arm's coordinates then its gripper command, per arm -- and it is
        assembled here rather than through a space, because a space writes one
        world's control vector and this reads one world's out of many.
        """
        embodiment = self.fleet.scene.embodiment
        if embodiment is None:
            return None
        del mode  # one layout is built here; a wave drives joint positions
        # The fleet's copy, not this world's own read: every world in the wave
        # asks for this every control step, and the transfer is the whole array
        # whoever asks. See ``FleetEnv.host_joint_q``.
        coords = self.fleet.worlds.coord_view(self.fleet.host_joint_q(), self.world)
        parts = []
        for name, index in embodiment.arms.items():
            parts.append(np.asarray(coords[index.arm_coords], dtype=np.float32))
            bank = self.fleet.grippers.get(name)
            parts.append(np.array([float(bank.commands()[self.world]) if bank is not None else 0.0], dtype=np.float32))
        return np.concatenate(parts) if parts else None

    @property
    def post_step_hooks(self) -> list:
        """This world's after-the-step hooks, which the driver runs.

        A list per world, not one shared: a hook is a thing a *policy* installed
        mid-episode -- a grasp pin that has to be re-asserted every step -- and
        one world's pin is not another's. Sharing the list would have every
        world's hook run once per world per step, against whichever world the
        hook happened to be built for.

        Held here rather than on the fleet because this is the object a routine
        reaches for, and the fleet has no per-world anything to hang it on.
        """
        return self._hooks

    def make_context(self):
        """This world's scoring inputs, for a tool that measures the task."""
        snapshot = self.fleet.snapshot()
        settled = self.fleet.settled_body_q(self.world)
        return self.fleet.context(self.world, snapshot, settled_body_q=settled)
