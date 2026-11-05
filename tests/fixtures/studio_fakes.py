# SPDX-License-Identifier: Apache-2.0
"""Stand-ins for what a studio panel reads: an arm panel, an env, and a session.

A panel is bookkeeping between viser's callbacks and its own ``drain``, and the
part of it worth a fast test -- which arm a routine takes, where a reach sends
the cursor, which panels a selection keeps -- needs no scene, no IK solve and
no browser. These hold the state a panel asks for and record what it does,
the way :mod:`tests.fixtures.viser_stubs` does for the browser; they compute
nothing a real panel would, so they cannot agree with it by accident.

:class:`FakeArmPanel`
    An arm panel as a routine or a policy runner sees it: a cursor, ``aim``,
    an ``ik`` whose ``ee_pose`` is where the hand is, ``suspend``/``resume``
    (counted, and logged in order) and a ``grip_request``. With
    ``tracks=True`` the hand is wherever the cursor was last put -- an arm that
    keeps up; with ``tracks=False`` it stays where it was -- one that cannot.
:class:`FakeEnv`
    A control rate, a state, a step count, the robot's arms and grippers, the
    handles and the cameras -- enough to construct a real
    :class:`~robosimstudio_studio.panels.PanelArm`.
:class:`FakeSession`
    :class:`~tests.fixtures.viser_stubs._StubSession` and the rest of what a
    panel asks of its session: the arm panels, the panel list, ``select``,
    ``each_step`` and the driver and record slots.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tests.fixtures.viser_stubs import _StubGui, _StubSession

__all__ = ["FakeArmPanel", "FakeEnv", "FakeGripper", "FakeIK", "FakeSession", "FakeSpace"]

HOME = ((0.3, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
"""Where a fake hand starts: 300 mm out, 200 mm up, unrotated (xyzw)."""


class FakeIK:
    """An arm's IK as a panel calls it: a home pose, the hand's pose now, and the targets it was set."""

    def __init__(self, position=HOME[0], quat=HOME[1]) -> None:
        self.position = np.array(position, dtype=np.float64)
        self.quat = np.array(quat, dtype=np.float64)
        self.home_pose = (self.position.copy(), self.quat.copy())
        self.solver = object()
        """Set, as an LM solver is: :class:`PanelArm` then drives through its setpoint law every step."""
        self.targets: list[tuple[np.ndarray, np.ndarray]] = []
        self.solves = 0
        self.holds = 0

    def ee_pose(self, state=None) -> tuple[np.ndarray, np.ndarray]:
        return self.position.copy(), self.quat.copy()

    def set_target(self, position, quat) -> None:
        self.targets.append((np.array(position, dtype=np.float64), np.array(quat, dtype=np.float64)))

    def solve_into(self, joint_q, joint_target_q) -> None:
        self.solves += 1

    def hold_into(self, joint_q, joint_target_q) -> None:
        self.holds += 1


class FakeGripper:
    """A hand's fingers: the command last written, and every write ``apply`` made."""

    def __init__(self, command: float = 0.0, open_width: float = 0.08) -> None:
        self.command = float(command)
        self.open_width = float(open_width)
        self.applied: list[float] = []

    def apply(self, joint_target_q) -> None:
        self.applied.append(self.command)

    def width(self, state=None) -> float:
        return self.open_width * (1.0 - self.command)


class FakeArmPanel:
    """An arm panel as a routine, a policy runner or a leader arm sees it."""

    def __init__(
        self,
        name: str = "right",
        position=HOME[0],
        quat=HOME[1],
        *,
        suspended: bool = False,
        tracks: bool = True,
    ) -> None:
        self.name = name
        self.ik = FakeIK(position, quat)
        self.cursor_pos, self.cursor_quat = self.ik.ee_pose()
        self.tracks = tracks
        self.suspended = suspended
        self.suspend_calls = 0
        self.resume_calls = 0
        self.calls: list[str] = []
        """``"suspend"`` and ``"resume"``, in the order they were called."""
        self.aims: list[tuple[np.ndarray, np.ndarray, bool]] = []
        """Every ``aim``: the cursor's position, its quaternion and whether the drag was still moving."""
        self.grip_request: float | None = None
        self.dragging = False

    def aim(self, position, quat=None, *, moving: bool = True) -> None:
        self.cursor_pos = np.array(position, dtype=np.float64)
        if quat is not None:
            self.cursor_quat = np.array(quat, dtype=np.float64)
        self.dragging = bool(moving)
        self.aims.append((self.cursor_pos.copy(), self.cursor_quat.copy(), bool(moving)))
        if self.tracks:
            self.ik.position, self.ik.quat = self.cursor_pos.copy(), self.cursor_quat.copy()

    def suspend(self) -> None:
        self.suspended = True
        self.suspend_calls += 1
        self.calls.append("suspend")

    def resume(self) -> None:
        self.suspended = False
        self.resume_calls += 1
        self.calls.append("resume")


class FakeSpace:
    """A control space as the leader arm's preflight reads it: joints per arm, and its arms."""

    def __init__(self, per_arm: int) -> None:
        self.per_arm = per_arm
        self.arms: list = []


class FakeEnv:
    """An env as the studio's panels and the leader arm's preflight read it.

    ``arms`` are the robot's arm names in declaration order, which is the
    order the action vector takes them in; ``per_arm`` is how many joints the
    ``joint_position`` control space gives each.
    """

    def __init__(
        self,
        arms: tuple[str, ...] = (),
        per_arm: int = 0,
        *,
        control_hz: float = 50.0,
        grippers: bool = True,
        cameras: tuple = (),
        obs: dict | None = None,
        work_arm: str = "",
    ) -> None:
        self.control_hz = float(control_hz)
        self.request = SimpleNamespace(physics=SimpleNamespace(control_hz=float(control_hz)))
        self.state = SimpleNamespace(joint_q=None, body_q=None, particle_q=None)
        self.control = SimpleNamespace(joint_target_q=None)
        self.step_count = 0
        self.iks = {name: FakeIK() for name in arms}
        self.grippers = {name: FakeGripper() for name in arms} if grippers else {}
        self.scene = SimpleNamespace(
            embodiment=SimpleNamespace(iks=self.iks, grippers=self.grippers, suctions={}, arms={})
        )
        self.handles = SimpleNamespace(
            arms=tuple(arms), work_arm=work_arm, objects={}, statics={}, regions={}, deformables={}
        )
        self.cameras = tuple(cameras)
        self.obs = dict(obs or {})
        self.last_result = None
        self._space = FakeSpace(per_arm)

    def arm(self, name: str) -> FakeIK:
        return self.iks[name]

    def use_control(self, mode: str) -> FakeSpace:
        return self._space


class FakeSession(_StubSession):
    """:class:`_StubSession` and the rest of a session's surface a panel or the leader arm reaches for."""

    def __init__(self, env=None, arms: dict | None = None, *, gui=None) -> None:
        super().__init__(env if env is not None else FakeEnv(), gui if gui is not None else _StubGui())
        self.arms = dict(arms or {})
        self.panels: list = []
        self.hooks: list = []
        self.leader_arm = None
        self.quit_requested = False
        self.closed = 0
        self.selected = None
        self.selections: list = []
        self.driver = None
        self.record = None
        self.picker = None
        self.reset_requested = False

    def each_step(self, fn) -> None:
        self.hooks.append(fn)

    def close(self) -> None:
        self.closed += 1

    def select(self, subject) -> None:
        self.selected = subject
        self.selections.append(subject)
