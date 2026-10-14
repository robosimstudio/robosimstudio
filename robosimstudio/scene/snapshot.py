# SPDX-License-Identifier: Apache-2.0
"""One step of a running scene, written down so somebody can argue with it.

**Not** :class:`robosimstudio.assetlib.semantics.snapshot.Snapshot`, which is the
*evaluator's* view of a world -- the handles and arrays a goal atom reads. This
is a debugging artefact: a JSON file of what the simulator currently believes,
aimed at the questions that come up when a scene looks wrong and nothing has
raised.

**Why a file rather than a print.** The failures this is for are the ones you
cannot see and cannot reproduce on demand: an object a few millimetres inside a
table, a pile that is still creeping after it looks still, a contact so soft that
a gripper drives straight through. Every one of them is a number nobody was
looking at, and each was found here by hand, twice, before anything wrote them
down. A snapshot is that hand-search, saved, timestamped and diffable against the
one you took before the change.

**What it records, and why each is on the list rather than a general dump:**

``penetration``
    How far every touching pair overlaps, worst first. The headline. A resting
    contact under a soft model always overlaps -- that overlap is how the force
    is generated -- so the figure is never zero and the only question is the
    magnitude. Hundredths of a millimetre is a healthy scene; whole millimetres
    is a contact model nobody set.
``bodies``
    Pose, speed and mass per object. "Is it where the document says" and "has it
    stopped" are the two questions a start pose has to answer, and a speed of
    zero next to a position that moved is how a limit cycle looks.
``materials``
    ``ke``, ``kd``, ``mu`` and the contact gap per colliding shape. On the list
    because a whole suite once ran four hundred times too soft and none of the
    other sections would have shown it: the poses were right, the penetration was
    small at rest, and the number that was wrong was never displayed anywhere.
``metrics``
    The evaluator's own readings, if the task has one, so a snapshot can be
    matched against what the goal panel was saying at the time.

Reading the model costs several device-to-host copies, so this is main-thread
work and the studio calls it from ``drain`` rather than from a viser callback.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["penetration_rows", "snapshot", "summarise", "write_snapshot"]

FLAGGED_PENETRATION = 0.001
"""Overlap worth complaining about in the summary line [m].

A millimetre. Chosen from the two sides it separates rather than as a round
number: a bin of sorted objects settles at 0.01-0.09 mm once their contact model is the
one their own files declare, and the bug that model was hiding showed as 7.6 to
10.4 mm. Nothing measured here has ever landed between 0.1 and 1 mm.
"""


def _label(model, shape: int) -> str:
    """A shape's own label, or something honest when it has none."""
    if shape < 0:
        return "world"
    try:
        return str(model.shape_label[shape])
    except (IndexError, TypeError):
        return f"shape{shape}"


def _owner(env, shape: int, bodies: dict[int, str], model, shape_body: np.ndarray | None = None) -> str:
    """The object a shape belongs to, falling back to the shape's label.

    An object name is what a reader recognises; a shape label is what is left
    when the shape belongs to the arena, the robot, or a collider substitute
    that no handle owns. ``shape_body`` is ``model.shape_body`` already on the
    host, for a caller naming many shapes: each read is a full device copy.
    """
    if shape < 0:
        return "world"
    if shape_body is None:
        shape_body = model.shape_body.numpy()
    body = int(shape_body[shape])
    named = bodies.get(body)
    return named if named else _label(model, shape).split("/")[-1]


def _bodies(env) -> dict[int, str]:
    return {h.body: name for name, h in env.handles.objects.items() if getattr(h, "body", -1) >= 0}


def _penetration(env, model, bodies: dict[int, str]) -> list[dict[str, Any]]:
    """Every overlapping pair, worst first, named.

    Empty when the scene has no MuJoCo contact probe -- a cloth-only scene, or a
    solver running with contacts handed in from outside -- rather than raising:
    a snapshot of a scene that cannot answer this is still worth having for
    everything else it says.
    """
    probe = getattr(env, "contacts", None)
    if probe is None:
        return []
    probe.update()
    out = []
    shape_body = None
    for shape_a, shape_b, depth in probe.penetrations():
        a, b = int(shape_a), int(shape_b)
        if shape_body is None:
            shape_body = model.shape_body.numpy()
        out.append(
            {
                "depth_mm": round(float(depth) * 1000.0, 4),
                "between": [_owner(env, a, bodies, model, shape_body), _owner(env, b, bodies, model, shape_body)],
                "shapes": [_label(model, a), _label(model, b)],
            }
        )
    return out


def penetration_rows(env) -> list[dict[str, Any]]:
    """Every overlapping pair in ``env`` as it stands, worst first, named.

    The ``penetration`` rows of :func:`snapshot` without the rest of it: each a
    ``depth_mm``, the two objects ``between`` and the two ``shapes``. Empty when
    the scene has no contact probe.
    """
    return _penetration(env, env.model, _bodies(env))


def _objects(env, model) -> list[dict[str, Any]]:
    """Pose, speed and mass per object handle."""
    body_q = env.state.body_q.numpy()
    body_qd = env.state.body_qd.numpy()
    mass = model.body_mass.numpy()
    out = []
    for name, handle in env.handles.objects.items():
        body = int(getattr(handle, "body", -1))
        if body < 0:
            continue
        pose = np.asarray(body_q[body], dtype=float)
        velocity = np.asarray(body_qd[body], dtype=float)
        out.append(
            {
                "name": name,
                "static": bool(getattr(handle, "static", False)),
                "position": [round(float(v), 6) for v in pose[:3]],
                "quat_xyzw": [round(float(v), 6) for v in pose[3:7]],
                # Split because they answer different questions: a pile that is
                # sliding and a pile that is buzzing in place look the same in a
                # single magnitude. Newton's body_qd is (linear, angular),
                # linear first (see SceneState.body_qd).
                "linear_speed": round(float(np.linalg.norm(velocity[:3])), 6),
                "angular_speed": round(float(np.linalg.norm(velocity[3:])), 6),
                "mass_kg": round(float(mass[body]), 6),
            }
        )
    return out


def _materials(env, model, bodies: dict[int, str]) -> list[dict[str, Any]]:
    """The contact model each *colliding* shape actually got.

    Colliding only: a visual shape's material is not consulted by anything, and
    listing it doubles a section whose whole value is being short enough to read.
    """
    import newton

    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    flags = model.shape_flags.numpy()
    ke, kd = model.shape_material_ke.numpy(), model.shape_material_kd.numpy()
    mu = model.shape_material_mu.numpy()
    gap = getattr(model, "shape_gap", None)
    gap = gap.numpy() if gap is not None else None
    shape_body = model.shape_body.numpy()
    out = []
    for shape in range(int(model.shape_count)):
        if not int(flags[shape]) & collide:
            continue
        out.append(
            {
                "shape": _label(model, shape),
                "owner": _owner(env, shape, bodies, model, shape_body),
                "ke": float(ke[shape]),
                "kd": float(kd[shape]),
                "mu": round(float(mu[shape]), 4),
                "gap": round(float(gap[shape]), 6) if gap is not None else None,
            }
        )
    return out


def _metrics(env, evaluate: bool) -> dict[str, Any]:
    """What the evaluator says, if the caller is willing to ask.

    Off by default, and that is the whole subtlety of this section.
    ``Env.evaluate`` is **stateful** -- ``once`` latches and ``hold-for``
    accumulates simulated seconds -- so a snapshot that called it would satisfy
    a five-second hold in four and a half by being taken. The studio drains its
    goal panel every step and the panel is the one caller allowed to evaluate,
    so in the studio this is left off and the readings come from there.

    A script stepping an environment itself has no such panel and nothing else
    is latching, so it may pass ``evaluate=True`` and pay the advance knowingly.
    """
    if not evaluate:
        return {}
    _terminated, _truncated, info = env.evaluate()
    return {
        "reward": _plain(info.get("reward")),
        "success": _plain(info.get("success")),
        "metrics": {k: _plain(v) for k, v in dict(info.get("metrics", {})).items()},
    }


def _plain(value: Any) -> Any:
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return round(float(value), 6)
    return str(value)


def snapshot(env, *, note: str = "", evaluate: bool = False) -> dict[str, Any]:
    """One step of ``env``, as plain data.

    Args:
        env: A built ``Env`` from the benchmark tier.
        note: Free text kept in the header -- what you were doing when you took
            it, which is the thing nobody remembers a day later.
        evaluate: Also score the step. **Advances the evaluator**, so it is off
            by default; see :func:`_metrics`.

    Returns:
        A JSON-able dict. Every length is metres and every angle a quaternion,
        except ``depth_mm``, which is millimetres because that is the scale the
        answer lands on and a reader should not have to count zeros.
    """
    model = env.model
    bodies = _bodies(env)
    penetration = _penetration(env, model, bodies)
    # A bare ``Sim`` has no task; its header still carries the step.
    task = getattr(env, "task", None)
    return {
        "taken": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note,
        "task": {
            "name": getattr(task, "full_name", None),
            "scene": getattr(task, "scene", None),
            "embodiment": getattr(task, "embodiment", None),
            "gripper": getattr(task, "gripper", None),
            # `env.episode` is the resolved Episode, not an index, so the id is
            # read off it rather than off the environment.
            "episode": int(getattr(getattr(env, "episode", None), "episode_id", 0) or 0),
            "step": int(getattr(env, "step_count", 0)),
            "instruction": getattr(task, "instruction", None),
        },
        "finite": bool(env.state_is_finite()),
        "worst_penetration_mm": penetration[0]["depth_mm"] if penetration else 0.0,
        "penetration": penetration,
        "bodies": _objects(env, model),
        "materials": _materials(env, model, bodies),
        "evaluation": _metrics(env, evaluate),
    }


def summarise(snap: dict[str, Any]) -> str:
    """One line for a readout: what a reader needs before opening the file."""
    worst = float(snap.get("worst_penetration_mm", 0.0))
    moving = [b for b in snap.get("bodies", ()) if not b["static"] and b["linear_speed"] > 0.01]
    parts = [f"step {snap['task']['step']}"]
    if not snap.get("finite", True):
        parts.append("STATE NOT FINITE")
    parts.append(f"worst overlap {worst:.3f} mm" + ("  <-- check" if worst > FLAGGED_PENETRATION * 1000 else ""))
    parts.append(f"{len(moving)} still moving")
    return ", ".join(parts)


def write_snapshot(env, directory, *, note: str = "", evaluate: bool = False) -> Path:
    """Take a snapshot and write it beside the run. Returns the file.

    Named for the step it was taken at rather than only the clock, so a
    directory of them sorts into the order they happened and two taken in the
    same second do not collide.
    """
    snap = snapshot(env, note=note, evaluate=evaluate)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"step{snap['task']['step']:06d}-{time.strftime('%H%M%S')}.json"
    path.write_text(json.dumps(snap, indent=2) + "\n")
    return path
