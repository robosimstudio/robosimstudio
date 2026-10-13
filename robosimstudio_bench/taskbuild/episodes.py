# SPDX-License-Identifier: Apache-2.0
"""Making episodes: deterministic, and reproducible one at a time.

The rule this module exists to enforce is that **an episode depends on its seed
and nothing else**. The seed is ``spec.seed + episode_id`` (:func:`episode_seed`),
so an episode samples the same way whether it is run first, run alone, or run
after a thousand others. Anything else makes a failure impossible to reproduce
and a re-render impossible to trust.

The seed, not the index, is what a draw may read. A benchmark instance is its base document at
one seed (``(:count 1) (:seed k)``), which is always episode 0 of a document of
its own, so a sampler that reads ``episode_id`` reports the same draw for every
instance in the set. A sampler that enumerates rather than draws asks
:func:`episode_seed` for the number to enumerate by.

Samplers are plain functions a task references directly. The three here cover
what the initial suites need -- fixed layouts, a jittered grid, a near-placement
-- and a task with a stranger requirement writes its own beside itself rather
than extending a shared vocabulary it would then have to share.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np

from robosimstudio.specs import Episode, TaskSpec, replace

__all__ = [
    "active_names",
    "episode_range",
    "episode_seed",
    "fixed",
    "grid_xy_quat",
    "make_episode",
    "model_key_for",
    "move_near",
    "pick_instruction",
    "with_base_seed",
]


def episode_seed(task: TaskSpec, episode_id: int) -> int:
    """The seed episode ``episode_id`` of ``task`` draws at: ``task.episodes.seed + episode_id``.

    The one number an episode's draw may depend on, and the reason it is a
    function rather than a line inside :func:`make_episode`: a sampler that
    enumerates a table instead of drawing from the generator needs the same
    number, and computing it a second way is how the two come apart. A
    benchmark instance -- one document at ``(:count 1) (:seed k)`` -- is
    episode 0 of itself and episode ``k - base seed`` of the document it was
    written from, and it is this number that is equal across the two.

    Args:
        task: The variant.
        episode_id: Index. Outside ``range(task.episodes.count)`` is legal, as
            it is for :func:`make_episode`.

    Returns:
        The seed.
    """
    return int(task.episodes.seed) + int(episode_id)


def with_base_seed(task: TaskSpec, seed: int) -> TaskSpec:
    """``task`` with its base seed replaced by ``seed``, or ``task`` itself for ``-1``.

    How ``JobSpec.seed`` reaches an episode. The override is applied to the
    document rather than threaded beside it, so :func:`episode_seed` -- and
    every sampler that asks it for the number to enumerate by -- sees the one
    base seed the job asked for, whichever runner made the episode. ``-1`` keeps
    the very object passed in, so identity checks downstream are undisturbed.
    """
    seed = int(seed)
    if seed == -1 or seed == int(task.episodes.seed):
        return task
    return replace(task, episodes=replace(task.episodes, seed=seed))


def make_episode(task: TaskSpec, episode_id: int) -> Episode:
    """Build episode ``episode_id`` of a task.

    Args:
        task: The variant to sample from.
        episode_id: Index. Values outside ``range(task.episodes.count)`` are
            allowed and keep sampling -- an eval job may ask for more episodes
            than a task nominally declares, and refusing would be arbitrary.

    Returns:
        The :class:`~robosimstudio.specs.Episode`, with its model key
        computed from whichever sampled fields the task said change the model.
    """
    spec = task.episodes
    seed = episode_seed(task, episode_id)
    overrides: dict[str, Any] = {}
    if spec.sampler is not None:
        rng = np.random.default_rng(seed)
        overrides = dict(spec.sampler(task, episode_id, rng) or {})
    instruction = str(overrides.pop("instruction", "") or task.instruction)
    if "active" in overrides:
        overrides["active"] = active_names(task, overrides["active"])
    return Episode(
        episode_id=int(episode_id),
        seed=seed,
        instruction=instruction,
        overrides=tuple(sorted(overrides.items())),
        model_key=model_key_for(task, overrides, instruction=instruction),
    )


def active_names(task: TaskSpec, names: Any) -> tuple[str, ...]:
    """A sampler's ``active`` value as the episode carries it: sorted, unique, every name the task's.

    Sorted so two draws of the same set are the same override -- the tuple is
    what a wave compares to decide whether a model was built for a bench --
    and checked against the task's entities so a sampler naming an object the
    document does not have fails here, at the draw, rather than as a
    ``KeyError`` from whichever world happened to reach for it first.
    """
    if isinstance(names, str):
        names = (names,)
    wanted = tuple(sorted({str(name) for name in names}))
    known = {getattr(spec, "name", "") for spec in task.entities}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise ValueError(
            f"task {task.full_name!r}: the sampler's active set names {unknown}, which the task does not "
            f"declare; it has {sorted(known)}"
        )
    return wanted


def model_key_for(task: TaskSpec, overrides: dict[str, Any], *, instruction: str = "") -> str:
    """Identity of the model an episode needs.

    Two episodes with the same key differ only in state, so an environment
    re-poses instead of rebuilding -- which is the difference between a
    hundred-episode eval taking minutes and taking an hour. The key covers the
    variant plus whichever sampled fields the task declared as model-changing;
    an instruction never is, since language does not change geometry.
    """
    del instruction
    # Scene and embodiment are part of the model's identity, not just of its
    # provenance: the same variant staged on two tables or driven by two robots
    # is two different models, and a key that ignored them would let a cache --
    # or a reset that re-poses instead of rebuilding -- hand back the wrong one.
    parts: dict[str, Any] = {"task": task.full_name, "scene": task.scene, "embodiment": task.embodiment}
    for field_name in task.episodes.model_key_fields:
        parts[field_name] = overrides.get(field_name)
    blob = json.dumps(parts, sort_keys=True, default=repr)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def episode_range(task: TaskSpec, start: int = 0, stop: int | None = None) -> list[Episode]:
    """Build a contiguous run of episodes."""
    stop = task.episodes.count if stop is None else stop
    return [make_episode(task, i) for i in range(start, stop)]


# ---------------------------------------------------------------- samplers
def fixed(task: TaskSpec, episode_id: int, rng: np.random.Generator) -> dict[str, Any]:
    """No variation: every episode is the task's own layout.

    The right default for a task whose scene came from an export that already
    records where everything settled -- re-sampling those poses would discard
    the one arrangement known to be physically consistent.
    """
    del task, episode_id, rng
    return {}


def grid_xy_quat(
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    grid: tuple[int, int] = (3, 3),
    yaw_choices: Sequence[float] = (0.0,),
    objects: Sequence[str] = (),
    jitter: float = 0.0,
    keep_rest_orientation: bool = False,
):
    """A sampler placing named objects on a grid of positions and yaws.

    Enumerates rather than randomises: the episode at seed ``k``
    (:func:`episode_seed`) takes the ``k``-th cell of the ``grid x yaw`` product, so a run of ``n`` episodes covers the layout
    space evenly instead of clumping the way independent draws do. ``jitter``
    adds a small random offset on top, which is what keeps a policy from
    memorising exact cell centres.

    Args:
        x_range: Low and high x of the grid [m].
        y_range: Low and high y [m].
        grid: Cells in x and y.
        yaw_choices: Yaw angles [rad] to enumerate alongside the cells.
        objects: Which objects to place. Empty places every non-static object.
        jitter: Uniform position noise half-width [m].
        keep_rest_orientation: Compose each yaw with the object's *declared*
            orientation instead of replacing it. ``object_poses`` rebuilds a
            pose as pure yaw about world z, which is right for anything that
            rests yaw-symmetrically — a can, a box — and wrong for a tool
            whose settled pose carries roll: a hammer respawned yaw-only lies
            un-settled and topples into a tilt no layout asked for. With this
            on, the sampler emits full ``object_xforms`` of the settled
            quaternion yawed about world z, so every cell is the declared rest
            pose *turned*, not a new physics problem.

    Returns:
        A sampler callable for :attr:`EpisodeSpec.sampler`.
    """
    nx, ny = grid
    cells = [
        (
            x_range[0] + (x_range[1] - x_range[0]) * (i + 0.5) / nx,
            y_range[0] + (y_range[1] - y_range[0]) * (j + 0.5) / ny,
        )
        for i in range(nx)
        for j in range(ny)
    ]
    layouts = [(cell, yaw) for cell in cells for yaw in yaw_choices]

    def sample(task: TaskSpec, episode_id: int, rng: np.random.Generator) -> dict[str, Any]:
        names = list(objects) or [o.name for o in task.objects if not o.static]
        poses: dict[str, tuple[float, float, float, float]] = {}
        xforms: dict[str, tuple[float, ...]] = {}
        index = episode_seed(task, episode_id)
        for k, name in enumerate(names):
            (x, y), yaw = layouts[(index + k * 7) % len(layouts)]
            if jitter:
                x += float(rng.uniform(-jitter, jitter))
                y += float(rng.uniform(-jitter, jitter))
            spec = task.object(name)
            if keep_rest_orientation:
                # The declared rest orientation, yawed about world z: a
                # world-frame turn left-multiplies (xyzw convention).
                qx, qy, qz, qw = (float(v) for v in spec.quat)
                s, c = float(np.sin(yaw * 0.5)), float(np.cos(yaw * 0.5))
                xforms[name] = (
                    x,
                    y,
                    float(spec.pose[2]),
                    c * qx - s * qy,
                    c * qy + s * qx,
                    c * qz + s * qw,
                    c * qw - s * qz,
                )
            else:
                poses[name] = (x, y, float(spec.pose[2]), float(yaw))
        out: dict[str, Any] = {}
        if poses:
            out["object_poses"] = poses
        if xforms:
            out["object_xforms"] = xforms
        return out

    return sample


def move_near(*, radius: float = 0.12, source: str = "", target: str = ""):
    """A sampler placing one object a short distance from another.

    The layout family a "move A near B" task needs: the pair must start apart
    by more than the success threshold, or the episode is already solved.

    Args:
        radius: Distance from the target the source starts at [m].
        source: Object to move. Empty uses the task's first object.
        target: Object to place relative to. Empty uses the second.
    """

    def sample(task: TaskSpec, episode_id: int, rng: np.random.Generator) -> dict[str, Any]:
        movable = [o for o in task.objects if not o.static]
        src = task.object(source) if source else movable[0]
        tgt = task.object(target) if target else movable[1]
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        poses = {
            src.name: (
                float(tgt.pose[0] + radius * np.cos(angle)),
                float(tgt.pose[1] + radius * np.sin(angle)),
                float(src.pose[2]),
                float(rng.uniform(-np.pi, np.pi)),
            )
        }
        return {"object_poses": poses, "source": src.name, "target": tgt.name}

    return sample


def pick_instruction(*instructions: str):
    """A sampler cycling through phrasings of the same goal.

    Language variation with no physical variation, which is what a
    vision-language policy should be robust to and a scripted one should not
    notice at all. Cycled by :func:`episode_seed`, not the bare index, so a
    benchmark instance (episode 0 of a document at ``(:seed k)``) gets the
    phrasing its seed names.
    """

    def sample(task: TaskSpec, episode_id: int, rng: np.random.Generator) -> dict[str, Any]:
        del rng
        return {"instruction": instructions[episode_seed(task, episode_id) % len(instructions)]}

    return sample


def compose(*samplers):
    """Run several samplers in order, later overrides winning.

    Lets a task combine an independent layout sampler and an instruction
    sampler without either knowing about the other.
    """

    def sample(task: TaskSpec, episode_id: int, rng: np.random.Generator) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for sampler in samplers:
            out.update(sampler(task, episode_id, rng) or {})
        return out

    return sample
