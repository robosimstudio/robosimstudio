# SPDX-License-Identifier: Apache-2.0
"""Raw point and span access over a recorded episode.

This module intentionally contains no scene interpretation.  A point is a set
of simulator arrays at one recorded step; a span is two point references plus
a lazy slice of the same tape.  Names enter only through a static index
manifest written by :class:`~robosimstudio.record.recorder.EpisodeRecorder`.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CAPTURE_SCHEMA",
    "EpisodeCapture",
    "RawPoint",
    "RawSpan",
    "ToolSpanRecorder",
    "capture_manifest",
]

CAPTURE_SCHEMA = "robosimstudio.capture.v1"
"""The manifest's wire format, written into every tape and never read back here.

Readers outside this repo parse it, so the *keys* are the contract and an
attribute rename on this side is not a schema change -- ``cloths`` still names
what ``handles.deformables`` now holds. Bump the version when a key changes
meaning or leaves; adding one is backward compatible and does not.

Poses are ``[x, y, z, qx, qy, qz, qw]`` in world metres, hence the ``_xyzw``
suffix on ``xform_xyzw`` and ``ee_offset_rot_xyzw``; ``bbox`` is full extents in
metres, not half-extents.
"""


def _ints(values: Any) -> list[int]:
    return [int(v) for v in values]


def _floats(values: Any) -> list[float]:
    return [float(v) for v in values]


def _asset_types(env: Any) -> dict[str, str]:
    doc = getattr(env, "doc", None)
    return {
        str(getattr(item, "name", "")): str(getattr(item, "type", ""))
        for item in getattr(doc, "objects", ()) or ()
        if getattr(item, "name", "")
    }


def _chain_radius(env: Any, shapes: Any) -> float | None:
    """The segment radius of a body chain, or ``None`` when it has no round section.

    A reader deciding whether a jaw closed *on* a chain needs the section it
    closes on, and a chain has no ``bbox`` to say it. Read off the first
    segment's own collision geometry -- a capsule, cylinder or sphere carries
    its radius in ``shape_scale[0]`` -- rather than declared, so it cannot
    drift from the chain the solver actually steps.
    """
    shapes = list(shapes)
    model = getattr(env, "model", None)
    if not shapes or model is None:
        return None
    try:
        import newton

        round_types = {int(newton.GeoType.CAPSULE), int(newton.GeoType.CYLINDER), int(newton.GeoType.SPHERE)}
        shape = int(shapes[0])
        if int(model.shape_type.numpy()[shape]) not in round_types:
            return None
        return float(model.shape_scale.numpy()[shape][0])
    except Exception:
        return None


def _features(asset_type: str) -> dict[str, Any]:
    """Declared feature payloads for one asset type, JSON-shaped; {} unknown."""
    if not asset_type:
        return {}
    try:
        from robosimstudio.assetlib.registry import ASSETS

        record = ASSETS.get(asset_type)
    except Exception:
        return {}
    out: dict[str, Any] = {}
    for feature in getattr(record, "features", ()) or ():
        name = getattr(feature, "name", "")
        payload = getattr(feature, "payload", None)
        if name and isinstance(payload, dict):
            out[name] = {
                "kind": str(getattr(feature, "kind", "")),
                **payload,
            }
    return out


def capture_manifest(env: Any, episode: Any = None) -> dict[str, Any]:
    """Serialize stable simulator indices and geometry, without deriving facts.

    Args:
        episode: Whose episode the ``task`` block names. Defaults to the env's
            own, which is the only one a single-world env has. **A wave must
            pass one per world**: the indices and geometry below are the
            template's and are shared, but the episode id, seed and sampled
            overrides are not -- and a manifest built once for a wave labels
            every world's tape with world zero's episode, which is a tape that
            lies about which episode it is evidence of.
    """
    handles = env.handles
    types = _asset_types(env)
    entities: dict[str, Any] = {}
    for group, values in (("object", handles.objects), ("static", handles.statics)):
        for name, handle in sorted(values.items()):
            entities[name] = {
                "kind": group,
                # ``.category``, not ``.kind``: EntityHandle renamed the field
                # and this reader never followed, so every tape written since
                # carried an empty category. It is read straight off the
                # dataclass now, and every declared field below with it -- a
                # ``getattr`` with a default cannot raise, so the next rename
                # would go silent again exactly the same way.
                "category": str(handle.category),
                "asset_type": types.get(name, ""),
                "body": int(handle.body),
                # Every body a chain owns, in order -- a rope's segments. Empty
                # for the ordinary one-body entity, whose ``body`` is the whole
                # story. Without it a reader has ``body`` alone, which for a
                # chain is the first segment -- the one welded in the anchor --
                # so every pose, carry and containment it derived described a
                # capsule that never moves.
                "bodies": _ints(handle.bodies),
                "segment_radius": _chain_radius(env, handle.shapes) if len(handle.bodies) > 1 else None,
                "coord_start": int(handle.coord_start),
                "shapes": _ints(handle.shapes),
                "collision_shapes": _ints(handle.collision_shapes),
                "visual_shapes": _ints(handle.visual_shapes),
                "static": bool(handle.static),
                "xform_xyzw": _floats(handle.xform),
                "bbox": _floats(handle.bbox) if handle.bbox is not None else None,
                # Declared subpart geometry, local frame — the ring's centre,
                # a bore's axis. Static per asset and tiny, but it is the
                # difference between feedback that says "0.08 m from the hook"
                # and feedback that can say "the ring ended 184 mm BELOW the
                # shank": without the feature, every consumer reasons about
                # centroids.
                "features": _features(types.get(name, "")),
            }

    arms: dict[str, Any] = {}
    ordered = [n for n in (handles.work_arm, handles.support_arm) if n in handles.arms]
    ordered += [n for n in handles.arms if n not in ordered]
    for arm_id, name in enumerate(ordered):
        arm = handles.arms[name]
        arms[name] = {
            "arm_id": arm_id,
            "role": "work" if name == handles.work_arm else "support" if name == handles.support_arm else "",
            "arm_coords": _ints(arm.arm_coords),
            "arm_dofs": _ints(arm.arm_dofs),
            "gripper_coords": _ints(arm.gripper_coords),
            "gripper_dofs": _ints(arm.gripper_dofs),
            "ee_body": int(arm.ee_body),
            "base_body": int(arm.base_body),
            "ee_offset": _floats(arm.ee_offset),
            "ee_offset_rot_xyzw": _floats(arm.ee_offset_rot),
            "approach_axis": _floats(arm.approach_axis),
            "close_axis": _floats(arm.close_axis),
            "gripper_span": float(arm.gripper_span),
            "gripper_open": float(arm.gripper_open),
            "gripper_open_q": _floats(arm.gripper_open_q),
            "gripper_closed_q": _floats(arm.gripper_closed_q),
            "open_axis_signs": _floats(arm.open_axis_signs),
            "finger_groups": [
                [{"body": int(body), "shapes": _ints(shapes)} for body, shapes in side] for side in arm.finger_groups
            ],
            # Flat per side, and NOT redundant with ``finger_groups`` for the
            # reader that wants it: attributing a contact to a finger SIDE is a
            # set-membership test on shape indices, and the grouped form makes
            # every consumer rebuild this to do it. Downstream tape readers use
            # exactly this key to decide which side of which hand a contact
            # belongs to -- which is how a grasp, a hold, a carry and a release
            # are derived from the tape. A reader that defaults it with
            # ``.get(..., ())`` does not raise if it is dropped: it silently
            # reports that nothing is held.
            "finger_shapes": [[int(shape) for _body, shapes in side for shape in shapes] for side in arm.finger_groups],
            "shapes": _ints(arm.shapes),
        }

    # ``handles.deformables`` is where cloth, centerlines and soft bodies live
    # now -- ``SceneHandles.cloths``/``ClothHandle`` were widened to cover every
    # substrate with particles instead of one. The manifest's own key stays
    # ``cloths``: it is the wire format of ``robosimstudio.capture.v1``, which
    # readers outside this repo parse, and an attribute rename is not a schema
    # change. Reading the retired attribute raised ``AttributeError`` from
    # inside the recorder's ``begin``, which disables recording rather than
    # failing the run -- so every ``--capture-state`` episode wrote no tape at
    # all and said so only in one line of the runner's log.
    cloths = {
        name: {
            "particles": _ints(handle.particles),
            "sets": {key: _ints(indices) for key, indices in sorted(handle.sets.items())},
            "triangles": np.asarray(handle.triangles, dtype=np.int32).tolist(),
        }
        for name, handle in sorted(handles.deformables.items())
    }
    regions = {
        name: {
            "frame": str(handle.frame),
            "frame_body": int(handle.frame_body),
            "origin": _floats(handle.origin),
            "xy_ranges": _floats(handle.xy_ranges),
            "z_range": _floats(handle.z_range),
        }
        for name, handle in sorted(handles.regions.items())
    }
    episode = episode if episode is not None else getattr(env, "episode", None)
    # Off the task when there is one, off the request when there is not: the
    # recorder accepts a bare ``Sim``, which has no task (``EpisodeRecorder``'s
    # results row reads provenance the same way).
    task = getattr(env, "task", None)
    source = task if task is not None else getattr(env, "request", None)
    embodiment = getattr(source, "embodiment", "")
    physics = getattr(source, "physics", None)
    return {
        "schema": CAPTURE_SCHEMA,
        "task": {
            "name": getattr(source, "full_name", "") if task is not None else "",
            "scene": getattr(source, "scene", "") or getattr(source, "cell", ""),
            "embodiment": embodiment if isinstance(embodiment, str) else str(getattr(embodiment, "name", embodiment)),
            "gripper": getattr(source, "gripper", None),
            "instruction": str(getattr(episode, "instruction", getattr(source, "instruction", ""))),
            "control_hz": float(physics.control_hz) if physics is not None else None,
            "episode_id": int(getattr(episode, "episode_id", -1)),
            "seed": int(getattr(episode, "seed", -1)),
            "sampled": _json_safe(dict(getattr(episode, "overrides", ()) or ())),
        },
        "surface_z": float(handles.surface_z),
        "entities": entities,
        "arms": arms,
        "cloths": cloths,
        "regions": regions,
        "robot_shapes": _ints(handles.robot_shapes),
        "segmentation_names": {str(k): v for k, v in handles.segmentation_names().items()},
    }


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return "<nested>"
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return value.tolist()
        return {"array_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _json_safe(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            return {"sequence_length": len(value)}
        return [_json_safe(v, depth=depth + 1) for v in value]
    return str(value)


class ToolSpanRecorder:
    """Thread-safe lifecycle markers for state-mutating graph tool calls."""

    def __init__(self, step_fn) -> None:
        self._step_fn = step_fn
        self._lock = threading.Lock()
        self._next = 0
        self._episode_id = -1
        self._rows: list[dict[str, Any]] = []
        self._active: dict[int, dict[str, Any]] = {}

    def reset(self, episode_id: int) -> None:
        with self._lock:
            self._next = 0
            self._episode_id = int(episode_id)
            self._rows = []
            self._active = {}

    def _step(self) -> int:
        try:
            return int(self._step_fn())
        except Exception:
            return 0

    def begin(self, tool: str, tags: tuple[str, ...], ctx: Any, request: dict[str, Any]) -> dict[str, Any] | None:
        if "sim_step" not in tags:
            return None
        start_step = self._step()
        with self._lock:
            span_id = self._next
            self._next += 1
            token = {
                "schema": CAPTURE_SCHEMA,
                "span_id": span_id,
                "episode_id": self._episode_id,
                "kind": "tool",
                "tool": str(tool),
                "node": str(getattr(ctx, "_node_id", "") or ""),
                "arm_id": int(request.get("arm_id", 0) or 0),
                "start_step": start_step,
                "started_at": time.time(),
                "request": _json_safe(request),
            }
            self._active[span_id] = token
            return token

    def end(self, token: dict[str, Any] | None, *, result: Any = None, error: BaseException | None = None) -> None:
        if token is None:
            return
        span_id = int(token["span_id"])
        row = {
            **token,
            "end_step": self._step(),
            "finished_at": time.time(),
            "status": "error" if error is not None else "ok",
            "result": _json_safe(result),
            "error": f"{type(error).__name__}: {error}" if error is not None else None,
        }
        with self._lock:
            # A reset can retire an old episode while its worker unwinds. Do
            # not append that late completion to the next episode's marker set.
            if self._active.pop(span_id, None) is None:
                return
            self._rows.append(row)

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            end_step = self._step()
            finished_at = time.time()
            rows = list(self._rows)
            rows.extend(
                {
                    **token,
                    "end_step": end_step,
                    "finished_at": finished_at,
                    "status": "interrupted",
                    "result": None,
                    "error": "episode ended before the tool returned",
                }
                for token in self._active.values()
            )
            return [dict(row) for row in sorted(rows, key=lambda r: int(r["span_id"]))]


@dataclass(frozen=True)
class RawPoint:
    """One recorded instant; array values are views into the episode tape."""

    index: int
    step: int
    sim_time: float
    wall_time: float
    arrays: dict[str, np.ndarray]
    manifest: dict[str, Any]
    contacts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RawSpan:
    """A tool interval over one canonical tape."""

    source: EpisodeCapture
    metadata: dict[str, Any]

    @property
    def before(self) -> RawPoint:
        return self.source.point(step=int(self.metadata["start_step"]))

    @property
    def after(self) -> RawPoint:
        return self.source.point(step=int(self.metadata["end_step"]))

    def points(self) -> Iterator[RawPoint]:
        start, end = int(self.metadata["start_step"]), int(self.metadata["end_step"])
        yield from self.source.points(start_step=start, end_step=end)

    @property
    def id(self) -> int:
        return int(self.metadata["span_id"])


class EpisodeCapture:
    """Read point and span views from ``capture.npz`` and marker files."""

    def __init__(self, episode_dir: str | Path) -> None:
        self.path = Path(episode_dir)
        self.manifest = json.loads((self.path / "capture_manifest.json").read_text())
        with np.load(self.path / "capture.npz", allow_pickle=False) as data:
            self.arrays = {key: data[key] for key in data.files}
        self._step_to_index = {int(step): i for i, step in enumerate(self.arrays["step"])}
        self.spans = self._jsonl(self.path / "spans.jsonl")
        self.events = self._jsonl(self.path / "events.jsonl")
        contacts = self._jsonl(self.path / "contacts.jsonl")
        self._contacts = {int(row["step"]): list(row.get("pairs") or []) for row in contacts}

    @staticmethod
    def _jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def point(self, *, step: int | None = None, index: int | None = None, event: str | None = None) -> RawPoint:
        if event is not None:
            matches = [row for row in self.events if row.get("event") == event]
            if not matches:
                raise KeyError(f"no capture event {event!r}")
            step = int(matches[-1]["step"])
        if index is None:
            if step is None:
                raise ValueError("one of step, index or event is required")
            try:
                index = self._step_to_index[int(step)]
            except KeyError as exc:
                raise KeyError(f"step {step} was not recorded; have {sorted(self._step_to_index)}") from exc
        index = int(index)
        values = {
            key: value[index]
            for key, value in self.arrays.items()
            if key not in {"step", "sim_time", "wall_time"} and len(value) == len(self.arrays["step"])
        }
        actual_step = int(self.arrays["step"][index])
        return RawPoint(
            index=index,
            step=actual_step,
            sim_time=float(self.arrays["sim_time"][index]),
            wall_time=float(self.arrays["wall_time"][index]),
            arrays=values,
            manifest=self.manifest,
            contacts=self._contacts.get(actual_step, []),
        )

    def points(self, *, start_step: int = 0, end_step: int | None = None) -> Iterator[RawPoint]:
        for index, step in enumerate(self.arrays["step"]):
            value = int(step)
            if value < start_step or (end_step is not None and value > end_step):
                continue
            yield self.point(index=index)

    def span(self, span_id: int) -> RawSpan:
        for row in self.spans:
            if int(row.get("span_id", -1)) == int(span_id):
                return RawSpan(self, row)
        raise KeyError(f"no span {span_id}; have {[r.get('span_id') for r in self.spans]}")

    def iter_spans(self) -> Iterator[RawSpan]:
        for row in self.spans:
            yield RawSpan(self, row)
