# SPDX-License-Identifier: Apache-2.0
"""Re-render a recorded episode into any backend.

The economics this exists for: collecting an episode with a path tracer in the
loop costs orders of magnitude more than collecting it with the tiled ray
tracer, and most of that cost buys nothing during collection -- the policy is
not looking at those pixels, the evaluator is not scoring them. So collect
cheap, then re-render.

What makes that sound rather than a re-simulation is that nothing is
re-simulated. :class:`OfflineRenderer` rebuilds the deterministic scene, then
*replays* the recorded ``body_q`` and ``particle_q`` straight into a
:class:`newton.State` and renders it. No solver runs, so there is no question of
whether the second pass diverged from the first: it is the same trajectory, by
construction, photographed differently.

Only the state channels a run recorded can be replayed. An episode that stored
subsampled particles can re-render its rigid bodies faithfully and its cloth
only as far as the subsample goes, so :meth:`OfflineRenderer.replay` says which
it got rather than quietly interpolating.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

import newton
import numpy as np

from .base import FrameSet, RendererBase

__all__ = ["REPLAY_CHANNELS", "OfflineRenderer", "StateReplay", "resolve_live_index"]

REPLAY_CHANNELS: tuple[str, ...] = ("body_q", "body_qd", "particle_q", "particle_qd", "joint_q", "shape_color_live")
"""The channels :meth:`StateReplay.write` honours, and nothing else.

``shape_color_live`` is the one that is not a :class:`newton.State` field: the
colours of the shapes a scene recolours while it runs, written back into
``model.shape_color`` so a backend that reads colour off the model shows the
mark fading rather than frozen at the frame the stage was built. It needs the
recording's ``shape_color_live_index`` anchor to say which rows those are.

Named rather than left implicit because it is what a reader has to select *on*
before slicing. A recording that stores an episode's constants beside its
per-step state -- bond indices, bond anchors, a particle subset -- otherwise
hands :func:`steps_from_arrays` a length belonging to none of its steps, and
that function truncates to the shortest array it is given.
"""


class StateReplay:
    """Writes recorded arrays into a state, one step at a time.

    Args:
        model: The model the recording was made against. It must be the *same*
            scene -- same body and particle counts -- which is what the episode
            metadata's model key is for.
        strict: Raise when a recorded array does not match the model. Off,
            a mismatched particle array is skipped and reported through
            :attr:`skipped`.
    """

    def __init__(self, model: newton.Model, *, strict: bool = True, live_index=None) -> None:
        self.model = model
        self.state = model.state()
        self.strict = bool(strict)
        self.skipped: set[str] = set()
        """Which channels were dropped for not matching the model."""
        self.live_index: np.ndarray | None = None if live_index is None else np.asarray(live_index, dtype=np.int32)
        """Which shapes a recorded ``shape_color_live`` row addresses, from the
        recording's own anchor. Set per episode; ``None`` drops the channel."""

    def write(self, step: Mapping[str, np.ndarray]) -> newton.State:
        """Load one recorded step into the state and return it.

        Args:
            step: Any of ``body_q``, ``body_qd``, ``particle_q``,
                ``particle_qd``, ``joint_q``. Missing keys keep their previous
                values, so a recording that only stored ``body_q`` replays a
                static cloth rather than failing.
        """
        for key, expected in (
            ("body_q", self.model.body_count),
            ("body_qd", self.model.body_count),
            ("particle_q", self.model.particle_count),
            ("particle_qd", self.model.particle_count),
        ):
            value = step.get(key)
            if value is None:
                continue
            arr = np.asarray(value)
            if len(arr) != expected:
                message = f"recorded {key} has {len(arr)} rows, model has {expected}"
                if self.strict:
                    raise ValueError(message)
                self.skipped.add(key)
                continue
            getattr(self.state, key).assign(arr)

        live = step.get("shape_color_live")
        if live is not None:
            self._write_live_colors(np.asarray(live, dtype=np.float32))

        joint_q = step.get("joint_q")
        if joint_q is not None and len(joint_q) == self.model.joint_coord_count:
            self.state.joint_q.assign(np.asarray(joint_q, dtype=np.float32))
            if step.get("body_q") is None:
                # No recorded poses, so derive them -- a recording that stored
                # only joint angles still re-renders correctly.
                newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        return self.state

    def _write_live_colors(self, rows: np.ndarray) -> None:
        colors = getattr(self.model, "shape_color", None)
        if self.live_index is None or colors is None or not hasattr(colors, "numpy"):
            self.skipped.add("shape_color_live")
            return
        if len(rows) != len(self.live_index):
            message = f"recorded shape_color_live has {len(rows)} rows, the anchor names {len(self.live_index)} shapes"
            if self.strict:
                raise ValueError(message)
            self.skipped.add("shape_color_live")
            return
        table = np.asarray(colors.numpy(), dtype=np.float32)
        table[self.live_index, :3] = rows[:, :3]
        colors.assign(table)


def resolve_live_index(model: newton.Model, privileged: Mapping[str, np.ndarray], fallback=None) -> np.ndarray | None:
    """Which rows of the rebuilt model a recording's ``shape_color_live`` addresses.

    Shape indices are not stable across builds of one task -- a convex
    decomposition upstream of the decals may cut a different number of hulls
    -- so the recording's ``shape_color_live_label`` anchor is matched against
    the rebuilt model's labels first. A recording that carries only the index
    anchor is mapped *positionally* onto ``fallback`` (the rebuilt scene's own
    live shapes) when the two agree in length: the decal fields add their cells
    in a fixed order, so the k-th live shape then is the k-th live shape now.
    Anything else returns ``None`` and the channel is skipped.
    """
    labels = privileged.get("shape_color_live_label")
    have = getattr(model, "shape_label", None)
    if labels is not None and have is not None:
        lookup = {str(label): i for i, label in enumerate(have)}
        found = [lookup.get(str(label), -1) for label in labels]
        if all(i >= 0 for i in found):
            return np.asarray(found, dtype=np.int32)
    index = privileged.get("shape_color_live_index")
    if index is None:
        return None
    index = np.asarray(index, dtype=np.int32)
    if fallback is not None and len(fallback) == len(index):
        return np.asarray(fallback, dtype=np.int32)
    if int(model.shape_count) > int(index.max()):
        return index
    return None


class OfflineRenderer:
    """A renderer driven by recorded state rather than by a simulation.

    Args:
        model: The rebuilt scene's model.
        renderer: Any :class:`~robosimstudio.core.render.base.RendererBase`.
            RTX is the usual choice here, which is the whole point.
        strict: Forwarded to :class:`StateReplay`.
    """

    def __init__(self, model: newton.Model, renderer: RendererBase, *, strict: bool = True, live_index=None) -> None:
        self.model = model
        self.renderer = renderer
        self.replay_state = StateReplay(model, strict=strict, live_index=live_index)

    def replay(self, steps: Iterable[Mapping[str, np.ndarray]]) -> Iterator[FrameSet]:
        """Render every recorded step in order.

        Yields rather than accumulating, so a long episode is bounded by one
        frame set rather than by the whole video.
        """
        for step in steps:
            yield self.renderer.render(self.replay_state.write(step))

    def close(self) -> None:
        self.renderer.close()


def steps_from_arrays(arrays: Mapping[str, np.ndarray]) -> Iterator[dict[str, np.ndarray]]:
    """Slice a recorded episode's stacked arrays into per-step dicts.

    An episode store holds ``body_q`` as ``[steps, bodies, 7]``; a replay wants
    ``[bodies, 7]`` at a time. Arrays of differing length are truncated to the
    shortest, which is what a run interrupted mid-write leaves behind.
    """
    keys = [k for k, v in arrays.items() if isinstance(v, np.ndarray) and v.ndim >= 1]
    if not keys:
        return
    n = min(len(arrays[k]) for k in keys)
    for i in range(n):
        yield {k: arrays[k][i] for k in keys}
