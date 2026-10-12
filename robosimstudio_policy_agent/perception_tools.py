# SPDX-License-Identifier: Apache-2.0
"""Unprivileged ``sam3.*`` tools: an image and a phrase in, masks out.

The same argument :mod:`~robosimstudio_policy_agent.geometry_tools` makes, for
the bundle one layer up. The reference stack supplies ``sam3.*`` from
``open-robot-skills``, which is a separate distribution this package does not
depend on and cannot resolve; a perception body that calls into it fails at its
first node with ``Tool 'sam3.segment_text' not found`` on a checkout that
installed everything this project declares. What is reimplemented here is the
contract those bodies are written against -- ``image`` and ``query`` in, a
score-sorted ``masks``/``scores``/``boxes`` triple out -- and nothing else.

**Unprivileged in the structural sense.** There is no environment here, no bus
and no simulator: the tool sees an array of pixels and a phrase, which is what a
camera and a prompt give a real cell. It is registered beside ``geometry.*`` and
*before* the privileged branch for that reason -- a graph that grounds its
targets through here still runs under ``--no-privileged``, which is the whole
point of having a vision body at all.

**Lazily loaded, and it must stay that way.** Importing this module must not
import torch: it is imported by the connector on every graph run, and most
graphs never ground anything. The model is built on the first call and held
after, which costs one load per process rather than one per node.

**Weights.** ``facebook/sam3`` is a gated Hugging Face repository. The native
``sam3`` package wants a ``sam3.pt`` checkpoint; the *transformers* port of the
same weights is a ``model.safetensors`` under the same repo id, and that is what
this loads, because it is what a machine that has ever run SAM3 through
transformers already has cached. With the cache warm, ``HF_HUB_OFFLINE=1``
avoids the gate entirely; without it, ``HF_TOKEN`` is required and the error
below says so rather than surfacing a bare 401.
"""

from __future__ import annotations

import os
import threading
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SAM3_MODEL_ID",
    "SEGMENT",
    "PerceptionTools",
    "SegmentAsk",
    "reference_bundle_supplies_sam3",
    "register_perception_tools",
    "sam3_available",
    "segment_many",
]

#: The Hugging Face repo id the weights come from, image and video alike.
SAM3_MODEL_ID = os.environ.get("ROBOSIMSTUDIO_SAM3_MODEL_ID", "facebook/sam3")

#: Where the model runs. Overridable because a machine whose torch and driver
#: disagree about CUDA can still run this on the CPU, slowly, rather than not
#: at all -- and a perception body that runs slowly is diagnosable where one
#: that refuses to load is not.
SAM3_DEVICE = os.environ.get("ROBOSIMSTUDIO_SAM3_DEVICE", "cuda")

#: Score below which a detection is dropped before sorting. Zero by default:
#: thresholding is the *caller's* judgement -- a node that knows it is looking
#: for exactly three spools wants a different floor from one asking whether a
#: rod is present at all -- and a tool that pre-filters takes that judgement
#: away while looking like it did nothing.
SAM3_FLOOR = 0.0

_lock = threading.Lock()

#: The loaded pair, or ``(None, None)``. One container rather than two module
#: names so :func:`_load` rebinds a cell instead of declaring ``global`` twice:
#: the singleton is one fact, and a reader who finds half of it set has found a
#: bug rather than a state.
_loaded: list[Any] = [None, None]
_load_failure: list[Exception | None] = [None]


def sam3_available() -> bool:
    """Whether the stack this needs is importable, without importing torch."""
    from importlib.util import find_spec

    return find_spec("transformers") is not None and find_spec("torch") is not None


def reference_bundle_supplies_sam3(registries: Sequence[str | Path] | None = None) -> bool:
    """Whether a resolvable skill registry ships the real ``sam3`` bundle.

    Args:
        registries: The run's registry roots
            (:func:`~robosimstudio_policy_agent.skills.pinned_registries`), or ``None``
            to resolve gap's own set.

    The same race :func:`~robosimstudio_policy_agent.geometry_tools.reference_bundle_supplies_geometry`
    documents, and the same answer. ``register_callable`` refuses a duplicate
    name, this package registers before ``gap.execute`` discovers the skills
    bundle, and so the bundle loses: every graph run on a machine that HAS the
    reference stack would die at load with ``Tool name collision:
    'sam3.segment_text'``. That is the machine the perception graphs are meant
    to run on, and the collision would not be confined to the bodies that want
    the in-process copy -- ``tool_sort``, ``tool_hanging`` and ``rgbs`` ground
    their targets through the bundle's ``sam3`` and would go down with it.

    So the reference wins where it exists. It carries the rest of ``sam3.*``
    besides the one name reimplemented here -- ``segment_box`` among them, which
    the tool-hanging bodies call and this module does not have -- and it is the
    thing this module is a copy of.
    """
    from .geometry_tools import _registry_ships

    return _registry_ships("sam3", registries)


def _requested_backend() -> str:
    """``ROBOSIMSTUDIO_SEGMENTATION_BACKEND``, normalised; ``auto`` when unset."""
    return os.environ.get("ROBOSIMSTUDIO_SEGMENTATION_BACKEND", "auto").strip().lower()


SEGMENT = "perceive.sam3.segment_text"
"""The rendezvous group a text-prompted segmentation is asked for through.

A wave running one graph reaches its perception node together, so the worlds
asking are almost always *every* world -- and a model call is the most expensive
thing any of them does. Answered one at a time it is also the most serialising:
the barrier holds the fleet for the whole of it, once per world.

Batched it is one forward pass. The arithmetic per world is unchanged, which is
what a per-episode parity bar requires and what :func:`segment_many` is written
to guarantee: the single-image path *is* the batch path with one element.
"""


def register_perception_tools(registry, *, registries: Sequence[str | Path] | None = None, bus: Any = None) -> None:
    """Register the unprivileged perception tools on a GaP tool registry.

    A no-op where the reference bundle is resolvable -- see
    :func:`reference_bundle_supplies_sam3`.

    Registered otherwise whether or not the stack is importable. A body that
    calls a tool which is *present but cannot load its weights* gets an error
    naming the cause; one that calls a tool which was never registered gets
    "not found", which reads as a graph authoring mistake and sends the reader
    to the wrong file. The tool-hanging port lost an afternoon to exactly that
    distinction.
    """
    if reference_bundle_supplies_sam3(registries) and _requested_backend() != "palette":
        return
    # `ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette` is an explicit request for THIS module's
    # matcher, and it is honoured only by this module: the reference bundle is
    # real SAM3 and never reads the variable. So under `palette` the in-process
    # tool registers even where the reference is resolvable -- and it is safe
    # to, because the runtime keeps an in-process registration over a booted
    # bundle's copy (`tool_bundle_manager`: "in-process wins, and the bundle's
    # copy is simply not offered"; `tool_bundle_boot` does not even boot a
    # bundle whose tools are all provided in-process). The collision the
    # docstring above feared is from an older runtime.
    #
    # Measured before this: the cable delivery's `--segmentation palette` was
    # its whole evaluation condition (its in-tree registry shipped no tool
    # bundles, so this module always served), and the same runs on a checkout
    # with the sibling registry resolvable silently switched to real SAM3 --
    # USB-C Medium went 2/30 to 0/30 with every scene ending in
    # `no 'orange plug' in 'topdown'`, and nothing in the log said why.
    tools = PerceptionTools(bus=bus)
    for name, fn, summary in tools.entries():
        registry.register_callable(name, fn, summary=summary, tags=("perception",))


def _load() -> tuple[Any, Any]:
    """Build the SAM3 image model and processor once per process."""
    with _lock:
        if _load_failure[0] is not None:
            raise RuntimeError(str(_load_failure[0])) from _load_failure[0]
        if _loaded[0] is None:
            try:
                import torch
                import transformers
            except ImportError as exc:  # pragma: no cover -- depends on the extra
                raise RuntimeError(
                    "sam3.segment_text needs transformers and torch. They are not part of this "
                    "package's declared dependencies: install them, or run the oracle "
                    "perception body (--perception oracle)."
                ) from exc
            if not hasattr(transformers, "Sam3Model"):
                raise RuntimeError(
                    f"transformers {transformers.__version__} has no Sam3Model; SAM3 support "
                    "landed in 4.57. Upgrade transformers or run --perception oracle."
                )
            device = SAM3_DEVICE
            if device.startswith("cuda") and not torch.cuda.is_available():
                # Not an error. A torch built for one CUDA minor against a driver
                # from another reports exactly this, and the useful behaviour is
                # to run and say so -- the alternative is a perception body that
                # cannot be told apart from a missing bundle.
                device = "cpu"
            try:
                processor = transformers.Sam3Processor.from_pretrained(SAM3_MODEL_ID)
                model = transformers.Sam3Model.from_pretrained(
                    SAM3_MODEL_ID, dtype=torch.bfloat16 if device != "cpu" else torch.float32
                )
            except Exception as exc:
                error = RuntimeError(
                    f"could not load {SAM3_MODEL_ID!r}. It is a gated repository: accept its "
                    "licence and set HF_TOKEN, or run with the weights already cached and "
                    f"HF_HUB_OFFLINE=1. Underlying error: {exc}"
                )
                _load_failure[0] = error
                raise error from exc
            model = model.to(device).eval()
            model._rss_device = device
            _loaded[:] = (model, processor)
        return _loaded[0], _loaded[1]


def _palette_segment(arr: np.ndarray, query: str, max_results: int) -> dict[str, Any]:
    """Segment the suite's declared synthetic palette when SAM3 is unavailable.

    This consumes RGB only and has no simulator access. It is deliberately
    limited to colour-bearing prompts used by these scenes; arbitrary language
    still requires SAM3. Component scores favour thin, elongated achromatic
    regions for cable prompts so the white robot is not ranked as the cable.
    """
    import cv2

    q = str(query).lower()
    rgb = arr.astype(np.float32)
    hi = np.max(rgb, axis=2)
    lo = np.min(rgb, axis=2)
    chroma = (hi - lo) / np.maximum(hi, 1.0)
    # RTX's lab panel is intentionally soft and bright.  It preserves hue but
    # compresses channel ratios: the plug is typically RGB (235, 218, 174), so
    # the old ``r > 1.12*g`` test retained highlights and discarded the body.
    # HSV hue is the invariant the synthetic palette actually promises, across
    # both the tiled diagnostic renderer and the RTX observation renderer.
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hue, saturation, value = (hsv[..., i] for i in range(3))
    if "orange" in q:
        # The tiled routing fittings cluster at hue 9; a floor of 10
        # retains only sparse highlights and makes the terminal disappear.
        mask = (hue >= 8) & (hue <= 40) & (saturation >= 24) & (value > 48)
        thin = False
    elif "blue" in q:
        # The slightly blue-grey bench has HSV saturation 23 and is larger
        # than either receptacle. Accepting it makes the support surface win
        # the component ranking. Pale RTX blue still measures saturation 71.
        mask = (hue >= 85) & (hue <= 120) & (saturation >= 40) & (value > 45)
        thin = False
    elif "white" in q:
        achromatic = chroma < 0.16
        values = hi[achromatic]
        # Keep the grey bench out of the cable component.  The former 72-level
        # floor joined the white jacket to the entire achromatic background;
        # its skeleton then measured a 600 mm cable as 836--870 mm. 128 remains
        # below the lit synthetic jacket while cleanly separating the neutral
        # work surface in both tiled and RTX-derived observations.
        floor = max(128.0, float(np.percentile(values, 68.0))) if values.size else 128.0
        mask = achromatic & (hi >= floor)
        thin = "cable" in q or "rod" in q or "wire" in q
    else:
        raise RuntimeError(
            f"SAM3 weights are unavailable and the RGB palette fallback does not support {query!r}; "
            "use a colour-bearing query or provide HF_TOKEN."
        )

    binary = mask.astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    found: list[tuple[float, int, np.ndarray, list[float]]] = []
    for label in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label])
        if area < 24:
            continue
        component = (labels == label).astype(np.uint8)
        if thin:
            points = np.column_stack(np.nonzero(component)).astype(np.float64)
            covariance = np.cov(points, rowvar=False) if len(points) > 2 else np.eye(2)
            eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1.0e-9)
            elongation = float(1.0 - np.sqrt(eigenvalues[0] / eigenvalues[-1]))
            score = 0.70 + 0.29 * elongation
        else:
            score = 0.90 + 0.09 * min(1.0, area / 1000.0)
        found.append((score, area, component, [float(x), float(y), float(x + width), float(y + height)]))
    found.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if max_results > 0:
        found = found[:max_results]
    return {
        "masks": [item[2] for item in found],
        "scores": [float(item[0]) for item in found],
        "boxes": [item[3] for item in found],
        "backend": "palette",
    }


class PerceptionTools:
    """The SAM3 implementations. Stateless; the model is a module singleton.

    Args:
        bus: The world's bus, when there is one offering :data:`SEGMENT`. With
            it a call joins the wave's group and is answered in one forward pass
            with every other world that is looking; without it -- a single-world
            run, or a wave whose driver registered no group -- the same call runs
            on its own, through the same code.
    """

    def __init__(self, bus: Any = None) -> None:
        self.bus = bus

    def entries(self) -> list[tuple[str, Any, str]]:
        return [
            (
                "sam3.segment_text",
                self.segment_text,
                "Text-prompted instance segmentation: one mask per detected instance, score-sorted descending.",
            ),
        ]

    def segment_text(
        self,
        image: Any = None,
        query: str = "",
        max_results: int = 0,
        threshold: float = SAM3_FLOOR,
        mask_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Segment every instance matching ``query``.

        Args:
            image: RGB uint8 ``[H, W, 3]``.
            query: What to find, as a noun phrase.
            max_results: Cap on returned detections, best first. ``<= 0`` returns
                everything, which is what a caller counting instances wants.
            threshold: Detection score floor.
            mask_threshold: Where a soft mask becomes a boolean one.

        Returns ``{"masks": [...], "scores": [...], "boxes": [...]}`` with masks
        as ``uint8`` arrays at the image's own resolution -- the shape the
        ``open-robot-skills`` bundle returns, so a body written against either
        runs on both.
        On a wave this joins :data:`SEGMENT` and is answered with every other
        world that is looking, in one forward pass. That is a change of *when*
        and not of *what*: the answer comes from :func:`segment_many`, which is
        also what a single call runs.
        """
        ask = SegmentAsk(
            image=np.ascontiguousarray(np.asarray(image, dtype=np.uint8)),
            query=str(query),
            max_results=int(max_results),
            threshold=float(threshold),
            mask_threshold=float(mask_threshold),
        )
        ask.validate()
        # The backend switch comes before any of that, because `palette` is a
        # request for THIS module's matcher and there is nothing to batch: it is
        # a pure function of one image. Measured on the cable delivery, whose
        # `--segmentation palette` was its whole evaluation condition -- the same
        # runs on a checkout where the sibling registry resolves silently
        # switched to real SAM3, and USB-C Medium went 2/30 to 0/30 with every
        # scene ending in `no 'orange plug' in 'topdown'`.
        backend = _requested_backend()
        if backend not in ("auto", "sam3", "palette"):
            raise ValueError("ROBOSIMSTUDIO_SEGMENTATION_BACKEND must be auto, sam3, or palette")
        if backend == "palette":
            return _palette_segment(ask.image, ask.query, ask.max_results)
        try:
            _load()
        except RuntimeError as exc:
            if backend == "sam3":
                raise
            warnings.warn(
                f"{exc} Falling back to RGB palette segmentation for the synthetic cable tasks.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _palette_segment(ask.image, ask.query, ask.max_results)
        batchable = getattr(self.bus, "batchable", None)
        if callable(batchable) and batchable(SEGMENT):
            return self.bus.batched(SEGMENT, ask)
        return segment_many([ask])[0]


@dataclass
class SegmentAsk:
    """One world's segmentation request, as the group carries it.

    A record rather than a tuple because the *thresholds* decide what can share a
    call: the post-processing takes one score floor for a whole batch, so asks
    that disagree about it are batched through the model together and separated
    afterwards. Every shipped wave runs one graph and therefore one threshold,
    so that grouping is almost always the identity -- it exists so that a graph
    which varies its floor per world is answered correctly rather than quietly
    scored against somebody else's.
    """

    image: np.ndarray
    query: str
    max_results: int = 0
    threshold: float = SAM3_FLOOR
    mask_threshold: float = 0.5

    def validate(self) -> None:
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"sam3.segment_text expects RGB uint8 [H, W, 3], got {self.image.shape}")
        if not self.query.strip():
            raise ValueError("sam3.segment_text needs a non-empty query")


def segment_many(asks: Sequence[SegmentAsk]) -> list[dict[str, Any]]:
    """Answer a whole group of segmentations, in the order it was asked.

    One forward pass per distinct ``(threshold, mask_threshold)`` -- which for a
    wave running one graph is one pass for the fleet. The model is prompted with
    a list of images and a list of phrases, which is the port's own batched form,
    so a world's answer is the same tensor it would have got alone.

    **Written as the one implementation, and the single-image call goes through
    it.** A batched path beside a serial one is two pieces of arithmetic that
    have to be kept equal by hand; this way a wave and a single run cannot
    disagree, because there is nothing for them to disagree about.
    """
    import torch
    from PIL import Image

    items = list(asks)
    for ask in items:
        ask.validate()
    model, processor = _load()
    device = getattr(model, "_rss_device", "cpu")

    answers: list[dict[str, Any]] = [{} for _ in items]
    groups: dict[tuple[float, float], list[int]] = {}
    for index, ask in enumerate(items):
        groups.setdefault((ask.threshold, ask.mask_threshold), []).append(index)

    for (threshold, mask_threshold), indices in groups.items():
        images = [Image.fromarray(items[i].image) for i in indices]
        texts = [items[i].query for i in indices]
        inputs = processor(images=images, text=texts, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=float(threshold),
            mask_threshold=float(mask_threshold),
            target_sizes=[(items[i].image.shape[0], items[i].image.shape[1]) for i in indices],
        )
        for slot, index in enumerate(indices):
            answers[index] = _detections(results[slot], items[index], mask_threshold)
    return answers


def _detections(result: Any, ask: SegmentAsk, mask_threshold: float) -> dict[str, Any]:
    """One image's detections, score-sorted and capped."""
    scores = result.get("scores")
    if scores is None or len(scores) == 0:
        return {"masks": [], "scores": [], "boxes": [], "backend": "sam3"}
    scores_np = scores.float().cpu().numpy()
    order = np.argsort(-scores_np)
    if ask.max_results > 0:
        order = order[: ask.max_results]
    masks, boxes = result.get("masks"), result.get("boxes")
    return {
        "backend": "sam3",
        "masks": [(masks[i].cpu().numpy() > float(mask_threshold)).astype(np.uint8) for i in order],
        "scores": [float(scores_np[i]) for i in order],
        "boxes": [[float(v) for v in boxes[i].float().cpu().numpy()] for i in order] if boxes is not None else [],
    }
