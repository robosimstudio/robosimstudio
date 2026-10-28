# SPDX-License-Identifier: Apache-2.0
"""The constructor for a jointed object: a free function that returns a spec.

The articulated sibling of :func:`robosimstudio.domains.rigid.make.body`.
"""

from __future__ import annotations

from robosimstudio.specs import ArticulatedSpec, AssetSpec, JointSpec, Vec3

__all__ = ["articulated"]

_ORIGIN: Vec3 = (0.0, 0.0, 0.0)


def articulated(
    *,
    name: str,
    asset: str | AssetSpec,
    at: Vec3 = _ORIGIN,
    fixed_base: bool = True,
    joints: dict[str, float] | tuple[JointSpec, ...] = (),
    **kwargs,
) -> ArticulatedSpec:
    """A jointed object from a USD stage -- a cabinet, a door, a box with a lid.

    ``at`` places the asset's origin in the work-surface frame, as for
    :func:`~robosimstudio.domains.rigid.make.body`; a SimReady asset's origin is
    its base, so ``at=(x, y, 0)`` stands it on the bench. ``joints`` starts named
    joints somewhere other than where the asset has them -- ``{"drawer_01": 0.15}``
    opens a drawer 150 mm -- or is a tuple of
    :class:`~robosimstudio.specs.JointSpec` for drive overrides too. Extra keyword
    arguments go to :class:`~robosimstudio.specs.ArticulatedSpec`.
    """
    spec = asset if isinstance(asset, AssetSpec) else AssetSpec(path=str(asset))
    if isinstance(joints, dict):
        joints = tuple(JointSpec(name=joint, q=float(q)) for joint, q in joints.items())
    return ArticulatedSpec(name=name, asset=spec, pose=at, fixed_base=fixed_base, joints=tuple(joints), **kwargs)
