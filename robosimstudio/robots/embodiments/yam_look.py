# SPDX-License-Identifier: Apache-2.0
"""The real YAM's two-tone shell: black arm, white upper-arm and forearm tubes.

Ported from an earlier YAM simulation, whose numbers came out of a colour
derivation against photographs of the real arm -- and the derivation is the
reason this is a module rather than a colour constant.

The i2rt MJCF ships rainbow per-link ``rgba`` (purple base, tan link1, blue
link2, olive link3, pink link5, green gripper) left over from the CAD export.
Those are placeholder part colours, not the robot, which is why
:class:`~robosimstudio.specs.EmbodimentSpec` states one flat colour over the
lot -- :data:`YAM_BLACK`, the near-black the real arm's anodised parts read as
(it was the palette's ``ROBOT_DARK`` mid-grey, which a path tracer draws as
silver). Flat black is right for every link but three: on the real arm the
base and the upper-arm and forearm **tube shells** are white, and an arm
without them does not read as a YAM.

The tube cannot simply be a differently coloured geom, because i2rt welds one
mesh per link where the reference bimanual description
(``robots_realtime/.../yam_bimanual_scene.xml``, materials ``black`` at 0.30
and ``white`` at 0.90) has a per-part mesh set. So the tube is selected
*geometrically* inside the link mesh, in the mesh's own STL coordinates: a band
along the link's long z axis plus a radius about the tube axis, so the
forearm's side brackets stay black. Those five numbers per link were derived by
registering the reference part meshes onto the i2rt link meshes and labelling
every i2rt vertex by the reference part nearest it; the rule reproduces that
per-part labelling to within 0.3% (link2) and 1.0% (link3) of vertices.

Nothing here touches physics. The split adds two massless, non-colliding
visual shapes and hides the original's *rendering* only -- the original keeps
its collision geometry and its inertia.
"""

from __future__ import annotations

import newton
import numpy as np

from robosimstudio.core.render.finish import GLOSS, SATIN, register_color_finish

__all__ = ["YAM_BLACK", "YAM_WHITE", "YAM_WHITE_LINKS", "YAM_WHITE_TUBES", "split_yam_white_tubes"]

YAM_WHITE: tuple[float, float, float] = (0.90, 0.90, 0.90)
"""Material ``white`` in the reference scene: the two tube shells, and the base."""

YAM_BLACK: tuple[float, float, float] = (0.040, 0.040, 0.043)
"""Everything else on the arm: anodised joints, the forearm, the printed fingers.

The reference clips read the arm's dark parts at 40-65 of 255 in sRGB, which
is an albedo near 0.04 -- black, not the ``ROBOT_DARK`` mid-grey (0.30) the
palette gives a converter-authored arm so that it is not crushed under the
tiled tracer. Under that tracer this renders around pixel 45, which is what a
black arm on a lit bench looks like, and under the path tracer it is the
black the clips show."""

register_color_finish(YAM_WHITE, GLOSS)
register_color_finish(YAM_BLACK, SATIN)

YAM_WHITE_LINKS: tuple[str, ...] = ("base_geom_0",)
"""Geom-name suffixes painted white whole. The base is a white shell in every
frame of the reference clips, and it is one part, so it needs no split."""

YAM_WHITE_TUBES: dict[str, tuple[float, float, float, float, float]] = {
    "link2_geom_0": (0.0580, 0.2461, 0.0745, 0.0478, 0.0385),
    "link3_geom_0": (0.0856, 0.2545, 0.0119, 0.0481, 0.0448),
}
"""Geom-name suffix -> ``(z_lo, z_hi, axis_x, axis_y, radius)`` [m, mesh-local].

Keyed by *suffix* rather than by the whole name: a bimanual MJCF prefixes its
geoms per side (``right_link2_geom_0``), and the tube is the same part on both
arms."""


def split_yam_white_tubes(builder: newton.ModelBuilder, shape_range: tuple[int, int]) -> int:
    """Give the arm its white tubes, as visual-only shapes. Returns links split.

    Called after the embodiment's flat colour has been applied, so it repaints
    exactly the two links that are not flat black and leaves everything else as
    the spec stated it.

    A mesh that no longer spans what the derivation saw is skipped rather than
    sliced at stale coordinates -- a re-exported or re-centred link would
    otherwise get a white band across the wrong part of it. Returning 0 means
    no link matched and the arm simply stays flat black, which is a degraded
    look and never a broken model.

    Args:
        builder: The builder, pre-finalize.
        shape_range: ``(start, end)`` shape indices this robot owns.

    Returns:
        How many links were split.
    """
    n_split = 0
    for i in range(*shape_range):
        tail = builder.shape_label[i].rsplit("/", 1)[-1]
        if any(tail.endswith(k) for k in YAM_WHITE_LINKS):
            builder.shape_color[i] = YAM_WHITE
            continue
        spec = next((v for k, v in YAM_WHITE_TUBES.items() if tail.endswith(k)), None)
        if spec is None:
            continue
        source = builder.shape_source[i]
        vertices = getattr(source, "vertices", None)
        if vertices is None:
            continue
        v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        f = np.asarray(source.indices, dtype=np.int32).reshape(-1, 3)
        z_lo, z_hi, cx, cy, radius = spec
        if not (v[:, 2].min() < z_lo - 0.02 and v[:, 2].max() > z_hi + 0.02):
            continue
        white_v = (v[:, 2] >= z_lo) & (v[:, 2] <= z_hi) & (np.hypot(v[:, 0] - cx, v[:, 1] - cy) <= radius)
        white_f = white_v[f].all(axis=1)
        if not white_f.any() or white_f.all():
            continue
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        for faces, colour, suffix in ((f[white_f], YAM_WHITE, "_white"), (f[~white_f], YAM_BLACK, "_black")):
            builder.add_shape_mesh(
                body=int(builder.shape_body[i]),
                xform=builder.shape_transform[i],
                mesh=newton.Mesh(v, faces.reshape(-1), compute_inertia=False, is_solid=False),
                cfg=cfg,
                color=colour,
                label=builder.shape_label[i] + suffix,
            )
        builder.shape_flags[i] = int(builder.shape_flags[i]) & ~int(newton.ShapeFlags.VISIBLE)
        n_split += 1
    return n_split
