# SPDX-License-Identifier: Apache-2.0
"""The scene ``00_wipe_the_bar.py`` drives: a real cell's grab bar, sponge and marker stripe, measured off its clip.

The cell is a real one: a YAM pair, a chrome grab bar bolted into two black
blocks on a gridded foam mat, a kitchen sponge, and a blue marker line drawn
along the bar's crown. Every number in :class:`Layout` was measured off the
clip's frames against the mat's 25 mm grid. They are what makes this a twin;
the rest of the file builds what they describe.

In order:

1. **The layout**: where everything stands, and the physics it was measured under.
2. **The bar**: a 25 mm tube on two elbows, welded into its blocks as the real one is bolted.
3. **The stripe**: the marker line, which wears off where the foam presses *and* slides.
4. **The scene**: two YAMs, the mat, the blocks, the bar, the sponge in the air where the hand will hold it.
5. **The holds**: where each hand goes, and the latch that keeps the sponge in the right one.
6. **The contact map**: the stripe unrolled, with the press drawn on it.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace

import newton
import numpy as np
import warp as wp

import robosimstudio as rss
from robosimstudio.core.math import quat_multiply, quat_to_mat
from robosimstudio.core.render.finish import BRUSHED_METAL, MATTE, SATIN
from robosimstudio.core.render.room import LAB_GRID_MAT, LAB_MAT_GREY
from robosimstudio.domains.capabilities import host_bridge
from robosimstudio.domains.rigid import EntityRigid
from robosimstudio.scene.entity import label_index
from robosimstudio.specs import ActuationSpec

# =============================================================================
# 1. The layout
# =============================================================================


@dataclass(frozen=True)
class Layout:
    """Where everything stands [m], in the bench's work-surface frame: +x away from the arms, -y the right arm.

    Measured off the clip against the mat's 25 mm grid (300 mm tiles): the tube is
    one square across, the sponge three squares long, the end blocks three wide.
    """

    mat_thickness: float = 0.010  # an EVA puzzle mat over the whole bench top
    tube_radius: float = 0.0125  # a 25 mm tube: one grid square across
    straight_half: float = 0.170  # half the straight run: 13.5 squares between the elbows
    bend_radius: float = 0.032  # centreline radius of the elbows
    leg_drop: float = 0.050  # bar axis to the blocks' top face: two tube diameters
    block_half: tuple[float, float, float] = (0.038, 0.040, 0.035)  # 76 deep, 80 along the bar, 70 tall
    block_inset: tuple[float, float] = (0.018, 0.020)  # block centre from its leg: +x, and outward along the bar
    bar_x: float = 0.0
    bar_y: float = -0.02  # a shade toward the working (right, -y) arm
    sponge_size: tuple[float, float, float] = (0.050, 0.075, 0.030)  # three squares by two, long axis along the bar
    sponge_cells: tuple[int, int, int] = (9, 13, 6)  # ~5.5 mm nodes, 980 of them
    stripe_half: float = 0.100  # the marker line covers the middle 200 mm of the run
    stripe_half_angle_deg: float = 9.0  # and is ~4 mm wide round the tube
    support_u: float = 0.170  # the left hand steadies the far elbow's tangent point

    @property
    def mat_top(self) -> float:
        return self.mat_thickness

    @property
    def bar_z(self) -> float:
        """The bar's axis height: the blocks' top plus the leg drop."""
        return self.mat_top + 2.0 * self.block_half[2] + self.leg_drop

    @property
    def crown_z(self) -> float:
        return self.bar_z + self.tube_radius

    def block_centre(self, sign: float) -> tuple[float, float, float]:
        """A block's centre; ``sign`` is +1 for the far (+y) end, -1 for the near."""
        leg_u = self.straight_half + self.bend_radius
        return (
            self.bar_x + self.block_inset[0],
            self.bar_y + sign * (leg_u + self.block_inset[1]),
            self.mat_top + self.block_half[2],
        )


LAYOUT = Layout()

PHYSICS = rss.ArrangementSpec(
    control_hz=60.0,
    substeps=16,
    backends=(
        # The arms, the bar and the bench on MuJoCo; the sponge on SuperDex, two engine steps a frame.
        (
            "*",
            rss.backend(
                "mujoco",
                use_mujoco_contacts=True,
                nconmax=2048,
                njmax=4096,
                iterations=100,
                ls_iterations=50,
                cone="elliptic",
                impratio=10.0,
                enable_multiccd=True,
            ),
        ),
        # feedback="off": the foam does not push back on the arm. Squeezed in the jaws it hands
        # the fingers a steady couple, and the page, which re-seats an idle gizmo on the hand,
        # turns that into a hand creeping 0.5 mm/s. The arm presses by position instead; at a
        # few newtons the stiffened servos below would yield about a millimetre.
        ("soft", rss.backend("superdex", substeps=2, feedback="off")),
    ),
    collision=rss.CollisionSpec(contact_ke=6.0e4, contact_kd=500.0, rigid_contact_max=8192, rigid_gap=2.0e-4),
)
"""The arrangement the twin was measured under, less the foam's push on the arm."""

ARM = ActuationSpec(target_ke=1600.0, target_kd=70.0)
"""Stiffened arm servos: at the robot file's own gain the hand lags ~10 mm under a press."""
GRIPPER = ActuationSpec(target_ke=20000.0, target_kd=20.0)

CHROME = (0.80, 0.80, 0.80)
INK = (0.060, 0.090, 0.300)
STEEL = replace(BRUSHED_METAL, roughness=0.3, metallic=0.95)

# =============================================================================
# 2. The bar
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class BarSpec(rss.ObjectSpec):
    """A tube on two elbows whose legs drop into the blocks. Its axis is local z, its crown local +x."""

    entity: str = "grab_bar_tube"
    name: str = "bar"
    static: bool = True
    radius: float = LAYOUT.tube_radius
    half_len: float = LAYOUT.straight_half
    bend_radius: float = LAYOUT.bend_radius
    leg_drop: float = LAYOUT.leg_drop
    buried: float = 0.015  # how far each leg runs into its block
    color: tuple[float, float, float] = CHROME
    friction: float = 0.6

    def __post_init__(self) -> None:
        if self.shape is None:  # the straight run is the spec's own cylinder
            object.__setattr__(
                self, "shape", rss.ShapeSpec(kind="cylinder", radius=self.radius, half_height=self.half_len)
            )


@rss.register_entity("grab_bar_tube")
class GrabBar(EntityRigid):
    """The straight run, built by ``EntityRigid``, plus an elbow and a leg at each end, welded to the world."""

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        super().build(builder, episode)
        spec = self.spec
        # A static object is a jointless link, which MuJoCo's converter refuses: weld it to the world.
        builder.add_joint_fixed(-1, self.body, parent_xform=builder.body_q[self.body], label=f"{spec.name}_weld")
        cfg, r, bend = self._contact_cfg(builder), float(spec.radius), float(spec.bend_radius)
        for end, sign in (("far", 1.0), ("near", -1.0)):
            centre = np.array([-bend, 0.0, sign * spec.half_len])  # the bend's centre of curvature
            points = [
                centre + bend * np.array([math.sin(a), 0.0, sign * math.cos(a)]) for a in np.linspace(math.pi / 2, 0, 6)
            ]
            points.append(points[-1] + np.array([-(spec.leg_drop - bend + spec.buried), 0.0, 0.0]))  # the leg
            for k, (a, b) in enumerate(itertools.pairwise(points)):
                self._tube(builder, a, b, r, cfg, f"{spec.name}_{end}_{k}")
                builder.add_shape_sphere(
                    self.body, xform=wp.transform(wp.vec3(*b), wp.quat_identity()), radius=r, cfg=cfg, color=CHROME
                )

    def _tube(self, builder, a, b, radius, cfg, label) -> None:
        d = b - a
        length = float(np.linalg.norm(d))
        axis = np.cross([0.0, 0.0, 1.0], d / length)  # a cylinder's axis is its local z: turn z onto d
        s = float(np.linalg.norm(axis))
        q = (
            wp.quat_from_axis_angle(wp.vec3(*(axis / s)), math.atan2(s, d[2] / length))
            if s > 1e-9
            else wp.quat_identity()
        )
        builder.add_shape_cylinder(
            self.body,
            xform=wp.transform(wp.vec3(*(0.5 * (a + b))), q),
            radius=radius,
            half_height=0.5 * length,
            cfg=cfg,
            color=CHROME,
            label=label,
        )

    def apply_display(self, builder: newton.ModelBuilder) -> None:
        super().apply_display(builder)
        for s in range(*self.range.shapes):
            self.finishes[s] = STEEL


# =============================================================================
# 3. The stripe
# =============================================================================
# The marker line is cells on the tube's surface, each with ink in [0, 1], drawn as
# thin decals on the bar. Preston's law wears them: ink falls at k_p * pressure *
# sliding speed, so a still press removes nothing and nothing ever comes back.
# The pressure is the foam's own: SuperDex reports every contact sample's force
# and quadrature area, and a cell is worn at the rate of the nearest sample.


@dataclass(frozen=True, kw_only=True)
class StripeSpec:
    """A marker line along the bar's crown, as cells on the tube that wear off."""

    entity: str = "marker_stripe"
    name: str = "stripe"
    bar: str = "bar"
    sponge: str = "sponge"
    radius: float = LAYOUT.tube_radius
    half_length: float = LAYOUT.stripe_half
    half_angle: float = math.radians(LAYOUT.stripe_half_angle_deg)
    pitch: float = 0.0025  # cell size on the surface: about the contact samples' spacing
    reach: float = 0.004  # how far one contact sample wipes
    k_p: float = 1.2e-2  # Preston coefficient [1/(Pa m)]: a 4 N, 30 mm/s stroke clears a cell in ~2 s
    p_min: float = 300.0  # [Pa] the contact's 2 mm padding band reports samples at ~0 N; they wipe nothing
    color: tuple[float, float, float] = INK
    bare: tuple[float, float, float] = CHROME


@rss.register_entity("marker_stripe")
class MarkerStripe(rss.Entity):
    """Decals on the bar, the ink under them, and the wear law, run once per control step."""

    recolors_live = True  # tells the studio to redraw the decals' colours as they change

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        spec = self.spec
        self.bar_body = label_index(list(builder.body_label), spec.bar)
        nu = round(spec.half_length / spec.pitch)
        na = round(spec.half_angle * spec.radius / spec.pitch)
        u, a = np.meshgrid(np.arange(-nu, nu + 1) * spec.pitch, np.arange(-na, na + 1) * spec.pitch / spec.radius)
        self.cells = np.stack([u.ravel(), a.ravel()], axis=1)  # (axial [m], angle from the crown [rad])
        self.ink = np.ones(len(self.cells))
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        self.first_decal = builder.shape_count
        lift = spec.radius + 2.0e-4
        for i, (ui, ai) in enumerate(self.cells):
            turn = wp.quat_from_axis_angle(wp.vec3(-math.sin(ai), math.cos(ai), 0.0), math.pi / 2)  # +z outward
            builder.add_shape_box(
                self.bar_body,
                xform=wp.transform(wp.vec3(lift * math.cos(ai), lift * math.sin(ai), ui), turn),
                hx=spec.pitch / 2,
                hy=spec.pitch / 2,
                hz=1.0e-4,
                cfg=cfg,
                color=spec.color,
                label=f"{spec.name}_{i:03d}",
            )
        self.bridge, self.press, self.peak, self.samples = None, 0.0, 0.0, np.zeros((0, 3))

    def apply_display(self, builder: newton.ModelBuilder) -> None:
        super().apply_display(builder)
        for s in range(*self.range.shapes):
            self.finishes[s] = STEEL

    def contribute_hooks(self, hooks) -> None:
        # After the step, where SuperDex has just stepped and its contact query is fresh.
        hooks.add("after", self.wipe, name="marker-stripe", cadence=1)

    def on_scene_ready(self, pipeline) -> None:
        self.bridge = host_bridge(pipeline, self.spec.sponge)
        if self.bridge is None:
            raise RuntimeError(f"{self.spec.sponge!r} is not on SuperDex, so there is no contact patch to wipe with")
        pose = pipeline.state_0.body_q.numpy()[self.bar_body]  # welded: read once
        self.origin, self.rot = pose[:3].astype(np.float64), quat_to_mat(pose[3:7].astype(np.float64))

    def wipe(self, state, dt) -> None:
        """One control step of wear, from this step's contact samples against the bar."""
        if self.bridge is None:
            return
        spec = self.spec
        s = self.bridge.contact_points(spec.sponge, against=self.bar_body)
        local = (s["pos"] - self.origin) @ self.rot  # world -> the bar's own frame
        on_tube = np.abs(np.hypot(local[:, 0], local[:, 1]) - spec.radius) <= 0.004
        force = np.linalg.norm(s["force"], axis=1)
        pressure = force / np.maximum(s["area"], 1.0e-9)
        u, angle = local[:, 2], np.arctan2(local[:, 1], local[:, 0])
        self.press = float(force[on_tube].sum())
        self.peak = float(pressure[on_tube].max()) if on_tube.any() else 0.0
        self.samples = np.stack([u, angle, pressure], axis=1)[on_tube]
        live = on_tube & (pressure >= spec.p_min)
        if not live.any():
            return
        # Only the sliding part of the relative velocity wears: pressing straight down is a stamp, not a wipe.
        normal = s["force"] / np.maximum(force, 1.0e-12)[:, None]
        rel = s["v_soft"] - s["v_other"]
        slide = np.linalg.norm(rel - normal * np.sum(rel * normal, axis=1)[:, None], axis=1)
        du = self.cells[:, :1] - u[live][None]
        da = (self.cells[:, 1:] - angle[live][None] + math.pi) % (2 * math.pi) - math.pi
        d2 = du**2 + (da * spec.radius) ** 2
        nearest = np.argmin(d2, axis=1)
        reached = d2[np.arange(len(self.cells)), nearest] <= spec.reach**2
        rate = spec.k_p * pressure[live] * slide[live]
        self.ink[reached] = np.maximum(0.0, self.ink[reached] - rate[nearest[reached]] * float(dt))
        self.paint()

    def paint(self) -> None:
        """Fade each decal from the ink's colour to the bare chrome under it."""
        bare, ink = np.asarray(self.spec.bare), np.asarray(self.spec.color)
        colours = bare + (ink - bare) * self.ink[:, None]
        model = self.stepping_model if self.stepping_model is not None else self.model
        src = wp.array(colours.astype(np.float32), dtype=wp.vec3, device=model.device)
        wp.copy(model.shape_color, src, dest_offset=self.first_decal, count=len(colours))

    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        self.ink[:] = 1.0
        self.press, self.peak, self.samples = 0.0, 0.0, np.zeros((0, 3))
        if self.model is not None:
            self.paint()


# =============================================================================
# 4. The scene
# =============================================================================

TILT_DEG = 35.0
"""How far the right hand leans along the bar. The real hand does; upright, the
wrist runs out of orientation at the far end of the stripe."""
GRIP_GAP = 0.032  # [m] the right jaws squeeze the 50 mm sponge to this, 36 % strain
SUPPORT_GAP = 2 * LAYOUT.tube_radius + 0.004  # [m] the left jaws rest on the tube: a steadying hand, not a clamp
PAD_OFFSET = (0.001, 0.0, 0.007)
"""Where the YAM's pads touch a held object, in the tool frame [m]: measured off
MuJoCo's contact table on a working grasp. The pads lead the tool centre by 7 mm."""
HOVER = 0.010  # [m] the sponge's lowest point starts this far above the crown


def hand_quat(tilt_deg: float = 0.0) -> np.ndarray:
    """Top-down, jaws closing across x, leaning ``tilt_deg`` about x so the hand tips toward +y."""
    down = quat_multiply((0.0, 0.0, -math.sin(math.pi / 4), math.cos(math.pi / 4)), (1.0, 0.0, 0.0, 0.0))
    t = math.radians(tilt_deg)
    return np.asarray(quat_multiply((math.sin(t / 2), 0.0, 0.0, math.cos(t / 2)), down), dtype=np.float64)


def sponge_in_hand(layout: Layout = LAYOUT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where the right hand holds the sponge at the start [surface frame].

    Over the stripe's near end, leaning, the foam's lowest point :data:`HOVER`
    above the crown. Returns the tool centre, the hand's quaternion and the
    sponge's bottom-face centre -- one calculation, so the sponge is built
    exactly where the hand is then posed.
    """
    half = 0.5 * np.asarray(layout.sponge_size)
    tilt = math.radians(TILT_DEG)
    rot = quat_to_mat((math.sin(tilt / 2), 0.0, 0.0, math.cos(tilt / 2)))
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
    u = -layout.stripe_half + half[1] - 0.025  # the sponge's near edge 25 mm past the stripe's end
    centre = np.array([layout.bar_x, layout.bar_y + u, 0.0])
    centre[2] = layout.crown_z + HOVER - float((corners @ rot.T)[:, 2].min())
    quat = hand_quat(TILT_DEG)
    tcp = centre - quat_to_mat(quat) @ np.asarray(PAD_OFFSET)
    return tcp, quat, centre - rot @ np.array([0.0, 0.0, half[2]])


def scene(layout: Layout = LAYOUT) -> rss.Scene:
    """Two YAMs on the bench, the gridded mat, the blocks, the bar, the sponge in the air where the hand will hold it."""
    sc = rss.Scene(cell="bench_075", name="wipe_the_bar", physics=PHYSICS)
    sc.add(rss.robot("yam_i2rt", arm=ARM, gripper=GRIPPER))
    fixture = rss.SimRequirements(domain="static", contacts=rss.contacts_with("objects", "robot", "deformables"))
    hx, hy = rss.get_cell("bench_075").half[:2]
    t = layout.mat_thickness / 2
    sc.add(
        rss.StaticSpec(
            name="mat",
            half_extents=(hx, hy, t),
            xform=(0.0, 0.0, t, 0.0, 0.0, 0.0, 1.0),
            friction=0.9,
            color=LAB_MAT_GREY,
            finish=LAB_GRID_MAT,
            sim=fixture,
        )
    )
    for which, sign in (("near", -1.0), ("far", 1.0)):
        sc.add(
            rss.StaticSpec(
                name=f"block_{which}",
                half_extents=layout.block_half,
                xform=(*layout.block_centre(sign), 0.0, 0.0, 0.0, 1.0),
                friction=0.9,
                color=(0.018, 0.018, 0.021),
                finish=SATIN,
                sim=fixture,
            )
        )
    sc.add(
        BarSpec(
            pose=(layout.bar_x, layout.bar_y, layout.bar_z),
            quat=(-0.5, -0.5, -0.5, 0.5),  # local z (the axis) onto world y, local x (the crown) onto world z
            sim=rss.SimRequirements(
                domain="rigid", contacts=rss.contacts_with("surface", "fixtures", "robot", "deformables")
            ),
        )
    )
    _, _, bottom = sponge_in_hand(layout)
    tilt = math.radians(TILT_DEG)
    sc.add(
        rss.sponge(
            name="sponge",
            size=layout.sponge_size,
            cells=layout.sponge_cells,
            at=tuple(bottom),
            quat=(math.sin(tilt / 2), 0.0, 0.0, math.cos(tilt / 2)),
            settle_frames=0,  # it is held, not dropped: the example closes the hand on it
            color=(0.93, 0.94, 0.94),
            finish=replace(MATTE, roughness=0.95),
            sim=rss.SimRequirements(
                domain="soft", contacts=rss.contacts_with("surface", "hands", "fixtures", "objects")
            ),
        )
    )
    sc.add(StripeSpec(radius=layout.tube_radius, half_length=layout.stripe_half))
    return sc


# =============================================================================
# 5. The holds
# =============================================================================

GRIPS = {"right": GRIP_GAP, "left": SUPPORT_GAP}
"""Jaw gap [m] each hand closes to."""


def holds(sim, layout: Layout = LAYOUT) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Where each tool centre goes [m, world] and how it is turned: the right hand
    where the sponge was built, the left astride the tube at the far elbow."""
    cell = rss.get_cell("bench_075")
    tcp, quat, _ = sponge_in_hand(layout)
    bar = sim["bar"].pose
    elbow = bar[:3] + quat_to_mat(bar[3:7]) @ np.array([0.0, 0.0, layout.support_u])
    return {
        "right": (tcp + np.array([cell.center[0], cell.center[1], cell.surface_z]), quat),
        "left": (elbow + np.array([0.0, 0.0, 0.004]), hand_quat(0.0)),  # pads straddling the tube
    }


class Latch:
    """The foam under the right pads, pinned to the pad frame and carried with it every control step.

    The jaws hold the sponge by friction, and SuperDex's friction has no static
    part below 10 mm/s of slip: an idle hand lets the foam creep out in about ten
    seconds, and a press into a round bar with the hand leaning twists it out on
    the first stroke. So, as the real rig's twin does, the patch under each pad
    is pinned with the engine's own compliant node constraints to where it sits
    in the hand. The rest of the block is free foam and still wraps the tube.
    Opening the right hand lets go of it. Install it in ``sim.post_step_hooks``.
    """

    def __init__(self, sim) -> None:
        sponge = sim["sponge"]
        self.bridge = sponge.entity.bridge
        self.command = sim.gripper("right").command_for_width(GRIP_GAP)
        pos, quat = sim.arm("right").ee_pose(sim.state)
        local = (sponge.particle_q - pos) @ quat_to_mat(quat)  # every node, in the hand's frame
        pad = np.asarray(PAD_OFFSET)
        # A 14 mm disc on each gripped face: the jaws close along the hand's y.
        disc = np.hypot(local[:, 0] - pad[0], local[:, 2] - pad[2]) <= 0.014
        face = np.abs(local[:, 1]) >= 0.5 * GRIP_GAP - 0.006
        self.local = local[disc & face]
        self.pin = self.bridge.pin_nodes(
            "sponge", sponge.particles[disc & face], self._targets(pos, quat), stiffness=4000.0, damping=20.0
        )

    def _targets(self, pos, quat) -> np.ndarray:
        return np.asarray(pos) + self.local @ quat_to_mat(quat).T

    def __call__(self, sim) -> None:
        if self.pin is None:
            return
        if sim.gripper("right").command < 0.5 * self.command:
            self.bridge.unpin(self.pin)
            self.pin = None
            return
        try:
            self.bridge.set_pin_targets(self.pin, self._targets(*sim.arm("right").ee_pose(sim.state)))
        except (KeyError, IndexError):  # a reset took the pins with it
            self.pin = None


# =============================================================================
# 6. The contact map
# =============================================================================

EYE, LOOK = (0.02, -0.04, 1.32), (0.58, -0.02, 0.86)
"""[m, world] Where the page opens looking from, and at: from behind the arms, over their shoulders."""
MAP_SPAN = (0.135, 0.015)  # [m] the map's half-extent: along the bar, round it
MAP_SCALE = (1.5, 4.0)  # [px/mm] along, round: the stripe is fifty times longer than it is wide
MAP_FULL_SCALE = 40000.0  # [Pa] red: the per-sample peaks of a firm wipe, where the foam wraps the tube


def contact_map(stripe: MarkerStripe) -> np.ndarray:
    """The bar's surface unrolled round the stripe: the ink left on each cell, and the press on top.

    Along the bar runs across the picture, the right hand's end on the right as
    the page opens; round the tube runs up and down, the far side on top. Each
    contact sample is a dot, yellow to red with its pressure, and faint where it
    is under the floor that wears anything.
    """
    ku, ka = 1000.0 * MAP_SCALE[0], 1000.0 * MAP_SCALE[1]
    w, h = int(2 * MAP_SPAN[0] * ku), int(2 * MAP_SPAN[1] * ka)
    img = np.full((h, w, 3), 60.0)
    spec = stripe.spec

    def px(u, angle):
        return ((MAP_SPAN[0] - u) * ku).astype(int), ((MAP_SPAN[1] - angle * spec.radius) * ka).astype(int)

    bare, ink = np.asarray(spec.bare) * 255, np.asarray(spec.color) * 255
    hu, ha = int(spec.pitch * ku / 2) + 1, int(spec.pitch * ka / 2)
    for x, y, left in zip(*px(stripe.cells[:, 0], stripe.cells[:, 1]), stripe.ink, strict=True):
        img[max(0, y - ha) : y + ha, max(0, x - hu) : x + hu] = bare + (ink - bare) * left
    xs, ys = px(stripe.samples[:, 0], stripe.samples[:, 1])
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    for x, y, p in zip(xs[inside], ys[inside], stripe.samples[inside, 2], strict=True):
        t = min(1.0, p / MAP_FULL_SCALE)
        alpha = 0.35 + 0.6 * t if p >= spec.p_min else 0.18
        dot = img[max(0, y - 3) : y + 4, max(0, x - 2) : x + 3]
        dot[:] = dot * (1 - alpha) + np.array([255.0, 230.0 * (1 - t), 40.0 * (1 - t)]) * alpha
    return img.clip(0, 255).astype(np.uint8)
