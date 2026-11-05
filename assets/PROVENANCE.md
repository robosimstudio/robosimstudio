# Vendored assets

Each directory here is a copy of geometry authored elsewhere, vendored so a task
suite is self-contained and reproducible. This file records where each came
from and what was changed.

Robot descriptions (embodiments) live under `embodiments/`. Geometry that one
task declares lives beside that task, in `<collection>/tasks/<name>/assets/`,
with its own `PROVENANCE.md`; `register_asset` rewrites a record's `files` to
absolute paths at registration. An export that a collection's documents stage
lives with that collection, with its own `PROVENANCE.md`; a document names its
path in `(:staging (:export …))`.

`robosimstudio.paths.resolve_path` searches the checkout's `assets/`, then each
collection's `assets/`, then the copy a wheel force-includes at
`robosimstudio/assets/`. `ROBOSIMSTUDIO_ASSETS` overrides the search outright,
for a checkout that keeps large binaries out of the package.

Files marked *generated* below were written by development scripts that are not
in this repository and checked against their source; the checked-in files are
the source of truth.

---

## `embodiments/yam/` — I2RT YAM bimanual arena

**Source:** the arena's `sim_models/` directory, in a read-only reference
checkout of the Isaac Lab Arena YAM stage
([isaac-sim/IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena),
Apache-2.0, Copyright 2025 NVIDIA Corporation and the Isaac Lab Arena Project
Developers). The arm geometry and meshes in it are i2rt's YAM (MIT). `LICENSE`
beside `yam_bimanual_arms.xml` is the Apache License 2.0 text that covers the
arena-authored files; `i2rt_yam/LICENSE` is i2rt's MIT licence for the meshes.

**Files**

| File | What it is |
| --- | --- |
| `yam_bimanual_arms.xml` | *Generated.* The arena's two YAM arms and nothing else, at their own base origin: the scene as authored (two YAM arms, a gate, cameras, a table, seven free-jointed props) with every free-jointed body and the home keyframe removed, the cell's scenery deleted, and the arm roots moved to the origin. **What the embodiment imports.** |
| `yam_bimanual_arm.urdf` | One arm's kinematic chain (`base_link`, `link_1`..`link_6`, joints `joint1`..`joint6`) transcribed from `yam_bimanual_arms.xml`'s bodies: each body's `pos`/`quat` as the joint origin, each joint's axis and range, in chain order. No geometry or inertia. For the PyRoKi IK solver, which checks it against the MJCF at construction and refuses drift. Named by `yam_bimanual` (`robots/embodiments/yam_bimanual.py`). |
| `i2rt_yam/` | The meshes the MJCF names. Unmodified. Carries its own `LICENSE`. |
| `LICENSE` | Isaac Lab Arena's `LICENSE.md` (Apache License 2.0), under which the arena distributes the files this directory derives from; fetched from `isaac-sim/IsaacLab-Arena` @ `9abd5dd0` on 2026-10-13. |

Not shipped: i2rt's per-arm `yam.urdf` and its meshes under `i2rt_yam/urdf/`
(unused; the IK drift check refuses it, since it lists `joint6` first), and the
arena's 9 MB `base_visual_gate.stl` (unreferenced after stripping).

**Stripped:** the free-jointed props (`bottle_1` … `bottle_6`,
`bin_container`) and the home keyframe, whose `qpos` indexed them; a task
supplies its own objects. The strip deletes only `<body>` elements carrying a
`<freejoint>` and the `<keyframe>` block, and `meshdir` still resolves because
the file sits beside `i2rt_yam/`. `robosimstudio.scene.build.imports.strip_free_bodies`
does the same deletion at runtime; the result is checked in so the import reads
this file.

**Scenery removed:** a light, a floor plane, a gantry mesh with its collision
posts and a camera bar, three wall panels, a visual table slab, and an
invisible `table_plane` collider at z = 0.728 (22 mm below the surface of a
0.750 bench). The arm roots moved from `(0.2525, ±0.31, 0.75)` to the origin.
The ±0.31 shoulder separation is kept; 0.2525 and 0.75 became
`EmbodimentSpec.base_offset` against a scene's work surface.

**What changed on the way in** -- the hand, edited in the checked-in file on
2026-10-26 so it grips like the robot it models (`i2rt-robotics/i2rt` @
`120c3c8`, v1.3.6). Measured on a top-down grasp of a free box (mu 0.3) with the
right hand, the box in the arm's MuJoCo island (`MUJOCO_500HZ`):

* **Collider friction 1.0**, from 2.0 on the `collision` class and 4.0 on the
  0.6 mm pad nubs (`sphere_collision`). MuJoCo gives a contact the larger of its
  two geoms' frictions, so every finger contact ran at 2 to 4 whatever the
  object said: the mu 0.3 box read `[2.0, 4.0]` in `mjw_data.contact.friction`,
  and reads `[1.0]` now. 1.0 is what i2rt's own `linear_4310.xml` leaves on its
  tips (MuJoCo's default).
* **The finger equality is stiff** (`solref="0.004 1" solimp="0.95 0.99 0.001"`,
  from MuJoCo's default 0.02). At the default the carried finger gave 4 mm under
  a 24 N squeeze, so the hand pinched off centre and lost force to the coupling;
  the box now sits within 0.2 mm of the grasp point.
* **The finger actuator's `forcerange="-100 100"`** (it had none): the driver's
  limit. i2rt runs the real arm with `limit_gripper_force=50.0`
  (`i2rt/robots/get_robot.py`), and its `GripperForceLimiter`
  (`i2rt/robots/utils.py`) sees a blocked jaw and holds the motor at the torque
  that makes 50 N of grip, whatever the width. The one actuator drives both
  fingers through the equality, which splits its force onto the two jaws, so
  100 N is 50 N a jaw.

**The finger gains are not the file's.** The file keeps the arena's, kp 200 N/m,
kv 14: a spring whose squeeze grew with the object, 0.3 N a jaw on a 6 mm box,
1.5 N on 30 mm and 2.5 N on 50 mm. The library drives the hand with its spec's
own servo instead (`yam_linear_arena.CLAMP`, a `GripperSpec.actuation`), so a
page that states its own servo replaces it and gets this file's joint physics
back, under the same cap:

* **A stiff servo**, kp 300000, kv 300, the real one's stiffness in kind
  (`i2rt/robots/config/linear_4310.yml`: a DM4310 at kp 20, 6.57 rad of motor per
  0.096 m of jaw, so about 94 kN/m at the jaw before the limiter): the cap is
  reached 0.33 mm past contact, and 50 N a jaw measured on boxes of 1, 2, 6, 30
  and 60 mm.
* **230 N*s/m of damping on each finger joint.** It sets the closing speed: a
  clamp this stiff and undamped slams the pads, and at kp 200 kN/m they crossed
  56 mm through each other shut on nothing and went through a 1-2 mm plate. The
  equality ties the two joints, so the pair carries 460: pulling at the 100 N
  cap a finger runs at 100/460 = 0.22 m/s and the jaw at 0.43 m/s, the DM4310 at
  30 rad/s (the velocity range i2rt's motor table gives it) through 6.57 rad per
  96 mm. That is an upper bound, since no closing time is published for the
  hardware: shut on nothing from full open takes 0.20 s to 90 % at 50, 60 and
  240 Hz control (0.16 s on the spring). On both joints rather than the driven
  one: with the carried finger left at 0.5 it rang against the stiff equality,
  flung a held box under `GENERIC` and crossed a 6 mm plate at 240 Hz.

After: every box from 1 to 60 mm held to the 2 kg the probe stops at (the YAM's
payload; the arm itself sags 28 mm at 2 kg), where the arena hand dropped the
6 mm box at 0.29 kg and the 30 mm one at 1.31 kg on its mu 2 to 4. Under
`GENERIC`, the box on VBD behind the proxy coupler (the finger contacts resolved
by VBD, which mixes friction as the geometric mean), the 6, 30 and 60 mm boxes
lifted and held to 1.9, 1.5 and 1.8 kg, where the arena hand never lifted the
6 mm box and dropped the 30 mm one at 0.21 kg. Two limits: a 1-2 mm plate
standing on edge is not reached under `GENERIC` by either hand (the tool centre
stops 11 mm high over it), and at 240 Hz (`rigid_mujoco@v1`'s rate) with the
plate in MuJoCo the pads cross it in two steps.

The arm's gains stay the arena's, kp 80 kv 6 on joints 1-3 and kp 30 kv 2 on
the wrist, though i2rt's `yam_v1.yml` runs kp 80/80/80/10/10/10 and kd
5/5/5/1.5/1.5/1.5. Decided by measurement: with i2rt's gains in the classes,
the medium tier lost six shipped tasks that pass on the arena's -- the towel
example's edge slipped out of the jaws (0 of 10 particles held), the
fold-the-towel demo scored 0.74 against 0.85 (its near half dragged 27 mm), a
scrub-a-stain stroke inside its force band wiped 0.56 of the stain against 0.8,
the suction hand stopped 7.5 mm short of its plate, the paper fold kept
2.54 rad against 2.8 and the stuck label bonded 0.07 against 0.8 -- and the arm
sagged 32 mm, not 28, under the 2 kg probe box. The routines were written on
the stiffer wrist; until they are re-planned, it stays.

**Arena constants**, restated in `robosimstudio/robots/embodiments/yam_bimanual.py` rather than
imported from the reference: table top at `z = 0.728 m`, table centred at
`(0.6, 0.0)` with half-extents `(0.30, 0.40)`, per-arm home
`[0, 1.047, 1.047, 0, 0, 0]`, grasp site `0.1347 m` along the wrist's `+z` with
a −90° twist about `z`.

---

## `embodiments/panda/` — Franka Panda: robosuite's model, and a kinematics URDF

### The physics model — robosuite 1.4.0's Panda, as the VAB tool-sort export carries it

robosuite 1.4.0 (MIT) arm and hand, cut from VAB-industrial's `ToolSortPickPlace`
export; mesh geometry from Franka's `franka_description` (Apache-2.0).
`LICENSE` is robosuite's, fetched on 2026-10-04 and byte-identical to the
`v1.4.0` tag's. Source, files, changes and the measurements across the copy:
[`embodiments/panda/README.md`](embodiments/panda/README.md).

### The kinematics URDF — for full-orientation IK

**Source:** `example-robot-data` @ `d0d9098d752014aec3725b07766962acf06c5418`
(`robots/panda_description/urdf/panda.urdf`, via the machine's
`robot_descriptions` cache). BSD 3-Clause, Copyright (C) 2018-2020 LAAS-CNRS,
University of Edinburgh, INRIA: `LICENSE.example-robot-data` is that
repository's `LICENSE` at the same commit, and `NOTICE` credits it.

**Files**

| File | What it is |
| --- | --- |
| `panda_kinematics.urdf` | The URDF with every `<visual>` and `<collision>` element removed — joints, links, limits and inertials only. |

**Stripped** because the PyRoKi IK backend (`robosimstudio.solvers.ik.pyroki`)
reads kinematics only, and a kinematics-only file has no mesh paths to break.
The physics Panda is the MJCF above; this file is never simulated, only solved
against.

**Tool-frame check:** the PyRoKi backend measures `T_hand→tcp =
fk_urdf(q)⁻¹ · T_base⁻¹ · fk_sim(q)` at two configurations at construction and
refuses drift over 1 mm / 0.1°. Measured on the tool-sort scene: ≤0.001 mm,
0.0000°.

**Per-link parity, for a planner.** The tool-frame check is enough for IK,
which needs only the tool point, but not for a planner, which also needs the
elbow. Measured link by link against the tool-sort MJCF at 64 configurations
(seed 0, home, both limit corners, 61 uniform), with a per-link frame
comparator that has since been removed:

| Frame | vs MJCF | Verdict |
| --- | --- | --- |
| `panda_link0` … `panda_link7` | ≤ **0.0016 mm** / ≤ 0.00002° | the same arm, to float32 |
| `panda_hand` | 0.4994 mm along the link's own −z, spread 0.0025 mm | a constant |
| `panda_leftfinger`, `panda_rightfinger` | 6.4989 mm along −z, spread 0.0031 mm | a constant |

The hand and finger offsets are frame-origin conventions: robosuite places the
hand's origin half a millimetre from franka_description's, and inserts a
`gripper0_right_gripper` body between hand and fingers that accounts for the
further 6 mm. The offsets are constant in the link's own axes to five decimal
places over the whole sample, and the orientations agree to 1e-5°. The spread
is recorded beside each magnitude because a small spread is what shows the
offset is constant.

cuRobo's hand and fingertip spheres are authored in franka_description's frames
and must be re-anchored, or inflated to stay enclosing, before they are
trusted. The arm spheres need neither.

**Before pointing a planner at this file:**

* `panda_finger_joint2` is a `<mimic>` of joint 1 (franka_description declares
  two independent prismatics). cuRobo folds a mimic joint out of the
  independent set, so `franka.yml`'s `lock_joints` entry for joint 2 names a
  missing joint; lock joint 1 alone.
* The MJCF mirrors the fingers by sign (`joint_q` holds `+x, −x`), the URDF by
  axis. A parity check must lock the URDF finger where the scene holds it, or
  it reports a phantom 35 mm error.

---

## `embodiments/yam_i2rt/` — i2rt YAM bimanual pair with linear_4310 grippers

**Source:** `i2rt-robotics/i2rt` @ `c40b4125` (read from a mirror of that
repository), files
`i2rt/robot_models/arm/yam/{yam.xml,assets/*.stl}` and
`i2rt/robot_models/gripper/linear_4310/{linear_4310.xml,assets/*.stl}`. MIT
licensed; `LICENSE` copied beside the meshes.

**Files**

| File | What it is |
| --- | --- |
| `yam_i2rt_bimanual.xml` | *Generated.* Two composed YAM+linear_4310 arms, `left_`/`right_` prefixed, bases baked at the reference workcell's mounts. |
| `yam_i2rt_arms.xml` | *Generated.* The same two arms with the baked mount taken out: the arm roots moved from `(0.2525, ±0.31, 0.75)` to the origin, the ±0.31 shoulder separation kept, 0.2525 and 0.75 moved to `EmbodimentSpec.base_offset`. **What `yam_i2rt` imports.** |
| `assets/` | The 9 STLs the MJCF references. Byte-identical copies. |
| `LICENSE` | i2rt's MIT license. |
| `yam_i2rt_arm.urdf` | *Generated.* One arm's chain (`base_link`, `link_1`..`link_6`, `joint1`..`joint6`) transcribed from `yam_i2rt_arms.xml`'s `right_` bodies: each body's `pos`/`quat` as the joint origin, each joint's own axis and range, in chain order; no geometry or inertia. Used by PyRoKi IK, which refuses i2rt's stock `yam.urdf` (not shipped; it lists `joint6` first and puts `joint1` at 63.1 mm where the MJCF has 67, and the check read 268.58 mm / 10.524° tool-to-TCP drift between two configurations on the crate-washing scene, measured 2026-10-03) and accepts this one on both arms; its drift check holds this file to the MJCF. Named by `yam_i2rt`; `yam_i2rt_arena_narrow` has the same chain but names no URDF (see `robots/embodiments/yam_i2rt_arena.py`). |

**What changed on the way in** (all in the generator, none in the meshes):

* The single-arm composition replicates the YAM builder in the original reference sims
  (itself a re-implementation of i2rt's `combine_arm_and_gripper_xml()`):
  fixed `base` wrap, linear_4310 mounted on `link6` with the transform from
  i2rt's `linear_4310.yml`, `joint6` axis overridden to `0 0 -1`.
* Actuator physics the stock MJCF omits, from that builder verbatim:
  reflected armature `N²·I_rotor`, damping, frictionloss and true DM4340/DM4310
  torque limits per joint; position actuators kp=300/kv=30 (arms) and
  kp=600/kv=20 (fingers) -- the fingers' not what the library drives now,
  below.
* `gravcomp="1"` baked onto every body — the sims set `mujoco:gravcomp = 1.0`
  on every robot body in code; baking it lets the importer harvest it.
* **Both fingers actuated and the stock `joint7 == joint8` equality dropped**:
  Newton's MJCF import applies that equality with inverted polarity
  (`joint8 = -joint7`), which translates the jaw pair sideways instead of
  closing it. The original reference sims carry the same fix
  (`skip_equality_constraints=True` plus a second actuator); both fingers are
  always commanded to the same value, so the coupling is preserved by
  construction. (Both fingers are still actuated, but the equality is back
  since: Newton 1.7 imports it the right way round, below.)
* Two prefixed copies merged into one worldbody at bases
  `(0.2525, ∓0.31, 0.75)` (right = active), per-side contact excludes kept.

**Edited in the checked-in `yam_i2rt_arms.xml` and `yam_i2rt_bimanual.xml` on 2026-10-26** (the generator
does not make these: a file regenerated from it drops them): the finger
actuators' `forcerange` and the joints' `actuatorfrcrange` are -50..50, from
+-80, the driver's 50 N a blade (each blade has its own actuator; sources under
`embodiments/yam/` above); and **the `joint7 == joint8` equality is back** (`polycoef="0 1 0 0 0"`, with the stiff `solref="0.004 1"
solimp="0.95 0.99 0.001"`). Newton 1.7 imports it with the right sign
(`eq_data` `[0, 1, 0, 0, 0]`; both fingers commanded to 20 mm read 20.0 and
20.0), and it is what the real hand is, one motor: two capped servos without it
hold a part anywhere between them. Both fingers stay actuated and commanded
alike, as before. Tip and link friction are unchanged: the file authors none,
so MuJoCo's 1.0.

The file's finger gains (kp 600, kv 20, damping 2) are not what the library
drives: `linear_4310.CLAMP` is, as under `embodiments/yam/` above -- kp 150000,
kv 150 and 230 N*s/m a blade (0.22 m/s at the cap). At kp 600 the squeeze grew with the part, 0.9 N a blade on a 1 mm box, 2.3 N on 6 mm,
9.5 N on 30 mm; full open to shut took 0.08 s, now 0.20 s. After
(`MUJOCO_500HZ`): 50 N a blade on 30 and 60 mm boxes, both held to 2 kg. A thin
plate standing on edge is the limit of these colliders at the real force: the
decomposed tip hulls' faces are not parallel, and at 50 N they turn a 6 mm
plate in the pinch (83 degrees; it dropped at 0.47 kg, where the kp 600
servo's 2.3 N held 0.55 kg). The arm's kp 300/kv 30 are the reference sims',
not the hardware's (i2rt's `yam_v1.yml`: kp 80/80/80/10/10/10, kd
5/5/5/1.5/1.5/1.5), and are unchanged.

---

## `embodiments/yam_i2rt_arena/` — the composed i2rt YAM wearing the arena's hand

**Source:** the two directories above. `yam_i2rt_arena_narrow_arms.xml`
(*generated*) is `yam_i2rt/`'s links, meshes, inertias and joint physics with
the arena's hand grafted onto the wrist unchanged: the `{side}_link_6` subtree
of `yam/yam_bimanual_arms.xml` — wrist inertial, palm capsules, D405 body and
its collider, both finger linkages with their `<equality>` coupling — hung
under the composed `{side}_link6` as a fixed child, rotated by `quat 0 -1 1 0`
and not translated. Shoulders at ±0.17 m. `assets/` holds the STLs the file
names; `LICENSE` is i2rt's MIT licence, as for `yam_i2rt/`; the grafted hand
comes from Isaac Lab Arena (Apache-2.0, see `embodiments/yam/` above and
`NOTICE`). Placement rationale and measurements:
`robosimstudio/robots/embodiments/yam_i2rt_arena.py`.

**Edited in the checked-in file on 2026-10-26:** the grafted hand carries the
same edits as `embodiments/yam/` above -- collider friction 1.0, the stiff
finger equality and the actuator's 100 N range -- and is driven by the same
`yam_linear_arena.CLAMP`. Measured
the same way (`MUJOCO_500HZ`): 2, 6 and 30 mm boxes held to the probe's 2 kg,
where the arena hand dropped them at 0.11, 0.29 and 1.31 kg. The arm keeps its
arena gains, as `embodiments/yam/` says.

---

## `embodiments/robotiq_2f85/` — Robotiq 2F-85, from MuJoCo Menagerie

**Source:** `google-deepmind/mujoco_menagerie` @ `affef083`, folder
`robotiq_2f85`. The commit is Newton's own `MENAGERIE_REF`, read off
`newton._src.utils.download_assets`, and fetched by a development script not in
this repository. The model is **BSD 2-Clause**: `LICENSE` is the file
Menagerie ships beside it, "Copyright (c) 2013, ROS-Industrial", and the
vendored `README.md` says the same.

**Files:** Menagerie's bytes, unmodified (`2f85.xml`, `assets/*.stl`,
`LICENSE`), and its `README.md` with the two links to files not vendored here
(`CHANGELOG.md`, `2f85.png`) pointed at Menagerie.

**At build**, `EntityEmbodiment._attach_hands` imports it with `parent_body` set
to the arm's flange, switches its MuJoCo actuators to `JOINT_TARGET` (a live
tendon actuator fights a joint target; Newton's composer test uses the same
setting), and puts `right_driver_joint` / `left_driver_joint` in position mode.
Nothing on disk changes.

**Measured** on a tool-sorting scene, welded to a UR5e, closing under physics:
pad centres 0.09793 m apart at full open and 0.01252 m at full closed, i.e.
**0.0854 m of travel against the published 0.085 m stroke**. Closing axis
tool-local ±x, a right angle from the `GripperSpec` default. `housing_half`
measures 0.0424 here against the Isaac stage export's 0.0425.

---

## `embodiments/suction/` — a single-cup vacuum hand, authored here

**Source:** none; `suction_cup.xml` was written for this repository and is
Apache-2.0 like the code. Four cylinders: an 18 mm flange and a 6 mm stem on
`suction_mount`, an 11 mm bellows and a 15 mm face on `suction_pad`, whose
origin is the face, 60 mm from the mount along +z. Masses are stated per geom,
70 g in all.

**At build**, `EntityEmbodiment._attach_hands` imports it with `parent_body`
set to the arm's wrist at the arm's `ArmSpec.gripper_mount`, and
`after_scene_built` adds one disabled MuJoCo weld row from `suction_pad` to
each part with a flat face; `robosimstudio.core.suction` switches them.
Nothing on disk changes.
