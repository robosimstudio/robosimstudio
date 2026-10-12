# Vendored assets

Every directory here is a copy of geometry authored elsewhere, vendored so a
task suite is self-contained and reproducible. This file records where each came
from and what was changed on the way in. **Nothing here is re-authored** — a
benchmark whose geometry is a re-authoring of the original is not the same
benchmark.

**What lives here, and what does not.** The rule is *who owns it*: geometry a
single declaration owns lives beside that declaration, in the corpus. What is
left is here, under `embodiments/`.

* **`embodiments/`** — robot descriptions: the Panda, the YAM pair, the i2rt
  composition of the same arm and the arena-handed variant of it, the Robotiq
  2F-85. Embodiments are not assets, which is a standing rule in this
  stack rather than a filing accident, so they do not live in a corpus of assets;
  and the declaration that owns one is an `EmbodimentSpec` in the *library*, so
  the geometry cannot sit beside it either.

**An export a collection's documents stage lives with that collection**, with
its own `PROVENANCE.md`; a document states the export's path in
`(:staging (:export …))` and `robosimstudio.paths.resolve_path` searches each
collection's `assets/` after this one, so the record stays as written. The
`panda` embodiment used to import its robot out of the benchmark's tool-sort
export by path — a collection-owned file the library read and did not ship. The arm is vendored
under `embodiments/panda/` now, as a byte copy, so the library's own robot
builds where the collection is not installed; the collection keeps its
originals, which its merged scenes still read. See that section.

Geometry that a declaration *does* own is in **`<collection>/tasks/<name>/assets/`**,
beside the `AssetRecord` that declares it, and carries its own `PROVENANCE.md`
there. The point is that such a folder is the whole asset and can be handed to
somebody: geometry, declaration and attribution together, none of it in a file
that did not come along. A layout test holds the line from this side
too — the one directory here is `embodiments/`, so a new top-level folder has
to be argued for rather than added. A surgical
needle once arrived as `assets/props/surgical_needle/` and moved into the
suite that declared it for precisely that reason: one declaration owns it.

Only this tree is searched by `robosimstudio.paths.resolve_path` — the checkout's
`assets/` and then the copy a wheel force-includes at
`robosimstudio/assets/`. It used to search a content tree first, and no longer
does, because task-owned geometry is resolved beside its declaration instead:
`register_asset` rewrites a record's `files` to absolute paths at registration.
`ROBOSIMSTUDIO_ASSETS` overrides the search outright, for a checkout that keeps
large binaries out of the package.

---

## `embodiments/yam/` — I2RT YAM bimanual arena

**Source:** the arena's `sim_models/` directory, in a read-only reference
checkout of the Isaac Lab Arena YAM stage.

**Files**

| File | What it is |
| --- | --- |
| `yam_bimanual_arms.xml` | The arena's two YAM arms and nothing else, authored at their own base origin: the scene as authored (two YAM arms, a gate, cameras, a table, seven free-jointed props) with every free-jointed body and the home keyframe removed, then the cell's scenery deleted and the arm roots moved to the origin. Written by `scripts/strip_yam_scenery.py` (development tree), which checks its result against the source. **This is what the embodiment imports.** |
| `i2rt_yam/` | Meshes, and the per-arm URDF. Unmodified. Carries its own `LICENSE`. |

**Why stripped:** the arena declares six bottles and a bin container as loose
props. A task supplies its own objects, so those are removed — and the home
keyframe with them, because a keyframe indexes the `qpos` of the bodies just
removed and an import that keeps it reads the wrong coordinates. Free bodies
removed: `bottle_1` … `bottle_6`, `bin_container`.

The strip is a pure deletion of `<body>` elements carrying a `<freejoint>` plus
the `<keyframe>` block; `meshdir` still resolves because the file sits beside
`i2rt_yam/`. `robosimstudio.scene.build.imports.strip_free_bodies` performs the
same operation at runtime, and the result is checked in so the vendored
asset is what actually gets imported rather than a temp file derived from it.

**Why the scenery is gone too:** an embodiment asset should contain the robot
and nothing else. The arena file carries a whole cell — a light, a floor plane,
a gantry mesh with its collision posts and a top bar holding a camera, three
wall panels, a visual table slab, and an invisible `table_plane` collider at
z = 0.728, which dropped into a workcell whose table is at 0.750 adds a second,
invisible surface 22 mm below the real one. The arm roots were baked at
`(0.2525, ±0.31, 0.75)`; the ±0.31 shoulder separation stays, because that *is*
the robot, and the 0.2525 and the 0.75 become `EmbodimentSpec.base_offset`
against a scene's work surface.

**Arena constants**, restated in `embodiments/yam_bimanual.py` rather than
imported from the reference: table top at `z = 0.728 m`, table centred at
`(0.6, 0.0)` with half-extents `(0.30, 0.40)`, per-arm home
`[0, 1.047, 1.047, 0, 0, 0]`, grasp site `0.1347 m` along the wrist's `+z` with
a −90° twist about `z`.

---

## `embodiments/panda/` — Franka Panda: robosuite's model, and a kinematics URDF

Two things from two sources, in one directory because they describe one arm.

### The physics model — robosuite 1.4.0's Panda, as the VAB tool-sort export carries it

**Source:** `ARISE-Initiative/robosuite` 1.4.0 (MIT), compiled into
VAB-industrial's `ToolSortPickPlace` scene and dumped by the benchmark
collection's scene-dump script — the export that collection's own
`PROVENANCE.md` records. The mesh geometry originates from Franka's `franka_ros` robot
description (`franka_description`, Apache-2.0). `LICENSE` is robosuite's licence
text, fetched from
`https://raw.githubusercontent.com/ARISE-Initiative/robosuite/master/LICENSE` on
2026-10-04 and byte-identical to the `v1.4.0` tag's; `README.md` says all of
this again, for whoever is handed the directory without this file.

**Files**

| File | What it is |
| --- | --- |
| `panda_standalone.xml` | The export's `robot0_base` subtree at its own origin, the `mount0_*` pedestal dropped, with its nine `<position>` actuators and two sensors. **What `panda` imports.** Written in the collection by its tool-sort split script. |
| `panda_nohand_standalone.xml` | The same with the `gripper0_right_gripper` subtree and everything naming it removed, ending at `robot0_right_hand` with the seven arm actuators. **What `panda_robotiq` and `panda_robotiq_roll` import**, to weld a 2F-85 onto. |
| `assets/` | The 61 meshes those two files name — 11 STL, 50 OBJ, 39.9 MB — and no others. |
| `LICENSE`, `README.md` | robosuite's MIT licence, and the note that travels with the files. |

**Copied, not moved.** These are byte copies of the two MJCFs the benchmark's
collection keeps beside its tool-sort export, and of the 61 files under that
export's `assets/` which the two name. The
originals stay: the collection's merged scene MJCFs name the same meshes at the
same relative paths, and git stores identical bytes once. Both MJCFs say
`meshdir="."` and name each mesh as `assets/<file>`, so the copies needed no
edit. `tests/test_panda_vendored.py` holds the copies byte-identical to the
originals while both are on disk, and holds `assets/` to exactly the meshes
named.

**Why it is here.** `panda` is an `EmbodimentSpec` the *library* registers, and
until this copy its asset path resolved into a collection — so the robot every
first example reaches for built only where the benchmark's collection was installed. An embodiment
the library declares needs geometry the library ships.

**What changed on the way in** — against robosuite's own files, and all of it
upstream of this directory: `<position>` actuators for robosuite's `<motor>`s
(the export's entry says why), the cut out of the arena, the hand removed for
the second file, and mesh paths flattened into one `assets/`. The meshes are
robosuite's bytes: checked against the `v1.4.0` tag on 2026-10-04, the eleven
STLs are byte-identical and the fifty OBJs differ only in line endings (CRLF
here, LF upstream).

**Measured across the copy** (2026-10-04): a `panda` and a `panda_robotiq`
alone on `bench_075`, and a level-1 and a level-2 tool-sorting scene, built
from the collection's files and then from these. Every count agrees — 16 bodies,
73 shapes, 16 joints and 19.02 kg for the `panda` scene; 24, 88, 26 and 19.47 kg
for `panda_robotiq` — and so does every model array compared, bit for bit: 124
of them across the four scenes (masses, inertias, shape transforms, joint
frames, limits, gains), with the hash of every mesh's vertices. Over the first
200 steps the joint positions differ by at most 6.7e-13 for `panda` and 6.3e-7
for `panda_robotiq` (radians; metres for the fingers), which is no more than two
runs of *one* build differ by: 1.5e-13 and 1.2e-6, the 2F-85's linkage being in
contact.

### The kinematics URDF — for full-orientation IK

**Source:** `example-robot-data` @ `d0d9098d752014aec3725b07766962acf06c5418`
(`robots/panda_description/urdf/panda.urdf`, via the machine's
`robot_descriptions` cache).

**Files**

| File | What it is |
| --- | --- |
| `panda_kinematics.urdf` | The URDF with every `<visual>` and `<collision>` element removed — joints, links, limits and inertials only. |

**Why stripped:** the PyRoKi IK backend
(`robosimstudio.robot.ik.pyroki`) consumes kinematics alone, and a
kinematics-only file has no mesh paths to break. The scene's *physics* Panda
is the MJCF above, out of the tool-sort export — this file is never simulated,
only solved against.

**Why it can be trusted:** the backend measures `T_hand→tcp =
fk_urdf(q)⁻¹ · T_base⁻¹ · fk_sim(q)` at two configurations at construction and
refuses to run if it drifts (>1 mm / 0.1°). Measured on the tool-sort scene the
drift is ≤0.001 mm and 0.0000°, i.e. this URDF and the imported MJCF describe
the same arm.

**Per-link parity, for the collision model.** The claim above is about one
frame, measured at two configurations — enough for an IK backend, which only
needs the tool point, and not enough for a *planner*, which is trusted about the
elbow. Re-measured link by link against the tool-sort MJCF at 64 sampled
configurations (seed 0, home plus both limit corners plus 61 uniform; measured
with a per-link frame comparator that has since been removed, the numbers being
the part worth keeping):

| Frame | vs MJCF | Verdict |
| --- | --- | --- |
| `panda_link0` … `panda_link7` | ≤ **0.0016 mm** / ≤ 0.00002° | the same arm, to float32 |
| `panda_hand` | 0.4994 mm along the link's own −z, spread 0.0025 mm | a constant |
| `panda_leftfinger`, `panda_rightfinger` | 6.4989 mm along −z, spread 0.0031 mm | a constant |

The hand and finger rows are **frame-origin conventions, not geometry**:
robosuite names the hand's origin half a millimetre from where
franka_description does, and inserts a `gripper0_right_gripper` body between
hand and fingers that accounts for the further 6 mm. The offsets are constant in
the link's own axes to five decimal places over the whole sample, and the
orientations agree to 1e-5° — which is what distinguishes a naming difference
from a broken model, and is why the spread is recorded beside the magnitude
rather than the magnitude alone.

Consequence for a collision planner: cuRobo's hand and fingertip spheres are
authored in franka_description's frames and must be re-anchored, or inflated to
stay enclosing, before they are trusted. The arm spheres need neither.

**Two more differences worth knowing before pointing a planner at this file:**

* It declares `panda_finger_joint2` as a `<mimic>` of joint 1, where
  franka_description declares two independent prismatics. cuRobo folds a mimic
  joint out of the independent set, so `franka.yml`'s `lock_joints` entry for
  joint 2 names a joint that is no longer there; lock joint 1 alone.
* The MJCF mirrors the fingers by *sign* (`joint_q` holds `+x, −x`) where the
  URDF mirrors them by *axis*. Same motion, and a parity check that locks the
  URDF's finger somewhere the scene does not hold it reads 35 mm of
  disagreement that is not there.

---

## `embodiments/yam_i2rt/` — i2rt YAM bimanual pair with linear_4310 grippers

**Source:** `i2rt-robotics/i2rt` @ `c40b4125` (the `newton_tasks` submodule
pin; read from a mirror of that repository), files
`i2rt/robot_models/arm/yam/{yam.xml,assets/*.stl}` and
`i2rt/robot_models/gripper/linear_4310/{linear_4310.xml,assets/*.stl}`. MIT
licensed; `LICENSE` copied beside the meshes.

**Files**

| File | What it is |
| --- | --- |
| `yam_i2rt_bimanual.xml` | Two composed YAM+linear_4310 arms, `left_`/`right_` prefixed, bases baked at the newton_tasks workcell mounts. **Generated by `scripts/compose_yam_i2rt.py` (development tree) — edit that, not this.** |
| `yam_i2rt_arms.xml` | The same two arms with the baked mount taken out: the arm roots moved from `(0.2525, ±0.31, 0.75)` to the origin, the ±0.31 shoulder separation kept, the 0.2525 and the 0.75 becoming `EmbodimentSpec.base_offset`. Written by `scripts/strip_yam_scenery.py` (development tree). **What `yam_i2rt` imports.** |
| `assets/` | The 9 STLs the MJCF references. Byte-identical copies. |
| `LICENSE` | i2rt's MIT license. |
| `yam_i2rt_arm.urdf` | One arm's chain (`base_link`, `link_1`..`link_6`) transcribed from `yam_i2rt_arms.xml`'s `right_` bodies: each body's `pos`/`quat` as the joint origin, each joint's own axis and range, joints `joint1`..`joint6` in chain order. For the PyRoKi solver family, which refuses the stock `embodiments/yam/i2rt_yam/urdf/yam.urdf` on this arm -- its check reads the tool-link-to-TCP transform drifting 268.58 mm / 10.524 deg between two configurations on the crate-washing scene, measured 2026-10-03 (that file lists `joint6` first, and puts `joint1` at 63.1 mm where the MJCF has 67) -- and accepts this one on both arms. No geometry or inertia, by design: it is a kinematic chain. Named by `yam_i2rt`; `yam_i2rt_arena_narrow` has the same chain but deliberately names no URDF (see `robots/embodiments/yam_i2rt_arena.py`). **Generated** by the crate-washing task's embodiment script (development tree), and drift-tested there. |

**What changed on the way in** (all in the generator, none in the meshes):

* The single-arm composition replicates `newton_tasks/sim/yam_asset.py`
  (itself a re-implementation of i2rt's `combine_arm_and_gripper_xml()`):
  fixed `base` wrap, linear_4310 mounted on `link6` with the transform from
  i2rt's `linear_4310.yml`, `joint6` axis overridden to `0 0 -1`.
* Actuator physics the stock MJCF omits, from `yam_asset.py` verbatim:
  reflected armature `N²·I_rotor`, damping, frictionloss and true DM4340/DM4310
  torque limits per joint; position actuators kp=300/kv=30 (arms) and
  kp=600/kv=20 (fingers).
* `gravcomp="1"` baked onto every body — the sims set `mujoco:gravcomp = 1.0`
  on every robot body in code; baking it lets the importer harvest it.
* **Both fingers actuated and the stock `joint7 == joint8` equality dropped**:
  Newton's MJCF import applies that equality with inverted polarity
  (`joint8 = -joint7`), which translates the jaw pair sideways instead of
  closing it. The newton_tasks kamino sims carry the same fix
  (`skip_equality_constraints=True` plus a second actuator); both fingers are
  always commanded to the same value, so the coupling is preserved by
  construction.
* Two prefixed copies merged into one worldbody at bases
  `(0.2525, ∓0.31, 0.75)` (right = active), per-side contact excludes kept.

---

## `embodiments/yam_i2rt_arena/` — the composed i2rt YAM wearing the arena's hand

**Source:** the two directories above. `yam_i2rt_arena_narrow_arms.xml` is
`yam_i2rt/`'s links, meshes, inertias and joint physics with the arena's hand
grafted onto the wrist unchanged: the `{side}_link_6` subtree of
`yam/yam_bimanual_arms.xml` — wrist inertial, palm capsules, D405 body and its
collider, both finger linkages with their `<equality>` coupling — hung under the
composed `{side}_link6` as a fixed child, rotated by `quat 0 -1 1 0` and not
translated. Shoulders at ±0.17 m. `assets/` holds the STLs the file names;
`LICENSE` is i2rt's MIT licence, as for `yam_i2rt/`. **Generated by
`scripts/graft_arena_hand.py` (development tree) — edit that, not this.**
`robosimstudio/robots/embodiments/yam_i2rt_arena.py` records why the hand sits
where it does and what was measured.

---

## `embodiments/robotiq_2f85/` — Robotiq 2F-85, from MuJoCo Menagerie

**Source:** `google-deepmind/mujoco_menagerie` @ `affef083`, folder
`robotiq_2f85`. The commit is Newton's own `MENAGERIE_REF`, read off
`newton._src.utils.download_assets`; fetched by `scripts/fetch_menagerie.py`
(development tree). The model is **BSD 2-Clause**: `LICENSE` is the file
Menagerie ships beside it, "Copyright (c) 2013, ROS-Industrial", and the
vendored `README.md` says the same.

**Files:** Menagerie's own bytes, untouched — `2f85.xml`,
`assets/*.stl`, `LICENSE`, `README.md`. Nothing is generated here: this is the
first hand in the repo that is *welded on at build time* rather than composed
into an arm asset, so the file the weld imports is the file that was vendored.

**What the toolkit does to it at build**, in `EntityEmbodiment._attach_hands`
rather than on disk: imports it with `parent_body` set to the arm's flange,
switches its MuJoCo actuators to `JOINT_TARGET` (leaving the tendon actuator live
alongside a joint target makes the two fight — the same setting, and the same
reason, as Newton's composer test), and puts `right_driver_joint` /
`left_driver_joint` in position mode.

**Measured** on a tool-sorting scene, welded to a UR5e, closing under physics: pad centres
0.09793 m apart at full open and 0.01252 m at full closed, i.e. **0.0854 m of
travel against the published 0.085 m stroke**. Closing axis tool-local ±x, which
is a right angle from the `GripperSpec` default. `housing_half` measures 0.0424
here against the Isaac stage export's 0.0425 — two independent exports of one
piece of hardware, agreeing to 0.1 mm.
