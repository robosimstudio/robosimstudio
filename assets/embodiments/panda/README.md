# Franka Panda — robosuite's model, as the VAB tool-sort export carries it

What the `panda`, `panda_robotiq` and `panda_robotiq_roll` embodiments import
(`robosimstudio/robots/embodiments/panda.py`). The longer record — including the
kinematics URDF's parity measurements — is the `embodiments/panda/` section of
`assets/PROVENANCE.md`; this note is the part that has to travel with the files.

## Where it came from

* **robosuite 1.4.0** (`ARISE-Initiative/robosuite`, MIT): its Panda arm
  (`robosuite/models/assets/robots/panda/`) and its Franka hand
  (`robosuite/models/assets/grippers/`, `panda_gripper`). `LICENSE` beside this
  file is robosuite's licence text, fetched on 2026-10-04 from
  `https://raw.githubusercontent.com/ARISE-Initiative/robosuite/master/LICENSE`;
  the `v1.4.0` tag's file is byte-identical to it.
* **Exported by the VAB pipeline.** VAB-industrial's `ToolSortPickPlace`
  (robosuite 1.4.0, MuJoCo 3.3.3) compiles that robot into one merged MJCF with
  its arena, and the dump of that scene is what these files were cut from. Not
  re-authored: the bodies, joints, inertias, limits and names are robosuite's,
  `robot0_` and `gripper0_` prefixes included.
* **The mesh geometry originates from Franka's `franka_ros` robot description**
  (`franka_description`, Apache-2.0), which is what robosuite's Panda meshes
  were made from.

## Files

| File | What it is |
| --- | --- |
| `panda_standalone.xml` | The arm with its Franka hand: the export's `robot0_base` subtree at its own origin, with the nine actuators and two sensors that drive it. What `panda` imports. |
| `panda_nohand_standalone.xml` | The same arm ending at `robot0_right_hand`, so a hand can be welded on at build time. What `panda_robotiq` and `panda_robotiq_roll` import. |
| `assets/` | The 61 meshes those two files name, and nothing else: eleven STL collision meshes and fifty OBJ visual meshes, 39.9 MB. |
| `LICENSE` | robosuite's MIT licence, which the two MJCFs and the meshes arrive under. |
| `panda_kinematics.urdf` | **A different source**: `example-robot-data`'s `panda_description` (itself Franka's `franka_description`, Apache-2.0), stripped to joints, links, limits and inertials. Never simulated, only solved against. See `assets/PROVENANCE.md`. |

Both MJCFs say `meshdir="."` and name every mesh as `assets/<file>`, so they
resolve against this directory as they stand.

## What was changed

Against robosuite's own files, and all of it before the copy into this
directory:

1. **`<position>` actuators instead of robosuite's `<motor>`s.** Newton reads a
   joint target only for a position-driven DOF, so a torque-actuated export
   takes zero torque from every target. The force limits in the files are
   ±80 N·m on the first five joints, ±12 on the last two and ±20 N on each
   finger.
2. **Cut out of the arena.** The merged scene holds bins, a workbench and tools
   besides the robot. `panda_standalone.xml` keeps the `robot0_base` subtree and
   drops the `mount0_*` pedestal chain, which is furniture: it holds the arm at
   one height over one bench. That height is `EmbodimentSpec.base_offset` now.
3. **The hand removed, for the second file.** `panda_nohand_standalone.xml` is
   the standalone with the `gripper0_right_gripper` subtree and every mesh,
   actuator and sensor naming it dropped: the seven arm actuators remain, and 57
   of the 61 meshes are named.
4. **Mesh paths flattened** by the dump into one `assets/` directory, where
   robosuite keeps one directory per link.

The meshes themselves are robosuite's bytes. Checked against the `v1.4.0` tag on
2026-10-04, all 61: the eleven STLs are byte-identical, and the fifty OBJs are
identical apart from line endings (CRLF here, LF upstream). Their `mtllib` lines
name `.mtl` files that neither the export nor this directory carries; the
colours come from the MJCFs' own `<material>` elements.

## The copy the benchmark's collection keeps

The same two MJCFs and the same meshes also sit in the benchmark's collection,
beside the arena exports they were cut from, and that collection's own scenes
read them there. The files here are byte copies of those. The collection's
tool-sort split script regenerates `panda_standalone.xml` there; after running
it, copy the result here — `tests/test_panda_vendored.py` fails while the two
differ.
