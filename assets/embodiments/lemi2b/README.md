# Sunday Robotics Lemi 2B (`lemi2b`)

Standalone MJCF and URDF asset bundle for the **Sunday Robotics Lemi 2B** bimanual mobile manipulator.

## Provenance

Staged from `sunday-robotics/lemi`:

* `hardware/model/lemi2/lemi2b/robot.xml` and `collisions.xml`
* `hardware/model/lemi2/lemi2b/robot.urdf`
* `hardware/model/lemi2/lemi2b/assets/*.stl` (27 full-resolution visual meshes, staged under `assets/`)
* `sunday/config/common/robot_config.textproto` (`LEMI_2B` nominal joint posture and joint configuration)
* `sunday/config/common/sim/controller_config.textproto` (`STIFF` actuator PD gains)

## Structure

* **30 actuated degrees of freedom**:
  * Planar mobile base (`basex`, `basey`, `baseth`)
  * Vertical lift (`lift`, range `[0.66, 1.259]` m, nominal `0.85` m) and torso pitch (`TP`)
  * Neck yaw/pitch (`NY`, `NP`) and head camera pitch (`head_camera_dof`)
  * Right 7-DoF arm (`RJ1`..`RJ7`) + 4-DoF 2-finger hand (`RH_PI`, `RH_DI`, `RH_P3F`, `RH_D3F`)
  * Left 7-DoF arm (`LJ1`..`LJ7`) + 4-DoF 2-finger hand (`LH_PI`, `LH_DI`, `LH_P3F`, `LH_D3F`)
* **Frames and sites**:
  * Wrist TCP sites: `right_wrist_axis_center`, `left_wrist_axis_center`
  * Thumb reference sites: `right_thumb`, `left_thumb`
  * Camera sites: `head_camera`, `right_teleop_camera`, `left_teleop_camera`
  * LiDAR sites: `base_jt128`, `torso_jt128`
