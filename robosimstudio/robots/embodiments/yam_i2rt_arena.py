# SPDX-License-Identifier: Apache-2.0
"""The composed i2rt YAM wearing the arena's hand: the suites that were measured
on the arena, on the one arm.

The tool-hanging cell, written by the development-tree ``graft_arena_hand.py``
(not shipped; see ``assets/PROVENANCE.md``): shoulders at +/-0.17 m and the
raised, forward mount that cell needs for lift-and-rotate clearance. It replaced
an earlier tool-hanging embodiment, since retired, cell for cell, measured in
the benchmark's YAM campaign record of 2026-09-06.

The generator takes ``--shoulder-y``, so the play table's +/-0.31 m pair is one
flag away. It is not registered because nothing asks for it: the cloth-folding
suite is the only one that would, and it stays on ``yam_bimanual`` by the 2026-09-06
ruling. Re-opening that would have to answer the arm-clearance measurement taken
during the arena migration.

This is ``yam_i2rt``'s links, meshes, inertias and joint physics -- the arm
composed from i2rt's own part files with the YAM motor model -- and
the ``yam_linear_arena`` hand grafted onto the wrist unchanged, palm capsules
and camera included, in the ``{side}_link_6`` frame every one of that hand's
constants is measured in. See the generator's docstring for where the graft
sits and why; the measurement it rests on is an arm-parity comparison made
during development.

**What moved and by how much.** The arena export and this arm are not one
kinematic chain: axis directions agree to 0.0000 deg, axis lines differ by
about 1 mm through the arm and 3.6-5.3 mm at the wrist. ``base_offset`` here is
the arena's less :data:`~tools.graft_arena_hand.SHOULDER_HEIGHT_DELTA_M`, which
puts the shoulder at the height the suites had it; the wrist residual is the
arm, and is what the suites' own graphs were re-measured against before these
became their native pairing.

**Gains** are the arena's actuator classes, stated in the asset -- a cell fact,
as the USB-C insertion suite says of its own 1600 -- so ``arm_actuation`` and
``gripper_actuation`` stay ``None``: the MJCF's, only checked.
"""

from __future__ import annotations

from robosimstudio.specs import ArmSpec, AssetSpec, EmbodimentSpec

from ..registry import register_embodiment
from .yam_look import YAM_BLACK, split_yam_white_tubes

__all__ = ["ARM_HOME", "SHOULDER_HEIGHT_DELTA_M", "YAM_I2RT_ARENA_NARROW"]

ARM_HOME = (0.0, 1.047, 1.047, 0.0, 0.0, 0.0)
"""Per-arm home posture [rad] -- the arena's, unchanged, because the parity
measurement shows the two exports share joint zeros and signs."""

SHOULDER_HEIGHT_DELTA_M = 0.007517
"""How much higher this arm's shoulder sits above its base origin than the
arena's did (measured by the graft generator). Subtracted from the arena's mounts."""


def _arm(side: str) -> ArmSpec:
    return ArmSpec(
        name=side,
        # Named, not scanned by prefix: with fixed bodies kept as bodies, the
        # grafted wrist, its linkages and the camera each carry a fixed joint
        # under the same prefix, and a prefix scan sweeps them into the arm's
        # joint list -- measured as a DOF mask that partially covered an
        # object's FREE joint at build. yam_i2rt names its six for the same
        # reason.
        joint_names=tuple(f"{side}_joint{i}" for i in range(1, 7)),
        # The arena's wrist frame, grafted verbatim: every hand constant is
        # measured against it, and the wrist camera rig is stated against it.
        ee_body_name=f"{side}_link_6",
        gripper="yam_linear_arena",
        home_q=ARM_HOME,
    )


def _spec(name: str, base_offset: tuple[float, ...]) -> EmbodimentSpec:
    return EmbodimentSpec(
        name=name,
        # No URDF, on purpose: this host is served by damped least squares, as
        # the retired tool-hanging embodiment (which also named none) was. The vendor yam.urdf it
        # used to name was refused by PyRoKi on both arms anyway (231.53 mm /
        # 15.994 deg of drift on the level-1 pliers scene), so PyRoKi here
        # already fell back to DLS. yam_i2rt's transcribed
        # embodiments/yam_i2rt/yam_i2rt_arm.urdf matches this chain and PyRoKi
        # accepts it (0.001 mm at the TCP over 20 configurations), but serving
        # this host with PyRoKi took the tool-hanging vision graph from 15/22 to
        # 1/22 (A/B, 2026-10-04). Until that is understood, DLS is the choice.
        urdf="",
        asset=AssetSpec(path=f"embodiments/yam_i2rt_arena/{name}_arms.xml"),
        base_offset=base_offset,
        # The composed arm's conventions (see yam_i2rt.py): fixed bodies stay
        # bodies, so the grafted ``{side}_link_6`` keeps its label; one stated
        # colour with the white tube shells cut back in; the arm's dual-purpose
        # mesh geoms shown. No decomposition: the arena hand's pads are
        # primitives and collide as authored.
        collapse_fixed_joints=False,
        color=YAM_BLACK,
        appearance=split_yam_white_tubes,
        show_collision_shapes=True,
        enable_self_collisions=False,
        arms=(_arm("left"), _arm("right")),
        arm_actuation=None,
        gripper_actuation=None,
        ik_iterations=8,
    )


YAM_I2RT_ARENA_NARROW = register_embodiment(
    _spec(
        "yam_i2rt_arena_narrow",
        # The tool-hanging cell's mount -- 50 mm forward and 60 mm higher than
        # the arena's -- less the same convention: same shoulder height as
        # the retired tool-hanging embodiment.
        (-0.2975, 0.0, 0.082 - SHOULDER_HEIGHT_DELTA_M, 0.0, 0.0, 0.0, 1.0),
    )
)
