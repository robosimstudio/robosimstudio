# SPDX-License-Identifier: Apache-2.0
"""Paper: fold a sheet

Bring each hand in along the edge of the A4 sheet that runs past the box, from its own corner, with **Grip amount** 0.79, and press **Close**: the fingers stall on the sheet at the grip's 10 N cap. Carry the edge up and over, each hand turning with the flap to 88 degrees and no further, and press **Open** 100 mm over the far half: the flap falls onto it. Then, at **Grip amount** 0.6, run the side of each hand's lower finger along its half of the fold, the sheet pushing it up its slide against the same 10 N cap: the sheet stays folded in half.

On the YAM's pads, at friction 1 (paper's 0.3 mixes to 0.55 on them), the arms, which collide, kept 79.5 mm
apart and 12.0 and 9.0 mm on their own sides, each hand turned 88.0 degrees holding the edge, and no finger pushed
over 15 N at any step (8.7 N holding, 4.8 after **Open**, 12.4 pressing). The edges ended 0.3 to 2.3 mm apart and the
fold kept 2.89 rad, unchanged 3 s after the hands (``tests/test_fold_paper_robot``).
"""

import numpy as np

import robosimstudio as rss
from robosimstudio.assetlib.semantics.snapshot import Snapshot
from robosimstudio.assetlib.vocabulary.measures import cloth_fold
from robosimstudio.robots.registry import get_embodiment
from robosimstudio_materials.paper import PAPER, crease_angle


def main():
    size, cells = (0.297, 0.21), (28, 20)  # A4, its long side away from the robot; a 10.6 x 10.5 mm cell
    box = (0.295, 0.28, 0.08)  # what the sheet lies on [m]: flat on the bench, a pad has nothing to get under
    corner = (-0.05, -0.105, 0.085)  # the sheet's near corner (bench frame), across the middle, on the box
    overhang = 0.035  # how far the near edge runs past the box: the pads close on it from either corner
    # Copy paper at 20 VBD iterations: a 5.1 cm bending length held elastic on the drape bench (PAPER's 30: 9.7),
    # and a hinge bent 0.2 rad past its rest keeps the rest, so a fold pressed flat keeps up to 2.94 rad.
    sheet = rss.replace(PAPER, edge_ke=6.0, yield_angle=0.2)
    # Each finger feels the sheet's push along its slide; at 20 iterations, not 5, the sheet rests at 2.6 N, not 11.
    coupling = rss.coupler("one_way", contact_forces=True, slide_feedback=True)
    physics = rss.replace(rss.GENERIC, coupling=coupling, collision=rss.CollisionSpec(full_surface_contact=True))
    vbd = {"iterations": 20, "friction_epsilon": 1.0e-3, "rigid_body_contact_buffer_size": 256}
    layers = {"particle_self_contact_radius": 1.0e-3, "particle_self_contact_margin": 1.0e-3}
    physics = physics.with_backend("*", rss.backend("vbd", **vbd, **layers))
    # The arms collide (the YAM's own). A hand's motor pushes its driven finger 10 N at most; the other follows.
    yam = get_embodiment("yam_bimanual")
    hands = rss.ActuationSpec(target_ke=1.0e5, target_kd=20.0, effort_limit=10.0, joint_damping=100.0)  # 200 a pair

    scene = rss.Scene(cell="bench_0728", physics=physics, name="paper")
    scene.add(rss.robot(yam, gripper=hands))
    at = (corner[0] + overhang + box[0] / 2, 0.0, box[2] / 2)  # a rubber-topped box: paper's 0.3 mixes to 0.95
    scene.add(rss.slab(name="box", half=tuple(v / 2 for v in box), at=at, color=(0.62, 0.47, 0.33), friction=3.0))
    scene.add(rss.paper(name="paper", size=size, cells=cells, at=corner, material=sheet))
    sim = scene.build(control_mode="joint_position")
    # The fingers take the sheet's contact damping, not MuJoCo's 500 N s/m, which VBD averages in: a pad read 19 N.
    fingers = [b for arm in sim.scene.embodiment.arms.values() for side in arm.finger_bodies for b in side]
    kd = sim.model.shape_material_kd
    kd.assign(np.where(np.isin(sim.model.shape_body.numpy(), fingers), sheet.contact_kd, kd.numpy()))
    folded = sim["paper"].entity
    line = {"crease": {"axis": 0, "at": corner[0] + size[0] / 2}}  # the fold line, half way along the sheet

    def page(studio):
        with studio.gui.add_folder("Crease", order=-1.0):  # first on the panel: the readouts are the evidence
            angle = studio.gui.add_number("crease_angle (rad)", 0.0, step=0.01, disabled=True)
            extent = studio.gui.add_number("fold_extent", 1.0, step=0.01, disabled=True)

        def each_step() -> None:
            folded.update_probe(sim.state, sim.step_count)
            snapshot = Snapshot(sim.make_context())
            angle.value = round(crease_angle(snapshot, "paper", line)["angle"], 2)
            extent.value = round(cloth_fold(snapshot.ctx, "paper", axis=0)[0], 2)

        studio.each_step(each_step)

    rss.view(sim, page=page, panels=("inspector", "overlay", "arm", "object", "cloth", "tune"))


if __name__ == "__main__":
    main()
