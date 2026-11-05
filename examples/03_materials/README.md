# Materials

Each script here is one kind from
[`robosimstudio_materials`](../../robosimstudio_materials/), with the robot doing
its task: the hand takes hold, works the material, and the page reads how far it
got. Cloth and the sponge are introduced in [`01_cloth`](../01_cloth/README.md)
and [`02_soft`](../02_soft/README.md).

- **Rope**, `00` to `02`: [`rod/`](../../robosimstudio_materials/rod/) is `rss.rod`, a
  capsule chain on Newton, or a Cosserat rod when SuperDex steps the `rod` kind.
- **Knot**, `03`: [`rod/knot.py`](../../robosimstudio_materials/rod/knot.py) lays an
  overhand knot (`overhand`) and reads `crossings`, `knot_length` and
  `self_clearance`; `RodSpec.self_contact` is what keeps it a knot.
- **Band**, `04`: [`rod/band.py`](../../robosimstudio_materials/rod/band.py) closes a
  rod into a loop on SuperDex. `RUBBER` and `O_RING` ship.
- **Paper**, `05`: [`paper.py`](../../robosimstudio_materials/paper.py) is `rss.paper`,
  cloth whose folds stay. `rss.papers.PAPER` and `rss.papers.CARDBOARD` ship.
- **Bag**, `06`: [`bag.py`](../../robosimstudio_materials/bag.py) is `rss.bag`, two
  layers of cloth sewn into a pocket that SuperDex keeps apart. `rss.bags.CANVAS`
  ships.
- **Brush**, `07`: [`brush.py`](../../robosimstudio_materials/brush.py) is `rss.brush`,
  bristles of capsule chains under a sole; `nylon` and `soft` ship. The hand holds a
  named handle on the arm's own solver (`rss.brush(handle=...)` under
  `rss.BRUSH_60HZ`), and the module's docstring says why and what that costs.
- **Tape**, `08` and `09`: [`tape.py`](../../robosimstudio_materials/tape.py) is
  `rss.tape`, a sheet stuck to a free body on the tape's solver (`rss.TAPE_60HZ`) by
  bonds that let go. `rss.tapes.PAPER_LABEL` is built stuck; `LABEL`,
  `MASKING_TAPE` and `DUCT_TAPE` stick where they are pressed.
- **Cutting**, `10`: [`cutting/`](../../robosimstudio_materials/cutting/__init__.py) is
  `rss.cuttable` and `rss.knife`, on a vendored walk-on-spheres cage cutter.
  `rss.cuttables.TOFU`, `CAKE` and `FOAM` ship.
- **Granular, liquid and dough**, `11` to `13`:
  [`granular.py`](../../robosimstudio_materials/granular.py),
  [`liquid.py`](../../robosimstudio_materials/liquid.py) and
  [`dough.py`](../../robosimstudio_materials/dough.py) are `rss.heap`, `rss.liquid`
  and `rss.dough`, on Newton's implicit MPM in
  [`mpm/`](../../robosimstudio_materials/mpm/). The presets are `rice`, `sand`,
  `beans`, `flour`; `water`, `honey`, `ketchup`; and `dough`, `plasticine`, `clay`,
  `butter`. Two caveats for a liquid: the `water` preset loses volume, so it is
  read by counts, not heights; and a wall thinner than two voxels leaks unless its
  body is in the liquid's `containers`.
- **Water**, `14`: [`water.py`](../../robosimstudio_materials/water.py) is `rss.water`,
  SPH on Genesis that keeps its volume. `water` and `oil` ship.

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Rope: drag a cable

![Rope: drag a cable](media/00_rope.webp)

A cable lying on the bench under an open hand: drag the hand down onto it, close it and lift, or click any segment and drag it.

Drag the right hand down until the fingertips are just over the bench, press
**Close**, and lift slowly: friction alone holds the cable, which takes the
static friction `contact_matching="sticky"` supplies. Clicking a segment grabs
it instead, and `at` places the cable's first end, not its middle.

```bash
uv run examples/03_materials/00_rope.py
```

Source: [`00_rope.py`](00_rope.py)

## Rope memory: wrap a post

![Rope memory: wrap a post](media/01_rope_memory.webp)

A cable with plastic memory lies on the bench, clamped at its far end: drag the right hand down onto it, press **Close**, lift it a little, carry its end round the post while turning the gizmo's ring a third of a turn, lay it down, press **Open** and lift away, and it stays wrapped.

`dahl_eps_max` turns on Newton's Dahl bending hysteresis: a bend put into a
joint past `dahl_tau` radians is kept. The pads alone hold the cable (sticky
contacts, as in `00_rope`). Carried 120 degrees round the post and let go,
it reads straightness 0.69; built at 0, the same drive springs back to 0.99
(`tests/test_cable_robot`, GPU 2). The bench and posts have no friction, so
the bend is the cable's: with 0.24 or 0.024 left, the elastic cable stayed
wrapped too, 0.74 and 0.78 (`mem_drive` probe, GPU 2).

```bash
uv run examples/03_materials/01_rope_memory.py
```

Source: [`01_rope_memory.py`](01_rope_memory.py)

## Twist: turn two cables

![Twist: turn two cables](media/02_twist.webp)

Two cables hang from a clamp, one twisting ten times more easily than it bends and one ten times less: drag each hand onto its cable's tip, press **Close**, turn each gizmo's ring a third of a turn about the cable and press **Open**, and the stiff one springs the turn back out while the soft one keeps it.

`twist_ke` splits Newton's rod joint from its bend. The pads alone hold each
tip, turned about the vertical. Turned 120 degrees, the soft cable takes 0.32
turns and the stiff 0.32 too, held by the hand's 50 N clamp; opened, the stiff
reads 0.001 in 30 steps, the soft 0.106 in 60 (`tests/test_cable_robot`, GPU 7).

```bash
uv run examples/03_materials/02_twist.py
```

Source: [`02_twist.py`](02_twist.py)

## Knot: pull it tight

![Knot: pull it tight](media/03_knot.webp)

A loose overhand knot in a rope lies on the bench between the two hands: drag each hand down onto its tail, slide its **Grip amount** to 0.90, drag both hands up 120 mm, then each out 85 mm, and the knot draws up and stays a knot.

A knot is a rope crossing itself, held apart where it crosses by its own contact
(`self_contact`: SuperDex tests the rope's skin against itself). The pads hold the
tails by friction, nothing pinned. Pulled to 490 mm across, the knot gives back 59
of its 361 mm of rope and keeps its three crossings, the closest strands 0.29 mm
into each other on any step (`k_knot.py`); with self-contact off one strand goes
straight through another, a whole 12 mm, and one crossing is left
(`tests/test_knot_pull`, GPU 1). BDF2 drew it to 239 mm but let a tail slide 88 mm
through a hand (`probe_c3`). 20 to 45 ms a control step. Needs `[superdex]`.

```bash
uv run --extra superdex examples/03_materials/03_knot.py
```

Source: [`03_knot.py`](03_knot.py)

## Band: hook it on pegs

![Band: hook it on pegs](media/04_band.webp)

A rubber band hangs on the upper peg of a pegboard: press **Close** and drag the right hand's shut jaws into the loop, down past the lower peg, in toward the board under it, up, and back out, and the band stays stretched round both pegs.

A band is a closed rod (`RodSpec.closed`): SuperDex closes its polyline with an
element from the last node back to the first, so it has no ends (the capsule chain
refuses one). The shut jaws carry it by contact alone; nothing is pinned. It ends
round both pegs about 19% longer than at rest, held by the pegs; the same cord cut,
driven the same way, is left on the upper peg at its rest length, so what keeps it
is the loop (`tests/test_band_on_pegs`, GPU 1). Backing out, the jaws drag the
band along the lower peg: in one filmed take of two it went off the end, and the
peg's upturned tip is what keeps it. 18 ms a control step. Needs `[superdex]`.

```bash
uv run --extra superdex examples/03_materials/04_band.py
```

Source: [`04_band.py`](04_band.py)

## Paper: fold a sheet

![Paper: fold a sheet](media/05_paper.webp)

Bring each hand in along the edge of the A4 sheet that runs past the box, from its own corner, with **Grip amount** 0.79, and press **Close**: the fingers stall on the sheet at the grip's 10 N cap. Carry the edge up and over, each hand turning with the flap to 88 degrees and no further, and press **Open** 100 mm over the far half: the flap falls onto it. Then, at **Grip amount** 0.6, run the side of each hand's lower finger along its half of the fold, the sheet pushing it up its slide against the same 10 N cap: the sheet stays folded in half.

On the YAM's pads, at friction 1 (paper's 0.3 mixes to 0.55 on them), the arms, which collide, kept 79.5 mm
apart and 12.0 and 9.0 mm on their own sides, each hand turned 88.0 degrees holding the edge, and no finger pushed
over 15 N at any step (8.7 N holding, 4.8 after **Open**, 12.4 pressing). The edges ended 0.3 to 2.3 mm apart and the
fold kept 2.89 rad, unchanged 3 s after the hands (`tests/test_fold_paper_robot`).

```bash
uv run examples/03_materials/05_paper.py
```

Source: [`05_paper.py`](05_paper.py)

## Bag: open its mouth

![Bag: open its mouth](media/06_bag.webp)

A canvas bag lies with its mouth toward the right hand and its top layer's lip turned up: drag the right hand down until its open jaws straddle the lip, press **Close**, drag it up 80 mm and back over the bag, and the mouth opens.

Nothing is pinned: the lip comes up because the pads squeeze it, about 0.3 N a side, and
nothing holds the bag down; its closed end moved 5 mm. Closed 25 mm over the bench and drawn
60 mm back, the mouth read 75-77 mm in seven drives; closed 12-30 mm over it, 66-90 mm; closed
over the lip's top, 45 mm up, the hand lifted nothing and the bag read shut (`f_bag.py`, GPUs
1 and 3). The bag is one SuperDex shell that collides with itself: its layers rest 3.9 mm apart,
a contact radius, and stayed 3.5 mm apart while the hand opened it. Set `self_contact` to
False and the top layer lies in the bottom one instead: -0.24 mm at rest, -0.29 mm held open.
The grip is the page's own servo, 8 N a finger: on the YAM's old kp 200 spring the pads squeezed
the lip with 0.02-0.05 N, it slipped, and the mouth opened 64-67 mm (its hand is a 50 N clamp now). The **Bag** folder reads
`mouth_opening`, the widest span across the rim and the area it encloses, and `layer_gap`.
Through the page, 11 ms per control step at rest and 41 ms while the pads hold the cloth (RTX
PRO 6000, GPU 1). Needs the `[superdex]` extra (Python 3.12).

```bash
uv run --extra superdex examples/03_materials/06_bag.py
```

Source: [`06_bag.py`](06_bag.py)

## Brush: scrub a patch

![Brush: scrub a patch](media/07_brush.webp)

A nylon dish brush stands on the bench under the right hand, which points down at its handle, beside a marked square: drag the hand down onto the handle, close, press the bristles to the bench and scrub the square.

The brush is 400 bristles of four capsules each on VBD, rooted in a sole of
twenty strips, under a block on the arm's solver that the hand closes on
(`rss.BRUSH_60HZ`); the field cannot hold that block up across the two
solvers, so it is gravity-compensated, and the arm does not feel the bristles
push back. The panel reads the share of the square's 5 mm cells a bristle tip
has touched while down on the bench. `tests/test_scrub_bench` closes the hand,
presses 4 mm and makes three strokes across the square: all of its cells
scrubbed, every tip down throughout, the handle slipping 0.5 mm in the
fingers, at 22.4 ms per control step on an RTX PRO 6000, GPU 3,
graph-captured.

```bash
uv run examples/03_materials/07_brush.py
```

Source: [`07_brush.py`](07_brush.py)

## Tape: peel a label

![Tape: peel a label](media/08_tape.webp)

A paper label stuck on a box, its end in a blue pull tab under the right hand: drag the hand down until its open jaws straddle the tab, press **Close**, drag it up 30 mm, then back toward the arm 20 mm and up 10 mm, seven times, and the label peels off the box row by row.

The grasp is contact: the pads squeeze the tab and nothing pins it to the hand. The label is a cloth
held to the box by Newton 1.7 particle-body attachments, one per particle, and a bond lets go once the
label has lifted 1 mm off the box, so the peel front travels with the hand. In 24 drives all 144 bonds
let go, the tab stayed within 3.4 mm of where the pads took it and the free box moved 1.7 mm at most
(`peel_probe.py`, two drives to a GPU). The jaws are damped to reach the tab in 0.4 s: shutting at
8 mm a step they knocked it aside and closed on nothing in 3 of 116 closes (`peel_probe.py`,
`peel_close.py`, the round-3 review). Tick **Never let go** and no bond lets go: the pull drags the
box until the tab slides out of the pads (10 drives of 24), or, where the pads hold, drags the whole
box 46 mm by its label (14 of 24). The tab is rigid, because the pads cannot hold a cloth tab under
the coupling the attachments need, and weighs 30 g (`robosimstudio_materials/tape.py`). The pads
hold it with 5.8 N; only Close's first push (71 N) meets the 12 N finger cap. The **Label** folder
reads `peeled_fraction` and the peel front. 38 ms a control step through the page (RTX PRO 6000).

```bash
uv run examples/03_materials/08_tape.py
```

Source: [`08_tape.py`](08_tape.py)

## Adhesion: stick a label

![Adhesion: stick a label](media/09_adhesion.webp)

A loose paper label lies on the bench beside a box, its end in a blue pull tab under the right hand: set **Grip amount** to 0.6, drag the hand down onto the tab and press **Close**; drag it up, over the box's far end and down toward its near end, and the label lies down; set the grip back to 0.6 and lift the hand. Then press **Close** and rub the shut fingertips along the label, four strokes 10 mm apart: it sticks where they pressed, and **bonded_fraction** passes 0.8.

The label carries ipc-toolkit's adhesion law (`rss.tapes.LABEL`, `robosimstudio_materials/tape.py`):
a bond forms where the sheet is pressed onto the box harder than 5 kPa, pulls hardest, 4 kPa, at a
0.5 mm lift and lets go at 1 mm. The grasp is contact, the pads on the rigid tab. Laid, the label held
1-6 of its 126 bonds; in twelve drives the four strokes made 85-93% of them, and in eight the tab drawn
straight up 120 mm peeled all but 0-3 at 2.15-2.38 N on **pull on the tab**; the box moved 0.7 mm at most, as much as
it does with no bond. With **Adhesive off** the strokes made no bond (`tests/test_stick_a_label`). Rub lightly,
the gizmo 3 mm under the label: at 5 mm the fingertips dragged it along in two drives of three. 35 ms a control
step (RTX PRO 6000).

```bash
uv run examples/03_materials/09_adhesion.py
```

Source: [`09_adhesion.py`](09_adhesion.py)

## Cutting: cut the tofu

![Cutting: cut the tofu](media/10_cutting.webp)

A block of firm tofu on a cutting board, and a kitchen knife standing on its edge in a rest under the right hand: drag the hand down onto the knife's handle, press **Close**, drag it up 70 mm, across to the tofu and down until the edge meets the board, and the tofu parts in two.

The grasp is contact: the pads close on the 20 mm handle at 20.4 mm and nothing pins it to the hand.
The tofu is a cage on the vendored walk-on-spheres cutter, its base held by the board; the cut starts
in the blade's plane, runs where the edge pushes past the blade's threshold (0.18 N across 60 mm of
tofu), and the knife feels that push. In three drives through the arm panel (`tests/test_cut_tofu`)
the tofu read 2 `pieces` with the edge on the board, and 1 piece, cut 88% through, with the hand
stopped 15 mm higher. Tick **Dull blade**, a blade a thousand times duller, and push the same way: the
push swings from step to step, up to 18-21 N on the way down and 3-19 N with the hand held low; the
knife tips 14-17 degrees in the pads and the arm stalls with the edge's lowest point 7-11 mm into the
tofu, which stays whole (13 drives, the test's and one without its pause). 13 ms a control
step cutting through the page, 9 ms idle (RTX PRO 6000).

```bash
uv run examples/03_materials/10_cutting.py
```

Source: [`10_cutting.py`](10_cutting.py)

## Granular: sweep rice

![Granular: sweep rice](media/11_granular.webp)

A heap of rice under the right hand, a pusher standing on its edge beyond it and a three-walled tray on the near side: turn the hand to point down, drag it onto the pusher, close on it, and sweep the heap into the tray at a walking pace.

The rice is 16,000 particles on Newton's implicit MPM, stepped once per 60 Hz
control step beside the YAM (`rss.GRANULAR_60HZ`: 16.5 ms per control step on
an RTX PRO 6000, GPU 3, graph-captured, measured by `tests/test_sweep_rice`
on this scene). The pusher is 20 mm thick and 40 mm tall so the jaws close on
it from above, and the heap names it in `containers` so the grid sees it one
voxel fatter than the hand does; without that the bottom voxel of rice passes
under it. The tray's walls are five voxels thick and its open side faces the
heap, so the sweep ends inside it. Swept at 0.15 m/s as the test sweeps it,
93% of the rice is in the tray and the rest is on the bench, but for a grain
the squeeze shot off the bench's edge in two sweeps of thirty (the others lost
none); a tool faster than ~0.36 m/s outruns the grid. What reaches the tray
is packed, not heaped: 88% of the grains end in a strip 32 mm deep along the
pusher's face, half of them within 1.3 mm of the bench, and a tenth runs
round the pusher's ends (`sweep_where.py`, GPU 0). The panel reads the
share of the rice inside the tray.

```bash
uv run examples/03_materials/11_granular.py
```

Source: [`11_granular.py`](11_granular.py)

## Liquid: pour honey

![Liquid: pour honey](media/12_liquid.webp)

A cup of honey under the right hand and a bowl beside it: drag the hand down until its open jaws straddle the cup, close on it, drag it up and across over the bowl, then turn the gizmo 40 degrees about the vertical and 140 degrees about the axis pointing away from the robot.

The jaws open 94 mm and the cup is 66 mm across, so they close on its outside walls; it is a rigid body on the
arm's island (`rss.LIQUID_120HZ`), held by friction, its 8 mm walls named in the honey's `containers` for
the margin a liquid needs. The arm sags about 30 mm under 0.48 kg of cup and honey: lift the gizmo 150 mm over
the bench to clear the bowl's 40 mm walls. The wrist's last joint stops at 120 degrees: turned 140 about the
away axis alone, the hand turns 122 and the honey creeps along the wall, about 56% in the bowl after 12 s held
(three drives, `fixprobe.py`, GPUs 1 and 6). The turn about the vertical first lets the arm reach 140, where the
honey leaves the cup in one mass within a second: 99% or more in the bowl, 0 to 8 of the 8,000 particles through
the bench where it lands, the cup 7 mm along the pads (`tests/test_pour_honey`, GPU 1, RTX PRO 6000). Honey,
because Newton 1.6's water preset loses volume (`robosimstudio_materials.liquid`). The panel counts the honey
in the bowl. 29.8 ms per control step, graph-captured.

```bash
uv run examples/03_materials/12_liquid.py
```

Source: [`12_liquid.py`](12_liquid.py)

## Dough: press a ball

![Dough: press a ball](media/13_dough.webp)

A puck of dough under the right hand: close the gripper, drag the hand down into the dough and lift it away again.

What a plastic solid keeps is the measure: the press probe behind the `dough`
preset kept 83% of a 15 mm compression after the plate lifted (plasticine 75%,
clay 66%). The panel reads the puck's height as a share of the height it
settled at, and its footprint against the settled one: the closed hand driven
15 mm under the top through the arm panel and lifted away leaves 76% under
the hand and 76% after (`tests/test_press_dough`, GPU 0). Change `material` to
`"plasticine"` or `"clay"` for a stiffer lump. 23.4 ms per control step on
an RTX PRO 6000, graph-captured.

```bash
uv run examples/03_materials/13_dough.py
```

Source: [`13_dough.py`](13_dough.py)

## Water: pour into a glass

![Water: pour into a glass](media/14_water.webp)

A cup of water under the right hand and a glass beside it: drag the hand down until its open jaws straddle the cup, close on it, drag it up to 170 mm over the bench and 125 mm across, then turn the gizmo 120 degrees about the axis pointing away from the robot, and the water runs into the glass past its red line.

The water is Genesis's SPH (`rss.water` under `rss.WATER_60HZ`), which keeps its volume where Newton's MPM
water loses it; it runs where the `[genesis]` extra is installed, in a virtual environment of its own (this
folder's README). The jaws close on the cup's 66 mm walls, and the water's reaction comes back onto the cup:
carried still, it reads 1.09 N where the water weighs 1.085 N. The wrist stops at 115 degrees: in three drives
99.2 to 99.9% of the 1,728 particles ended in the glass, standing 29.4 to 29.7 mm where their count fills 30.5 to
30.7 mm, 0.1 to 0.8% on the bench (`tests/test_pour_water`'s gesture, GPU 1, RTX PRO 6000). Turned 60 degrees
it keeps the water (at most 0.1% in the glass); carried 100 mm aside, 99.2% lands on the bench. 24 ms per step.

```bash
uv run examples/03_materials/14_water.py
```

Source: [`14_water.py`](14_water.py)

<!-- examples:end -->

## Water's own environment

The water runs on Genesis, which lives in a virtual environment of its own,
beside the checkout's, because installing the `[genesis]` extra moves twenty
packages. From the repository root:

```bash
uv export --frozen --extra genesis --extra dev --no-group rtx --no-emit-project --no-hashes > /tmp/genesis-req.txt
uv venv --python 3.12 /tmp/gvenv && uv pip install --python /tmp/gvenv/bin/python -r /tmp/genesis-req.txt
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 QD_VISIBLE_DEVICE=0 /tmp/gvenv/bin/python examples/03_materials/14_water.py
```

Without Genesis the water scripts stop at the build and name the extra.
