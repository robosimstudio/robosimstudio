# Wipe the bar

![Wipe the bar](media/wipe_the_bar.webp)

The twin of a real cell, and the whole job in it: two YAMs at their home pose,
a kitchen sponge lying on a gridded foam mat, and a chrome grab bar bolted into
two black blocks with a blue marker stripe drawn along its crown. Press **Wipe
the bar** on the page and the arms do what the clip shows: the left hand
steadies the bar at its far elbow, the right takes the sponge off the mat,
presses it onto the bar and strokes the marker off, puts the sponge back where
it lay, and both arms go home.

**The grasp is physics; the hold after it is a latch.** The jaws close on the
foam by their servo and SuperDex contact alone. Once they have settled on it,
both pads pressing and the foam squeezed, the foam under the pads is pinned to
the hand where it is, and stays pinned through the wipe until the jaws are told
to open. Held by friction alone, the sponge slid out of the jaws in the wipe;
the numbers are under [Why a latch](#why-a-latch).

```bash
uv run --extra superdex python demo/wipe_the_bar/wipe_the_bar.py
```

The first run builds the YAM's fingertip collision and compiles the kernels, so
give it a minute. Needs the `[superdex]` extra (Python 3.12) and a GPU.

## The files

| | |
| --- | --- |
| [`wipe_the_bar.py`](wipe_the_bar.py) | builds the scene, installs the latch and opens the page, with the press readout, a contact map and the Run button |
| [`bar.py`](bar.py) | the scene: the layout measured off the clip, the welded bar, the stripe that wears off, the physics, the latch |
| [`routine.py`](routine.py) | the job, as a generator the page plays one control step at a time |

The layout is the `surface_prep` task's, measured off the *Surface Preparing*
clip against the mat's 25 mm grid: a 25 mm tube with a 340 mm straight run on
two 32 mm elbows, its legs 50 mm into 76 x 80 x 70 mm blocks on a 10 mm mat; the
stripe is the middle 200 mm of the crown and 9 degrees wide (243 cells of
2.5 mm); the sponge is 50 x 75 x 30 mm, 70 mm past the near block's outer face.
The stripe wears by Preston's law, `ink' = -k_p * pressure * sliding speed`, fed
by SuperDex's own contact samples on the bar, with no wear under 300 Pa. Its
**cleaned fraction** -- the share of cells with a tenth of their ink or less --
is the job's measure.

## The grasp, the latch, the release

In that order, and the same whether the routine presses the buttons or you do:

1. **The friction grasp.** **Close** on the sponge, and the jaws close at no more
   than 80 mm/s and squeeze it at their 8 N cap: by contact, with nothing pinned.
   They stop 45.6 mm apart on the 50 mm foam, each pad pressing it with 8.6-8.9 N.
2. **The latch, once the grasp is made.** While the jaws are told to close,
   `bar.Latch` checks after every control step for all three of:
   - the opening still, moving no more than **0.3 mm over 12 steps (0.2 s)**;
   - each pad pressing the foam with at least **2 N** of normal force, read off
     SuperDex's contact records;
   - the opening at least **2 mm under the foam's own width** across the jaws.

   Then it pins the 106 foam nodes within 6 mm of the pads' contact samples
   where they stand, and the bridge carries them on the hand, re-aimed every
   engine substep. A hand closed on air or on one edge never meets the test and
   never latches, and the routine stops and says what it read. Measured: it
   engaged 1.0 s after Close; the pinned foam moved 0.006-0.008 mm in the hand in
   the next 0.1 s, 0.008-0.012 mm by the time the hand had risen 10 mm, and
   0.06 mm through the whole wipe.
3. **The release.** **Open** unpins the foam on the step the command lands. The
   sponge is set down and let go by the jaws.

## What the routine does

Every move is a drag of an arm's gizmo and every grip a press of its Grip
control, so the page shows what a person would do by hand:

1. **Steady.** The left gizmo goes 50 mm over the far elbow, down until the jaws
   straddle the tube 4 mm over its axis, and the jaws close to 29 mm: a hand
   resting on the bar, not clamping it, and never latched.
2. **Pick.** The right gizmo goes 100 mm over the sponge, top-down with the jaws
   across its 50 mm side, down until the pads are on its middle, and **Close**;
   the take waits up to 1.5 s for the latch.
3. **Carry.** Up 100 mm, over the near block at the crossing height leaning 35
   degrees along the bar, then over the near end of the stripe.
4. **Wipe.** Down a millimetre at a time until the stripe reads 4 N, then a
   stroke toward the far end at 60 mm/s, the hand following the press up and
   down; up, back, and again, four times or until the stripe is 96 % clean.
   Strokes go one way only, the low edge of the leaning sponge trailing.
5. **Put back.** Back over the near block, down onto the spot the sponge came
   from, **Open**, up; the left hand opens and lifts; both gizmos go home.

A take is 59.5 s of simulated time.

## Measured

Three takes of the test (`python -m unittest tests.test_demo_wipe_the_bar`),
2026-10-13, one RTX PRO 6000 Blackwell, nine assertions each:

| take | cleaned | peak press | peak pressure | sponge put back | arms home | finite | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.955 | 4.93 N | 84 kPa | 0.1 mm from its spot | yes | yes | 9/9 |
| 2 | 0.959 | 4.91 N | 84 kPa | 0.1 mm from its spot | yes | yes | 9/9 |
| 3 | 0.959 | 4.91 N | 84 kPa | 0.0 mm from its spot | yes | yes | 9/9 |

In every take nothing was pinned before the first Close or after the last
Open, the latch engaged once and released once, and the four strokes cleaned
0.91, 0.94, 0.95 and 0.95-0.96 of the stripe in turn.

## Why a latch

Without it -- the same grasp, the foam held by the jaws' squeeze and its
friction alone, at the sponge's lowest workable `friction_falloff` (0.1 mm/s) --
three takes cleaned 0.761, 0.724 and 0.687 of the stripe, the foam under the
pads slid 69, 226 and 218 mm in the hand, and in two of the three the sponge
left the jaws before the last stroke. The pick and the carry held (3-9 mm of
slide); every stroke slid it 10-20 mm, back along the bar and up toward the
palm, with the pads reporting 8-9 N of grip and the bar 0.4-0.6 N of drag.

Tried over 30 more takes, none of which kept the slide under 10 mm: the
sponge's falloff at 1 and 0.1 mm/s; a 12 N grip; presses of 1.5 to 4 N; a 25
degree lean; strokes both ways; 120 mm/s strokes; bar friction 0.2 and sponge
friction 2.0; the clip's soft foam, which the jaws closed through on the pick.
And not the fingertips' decomposition: mirrored as one convex hull of each tip's
last 30-60 mm instead of 48 and 52 pieces, the jaws held a lifted sponge just as
well (0.11-0.14 mm of slip over 3 s) and three wipes still lost it, cleaning
0.65, 0.81 and 0.71.

## Tests

```bash
uv run python -m unittest tests.test_demo_wipe_the_bar   # slow: the whole job, through the Run button
```
