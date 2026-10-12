# live: a page that rebuilds `scene.py` every time you save it

Point a coding agent at this folder and it builds a scene while you watch: it
edits `scene.py`, the open page rebuilds, and it looks at the result through
snapshots. The page is the studio, so you can drag things on it between edits.

What is here:

- `scene.py`: the starter scene, and the agent's working file. Whatever it last
  built is what the file says. Bind `routine` (or `routines = {name: fn}`)
  and the page grows a Run button for it.
- `ctl`: the command line onto the running page (`python -m robosimstudio_real2sim_agent`,
  with `run/` pinned beside the file). It uses the checkout's `.venv` when there
  is one, and otherwise `$PYTHON` or `python3`, so it also works after
  `pip install robosimstudio`.
- `run/`: made on first use and not committed. It holds the socket, the
  server's log and every snapshot.

## Start

```bash
cd examples/live
./ctl up                                    # the page; prints http://localhost:8080 (--share adds a public URL)
claude --permission-mode acceptEdits        # Claude Code; edits go through without a prompt
codex -a on-request -s workspace-write      # or Codex; the sandbox may write here and reach the socket
```

The page opens on the starter scene: two arms on a bare floor. On a remote GPU
box, pick an idle card with `CUDA_VISIBLE_DEVICES` and use `--share` or an SSH
tunnel (`ssh -L 8080:localhost:8080 <box>`). `./ctl quit` stops the server.

Both agents pick the procedure up on their own, from [`AGENTS.md`](../../AGENTS.md)
at the root, so the prompt only has to say what to build.

## The prompt, with a clip

Drop the clip beside the checkout (not in it) and say:

```
A live simulation page is running on this box: it rebuilds scene.py every
time you save the file, and ./ctl is your window into it: status, stills
cut from a clip, the scene measured against a still in numbers, a gate, a
film of what the robot does held against the clip, and python against the
live scene.

../../../<clip>.mov is a real clip of a person doing one thing at a table.
Rebuild it in scene.py with this library's core API: the table, two YAM
arms where the person sat, what is on the table, a camera where the phone
was, what stands behind the table, and then the arms doing what the person
did, bound as `routine` in scene.py so I get a Run button on the page. Look at the pairs, the look sheets and the strip with your own eyes,
not only the numbers: the things in their detail, the background, what happened. I am
watching the page. When you are done, report the readings and what you saw:
the first frame against the clip, the gate, and the take against the clip's
timeline. Say in one line what you are about to do before each edit.
```

## The prompt, with your own photo

Make a `ref/` folder here, drop your photo into `ref/`, and say:

```
A live simulation page is running on this box: it rebuilds scene.py every
time you save the file, and ./ctl is your window into it -- status, snapshots
you can look at, stepping, and python against the live scene.

ref/<your photo> is a real photo of a robot bench. Rebuild that scene in
scene.py with this library's core API, one thing per save so I can watch each
one appear on the page: the bench, the robot, what is on the bench, and a
camera where the photo was taken from. Measure sizes and placements from the
photo. The repo's examples are yours to read for the API and the
robot names. After each save, snapshot through your camera beside the photo
(./ctl snapshot --beside ref/<your photo>), look at the pair, and adjust until
they line up. When the scene matches, make the robot do something to an object
and step it on the page so I can watch. Say in one line what you are about to
do before each edit.
```

## The prompt, with no photo

```
A live simulation page is running on this box: it rebuilds scene.py every
time you save the file, and ./ctl is your window into it -- status, snapshots
you can look at, stepping, and python against the live scene.

Build this in scene.py with this library's core API, one thing per save so I
can watch each one appear on the page: two YAM arms, a towel and a box on a
bench. The repo's examples are yours to read for the API and the
robot names. After each save, take a snapshot and look at it before the next
edit. When the scene is built, make an arm push the box and step it on the
page so I can watch. Say in one line what you are about to do before each
edit.
```

## Where things go

- Snapshots: `run/snap_001.png`, `run/snap_002.png`, and so on. With
  `--beside`, the pair is written next to it as `run/snap_NNN_vs_ref.png`.
- Comparisons: the pair `run/cmp_NNN_beside.png`, the look sheet
  `run/cmp_NNN_look.png` and the numbers in `run/cmp_NNN.json`; a take's
  strip as `run/cmp_NNN_strip.png`, written by itself when the take ends
  (with the last compare's keys) and shown on the page under its verdict.
  The numbers continue from the files already in `run/` across restarts.
- Takes: `run/<name>.mp4`, `run/<name>.json` (every frame's step, time and
  positions) and `run/<name>_seg.npz`.
- The log: `run/log.jsonl`, one line per snapshot, compare and check, with
  the hash of the `scene.py` it measured; `./ctl log` prints it.
- The clip's stills: `ref/tSSS.SS.jpg`, `ref/contact.jpg` and
  `ref/timeline.json`, from `./ctl ref frames`.
- The server's output: `run/serve.log`. A file that does not build leaves the
  previous scene up, and `./ctl status` prints the traceback.
- The result: `scene.py` itself. Copy it aside to keep it, and put the starter
  back for the next session:

```python
import robosimstudio as rss

scene = rss.Scene(cell="void", name="live")
scene.add(rss.robot("yam_i2rt"))
```

Every `ctl` command, the contract for `scene.py` and the frames and units are
in [`REFERENCE.md`](REFERENCE.md).
