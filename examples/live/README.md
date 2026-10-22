# live: a page that rebuilds `scene.py` every time you save it

Point a coding agent at this folder: it edits `scene.py`, the open page
rebuilds on every save, and the agent checks the result against your photo or
clip. The page is the studio, so you can drag things on it between edits.
This file is for you, starting the page and the agent. The agent's method is
[`AGENTS.md`](AGENTS.md); every command and reading is in
[`REFERENCE.md`](REFERENCE.md).

What is here:

- `scene.py`: a starter scene for trying the page without a twin. A twin's
  own `scene.py` is in its folder under `twins/`. Bind `routine` (or
  `routines = {name: fn}`) to get a Run button.
- `ctl`: the command line onto the running page; it also runs from the
  repository root as `examples/live/ctl`. It uses the checkout's `.venv`, else
  `$PYTHON` or `python3`, so it also works after
  `pip install "robosimstudio @ git+https://github.com/robosimstudio/robosimstudio"`.
- `run/`: made on first use and not committed. It holds the socket, the
  server's log and every snapshot of a page served on `scene.py` here; a twin
  keeps its own `run/` in its folder.
- `twins/`: one folder per twin, made by `./ctl new`, and not committed. Each
  is a twin on its own -- its `scene.py`, helpers, assets, `ref/`, `run/`,
  `REPORT.md` and its own `ctl` -- so zip one to hand it on, without its
  readings and bytecode (`zip -r lid.zip lid -x 'lid/run/*' '*/__pycache__/*'`
  from `twins/`). Unzipped outside a checkout, its `./ctl` runs on `$PYTHON`
  or `python3` with robosimstudio installed.

## Start

```bash
cd examples/live
./ctl new lid --ref ../../../IMG_8983.MOV   # the twin's own folder, twins/lid/, the clip copied into its ref/
cd twins/lid
./ctl up                                    # the page; prints its URL, http://localhost:8080 or the next free port (--share adds a public URL)
codex -a on-request -s workspace-write      # Codex; the sandbox may write here and reach the socket
claude --permission-mode acceptEdits        # or Claude Code; edits go through without a prompt
```

Start the agent in the twin's folder: Codex reads `examples/live/AGENTS.md` by
itself on the way down from the repository, and Claude Code reads it through
`CLAUDE.md`. An agent started anywhere else is pointed to it by the
repository's `AGENTS.md` (and Claude Code by the `real-to-sim` skill), and
makes the folder itself. Where Codex's sandbox cannot
start (`bwrap: loopback: Failed RTM_NEWADDR` on some hosts), run it in a
throwaway clone or worktree of the repository with its sandbox off, never in a
checkout you care about.

The page opens on the twin's starter scene: two arms on a bench. On a remote
GPU box, pick an idle card with `CUDA_VISIBLE_DEVICES=N ./ctl up` (the page
then calls it `cuda:0`; naming it with `ROBOSIMSTUDIO_DEVICE` alone still
holds memory on every card), and use `--share` or an SSH tunnel
(`ssh -L 8080:localhost:8080 <box>`). The page listens on all interfaces
(viser's default), so on a shared network firewall the port or use the
tunnel. `./ctl quit` stops the server once a build, a render or a routine's
step under way ends.

The method is in `AGENTS.md`, so the prompt only has to say what to build; one
line naming the clip or the photo is enough. The prompts below say more, for
a person who wants to steer. Start each with this line:

```
A live page is running: it rebuilds scene.py on every save. Use ./ctl to inspect it (see ./ctl --help).
```

## The prompt, with a clip

With the clip in the twin's `ref/` (`./ctl new <name> --ref <clip>`), say:

```
ref/<clip>.mov is a real clip of a person doing one thing at a table.
Rebuild it in scene.py with this library's core API: the table, two YAM
arms where the person sat, what is on the table, a camera where the phone
was, what stands behind the table, and then the arms doing what the person
did, by contact alone, bound as `routine` in scene.py so I get a Run button
on the page. I am watching the page. When you are done, report as AGENTS.md
says: what still differs from the clip, every shortcut, the gate, and the
look sheet and strip you judged. Say in one line what you are about to do
before each edit.
```

## The prompt, with your own photo

With the photo in the twin's `ref/` (`./ctl new <name> --ref <photo>`), say:

```
ref/<your photo> is a real photo of a robot bench. Rebuild that scene in
scene.py with this library's core API, one thing per save so I can watch each
one appear on the page: the bench, the robot, what is on the bench, and a
camera where the photo was taken from. When the scene matches, make the robot
do something to an object, by contact alone, bound as `routine` so I get a Run
button. Say in one line what you are about to do before each edit.
```

## The prompt, with no photo

```
Build this in scene.py with this library's core API, one thing per save so I
can watch each one appear on the page: two YAM arms, a towel and a box on a
bench. When the scene is built, make an arm push the box, bound as `routine`
so I get a Run button. Say in one line what you are about to do before each
edit.
```

## Where things go

A twin is its folder under `twins/`: its `scene.py` and the helpers and
assets beside it, `ref/` (the references, the clip's stills and
`timeline.json`), `run/` (snapshots, comparisons, takes, `log.jsonl`,
`serve.log`; [`REFERENCE.md`](REFERENCE.md#ctl) has the full list) and the
agent's `REPORT.md`. Zip the folder to hand it on. `scene.py` here is the
starter for trying the page without a twin.
