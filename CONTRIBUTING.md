# Contributing to RoboSimStudio

Thanks for helping. Issues, fixes and examples are all welcome.

## Set up

You need an NVIDIA GPU and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/robosimstudio/robosimstudio.git
cd robosimstudio
uv run examples/00_getting_started/00_hello_world.py   # builds the environment on first run
uvx pre-commit install                                  # ruff on every commit
```

## Check your change

```bash
uv run python scripts/testrun.py --tier fast     # no GPU; CI runs this
uv run --extra gap python scripts/testrun.py --tier medium   # needs the GPU; builds and steps scenes
uvx ruff@0.16.1 check . && uvx ruff@0.16.1 format . --check
```

Every test module declares `TIER = "fast" | "medium" | "slow"`;
`scripts/testrun.py` exits non-zero, naming the module, while one does not. A
test that builds a scene is `medium`. CI runs the fast tier only; run the medium
tier on a GPU machine before a change that touches simulation.

## Where things go

- **A new example** is one script under `examples/<NN_category>/`. Line 1 of its
  docstring is the title, the next sentence says what it shows, and it defines
  `main()`. A looping clip of it goes in that folder's `media/` as
  `<script>.webp`; then `uv run python scripts/example_readmes.py` writes its
  section of the folder's README.
- **A new kind of thing** or **a new robot** registers itself; the library is not
  edited. See `examples/04_extending/`.
- **A task** is a document and **a policy** is a graph folder; neither needs a
  change to the library. See `examples/07_tasks/` and `examples/08_policies/`.
- **Imports go one way.** The library imports neither the studio nor the agent,
  and a lower layer never imports a higher one.
- **A number in a docstring was measured.** A new one should be too.

## Commits and pull requests

A commit message is a lowercase `area:` prefix and one declarative sentence:
`studio: the room, and lights that match the render`. Keep a pull request to one
change, say how you tested it, and include a picture when it changes what a
scene looks like.

By contributing you agree that your contribution is licensed under the
repository's Apache-2.0 licence.
