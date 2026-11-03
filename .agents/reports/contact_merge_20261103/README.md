# One validation of the 2026-11-03 contact merge

Result: **FAILED pickup**, not a release failure. These files describe the paired candidate RSS `ac28ac3` + Newton `3d3f9e4c`. No retry or parameter sweep was run. The exact task source hash is in metadata.json; repository revisions/import paths are in revisions.json.

`states.npz` stores all 300 frames, with `particle_q` shape (300,1840,3) and `body_q` shape (300,4,7). Frame 1 is at 1/30 s. Diagnostics are losslessly compressed. The MP4 displays all 300 control frames, each held twice at 60 fps (10 s duration).

The launcher files are archived verbatim as provenance. They were executed from `outputs/contact_merge_20261103/` in the paired integration worktree; their directory-relative assumptions refer to that original location, not this archive.

See ../MERGE_contact_20261103.md for merge status and measured limitations. Historical successful H15 evidence is in ../contact_20261102/ and uses a different configuration.
