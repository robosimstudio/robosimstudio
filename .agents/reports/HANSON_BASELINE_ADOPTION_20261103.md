# Hanson runtime adoption — 2026-11-03

## Decision

Adopt Hanson's runtime unchanged, retaining our investigation reports and evidence.
The user explicitly authorized merging this development baseline despite the known
failed default pick/drop result. This is not a claim of successful pick-and-place.

- RSS source: `hanson/vbd-improvements-3`, `d50f86f51b31a1fbc42844ef78a4b851eb693159`.
- Newton source: `hanson/sticky-contacts`, `3d3f9e4c13fd0e60e7e53a9870946c0983d1d14c`.
- All RSS files outside `.agents/reports/` and the example README match that RSS source.
- Newton's complete source tree is taken unchanged from that Newton source.
- Our owned-pipeline wiring, tolerance override, controller/settings and retained
  main-derived coupling differences are not included in the selected runtime.
- Historical reports and their launch scripts describe their recorded revisions;
  they are evidence, not launch instructions for the new baseline.

## Existing validation and limitation

Merge checks: runtime tree equality, changed Python syntax and Git whitespace checks pass.
The required fast suite was attempted but could not start: the existing simulation
Python is ARM64 while this login host is x86_64 (Exec format error). No replacement
environment was built for this documentation-and-baseline merge.

No new simulation was run for this adoption. The already completed clean-source
run (job 4860245) used exactly this source pairing: 600 frames / 20 seconds,
146.53 mm maximum clearance, then loss of finger contact at 2.567 seconds before
transport and the programmed opening. Native task success was false. No inverted
or nonfinite tetrahedra were recorded; maximum recorded penetration was 0.035885 mm.
See [baseline comparison](BUG_FIX_hanson_baseline_20261103.md) and its linked video,
configuration, diagnostics and numerical results. Hanson's supplied videos show
better outcomes but their exact launch configuration has not been matched.

The next simulation should use Hanson's exact video launcher and paired revisions,
when available, rather than guess settings or add another solver modification.

## Preserved history

RSS branch `yiduo/archive-contact-work-20261103` at
`63bc6ac89d84d4bda43d597cceb6a65b76b3e6d6` preserves these histories as parents:

- `e2f84a5`: final prior integration candidate, including failed merged and clean
  Hanson comparison evidence, and the earlier Chebyshev integration/revert history.
- `73e0601`: H15 full placement configuration and results (one pass; about 92 minutes).
- `1a89afe`: earlier experimental solver/backend work.
- The local release stash, including its own parent snapshots. The local stash
  remains in place; it is also reachable from the remote archive.

Newton branch `yiduo/archive-contact-work-20261103` preserves `d960ec3b`, including
our USB-hub example work. Existing development and older archive branches remain.
Temporary integration branch names can be deleted after their commits are verified
reachable from main or the archive. Unrelated untracked AIRBOT work stays local
and untouched; this archive does not claim to include those untracked files.
