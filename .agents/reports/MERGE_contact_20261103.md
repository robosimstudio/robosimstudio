# Contact merge and one validation — 2026-11-03

> Historical investigation record. The subsequent user-authorized adoption of
> Hanson runtime is recorded in [the adoption decision](HANSON_BASELINE_ADOPTION_20261103.md).

## Current candidate

At the user's subsequent request to replicate Hanson's default coarse solver, the active RSS candidate reverts `ac28ac3` and restores the published pick/drop scene and original collision path. The prior Chebyshev candidate is preserved on `yiduo/archive-chebyshev-merge-20261103` at `3f088be`; all evidence remains in this branch. The simulation sources match the configuration already measured in job 4860245; see [comparison](BUG_FIX_hanson_baseline_20261103.md). No additional simulation was run for this restoration. The known early loss remains, and neither main branch has been updated.

## Initial merge outcome

The branch merge candidates are prepared and pushed, but **neither remote main was advanced** because the single agreed standalone validation failed pickup. No retry, tuning campaign or additional simulation was run during the initial merge validation. The user subsequently authorized a separate unmodified Hanson baseline comparison; see [that comparison](BUG_FIX_hanson_baseline_20261103.md).

- Newton integration branch: `yiduo/integrate-sticky-contacts-20261103`, commit `3d3f9e4c13fd0e60e7e53a9870946c0983d1d14c`, fast-forwarding main `1a640e4b5e6ce9e530b359ea1d99d07600703198` with all three Hanson sticky-contact commits.
- RSS integration branch: `yiduo/integrate-hanson-v3-20261103`. Merge `ecca1ef` combines main `f1b648c1d987aee080e2eff742752e9f64048fab` and Hanson `d50f86f51b31a1fbc42844ef78a4b851eb693159`. `8d81fe6` preserves historical evidence/documentation. **Tested code revision: `ac28ac3`**, connecting the native owned pipeline and the standalone validation.
- Original RSS yiduo/dev and Newton yiduo/main checkouts, including unrelated AIRBOT and USB-hub work, are preserved.

## What changed

| Area | Merge decision / purpose |
| --- | --- |
| Newton | Keep Hanson's rigid-soft and self-contact history ports plus reset-tracker fix unchanged. No new Newton solver changes. |
| RSS scenes and gripper | Take Hanson versions, including the kinematic XYZ guide needed by the X-carry scene. Keep shared coupled infrastructure dormant. |
| Collision integration | Pass existing RSS sticky pipeline to a single VBD solver, expose solver-owned contacts to diagnostics, and skip duplicate external detection. This is required to activate the self-contact port. Other solver/coupler paths retain their existing route. |
| Acceleration | Explicitly select existing SolverVBDChebyshev, rho 0.9, gamma 1, start 0, restart false. No coarse translation correction or separate rigid solver in this run. |
| Edge tolerance | Keep scene-level particle_edge_parallel_epsilon=1e-12, verified in solver, detector and self-contact matcher. No global Newton tolerance change. |
| Validation | Reuse mesh-quality/contact diagnostics and strict pickup/carry/release checks. Save states and separate stepping/readback/render timing. |
| History | Preserve older report, video, states and numerical results; exclude old expensive trajectory/settings and ADMM friction-routing edits. |

The paired Newton revision above is required for this RSS integration; the checkout still uses the existing editable sibling Newton dependency. Source asset checks confirmed one labelled 1840-vertex/6850-tet mesh, original stiffness/density, preserved tet order and nodal masses, and zero grasp attachments. Scene viscosity stays at 5000 Pa.s; the doc now explicitly describes its physical rate-dependent effect.

## Single validation

Slurm job **4859713**, GH200, 300 frames at 30 Hz, 20 substeps x 12 iterations. Original 5 N per-jaw and 200 N carrier caps; vertex-only rigid-soft collision; contact matching sticky; both persistent contact paths allocated and active. No adhesion.

| Criterion | Measurement | Result |
| --- | --- | --- |
| Complete episode | 10 simulated seconds / 300 frames | Pass |
| Retain until opening at 5.3 s | Last finger contact frame 43 (1.433 s), none from frame 44 (1.467 s) onward | Fail |
| Whole-object transit clearance >=20 mm | 0.4365 mm | Fail |
| Carry 80 +/-20 mm over 3.5-4.7 s | -0.000119 mm (effectively zero) | Fail |
| Final finger contacts | Zero | Pass; object was never picked up |
| Final table support | 0.4364 mm clearance | Pass |
| Final peak vertex-centroid speed <=5 mm/s | 0.00566 mm/s | Pass |
| Inversion / nonfinite tet-substeps | Zero / zero | Pass |
| Recorded penetration <=0.5 mm | 0.3455 mm maximum | Pass |
| Jaw drive peaks <=5 N | 5 N / 5 N | Pass |

Saved body poses confirm the gripper lifted and moved horizontally while the toy remained on the table. This localizes the failure to acquisition/early lift, before transport or commanded release. It does **not** establish why the grasp failed, invalidate Hanson's rod-scene results, or test release under a retained load.

Both persistent paths were verified at startup, but this nearly stationary object produced **zero self-contact pairs**. Consequently this episode does not validate loaded self-contact friction response. Existing penetration/contact checks are frame sampled and only cover detected contacts; the quality latch samples every substep. The 5 N bound is on jaw drive effort, not every constraint/contact force. Reset/batched/coupled behavior and other scenes were not tested.

Timing: 151.25 s stepping including the existing quality hook and lazy kernel compilation, 0.86 s diagnostics/readback, 0.91 s rendering/video append, 23.19 s scene setup; whole Python process 194.25 s, Slurm elapsed 201 s. About 0.504 s/frame for stepping, but this failed stationary-object episode is not evidence of successful grasp throughput.

## Evidence and next decision

[Video](contact_merge_20261103/overview.mp4), [numerical result](contact_merge_20261103/result.json), [configuration](contact_merge_20261103/metadata.json), [revisions](contact_merge_20261103/revisions.json), [states](contact_merge_20261103/states.npz), and [log](contact_merge_20261103/slurm-4859713.log). The archive also holds compressed per-frame diagnostics and original launchers.

Smallest next investigation: inspect pad/rim geometry and the contact load during 1.1-1.47 s, distinguishing poor grasp acquisition from insufficient friction retention before changing any solver. That is a separate debugging task; no change or retry was made here. Publishing these candidates despite the failed acceptance run would require a decision to relax the publication gate, not a claim that the task passed.
