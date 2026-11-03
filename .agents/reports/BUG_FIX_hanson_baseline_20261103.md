# Unmodified Hanson baseline comparison — 2026-11-03

> Historical investigation record. The subsequent user-authorized adoption of
> Hanson runtime is recorded in [the adoption decision](HANSON_BASELINE_ADOPTION_20261103.md).

## Outcome

**Removing our integration changes restores substantial lift, but Hanson's published standalone example still loses the toy before transport.** This is one clean baseline run, not a completed manipulation task or an isolated diagnosis of a particular code change. No simulation source was edited.

| Same measurement | Merged candidate | Unmodified Hanson |
| --- | --- | --- |
| RSS revision | ac28ac3 | d50f86f |
| Newton revision | 3d3f9e4c | 3d3f9e4c |
| Peak whole-object clearance | 0.437 mm; never picked up | 146.530 mm |
| Last finger-contact frame | 43, at 1.433 s | 76, at 2.533 s |
| First subsequent zero-contact frame | 44, at 1.467 s | 77, at 2.567 s |
| X travel during commanded carry, 3.5–4.7 s | -0.000119 mm | -0.202924 mm |
| Retained until opening at 5.3 s | No | No |
| Inverted / nonfinite tet-substeps | 0 / 0 | 0 / 0 |
| Maximum recorded penetration | 0.345531 mm | 0.035885 mm |
| Peak final-half-second centroid speed, common 10 s endpoint | 0.005659 mm/s | 7.450368 mm/s |
| Native task success flag | False | False |

Hanson's run lasted its default 20 seconds / 600 frames. At 20 seconds the peak centroid speed over the final half second was 8.083679 mm/s, above the unchanged 5 mm/s settling criterion. Final table clearance was 0.688–0.694 mm. Finger contact never returned after frame 76. Therefore the loss was not caused by the programmed opening at 5.3 seconds. Slurm job 4860245 completed normally in 7 min 55 s; Python process time was 468.35 s. A normal process exit does not mean task success.

## What was removed for this comparison

| Runtime choice | Our merged candidate | Hanson's exact published default |
| --- | --- | --- |
| Solver wrapper | SolverVBDChebyshev | SolverVBDCoarse, including its common-translation solve |
| Collision execution | VBD-owned pipeline | RSS external collision calls |
| Persistent self-contact anchors | Owned pipeline initialized | Not activated by this RSS route; detector-based self-contact remains enabled |
| Scene edge degeneracy tolerance | 1e-12 | No override; Newton default 1e-5 |
| Episode duration | 10 s | 20 s |

Both runs use the same source mesh, labels, material estimates, 5000 Pa.s viscosity, 20 substeps x 12 iterations, gripper geometry, 5 N configured jaw cap, X-carry trajectory and Newton commit. The longer baseline duration only extends settling; it cannot explain the pickup difference before 3 seconds. The initial collision setup still uses sticky rigid-soft matching in both runs. Main-derived coupled-backend code is also absent from the clean Hanson checkout, but those coupled routes were dormant in the candidate's single-VBD scene.

I selected Chebyshev in the merge candidate; that removed the published pick/drop default's coarse translation solve. The earlier candidate was consequently not a clean test of Hanson's published scene. This comparison does not isolate that solver choice from the collision ownership and tolerance changes.

## H1: remove our integration bundle

Recorded before execution: if the unchanged Hanson configuration lifts and carries while the candidate does not, our integration bundle matters and its individual differences warrant isolation. If the same acquisition failure occurs unchanged, our additions are not necessary for that failure. Either outcome alone cannot identify a particular contact mechanism.

Experiment: clean detached paired checkouts at RSS d50f86f51b31a1fbc42844ef78a4b851eb693159 and Newton 3d3f9e4c13fd0e60e7e53a9870946c0983d1d14c. Reused the ARM64 environment, Warp 1.18.0 and an allocated GH200. Invoked the original run with its default 600-frame duration and physics parameters, selecting only output directory and video frequency. No monkeypatches, solver edits, trajectory edits or material changes. Verified exact import paths/revisions and empty tracked diffs before and after execution.

**Verdict: PARTIAL.** The untouched configuration recovers lift, but the complete task still fails. Our changes are not necessary for premature loss, although the failed merged configuration lost contact earlier and never lifted. The coarse-versus-Chebyshev difference is a candidate explanation for the improvement, not an isolated causal result. A single run of each bundle does not establish robustness.

Existing diagnostics were postprocessed against the same criteria: minimum whole-object clearance at least 20 mm throughout 2.7–5.3 s, retention until opening, 80 +/-20 mm X carry during 3.5–4.7 s, final release, support within -0.5..2 mm, and peak centroid speed no greater than 5 mm/s. The baseline fails retention, sustained transit clearance, carry distance and settling speed. Its original success flag is also false. Mesh-quality and penetration measurements pass their existing limits.

## Evidence and limitations

[Video](hanson_baseline_20261103/overview.mp4), [native results](hanson_baseline_20261103/result.json), [comparison](hanson_baseline_20261103/comparison.json), [provenance](hanson_baseline_20261103/provenance.json), [log](hanson_baseline_20261103/slurm-4860245.log). Compressed diagnostics and original launcher/postprocessor are archived alongside them.

The unmodified example does not save particle/body states or peak jaw-drive readings; the 5 N value is the unchanged configured cap, not a newly recorded force measurement. Contact/penetration measurements are frame sampled and only cover detected contacts. Mesh quality is latched each substep. Unlike the candidate, this run did generate self-contact pairs (peak 46), but without the new persistent self-contact anchors enabled through an owned pipeline.

No additional trial, ablation, tuning or implementation change was made. Both main branches and the preserved candidate code remain unchanged. Recommended next scope, if requested: start from this clean Hanson baseline and isolate the loss near 2.53–2.57 s before reintroducing other changes. This comparison does not validate the rod scene or claim that every scene on Hanson's branch has the same failure.
