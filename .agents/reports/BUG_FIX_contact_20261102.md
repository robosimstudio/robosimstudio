# Clawhauser contact investigation — 2026-11-02

## Current result and retained changes

**The nominal standalone task passed in H15, job 4807226:** 477 frames / 15.9 s,
including pickup, hold, transport, supported lowering, deliberate release,
withdrawal and settling. Carry was 92.211 mm for the 80 +/- 20 mm criterion;
minimum transit clearance was 40.547 mm. Maximum recorded penetration was
0.036272 mm, final peak vertex-centroid speed 0.14633 mm/s, and both jaw drives
stayed within 5 N. No inverted or nonfinite tetrahedra were recorded at frames
or by the existing substep quality latch. Full results and the saved-state video
are under `outputs/debug/contact_20261102/agent_b/12_main_edge_epsilon` and
`agent_a/main_edge_epsilon_render/overview_saved_passed.mp4`, respectively.

The last correction was one native scene parameter: edge degeneracy tolerance
1e-5 -> 1e-12. Installed Warp compares it with squared edge lengths; the old
value incorrectly treated many millimetre-scale edges as points, admitting
false self-contacts and omitting valid ones. H14 had already released cleanly
but inverted after opening. H15 keeps that trajectory and all acceptance
thresholds, corrects the tolerance, and completes the full cycle. Earlier
acquisition/slip and placement-path failures required separate fixes below.

No Newton source changes, new solvers, contact welding, attachments or prescribed
object motion were introduced. The existing Hanson Chebyshev wrapper remains
present and masked during contact; the successful run uses native VBD updates.
This is one nominal-scene pass, not a robustness or policy-throughput claim.

The final H15 evidence is also archived beside this report for branch sharing:
[video](contact_20261102/overview.mp4),
[results](contact_20261102/result.json),
[configuration](contact_20261102/metadata.json),
[independent verification](contact_20261102/verification.json), and
[saved states](contact_20261102/states.npz).
`contact_20261102/diagnostics.json.gz` is a lossless compressed copy of the
recorded diagnostics. The directory also contains compressed support geometry and the rendering
manifest. Absolute paths in the metadata preserve the original run's provenance.
The original diagnostic driver, Slurm script and earlier exploratory artifacts
referenced below remain local outputs; the main example accepts `--steps 477`
for the same scene and duration.

| Change | Observed reason | Status |
| --- | --- | --- |
| Route the same scene friction to `particle_mu` and `soft_contact_mu` | ADMM was reading a different coefficient from native VBD | Correctness fix; ADMM trial still failed |
| Read ADMM-owned contacts from its own buffer | Main-buffer-only readout omitted coupling contacts | Diagnostic fix; no contact-law change |
| Carry along Y | Existing carrier cannot execute the old X command | Final run: 92.211 mm during the commanded carry phase; phase scoring is explained below |
| 80 rather than 20 substeps | Smaller timestep retained early acquisition longer | Integrated with other supported corrections |
| Full-surface contact | Saved mesh triangles crossed pads without a vertex contact covering them | Physical surface coverage corrected; alone insufficient |
| Friction epsilon 0.001 rather than 0.01 m/s | Measured low-speed contact creep fell about tenfold | Integrated; alone insufficient under later load |
| Scene viscosity 500 rather than 5000 Pa s | Lower loaded resistance and ring distortion permitted whole-object pickup | Integrated; explicitly changes physical rate response, not source E/nu/density |
| Measured static grasp offsets | Changed settling otherwise put both rim contacts at pad corners | Bilateral pickup/hold passed with the corrected pose |
| Supported laydown, stop before over-travel, then open and withdraw | Vertical lowering retained the ring on a post; continued closed-grip arc motion pushed it out early | Final run retains the grasp until opening and then releases |
| Native edge degeneracy tolerance 1e-12 instead of 1e-5 | Squared-length threshold collapsed millimetre edges to points, misclassifying junction self-contact | Same H14 trajectory now completes all 477 frames with valid mesh |
| 320 native iterations | Fixed-substep comparison showed incomplete convergence at 80 | Pickup/hold passed with original 5000 Pa s viscosity; expensive alternative archived, not integrated |
| Actual hold, clearance, rest-speed and substep-quality gates | Final release alone could mislabel a failed pickup as a completed task | Integrated; force and penetration limits preserved |

Generated face-contact compilation uses Warp optimization level 1 to avoid the
observed compilation stall. This is a compiler setting, not a numerical correction.
Experiment scripts and state analyses stay under ignored outputs; persistent code
changes are confined to three existing RSS files. The shared viscosity default
and other demos' material profiles are unchanged.

## Scope and baseline

User authorizes autonomous exploratory work until successful or a genuine blocker,
with only core functionality tests. Goal: contact-based standalone grasp, hold,
transport, release and supported settling. Arm construction, real-world calibration,
performance work and learned simulation are excluded. Temporary sticking is a last
resort requiring an explicit decision after ordinary contact methods fail. A large
solver/coupling rewrite must be reported before implementation.

RSS yiduo/dev starts at f1b648c, including Hanson 538de6c. Newton sibling is
d960ec3b (1.8.0.dev0), Warp 1.18.0, existing ARM64 environment on allocated GH200.
Prior local release edits remain in stash 076f699 and are not the active baseline.
Source Clawhauser: one labelled 1,840-vertex / 6,850-tetrahedron mesh, original
stiffness/density estimates. Keep these and current force limits fixed initially.
Current pick_drop defaults to one Chebyshev VBD island; on_rod defaults to
MuJoCo + ADMM. Historical README success claims are not current-run evidence.

Root owns integration and GPU scheduling. Agent A owns rss-contact-a friction /
coupling investigation; Agent B owns rss-contact-b collision / release. Reuse
diagnostics; no broad regression suites. Each experiment needs a distinguishing
prediction; no blind parameter sweep or new controller framework.

## Acceptance and sequence

1. Reproduce current pickup and hold, with actual active contacts and motion.
2. Separate holding/friction, geometry/detection and release causes.
3. Retain small supported corrections, one at a time.
4. Complete transport, release and supported table settling in the nominal scene.
5. Save the video and numerical evidence, report limitations, and stop after this
   core task passes. Additional placement variants and robustness sweeps are
   deferred to avoid expanding this contact investigation after nominal success.

Task and physics validity are separate: finite state, measured penetration,
substep mesh-quality diagnostics and preserved force limits must be reported.
Do not call an episode successful merely because its final pose looks correct.
Successful free release does not establish stable table impact. Candidate counts
do not establish force-bearing contacts. Current example scoring is reused, with
its quality outputs independently checked rather than silently ignored.

## H0 — current holding baseline

Before execution: run the unmodified pick_drop through 3.5 seconds (105 frames,
close/lift/initial stationary hang) with default VBD, then the same command with
existing --claw mujoco --coupler admm. All source mesh/material/trajectory and
limits remain unchanged. This is a comparison of existing configurations, not
an isolated proof of friction causality. No release is expected within this prefix.

Predictions: measured body lift and active pad contacts identify whether each
configuration actually gets to a hold. Zero/vanishing pad contacts favors failed
acquisition or lost contact over release-specific explanations. Retained pad
contacts with continuous descent motivates slip/load diagnostics. Invalid states
or mesh damage must be identified before interpreting friction. A short prefix's
full-task success flag is expected to be false and is not itself the diagnosis.

Evidence: outputs/debug/contact_20261102/00_vbd_prefix and 01_admm_prefix.
VBD verdict: pickup failure reproduced (job 4774119, 105 frames). Lowest-point
lift is 0.0593 mm, active finger contacts peak at four and disappear during lift,
last-contact horizontal displacement is -0.818 mm. Mesh stays valid throughout:
minimum J by part [0.8936, 0.6171, 0.9295], zero inverted or nonfinite tet-substeps.
Maximum recorded penetration is 0.201 mm. This is acquisition/holding failure,
not a release or impact failure. The printed `released: true` merely means no
contacts in the final window; `stop_reason: not-lifted` correctly distinguishes it.

The redundant root ADMM run was cancelled after VBD completion. Agent A's
instrumented ADMM control provides that baseline with actual coupler contacts,
which the existing pick_drop readout incorrectly omits. Root default run uses
the main VBD buffer correctly; its failure is not that ADMM diagnostic issue.

## Source-confirmed issues pending runtime validation

- ADMM reads `particle_mu` (default 0.5), while the heterogeneous entity sets
  `soft_contact_mu=0.9` only. At pad mu=4, paired coefficients differ 1.414 vs
  1.897. A matched .5/.9 test checks impact on the current failure.
- Cross-island finger contacts live in ADMM buffers, not the main pipeline
  buffer. Agent B has a diagnostic-only patch; no changed physics or scoring.
- The gripper has world-y and world-z carrier travel, while pick_drop commands
  an x carry. This cannot execute the requested transport; use the existing y
  travel once pickup is established rather than adding a new mechanism.
- Source geometry starts the ring 32–43 mm above the bench, with pad faces at
  1–13 mm. The ring moves while settling, so source geometry alone does not
  establish a missed grasp. Agent B records actual particle/finger poses through
  closure to distinguish pad acquisition from posts touching the ring.

Agent probes: A job 4774214 (45-frame ADMM mu controls); B job 4774806
(45-frame unchanged VBD acquisition with state/contact identities). These are
core phase checks, not full-cycle retries or broad tests.

## H1 — collision coverage versus slipping acquisition

Agent B's unchanged 45-frame VBD probe exactly reproduces root frames 1/16/31.
Both pads (shapes 2 and 4) acquire the ring; no posts are involved. Saved-state
box-distance measurements agree with the detected contacts. Contacts decrease
4 -> 3 -> 2 -> 1 -> 0 over frames 41–45 while the fingers slide upward past the
ring. The last right contact reaches 5.857 mm below the pad centre, near its
6 mm bottom edge. At frame 45 both pads have geometrically cleared the ring.
No inverted/nonfinite tet-substeps occur; minimum J is 0.6171. Thus a missing
initial grasp or missed vertex detection is not supported for this prefix.
The observed failure is grasp slip during lifting, not retained release.

ADMM control at frame 16 also has bilateral pad contacts. Actual group mu is
1.4142. Applied coupling force agrees with W*lambda to about 3e-5 N at that
sample. Contact penetration exceeds 1 mm, so a retained grasp alone would not
pass the existing 0.5 mm penetration gate. Native baumgarte defaults to 0 in
pick_drop, while rod sets 0.1; this is a possible later targeted correction,
not yet an executed experiment or established fix.

## H2 — timestep sensitivity of the VBD slip

Recorded before execution by Agent B. Given actual pad acquisition and nominal
friction capacity above object weight, test the numerical solve before changing
materials or geometry: change only SUBSTEPS 20 -> 80, retain 80 VBD iterations,
original physical timing, material map, geometry, contacts and force budgets.
Run the same 45-frame prefix with existing quality/contact/state measurements.
Improved relative pad-ring retention would support timestep/solve sensitivity;
the same escape would reject this as a sufficient fix. Contact depth, actual
carrier lag and mesh integrity distinguish an apparent improvement caused by
incorrect trajectory execution or excess deformation. Job 4775655, output
agent_b/01_vbd_substeps80; no retained physics-parameter edit yet.

## Small source corrections independent of the lift investigation

The carrier has world-y and world-z translation; x is jaw separation only.
Change the requested carry to the existing y axis, preserving its 80 mm length,
and measure that axis in the result. This cannot affect the close/lift prefixes.
Validate the actual carrier/object displacement when a grasp reaches transport.

The existing judge computes `not-lifted` and substep mesh damage but omits them
from `success`. Independent review caught that its unused lift threshold (half
the pad travel, 120 mm) is inconsistent with the hanging geometry: 248 mm final
pad height minus the 198 mm toy predicts about 50 mm tail clearance. Before any
successful full run, define actual pickup as at least 20 mm whole-object table
clearance. Require this and substep integrity in addition to all prior success
gates. Check continuous clearance through hang/carry in the phase evidence.
Also require the existing table band throughout the final half-second and centre
speed <=5 mm/s, computed from consecutive existing samples so oscillation cannot
cancel in an endpoint displacement. These are scoring corrections only; no new
state measurement or changed dynamics. Runtime validation remains pending.

H2 early verdict: PARTIAL. At 1.5 s, 80 substeps retains bilateral pad contacts
(3 vs 0); ring-centroid rise from 1.1 s is 14.43 mm vs 1.19 mm. Relative vertical
pad/ring slip falls from 19.57 to 7.77 mm. The whole toy still touches the table,
so this is not yet pickup. Minimum J is 0.338 vs 0.617, with zero inverted or
nonfinite tet-substeps. Continue the same candidate through 3.5 s/105 frames
(job 4777200, agent_b/02_vbd_substeps80_hold); no additional physics changes.

Read-only native solve audit: mapped source elasticity and mass are correct.
Tet damping is objective deformation-rate viscosity, not world drag, but its
local 1/dt Hessian worsens block conditioning. Rest-state median local block /
inertia ratios are approximately 22,625–35,986 at 20 substeps versus 4,188–6,116
at 80. This motivates the native accuracy test, not a new solver. Geometric
table-penalty force estimates also fall substantially; they are not measured
time-integrated forces and are not alone proof of a force-transfer defect.

## Friction-routing comparison verdict

Agent A's matched ADMM .5/.9 pair completed (job 4774214, exit 0). The requested
coefficient reaches contact groups: paired mu changes from 1.4142 to 1.8974.
Both lose the grasp: last contact is frame 42 versus 43, only 33 ms later.
Peak sampled upward coupling force rises from 4.947 to 6.273 N, while maximum
finger penetration is 1.723 versus 1.747 mm (both fail the 0.5 mm gate). Maximum
jaw drive remains below 4.501 N; minimum body J is 0.617, no inversion/nonfinite.
Verdict: routing defect CONFIRMED; sufficient explanation/fix for pickup REFUTED.
Evidence: agent_a/mu_comparison.json and each run's states/diagnostics/metadata.

Retained corrections: both friction fields receive the existing scene value;
the rod's existing override also updates both. No source material estimate was
changed. Agent A independently verified Agent B's ADMM finger readout against
owned coupler contacts on all 45 frames. The integrated helper additionally
returns all owned depths, including carrier contacts, for the existing overall
penetration gate. Source compilation/diff checks pass; final integrated runtime
validation remains pending. Root's initial `python` command was unavailable on
the login node; the source check was rerun successfully with `python3`.

Native force routing exists from the ADMM buffer to particle_f to VBD's inertial
target. Each ADMM iteration restarts the FEM subproblem from the substep input;
five coupler iterations are not 400 continuous VBD convergence iterations. Thus
raising the coupler iteration count is not the next proposed fix. Prioritize the
native timestep candidate already showing better contact retention.

## H2 continuation — insufficient for a complete lift

The 80-substep continuation lost all pad contacts at frame 54 / 1.8 s; it never
lifted the whole object clear. Stop the remaining open-loop prefix after this
reproduced failure; SIGINT preserved 68 frames in agent_b/02_vbd_substeps80_hold.
Minimum J remains 0.338, zero inverted/nonfinite tet-substeps. The ring rose by
up to 42.9 mm, but the lowest object point stayed within 0.676 mm of the table.
Verdict: timestep sensitivity confirmed, timestep change alone insufficient.

## H3 — missed surface contact near the pad edges

The earlier geometry check covered vertices, not the interiors of boundary
triangles. B's saved-state triangle/box check finds physical surface intersections
with the pads at frames 48–52, followed by a genuinely clear object at frame 54.
The contact points also approach the pads' rear edges; this is not pure vertical
sliding. Independently validate the surface intersection and its depth, then test
the existing full_surface_contact flag at the same 80 substeps, with everything
else unchanged. Prediction: barycentric surface contacts appear and prevent the
same missed overlap/edge escape. If the toy still slips with those contacts
resolved, inspect grasp alignment or native convergence rather than declaring
collision coverage a sufficient fix. No new collision algorithm or solver.

Independent confirmation: A rebuilt the global source boundary (2,356 faces,
887 ring faces) and clipped polygons against saved pad OBBs. Its face sets match
B's check exactly at frames 48–54. At frame 48, a left triangle has a point at
least 0.308 mm inside the physical pad, while the vertex contact depth is only
0.073 mm. This proves missed surface coverage, not yet causality for grasp loss
or a violation of the 0.5 mm gate. Approve a single-flag full-surface comparison
through 2.1 s / 63 frames at 80 substeps and 80 iterations; check that actual
barycentric surface contacts are generated. Keep main's default unchanged until
the result supports a correction. Native iteration changes remain in reserve.

Full-surface trial is job 4779543, agent_b/03_vbd_full_surface. Its actual contact
buffer flag is true. Source compilation precedes the first physical frame.

Alignment backup, not yet a trial: using the fitted hole and outer profile rather
than the stem-biased vertex centroid, A finds the loop shifts rearward before
contact. Acquired points sit near pad y=-5.1/-5.6 mm inside the 6 mm half-depth.
At frame 10, a band 4 mm behind the current aim spans 37.65 mm at an accessible
height; the current centre band spans 28.55 mm versus the 34.26 mm closed inner
gap. Thus a small jaw-y offset is a concrete trajectory alternative if surface
coverage alone fails. Do not substitute the ring vertex centroid for hole centre.

Y-carry scoring measures movement from the beginning of the carry phase, not the
initial lying pose: natural swinging into a hanging pose also moves the centre
along y and must not count as commanded transport. Jaw-force peaks already
recorded by the controller are included in the result and checked against 5 N.

H3 execution note: job 4779543 made no physical frames after 8m03 of face-kernel
compilation and was canceled. Preserved stock_release_20261028 report lines84–89
documents the same compiler problem on this Newton revision and Warp's public
`wp.set_module_options({'optimization_level': 1}, module='soft_face_contacts_False')`
workaround. Replacement job 4780794 uses that option only for the face module;
algorithm, solver source and cache contents are unchanged. Evidence is under
agent_b/03_vbd_full_surface_o1. Default-compiler parity remains unverified.

## H4 — a small independent grasp-alignment comparison

While H3 retries compilation, use A's measured accessible-band geometry to test
a single -4 mm gripper-y offset at the same vertex-only 80-substep/80-iteration
baseline. The object pose, jaw shape, materials and force caps stay fixed. Run
through 2.1 s, preserving the existing state/contact/quality evidence. Prediction:
contacts move away from the rear pad edge and remain acquired through the former
loss time. Persistent escape would reject this offset as sufficient. This is
parallel to H3 because acquisition alignment and surface coverage are separate
observed deficiencies; do not attribute a combined change to either alone.

H3 verdict: coverage correction CONFIRMED, sufficient pickup fix REFUTED.
Job 4780794 completed 63 frames. Actual edge/face contacts were present, and
independent boundary/box checks found zero physical pad/triangle intersections
at frames 44–54, versus several at frames 48–52 without full-surface contact.
The last contact was nevertheless frame 52 (baseline 53). Whole-object clearance
never exceeded 0.697 mm. Maximum recorded pad penetration was 0.038 mm, jaw
drives 3.59/3.64 N, minimum J 0.335, with no invalid tetrahedra. Contacts still
approached the rear pad boundary. Evidence: agent_b/vbd_full_surface_escape.json
and agent_b/vbd_full_surface_escape_intersections.json.

H4 verdict: the -4 mm offset alone is insufficient. Job 4780944 stopped after
58 frames with actual post-escape clearance. Last contact was frame 56, only
100 ms later than the matching baseline. Lowest-point rise was 0.0566 mm,
maximum penetration 0.0843 mm, jaws 4.999/4.997 N, minimum J 0.338, no invalid
tetrahedra. The right contact remained near the rear edge. Do not retain this
offset merely for a small delay. Evidence: agent_a/02_vbd_y_minus4 and
agent_a/y_minus4_contact_geometry.json.

## H5 — native convergence at a fixed physical step

Job 4782789 replays the exact 80-substep/80-iteration baseline through frame 40.
Only the first substep of frame 41 uses 320 native iterations, recording the
80/160/320 iteration states at the same inertial target. Remaining substeps
stay at 80. A meaningful additional displacement would establish insufficient
convergence of the coupled native VBD step (both rigid and particle sweeps),
not FEM alone. A plateau would weaken this explanation. No solver source,
equations, force instrumentation framework or source material changes.

Read-only friction context: native epsilon is 0.01 m/s, giving a 4.167 micrometre
tangential displacement transition at dt=1/2400. Below it the Coulomb-force
fraction is 2r-r^2, r=slip_speed/epsilon. The frame-40 depth-based normal-load
estimate would need about 0.706 mm/s creep to support the weight ideally. The
19.4 mm/s pad/ring-centroid relative motion over 0.4 s is NOT contact-point slip:
reconstruction at the two actual contacting vertices gives 2.10/1.46 mm/s at
sweep 80 and 1.84/1.30 mm/s at 320. Those contacts remain in the regularized
branch (about 38%/27% and 33%/24% of Coulomb capacity). Thus the centroid figure
does not rule out regularization as a contributor. These are geometric
estimates, not a force balance. Reducing epsilon also increases the near-zero
tangential Hessian; leave it unchanged pending convergence and alignment results.

H5 probe verdict: underconvergence CONFIRMED at the sampled loaded step. Job
4782789 completed in 8m16. Incoming positions exactly match B frame 40. Centre
z gains incoming-to-80 / 80-to-160 / 160-to-320 were 1.784 / 1.388 / 2.097
micrometres. All 1,840 positions changed; target and collision reference stayed
identical; truncation factors were all one. Ring z gained 4.575 micrometres from
80 to 320 while pads gained 0.566, reducing relative slip in that substep from
12.63 to 8.62 micrometres. Mesh stayed valid. This supports a native iteration
phase test, not a claim of successful manipulation. Evidence:
agent_c/00_native_iteration_probe/comparison.json.

Next bounded H5 check: replay the full-surface control through lift start at
80 iterations, then use 320 native iterations during the loaded phase. Keep
all physical parameters and the original pose/trajectory. Inspect through
1.8 s; stop after escape or invalid mesh, and only continue through 3.5 s hold
if healthy. Using full-surface coverage avoids repeating the already proven
surface-detection defect. No change to viscosity: source history shows the
current 5,000 Pa s is a scene solver setting (commit 56054df), but changing it
would alter physical deformation-rate response as well as conditioning.

H5 phase outcome: the 320-iteration original-pose trial reaches frame 54 with
zero finger contacts, at essentially the control's loss time. Early rise was
larger (frame-40 centre gain 2.608 versus 0.938 mm), but that does not establish
a usable grasp improvement. Preserve job 4784521 evidence under
agent_c/01_loaded_iterations320 and stop at the failed checkpoint. Do not retain
the higher iteration cost on this result alone; inspect final contact geometry
to determine whether the original alignment remained the limiting factor.

Final H5 geometry confirms the original alignment bottleneck: both cases have
last contact at frame 52; the 320 case ends with right-pad local y=-6.565 mm,
z=-0.035 mm, leaving the rear edge while still at mid-height. Control ends at
y=-6.552 mm, z=+0.094 mm. Maximum centre rise is 14.30 versus 12.91 mm, whole
object still table-supported. Maximum penetration 0.0333 mm, peak jaw 3.65 N,
minimum body J 0.3367, no invalid tetrahedra. Job completed in 19m13. This
does not test increased iterations on the corrected pose. Evidence:
agent_c/01_loaded_iterations320/comparison.json and states.npz.

## H6 — align the grasp with the settled loop

The -4 mm test did not centre both contacts. From full-surface precontact frame
10, the projected hole is at y=-16.862 mm. Within its 12 mm pad-width band,
the outer left/right rim points have z=0.765108/0.765265 m. Test one measured
pose: gripper y=-17 mm and pad centre z=0.7652 m, keeping source object pose,
open/closed gaps and force caps. The existing builder reduces lift by 8.2 mm
to preserve final carrier height. Open-pose mesh/box checks find no intersection;
loop clearance is at least 12 mm from pads and 18 mm from posts. Use the now
verified full-surface coverage and 80 substeps/80 iterations. Stop after escape
or a quality failure; extend only through the 3.5 s hold if acquisition remains
healthy. Evidence before launch: agent_b/measured_grasp_pose_candidates.json
and agent_b/measured_grasp_open_clearance.json. This tests measured pose, not a
material or force-cap change.

H6 verdict: acquisition correction CONFIRMED; complete pickup still FAILED.
Job 4783634 stopped normally after two clear frames, saving 69 frames. Last
contact was frame 67 versus 52 in the full-surface control. Contact points now
start on the pad faces rather than rear corners. The remaining escape is
predominantly downward slip: right local z moves from -4.88 mm at frame 61
to -5.39 at 63 and -6.00 at 65; left reaches the bottom edge at 66–67; both
are clear at 68. Ring rise reached 152.3 mm but whole-object clearance never
exceeded 0.698 mm. Boundary clipping found zero physical pad/post intersections
at frames 60–69. Maximum recorded contact depth was 0.030 mm, jaw drives 5 N,
minimum J 0.334, no invalid tetrahedra. A independently reconstructed the same
barycentric point path. Evidence: agent_b/vbd_measured_pose_escape.json,
agent_b/vbd_measured_pose_escape_intersections.json, and
agent_a/measured_pose_contact_geometry.json. Keep this pose experimental until
loaded retention passes. Compare native convergence next; reconstruct actual
contact-point tangential motion from saved states before any friction-epsilon
trial. Do not substitute centroid motion for friction slip speed.

## H7 — native low-speed friction smoothing

B reconstructed each current barycentric point at current and previous saved
states, subtracting the motion of the same body-attached pad surface point.
These are 30 Hz interval means, not physical-substep velocities. At frames
59–64, median tangential speeds are 6.2–8.2 mm/s on the left and 7.6–8.7 on
the right, predominantly downward. All sampled records in those frames are
below the native 10 mm/s regularization threshold, although the right side
earlier reaches 24–28 mm/s. Evidence:
agent_b/vbd_measured_pose_contact_slip_velocity.json.

Test one existing parameter: friction_epsilon 0.01 -> 0.001 m/s on the measured
pose, full-surface, 80-substep/80-iteration configuration. Prediction: less
loaded creep and longer retention. Keep mu, source materials and force limits
unchanged; no temporary welding or equation changes. Use the constant setting
from build, so startup/table/self/rigid friction smoothing changes too. Check
actual acquisition geometry before attributing any outcome purely to loaded
creep. Same early failure stop and maximum 3.5 s hold. No value sweep. This runs
in parallel with the isolated native-iteration comparison.

Interpretation caveat: 6.2–8.7 mm/s already gives approximately 86–98% of
the Coulomb limit under the current smooth law. Any large benefit may come
from earlier low-slip acquisition/loading, not the last few escape frames.
At epsilon=0.001, the displacement threshold is 0.417 micrometres at dt=1/2400,
about seven world-z float32 spacings. Job 4786110 uses the existing constructor
parameter and records the actual solver attribute; output agent_b/05_vbd_epsilon001.

H7 completed with failure at frame 72 after two clear frames. Last contact was
70 versus 67 in the measured-pose control, only 0.1 s later. Whole-object
pickup did not occur. Acquisition used the same rim vertices with less than
0.3 mm pre-lift centre difference, so there was no large startup re-aiming
confound. Reducing friction smoothing is insufficient; do not sweep epsilon
values or retain this change based solely on the small delay. Saved terminal
geometry is being compared before the next trial.

H7 mechanism verdict from saved states: low-speed-creep improvement CONFIRMED,
complete pickup still FAILED. During frames 59–64, left vertical contact creep
fell from 5.9–8.0 to 0.47–0.53 mm/s; right from 7.2–8.3 to 0.75–0.80 mm/s.
The right patch remains around local z=-1 mm at frame 64 versus -5.66 mm in
the control. It then rapidly slips: right vertical speed rises through 2.6,
10.1, 19.4, 40, 66 and 152 mm/s at frames 65–70. Thus the small retention-time
gain hides a real reduction in prior creep and a later load-related failure.
Maximum penetration 0.019 mm, jaws 5 N, minimum J 0.334, no invalid tetrahedra;
whole object never clears the table. Evidence: agent_b/vbd_epsilon001_escape.json.
Retain epsilon=0.001 in the experimental baseline because of the measured creep
reduction, not merely the 0.1 s retention gain. No main configuration change yet.

Load context: depth-based normal estimates are about 4 N per jaw and nominal
Coulomb capacity about 15 N at frames 59–64, declining toward 14.25 N at 67.
Carrier drive minus rigid gravity/acceleration suggests 12.5–14.1 N external
vertical load, much greater than the object's 1.227 N weight. These estimates
are not converged solver reactions and can include constraint residuals; they
motivate checking viscous response/convergence, not a claim of a force bug.
Evidence: agent_b/vbd_epsilon001_load_context.json.

## H8/H9 — loaded response after acquisition and creep corrections

Use B05 (measured pose, full-surface/opt1, epsilon=0.001, 80 substeps, 80 native
iterations, viscosity=5000 Pa s) as the next control. Two independent existing
settings address the remaining loaded slip without new equations or solvers:

- H8: constant 320 native iterations from build. Unlike the previous diagnostic,
  this does not introduce a phase-dependent solver setting. Source physical
  parameters remain unchanged. Check whether startup changes the acquired patch;
  then test retention/clearance through the 3.5 s hold, stopping early on failure.
- H9: constant tet viscosity 500 rather than 5000 Pa s, using the existing scene
  spec argument, with 80 iterations. This changes physical deformation-rate
  response as well as conditioning; it is NOT just an accuracy adjustment.
  The viscous local coefficient 2*d/dt falls from 24 to 2.4 MPa, nearer the
  body's elastic lambda+mu of 1.724 MPa. Source E/nu/density and force caps stay
  fixed. First independently check H7's terminal surface coverage, then launch
  only if no newly missed collision explains the failure. Check acquisition
  geometry and all existing quality limits. No viscosity grid or shared-default
  change, and no source solver changes.

Both are output-local core manipulation trials, with early failure stops and
at most 105 frames. A passing prefix is required before a full-cycle validation.

H7 terminal coverage independently cleared: A clipped all 2,356 global exterior
faces against both pads and posts at saved frames 65–72 and found zero physical
intersections. Both contacts leave the pad bottoms; their y positions remain
within pad width. This cannot exclude unsaved-substep overlap, but supplies no
new missed-surface explanation. Evidence:
agent_a/epsilon001_terminal_surface_validation.json. H9 may proceed.
H8 is job 4788239, agent_c/02_measured_pose_epsilon001_iterations320, with actual
constant 320 iterations and epsilon=0.001 verified from the runtime object.

H9 is job 4788459, agent_b/06_vbd_damping500; runtime metadata confirms all
tet damping values are 500 Pa s. At frame 16 its centre differs from B05 by
about 7 mm (y +6.22 mm, z +2.96 mm), and it acquires different bilateral rim
patches. Minimum J is 0.623 versus 0.334, no invalid tetrahedra, jaw peaks
3.83/4.82 N, maximum penetration 0.020 mm. Interpret any later outcome with
this changed settling/acquisition geometry; it is not an isolated loaded-rate
comparison.

H9 outcome: transient whole-object pickup, stationary hold FAILED. At frame 76
clearance reaches 35.06 mm, and at 91 it is 38.91 mm, with one remaining left-pad
face contact [392,434,410] (all ring vertices). Contacts disappear at 97–98,
before carry or opening; the trial stops at 98 / 3.267 s. No inversion/nonfinite
state; minimum body J 0.623, ring J 0.975, jaw peaks below 4.90 N. The lower
loaded force and reduced ring compression are useful observations but do not
establish a completed grasp. Saved acquisition/support geometry is being audited
before any retry. If changed settling misaligns the old grasp, derive a measured
pose correction; do not introduce another solver/material knob.

H9 saved-state classification confirms misalignment after changed settling.
At frame 31 both contacts are at front (+y) pad corners, median y=+6.22/+6.23 mm.
By 68 the left contact reaches the upper-front corner; frame 72 briefly touches
the left post. Frames 74–96 are left-pad upper/front/inner-corner support,
typically local [3.2,6.4,6.5] mm, followed by loss at 97. This is genuine contact
support, not a pure friction hold. Peak whole-object clearance is 58.46 mm;
maximum penetration 0.046 mm. Startup shape RMS differences from B05 are
12.1 mm at frame 10 and 10.8 mm at 31, much larger than centre differences.
Evidence: agent_b/vbd_damping500_support_geometry.json. Derive one fresh grasp
pose from the new precontact geometry; keep H9 physics, gaps and caps fixed.

Core output preflight: main's unchanged renderer/VideoSet path passed one
control frame, job 4788570, exit 0 in 35 s. MP4 H.264/yuv420p 960x540, 60 fps,
two encoded frames as expected for one 30 Hz control frame; video_error=null.
The existing GH200 tiled fallback renders collision geometry. No renderer
changes or visual polish. Evidence: agent_a/render_preflight, including
video_metadata.json and inspected first_frame.png.

Existing-accelerator audit (no trial): Hanson's Chebyshev mask can be disabled
through its existing option, but the blend extrapolates positions after native
penetration-free truncation without reapplying that truncation. There is no
energy/residual rejection, adaptive restart or inversion rollback. The contact
mask is its only contact safeguard; rho changes with the mask enabled cannot
help this contact-rich scene. Do not bypass that safeguard or add a new guarded
accelerator as part of this task. Continue the native-setting comparisons.

## H10 — re-centre the grasp after H9's changed settling

The H9 last sampled contact-free frame is 9 (first contact 10). Its projected
hole y=-6.954 mm and outer-rim midpoint x/z=(0.475200, 0.760725) m motivate
one static grip centre (0.4752, -0.0070, 0.7607) m. This changes x/y/z by
(+1.861, +10.0, -4.5) mm from H9, preserving final pad height, gaps, force
caps and all H9 physics. Frames 5–9 still oscillate; measured rim z ranges
759.4–765.0 mm, within the proposed pad face. Independent whole-surface
clipping confirms source/frame-9 open pads/posts/carrier are clear. A's
pad-relative rim y/z values are (-2.707,-0.947) mm left and (+0.105,+0.997)
mm right, inside the +/-6 mm faces. This is a measured correction to the
observed corner grasp, not a new material or solver sweep. Verify actual
acquisition geometry again because a different pose changes first-contact time.
Evidence: agent_b/h9_remeasured_pose_proposal.json and
agent_a/h9_pose_independent_review.json.

Initial H10 job 4792013 was invalidated immediately: the output-local wrapper
used the world x as a spec-local x, adding RSS's +0.6 m scene transform twice.
It was stopped before pickup. Correcting the same approved pose requires adding
+1.861 mm to the existing spec x, with a runtime world-centre assertion before
simulation. Preserve the invalid run separately; it is not physics evidence.

H8 frame 33 remains a valid centred bilateral acquisition. Constant 320 iterations
changes the settled shape (3.195 mm RMS, max 5.761 mm versus B05); pad-local
median y/z is (-1.654,-1.677) mm left and (-0.104,+2.582) mm right. Interpret
later retention with this startup difference. No post contacts or quality failure.
Evidence: agent_c/02_measured_pose_epsilon001_iterations320/startup_comparison.json.

H10 corrected job 4792320 PASSED the 105-frame pickup/hold prefix. Actual world
centre was asserted before stepping. Minimum whole-object clearance during
2.7–3.5 s is 40.550 mm, final 47.307 mm, and all hold frames retain both pads.
Maximum recorded penetration is 0.020412 mm; jaw peaks 5/5 N; minimum J by
part [0.9771,0.6227,0.9704], zero inverted/nonfinite tet substeps. At acquisition,
both patches are on inner x faces (local y approximately +2.7/+2.4 mm and z
+4.0/+3.8 mm). During hold the left moves toward its front edge, while the right
stays central; no post contact or single pad-top hook explains retention.
Evidence: agent_b/07_vbd_damping500_reaim, agent_b/h10_hold_summary.json,
agent_b/h10_support_geometry.json. This is a prefix pass, not a complete task.

Main now contains this candidate's exact small scene changes: 80 substeps,
80 native iterations, friction_epsilon=0.001 m/s, full-surface contact, generated
face kernel compiler optimization level 1, scene viscosity 500 Pa s, pad bottom
clearance 4.7 mm and measured x/y offsets (+1.8613155,-7.0) mm. Final carrier
height, source geometry/labels/E/nu/density, gaps and 5 N jaw limits are unchanged.
The shared viscosity default is unchanged. Independent configuration review
precedes a single 300-frame (10 s) full-cycle test using main's Y carry and exact
judge, with 4.3 s available after opening. The output-local probe reuses saved
states/contact identities for release diagnosis and retains core early failure
stops; no new solver, controller or testing framework.

H9's failed pickup/hold has also been rendered from its existing 98 saved states,
without any physics steps: agent_a/h9_saved_state_render/overview.mp4. The
manifest identifies it as saved-state rendering. It visually confirms suspension
and terminal loss; exact contact classification comes from numerical geometry.

H8 FINAL: the independent constant-320-iteration / 5000 Pa s alternative also
PASSED all 105 pickup/hold frames, job 4788239 exit 0 in 85m10s. Whole-object
clearance exceeds 20 mm from frame 70; minimum hold clearance 37.880 mm,
final 46.054 mm. Maximum recorded penetration 0.007809 mm; minimum body
J 0.456843, zero inverted/nonfinite samples, 5 N jaw caps. Both pads remain
in contact throughout lift and hold. Independent geometry confirms an inner-face
pinch without post or pad-top support; the right patch reaches its rounded rear
edge by 105, so grasp margin remains limited. Startup geometry differs from the
80-iteration control, so loaded convergence is not isolated by this whole-prefix
comparison alone. It complements the earlier fixed-substep convergence evidence.
Carry, release and landing are untested for this alternative. Archive it; no new
320-iteration trial. Main continues on the selected H10 500 Pa s / 80-iteration
configuration. Evidence: agent_c/02_measured_pose_epsilon001_iterations320/
final_comparison.json, states.npz, startup_comparison.json, and
agent_b/c_h8_support_review.json.

## H11 — full cycle releases correctly, then fails on secondary body impact

Job 4795749 stopped at frame 176 / 5.867 s. Pickup/hold/transport succeeded:
minimum transport clearance 40.577 mm, actual Y transport 85.386 mm,
held_until_open=true, 5 N jaw caps. Last finger contact is 164 (5.467 s),
and frames 165–176 are clear. At 165 all exterior triangles are conservatively
at least 2.019/2.162 mm from left/right pads (greater than the 0.7 mm particle
radius), posts >8 mm, carrier >24 mm. The ring genuinely releases.

Tail (part 0) touches the table at 167 while the body is still 56.1 mm above it.
The tail rebounds at 170; the toy tips onto its main body at 175. Inversion begins
at 176: body tet 6686 has saved J=-0.185 and latched substep minimum J=-0.476,
with 51 inverted body-tet substeps. Maximum recorded contact penetration is only
0.02121 mm. This is secondary body impact after release, not finger retention.
The old judge's released=false/stop_reason=held on this aborted run is a truncated
half-second-window artifact: only 12 clear frames were available before the mesh
failure. Success remains false for the correct physics reason. Future early
failures should prioritize invalid-mesh as the stop reason without relaxing any
gate. Evidence: agent_b/08_main_full_cycle, agent_b/full_cycle_release_impact_analysis.json,
agent_a/full_cycle_release_impact_validation.json. Saved-state video (zero physics
rerun): agent_a/main_full_cycle_render/overview_saved_failed.mp4.

## H12 — support the main body before opening

One bounded trajectory retry, with H10 scene physics/grasp/caps unchanged.
Preserve the entire successful motion through 5.3 s; then lower 103 mm using the
existing quintic smooth() over 5.3–6.8 s, pause for support 6.8–7.3 s, and open
7.3–7.7 s. Final pad centre z=0.8949296 m; peak commanded lowering speed
0.12875 m/s and acceleration 0.2643 m/s^2. The measured main-body lower-edge
clearance before opening was 101.24–104.85 mm (median 103.02), whereas tail
clearance alone was about 41 mm. Thus this targets body support rather than
merely repeating the observed unstable tail touchdown. Physics must determine
whether the body actually settles into support; the trajectory prescribes only
the gripper, never object motion.

Run 360 frames / 12 s, preserving the previous 4.3 s after opening. Keep the
20 mm clearance requirement over the exact original transport window 2.7–5.3 s;
intentional lowering begins afterward. Extend continuous grip through the new
opening time 7.3 s. Measure the same 80 +/-20 mm Y transport over the stated
commanded carry phase, 3.5–4.7 s, so subsequent swing and placement reorientation
are not counted as commanded transport. Also retain the pre-lowering movement
at 5.3 s as a diagnostic; do not conceal the pause's overshoot.
Keep final table-support/rest-speed, penetration, force, and zero-inversion /
nonfinite limits unchanged. Use existing saved states to distinguish body support
from tail-only contact during the pause. No new feedback controller, solver,
material tuning or infrastructure. Independent diff review precedes launch.

Carry-measurement decision before the retry: the saved free-drop trajectory gives
93.067 mm at the commanded carry end (4.7 s), 104.519 mm at the lower-start
boundary (5.3 s), and 85.386 mm at the old last-held sample (5.467 s). The earlier
proposal to score 5.3 s would fail the unchanged 60–100 mm interval because of
pendulum phase. Use the explicitly commanded phase endpoint, not a chosen instant
within the oscillation, and report the 5.3 s value separately. This is a disclosed
phase-definition adjustment; force, clearance, grip, mesh and settling thresholds
remain unchanged. Placement accuracy after lowering is not being optimized.

H12 final diff/driver independently approved; job 4802149 runs exactly 360 frames
with explicit phase/config assertions, output agent_b/09_main_placement. Startup
matches H11 frame 1 exactly. Dependent saved-state render job 4802196 writes
agent_a/main_placement_render using current phase times and no physics steps.

Read-only H11 impact audit: the established construction masks Chebyshev
extrapolation off. Soft-contact candidate buffers remain nonempty at saved
frames 159–176, which globally masks acceleration even when active force is
zero. In addition, RSS's external contact buffer lacks the self-contact struct
while self-contact is enabled, triggering the unconditional global-mask fallback
in chebyshev_acceleration.py. This is a source-path conclusion; the GPU mask was
not recorded. Thus release is not evidence of acceleration turning on.

Tet 6686 is a correctly oriented body surface tet beside the ring interface:
rest volume 0.865 mm^3, minimum altitude 0.874 mm, normalized shape quality
0.648 (approximately the lowest 2.8% of body tets, not degenerate). Its saved
J falls 0.781 -> 0.376 -> -0.185 over frames 174–176, while its vertices remain
12–16 mm above the table. The native stable Neo-Hookean energy stays finite
through J=0; neither it nor the local collision truncation enforces positive
tetrahedral volume. Shallow table penetration therefore does not rule out
internal inversion. The aggregate substep latch cannot identify which tet
inverted first within frame 176. This audit does not isolate geometry, stiffness
contrast or insufficient convergence as the sole cause, and adds no solver fix.

H12 stopped after a post-opening retention plateau, preserving 273 frames / 9.1 s
(job 4802149, 55m31 wall time). At frames 241, 256 and 271, all 16 remaining
finger contacts are on left post shape 3, not the pads. Interval centroid speed
falls from 1.024 to 0.371 mm/s, with no further commanded motion. SIGINT saved
the in-memory evidence rather than consuming the remaining planned 360 frames;
this is a failed partial trial, not a complete 12-second episode. Its exact judge
reports held/released=false, success=false, carry 91.892 mm, pre-lowering
movement 103.249 mm, minimum transit clearance 40.651 mm, maximum recorded
penetration 0.074710 mm, zero inversion/nonfinite, 5 N jaw caps. The latched
minimum body J is 0.057620, requiring saved-geometry inspection despite no
inversion. Evidence: agent_b/09_main_placement. Saved-state render job 4805723
replaces the cancelled afterok render dependency, with no new physics steps.

During table contact, Newton warned that its per-body particle-contact adjacency
list exceeded 4096 candidates. Its native fallback scans all globally stored
contacts and preserves their body reactions, including edge/face records. This
warning is not itself dropped-contact evidence. Global soft_contact_count
exceeding soft_contact_max would be a different overflow; raw values were not
saved. This primitive box/plane scene uses default capacity sized for the
particle/edge/face query maxima. No buffer or solver change follows from the
warning alone.

H12 saved-pose diagnosis: the jaws actually opened to 68.448 mm centre separation
(68 mm commanded), with a 62.45 mm inner-pad gap. Final contacts lie on the left
post's rear Y face (14) and bottom (2), at the expected approximately 0.7 mm
particle shell rather than deep geometric overlap. Left-post contact begins at
frame 194, right-pad contact lasts through 202, and left-pad contact through 225.
The tail touches the table at 179, but the main body's minimum clearance never
falls below 45.94 mm (190) and is still 54.23 mm at 273. Thus the 103 mm descent
estimate was insufficient: the toy reoriented around tail support instead of
translating rigidly. This is not a supported main-body placement or stalled jaw.
Simply withdrawing would still leave a substantial body fall.

Independent volume audit finds the minimum-J event localized near the fine neck
interface, not bulk collapse. Current minimum body J at 196/204/219/273 is
0.3683/0.5402/0.5225/0.5234; the worst saved snapshot is 0.2347 at 195, while
the 0.0576 substep latch first appears at 196 without a tet identity. Final body
volume is 99.967% of rest volume, median J=0.99990, and no body tet has J<0.5.
Directional distortion persists in 167 fine neck tets comprising 0.065% of body
rest volume; J alone does not establish visual realism. Evidence:
agent_c/h12_body_compression_audit.json. The saved-state video is
agent_a/main_placement_render/overview_saved_failed.mp4 (render job 4805723,
zero physics steps).

## H13 — lay the body down before opening, then withdraw

H12 established that vertical lowering alone folded/reoriented the toy onto its
tail and moved contact onto a post. One trajectory-only retry preserves every
command through 5.3 s and all H10 physics. Approach 44.0953 mm downward over
5.3–6.1 s, then use a 75-degree Y/Z arc toward -Y over 6.1–10.1 s, constructed
from H12's first-tail-touch geometry with a gradual 9 mm clearance allowance.
Pause 10.1–10.6, open 10.6–11.0, withdraw another 20 mm along -Y over
11.0–11.6, and observe 4.3 seconds afterward (477 frames / 15.9 s).

The final arc command is (0.4752, -0.1219723, 0.8110253) m; withdrawal ends at
Y=-0.1419723, leaving 15.03 mm inside the existing joint limit. Peak approach
Z speed is 0.10335 m/s; peak arc Z speed is 0.08611 m/s, total speed 0.1220 m/s.
Existing quintic timing supplies continuous positions and zero endpoint rates.
No object pose, rotation or tail attachment is prescribed. The positive-Y
alternative exceeds the current guide range; a straight diagonal chord would
shorten the ring-to-tail span by about 40 mm mid-motion.

Independent frozen-geometry checks at 151 arc poses find no physical collision
with any of the five gripper shapes; carrier/posts also clear boxes expanded by
5 mm. The proxy footprint stays inside the bench. This is a clearance
construction, not a dynamics prediction: the level gripper does not command
object rotation, the tail can slide, the reference pivot rises, and actual
tracking lag/deformation can consume the margin. Final proxy body clearance
1.129 mm exceeds the 0.7 mm particle radius and does not itself establish support.

Add only an output-probe scalar for actual main-body table clearance, using
existing saved particles and part labels, to distinguish body support from tail
touch. Existing task/physics success thresholds and early quality/loss stops are
unchanged. Report actual body support before opening separately from any short
drop; do not infer it from the proxy. Persistent post retention or invalid mesh
remains failure. Evidence: agent_b/h12_laydown_proxy.json,
agent_b/h13_trajectory_source_check.json and agent_b/laydown_trajectory.patch.
The source check confirms commands through 5.3 s are bitwise identical to H12
and build_scene/judge are unchanged. A 2-hour allocation protects the single
trial against the observed table-contact slowdown; no additional trial is queued.

Final code review passed. H13 job 4806053 uses the reviewed 477-frame driver,
output agent_b/10_main_laydown. Dependent saved-state-only render job 4806059
uses afterany so a controlled early-failure stop can still be rendered; output
agent_a/main_laydown_render. No further physics run is queued.

H13 stopped at frame 294 / 9.8 s because frames 293–294 had no finger contact
before the commanded 10.6 s opening. This is a failed placement, despite
successful body support. Main-body clearance reaches the 0.7 mm contact shell
at frame 268 / 8.933 s and remains there through frame 292. No posts contact;
both pads retain contact through 291, then one left-pad contact remains at 292.
Body minimum-J latch stays 0.6233 through 273, falls to 0.3749 by 275 and
0.2169 by 287 as the arc continues after body support. No inversion/nonfinite
events occur. Carry is 92.752 mm, transit clearance 40.574 mm, maximum recorded
penetration 0.039395 mm and jaw drives remain capped at 5 N. This distinguishes
late release scheduling from the unsupported-body/post-retention failure in H12.
The exact driver stop and full rows establish premature loss; the truncated
judge's `held` label is not evidence that the toy remains on the fingers.
Saved-state render 4806059 completed: agent_a/main_laydown_render/overview_saved_failed.mp4.
A bounded saved-geometry review will determine the smallest timing correction;
no physics or acceptance changes are proposed.

## H14 — stop the arc at the measured supported pose

An earlier opening at 9.1–9.5 s with the unchanged descending arc was considered
and rejected before simulation. B's saved-frame-273 clearance check finds that
opening clears the pads/posts, but the continuing carrier descent intersects
the frozen ring geometry from about 9.3 s. Released dynamics might avoid it,
but the existing evidence does not establish a clear sweep. Evidence:
agent_b/h13_early_open_clearance.json. No early-opening trial was submitted.

Instead, preserve all H13 phase windows and shorten the existing arc endpoint
to the H13 9.1 s commanded pose: angle 67.236328125 degrees and clearance
8.068359375 mm. These are the original 75 degrees / 9 mm multiplied by the
existing quintic's value 0.896484375 at 9.1 s. This follows the same geometric
arc prefix more slowly over 6.1–10.1 s, then uses the existing 0.5 s pause,
10.6–11.0 opening and 11.0–11.6 withdrawal. Only two trajectory constants
change; no new path transition, contact rule or controller is introduced.

At the measured endpoint H13 had sustained main-body table contact since
8.933 s, both pads still contacting, no posts and no new compression event.
Later closed descent worsened neck deformation and lost the rim at 9.767 s.
Stopping at the supported pose should avoid that over-travel and also keep the
carrier clear after release. Changing traversal speed can change the dynamics;
the saved pose is a hypothesis, not a promised support outcome. The final run
must still pass retention until deliberate opening, release, settling and all
existing physics checks. H13 remains failed and thresholds stay unchanged.
Endpoint opening/withdrawal clearance is under bounded review before launch.

B's actual-frame-273 swept-box check clears stationary opening and subsequent
20 mm -Y withdrawal for all five gripper shapes; final particle-radius-inflated
envelopes are also clear. New command endpoint is (0.4752, -0.113263366,
0.836156921) m, and withdrawn Y=-0.133263366 leaves 23.74 mm joint margin.
The measured prior pad center differs by 1.26 mm vertically, so support remains
a dynamic outcome to verify. Independent review confirms the geometric prefix
and unchanged smooth endpoints/phase windows; peak arc speed falls from
0.1220 to 0.1096 m/s. Source syntax and the output-only driver diff pass review.
Evidence: agent_b/h13_supported_endpoint_clearance.json. Exactly two main
trajectory constants changed for H14; build_scene and judge remain unchanged.

H14 submitted as job 4806831, output agent_b/11_main_supported_laydown,
477 frames / 15.9 s with a 2-hour allocation. Saved-state-only render job
4806832 depends afterany on it, output agent_a/main_supported_laydown_render.
No other physics trial is queued.

H14 completed its contact phases but stopped at frame 333 / 11.1 s on body
inversion. It retained contact until deliberate opening (`held_until_open=true`),
then cleared the fingers; the frame-331 diagnostic has zero finger contacts.
The full-task result is `success=false`, `stop_reason=invalid-mesh`, with one
inverted body tetrahedron in the final saved frame and 86 inverted body
tet-substeps (minimum body J=-1.79817), zero nonfinite values. Carry92.163 mm,
minimum transit clearance40.574 mm, maximum recorded penetration0.037245 mm
and5 N jaw caps pass their limits. The short final window still includes
pre-release frames, so `released=false` is not evidence of retained fingers.
Supported-contact minimum height alone did not establish stable body support:
post-release center speed reaches0.393 m/s and the body rolls/moves strongly.
Terminal geometry is under review before deciding on any additional trajectory
change. Saved-state rendering4806832 completed, output
agent_a/main_supported_laydown_render/overview_saved_failed.mp4.

H14 terminal geometry confirms last finger contacts at322, zero323–333, no
post contacts anywhere. All2,356 exterior faces clear the physical carrier,
pads and posts at every saved frame318–333; even0.7 mm-expanded boxes are
clear from324. The eventual inversion therefore occurs after10 fully separated
frames. Only four body vertices touch the table before opening. Ring underside
is still44.5 mm above the table at323, first reaches the table at330, and the
body rolls. Saved inverted tet4856 is near the fine body/ring interface, not
the H11 inverted tet6686. Evidence: agent_b/h14_terminal_analysis.json.

Independent elastic-energy reconstruction identifies a ring-energy spike before
table impact, so ordinary falling/rolling is not yet an adequate causal account.
RSS source inspection confirms both state force arrays clear every substep,
default main collision cadence is every substep, and native VBD refreshes
body-particle material/contact adjacency. No obvious stale RSS force explains
it. Gripper forces and self-contact work were not recorded at every substep;
saved-frame geometry/energy cannot uniquely establish the numerical cause.
No further physics trial or solver edit has been made during this audit.

## H15 — correct the native edge degeneracy tolerance for this mesh scale

The native detection-only snapshot job4807170 completed in25 s with zero
physics steps. It confirms four VT contacts and no EE contacts through324;
at325 there are14 VT and14 unique EE contacts, without buffer overflow.
The strongest admitted EE pair [358,1830]/[1645,1706] has current separation
1.1078 mm and an estimated89.2 N normal pair magnitude. These are contact-law
estimates at saved poses, not measured substep impulses or universal force caps.
Evidence: agent_c/03_h14_native_snapshot/result.json.

A concrete parameter-scale defect explains why this pair is admitted. Installed
Warp1.18 closest_point_edge_edge uses epsilon to compare SQUARED EDGE LENGTH,
not a parallel angle. Newton names the parameter particle_edge_parallel_epsilon
and defaults it to1e-5, so edges shorter than sqrt(1e-5)=3.162 mm become points.
This pair's rest edges are2.969 and3.073 mm. Their exact segment separation is
4.3766 mm, inside the configured6 mm rest exclusion, but treating both edges
as points gives6.0874 mm and incorrectly passes that exclusion. Stretching
an edge past3.162 mm can also change its closest-point calculation abruptly.
Source: installed warp/native/intersect.h:1151; Newton
newton/_src/geometry/tri_mesh_collision.py passes the same parameter into
rest filtering; native VBD contact evaluation uses it as well.

Before any full retry, compare unchanged saved poses with existing native
parameter1e-5 versus1e-12 (1 micrometre degeneracy cutoff), retaining all
interaction distances, rest exclusions, topology filters and force/quality
checks. Check source minimum boundary-edge length and actual retained pairs.
This is a native configuration correction hypothesis, not a new solver or
disabling self-contact. No main edit/full simulation has yet been made forH15.

Paired native snapshot4807223 completed in9 s with zero physics steps. All
sampled contacts are identical at1,31,61,105,151,181,318,323,324. At325 the
corrected tolerance removes9 false EE pairs with rest separations4.31–4.59 mm
and adds13 valid previously missed pairs with rest separations6.69–6.78 mm.
The largest estimated normal pair magnitude falls89.22 to24.52 N. VT pairs
are unchanged and no native buffers overflow. The minimum source boundary
edge is0.29235 mm, far above the proposed1 micrometre degeneracy cutoff.
Remaining valid contact potentials mean this is not proof of dynamic stability.
Evidence: agent_c/04_h14_edge_epsilon_snapshot/result.json.

Integrated exactly one scene parameter, particle_edge_parallel_epsilon=1e-12,
with a units comment. H14 trajectory, native80 iterations,80 substeps, material
profile, full-surface collision, contact distances, rest/topology exclusions,
force limits and all quality/task thresholds remain unchanged. H15 job4807226
uses the reviewed477-frame/2-hour driver, agent_b/12_main_edge_epsilon. Its
startup asserts the corrected tolerance in both native solver and detector.
No Newton or RSS solver implementation changed; no further physics run is queued.

H15 startup passed on n128 with source SHA256
e63fd3c272e266840765331479af8995225ee7503c7cc32fe5807c57250eb6b8.
Dependent saved-state-only render4807228 uses afterany; output
agent_a/main_edge_epsilon_render.

## Final H15 result and stopping point

Job 4807226 completed all 477 frames / 15.9 s in 1 h 32 min 15 s on one GH200,
exit 0. The existing task judge reports `success=true`, `held_until_open=true`,
`released=true`. Its legacy `stop_reason="dropped"` also labels supported
placement; it does not mean this run used the earlier free-drop trajectory.

| Core measurement | H15 result | Acceptance |
| --- | --- | --- |
| Carry displacement during commanded phase | 92.211 mm | 60–100 mm |
| Minimum whole-object clearance during transit | 40.547 mm | At least 20 mm |
| Retention until deliberate opening | Passed | No contact loss before opening |
| Last finger contact / first clear frame | 322 / 323 (10.767 s) | All later frames clear |
| Actual withdrawal, frames 330–348 | 20.036 mm along -Y | Commanded 20 mm; no recontact |
| Settling observed after withdrawal | 4.3 s | Full planned observation completed |
| Final half-second body/table clearance | 0.69791–0.69803 mm | Inside existing -0.5–2 mm band |
| Final peak whole-vertex-centroid speed | 0.14633 mm/s | At most 5 mm/s |
| Maximum recorded contact penetration | 0.036272 mm | At most 0.5 mm |
| Peak jaw drives | 5 N / 5 N | At most 5 N each |
| Inverted / nonfinite tetrahedra | 0 / 0, including substep latch | None |
| Minimum volume ratio J: tail / body / ring | 0.90384 / 0.43524 / 0.97045 | Positive |

Saved-state geometry independently verifies that the main body reaches support
before opening, the jaws actually open, and withdrawal actually executes. No
finger-post contacts occur in this run. All 2,356 exterior triangles are clear
of the physical carrier, both pads and both posts at the final frame, also when
those boxes are expanded by the 0.7 mm particle radius. This checks separation
independently of a zero contact count. The whole-vertex-centroid speed computed
directly from saved positions matches the task result.

Evidence, relative to `outputs/debug/contact_20261102/`:

- `agent_b/12_main_edge_epsilon/result.json`: task and physics result.
- `agent_b/12_main_edge_epsilon/diagnostics.json`: all 477 recorded frames.
- `agent_b/12_main_edge_epsilon/metadata.json`: configuration and source hash.
- `agent_b/12_main_edge_epsilon/states.npz`: recorded particle/body positions.
- `agent_b/h15_final_verification.json`: independent support, withdrawal and
  exterior-surface separation checks, using `agent_b/verify_h15_saved.py`.
- `agent_a/main_edge_epsilon_render/overview_saved_passed.mp4`: complete video.
- `agent_a/main_edge_epsilon_render/render_manifest.json`: rendering provenance.

Render job 4807228 completed in 25 s, exit 0, with zero physics steps. It renders
all 477 stored states and duplicates each 30 Hz sample twice for 60 fps encoding;
it does not invent intermediate motion. Held, pre-open and final stills were
visually inspected. Task and render source hashes both match
`e63fd3c272e266840765331479af8995225ee7503c7cc32fe5807c57250eb6b8`.
The source-state SHA256 is
`dd950ef6a4073342924867607f5458bf2c9a22133c1d926ad2dc6bac5cdebd16`.

The exact executed diagnostic driver is `agent_b/probe_main_edge_epsilon.py`,
with its Slurm environment in the adjacent `.sh`. It imports the main example's
scene, trajectory and judge. The existing example can request the same duration
with `--steps 477`; its unchanged default duration is 20 s, which this run does
not validate. No additional physics run follows this nominal full-task pass.

Limits: this is one configuration and one completed corrected-tolerance trial.
The previous free-drop impact scenario was not rerun after the tolerance fix;
ADMM coupling and AIRBOT remain unvalidated. The solve is expensive and no policy
throughput or general reliability is established. Settling uses an arithmetic
vertex centroid, not every local particle's speed; penetration is sampled at
recorded frames. The substep quality latch is not proof that every internal
solver iterate is inversion-free. The 5 N limits apply to jaw drives, not every
contact reaction; the existing carrier drive limit remains 200 N. Saved positions
are inspection/rendering evidence, not complete resumable checkpoints. Native
friction remains velocity-regularized; no persistent static-friction model was
added. Source E, nu, density, mesh and labels are preserved, but scene viscosity
is explicitly 500 rather than 5000 Pa s.

Final review found no blocking issue in the three changed RSS files. Newton has
no tracked source edits; unrelated AIRBOT work and the pre-merge stash are left
alone. Work stops at the agreed nominal task milestone. The subsequent requested
publication to `yiduo/dev` includes the three source files, this report and the
archived final H15 evidence; it excludes exploratory outputs and AIRBOT work.
