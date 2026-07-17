# DROID-W Omega: Improvements, Evidence, and Next Steps

Date: 2026-07-17

## Current Research Direction

The strongest current direction is calibration-robust dynamic monocular tracking:
VGGT-Omega supplies scene priors and an approximate camera model, while constrained
joint BA refines focal length together with pose and depth only when the trajectory
and the available evidence make that update safe. This extends DROID-W toward an
in-the-wild operating mode instead of relying on an accurately supplied K.

The present evidence supports this direction, but the claim must remain split into
two parts. The complete TUM eight-sequence run establishes parity with DROID-W under
known K. A strict unknown-K TUM first pass now establishes that the system completes
all sequences and that joint BA recovers part of the Omega-K gap, but it does not yet
recover to GT-K parity.

## Improvements over Original DROID-W

### 1. VGGT-Omega Prior Interface

- Added model and cache-backed Omega inference so expensive predictions can be
  generated once and reused reproducibly.
- Added access to Omega depth, confidence, uncertainty, compact patch tokens, and
  camera-intrinsics estimates.
- Kept every prior optional, allowing exact DROID-W-style controls from the same
  codebase.

### 2. Dynamic Correspondence Reliability

- Added edge-distance weighting so correspondences near uncertain object boundaries
  can be attenuated.
- Added bidirectional/cycle checks and residual gating so semantic uncertainty alone
  cannot suppress factors without geometric support.
- Added patch-token disagreement as a compact semantic change signal.
- Added v46 cross-modal agreement, temporal persistence, factor-graph observability
  protection, long-span protection, and a bounded information-suppression budget.
  The resulting mechanism is conservative: it can weaken an existing attenuation
  candidate but cannot create stronger suppression by itself.

### 3. Omega-Initialized Intrinsics and Joint BA

- Added extraction of approximate `fx, fy, cx, cy` from VGGT-Omega.
- Integrated the flow3rslam-compatible CUDA Schur backend for joint focal, pose, and
  depth optimization. The implementation updates `fx/fy` with a fixed aspect ratio;
  `cx/cy` remain fixed.
- Added a focal prior, Hessian/observability gate, bounded log step, bounded total
  deviation, reprojection-loss acceptance test, and full rollback on rejection.
- Added early bootstrap recovery for large initial-K error, trajectory-stability
  gating, and Omega-confidence routing for calibration-sensitive cases.
- Added per-update CSV diagnostics containing proposed/applied K updates, focal
  information, loss, trajectory update magnitude, route, and rejection reason.

### 4. Evaluation and Operational Robustness

- Added fixed-seed/multi-seed protocols, deterministic-mode support, complete-output
  checks, retryable runners, and separate Full/KF trajectory reporting.
- Added headless execution support by avoiding an unnecessary GUI import when the GUI
  is disabled.

## Evidence Available Now

All ATE values are RMSE in metres.

| Evidence | Result | Interpretation |
|---|---|---|
| TUM Dynamic, 8 sequences, known K | v46 Full mean `0.013875` vs DROID-W `0.013853` (+0.10%); KF +0.07% | The added tracking machinery is effectively neutral on TUM, but this is not an unknown-K result. |
| Bonn development, known K, 3 seeds | v25 balanced Full mean is 1.06% below DROID-W; v45 is +0.04% | Omega reliability can give a small gain, but later variants are not universally better. |
| Bonn unknown-K, clean DROID-W-O baseline, 8 sequences, seed 43 | DROID-W-O + Omega fixed K `0.026084`; confidence BA `0.023106` (-11.42%); DROID-W-O + GT K `0.023047` | Exact K-paired comparison shows that the proposed conditional BA recovers most of the direct-Omega calibration gap; Full wins are 5/8 and the gain is concentrated on box2, person_tracking2, crowd2, and person_tracking. |
| 7-Scenes chess seq-01, Omega fixed K | Full `0.036533` vs GT-K `0.037221` | A favorable unknown-K pilot; one sequence is insufficient for a general claim. |
| Bonn crowd2, Omega fixed K | Full `0.022644` vs GT-K `0.017673` | Approximate K can materially hurt, so explicit recovery is justified. |
| Bonn box2, 3 seeds | Omega fixed K `0.034557`; K recovery `0.020434`; confidence recovery `0.022394` | Joint BA recovers a calibration-sensitive degradation with low variance. |
| Bonn crowd2, 3 seeds | Omega fixed K `0.023774`; confidence recovery `0.020165` | Confidence routing improves both mean and stability. |
| Bonn person_tracking2, 3 seeds | Stability `0.029443`; confidence recovery `0.036018` with high variance | Current routing is not universally selective and can activate in the wrong regime. |
| TUM Dynamic unknown-K, 8 sequences, seed 43 | GT-K `0.013853`; Omega fixed-K `0.016008` (+15.56%); stability BA `0.015466` (+11.65%); confidence BA `0.015592` (+12.56%) | All 24 non-GT runs completed. Joint BA recovers 2.6--3.4% relative to fixed Omega K, but misses the 5% GT-K-parity target. |

The defensible current claim is therefore: Omega-initialized, constrained joint BA
can recover tracking in a subset of calibration-sensitive dynamic sequences, while
the Omega reliability path preserves DROID-W-level tracking when accurate K is
available. The unknown-K system is operational but does not yet match the GT-K
reference over the full TUM Dynamic suite.

## Immediate Next Task: Diagnose the Remaining Unknown-K Gap

The first TUM unknown-K pass revealed a specific unmodeled component: Omega returns
`cx=320, cy=240` on all eight sequences, while the TUM configuration has
`cx=320.1, cy=247.6`. The present joint BA updates focal only, so it cannot recover
the 7.6-pixel vertical principal-point offset. Before extending the solver, run the
following two-sequence diagnostic on `walking_rpy` and `sitting_xyz`, which have
large fixed-K gaps:

1. Omega K as used in the strict protocol.
2. Omega focal with only `cx/cy` replaced by the known diagnostic value.
3. Omega K with a constrained online `cy` update and a strong prior.

This is a diagnosis, not a result intended for the unknown-K claim. If principal
point correction removes most of the gap, extend the Schur state from focal-only to
constrained `fx/fy/cx/cy` optimization with separate Hessian gates, small principal
point steps, priors, and full rollback. If it does not, retain focal-only BA and
improve the confidence router or the Omega K estimator instead.

This diagnosis and a first constrained full-pinhole implementation were tested on
2026-07-17. Replacing only Omega `cx/cy` with the diagnostic TUM value improves
`walking_rpy` from `0.036339` to `0.031477` m, confirming that principal point is a
major factor there. A shared 4D Gaussian prior was then added directly to the CUDA
Schur system: it improves `walking_rpy` to `0.033285` m and `sitting_xyz` to
`0.010014` m, but is still not ready for a full suite. Locking focal while optimizing
only `cx/cy` regresses both pilots, so principal point must remain jointly coupled to
focal, pose, and depth. Details are recorded in
`experiments/tum_principal_point_full_pinhole_pilot.md`.

A subsequent state-consistency audit found that new keyframes inherited only the
current focal estimate, not the jointly optimized principal point. Full-K propagation
is now fixed. With the corrected code, continuous 4D prior BA reaches `0.032065` m
on `walking_rpy` (3.67% better than the earlier continuous run), but `0.010157` m on
`sitting_xyz` (1.43% worse). A freeze-after-120-keyframes control gives `0.032708` m
and `0.010115` m respectively. It does not establish a robust benefit for early
freezing, so the next mechanism experiment must improve the observation model rather
than tune a freeze boundary.

The first direct observation-model control was negative: a K-only subset using
low Omega uncertainty (`<=0.93`) and keyframe span `>=4`, while holding ordinary BA
fixed, changes corrected full-pinhole ATE from `0.032065` to `0.035807` m on
`walking_rpy` and from `0.010157` to `0.010263` m on `sitting_xyz`. The hard subset
retains only 9.2% and 20.7% of calibration weight on average, respectively. It
therefore removes useful coupled constraints rather than isolating a better camera
signal. Do not tune this threshold or extend it to the full suite.

A soft counterpart was then tested with the same uncertainty/span signals but no
hard masking: every edge is retained, with a 0.60--1.0 pixel scale and a 0.75--1.0
span scale. In a clean seed-43 rerun it changes ATE from `0.032065` to `0.031994` m
on `walking_rpy` (-0.22%) and from `0.010157` to `0.010166` m on `sitting_xyz`
(+0.09%). Mean calibration coverage is 64.4%/68.5% and accepted updates remain
57/426 and 1/319. Treat this as a neutral stability control, not a new method
claim: smooth weighting avoids the hard subset's information loss but does not
provide a measurable, repeatable gain.

## Tasks After TUM Closure

1. After resolving the principal-point diagnosis, rerun the frozen four-way TUM
   protocol with three seeds only for sequences that still regress by more than 1%.
2. Freeze one unknown-K preset and validate it without tuning on 7-Scenes, Wild-SLAM
   MoCap, and DROID-W in-the-wild sequences.
3. Run the minimal paper ablation: GT-K, Omega fixed-K, focal-only BA, constrained
   full-intrinsics BA, stability-gated BA, and confidence-routed BA.
4. Report failure rate and ATE distribution in addition to mean ATE; unknown-K
   robustness is weakened by rare catastrophic trajectories even when the mean is
   favorable.
5. Only revise the confidence router if the strict TUM protocol exposes repeatable
   false activations. Sub-percent ATE differences should not trigger more method
   complexity.
6. Complete runtime, memory, K-convergence, and qualitative trajectory diagnostics
   after the algorithm and preset are frozen.
