# Bonn DroidCalib Schur H100

All eight Bonn sequences use v25 tracking, seed 43, a 12-frame VGGT-Omega
intrinsics bootstrap, focal prior weight 5, and the constrained joint Schur
solver. A focal update is allowed only when the post-gating log-focal Hessian
is at least 100. ATE values are RMSE in metres.

The v25 reference uses the dataset calibration. The H100 condition instead
uses Omega's estimated K, so this table measures the unknown-intrinsics cost
and must not be read as a same-input ablation.

| Sequence | v25 Full | v25 KF | Omega + Schur H100 Full | Omega + Schur H100 KF |
| --- | ---: | ---: | ---: | ---: |
| balloon | 0.025262 | 0.026720 | 0.022525 | 0.023853 |
| balloon2 | 0.024713 | 0.027778 | 0.025827 | 0.029007 |
| crowd | 0.012888 | 0.014546 | 0.014723 | 0.015907 |
| crowd2 | 0.018261 | 0.019621 | 0.020362 | 0.021628 |
| moving_nonobstructing_box | 0.014816 | 0.014741 | 0.017419 | 0.018965 |
| moving_nonobstructing_box2 | 0.023439 | 0.025111 | 0.034278 | 0.037771 |
| person_tracking | 0.032727 | 0.032322 | 0.031577 | 0.031253 |
| person_tracking2 | 0.029830 | 0.029670 | 0.031521 | 0.031795 |
| mean | 0.022742 | 0.023814 | 0.024779 | 0.026272 |

The gate accepted 10/33, 21/38, 0/93, 43/94, 1/56, 1/78, 37/57, and 17/51
focal candidates in the table order. It prevents most calibration updates in
the two moving-box sequences and all updates in `crowd`, but an Omega K error
remains even when no update is accepted.

Outputs are stored below `/data1/czy/Output/DROID-omega/Bonn` using the
`bonn_<sequence>_omega_droidcalib_schur_h100` naming convention.

## Early Bootstrap Repair

The large-K-error moving-box sequences exposed an H100 failure mode: a strict
observability gate rejected nearly all early focal updates. The bootstrap runs
use the same solver and loss rollback, but from keyframes 30 through 80 evaluate
every BA call with Hessian >= 25, a 0.004 log-focal step limit, and a 0.10 total
deviation limit. They then return to the H100 tracking gate.

| Sequence | v25 Full/KF | H100 Full/KF | Bootstrap Full/KF |
| --- | ---: | ---: | ---: |
| moving_nonobstructing_box | 0.014816 / 0.014741 | 0.017419 / 0.018965 | 0.015068 / 0.016390 |
| moving_nonobstructing_box2 | 0.023439 / 0.025111 | 0.034278 / 0.037771 | 0.020429 / 0.022172 |

The final log-focal corrections were -0.068825 (box) and -0.039556 (box2),
leaving each estimate close to the shared Bonn calibration while retaining the
same loss-increase rollback used by H100.

## Full Bootstrap Comparison

The same early-bootstrap schedule was evaluated on all eight sequences.  It
uses Omega K as the only initialization and has no access to the dataset
calibration during tracking.  ATE is RMSE in metres; `Full` is the filled
trajectory and `KF` is the keyframe trajectory.

| Sequence | v25 Full/KF | H100 Full/KF | Bootstrap Full/KF |
| --- | ---: | ---: | ---: |
| balloon | 0.025262 / 0.026720 | 0.022525 / 0.023853 | 0.023787 / 0.025514 |
| balloon2 | 0.024713 / 0.027778 | 0.025827 / 0.029007 | 0.025921 / 0.028924 |
| crowd | 0.012888 / 0.014546 | 0.014723 / 0.015907 | 0.018127 / 0.017130 |
| crowd2 | 0.018261 / 0.019621 | 0.020362 / 0.021628 | 0.020366 / 0.021433 |
| moving_nonobstructing_box | 0.014816 / 0.014741 | 0.017419 / 0.018965 | 0.015068 / 0.016390 |
| moving_nonobstructing_box2 | 0.023439 / 0.025111 | 0.034278 / 0.037771 | 0.020429 / 0.022172 |
| person_tracking | 0.032727 / 0.032322 | 0.031577 / 0.031253 | 0.344645 / 0.351613 |
| person_tracking2 | 0.029830 / 0.029670 | 0.031521 / 0.031795 | 0.047424 / 0.047915 |
| mean | 0.022742 / 0.023814 | 0.024779 / 0.026272 | 0.064471 / 0.066386 |

Bootstrap improves the two deliberately diagnosed moving-box failures,
especially box2, but it is not safe as a global default: `person_tracking`
regresses by more than 10x and `person_tracking2` also regresses.  The
outlier's final focal correction is only -0.000324 in log space, so the
failure is caused by repeated early joint-BA state perturbations rather than a
large final focal error.  The next design must gate *when* bootstrap updates
are permitted using a trajectory-stability criterion, rather than lowering
the focal observability threshold for every sequence.

## Trajectory-Stability Pilot

The first stability-gated pilot keeps the bootstrap focal bounds, but permits
an update only after three consecutive ordinary BA calls whose maximum local
pose update is below 0.03 translation units and 0.025 rad.  An accepted joint
BA resets this streak.  ATE is RMSE in metres.

| Sequence | v25 Full/KF | Ungated bootstrap Full/KF | Stability-gated Full/KF |
| --- | ---: | ---: | ---: |
| crowd | 0.012888 / 0.014546 | 0.018127 / 0.017130 | 0.015513 / 0.015949 |
| person_tracking | 0.032727 / 0.032322 | 0.344645 / 0.351613 | 0.030549 / 0.030176 |
| person_tracking2 | 0.029830 / 0.029670 | 0.047424 / 0.047915 | 0.029966 / 0.030344 |

The gate reduced accepted bootstrap solves from 160/382/317 to 70/57/49 in
the table order and removed the `person_tracking` collapse.  It rejected
332/321/277 BA calls as trajectory-unstable, respectively; the earlier
interpretation that the magnitude thresholds did not reject any calls was a
CSV parsing error caused by carriage returns in the reason column.  The logged
distributions are now used to tighten this condition for the full benchmark.

## Tightened Stability Gate: Full Benchmark

The full benchmark tightens the pilot thresholds to a maximum ordinary-BA
update of 0.020 translation units and 0.0125 rad, retaining the three-call
streak and reset-after-accept rule.  All conditions use Omega K rather than
the dataset calibration; ATE is RMSE in metres.

| Sequence | v25 Full/KF | H100 Full/KF | Ungated bootstrap Full/KF | Tight stability Full/KF |
| --- | ---: | ---: | ---: | ---: |
| balloon | 0.025262 / 0.026720 | 0.022525 / 0.023853 | 0.023787 / 0.025514 | 0.022008 / 0.023273 |
| balloon2 | 0.024713 / 0.027778 | 0.025827 / 0.029007 | 0.025921 / 0.028924 | 0.025676 / 0.028891 |
| crowd | 0.012888 / 0.014546 | 0.014723 / 0.015907 | 0.018127 / 0.017130 | 0.015091 / 0.015865 |
| crowd2 | 0.018261 / 0.019621 | 0.020362 / 0.021628 | 0.020366 / 0.021433 | 0.021228 / 0.022953 |
| moving_nonobstructing_box | 0.014816 / 0.014741 | 0.017419 / 0.018965 | 0.015068 / 0.016390 | 0.016870 / 0.018467 |
| moving_nonobstructing_box2 | 0.023439 / 0.025111 | 0.034278 / 0.037771 | 0.020429 / 0.022172 | 0.032585 / 0.035931 |
| person_tracking | 0.032727 / 0.032322 | 0.031577 / 0.031253 | 0.344645 / 0.351613 | 0.030476 / 0.030015 |
| person_tracking2 | 0.029830 / 0.029670 | 0.031521 / 0.031795 | 0.047424 / 0.047915 | 0.029748 / 0.030126 |
| mean | 0.022742 / 0.023814 | 0.024779 / 0.026272 | 0.064471 / 0.066386 | 0.024210 / 0.025690 |

The tightened gate accepted 29, 25, 24, 35, 7, 3, 43, and 47 bootstrap
updates in the table order.  It improves the H100 mean from 0.024779 to
0.024210 m and avoids the person-sequence failures, but is too restrictive
for the large Omega-K errors in box and box2: only 7/3 updates are accepted,
compared with 233/172 for ungated bootstrap.  Therefore, the final policy
should use the stricter stability gate for small initial K disagreement and a
separate, bounded recovery mode for large focal disagreement; a single global
threshold cannot preserve both regimes.

## Initial-K Routing Pilot

For a controlled Bonn-only routing ablation, the initial Omega focal after the
tracker's image resize selects recovery when it lies in [456, 520] px.  This
routes box2 (463 px) to the bounded high-frequency bootstrap recovery and
keeps person_tracking2 (450 px) on the tightened stability gate.  The rule
uses no ground-truth K, but its numerical range is resolution- and
dataset-specific and is not a final in-the-wild decision rule.

| Sequence | Tight stability Full/KF | Initial-K branch Full/KF | Accepted updates |
| --- | ---: | ---: | ---: |
| moving_nonobstructing_box2 | 0.032585 / 0.035931 | 0.020488 / 0.022322 | 172 recovery |
| person_tracking2 | 0.029748 / 0.030126 | 0.029723 / 0.030105 | 47 stability |

The branch restores box2 to the ungated-bootstrap regime (0.020429 m Full)
without routing person_tracking2 into that risky path.  The next replacement
for this fixed focal interval must be an Omega-derived calibration confidence
or a scale-normalized uncertainty signal, so the branch generalizes beyond
Bonn's shared image resolution and camera family.

## Omega Depth-Confidence Routing: Rejected Proxy

Omega exposes dense `depth_conf` but no direct camera-intrinsics confidence.
We therefore tested the scale-free per-keyframe raw-confidence shape statistic
`mean(depth_conf) / median(depth_conf)`, using a 30-keyframe running median
and a 1.05 recovery threshold.  The offline input-frame summary appeared to
separate box2 (about 1.08) and person_tracking2 (about 1.00), but the actual
tracker keyframe stream did not preserve that separation: both sequences
entered recovery (95 and 100 CSV rows, respectively).  The complete v2 runs
gave box2 0.022413 m Full / 0.024624 m KF and person_tracking2 0.029719 m Full /
0.030143 m KF.  This proxy is therefore rejected; it measures content and
keyframe sampling effects rather than calibration reliability.

The implementation also now skips this statistic during batched trajectory
filling, where interpolated non-keyframes are written together and must not
alter the online routing state.
