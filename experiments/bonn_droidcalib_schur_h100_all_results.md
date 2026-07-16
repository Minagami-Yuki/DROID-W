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
