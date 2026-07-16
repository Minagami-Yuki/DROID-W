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
