# Bonn Full-Pinhole Soft Calibration Weight

Date: 2026-07-17. ATE is RMSE in metres. All methods use seed 43 and the
canonical per-sequence Omega K recorded in `bonn_omega_k_manifest_v1.json`.

## Protocol

This is the TUM soft-calibration pilot extended without retuning to all eight
Bonn sequences. It uses joint `fx, fy, cx, cy` Schur BA with a four-parameter
prior and the same trust regions, observability gates, and rollback rules as the
TUM experiment. Every calibration edge is retained. Cached Omega uncertainty
smoothly scales pixels from 1.0 at 0.85 uncertainty to 0.60 at 1.00; edge span
smoothly scales from 0.75 at span 1 to 1.0 at span 4. `omega_prior.uncertainty`
is `apply_to: none`, so normal DROID tracking weights remain unchanged. The
confidence-recovery route is disabled in this branch.

The two reference columns are the exact-K-paired clean DROID-W-O fixed-K baseline
and the canonical confidence-BA result (with the repeat-replaced balloon2 value).

## Full-Trajectory ATE

| Sequence | DROID-W-O + Omega fixed K | Full-pinhole soft calibration | Confidence BA | Soft vs fixed K | Soft vs confidence |
|---|---:|---:|---:|---:|---:|
| `balloon` | 0.022043046 | **0.021553226** | 0.022401659 | -2.22% | -3.79% |
| `balloon2` | **0.026718238** | 0.028639627 | 0.026742116 | +7.19% | +7.09% |
| `crowd` | 0.014587646 | **0.010875672** | 0.015098600 | -25.45% | -27.97% |
| `crowd2` | 0.022691415 | 0.023622214 | **0.020318232** | +4.10% | +16.26% |
| `moving_nonobstructing_box` | 0.017680265 | **0.016858135** | 0.016873716 | -4.65% | -0.09% |
| `moving_nonobstructing_box2` | 0.034604291 | 0.031073184 | **0.022425307** | -10.21% | +38.55% |
| `person_tracking` | 0.034312241 | **0.030897796** | 0.031391200 | -9.95% | -1.57% |
| `person_tracking2` | 0.036034631 | 0.037064749 | **0.029594133** | +2.86% | +25.24% |
| **Macro mean** | 0.026083972 | 0.025073075 | **0.023105620** | **-3.88%** | +8.52% |

Soft calibration wins 5/8 against fixed K and 4/8 against confidence BA, but its
macro Full ATE remains 8.52% above confidence BA. Its gains are concentrated on
`crowd`, `person_tracking`, and the two box sequences; it regresses on the two
sequences where routing is important (`crowd2`, `person_tracking2`).

## Keyframe ATE And Diagnostics

| Method | Macro KF ATE | vs DROID-W-O fixed K |
|---|---:|---:|
| DROID-W-O + Omega fixed K | 0.027521985 | - |
| Full-pinhole soft calibration | 0.026444432 | -3.92% |
| Confidence BA | **0.024395130** | -11.36% |

The soft branch retains all edges at every call. Mean calibration-weight coverage
ranges from 72.1% (`person_tracking2`) to 88.4% (`moving_nonobstructing_box`),
with accepted K updates ranging from 2/308 to 24/242. This rules out the hard-mask
information-loss failure, but does not make the continuous weight a sufficiently
selective activation rule.

## Decision

Keep this as a useful full-intrinsics ablation, not the Bonn mainline. The evidence
supports confidence-routed focal BA as the canonical unknown-K method; the soft
four-parameter update is complementary on some scenes but is not robust enough to
replace that route.
