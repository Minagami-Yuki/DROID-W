# TUM Dynamic Clean DROID-W-O + Omega Fixed-K Baseline

Date: 2026-07-17. ATE is Full-trajectory RMSE in metres. All runs use seed 43
and the same eight saved Omega K bundles used by the prior TUM unknown-K sweep.

## Protocol

The new baseline runs a clean DROID-W-O worktree at commit `c3414af`. Only
`cam.fx/fy/cx/cy` is replaced with the saved Omega K; no Omega cache, uncertainty
weighting, focal calibration, confidence route, or full-pinhole BA is loaded.
The current-code fixed-K and confidence columns are existing runs with exactly the
same K bundles. The GT-K reference is the prior TUM DROID-W evaluation and is read
only for context, not rerun in this pass.

## Full-Trajectory ATE

| Sequence | DROID-W GT K reference | Clean DROID-W-O + Omega K | Current Omega fixed K | Current confidence BA | Confidence vs clean DROID-W-O |
|---|---:|---:|---:|---:|---:|
| `walking_halfsphere` | 0.0157119 | 0.016766866 | 0.016744435 | 0.017019966 | +1.51% |
| `walking_rpy` | 0.0303817 | 0.036367878 | 0.036339078 | 0.037173011 | +2.21% |
| `walking_static` | 0.0047341 | 0.005266717 | 0.005268590 | 0.005265776 | -0.02% |
| `walking_xyz` | 0.0121554 | 0.014291443 | 0.014285611 | 0.013791262 | -3.50% |
| `sitting_halfsphere` | 0.0138364 | 0.016439456 | 0.016490614 | 0.014917425 | -9.26% |
| `sitting_rpy` | 0.0211892 | 0.023787181 | 0.023783738 | 0.021941275 | -7.76% |
| `sitting_static` | 0.0047721 | 0.004875403 | 0.004869692 | 0.004868020 | -0.15% |
| `sitting_xyz` | 0.0080418 | 0.010298483 | 0.010286099 | 0.009760547 | -5.22% |
| **Macro mean** | **0.0138528** | 0.016011678 | 0.016008482 | **0.015592160** | **-2.62%** |

Clean DROID-W-O fixed-K and current-code fixed-K differ by only 0.02% in macro
Full ATE (largest per-sequence difference 0.31%). Thus the prior current-code
fixed-K control is a faithful proxy for original DROID-W-O when no enhancement is
enabled. Confidence BA improves 6/8 sequences against the clean baseline, but
remains 12.56% above the GT-K reference; it is a partial recovery rather than
GT-K parity on TUM.

## Keyframe Macro ATE

| DROID-W GT K reference | Clean DROID-W-O + Omega K | Current Omega fixed K | Current confidence BA |
|---:|---:|---:|---:|
| 0.0158181 | 0.018071555 | 0.018070588 | 0.017620854 |

The original fixed-K baseline is 14.25% above the GT-K KF reference; confidence
BA recovers part of this gap but remains 11.40% above GT K.
