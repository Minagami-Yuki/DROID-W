# TUM Principal-Point Diagnosis and Constrained Full-Pinhole BA Pilot

Date: 2026-07-17. ATE is Full-trajectory RMSE in metres. All non-GT modes use the saved per-sequence Omega K bundle and seed 43.

The original Schur integration optimizes focal length only. This pilot first diagnoses the explanatory value of the principal point by replacing only Omega `cx/cy` with the TUM calibration, then evaluates a new constrained four-parameter (`fx, fy, cx, cy`) pinhole Schur path.

| Sequence | DROID-W GT-K | Omega fixed-K | Focal-only stability BA | Omega focal + diagnostic GT principal point | Constrained full-pinhole BA |
|---|---:|---:|---:|---:|---:|
| `walking_rpy` | 0.0303817 | 0.0363391 | 0.0361166 | **0.0314770** | 0.0347399 |
| `sitting_xyz` | 0.0080418 | 0.0102861 | 0.0102818 | **0.0096732** | 0.0102965 |

## Diagnosis

The diagnostic principal-point substitution improves `walking_rpy` by 13.38% relative to Omega fixed-K and leaves it only 3.61% above GT-K. It improves `sitting_xyz` by 5.96%, although that sequence retains a 20.29% gap. Principal point is therefore a major component of the `walking_rpy` unknown-K error but not the only component of the `sitting_xyz` error.

## Four-Parameter BA Pilot

The new mode uses the existing CUDA model-0 pinhole Schur branch and wraps it with:

- independent log-focal and pixel-principal-point curvature checks;
- per-update and total trust regions for focal and principal-point components;
- prior-aware reprojection acceptance; and
- rollback of poses, disparities, and intrinsics on a rejected proposal.

`walking_rpy` accepted 21/426 proposals and moved the internal tracker principal point from `(32.0, 24.0)` to `(32.163, 24.399)`, improving over focal-only BA by 3.81%. It did not reach the diagnostic principal-point setting, indicating that the present post-solve trust-region wrapper is too conservative to close the full gap.

`sitting_xyz` accepted 0/319 proposals because the regularized loss gate rejected them. Its four-parameter result is effectively the fixed-K trajectory and is 0.10% worse from normal run variation.

## CUDA Prior Extension

A shared 4D Gaussian prior was then added directly to the CUDA Schur calibration
block. This is a true joint pose-depth-intrinsics prior, not a post-solve penalty.
The original post-solve wrapper and the CUDA-prior version both use the same trust
regions and rollback rules.

| Sequence | Fixed-K | Post-solve full-pinhole | CUDA 4D prior | CUDA 4D prior, focal locked |
|---|---:|---:|---:|---:|
| `walking_rpy` | 0.0363391 | 0.0347399 | **0.0332852** | 0.0368946 |
| `sitting_xyz` | 0.0102861 | 0.0102965 | **0.0100145** | 0.0102040 |

The CUDA-prior version accepts 58/430 proposals on `walking_rpy` and reduces the
fixed-K error by 8.40%; it is the best online unknown-K result in this pilot. The
focal-locked ablation is worse, proving that principal point cannot be optimized
independently with the current dynamic trajectory state. On `sitting_xyz`, only 1/319
proposal is accepted and the gain is limited to 2.64% over fixed-K.

## Global-K Propagation and Early-Freeze Test

The initial four-parameter implementation propagated the current `fx/fy` to each new
keyframe but left new-frame `cx/cy` at the original Omega value. This violates the
single-camera assumption and makes per-frame reprojection/cache consumers disagree
with the global Schur state. The implementation now propagates all four shared
parameters to every new keyframe. The following full-sequence reruns use that fix;
the early-freeze variant disables only further K updates after keyframe 120, while
ordinary pose/depth BA continues.

| Sequence | GT-K | Omega fixed-K | Earlier continuous 4D prior | Corrected continuous 4D prior | Corrected 4D prior, freeze@120 |
|---|---:|---:|---:|---:|---:|
| `walking_rpy` | 0.0303817 | 0.0363391 | 0.0332852 | **0.0320648** | 0.0327076 |
| `sitting_xyz` | 0.0080418 | 0.0102861 | **0.0100145** | 0.0101572 | 0.0101149 |

On `walking_rpy`, coherent global-K propagation improves the earlier continuous
result by 3.67%; freezing at 120 keyframes then loses 2.00% relative to the corrected
continuous version. On `sitting_xyz`, the difference between corrected continuous
and freeze is only 0.42%; that run accepts exactly one early calibration update, so it
does not support a general freeze rule. The per-frame cached K now exactly matches the
final CSV K in both sequences. Keep continuous constrained updates as the canonical
four-parameter pilot and do not extend the early-freeze switch to the eight-sequence
evaluation.

## K-only Background-Observation Control

This control tests whether the K update should see only a putatively static
background subset. It keeps the ordinary DROID-W pose/depth BA graph and weights
unchanged. Only the auxiliary four-parameter Schur proposal and its acceptance
objective use source pixels with cached Omega uncertainty at most `0.93` and edges
spanning at least four keyframes. The restricted CUDA solve runs on temporary
pose/depth copies and only an accepted K is copied back, so the control does not
silently alter the main trajectory optimization.

| Sequence | Corrected continuous 4D prior | K-only background subset | Change |
|---|---:|---:|---:|
| `walking_rpy` | **0.0320648** | 0.0358070 | +11.67% |
| `sitting_xyz` | **0.0101572** | 0.0102631 | +1.04% |

The hypothesis is rejected in this form. `walking_rpy` retains only 9.2% of the
original calibration weight on average (0--20.4% across calls), accepts 16 updates
instead of 57, and ends at `cy=24.061` rather than `24.772`; it consequently loses
most of the continuous method's gain. `sitting_xyz` retains 20.7% of the weight and
accepts two updates instead of one, but still regresses. Hardly discarding dynamic
or short-span observations removes useful coupled pose--depth--K information; it is
not a suitable replacement for the current joint observation model.

## Soft Calibration-Observation Weight

The next control retains every calibration edge and continuously attenuates its
contribution using Omega uncertainty and keyframe span. Pixel weights map from 1.0
at uncertainty `0.85` to 0.60 at `1.00`; edge-span weights map from 0.75 at span 1
to 1.0 at span 4. The ordinary tracker graph is unchanged and the CUDA solve remains
joint over pose, depth, and all four intrinsics.

| Sequence | Corrected continuous 4D prior | Soft observation weight | Change |
|---|---:|---:|---:|
| `walking_rpy` | 0.0320648 | **0.0319940** | -0.22% |
| `sitting_xyz` | **0.0101572** | 0.0101662 | +0.09% |

The clean rerun uses the same seed 43 and an isolated output directory. Mean
calibration weight coverage is 64.4% on `walking_rpy` and 68.5% on `sitting_xyz`
(minimum 56.1% and 62.9%); all calibration edges remain present at every call.
Accepted K updates are unchanged on walking (57/426) and sitting (1/319). This is
therefore a neutral control: smooth weighting is less destructive than hard subset
masking, but does not yet resolve the K--pose--depth ambiguity.

## Decision

Do not extend four-parameter BA to all eight TUM sequences yet. The direct prior
block and coherent global-K propagation improve the calibration-sensitive pilot, but
the corrected method still regresses on `sitting_xyz` and trails the diagnostic
GT-principal-point upper bound substantially. The remaining work is not another
freeze-threshold sweep: it requires a more reliable calibration observation model
that breaks the K-pose-depth ambiguity before online dynamic tracking.
