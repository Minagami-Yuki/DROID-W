# Bonn DROID-W-O + Omega Fixed-K Baseline

Date: 2026-07-17. ATE is translation RMSE in metres. All runs use seed 43 and
the eight complete Bonn dynamic sequences.

## Protocol

- Baseline source: clean detached worktree of DROID-W-O commit `c3414af`.
- Initial K: the exact per-sequence Omega K used by the canonical confidence-BA
  runs, recorded in `experiments/bonn_omega_k_manifest_v1.json`.
- Baseline changes: only input/output paths, scene name, checkpoint path, fixed
  `cam.fx/fy/cx/cy`, and disabling diagnostic image export.
- No Omega depth/uncertainty prior, reliability weighting, focal/full-pinhole BA,
  calibration routing, or other DROID-omega code is loaded.
- Canonical baseline output: `/data1/czy/Output/DROID-W-O/Bonn_omega_fixed_k_baseline_v2`.

An earlier v1 run used newly re-extracted K for three sequences. Those values
differed from the canonical experiment K by up to 0.30 px, so v1 is retained only
as an audit and excluded from the comparison below.

## Full-Trajectory ATE

| Sequence | DROID-W-O + GT K | DROID-W-O + Omega fixed K | Ours: Omega + confidence BA | Ours vs fixed-K baseline |
|---|---:|---:|---:|---:|
| `balloon` | 0.026446989 | **0.022043046** | 0.022401659 | +1.63% |
| `balloon2` | **0.024622931** | 0.026718238 | 0.026742116 | +0.09% |
| `crowd` | **0.013215283** | 0.014587646 | 0.015098600 | +3.50% |
| `crowd2` | **0.018004461** | 0.022691415 | 0.020318232 | -10.46% |
| `moving_nonobstructing_box` | **0.014747677** | 0.017680265 | 0.016873716 | -4.56% |
| `moving_nonobstructing_box2` | 0.023466373 | 0.034604291 | **0.022425307** | -35.20% |
| `person_tracking` | 0.034277714 | 0.034312241 | **0.031391200** | -8.51% |
| `person_tracking2` | 0.029595351 | 0.036034631 | **0.029594133** | -17.87% |
| **Macro mean** | **0.023047097** | 0.026083972 | 0.023105620 | **-11.42%** |

Our confidence BA wins 5/8 Full-ATE comparisons against the clean fixed-K
baseline. Direct Omega K increases the DROID-W-O macro mean by 13.18% relative
to GT K, while our result is only 0.25% above the DROID-W-O GT-K mean.

## Keyframe ATE

| Sequence | DROID-W-O + GT K | DROID-W-O + Omega fixed K | Ours: Omega + confidence BA | Ours vs fixed-K baseline |
|---|---:|---:|---:|---:|
| `balloon` | 0.027787559 | **0.023468667** | 0.023735484 | +1.14% |
| `balloon2` | 0.027592747 | 0.029828776 | **0.029801522** | -0.09% |
| `crowd` | **0.015505714** | 0.015760175 | 0.015858509 | +0.62% |
| `crowd2` | **0.019121309** | 0.024339381 | 0.021756566 | -10.61% |
| `moving_nonobstructing_box` | **0.014664582** | 0.019268674 | 0.018473637 | -4.13% |
| `moving_nonobstructing_box2` | 0.025137302 | 0.038131804 | **0.024630349** | -35.41% |
| `person_tracking` | 0.033933032 | 0.033713883 | **0.030878032** | -8.41% |
| `person_tracking2` | **0.029435358** | 0.035664517 | 0.030026940 | -15.81% |
| **Macro mean** | **0.024147200** | 0.027521985 | 0.024395130 | **-11.36%** |

Our method wins 6/8 keyframe comparisons. Its keyframe macro mean is 1.03%
above the DROID-W-O GT-K reference, compared with +13.98% for direct Omega K.

## Interpretation

This baseline materially strengthens the conditional-calibration claim. Direct
Omega K is already sufficient on `balloon` and nearly tied on `balloon2`; online
calibration should not activate there. In contrast, `box2`, `person_tracking2`,
`crowd2`, and `person_tracking` expose real fixed-K degradation, and the proposed
method recovers 8.5--35.2% Full ATE on those sequences. The remaining problem is
therefore reliable routing, not making calibration BA universally active.
