# Bonn Edge-DTF Uncertainty Results

Date: 2026-07-03

## Motivation

This experiment adds an EPO-inspired edge-to-distance-transform-field prior to DROID-W without changing the CUDA DBA objective. The implementation follows the idea of using image edges and distance fields as a trackless structural consistency signal, but injects it conservatively as a cached per-factor edge weight.

## Implementation

Files:

- `src/utils/edge_dtf_prior.py`: computes Sobel edge maps and normalized distance transform fields at DROID-W BA resolution.
- `src/depth_video.py`: stores per-keyframe edge maps and DTF maps; exposes `edge_dtf_weight_from_coords`.
- `src/factor_graph.py`: computes a cached Edge-DTF factor weight when adding graph factors and multiplies it into recurrent update weights before BA.
- `configs/droid_w.yaml`: adds default-off `edge_dtf_prior` config.
- `src/utils/mono_priors/metric_depth_estimators.py`: prefers local Metric3D torch hub cache when available, avoiding network-dependent experiments.

The Edge-DTF factor weight is computed as:

```text
edge_weight(i -> j, p) = 1 - strength * Edge_i(p) * DTF_j(project_i_to_j(p))
```

The weight is cached at factor creation time, so baseline runtime is unchanged when disabled and enabled runtime stays close to the Omega best setting.

Follow-up static-edge gating adds an optional source-edge mask:

```text
source_edge = Edge_i(p) * 1[OmegaUncertainty_i(p) <= threshold]
```

The gate is disabled by default and only enabled in the `_ogate086` configs.

Cycle-consistent Edge-DTF adds a pair-level selector:

```text
mean_fwd = mean(Edge_i(p) * DTF_j(project_i_to_j(p))) / mean(Edge_i(p))
mean_rev = mean(Edge_j(p) * DTF_i(project_j_to_i(p))) / mean(Edge_j(p))
use_edge_dtf(i, j) = 1[abs(mean_fwd - mean_rev) <= max_asymmetry]
```

An optional `min_mean_residual` keeps only bidirectional high-residual edge factors. This is disabled by default and only used in the `_min004` config.

Soft cycle-consistent Edge-DTF keeps the same pair-level selector but attenuates rejected edges instead of removing them:

```text
source_edge = Edge_i(p) * (1.0 if cycle_consistent(i, j) else failed_scale)
```

`failed_scale=0.5` is used for the `_cycle008_soft050` configs. This keeps the EPO-inspired structural prior active as a calibration signal while reducing the failure mode where hard pair gating removes useful single-object dynamic constraints.

Pixel cycle-consistent Edge-DTF was also tested by comparing a per-pixel forward edge-DTF residual to an approximate reverse residual sampled at the target projection. It is available through `cycle_consistency.mode: pixel`, but the first Bonn tests were worse than pair-level soft gating.

## Commands

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_edge_dtf015.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf005.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_edge_dtf015_ogate086.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015_ogate086.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf005_ogate086.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_edge_dtf015_cycle008.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015_cycle008.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015_cycle008_min004.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_edge_dtf015_pcycle012.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015_pcycle012.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_edge_dtf015_cycle008_soft050.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015_cycle008_soft050.yaml
```

Configs:

- `configs/Dynamic/Bonn/bonn_person_tracking_omega_edge_dtf015.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf005.yaml`
- `configs/Dynamic/Bonn/bonn_person_tracking_omega_edge_dtf015_ogate086.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015_ogate086.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf005_ogate086.yaml`
- `configs/Dynamic/Bonn/bonn_person_tracking_omega_edge_dtf015_cycle008.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015_cycle008.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015_cycle008_min004.yaml`
- `configs/Dynamic/Bonn/bonn_person_tracking_omega_edge_dtf015_pcycle012.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015_pcycle012.yaml`
- `configs/Dynamic/Bonn/bonn_person_tracking_omega_edge_dtf015_cycle008_soft050.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015_cycle008_soft050.yaml`

## Results

ATE RMSE in meters.

| Scene | Variant | KF RMSE | Full RMSE | KF Mean | Full Mean |
| --- | --- | ---: | ---: | ---: | ---: |
| bonn_person_tracking | DROID-W | 0.033933 | 0.034278 | 0.028794 | 0.028837 |
| bonn_person_tracking | Omega best | 0.033721 | 0.033778 | 0.028989 | 0.028649 |
| bonn_person_tracking | Omega + Edge-DTF 0.15 | 0.033500 | 0.033652 | 0.028925 | 0.028662 |
| bonn_person_tracking | Omega + Edge-DTF 0.15 + Omega gate 0.86 | 0.033553 | 0.033661 | 0.028957 | 0.028647 |
| bonn_person_tracking | Omega + Edge-DTF 0.15 + cycle asym 0.08 | 0.033618 | 0.033792 | 0.028959 | 0.028715 |
| bonn_person_tracking | Omega + Edge-DTF 0.15 + pixel cycle 0.12 | 0.033930 | 0.033980 | 0.029173 | 0.028830 |
| bonn_person_tracking | Omega + Edge-DTF 0.15 + soft cycle 0.08/0.5 | 0.033462 | 0.033622 | 0.028902 | 0.028661 |
| bonn_crowd2 | DROID-W | 0.019121 | 0.018004 | 0.016788 | 0.015895 |
| bonn_crowd2 | Omega best | 0.018728 | 0.017846 | 0.016978 | 0.016107 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 | 0.019635 | 0.018819 | 0.017720 | 0.016868 |
| bonn_crowd2 | Omega + Edge-DTF 0.05 | 0.019381 | 0.018554 | 0.017543 | 0.016719 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 + Omega gate 0.86 | 0.019554 | 0.018706 | 0.017664 | 0.016822 |
| bonn_crowd2 | Omega + Edge-DTF 0.05 + Omega gate 0.86 | 0.019548 | 0.018474 | 0.017724 | 0.016646 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 + cycle asym 0.08 | 0.019146 | 0.018214 | 0.017333 | 0.016393 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 + cycle asym 0.08 + min residual 0.04 | 0.019533 | 0.018205 | 0.017005 | 0.016030 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 + pixel cycle 0.12 | 0.019562 | 0.018945 | 0.017758 | 0.017092 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 + soft cycle 0.08/0.5 | 0.019371 | 0.017985 | 0.016972 | 0.015784 |

Delta versus Omega best:

| Scene | Variant | Delta KF RMSE | Delta Full RMSE |
| --- | --- | ---: | ---: |
| bonn_person_tracking | Omega + Edge-DTF 0.15 | -0.000221 | -0.000126 |
| bonn_person_tracking | Omega + Edge-DTF 0.15 + Omega gate 0.86 | -0.000168 | -0.000116 |
| bonn_person_tracking | Omega + Edge-DTF 0.15 + cycle asym 0.08 | -0.000103 | +0.000014 |
| bonn_person_tracking | Omega + Edge-DTF 0.15 + pixel cycle 0.12 | +0.000209 | +0.000202 |
| bonn_person_tracking | Omega + Edge-DTF 0.15 + soft cycle 0.08/0.5 | -0.000259 | -0.000155 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 | +0.000907 | +0.000973 |
| bonn_crowd2 | Omega + Edge-DTF 0.05 | +0.000653 | +0.000708 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 + Omega gate 0.86 | +0.000826 | +0.000860 |
| bonn_crowd2 | Omega + Edge-DTF 0.05 + Omega gate 0.86 | +0.000820 | +0.000628 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 + cycle asym 0.08 | +0.000418 | +0.000368 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 + cycle asym 0.08 + min residual 0.04 | +0.000805 | +0.000359 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 + pixel cycle 0.12 | +0.000834 | +0.001100 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 + soft cycle 0.08/0.5 | +0.000643 | +0.000140 |

## Full Bonn Soft Cycle Sweep

Command template:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/<sequence>_omega_edge_dtf015_cycle008_soft050.yaml
```

ATE RMSE in meters. `soft vs` columns use full trajectory RMSE; negative is better.

| Sequence | DROID-W Full | Omega Best Full | Soft Cycle Full | Soft vs Omega Best | Soft vs DROID-W |
| --- | ---: | ---: | ---: | ---: | ---: |
| bonn_balloon | 0.026447 | 0.026513 | 0.026530 | +0.06% | +0.31% |
| bonn_balloon2 | 0.024623 | 0.024804 | 0.024729 | -0.30% | +0.43% |
| bonn_crowd | 0.013215 | 0.013787 | 0.013723 | -0.47% | +3.84% |
| bonn_crowd2 | 0.018004 | 0.017846 | 0.017985 | +0.78% | -0.11% |
| bonn_moving_nonobstructing_box | 0.014748 | 0.015095 | 0.015298 | +1.34% | +3.73% |
| bonn_moving_nonobstructing_box2 | 0.023466 | 0.023015 | 0.023023 | +0.04% | -1.89% |
| bonn_person_tracking | 0.034278 | 0.033778 | 0.033622 | -0.46% | -1.91% |
| bonn_person_tracking2 | 0.029595 | 0.029436 | 0.029346 | -0.31% | -0.84% |

Summary:

- Mean full RMSE: DROID-W `0.023047`, Omega best `0.023034`, soft cycle `0.023032`.
- Soft cycle wins on 4/8 sequences against Omega best and 4/8 against DROID-W.
- The soft cycle setting is roughly tied with Omega best on average (`-0.009%`), but its gains are concentrated on person/crowd-like sequences and it degrades `bonn_moving_nonobstructing_box`.

## Interpretation

- `bonn_person_tracking` improves beyond the previous best: full RMSE `0.033778 -> 0.033652`.
- `bonn_crowd2` degrades for both tested Edge-DTF strengths. This suggests the current ungated edge prior over-penalizes dynamic crowd edges.
- Omega uncertainty gating is a useful safety mechanism but not enough for crowded dynamic scenes:
  - `person_tracking` remains better than Omega best, but the ungated Edge-DTF 0.15 variant is still the best tested setting.
  - `crowd2` improves slightly versus ungated Edge-DTF, but remains worse than Omega best.
- Cycle-consistent Edge-DTF is a better selector than Omega uncertainty gating for crowded scenes:
  - `crowd2` full RMSE improves from ungated `0.018819` and Omega-gated `0.018706` to `0.018214`.
  - It still does not beat Omega best `0.017846`, so the current pair-level selector is useful but insufficient.
  - `person_tracking` loses the previous Edge-DTF gain, suggesting that pair-level cycle gating is too coarse for single-person dynamic scenes.
- `min_mean_residual=0.04` gives nearly identical `crowd2` full RMSE but worse KF RMSE, so the plain cycle asymmetry config is the more stable setting among these tests.
- Pixel-cycle gating is negative on both tested sequences and should not be the next default direction.
- Soft pair-cycle gating is the best tested setting for `bonn_person_tracking`: full RMSE `0.033778 -> 0.033622`.
- Full Bonn results show soft pair-cycle is not a universal replacement for Omega best. It is better framed as a selective uncertainty calibration mechanism that should be activated when the edge-DTF prior is reliable, rather than applied uniformly to every dynamic pattern.

## Metric Files

- `Outputs/Bonn/bonn_person_tracking_omega_edge_dtf015/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_person_tracking_omega_edge_dtf015/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_edge_dtf015/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_edge_dtf015/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_edge_dtf005/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_edge_dtf005/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_person_tracking_omega_edge_dtf015_ogate086/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_person_tracking_omega_edge_dtf015_ogate086/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_edge_dtf015_ogate086/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_edge_dtf015_ogate086/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_edge_dtf005_ogate086/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_edge_dtf005_ogate086/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_person_tracking_omega_edge_dtf015_cycle008/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_person_tracking_omega_edge_dtf015_cycle008/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_edge_dtf015_cycle008/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_edge_dtf015_cycle008/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_edge_dtf015_cycle008_min004/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_edge_dtf015_cycle008_min004/traj/metrics_full_traj.txt`
