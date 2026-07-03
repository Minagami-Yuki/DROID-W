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

## Commands

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_edge_dtf015.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf005.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_edge_dtf015_ogate086.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015_ogate086.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf005_ogate086.yaml
```

Configs:

- `configs/Dynamic/Bonn/bonn_person_tracking_omega_edge_dtf015.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf005.yaml`
- `configs/Dynamic/Bonn/bonn_person_tracking_omega_edge_dtf015_ogate086.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015_ogate086.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf005_ogate086.yaml`

## Results

ATE RMSE in meters.

| Scene | Variant | KF RMSE | Full RMSE | KF Mean | Full Mean |
| --- | --- | ---: | ---: | ---: | ---: |
| bonn_person_tracking | DROID-W | 0.033933 | 0.034278 | 0.028794 | 0.028837 |
| bonn_person_tracking | Omega best | 0.033721 | 0.033778 | 0.028989 | 0.028649 |
| bonn_person_tracking | Omega + Edge-DTF 0.15 | 0.033500 | 0.033652 | 0.028925 | 0.028662 |
| bonn_person_tracking | Omega + Edge-DTF 0.15 + Omega gate 0.86 | 0.033553 | 0.033661 | 0.028957 | 0.028647 |
| bonn_crowd2 | DROID-W | 0.019121 | 0.018004 | 0.016788 | 0.015895 |
| bonn_crowd2 | Omega best | 0.018728 | 0.017846 | 0.016978 | 0.016107 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 | 0.019635 | 0.018819 | 0.017720 | 0.016868 |
| bonn_crowd2 | Omega + Edge-DTF 0.05 | 0.019381 | 0.018554 | 0.017543 | 0.016719 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 + Omega gate 0.86 | 0.019554 | 0.018706 | 0.017664 | 0.016822 |
| bonn_crowd2 | Omega + Edge-DTF 0.05 + Omega gate 0.86 | 0.019548 | 0.018474 | 0.017724 | 0.016646 |

Delta versus Omega best:

| Scene | Variant | Delta KF RMSE | Delta Full RMSE |
| --- | --- | ---: | ---: |
| bonn_person_tracking | Omega + Edge-DTF 0.15 | -0.000221 | -0.000126 |
| bonn_person_tracking | Omega + Edge-DTF 0.15 + Omega gate 0.86 | -0.000168 | -0.000116 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 | +0.000907 | +0.000973 |
| bonn_crowd2 | Omega + Edge-DTF 0.05 | +0.000653 | +0.000708 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 + Omega gate 0.86 | +0.000826 | +0.000860 |
| bonn_crowd2 | Omega + Edge-DTF 0.05 + Omega gate 0.86 | +0.000820 | +0.000628 |

## Interpretation

- `bonn_person_tracking` improves beyond the previous best: full RMSE `0.033778 -> 0.033652`.
- `bonn_crowd2` degrades for both tested Edge-DTF strengths. This suggests the current ungated edge prior over-penalizes dynamic crowd edges.
- Omega uncertainty gating is a useful safety mechanism but not enough for crowded dynamic scenes:
  - `person_tracking` remains better than Omega best, but the ungated Edge-DTF 0.15 variant is still the best tested setting.
  - `crowd2` improves slightly versus ungated Edge-DTF, but remains worse than Omega best.
- The next useful version should add a stronger motion/static selector, for example suppressing edges whose forward/backward Edge-DTF residual is not cycle-consistent, or applying Edge-DTF only to low-motion/background regions.

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
