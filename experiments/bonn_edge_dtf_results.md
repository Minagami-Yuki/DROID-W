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

## Commands

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_edge_dtf015.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf005.yaml
```

Configs:

- `configs/Dynamic/Bonn/bonn_person_tracking_omega_edge_dtf015.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf015.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_edge_dtf005.yaml`

## Results

ATE RMSE in meters.

| Scene | Variant | KF RMSE | Full RMSE | KF Mean | Full Mean |
| --- | --- | ---: | ---: | ---: | ---: |
| bonn_person_tracking | DROID-W | 0.033933 | 0.034278 | 0.028794 | 0.028837 |
| bonn_person_tracking | Omega best | 0.033721 | 0.033778 | 0.028989 | 0.028649 |
| bonn_person_tracking | Omega + Edge-DTF 0.15 | 0.033500 | 0.033652 | 0.028925 | 0.028662 |
| bonn_crowd2 | DROID-W | 0.019121 | 0.018004 | 0.016788 | 0.015895 |
| bonn_crowd2 | Omega best | 0.018728 | 0.017846 | 0.016978 | 0.016107 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 | 0.019635 | 0.018819 | 0.017720 | 0.016868 |
| bonn_crowd2 | Omega + Edge-DTF 0.05 | 0.019381 | 0.018554 | 0.017543 | 0.016719 |

Delta versus Omega best:

| Scene | Variant | Delta KF RMSE | Delta Full RMSE |
| --- | --- | ---: | ---: |
| bonn_person_tracking | Omega + Edge-DTF 0.15 | -0.000221 | -0.000126 |
| bonn_crowd2 | Omega + Edge-DTF 0.15 | +0.000907 | +0.000973 |
| bonn_crowd2 | Omega + Edge-DTF 0.05 | +0.000653 | +0.000708 |

## Interpretation

- `bonn_person_tracking` improves beyond the previous best: full RMSE `0.033778 -> 0.033652`.
- `bonn_crowd2` degrades for both tested Edge-DTF strengths. This suggests the current ungated edge prior over-penalizes dynamic crowd edges.
- The next useful version should gate Edge-DTF with static-edge confidence, for example:
  - only use edges with high Omega confidence;
  - suppress edges with high DROID uncertainty;
  - suppress edges whose forward/backward Edge-DTF residual is not cycle-consistent;
  - apply Edge-DTF only to low-motion/background regions.

## Metric Files

- `Outputs/Bonn/bonn_person_tracking_omega_edge_dtf015/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_person_tracking_omega_edge_dtf015/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_edge_dtf015/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_edge_dtf015/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_edge_dtf005/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_edge_dtf005/traj/metrics_full_traj.txt`
