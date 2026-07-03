# Bonn Depth Ablation Results

Date: 2026-07-03

## bonn_person_tracking: Metric3D depth + Omega uncertainty

Command:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_uncertainty_metric_depth.yaml
```

Config:

- Base config: `configs/Dynamic/Bonn/bonn_person_tracking.yaml`
- Scene/output: `bonn_person_tracking_omega_uncertainty_metric_depth`
- Depth source: DROID-W original `mono_prior.depth: metric3d_vit_large`
- Omega depth: disabled with `omega_prior.depth.enable: False`
- Omega uncertainty: enabled with online VGGT-Omega confidence
- VGGT-Omega repo: `thirdparty/vggt-omega`
- VGGT-Omega checkpoint: `/data1/czy/Output/DROID-W/vggt_omega_1b_512.pt`

Result:

| Scene | Variant | KF RMSE | Full RMSE | KF Mean | Full Mean | Tracking FPS | System FPS | Keyframes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_person_tracking | Metric3D depth + Omega uncertainty | 0.039549 | 0.039997 | 0.036195 | 0.036287 | 9.07 | 5.61 | 79 |
| bonn_person_tracking | Omega depth + Omega uncertainty | 0.048816 | 0.049401 | 0.043352 | 0.043557 | 11.34 | - | 78 |

Delta versus Omega depth + Omega uncertainty:

- KF RMSE: `0.048816 -> 0.039549` (`-18.98%`)
- Full RMSE: `0.049401 -> 0.039997` (`-19.04%`)

Timer summary:

| Name | Count | Total Time (s) | Average Time (s) | FPS |
| --- | ---: | ---: | ---: | ---: |
| Metric Depth Estimation | 129 | 11.335699 | 0.087874 | 11.38 |
| DINO Feature Extraction | 129 | 1.132788 | 0.008781 | 113.88 |
| Tracking | 580 | 63.938822 | 0.110239 | 9.07 |
| Final Global BA | 1 | 8.535617 | 8.535617 | 0.12 |
| Full Trajectory Filling | 1 | 18.521495 | 18.521495 | 0.05 |
| Full System | 580 | 103.464421 | 0.178387 | 5.61 |

Metric files:

- `Outputs/Bonn/bonn_person_tracking_omega_uncertainty_metric_depth/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_person_tracking_omega_uncertainty_metric_depth/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_person_tracking_omega_uncertainty_metric_depth/timer_summary.csv`

## bonn_person_tracking: soft Omega weighting and aligned depth blend

Goal:

- Push `bonn_person_tracking` below `0.034 m` full-trajectory RMSE while keeping DROID-W baseline behavior unchanged by default.

Code changes:

- `omega_prior.uncertainty.edge_weight_strength` softens Omega confidence downweighting.
- `omega_prior.uncertainty.edge_weight_power`, `min_edge_weight`, and `max_edge_weight` expose the edge-weight mapping for ablation.
- `omega_prior.depth.align_to_mono: scale` robustly aligns online Omega depth to Metric3D before `blend` mode.
- Defaults preserve the previous behavior unless the new config keys are enabled.

Commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_u_soft025_metric_depth.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_u_soft030_metric_depth.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_blend005_u_soft025.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_blend010_u_soft025.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_blend010_u_soft030.yaml
```

Results:

| Variant | KF RMSE | Full RMSE | KF Mean | Full Mean | Tracking FPS | System FPS | Keyframes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DROID-W original | 0.033933 | 0.034278 | 0.028794 | 0.028837 | 14.14 | 7.03 | 79 |
| Metric3D depth + hard Omega uncertainty | 0.039549 | 0.039997 | 0.036195 | 0.036287 | 9.07 | 5.61 | 79 |
| Metric3D depth + Omega uncertainty strength 0.25 | 0.033693 | 0.034029 | 0.028831 | 0.028812 | 9.35 | 5.69 | 79 |
| Metric3D depth + Omega uncertainty strength 0.30 | 0.033635 | 0.033979 | 0.028826 | 0.028802 | 9.51 | 5.76 | 79 |
| Aligned Omega depth blend 0.05 + uncertainty strength 0.25 | 0.033737 | 0.033843 | 0.029104 | 0.028787 | 9.38 | 5.70 | 78 |
| Aligned Omega depth blend 0.10 + uncertainty strength 0.25 | 0.033721 | 0.033778 | 0.028989 | 0.028649 | 9.35 | 5.68 | 78 |
| Aligned Omega depth blend 0.10 + uncertainty strength 0.30 | 0.033794 | 0.033852 | 0.029102 | 0.028741 | 9.28 | 5.67 | 78 |

Best result:

- Config: `configs/Dynamic/Bonn/bonn_person_tracking_omega_blend010_u_soft025.yaml`
- Output: `Outputs/Bonn/bonn_person_tracking_omega_blend010_u_soft025`
- KF RMSE: `0.033720677324765705`
- Full RMSE: `0.033777686791906604`
- Full RMSE is below the `0.034 m` target.

Delta versus DROID-W original:

- KF RMSE: `0.033933 -> 0.033721` (`-0.63%`)
- Full RMSE: `0.034278 -> 0.033778` (`-1.46%`)

Delta versus the earlier hard Omega uncertainty setting:

- KF RMSE: `0.039549 -> 0.033721` (`-14.74%`)
- Full RMSE: `0.039997 -> 0.033778` (`-15.55%`)
