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
