# TUM RGB-D Dynamic Results

Date: 2026-07-04

## Omega + Edge-DTF Soft Cycle

Dataset root:

```text
/data1/czy/datasets/TUM_RGBD/dynamic
```

Tested configs:

```text
configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz_omega_edge_dtf015_cycle008_soft050.yaml
configs/Dynamic/TUM_RGBD/freiburg3_walking_halfsphere_omega_edge_dtf015_cycle008_soft050.yaml
configs/Dynamic/TUM_RGBD/freiburg3_walking_rpy_omega_edge_dtf015_cycle008_soft050.yaml
configs/Dynamic/TUM_RGBD/freiburg3_walking_static_omega_edge_dtf015_cycle008_soft050.yaml
```

This latest setting enables:

- online VGGT-Omega depth/confidence
- aligned Omega depth blend, `blend_alpha=0.10`
- soft Omega uncertainty edge weighting, `edge_weight_strength=0.25`
- Edge-DTF uncertainty calibration, `edge_weight_strength=0.15`
- pair cycle consistency with `max_asymmetry=0.08` and `failed_scale=0.5`

Command:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz_omega_edge_dtf015_cycle008_soft050.yaml
conda run -n droid-w python run.py --config configs/Dynamic/TUM_RGBD/freiburg3_walking_halfsphere_omega_edge_dtf015_cycle008_soft050.yaml
conda run -n droid-w python run.py --config configs/Dynamic/TUM_RGBD/freiburg3_walking_rpy_omega_edge_dtf015_cycle008_soft050.yaml
conda run -n droid-w python run.py --config configs/Dynamic/TUM_RGBD/freiburg3_walking_static_omega_edge_dtf015_cycle008_soft050.yaml
```

ATE RMSE in meters.

| Sequence | Frames | KF RMSE | Full RMSE | KF Mean | Full Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| freiburg3_walking_xyz | 858 | 0.012222 | 0.012143 | 0.010572 | 0.010480 |
| freiburg3_walking_halfsphere | 1065 | 0.016000 | 0.015706 | 0.014252 | 0.013899 |
| freiburg3_walking_rpy | 908 | 0.040519 | 0.030315 | 0.031031 | 0.023914 |
| freiburg3_walking_static | 742 | 0.005151 | 0.004750 | 0.004551 | 0.004191 |

Mean full RMSE over the four tested TUM dynamic walking sequences: `0.015728 m`.

Runtime summary:

| Sequence | Tracking FPS | Overall FPS | Metric Depth Calls |
| --- | ---: | ---: | ---: |
| freiburg3_walking_xyz | 10.22 | 6.35 | 145 |
| freiburg3_walking_halfsphere | 8.28 | 5.08 | 235 |
| freiburg3_walking_rpy | 6.11 | 4.15 | 244 |
| freiburg3_walking_static | 11.60 | 7.08 | 95 |

Metric files:

- `Outputs/TUM_RGBD/freiburg3_walking_xyz_omega_edge_dtf015_cycle008_soft050/traj/metrics_kf_traj.txt`
- `Outputs/TUM_RGBD/freiburg3_walking_xyz_omega_edge_dtf015_cycle008_soft050/traj/metrics_full_traj.txt`
- `Outputs/TUM_RGBD/freiburg3_walking_halfsphere_omega_edge_dtf015_cycle008_soft050/traj/metrics_kf_traj.txt`
- `Outputs/TUM_RGBD/freiburg3_walking_halfsphere_omega_edge_dtf015_cycle008_soft050/traj/metrics_full_traj.txt`
- `Outputs/TUM_RGBD/freiburg3_walking_rpy_omega_edge_dtf015_cycle008_soft050/traj/metrics_kf_traj.txt`
- `Outputs/TUM_RGBD/freiburg3_walking_rpy_omega_edge_dtf015_cycle008_soft050/traj/metrics_full_traj.txt`
- `Outputs/TUM_RGBD/freiburg3_walking_static_omega_edge_dtf015_cycle008_soft050/traj/metrics_kf_traj.txt`
- `Outputs/TUM_RGBD/freiburg3_walking_static_omega_edge_dtf015_cycle008_soft050/traj/metrics_full_traj.txt`

Notes:

- These are latest-method runs only. No baseline DROID-W runs for these exact TUM sequences were present under `Outputs/TUM_RGBD` at test time.
- `src/utils/datasets.py` now creates the GT-pose output directory before writing `gt_poses.txt`, matching the behavior needed for standalone dataset checks.
