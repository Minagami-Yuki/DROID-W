# TUM RGB-D Dynamic Results

Date: 2026-07-04

## Omega + Edge-DTF Soft Cycle

Sequence:

```text
/data1/czy/datasets/TUM_RGBD/dynamic/rgbd_dataset_freiburg3_walking_xyz
```

Config:

```text
configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz_omega_edge_dtf015_cycle008_soft050.yaml
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
```

ATE RMSE in meters.

| Sequence | Frames | KF RMSE | Full RMSE | KF Mean | Full Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| freiburg3_walking_xyz | 858 | 0.012222 | 0.012143 | 0.010572 | 0.010480 |

Runtime summary:

- Tracking: `858` frames, `83.94s`, `10.22 FPS`
- Full trajectory filling: `22.25s`
- Metric depth estimation: `145` calls, `12.81s`, `11.32 FPS`
- Final global BA: `14.85s`
- Overall: `135.10s`, `6.35 FPS`

Metric files:

- `Outputs/TUM_RGBD/freiburg3_walking_xyz_omega_edge_dtf015_cycle008_soft050/traj/metrics_kf_traj.txt`
- `Outputs/TUM_RGBD/freiburg3_walking_xyz_omega_edge_dtf015_cycle008_soft050/traj/metrics_full_traj.txt`

Notes:

- This is a latest-method run only. No baseline DROID-W run for this exact TUM sequence was present under `Outputs/TUM_RGBD` at test time.
- `src/utils/datasets.py` now creates the GT-pose output directory before writing `gt_poses.txt`, matching the behavior needed for standalone dataset checks.
