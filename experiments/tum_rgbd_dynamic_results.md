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

## Omega Patch-Token Evidence-Floor V11 Spot Check

Date: 2026-07-08

Config:

```text
configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz_omega_patch_token_uncertainty_v11_evidence_floor020.yaml
```

This run keeps the previous Omega + Edge-DTF soft-cycle setting and adds the Bonn-selected dense patch-token uncertainty with conditional Edge-DTF gating and `evidence_floor.fallback_min_gate=0.20`.

Command:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz_omega_patch_token_uncertainty_v11_evidence_floor020.yaml
```

ATE RMSE in meters. FPS is from `timer_summary.csv`.

| Sequence | Method | Frames | KF RMSE | Full RMSE | KF Mean | Full Mean | Tracking FPS | Full FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| freiburg3_walking_xyz | Omega + Edge-DTF soft cycle | 858 | 0.012222 | 0.012143 | 0.010572 | 0.010480 | 10.22 | 6.35 |
| freiburg3_walking_xyz | Omega patch-token v11 evidence floor | 858 | 0.012217 | 0.012142 | 0.010567 | 0.010478 | 8.11 | 5.47 |

Delta versus previous Omega + Edge-DTF soft cycle:

| Metric | Delta |
| --- | ---: |
| KF RMSE | -0.000005 m / -0.04% |
| Full RMSE | -0.000001 m / -0.01% |
| Tracking FPS | -2.11 FPS / -20.65% |
| Full FPS | -0.88 FPS / -13.86% |

Output files:

- `/data1/czy/Output/DROID-omega/TUM_RGBD/freiburg3_walking_xyz_omega_patch_token_uncertainty_v11_evidence_floor020/traj/metrics_kf_traj.txt`
- `/data1/czy/Output/DROID-omega/TUM_RGBD/freiburg3_walking_xyz_omega_patch_token_uncertainty_v11_evidence_floor020/traj/metrics_full_traj.txt`
- `/data1/czy/Output/DROID-omega/TUM_RGBD/freiburg3_walking_xyz_omega_patch_token_uncertainty_v11_evidence_floor020/timer_summary.csv`
- `/data1/czy/Output/DROID-omega/TUM_RGBD/freiburg3_walking_xyz_omega_patch_token_uncertainty_v11_evidence_floor020/final_point_cloud.ply`

Takeaway:

- On `freiburg3_walking_xyz`, patch-token v11 is accuracy-neutral relative to the earlier Omega + Edge-DTF setting, but slower. This suggests the dense patch-token uncertainty should be reported as a dynamic-scene robustness/calibration component, not as a speed improvement, unless used from cache or at reduced update rate.

## Omega Patch-Token V17 Frozen Cache Spot Check

Date: 2026-07-10

Configs:

```text
configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz_omega_cache_write.yaml
configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml
```

Commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz_omega_cache_write.yaml
conda run -n droid-w python run.py --config configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml
```

V17 uses cached Omega patch tokens only, disables Omega depth/uncertainty replacement, and applies residual-gated patch-token dynamic factor suppression with soft Edge-DTF `edge_weight_strength=0.10`.

ATE RMSE in meters. FPS is from the final v17-cache run, not from cache generation.

| Sequence | Method | Frames | KF RMSE | Full RMSE | KF Mean | Full Mean | Tracking FPS | Full FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| freiburg3_walking_xyz | Omega + Edge-DTF soft cycle | 858 | 0.012222 | 0.012143 | 0.010572 | 0.010480 | 10.22 | 6.35 |
| freiburg3_walking_xyz | Omega patch-token v11 evidence floor | 858 | 0.012217 | 0.012142 | 0.010567 | 0.010478 | 8.11 | 5.47 |
| freiburg3_walking_xyz | Omega patch-token v17 cache | 858 | 0.012266 | 0.012169 | 0.010616 | 0.010503 | 14.46 | 7.71 |

Delta versus previous Omega + Edge-DTF soft cycle:

| Metric | Delta |
| --- | ---: |
| KF RMSE | +0.000044 m / +0.36% |
| Full RMSE | +0.000026 m / +0.21% |
| Tracking FPS | +4.24 FPS / +41.49% |
| Full FPS | +1.36 FPS / +21.42% |

Cache generation runtime:

| Stage | Frames | Tracking FPS | Full FPS |
| --- | ---: | ---: | ---: |
| Omega cache write | 858 | 10.27 | 6.46 |

Output files:

- `/data1/czy/Output/DROID-omega/TUM_RGBD/freiburg3_walking_xyz_omega_patch_token_uncertainty_v17_patchonly_soft_edge010/traj/metrics_kf_traj.txt`
- `/data1/czy/Output/DROID-omega/TUM_RGBD/freiburg3_walking_xyz_omega_patch_token_uncertainty_v17_patchonly_soft_edge010/traj/metrics_full_traj.txt`
- `/data1/czy/Output/DROID-omega/TUM_RGBD/freiburg3_walking_xyz_omega_patch_token_uncertainty_v17_patchonly_soft_edge010/timer_summary.csv`

Takeaway:

- On `freiburg3_walking_xyz`, frozen v17 is within `0.3%` full RMSE of the prior online Omega setting and is faster in the final run because Omega inference is moved to cache generation.
