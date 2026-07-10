# 7Scenes Tracking Results

Date: 2026-07-04

## Setup

This adds a minimal loader for the original Microsoft 7-Scenes RGB-D format:

```text
frame-000000.color.png
frame-000000.depth.png
frame-000000.pose.txt
```

The first smoke/full tracking test uses `chess/seq-01` from:

```text
/data1/czy/datasets/7scenes/chess/seq-01
```

Config:

```text
configs/Static/7Scenes/chess_seq01.yaml
```

Camera settings:

- RGB/depth size: `640 x 480`
- Intrinsics: `fx=585.0`, `fy=585.0`, `cx=320.0`, `cy=240.0`
- PNG depth scale: `1000.0`

## Command

```bash
conda run -n droid-w python run.py --config configs/Static/7Scenes/chess_seq01.yaml
```

## Result

ATE RMSE in meters.

| Sequence | Frames | KF RMSE | Full RMSE | KF Mean | Full Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| chess/seq-01 | 1000 | 0.037822 | 0.037051 | 0.033394 | 0.033077 |

Runtime summary:

- Tracking: `1000` frames, `63.24s`, `15.81 FPS`
- Full trajectory filling: `27.34s`
- Metric depth estimation: `173` calls, `15.25s`
- Final global BA: `14.27s`

Metric files:

- `Outputs/7Scenes/chess_seq01/traj/metrics_kf_traj.txt`
- `Outputs/7Scenes/chess_seq01/traj/metrics_full_traj.txt`

## Notes

- This run uses the base DROID-W tracking configuration with Omega and Edge-DTF disabled.
- The loader normalizes poses relative to the first frame, matching the existing TUM/Bonn loader behavior.

## Omega + Edge-DTF Soft Cycle

Latest improved setting:

```text
configs/Static/7Scenes/chess_seq01_omega_edge_dtf015_cycle008_soft050.yaml
```

This keeps the same `chess/seq-01` input sequence and enables:

- online VGGT-Omega depth/confidence
- aligned Omega depth blend, `blend_alpha=0.10`
- soft Omega uncertainty edge weighting, `edge_weight_strength=0.25`
- Edge-DTF uncertainty calibration, `edge_weight_strength=0.15`
- pair cycle consistency with `max_asymmetry=0.08` and `failed_scale=0.5`

Command:

```bash
conda run -n droid-w python run.py --config configs/Static/7Scenes/chess_seq01_omega_edge_dtf015_cycle008_soft050.yaml
```

ATE RMSE in meters.

| Sequence | Method | Frames | KF RMSE | Full RMSE | KF Mean | Full Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chess/seq-01 | DROID-W baseline | 1000 | 0.037822 | 0.037051 | 0.033394 | 0.033077 |
| chess/seq-01 | Omega + Edge-DTF soft cycle | 1000 | 0.037764 | 0.037005 | 0.033363 | 0.033024 |

Delta versus baseline:

| Metric | Delta |
| --- | ---: |
| KF RMSE | -0.000058 m / -0.15% |
| Full RMSE | -0.000047 m / -0.13% |
| KF Mean | -0.000031 m / -0.09% |
| Full Mean | -0.000053 m / -0.16% |

Runtime summary:

| Method | Tracking FPS | Overall FPS |
| --- | ---: | ---: |
| DROID-W baseline | 15.81 | 8.16 |
| Omega + Edge-DTF soft cycle | 11.01 | 6.64 |

Metric files:

- `Outputs/7Scenes/chess_seq01_omega_edge_dtf015_cycle008_soft050/traj/metrics_kf_traj.txt`
- `Outputs/7Scenes/chess_seq01_omega_edge_dtf015_cycle008_soft050/traj/metrics_full_traj.txt`

## Omega Patch-Token Evidence-Floor V11 Spot Check

Date: 2026-07-08

Config:

```text
configs/Static/7Scenes/chess_seq01_omega_patch_token_uncertainty_v11_evidence_floor020.yaml
```

This run keeps the previous Omega + Edge-DTF soft-cycle setting and adds the Bonn-selected dense patch-token uncertainty with conditional Edge-DTF gating and `evidence_floor.fallback_min_gate=0.20`.

Command:

```bash
conda run -n droid-w python run.py --config configs/Static/7Scenes/chess_seq01_omega_patch_token_uncertainty_v11_evidence_floor020.yaml
```

ATE RMSE in meters. FPS is from `timer_summary.csv`.

| Sequence | Method | Frames | KF RMSE | Full RMSE | KF Mean | Full Mean | Tracking FPS | Full FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess/seq-01 | DROID-W baseline | 1000 | 0.037822 | 0.037051 | 0.033394 | 0.033077 | 15.81 | 8.16 |
| chess/seq-01 | Omega + Edge-DTF soft cycle | 1000 | 0.037764 | 0.037005 | 0.033363 | 0.033024 | 11.01 | 6.64 |
| chess/seq-01 | Omega patch-token v11 evidence floor | 1000 | 0.037730 | 0.036968 | 0.033326 | 0.032994 | 10.93 | 6.63 |

Delta versus DROID-W baseline:

| Metric | Delta |
| --- | ---: |
| KF RMSE | -0.000092 m / -0.24% |
| Full RMSE | -0.000083 m / -0.22% |
| Tracking FPS | -4.88 FPS / -30.87% |
| Full FPS | -1.53 FPS / -18.75% |

Delta versus previous Omega + Edge-DTF soft cycle:

| Metric | Delta |
| --- | ---: |
| KF RMSE | -0.000034 m / -0.09% |
| Full RMSE | -0.000037 m / -0.10% |
| Tracking FPS | -0.08 FPS / -0.73% |
| Full FPS | -0.01 FPS / -0.15% |

Output files:

- `/data1/czy/Output/DROID-omega/7Scenes/chess_seq01_omega_patch_token_uncertainty_v11_evidence_floor020/traj/metrics_kf_traj.txt`
- `/data1/czy/Output/DROID-omega/7Scenes/chess_seq01_omega_patch_token_uncertainty_v11_evidence_floor020/traj/metrics_full_traj.txt`
- `/data1/czy/Output/DROID-omega/7Scenes/chess_seq01_omega_patch_token_uncertainty_v11_evidence_floor020/timer_summary.csv`
- `/data1/czy/Output/DROID-omega/7Scenes/chess_seq01_omega_patch_token_uncertainty_v11_evidence_floor020/final_point_cloud.ply`

Takeaway:

- On static `chess/seq-01`, patch-token v11 does not hurt tracking accuracy and slightly improves full RMSE compared with both DROID-W and the earlier Omega + Edge-DTF setting. Runtime is essentially unchanged versus Omega + Edge-DTF, but remains slower than the original DROID-W baseline because Omega inference is online.

## Omega Patch-Token V17 Frozen Cache Spot Check

Date: 2026-07-10

Configs:

```text
configs/Static/7Scenes/chess_seq01_omega_cache_write.yaml
configs/Static/7Scenes/chess_seq01_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml
```

Commands:

```bash
conda run -n droid-w python run.py --config configs/Static/7Scenes/chess_seq01_omega_cache_write.yaml
conda run -n droid-w python run.py --config configs/Static/7Scenes/chess_seq01_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml
```

V17 freezes the Bonn-selected cache-based setting:

- cached Omega patch tokens only; no Omega depth replacement and no Omega uncertainty replacement
- residual-gated patch-token dynamic factor suppression
- soft Edge-DTF edge covariance with `edge_weight_strength=0.10`

ATE RMSE in meters. FPS is from the final v17-cache run, not from cache generation.

| Sequence | Method | Frames | KF RMSE | Full RMSE | KF Mean | Full Mean | Tracking FPS | Full FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chess/seq-01 | DROID-W baseline | 1000 | 0.037822 | 0.037051 | 0.033394 | 0.033077 | 15.81 | 8.16 |
| chess/seq-01 | Omega patch-token v17 cache | 1000 | 0.037811 | 0.037041 | 0.033384 | 0.033068 | 15.76 | 8.09 |

Delta versus DROID-W baseline:

| Metric | Delta |
| --- | ---: |
| KF RMSE | -0.000011 m / -0.03% |
| Full RMSE | -0.000010 m / -0.03% |
| Tracking FPS | -0.05 FPS / -0.32% |
| Full FPS | -0.07 FPS / -0.86% |

Cache generation runtime:

| Stage | Frames | Tracking FPS | Full FPS |
| --- | ---: | ---: | ---: |
| Omega cache write | 1000 | 11.08 | 6.75 |

Output files:

- `/data1/czy/Output/DROID-omega/7Scenes/chess_seq01_omega_patch_token_uncertainty_v17_patchonly_soft_edge010/traj/metrics_kf_traj.txt`
- `/data1/czy/Output/DROID-omega/7Scenes/chess_seq01_omega_patch_token_uncertainty_v17_patchonly_soft_edge010/traj/metrics_full_traj.txt`
- `/data1/czy/Output/DROID-omega/7Scenes/chess_seq01_omega_patch_token_uncertainty_v17_patchonly_soft_edge010/timer_summary.csv`

Takeaway:

- On static `chess/seq-01`, frozen v17 is effectively accuracy-neutral relative to DROID-W and keeps the final tracking FPS close to baseline once Omega priors are cached.
