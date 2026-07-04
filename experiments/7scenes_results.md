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
