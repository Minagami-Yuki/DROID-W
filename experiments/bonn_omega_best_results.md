# Bonn Omega Best Results

Date: 2026-07-03

## Setting

This run applies the best `bonn_person_tracking` setting to the remaining Bonn sequences:

- DROID-W original Metric3D depth remains the primary monocular depth.
- Online VGGT-Omega depth is scale-aligned to Metric3D and blended with `blend_alpha: 0.10`.
- Online VGGT-Omega confidence is used as a soft edge uncertainty prior with `edge_weight_strength: 0.25`.
- Original DROID-W configs and default behavior are unchanged. The new behavior is enabled only by the `_omega_best` configs.

Common Omega settings:

```yaml
omega_prior:
  enable: True
  source: model
  depth:
    enable: True
    mode: blend
    blend_alpha: 0.10
    align_to_mono: scale
    align_trim: 0.05
    fallback_to_mono: True
  uncertainty:
    enable: True
    apply_to: edge_weight
    normalize_confidence: True
    edge_weight_strength: 0.25
    freeze_droid_uncertainty_update: False
  model:
    repo_path: thirdparty/vggt-omega
    checkpoint: /data1/czy/Output/DROID-W/vggt_omega_1b_512.pt
    image_resolution: 512
    preprocess_mode: balanced
```

## Commands

The following configs were added for the remaining Bonn sequences:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_balloon_omega_best.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_balloon2_omega_best.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd_omega_best.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_best.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_best.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_moving_nonobstructing_box2_omega_best.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking2_omega_best.yaml
```

Batch helper:

```bash
bash scripts_eval/run_bonn_omega_best_remaining.sh
```

The batch helper writes logs to `Outputs/Bonn/omega_best_logs/`.

## Results

ATE RMSE in meters.

| Scene | DROID-W KF | Omega Best KF | Delta KF | DROID-W Full | Omega Best Full | Delta Full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_balloon | 0.027788 | 0.027853 | +0.000065 | 0.026447 | 0.026513 | +0.000066 |
| bonn_balloon2 | 0.027593 | 0.027811 | +0.000218 | 0.024623 | 0.024804 | +0.000181 |
| bonn_crowd | 0.015506 | 0.014779 | -0.000726 | 0.013215 | 0.013787 | +0.000572 |
| bonn_crowd2 | 0.019121 | 0.018728 | -0.000393 | 0.018004 | 0.017846 | -0.000159 |
| bonn_moving_nonobstructing_box | 0.014665 | 0.015012 | +0.000348 | 0.014748 | 0.015095 | +0.000347 |
| bonn_moving_nonobstructing_box2 | 0.025137 | 0.024697 | -0.000440 | 0.023466 | 0.023015 | -0.000452 |
| bonn_person_tracking | 0.033933 | 0.033721 | -0.000212 | 0.034278 | 0.033778 | -0.000500 |
| bonn_person_tracking2 | 0.029435 | 0.029277 | -0.000158 | 0.029595 | 0.029436 | -0.000159 |

Summary:

| Metric | DROID-W Mean | Omega Best Mean | Mean Delta | Improved Sequences |
| --- | ---: | ---: | ---: | ---: |
| KF RMSE | 0.024147 | 0.023985 | -0.000162 | 5 / 8 |
| Full RMSE | 0.023047 | 0.023034 | -0.000013 | 4 / 8 |

Notes:

- The best `bonn_person_tracking` config remains `configs/Dynamic/Bonn/bonn_person_tracking_omega_blend010_u_soft025.yaml`.
- `bonn_person_tracking` full RMSE stays below the `0.034 m` target: `0.033778 m`.
- The same recipe improves `bonn_person_tracking2` full RMSE from `0.029595 m` to `0.029436 m`.
- Gains are concentrated on `crowd2`, `moving_nonobstructing_box2`, and the two person-tracking sequences. `balloon`, `balloon2`, and `moving_nonobstructing_box` are slightly worse, so this setting should be treated as a robust person/dynamic-human setting rather than a universal Bonn optimum.

Metric files:

- `Outputs/Bonn/bonn_balloon_omega_best/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_balloon_omega_best/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_balloon2_omega_best/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_balloon2_omega_best/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_crowd_omega_best/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_crowd_omega_best/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_best/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_crowd2_omega_best/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_moving_nonobstructing_box_omega_best/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_moving_nonobstructing_box_omega_best/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_moving_nonobstructing_box2_omega_best/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_moving_nonobstructing_box2_omega_best/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_person_tracking_omega_blend010_u_soft025/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_person_tracking_omega_blend010_u_soft025/traj/metrics_full_traj.txt`
- `Outputs/Bonn/bonn_person_tracking2_omega_best/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/bonn_person_tracking2_omega_best/traj/metrics_full_traj.txt`
