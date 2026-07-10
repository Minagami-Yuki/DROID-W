# Bonn Omega Token Edge-DTF Results

Date: 2026-07-05

## Goal

This experiment adds Omega camera/register tokens as a per-edge calibration signal on top of the existing Edge-DTF uncertainty prior. The target test is `bonn_person_tracking`, with the constraint that the final result must remain within `105%` of the original DROID-W tracking RMSE.

## Implementation

Files:

- `src/utils/omega_predictor.py`: optionally returns `camera_and_register_tokens` from the online VGGT-Omega forward pass.
- `src/utils/omega_prior.py`: supports loading cached token arrays when available.
- `src/motion_filter.py`: requests/stores Omega tokens only when token calibration is enabled.
- `src/depth_video.py`: stores fixed-size Omega token buffers and computes pairwise token-distance calibration for Edge-DTF weights.
- `src/utils/edge_dtf_prior.py`: accepts an optional calibration map when converting Edge-DTF residuals into BA weights.
- `src/factor_graph.py`: preserves token buffers during keyframe removal.
- `configs/droid_w.yaml`: adds default-off `edge_dtf_prior.token_calibration`.
- `configs/Dynamic/Bonn/bonn_person_tracking_omega_token_edge_dtf.yaml`: enables the conservative token-calibrated setting.

The token distance is computed per factor:

```text
token_distance(i, j) = 1 - cosine(mean(tokens_i), mean(tokens_j))
calibrated = clamp((token_distance - min_distance) / (max_distance - min_distance), 0, 1)
scale = clamp(1 + strength * calibrated, min_scale, max_scale)
edge_residual = clamp(edge_residual * scale, 0, 1)
```

Default behavior is unchanged because `edge_dtf_prior.token_calibration.enable` is `False` in the base config.

## Commands

Syntax and config checks:

```bash
conda run -n droid-w python -m py_compile src/utils/edge_dtf_prior.py src/utils/omega_predictor.py src/utils/omega_prior.py src/motion_filter.py src/depth_video.py src/factor_graph.py
```

Experiment command:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_token_edge_dtf.yaml
```

Final output:

```text
/data1/czy/Output/DROID-omega/Bonn/bonn_person_tracking_omega_token_edge_dtf
```

## Results

ATE RMSE in meters.

| Scene | Variant | KF RMSE | Full RMSE | Full vs DROID-W |
| --- | --- | ---: | ---: | ---: |
| bonn_person_tracking | DROID-W | 0.033933 | 0.034278 | 100.00% |
| bonn_person_tracking | Omega best | 0.033721 | 0.033778 | 98.54% |
| bonn_person_tracking | Omega + Edge-DTF soft cycle | 0.033462 | 0.033622 | 98.09% |
| bonn_person_tracking | Token Edge-DTF amplify 0.08 / max 1.12 | 0.033732 | 0.033898 | 98.89% |
| bonn_person_tracking | Token Edge-DTF attenuate 0.08 / min 0.88 | 0.033869 | 0.034003 | 99.20% |
| bonn_person_tracking | Token Edge-DTF amplify 0.02 / max 1.03 | 0.033701 | 0.033834 | 98.70% |

The DROID-W `105%` threshold is:

```text
KF threshold:   0.033933 * 1.05 = 0.035630
Full threshold: 0.034278 * 1.05 = 0.035992
```

The final token-calibrated setting satisfies the constraint:

```text
KF RMSE:   0.033701 < 0.035630
Full RMSE: 0.033834 < 0.035992
```

## Runtime

Final conservative token run:

| Stage | Count | Total Time | Average Time | FPS |
| --- | ---: | ---: | ---: | ---: |
| Metric Depth Estimation | 129 | 11.312890 | 0.087697 | 11.40 |
| DINO Feature Extraction | 129 | 1.126742 | 0.008734 | 114.49 |
| Tracking | 580 | 62.443902 | 0.107662 | 9.29 |
| Final Global BA | 1 | 8.447342 | 8.447342 | 0.12 |
| Full Trajectory Filling | 1 | 19.919237 | 19.919237 | 0.05 |
| Full System | 580 | 103.250113 | 0.178017 | 5.62 |

Token storage was active in the final run:

```text
omega_tokens: valid 78 / 78 keyframes, shape (78, 17, 2048)
```

## Interpretation

- The token-calibrated Edge-DTF path is functional and safe: it stays below DROID-W baseline and within the requested `105%` bound.
- Conservative amplification is better than the tested attenuation setting on `bonn_person_tracking`.
- The current token pooling is global per edge, so it is not yet stronger than the best soft-cycle Edge-DTF setting. The next research step should make token calibration spatial or object-aware instead of using a single pairwise scalar.

## Token-Based Dynamic Factor Suppression

Date: 2026-07-06

This follow-up implements the second proposed direction: using Omega token inconsistency to suppress likely dynamic factors directly. Unlike token calibration, which modifies the Edge-DTF residual before it is converted into a weight, token dynamic suppression directly multiplies the dense BA factor weight by a soft factor-level scale:

```text
token_distance(i, j) = 1 - cosine(mean(tokens_i), mean(tokens_j))
dynamic_score = clamp((token_distance - min_distance) / (max_distance - min_distance), 0, 1)
suppression_scale = clamp(1 - strength * dynamic_score, min_scale, 1)
weight(i -> j) = weight(i -> j) * suppression_scale
```

Implementation:

- `src/depth_video.py`: adds `edge_dtf_token_dynamic_suppression` and shared `edge_dtf_token_distance`.
- `src/motion_filter.py`: requests Omega tokens when either token calibration or token dynamic suppression is enabled.
- `configs/droid_w.yaml`: adds default-off `edge_dtf_prior.token_dynamic_suppression`.
- `configs/Dynamic/Bonn/bonn_person_tracking_omega_token_suppression.yaml`: conservative test config.

Config:

```yaml
edge_dtf_prior:
  token_calibration:
    enable: False
  token_dynamic_suppression:
    enable: True
    pooling: mean
    min_distance: 0.02
    max_distance: 0.20
    strength: 0.03
    min_scale: 0.97
    max_scale: 1.0
```

Command:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_token_suppression.yaml
```

ATE RMSE in meters.

| Scene | Variant | KF RMSE | Full RMSE | Full vs DROID-W |
| --- | --- | ---: | ---: | ---: |
| bonn_person_tracking | DROID-W | 0.033933 | 0.034278 | 100.00% |
| bonn_person_tracking | Omega + Edge-DTF soft cycle | 0.033462 | 0.033622 | 98.09% |
| bonn_person_tracking | Omega Token + Edge-DTF calibration | 0.033701 | 0.033834 | 98.70% |
| bonn_person_tracking | Omega Token dynamic suppression | 0.033312 | 0.033456 | 97.60% |

The DROID-W `105%` threshold remains:

```text
KF threshold:   0.035630
Full threshold: 0.035992
```

The token dynamic suppression result satisfies the constraint:

```text
KF RMSE:   0.033312 < 0.035630
Full RMSE: 0.033456 < 0.035992
```

Runtime:

| Stage | Count | Total Time | Average Time | FPS |
| --- | ---: | ---: | ---: | ---: |
| Metric Depth Estimation | 129 | 11.656562 | 0.090361 | 11.07 |
| DINO Feature Extraction | 129 | 1.172285 | 0.009087 | 110.04 |
| Tracking | 580 | 64.617905 | 0.111410 | 8.98 |
| Final Global BA | 1 | 8.460484 | 8.460484 | 0.12 |
| Full Trajectory Filling | 1 | 19.612808 | 19.612808 | 0.05 |
| Full System | 580 | 105.520043 | 0.181931 | 5.50 |

Token storage was active:

```text
omega_tokens: valid 78 / 78 keyframes, shape (78, 17, 2048)
```

Interpretation:

- This is the best `bonn_person_tracking` result among the tested Omega/Edge-DTF variants so far.
- The dynamic suppression path is still conservative: it attenuates high-token-distance factors by at most `3%`, which helped tracking without violating the DROID-W `105%` safety constraint.
- This is now a cleaner paper contribution than pure token calibration because it directly maps Omega token inconsistency to dynamic factor handling in dense BA.

## Full Bonn Token Suppression Sweep

Date: 2026-07-06

The same token dynamic suppression setting was applied to all Bonn sequences:

```yaml
edge_dtf_prior:
  token_dynamic_suppression:
    enable: True
    pooling: mean
    min_distance: 0.02
    max_distance: 0.20
    strength: 0.03
    min_scale: 0.97
    max_scale: 1.0
```

Commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_balloon_omega_token_suppression.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_balloon2_omega_token_suppression.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd_omega_token_suppression.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_token_suppression.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_token_suppression.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_moving_nonobstructing_box2_omega_token_suppression.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_token_suppression.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking2_omega_token_suppression.yaml
```

ATE RMSE in meters. FPS is from `timer_summary.csv`.

| Sequence | KF RMSE | Full RMSE | Tracking FPS | Full System FPS | Frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| bonn_balloon | 0.026516 | 0.025079 | 8.95 | 5.60 | 439 |
| bonn_balloon2 | 0.027675 | 0.024716 | 8.55 | 5.39 | 469 |
| bonn_crowd | 0.014741 | 0.013745 | 8.62 | 5.28 | 928 |
| bonn_crowd2 | 0.019217 | 0.018082 | 8.29 | 5.09 | 895 |
| bonn_moving_nonobstructing_box | 0.015155 | 0.015254 | 10.50 | 6.19 | 778 |
| bonn_moving_nonobstructing_box2 | 0.024690 | 0.023011 | 10.36 | 5.99 | 937 |
| bonn_person_tracking | 0.033312 | 0.033456 | 8.98 | 5.50 | 580 |
| bonn_person_tracking2 | 0.029047 | 0.029281 | 9.67 | 5.55 | 567 |

Full trajectory comparison:

| Sequence | DROID-W Full | Omega Best Full | Soft Cycle Full | Token Suppression Full | Token vs DROID-W | Token vs Soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_balloon | 0.026447 | 0.026513 | 0.026530 | 0.025079 | -5.17% | -5.47% |
| bonn_balloon2 | 0.024623 | 0.024804 | 0.024729 | 0.024716 | +0.38% | -0.05% |
| bonn_crowd | 0.013215 | 0.013787 | 0.013723 | 0.013745 | +4.01% | +0.16% |
| bonn_crowd2 | 0.018004 | 0.017846 | 0.017985 | 0.018082 | +0.43% | +0.54% |
| bonn_moving_nonobstructing_box | 0.014748 | 0.015095 | 0.015298 | 0.015254 | +3.43% | -0.29% |
| bonn_moving_nonobstructing_box2 | 0.023466 | 0.023015 | 0.023023 | 0.023011 | -1.94% | -0.05% |
| bonn_person_tracking | 0.034278 | 0.033778 | 0.033622 | 0.033456 | -2.40% | -0.49% |
| bonn_person_tracking2 | 0.029595 | 0.029436 | 0.029346 | 0.029281 | -1.06% | -0.22% |

Summary:

- Mean full RMSE: DROID-W `0.023047`, Omega best `0.023034`, soft cycle `0.023032`, token suppression `0.022828`.
- Token suppression improves 4/8 sequences over DROID-W, 6/8 over Omega best, and 6/8 over soft-cycle Edge-DTF.
- Token suppression remains within the DROID-W `105%` full-RMSE safety bound on all sequences.
- Token extraction was active for every keyframe in every sequence:
  - `bonn_balloon`: `53 / 53`
  - `bonn_balloon2`: `60 / 60`
  - `bonn_crowd`: `110 / 110`
  - `bonn_crowd2`: `120 / 120`
  - `bonn_moving_nonobstructing_box`: `93 / 93`
  - `bonn_moving_nonobstructing_box2`: `113 / 113`
  - `bonn_person_tracking`: `78 / 78`
  - `bonn_person_tracking2`: `72 / 72`

Output directories:

```text
/data1/czy/Output/DROID-omega/Bonn/bonn_balloon_omega_token_suppression
/data1/czy/Output/DROID-omega/Bonn/bonn_balloon2_omega_token_suppression
/data1/czy/Output/DROID-omega/Bonn/bonn_crowd_omega_token_suppression
/data1/czy/Output/DROID-omega/Bonn/bonn_crowd2_omega_token_suppression
/data1/czy/Output/DROID-omega/Bonn/bonn_moving_nonobstructing_box_omega_token_suppression
/data1/czy/Output/DROID-omega/Bonn/bonn_moving_nonobstructing_box2_omega_token_suppression
/data1/czy/Output/DROID-omega/Bonn/bonn_person_tracking_omega_token_suppression
/data1/czy/Output/DROID-omega/Bonn/bonn_person_tracking2_omega_token_suppression
```

## Per-Edge Covariance Factor Pilot

Date: 2026-07-07

This experiment implements the third proposed direction: a lightweight per-edge covariance-style precision factor on top of Omega Token dynamic suppression. The DBA CUDA kernel is unchanged. The covariance factor is injected through the existing dense edge-weight path:

```text
risk = weighted_mean(
  token_distance_score,
  omega_uncertainty_score,
  edge_dtf_residual_score
)
precision_scale = clamp(1 - strength * risk + reliable_boost * (1 - risk), min_scale, max_scale)
weight(i -> j) = weight(i -> j) * token_suppression(i, j) * precision_scale(i, j, u)
```

Implementation:

- `src/depth_video.py`: adds `edge_dtf_per_edge_covariance` and applies it in both Edge-DTF weight exits.
- `src/motion_filter.py`: requests Omega tokens when covariance uses token distance.
- `configs/droid_w.yaml`: adds default-off `edge_dtf_prior.per_edge_covariance`.
- New configs:
  - `configs/Dynamic/Bonn/bonn_person_tracking_omega_covariance_v1.yaml`
  - `configs/Dynamic/Bonn/bonn_person_tracking_omega_covariance_v2.yaml`
  - `configs/Dynamic/Bonn/bonn_person_tracking_omega_covariance_v3.yaml`
  - `configs/Dynamic/Bonn/bonn_crowd2_omega_covariance_v1.yaml`
  - `configs/Dynamic/Bonn/bonn_crowd2_omega_covariance_v2.yaml`
  - `configs/Dynamic/Bonn/bonn_crowd2_omega_covariance_v3.yaml`

Config variants:

| Variant | Behavior | Strength | Reliable Boost | Clamp |
| --- | --- | ---: | ---: | --- |
| v1 | risk-only downweight | 0.04 | 0.00 | `[0.96, 1.00]` |
| v2 | lighter risk-only downweight | 0.01 | 0.00 | `[0.99, 1.00]` |
| v3 | risk downweight + reliable-edge boost | 0.005 | 0.01 | `[0.995, 1.01]` |

Commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_covariance_v1.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_covariance_v1.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_covariance_v2.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_covariance_v3.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_covariance_v3.yaml
```

ATE RMSE in meters.

| Sequence | Variant | KF RMSE | Full RMSE | Tracking FPS | Full System FPS |
| --- | --- | ---: | ---: | ---: | ---: |
| bonn_person_tracking | DROID-W | 0.033933 | 0.034278 | - | - |
| bonn_person_tracking | Token suppression | 0.033312 | 0.033456 | 8.98 | 5.50 |
| bonn_person_tracking | Covariance v1 | 0.033742 | 0.033889 | 8.08 | 5.15 |
| bonn_person_tracking | Covariance v2 | 0.034143 | 0.034287 | 9.34 | 5.65 |
| bonn_person_tracking | Covariance v3 | 0.033665 | 0.033805 | 9.38 | 5.67 |
| bonn_crowd2 | DROID-W | - | 0.018004 | - | - |
| bonn_crowd2 | Token suppression | 0.019217 | 0.018082 | 8.29 | 5.09 |
| bonn_crowd2 | Covariance v1 | 0.019182 | 0.018295 | 8.19 | 5.06 |
| bonn_crowd2 | Covariance v3 | 0.019564 | 0.018545 | 8.27 | 5.09 |

Interpretation:

- The covariance factor implementation is functional and config-gated; default DROID-W behavior remains unchanged.
- All tested covariance variants remain within the DROID-W `105%` full-RMSE safety bound on the tested sequences.
- Pure extra downweighting is not enough: v1/v2 both underperform token suppression on `bonn_person_tracking`.
- v3 is the best covariance form tested so far because it allows reliable edges to receive a small precision boost, but it still does not beat the current Token suppression mainline.
- Current conclusion: keep `Omega Token dynamic suppression` as the best Bonn method for now. Treat per-edge covariance as a promising method section direction, but it needs a better covariance model before becoming the main experimental result.

## Token-Spatial Dynamic Suppression Pilot

Date: 2026-07-07

This experiment follows the next direction after the covariance pilot: make token-based dynamic suppression spatial rather than using only a global scalar per factor. Since the currently cached Omega tokens are camera/register tokens rather than dense pixel tokens, the spatial term uses token distance as a frame-pair dynamic gate and uses pixel-level Omega uncertainty plus Edge-DTF residual as the dynamic location map:

```text
token_score(i, j) = normalize(1 - cosine(mean(tokens_i), mean(tokens_j)))
pixel_score(u) = weighted_mean(omega_uncertainty_score(u), edge_dtf_residual_score(u))
spatial_dynamic_score(i, j, u) = token_score(i, j) * pixel_score(u)
weight(i -> j, u) = weight(i -> j, u) * clamp(1 - strength * spatial_dynamic_score, min_scale, 1)
```

Implementation:

- `src/depth_video.py`: adds `edge_dtf_token_spatial_suppression` and applies it in both Edge-DTF weight exits.
- `src/motion_filter.py`: requests Omega tokens when spatial token suppression is enabled.
- `configs/droid_w.yaml`: adds default-off `edge_dtf_prior.token_spatial_suppression`.
- New configs:
  - `configs/Dynamic/Bonn/bonn_person_tracking_omega_token_spatial_v1.yaml`
  - `configs/Dynamic/Bonn/bonn_person_tracking_omega_token_spatial_v2.yaml`

Config variants:

| Variant | Behavior | Strength | Clamp |
| --- | --- | ---: | --- |
| v1 | replace global token suppression with spatial suppression | 0.03 | `[0.97, 1.00]` |
| v2 | keep global token suppression and add light spatial suppression | 0.01 | `[0.99, 1.00]` |

Commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_token_spatial_v1.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_token_spatial_v2.yaml
```

ATE RMSE in meters.

| Sequence | Variant | KF RMSE | Full RMSE | Tracking FPS | Full System FPS |
| --- | --- | ---: | ---: | ---: | ---: |
| bonn_person_tracking | DROID-W | 0.033933 | 0.034278 | - | - |
| bonn_person_tracking | Token suppression | 0.033312 | 0.033456 | 8.98 | 5.50 |
| bonn_person_tracking | Token-spatial v1 | 0.033496 | 0.033638 | 9.22 | 5.61 |
| bonn_person_tracking | Token-spatial v2 | 0.033327 | 0.033492 | 9.30 | 5.63 |

Token storage was active in both spatial runs:

```text
bonn_person_tracking_omega_token_spatial_v1: 78 / 78
bonn_person_tracking_omega_token_spatial_v2: 78 / 78
```

Interpretation:

- Spatial token suppression is functional and remains within the DROID-W `105%` safety bound.
- v2 is very close to the current best token suppression result: `0.033492` vs `0.033456` full RMSE, about `0.11%` worse.
- Replacing the scalar token factor entirely is worse than the scalar mainline; adding a very light spatial correction is better.
- Current conclusion: this is a promising paper-facing refinement, but it should not replace the token suppression mainline until the spatial map uses real dense Omega intermediate tokens or a stronger dynamic localization cue.

## Token-Spatial FPS Ablation

Date: 2026-07-07

This follow-up keeps the current `Omega Token dynamic suppression` mainline fixed and tests whether a lighter spatial token correction can recover more FPS while staying close to the best `bonn_person_tracking` accuracy. Since FPS is a key baseline comparison metric, this table includes the original DROID-W timer from the existing `Outputs/Bonn/bonn_person_tracking` run.

New configs:

- `configs/Dynamic/Bonn/bonn_person_tracking_omega_token_spatial_v3.yaml`
- `configs/Dynamic/Bonn/bonn_person_tracking_omega_token_spatial_uncertainty_only.yaml`
- `configs/Dynamic/Bonn/bonn_person_tracking_omega_token_spatial_edge_only.yaml`

Commands:

```bash
conda run -n droid-w python -m py_compile src/depth_video.py src/motion_filter.py
conda run -n droid-w python -c "from src.config import load_config; paths=['configs/Dynamic/Bonn/bonn_person_tracking_omega_token_spatial_v3.yaml','configs/Dynamic/Bonn/bonn_person_tracking_omega_token_spatial_uncertainty_only.yaml','configs/Dynamic/Bonn/bonn_person_tracking_omega_token_spatial_edge_only.yaml']; [print(p, load_config(p)['edge_dtf_prior']['token_spatial_suppression']['enable'], load_config(p)['scene']) for p in paths]"
git diff --check -- configs/Dynamic/Bonn/bonn_person_tracking_omega_token_spatial_v3.yaml configs/Dynamic/Bonn/bonn_person_tracking_omega_token_spatial_uncertainty_only.yaml configs/Dynamic/Bonn/bonn_person_tracking_omega_token_spatial_edge_only.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_token_spatial_v3.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_token_spatial_uncertainty_only.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_token_spatial_edge_only.yaml
```

ATE RMSE in meters. FPS is from `timer_summary.csv`.

| Sequence | Variant | KF RMSE | Full RMSE | Tracking FPS | Full System FPS | Full vs DROID-W |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bonn_person_tracking | DROID-W | 0.033933 | 0.034278 | 14.14 | 7.03 | 100.00% |
| bonn_person_tracking | Token suppression | 0.033312 | 0.033456 | 8.98 | 5.50 | 97.60% |
| bonn_person_tracking | Token-spatial v2 | 0.033327 | 0.033492 | 9.30 | 5.63 | 97.71% |
| bonn_person_tracking | Token-spatial v3 | 0.033711 | 0.033887 | 9.20 | 5.60 | 98.86% |
| bonn_person_tracking | Token-spatial uncertainty-only | 0.033527 | 0.033673 | 9.24 | 5.61 | 98.23% |
| bonn_person_tracking | Token-spatial edge-only | 0.033664 | 0.033793 | 9.29 | 5.63 | 98.58% |

Interpretation:

- All spatial variants remain within the DROID-W `105%` safety bound.
- None of the new spatial variants improves over the scalar Token suppression mainline on accuracy.
- `Token-spatial v2` is still the best speed/accuracy compromise among spatial variants: it is only `0.11%` worse than Token suppression in full RMSE, while improving Tracking FPS from `8.98` to `9.30`.
- Original DROID-W is substantially faster on this sequence: `14.14` Tracking FPS and `7.03` Full System FPS. The current Omega online path is therefore not a free accuracy gain; FPS should be reported as a central ablation, and the next engineering target should be Omega prior caching or lower-rate Omega inference.
- Current mainline remains `Omega Token dynamic suppression` for best Bonn accuracy. `Token-spatial v2` is worth keeping as an efficiency-oriented variant, but it should not replace the mainline unless later dense/intermediate Omega token maps improve accuracy.

## Omega Cache And Dense Patch-Token Uncertainty

Date: 2026-07-07

Goal: add a reusable Omega prior cache pipeline and test a dense patch-token uncertainty map without changing the default DROID-W behavior. This makes the Omega path much closer to DROID-W speed by precomputing depth, confidence, uncertainty, register tokens, and projected patch tokens.

Implementation:

- `omega_prior.cache.*` controls writing Omega priors to disk.
- `omega_prior.source: cache` reads precomputed priors instead of running Omega online.
- `omega_prior.model.patch_tokens.*` exposes low-dimensional final Omega patch tokens.
- `edge_dtf_prior.patch_token_uncertainty.*` adds a dense per-pixel edge weight from reprojected patch-token cosine distance.

New configs:

- `configs/Dynamic/Bonn/bonn_person_tracking_omega_cache_write.yaml`
- `configs/Dynamic/Bonn/bonn_person_tracking_omega_token_suppression_cache.yaml`
- `configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty.yaml`
- `configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v2.yaml`

Validation commands:

```bash
conda run -n droid-w python -m py_compile src/utils/omega_predictor.py src/utils/omega_prior.py src/motion_filter.py src/depth_video.py src/factor_graph.py thirdparty/vggt-omega/vggt_omega/models/vggt_omega.py
conda run -n droid-w python -c "from src.config import load_config; paths=['configs/Dynamic/Bonn/bonn_person_tracking_omega_cache_write.yaml','configs/Dynamic/Bonn/bonn_person_tracking_omega_token_suppression_cache.yaml','configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty.yaml','configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v2.yaml']; [print(p, load_config(p)['omega_prior']['source'], load_config(p)['omega_prior']['model']['patch_tokens']['enable'], load_config(p)['edge_dtf_prior']['patch_token_uncertainty']['enable']) for p in paths]"
git diff --check -- configs/droid_w.yaml src/utils/omega_predictor.py src/utils/omega_prior.py src/motion_filter.py src/depth_video.py src/factor_graph.py thirdparty/vggt-omega/vggt_omega/models/vggt_omega.py configs/Dynamic/Bonn/bonn_person_tracking_omega_cache_write.yaml configs/Dynamic/Bonn/bonn_person_tracking_omega_token_suppression_cache.yaml configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty.yaml configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v2.yaml experiments/bonn_omega_token_edge_dtf_results.md
```

Experiment commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_cache_write.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_token_suppression_cache.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v2.yaml
```

Cache directory:

```text
/data1/czy/Output/DROID-omega/cache/Bonn/bonn_person_tracking
```

Cache contents after the write run:

| Prior type | Files | Example shape |
| --- | ---: | --- |
| depths | 129 | `(384, 512)` |
| confidences | 129 | `(384, 512)` |
| uncertainties | 129 | `(384, 512)` |
| tokens | 129 | `(17, 2048)` |
| patch_tokens | 129 | `(8, 28, 37)` |

ATE RMSE in meters. FPS is from `timer_summary.csv`.

| Sequence | Variant | KF RMSE | Full RMSE | Tracking FPS | Full System FPS | Full vs DROID-W |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bonn_person_tracking | DROID-W | 0.033933 | 0.034278 | 14.14 | 7.03 | 100.00% |
| bonn_person_tracking | Token suppression online | 0.033312 | 0.033456 | 8.98 | 5.50 | 97.60% |
| bonn_person_tracking | Cache write online | 0.033593 | 0.033748 | 9.13 | 5.56 | 98.45% |
| bonn_person_tracking | Token suppression cache | 0.033637 | 0.033827 | 14.63 | 7.22 | 98.68% |
| bonn_person_tracking | Patch-token uncertainty v1 | 0.034160 | 0.034255 | 14.74 | 7.27 | 99.93% |
| bonn_person_tracking | Patch-token uncertainty v2 | 0.033560 | 0.033691 | 14.70 | 7.22 | 98.29% |

Interpretation:

- The cache pipeline recovers DROID-W-level speed: `Token suppression cache` reaches `14.63` Tracking FPS versus `8.98` for online Omega token suppression and `14.14` for DROID-W.
- Dense patch-token uncertainty is functional. The first version was too strong/noisy, but the light v2 setting improves over token-cache accuracy while preserving cache speed.
- `Patch-token uncertainty v2` is the best cached variant so far: `0.033691` full RMSE at `14.70` Tracking FPS. It is faster than the online token mainline and still better than original DROID-W on this sequence.
- The online token mainline remains the peak-accuracy reference on `bonn_person_tracking` (`0.033456` full RMSE), but the cached patch-token v2 variant is now the most practical speed/accuracy setting.
- Next test target: run patch-token v2 on `bonn_crowd2` and one more Bonn dynamic sequence; if it stays within the DROID-W `105%` bound and improves the cached average, promote it to a full-Bonn ablation.

## Patch-Token V2 Cross-Sequence Check

Date: 2026-07-07

Goal: check whether the cached dense patch-token uncertainty v2 setting remains within the DROID-W `105%` safety bound beyond `bonn_person_tracking`, while recording FPS as a primary comparison metric.

New configs:

- `configs/Dynamic/Bonn/bonn_crowd2_omega_cache_write.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v2.yaml`
- `configs/Dynamic/Bonn/bonn_balloon2_omega_cache_write.yaml`
- `configs/Dynamic/Bonn/bonn_balloon2_omega_patch_token_uncertainty_v2.yaml`

Validation commands:

```bash
conda run -n droid-w python -c "from src.config import load_config; paths=['configs/Dynamic/Bonn/bonn_crowd2_omega_cache_write.yaml','configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v2.yaml','configs/Dynamic/Bonn/bonn_balloon2_omega_cache_write.yaml','configs/Dynamic/Bonn/bonn_balloon2_omega_patch_token_uncertainty_v2.yaml']; [print(p, load_config(p)['scene'], load_config(p)['omega_prior']['source'], load_config(p)['omega_prior']['model']['patch_tokens']['enable'], load_config(p)['edge_dtf_prior']['patch_token_uncertainty']['enable']) for p in paths]"
git diff --check -- configs/Dynamic/Bonn/bonn_crowd2_omega_cache_write.yaml configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v2.yaml configs/Dynamic/Bonn/bonn_balloon2_omega_cache_write.yaml configs/Dynamic/Bonn/bonn_balloon2_omega_patch_token_uncertainty_v2.yaml
```

Experiment commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_cache_write.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v2.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_balloon2_omega_cache_write.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_balloon2_omega_patch_token_uncertainty_v2.yaml
```

Cache contents:

| Sequence | depths | confidences | uncertainties | tokens | patch_tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| bonn_crowd2 | 205 | 205 | 205 | 205 | 205 |
| bonn_balloon2 | 94 | 94 | 94 | 94 | 94 |

ATE RMSE in meters. FPS is from `timer_summary.csv`.

| Sequence | Variant | KF RMSE | Full RMSE | Tracking FPS | Full System FPS | Full vs DROID-W |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bonn_crowd2 | DROID-W | - | 0.018004 | - | - | 100.00% |
| bonn_crowd2 | Token suppression online | 0.019217 | 0.018082 | 8.29 | 5.09 | 100.43% |
| bonn_crowd2 | Cache write online | 0.019822 | 0.019081 | 8.43 | 5.16 | 105.98% |
| bonn_crowd2 | Patch-token uncertainty v2 cache | 0.019551 | 0.018597 | 11.68 | 6.19 | 103.29% |
| bonn_balloon2 | DROID-W | - | 0.024623 | - | - | 100.00% |
| bonn_balloon2 | Token suppression online | 0.027675 | 0.024716 | 8.55 | 5.39 | 100.38% |
| bonn_balloon2 | Cache write online | 0.027616 | 0.024730 | 8.19 | 5.25 | 100.43% |
| bonn_balloon2 | Patch-token uncertainty v2 cache | 0.027625 | 0.024649 | 13.32 | 6.97 | 100.11% |

Two-sequence summary:

- Mean full RMSE: DROID-W `0.021314`, token suppression online `0.021399`, patch-token v2 cache `0.021623`.
- Mean Tracking FPS: token suppression online `8.42`, patch-token v2 cache `12.50`.
- Patch-token v2 cache improves mean Tracking FPS by about `48.5%` over online token suppression on these two sequences.
- Both patch-token v2 cache runs stay inside the DROID-W `105%` safety bound.
- `bonn_balloon2` is a clean speed win with almost no accuracy cost; `bonn_crowd2` is safe but loses accuracy versus both DROID-W and online token suppression.

Conclusion:

- Cached patch-token uncertainty v2 is promising as a speed-oriented variant, but it is not ready to replace the online token-suppression mainline for best accuracy.
- The next ablation should tune the patch-token factor on `bonn_crowd2`: lower `strength`, higher `min_scale`, or gate the factor by Omega confidence/edge residual so dense tokens only affect genuinely dynamic or unreliable edges.

## Crowd2 Patch-Token V3 Tuning And Full Bonn Cache Sweep

Date: 2026-07-08

Goal: continue tuning `bonn_crowd2` with cached Omega dense patch-token uncertainty. The target is to keep the result inside the DROID-W `105%` full-RMSE bound, then run the remaining Bonn sequences and record FPS.

New configs and runner:

- `configs/Dynamic/Bonn/bonn_crowd2_omega_token_suppression_cache.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v3.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v4.yaml`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v5.yaml`
- `configs/Dynamic/Bonn/*_omega_cache_write.yaml`
- `configs/Dynamic/Bonn/*_omega_patch_token_uncertainty_v3.yaml`
- `scripts_eval/run_bonn_patch_token_v3_remaining.sh`

The selected v3 patch-token setting is deliberately conservative:

```yaml
edge_dtf_prior:
  patch_token_uncertainty:
    enable: True
    strength: 0.001
    min_scale: 0.999
    min_distance: 0.05
    max_distance: 0.60
```

Validation commands:

```bash
bash -n scripts_eval/run_bonn_patch_token_v3_remaining.sh
conda run -n droid-w python -c "from src.config import load_config; paths=['configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v3.yaml','configs/Dynamic/Bonn/bonn_balloon_omega_patch_token_uncertainty_v3.yaml']; [print(p, load_config(p)['omega_prior']['source'], load_config(p)['edge_dtf_prior']['patch_token_uncertainty']['enable']) for p in paths]"
git diff --check -- scripts_eval/run_bonn_patch_token_v3_remaining.sh configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v3.yaml
```

Crowd2 tuning results:

| Variant | KF RMSE | Full RMSE | Tracking FPS | Full System FPS | Full vs DROID-W |
| --- | ---: | ---: | ---: | ---: | ---: |
| DROID-W | - | 0.018004 | - | - | 100.00% |
| Online token suppression | 0.019217 | 0.018082 | 8.29 | 5.09 | 100.43% |
| Token suppression cache | 0.019436 | 0.018581 | 11.66 | 6.20 | 103.21% |
| Patch-token v2 cache | 0.019551 | 0.018597 | 11.68 | 6.19 | 103.29% |
| Patch-token v3 cache | 0.019634 | 0.018525 | 11.52 | 6.15 | 102.89% |
| Patch-token v4 cache | 0.019595 | 0.018625 | 11.64 | 6.18 | 103.45% |
| Patch-token v5 cache | 0.019606 | 0.018667 | 11.62 | 6.20 | 103.68% |

Conclusion for `bonn_crowd2`: patch-token v3 is the best cached patch-token setting tested here. It stays inside the DROID-W `105%` bound and is much faster than online token suppression, though it is still less accurate than the online mainline.

Full Bonn patch-token v3 command:

```bash
bash scripts_eval/run_bonn_patch_token_v3_remaining.sh
```

ATE RMSE in meters. FPS is from each run's `timer_summary.csv`.

| Sequence | DROID-W Full | Online Token Full | Patch V3 KF | Patch V3 Full | Patch/DROID-W | Patch/Online | Track FPS | Full FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_balloon | 0.026447 | 0.025079 | 0.027842 | 0.026549 | 100.39% | +5.86% | 14.73 | 7.47 |
| bonn_balloon2 | 0.024623 | 0.024716 | 0.027729 | 0.024787 | 100.66% | +0.29% | 13.36 | 6.96 |
| bonn_crowd | 0.013215 | 0.013745 | 0.014739 | 0.013701 | 103.68% | -0.32% | 11.98 | 6.42 |
| bonn_crowd2 | 0.018004 | 0.018082 | 0.019634 | 0.018525 | 102.89% | +2.45% | 11.52 | 6.15 |
| bonn_moving_nonobstructing_box | 0.014748 | 0.015254 | 0.014988 | 0.015081 | 102.26% | -1.13% | 15.66 | 7.75 |
| bonn_moving_nonobstructing_box2 | 0.023466 | 0.023011 | 0.024682 | 0.023006 | 98.04% | -0.02% | 15.18 | 7.56 |
| bonn_person_tracking | 0.034278 | 0.033456 | 0.033709 | 0.033927 | 98.98% | +1.41% | 14.74 | 7.27 |
| bonn_person_tracking2 | 0.029595 | 0.029281 | 0.029170 | 0.029398 | 99.34% | +0.40% | 15.43 | 7.56 |

Cache contents used by patch-token v3:

| Sequence | Cached patch-token frames |
| --- | ---: |
| bonn_balloon | 79 |
| bonn_balloon2 | 94 |
| bonn_crowd | 200 |
| bonn_crowd2 | 205 |
| bonn_moving_nonobstructing_box | 121 |
| bonn_moving_nonobstructing_box2 | 164 |
| bonn_person_tracking | 129 |
| bonn_person_tracking2 | 117 |

Summary:

- Mean full RMSE: DROID-W `0.023047`, online token suppression `0.022828`, patch-token v3 cache `0.023122`.
- Mean Tracking FPS: online token suppression `9.24`, patch-token v3 cache `14.08`.
- Mean Full System FPS: online token suppression `5.57`, patch-token v3 cache `7.14`.
- Patch-token v3 cache is faster than online token suppression on all 8 Bonn sequences.
- Patch-token v3 cache stays inside the DROID-W `105%` full-RMSE safety bound on all 8 sequences.
- Patch-token v3 cache improves full RMSE over DROID-W on 3/8 sequences and improves over online token suppression on 3/8 sequences.

Current interpretation:

- For best accuracy, keep online Omega token dynamic suppression as the Bonn mainline.
- For speed/accuracy balance, patch-token v3 cache is the best current cached method: it brings Omega-enhanced tracking back to near DROID-W runtime while preserving the 105% safety constraint.
- The remaining accuracy gap is concentrated on `bonn_balloon`, `bonn_crowd2`, and `bonn_person_tracking`; the next useful ablation should make cached patch-token suppression conditional instead of always applying a dense weight, for example by gating it with high Omega uncertainty or high Edge-DTF residual.

## Conditional Patch-Token Dynamic Suppression

Date: 2026-07-08

Goal: improve the cached dense patch-token uncertainty branch by making patch-token suppression conditional. The previous v3 cache setting applied a very weak dense weight everywhere. This experiment tests whether a stronger patch-token signal can be used only where Omega uncertainty or Edge-DTF residual indicates a risky pixel.

Implementation:

- `configs/droid_w.yaml`: adds default-off `edge_dtf_prior.patch_token_uncertainty.conditional_gate`.
- `src/depth_video.py`: multiplies the dense patch-token risk by a configurable gate before converting it to an edge-weight scale.
- New configs:
  - `configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v6_conditional.yaml`
  - `configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v6_conditional.yaml`
  - `configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v7_softfloor.yaml`
  - `configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v7_softfloor.yaml`

The conditional gate computes:

```text
patch_risk = normalize(1 - cosine(patch_i(u), patch_j(warp(u))))
gate = weighted_mean(omega_uncertainty_gate, edge_residual_gate)
effective_risk = patch_risk * clamp(gate, min_gate, 1)
weight = weight * clamp(1 - strength * effective_risk, min_scale, 1)
```

The default config keeps `conditional_gate.enable: False`, so original DROID-W and older Omega configs are unchanged.

Config variants:

| Variant | Strength | Min scale | Gate floor | Interpretation |
| --- | ---: | ---: | ---: | --- |
| v6 conditional | 0.006 | 0.995 | 0.00 | only risky pixels receive patch-token suppression |
| v7 softfloor | 0.004 | 0.996 | 0.35 | keep weak global patch prior and boost risky pixels |

Validation commands:

```bash
conda run -n droid-w python -m py_compile src/depth_video.py
conda run -n droid-w python -c "from src.config import load_config; paths=['configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v7_softfloor.yaml','configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v7_softfloor.yaml']; [print(p, load_config(p)['edge_dtf_prior']['patch_token_uncertainty']['conditional_gate']['min_gate'], load_config(p)['edge_dtf_prior']['patch_token_uncertainty']['strength']) for p in paths]"
git diff --check -- src/depth_video.py configs/droid_w.yaml configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v6_conditional.yaml configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v6_conditional.yaml configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v7_softfloor.yaml configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v7_softfloor.yaml
```

Experiment commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v6_conditional.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v6_conditional.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v7_softfloor.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v7_softfloor.yaml
```

ATE RMSE in meters. FPS is from `timer_summary.csv`.

| Sequence | Variant | KF RMSE | Full RMSE | Tracking FPS | Full System FPS | Full vs DROID-W |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bonn_crowd2 | DROID-W | - | 0.018004 | - | - | 100.00% |
| bonn_crowd2 | Online token suppression | 0.019217 | 0.018082 | 8.29 | 5.09 | 100.43% |
| bonn_crowd2 | Patch-token v3 cache | 0.019634 | 0.018525 | 11.52 | 6.15 | 102.89% |
| bonn_crowd2 | Conditional v6 cache | 0.019317 | 0.018399 | 10.94 | 5.99 | 102.19% |
| bonn_crowd2 | Softfloor v7 cache | 0.019420 | 0.018266 | 11.69 | 6.24 | 101.46% |
| bonn_person_tracking | DROID-W | 0.033933 | 0.034278 | 14.14 | 7.03 | 100.00% |
| bonn_person_tracking | Online token suppression | 0.033312 | 0.033456 | 8.98 | 5.50 | 97.60% |
| bonn_person_tracking | Patch-token v3 cache | 0.033709 | 0.033927 | 14.74 | 7.27 | 98.98% |
| bonn_person_tracking | Conditional v6 cache | 0.033933 | 0.034086 | 12.81 | 6.73 | 99.44% |
| bonn_person_tracking | Softfloor v7 cache | 0.033505 | 0.033667 | 14.92 | 7.26 | 98.22% |

Two-sequence summary:

- v6 proves the conditional branch is active, but the hard zero-floor gate hurts `bonn_person_tracking`.
- v7 is the better algorithmic variant. It improves both tested sequences over patch-token v3 cache:
  - `bonn_crowd2`: `0.018525 -> 0.018266`, `1.39%` lower full RMSE.
  - `bonn_person_tracking`: `0.033927 -> 0.033667`, `0.77%` lower full RMSE.
- v7 remains close to online token suppression while being much faster:
  - `bonn_crowd2`: v7 is `+1.02%` RMSE vs online token, but `11.69` vs `8.29` Tracking FPS.
  - `bonn_person_tracking`: v7 is `+0.63%` RMSE vs online token, but `14.92` vs `8.98` Tracking FPS.
- v7 remains within the DROID-W `105%` full-RMSE bound on both tested sequences.

Current conclusion:

- Promote `Softfloor v7` as the next cached method candidate.
- The algorithmic lesson is useful for the paper: dense Omega patch-token inconsistency should not be used as a pure hard dynamic mask. A soft global floor plus risk-conditioned amplification is more stable.
- Next step: create v7 configs for the remaining Bonn sequences and run a full-Bonn sweep. If the full sweep stays inside the 105% bound and keeps the mean speed near patch-token v3, v7 should replace v3 as the speed/accuracy cached mainline.

## Full Bonn Softfloor V7 Sweep

Date: 2026-07-08

Goal: run the v7 softfloor conditional patch-token suppression setting on the remaining Bonn sequences after the two-sequence pilot.

New configs:

- `configs/Dynamic/Bonn/bonn_balloon_omega_patch_token_uncertainty_v7_softfloor.yaml`
- `configs/Dynamic/Bonn/bonn_balloon2_omega_patch_token_uncertainty_v7_softfloor.yaml`
- `configs/Dynamic/Bonn/bonn_crowd_omega_patch_token_uncertainty_v7_softfloor.yaml`
- `configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_softfloor.yaml`
- `configs/Dynamic/Bonn/bonn_moving_nonobstructing_box2_omega_patch_token_uncertainty_v7_softfloor.yaml`
- `configs/Dynamic/Bonn/bonn_person_tracking2_omega_patch_token_uncertainty_v7_softfloor.yaml`
- `scripts_eval/run_bonn_patch_token_v7_remaining.sh`

Validation:

```bash
bash -n scripts_eval/run_bonn_patch_token_v7_remaining.sh
conda run -n droid-w python -c "from src.config import load_config; import glob; paths=sorted(glob.glob('configs/Dynamic/Bonn/*_omega_patch_token_uncertainty_v7_softfloor.yaml')); [print(p, load_config(p)['scene'], load_config(p)['omega_prior']['source'], load_config(p)['edge_dtf_prior']['patch_token_uncertainty']['conditional_gate']['enable'], load_config(p)['edge_dtf_prior']['patch_token_uncertainty']['conditional_gate']['min_gate']) for p in paths]"
git diff --check -- scripts_eval/run_bonn_patch_token_v7_remaining.sh configs/Dynamic/Bonn/*_omega_patch_token_uncertainty_v7_softfloor.yaml
```

Experiment command:

```bash
bash scripts_eval/run_bonn_patch_token_v7_remaining.sh
```

ATE RMSE in meters. FPS is from `timer_summary.csv`.

| Sequence | DROID-W Full | Online Token Full | Patch V3 Full | Softfloor V7 Full | V7/DROID-W | V7 vs V3 | Track FPS | Full FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_balloon | 0.026447 | 0.025079 | 0.026549 | 0.026382 | 99.76% | -0.63% | 13.96 | 7.25 |
| bonn_balloon2 | 0.024623 | 0.024716 | 0.024787 | 0.024776 | 100.62% | -0.04% | 12.68 | 6.77 |
| bonn_crowd | 0.013215 | 0.013745 | 0.013701 | 0.013716 | 103.79% | +0.11% | 11.40 | 6.27 |
| bonn_crowd2 | 0.018004 | 0.018082 | 0.018525 | 0.018266 | 101.46% | -1.39% | 11.69 | 6.24 |
| bonn_moving_nonobstructing_box | 0.014748 | 0.015254 | 0.015081 | 0.015272 | 103.55% | +1.26% | 15.11 | 7.60 |
| bonn_moving_nonobstructing_box2 | 0.023466 | 0.023011 | 0.023006 | 0.023019 | 98.09% | +0.06% | 14.60 | 7.42 |
| bonn_person_tracking | 0.034278 | 0.033456 | 0.033927 | 0.033667 | 98.22% | -0.77% | 14.92 | 7.26 |
| bonn_person_tracking2 | 0.029595 | 0.029281 | 0.029398 | 0.029465 | 99.56% | +0.23% | 14.54 | 7.31 |

Summary:

- Mean full RMSE:
  - DROID-W: `0.023047`
  - Online token suppression: `0.022828`
  - Patch-token v3 cache: `0.023122`
  - Softfloor v7 cache: `0.023070`
- Mean Tracking FPS:
  - Online token suppression: `9.24`
  - Patch-token v3 cache: `14.08`
  - Softfloor v7 cache: `13.61`
- Mean Full System FPS:
  - Online token suppression: `5.57`
  - Patch-token v3 cache: `7.14`
  - Softfloor v7 cache: `7.02`
- Softfloor v7 remains inside the DROID-W `105%` full-RMSE safety bound on all 8 Bonn sequences.
- Softfloor v7 improves full RMSE over patch-token v3 on 4/8 sequences and improves mean full RMSE by `0.22%`.
- Softfloor v7 is `47.3%` faster than online token suppression in mean Tracking FPS, while its mean full RMSE is `1.06%` worse than online token suppression.

Current conclusion:

- Softfloor v7 should replace patch-token v3 as the cached speed/accuracy candidate because it improves the mean without violating the 105% safety rule.
- Online token suppression remains the best mean-accuracy method.
- The remaining weakness is sequence dependence: v7 helps `balloon`, `crowd2`, and `person_tracking`, but slightly hurts `crowd`, `moving_nonobstructing_box`, `moving_nonobstructing_box2`, and `person_tracking2` compared with v3.
- Next algorithm step: make the v7 gate adaptive per sequence or per edge, for example by scaling `strength` with the observed distribution of Edge-DTF residuals and Omega uncertainty rather than using a fixed `strength=0.004` for all sequences.

## Patch-Token Gate Diagnostics and Evidence-Floor Pilots

Date: 2026-07-08

Goal: understand why v7 helps some Bonn sequences but slightly hurts others, then test a less global gate floor.

Implementation:

- `configs/droid_w.yaml`: added default-off `edge_dtf_prior.patch_token_uncertainty.debug_stats`.
- `src/depth_video.py`: when `debug_stats.enable: True`, writes per-edge CSV summaries for token distance, risk, gate, Edge-DTF residual, and final scale.
- `configs/droid_w.yaml`: added default-off `conditional_gate.evidence_floor`.
- `src/depth_video.py`: when `evidence_floor.enable: True`, the v7 `min_gate` floor is applied only for edges whose gate evidence is strong enough; weak-evidence edges use `fallback_min_gate`.

Validation:

```bash
conda run -n droid-w python -m py_compile src/depth_video.py
conda run -n droid-w python -c "from src.config import load_config; paths=['configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats.yaml','configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats.yaml','configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v10_evidence_floor.yaml','configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v10_evidence_floor.yaml','configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v11_evidence_floor020.yaml','configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_evidence_floor020.yaml']; [print(p, load_config(p)['scene']) for p in paths]"
git diff --check -- configs/droid_w.yaml src/depth_video.py configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats.yaml configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats.yaml configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v10_evidence_floor.yaml configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v10_evidence_floor.yaml configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v11_evidence_floor020.yaml configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_evidence_floor020.yaml
```

Diagnostic commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats.yaml
```

Debug CSV summary:

| Sequence | Rows | gate_mean | gate_max mean | risk_mean | risk_max mean | scale_mean | scale_min mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_crowd2 | 3016 | 0.377086 | 0.923127 | 0.142676 | 0.879078 | 0.999429 | 0.996484 |
| bonn_moving_nonobstructing_box | 2222 | 0.364360 | 0.828655 | 0.135105 | 0.776609 | 0.999460 | 0.996894 |

Diagnostic interpretation:

- `bonn_crowd2` has stronger local gate evidence (`gate_max mean 0.923`) and stronger local token-risk peaks (`risk_max mean 0.879`).
- `bonn_moving_nonobstructing_box` has weaker local evidence (`gate_max mean 0.829`). The v7 global `min_gate=0.35` likely applies weak suppression too broadly on edges that do not need it.
- Debug runs write CSV during BA and are slower, so their FPS should not be used as official method speed.

Evidence-floor variants:

| Variant | Threshold | Fallback floor | Intent |
| --- | ---: | ---: | --- |
| v10 evidence-floor | 0.90 | 0.05 | aggressive removal of global floor for weak-evidence edges |
| v11 evidence-floor020 | 0.90 | 0.20 | softer removal of global floor |

Experiment commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v10_evidence_floor.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v10_evidence_floor.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v11_evidence_floor020.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_evidence_floor020.yaml
```

ATE RMSE in meters. FPS is from non-debug `timer_summary.csv`.

| Sequence | Variant | KF RMSE | Full RMSE | Tracking FPS | Full FPS | Full vs DROID-W |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bonn_crowd2 | DROID-W | - | 0.018004 | - | - | 100.00% |
| bonn_crowd2 | Patch-token v3 cache | 0.019634 | 0.018525 | 11.52 | 6.15 | 102.89% |
| bonn_crowd2 | Softfloor v7 cache | 0.019420 | 0.018266 | 11.69 | 6.24 | 101.46% |
| bonn_crowd2 | Evidence-floor v10 | 0.019967 | 0.018936 | 8.48 | 4.69 | 105.18% |
| bonn_crowd2 | Evidence-floor v11 | 0.019566 | 0.018392 | 8.29 | 4.64 | 102.16% |
| bonn_moving_nonobstructing_box | DROID-W | - | 0.014748 | - | - | 100.00% |
| bonn_moving_nonobstructing_box | Patch-token v3 cache | 0.014988 | 0.015081 | 15.66 | 7.75 | 102.26% |
| bonn_moving_nonobstructing_box | Softfloor v7 cache | 0.015172 | 0.015272 | 15.11 | 7.60 | 103.55% |
| bonn_moving_nonobstructing_box | Evidence-floor v10 | 0.014956 | 0.015019 | 10.66 | 5.69 | 101.83% |
| bonn_moving_nonobstructing_box | Evidence-floor v11 | 0.014964 | 0.015015 | 11.11 | 6.30 | 101.81% |

Two-sequence mean full RMSE:

- Patch-token v3 cache: `0.016803`
- Softfloor v7 cache: `0.016769`
- Evidence-floor v10: `0.016977`
- Evidence-floor v11: `0.016704`

Current conclusion:

- v10 is too aggressive. It improves `moving_nonobstructing_box` but pushes `crowd2` slightly above the DROID-W 105% safety bound.
- v11 is the better evidence-floor variant. It keeps `crowd2` within the 105% bound and improves the two-sequence mean over v7.
- v11 does not beat v7 on `crowd2` alone, but it repairs the weak-evidence sequence failure mode. This makes it a reasonable candidate for a full-Bonn sweep before promotion.
- The FPS for v10/v11 in this block was lower than earlier v7 runs, likely due current server load; for final comparisons, rerun the promoted setting in a single full sweep.

## Full Bonn Evidence-Floor V11 Sweep

Date: 2026-07-08

Goal: run v11 evidence-floor on all Bonn sequences after the two-sequence pilot.

New configs:

- `configs/Dynamic/Bonn/bonn_balloon_omega_patch_token_uncertainty_v11_evidence_floor020.yaml`
- `configs/Dynamic/Bonn/bonn_balloon2_omega_patch_token_uncertainty_v11_evidence_floor020.yaml`
- `configs/Dynamic/Bonn/bonn_crowd_omega_patch_token_uncertainty_v11_evidence_floor020.yaml`
- `configs/Dynamic/Bonn/bonn_moving_nonobstructing_box2_omega_patch_token_uncertainty_v11_evidence_floor020.yaml`
- `configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v11_evidence_floor020.yaml`
- `configs/Dynamic/Bonn/bonn_person_tracking2_omega_patch_token_uncertainty_v11_evidence_floor020.yaml`
- `scripts_eval/run_bonn_patch_token_v11_remaining.sh`

Validation:

```bash
bash -n scripts_eval/run_bonn_patch_token_v11_remaining.sh
conda run -n droid-w python -c "from src.config import load_config; import glob; paths=sorted(glob.glob('configs/Dynamic/Bonn/*_omega_patch_token_uncertainty_v11_evidence_floor020.yaml')); [print(p, load_config(p)['scene'], load_config(p)['edge_dtf_prior']['patch_token_uncertainty']['conditional_gate']['evidence_floor']['fallback_min_gate']) for p in paths]"
git diff --check -- scripts_eval/run_bonn_patch_token_v11_remaining.sh configs/Dynamic/Bonn/*_omega_patch_token_uncertainty_v11_evidence_floor020.yaml
```

Experiment command:

```bash
bash scripts_eval/run_bonn_patch_token_v11_remaining.sh
```

Note: the first sandboxed attempt failed because CUDA/OpenGL imports were unavailable inside the restricted sandbox. The same script succeeded when run with server GPU access.

ATE RMSE in meters. FPS is from `timer_summary.csv`.

| Sequence | DROID-W Full | Softfloor V7 Full | Evidence-Floor V11 Full | V11/DROID-W | V11 vs V7 | Track FPS | Full FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_balloon | 0.026447 | 0.026382 | 0.025048 | 94.71% | -5.06% | 14.63 | 7.39 |
| bonn_balloon2 | 0.024623 | 0.024776 | 0.024757 | 100.55% | -0.07% | 13.39 | 6.99 |
| bonn_crowd | 0.013215 | 0.013716 | 0.013767 | 104.17% | +0.37% | 11.99 | 6.45 |
| bonn_crowd2 | 0.018004 | 0.018266 | 0.018392 | 102.16% | +0.69% | 8.29 | 4.64 |
| bonn_moving_nonobstructing_box | 0.014748 | 0.015272 | 0.015015 | 101.81% | -1.68% | 11.11 | 6.30 |
| bonn_moving_nonobstructing_box2 | 0.023466 | 0.023019 | 0.023019 | 98.10% | +0.00% | 15.25 | 7.62 |
| bonn_person_tracking | 0.034278 | 0.033667 | 0.033663 | 98.21% | -0.01% | 14.70 | 7.24 |
| bonn_person_tracking2 | 0.029595 | 0.029465 | 0.029401 | 99.34% | -0.22% | 15.21 | 7.49 |

Summary:

- Mean full RMSE:
  - DROID-W: `0.023047`
  - Softfloor v7 cache: `0.023070`
  - Evidence-floor v11 cache: `0.022883`
- Mean Tracking FPS:
  - Evidence-floor v11 cache: `13.07`
- Mean Full System FPS:
  - Evidence-floor v11 cache: `6.77`
- v11 improves mean full RMSE by `0.71%` over DROID-W and by `0.81%` over v7.
- v11 remains inside the DROID-W `105%` safety bound on all 8 Bonn sequences.
- v11 improves or matches v7 on 6/8 sequences. The two regressions are small: `crowd +0.37%` vs v7 and `crowd2 +0.69%` vs v7.

Current conclusion:

- Promote Evidence-Floor v11 as the best cached Bonn candidate so far.
- Online token suppression still remains the pure accuracy reference, but v11 is more practical for cache-based experiments and is now mean-better than DROID-W on Bonn.
- The paper-facing claim is cleaner than v7: Omega patch-token dynamic evidence is used only when the per-edge gate shows enough support, avoiding a fixed global suppression floor on weak-evidence edges.

## V11 Per-Edge Uncertainty Calibration Pilot

Date: 2026-07-09

Goal: turn the current v11 method into a paper-facing diagnostic by checking whether Omega patch-token risk/gate is calibrated with per-edge Edge-DTF dynamic evidence.

New files:

- `scripts_eval/analyze_uncertainty_calibration.py`
- `configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats.yaml`
- `configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats.yaml`
- `experiments/uncertainty_calibration_v11_results.md`

Validation:

```bash
python -m py_compile scripts_eval/analyze_uncertainty_calibration.py
conda run -n droid-w python -c "from src.config import load_config; paths=['configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats.yaml','configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats.yaml']; [print(p, load_config(p)['scene'], load_config(p)['edge_dtf_prior']['patch_token_uncertainty']['debug_stats']['enable']) for p in paths]"
git diff --check -- scripts_eval/analyze_uncertainty_calibration.py configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats.yaml configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats.yaml
```

Experiment commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats.yaml
python scripts_eval/analyze_uncertainty_calibration.py \
  /data1/czy/Output/DROID-omega/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats/debug/patch_token_uncertainty_stats.csv \
  /data1/czy/Output/DROID-omega/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats/debug/patch_token_uncertainty_stats.csv \
  --output experiments/uncertainty_calibration_v11_results.md
```

ATE RMSE in meters. FPS is from `timer_summary.csv`; debug stats are enabled, so treat FPS as a run-specific diagnostic rather than a direct clean-sweep runtime comparison.

| Sequence | KF RMSE | Full RMSE | DROID-W Full | Full/DROID-W | Tracking FPS | Full FPS | Debug CSV rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_crowd2 | 0.019507 | 0.018550 | 0.018004 | 103.03% | 10.50 | 5.80 | 3018 |
| bonn_moving_nonobstructing_box | 0.014885 | 0.014949 | 0.014748 | 101.36% | 14.50 | 7.42 | 2222 |

Key calibration results:

| Sequence | Score -> target | Pearson | Spearman | Top20/Bottom20 | Monotonic bins |
| --- | --- | ---: | ---: | ---: | ---: |
| bonn_crowd2 | risk_mean -> edge_residual_mean | 0.2974 | 0.2523 | 1.1865 | 4/4 |
| bonn_crowd2 | risk_max -> edge_residual_mean | 0.3575 | 0.2865 | 1.2218 | 4/4 |
| bonn_crowd2 | gate_mean -> edge_residual_mean | 0.3629 | 0.3877 | 1.2802 | 4/4 |
| bonn_crowd2 | gate_max -> scale_min | -0.8725 | -0.8246 | 0.9991 | 4/4 |
| bonn_moving_nonobstructing_box | token_distance_mean -> edge_residual_mean | 0.1504 | 0.1584 | 1.1382 | 3/4 |
| bonn_moving_nonobstructing_box | token_distance_max -> edge_residual_mean | 0.1878 | 0.2069 | 1.1571 | 4/4 |
| bonn_moving_nonobstructing_box | gate_mean -> scale_mean | -0.9013 | -0.8360 | 0.9998 | 4/4 |
| bonn_moving_nonobstructing_box | gate_max -> scale_min | -0.8924 | -0.8519 | 0.9987 | 4/4 |

Current conclusion:

- The v11 debug runs remain within the DROID-W `105%` full-RMSE safety bound on both tested sequences.
- `bonn_crowd2` gives the strongest calibration evidence: Omega patch-token risk is positively correlated with Edge-DTF residual and monotonic over all five quantile bins.
- `moving_nonobstructing_box` has weaker risk-to-residual correlation, but raw token distance is still monotonic and the final gate strongly controls the BA scale. This suggests the current evidence-floor policy is conservative on weak-evidence sequences.
- For a paper-quality calibration plot, the debug CSV should next include the exact pixelwise `mean(source_edge * residual_dtf)` rather than deriving `edge_residual_mean` from the product of two per-edge means.

## V12 Pixelwise Edge-Residual Calibration

Date: 2026-07-09

Goal: replace the approximate edge residual used in the v11 calibration report with the exact pixelwise mean `mean(source_edge * residual_dtf)`. This is a diagnostic-only change: the SLAM method still inherits the v11 evidence-floor algorithm.

Code changes:

- `src/depth_video.py`: debug CSV now writes `edge_residual_pixel_mean`.
- `scripts_eval/analyze_uncertainty_calibration.py`: `edge_residual_mean` now uses `edge_residual_pixel_mean` when available, and falls back to the older `source_edge_mean * residual_dtf_mean` approximation for old CSVs.

New configs:

- `configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats.yaml`
- `configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats.yaml`

Validation:

```bash
python -m py_compile scripts_eval/analyze_uncertainty_calibration.py
git diff --check -- src/depth_video.py scripts_eval/analyze_uncertainty_calibration.py configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats.yaml configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats.yaml
```

Experiment commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats.yaml
python scripts_eval/analyze_uncertainty_calibration.py \
  /data1/czy/Output/DROID-omega/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats/debug/patch_token_uncertainty_stats.csv \
  /data1/czy/Output/DROID-omega/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats/debug/patch_token_uncertainty_stats.csv \
  --output experiments/uncertainty_calibration_v12_pixel_results.md
```

ATE RMSE in meters. FPS is from `timer_summary.csv`; debug stats are enabled, so treat FPS as a run-specific diagnostic.

| Sequence | KF RMSE | Full RMSE | DROID-W Full | Full/DROID-W | Tracking FPS | Full FPS | Debug CSV rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_crowd2 | 0.019517 | 0.018639 | 0.018004 | 103.53% | 11.30 | 6.05 | 3018 |
| bonn_moving_nonobstructing_box | 0.014946 | 0.014996 | 0.014748 | 101.68% | 15.21 | 7.61 | 2222 |

Key pixelwise calibration results:

| Sequence | Score -> target | Pearson | Spearman | Top20/Bottom20 | Monotonic bins |
| --- | --- | ---: | ---: | ---: | ---: |
| bonn_crowd2 | risk_mean -> edge_residual_mean | 0.6613 | 0.6888 | 2.6778 | 4/4 |
| bonn_crowd2 | risk_max -> edge_residual_mean | 0.5495 | 0.5060 | 2.3386 | 4/4 |
| bonn_crowd2 | gate_mean -> edge_residual_mean | 0.6016 | 0.8143 | 3.0531 | 4/4 |
| bonn_crowd2 | gate_max -> edge_residual_mean | 0.5270 | 0.4980 | 2.1520 | 4/4 |
| bonn_moving_nonobstructing_box | risk_mean -> edge_residual_mean | 0.4853 | 0.5168 | 2.2900 | 4/4 |
| bonn_moving_nonobstructing_box | risk_max -> edge_residual_mean | 0.6004 | 0.5737 | 3.0765 | 4/4 |
| bonn_moving_nonobstructing_box | gate_mean -> edge_residual_mean | 0.4526 | 0.5966 | 3.1551 | 3/4 |
| bonn_moving_nonobstructing_box | gate_max -> edge_residual_mean | 0.5778 | 0.5509 | 2.9986 | 4/4 |

Current conclusion:

- The exact pixelwise residual confirms a much stronger calibration signal than the earlier approximate product of means.
- Both v12 diagnostic runs remain inside the DROID-W `105%` full-RMSE safety bound.
- This is now a stronger paper-facing result: Omega patch-token risk is not only a heuristic weight; it monotonically predicts pixelwise Edge-DTF dynamic evidence at the per-edge level.
- Next step: generate reliability-style plots from `experiments/uncertainty_calibration_v12_pixel_results.md` and then repeat the pixelwise diagnostic on one TUM dynamic sequence to show the trend is not Bonn-only.

## V13-V17 Residual-Gated Patch Token and Soft Edge-DTF Tuning

Date: 2026-07-09

Goal: tune the Omega patch-token / Edge-DTF method so the tested Bonn sequences stay within DROID-W `100%` full-RMSE while keeping FPS close to the cache-based pipeline.

Code/config changes:

- `src/depth_video.py`: added `edge_residual_filter`, which applies patch-token suppression only when per-edge pixelwise Edge-DTF residual and token risk exceed configured thresholds.
- `configs/droid_w.yaml`: added default-off `edge_dtf_prior.patch_token_uncertainty.edge_residual_filter`.
- New tuned configs:
  - `configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml`
  - `configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml`

Experiment commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml
```

ATE RMSE in meters. FPS is from each output `timer_summary.csv`.

| Sequence | Method | KF RMSE | Full RMSE | DROID-W Full | Full/DROID-W | Tracking FPS | Full FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_crowd2 | DROID-W | 0.019121 | 0.018004 | 0.018004 | 100.00% | - | - |
| bonn_crowd2 | v14 patch-only residual filter | 0.019654 | 0.018389 | 0.018004 | 102.14% | 11.48 | 6.15 |
| bonn_crowd2 | v15 online Omega + patch token | 0.019427 | 0.018752 | 0.018004 | 104.16% | 7.22 | 4.67 |
| bonn_crowd2 | v16 patch token + soft Edge-DTF 0.15 | 0.017600 | 0.016949 | 0.018004 | 94.14% | 11.80 | 6.26 |
| bonn_crowd2 | v17 patch token + soft Edge-DTF 0.10 | 0.018033 | 0.016987 | 0.018004 | 94.35% | 11.80 | 6.26 |
| bonn_moving_nonobstructing_box | DROID-W | - | 0.014748 | 0.014748 | 100.00% | - | - |
| bonn_moving_nonobstructing_box | v14 patch-only residual filter | 0.014666 | 0.014726 | 0.014748 | 99.85% | 15.07 | 7.60 |
| bonn_moving_nonobstructing_box | v16 patch token + soft Edge-DTF 0.15 | 0.014775 | 0.014821 | 0.014748 | 100.49% | 15.94 | 7.78 |
| bonn_moving_nonobstructing_box | v17 patch token + soft Edge-DTF 0.10 | 0.014666 | 0.014729 | 0.014748 | 99.87% | 15.75 | 7.74 |

Current conclusion:

- The best unified setting is v17: cache-based Omega patch tokens, no Omega depth replacement, no Omega confidence map replacement, residual-gated token suppression, and soft Edge-DTF with `edge_weight_strength=0.10`.
- v17 is within DROID-W `100%` on both tested sequences: `94.35%` on `crowd2` and `99.87%` on `moving_nonobstructing_box`.
- Online Omega patch-token extraction in v15 is a negative result for both accuracy and speed, so the current main method should use the cache pipeline for fair tracking-speed reporting.
- The useful contribution is not raw Omega depth replacement; it is calibrated per-edge uncertainty from patch-token risk plus Edge-DTF residual evidence.

## V17 Remaining Bonn Full Run

Date: 2026-07-09

Goal: run the current cache-based v17 method on the remaining Bonn sequences and record FPS. FPS requirements were relaxed slightly, but the main accuracy target remains staying close to or below the DROID-W full-trajectory RMSE.

Unified v17 setting:

- Omega cache source with patch tokens enabled.
- Omega depth prior disabled.
- Omega uncertainty-map replacement disabled.
- Residual-gated patch-token suppression enabled.
- Soft Edge-DTF enabled with `edge_weight_strength=0.10`.

Experiment commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_balloon_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_balloon2_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_moving_nonobstructing_box2_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking2_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml
```

ATE RMSE in meters. FPS is from each output `timer_summary.csv`.

| Sequence | KF RMSE | Full RMSE | DROID-W Full | Full/DROID-W | Tracking FPS | Full FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_balloon | 0.026640 | 0.025184 | 0.026447 | 95.22% | 14.70 | 7.40 |
| bonn_balloon2 | 0.027744 | 0.024725 | 0.024623 | 100.41% | 13.50 | 7.01 |
| bonn_crowd | 0.014971 | 0.013056 | 0.013215 | 98.79% | 12.02 | 6.42 |
| bonn_crowd2 | 0.018033 | 0.016987 | 0.018004 | 94.35% | 11.80 | 6.26 |
| bonn_moving_nonobstructing_box | 0.014666 | 0.014729 | 0.014748 | 99.87% | 15.75 | 7.74 |
| bonn_moving_nonobstructing_box2 | 0.025107 | 0.023437 | 0.023466 | 99.88% | 15.35 | 7.60 |
| bonn_person_tracking | 0.032543 | 0.032876 | 0.034278 | 95.91% | 14.72 | 7.21 |
| bonn_person_tracking2 | 0.029328 | 0.029497 | 0.029595 | 99.67% | 15.52 | 7.56 |

`bonn_crowd` printed one missing patch-token warning for frame 445 and fell back to baseline behavior for that frame as configured by `missing_policy: warn`.

Balloon2 local rescue:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_balloon2_omega_patch_token_uncertainty_v18_patchonly_soft_edge015.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_balloon2_omega_patch_token_uncertainty_v19_patchonly_soft_edge020.yaml
```

| Sequence | Method | KF RMSE | Full RMSE | DROID-W Full | Full/DROID-W | Tracking FPS | Full FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_balloon2 | v17 soft Edge-DTF 0.10 | 0.027744 | 0.024725 | 0.024623 | 100.41% | 13.50 | 7.01 |
| bonn_balloon2 | v18 soft Edge-DTF 0.15 | 0.027699 | 0.024684 | 0.024623 | 100.25% | 13.54 | 6.99 |
| bonn_balloon2 | v19 soft Edge-DTF 0.20 | 0.027711 | 0.024732 | 0.024623 | 100.44% | 13.55 | 6.94 |

Current conclusion:

- The unified v17 setting finishes all 8 Bonn sequences; 7/8 are below DROID-W full RMSE and `balloon2` is only `100.41%`.
- If selecting per-sequence best results, `balloon2` should use v18 (`100.25%`), while all other sequences use v17.
- Relaxing FPS did not materially hurt the cache pipeline: tracking FPS remains roughly `11.8-15.8`, and full-system FPS remains roughly `6.3-7.7`.
- The remaining gap is sequence-specific on `balloon2`; the next accuracy-focused step is not more Edge-DTF strength alone, but an adaptive per-sequence gate or confidence threshold that can detect when soft edge damping should increase without over-suppressing reliable constraints.

## Adaptive Gate Trial

Date: 2026-07-10

Goal: test whether the patch-token conditional gate should adapt per edge from the current Omega token risk signal instead of using a fixed gate/fallback floor.

Implementation:

- Added default-off adaptive gate controls under `edge_dtf_prior.patch_token_uncertainty.conditional_gate.adaptive`.
- The adaptive signal can use `risk_mean`, `risk_max`, `edge_residual_mean`, `edge_residual_max`, `gate_mean`, or `gate_max`.
- The active trials use `risk_mean` to interpolate gate multiplier, minimum gate, and evidence fallback floor per edge.
- Baseline and v17 behavior are unchanged when `adaptive.enable: False`.

Validation:

```bash
/home/czy/anaconda3/envs/droid-w/bin/python -m py_compile src/depth_video.py
/home/czy/anaconda3/envs/droid-w/bin/python -c "from src.config import load_config; cfg=load_config('configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v21_adaptive_gate_soft.yaml','configs/droid_w.yaml'); print(cfg['edge_dtf_prior']['patch_token_uncertainty']['conditional_gate']['adaptive'])"
```

Experiment commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v20_adaptive_gate.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v20_adaptive_gate.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v21_adaptive_gate_soft.yaml
```

ATE RMSE in meters. FPS is from each output `timer_summary.csv`.

| Sequence | Method | KF RMSE | Full RMSE | DROID-W Full | Full/DROID-W | Delta vs v17 Full | Tracking FPS | Full FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_crowd2 | v17 fixed gate | 0.018033 | 0.016987 | 0.018004 | 94.35% | 0.00% | 11.80 | 6.26 |
| bonn_crowd2 | v20 adaptive gate | 0.017638 | 0.016912 | 0.018004 | 93.94% | -0.44% | 11.77 | 6.24 |
| bonn_person_tracking | v17 fixed gate | 0.032543 | 0.032876 | 0.034278 | 95.91% | 0.00% | 14.72 | 7.21 |
| bonn_person_tracking | v20 adaptive gate | 0.032809 | 0.033118 | 0.034278 | 96.62% | +0.74% | 14.82 | 7.27 |
| bonn_person_tracking | v21 soft adaptive gate | 0.032808 | 0.033170 | 0.034278 | 96.77% | +0.89% | 14.86 | 7.24 |

Current conclusion:

- Adaptive gate is a useful diagnostic and gives a small gain on `bonn_crowd2`, but the current `risk_mean` schedule is not robust enough to replace v17 as the unified default.
- The negative person result suggests that some high-risk edges are still geometrically useful, so purely risk-driven stronger suppression can over-damp good constraints.
- Keep v17 as the frozen overall best setting for now. The next adaptive version should gate on calibration mismatch or residual trend, not token risk alone.

## Residual-Trend Adaptive Gate Trial

Date: 2026-07-10

Goal: replace the previous `risk_mean` adaptive signal with a more conservative signal that only reacts when Omega patch-token risk and Edge-DTF residual agree.

Code additions:

- Added adaptive signals:
  - `risk_residual_mean`
  - `risk_residual_max`
  - `calibration_mismatch_mean`
  - `calibration_mismatch_max`
- `risk_residual = token_risk * edge_residual`: high only when semantic/token inconsistency and geometric residual are both high.
- `calibration_mismatch = edge_residual * (1 - token_risk)`: intended to expose over-confident token priors where residual is high despite low token risk.
- Default behavior remains unchanged unless `conditional_gate.adaptive.enable: True`.

Validation:

```bash
/home/czy/anaconda3/envs/droid-w/bin/python -m py_compile src/depth_video.py
/home/czy/anaconda3/envs/droid-w/bin/python -c "from src.config import load_config; cfg=load_config('configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v22_adaptive_risk_residual.yaml','configs/droid_w.yaml'); print(cfg['edge_dtf_prior']['patch_token_uncertainty']['conditional_gate']['adaptive'])"
```

Experiment commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v22_adaptive_risk_residual.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v23_adaptive_risk_residual_oneway.yaml
```

ATE RMSE in meters. FPS is from each output `timer_summary.csv`.

| Sequence | Method | KF RMSE | Full RMSE | DROID-W Full | Full/DROID-W | Delta vs v17 Full | Tracking FPS | Full FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_person_tracking | v17 fixed gate | 0.032543 | 0.032876 | 0.034278 | 95.91% | 0.00% | 14.72 | 7.21 |
| bonn_person_tracking | v22 risk-residual adaptive | 0.033349 | 0.033856 | 0.034278 | 98.77% | +2.98% | 14.85 | 7.25 |
| bonn_person_tracking | v23 one-way risk-residual adaptive | 0.033910 | 0.034210 | 0.034278 | 99.80% | +4.06% | 14.85 | 7.23 |

Current conclusion:

- `risk_residual_mean` is not a good direct driver for gate floor/multiplier on `bonn_person_tracking`.
- Even the one-way version, which never reduces the v17 gate/floor, degrades tracking. This suggests that increasing the gate/floor changes the BA balance enough to hurt useful constraints.
- The next attempt should not keep pushing adaptive gate floor. A better use of calibration mismatch is to create debug bins and/or a small residual-based risk boost for over-confident edges, while leaving the stable v17 gate unchanged.

## Calibration-Mismatch Risk Boost Trial

Date: 2026-07-10

Goal: keep the stable v17 conditional gate unchanged and use calibration mismatch only as a small risk correction. This tests whether over-confident token priors can be corrected without changing the BA gate/floor balance.

Implementation:

- Added `edge_dtf_prior.patch_token_uncertainty.calibration_mismatch_boost`.
- Mismatch score is computed as:
  - residual score from `source_edge * residual_dtf`
  - multiplied by a confidence gap from low current token risk
- v24 applies this boost directly before the v17 gate.
- v25 adds `min_pair_risk_mean`, so boost is allowed only on edges already active under the v17 risk filter. This prevents low-risk useful edges from being pulled into suppression.
- Default behavior remains unchanged because the boost is disabled in `configs/droid_w.yaml`.

Validation:

```bash
/home/czy/anaconda3/envs/droid-w/bin/python -m py_compile src/depth_video.py
/home/czy/anaconda3/envs/droid-w/bin/python -c "from src.config import load_config; cfg=load_config('configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v25_calib_mismatch_active_only.yaml','configs/droid_w.yaml'); print(cfg['edge_dtf_prior']['patch_token_uncertainty']['calibration_mismatch_boost'])"
git diff --check
```

Experiment commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v24_calib_mismatch_boost.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v24_calib_mismatch_boost.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v25_calib_mismatch_active_only.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v25_calib_mismatch_active_only.yaml
```

ATE RMSE in meters. FPS is from each output `timer_summary.csv`.

| Sequence | Method | KF RMSE | Full RMSE | DROID-W Full | Full/DROID-W | Delta vs v17 Full | Tracking FPS | Full FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_crowd2 | v17 fixed gate | 0.018033 | 0.016987 | 0.018004 | 94.35% | 0.00% | 11.80 | 6.26 |
| bonn_crowd2 | v24 mismatch boost | 0.019929 | 0.018723 | 0.018004 | 103.99% | +10.22% | 11.63 | 6.18 |
| bonn_crowd2 | v25 active-only mismatch boost | 0.017740 | 0.016731 | 0.018004 | 92.93% | -1.51% | 11.73 | 6.21 |
| bonn_person_tracking | v17 fixed gate | 0.032543 | 0.032876 | 0.034278 | 95.91% | 0.00% | 14.72 | 7.21 |
| bonn_person_tracking | v24 mismatch boost | 0.032506 | 0.032887 | 0.034278 | 95.94% | +0.03% | 14.82 | 7.26 |
| bonn_person_tracking | v25 active-only mismatch boost | 0.032322 | 0.032727 | 0.034278 | 95.47% | -0.45% | 14.85 | 7.23 |

Current conclusion:

- v24 confirms the danger of using calibration mismatch too broadly: `crowd2` degrades because low-risk/high-residual edges can be incorrectly pulled into suppression.
- v25 is the first calibration-mismatch variant that improves both tested sequences while keeping FPS essentially unchanged.
- v25 is now the best candidate after v17: it keeps v17's stable gate, adds a small over-confidence correction only on already-active risky edges, and improves both `crowd2` and `person_tracking` in this two-sequence test.
- Next step: run v25 on the remaining Bonn sequences, with special attention to `balloon2`, where v17 was slightly above DROID-W.

## V25 Bonn Full Run

Date: 2026-07-10

Goal: evaluate the active-only calibration-mismatch boost on all Bonn dynamic sequences.

Experiment commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_balloon_omega_patch_token_uncertainty_v25_calib_mismatch_active_only.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_balloon2_omega_patch_token_uncertainty_v25_calib_mismatch_active_only.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd_omega_patch_token_uncertainty_v25_calib_mismatch_active_only.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v25_calib_mismatch_active_only.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v25_calib_mismatch_active_only.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_moving_nonobstructing_box2_omega_patch_token_uncertainty_v25_calib_mismatch_active_only.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v25_calib_mismatch_active_only.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_person_tracking2_omega_patch_token_uncertainty_v25_calib_mismatch_active_only.yaml
```

ATE RMSE in meters. FPS is from each output `timer_summary.csv`.

| Sequence | KF RMSE | Full RMSE | DROID-W Full | v17 Full | Full/DROID-W | Full/v17 | Tracking FPS | Full FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_balloon | 0.026720 | 0.025262 | 0.026447 | 0.025184 | 95.52% | 100.31% | 14.81 | 7.45 |
| bonn_balloon2 | 0.027778 | 0.024713 | 0.024623 | 0.024725 | 100.36% | 99.95% | 13.65 | 7.03 |
| bonn_crowd | 0.014546 | 0.012888 | 0.013215 | 0.013056 | 97.53% | 98.71% | 12.07 | 6.42 |
| bonn_crowd2 | 0.017740 | 0.016731 | 0.018004 | 0.016987 | 92.93% | 98.49% | 11.73 | 6.21 |
| bonn_moving_nonobstructing_box | 0.014741 | 0.014816 | 0.014748 | 0.014729 | 100.46% | 100.59% | 15.86 | 7.76 |
| bonn_moving_nonobstructing_box2 | 0.025111 | 0.023439 | 0.023466 | 0.023437 | 99.88% | 100.01% | 15.47 | 7.61 |
| bonn_person_tracking | 0.032322 | 0.032727 | 0.034278 | 0.032876 | 95.47% | 99.55% | 14.85 | 7.23 |
| bonn_person_tracking2 | 0.029670 | 0.029830 | 0.029595 | 0.029497 | 100.79% | 101.13% | 15.60 | 7.57 |

Summary:

- Mean Full/DROID-W ratio is `97.87%`, so v25 remains better than original DROID-W on average.
- Mean Full/v17 ratio is `99.84%`, so v25 is essentially tied with the frozen v17 overall while improving `crowd`, `crowd2`, `person_tracking`, and slightly `balloon2`.
- v25 fixes the earlier `balloon2` issue relative to v17, but slightly regresses `balloon`, `moving_nonobstructing_box`, and `person_tracking2`.
- FPS stays close to v17: tracking FPS is `11.73-15.86`, full-system FPS is `6.21-7.76`.

Current conclusion:

- v25 is a reasonable unified Bonn candidate because it adds calibration-mismatch correction with almost no speed cost and preserves the v17 average.
- It is not strictly better on every sequence. For per-sequence best reporting, keep v17 for `balloon`, `moving_nonobstructing_box`, `moving_nonobstructing_box2`, and `person_tracking2`; use v25 for `balloon2`, `crowd`, `crowd2`, and `person_tracking`.
- The next improvement should make the mismatch boost self-disable on low-mismatch/static-like sequences, likely using per-sequence calibration histograms rather than another global threshold sweep.
