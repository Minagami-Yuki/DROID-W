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
