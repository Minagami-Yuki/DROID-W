# Omega V17 Frozen Method Results

Date: 2026-07-10

## Frozen Method

V17 is the current overall best Bonn setting and is frozen as a cache-based method:

- `omega_prior.source: cache`
- Omega depth replacement disabled
- Omega uncertainty replacement disabled
- cached Omega patch tokens enabled with `dim=8`, `group_mean`, L2 normalization
- Edge-DTF enabled with `edge_weight_strength=0.10`
- residual-gated patch-token dynamic factor suppression:
  - `strength=0.004`
  - `min_scale=0.996`
  - `max_scale=1.0`
  - `edge_residual_filter.min_residual=0.030`
  - `edge_residual_filter.min_risk_mean=0.140`

Frozen cross-dataset configs:

```text
configs/Static/7Scenes/chess_seq01_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml
configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml
```

Cache generation configs:

```text
configs/Static/7Scenes/chess_seq01_omega_cache_write.yaml
configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz_omega_cache_write.yaml
```

## Commands

```bash
conda run -n droid-w python run.py --config configs/Static/7Scenes/chess_seq01_omega_cache_write.yaml
conda run -n droid-w python run.py --config configs/Static/7Scenes/chess_seq01_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml

conda run -n droid-w python run.py --config configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz_omega_cache_write.yaml
conda run -n droid-w python run.py --config configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz_omega_patch_token_uncertainty_v17_patchonly_soft_edge010.yaml
```

## Tracking Results

ATE RMSE in meters. FPS is from the final v17-cache run, not cache generation.

| Dataset | Sequence | Frames | KF RMSE | Full RMSE | KF Mean | Full Mean | Tracking FPS | Full FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7Scenes | chess/seq-01 | 1000 | 0.037811 | 0.037041 | 0.033384 | 0.033068 | 15.76 | 8.09 |
| TUM RGB-D dynamic | freiburg3_walking_xyz | 858 | 0.012266 | 0.012169 | 0.010616 | 0.010503 | 14.46 | 7.71 |

Cache generation runtime:

| Dataset | Sequence | Frames | Tracking FPS | Full FPS |
| --- | --- | ---: | ---: | ---: |
| 7Scenes | chess/seq-01 | 1000 | 11.08 | 6.75 |
| TUM RGB-D dynamic | freiburg3_walking_xyz | 858 | 10.27 | 6.46 |

## Calibration Figure

Generated with:

```bash
/home/czy/anaconda3/envs/droid-w/bin/python scripts_eval/plot_uncertainty_calibration.py \
  /data1/czy/Output/DROID-omega/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats/debug/patch_token_uncertainty_stats.csv \
  /data1/czy/Output/DROID-omega/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats/debug/patch_token_uncertainty_stats.csv \
  --output experiments/figures/omega_patch_token_calibration.png \
  --pdf experiments/figures/omega_patch_token_calibration.pdf \
  --bins 8
```

Outputs:

```text
experiments/figures/omega_patch_token_calibration.png
experiments/figures/omega_patch_token_calibration.pdf
```

The figure uses exact `edge_residual_pixel_mean` statistics from two Bonn dynamic sequences. Pearson correlation between patch-token scores and edge residual is positive:

| Scene | risk_mean vs edge residual | gate_mean vs edge residual |
| --- | ---: | ---: |
| Bonn crowd2 | 0.66 | 0.60 |
| Bonn moving box | 0.49 | 0.45 |

## Takeaways

- V17 is frozen as a cache-based method so the original DROID-W baseline remains unchanged and final-run FPS is not dominated by online Omega inference.
- On 7Scenes `chess/seq-01`, v17 is effectively equal to DROID-W tracking accuracy and speed.
- On TUM `freiburg3_walking_xyz`, v17 is within `0.3%` full RMSE of the previous online Omega setting and is faster in the final run.
- The calibration figure supports the paper claim that Omega patch-token inconsistency is predictive of geometric residual and can be used as a dynamic factor prior.
