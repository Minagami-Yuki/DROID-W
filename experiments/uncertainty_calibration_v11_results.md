# Omega Patch-Token Uncertainty Calibration

Date: 2026-07-09

Goal: test whether the Omega patch-token uncertainty/gate is calibrated with Edge-DTF dynamic evidence at the per-edge level. Higher token/risk/gate should correspond to higher Edge-DTF residual; higher gate should also correspond to lower final BA scale.

Input CSVs:

- `/data1/czy/Output/DROID-omega/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats/debug/patch_token_uncertainty_stats.csv`
- `/data1/czy/Output/DROID-omega/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats/debug/patch_token_uncertainty_stats.csv`

## Key Correlations

| Scene | Score | Target | Rows | Pearson | Spearman | Top20/Bottom20 | Monotonic bins |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats | gate_max | edge_residual_mean | 3018 | 0.3704 | 0.2766 | 1.2191 | 4/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats | gate_max | scale_min | 3018 | -0.8725 | -0.8246 | 0.9991 | 4/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats | gate_mean | edge_residual_mean | 3018 | 0.3629 | 0.3877 | 1.2802 | 4/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats | gate_mean | scale_mean | 3018 | 0.1580 | -0.8870 | 1.0031 | 3/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats | risk_max | edge_residual_mean | 3018 | 0.3575 | 0.2865 | 1.2218 | 4/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats | risk_mean | edge_residual_mean | 3018 | 0.2974 | 0.2523 | 1.1865 | 4/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats | token_distance_max | edge_residual_mean | 3018 | 0.0487 | -0.0215 | 0.9994 | 1/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats | token_distance_mean | edge_residual_mean | 3018 | 0.0252 | -0.0252 | 1.0056 | 2/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats | gate_max | edge_residual_mean | 2222 | 0.0147 | 0.0099 | 1.0096 | 1/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats | gate_max | scale_min | 2222 | -0.8924 | -0.8519 | 0.9987 | 4/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats | gate_mean | edge_residual_mean | 2222 | 0.0316 | -0.0010 | 1.0161 | 2/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats | gate_mean | scale_mean | 2222 | -0.9013 | -0.8360 | 0.9998 | 4/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats | risk_max | edge_residual_mean | 2222 | 0.0496 | 0.0529 | 1.0607 | 2/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats | risk_mean | edge_residual_mean | 2222 | 0.0520 | 0.0394 | 1.0383 | 2/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats | token_distance_max | edge_residual_mean | 2222 | 0.1878 | 0.2069 | 1.1571 | 4/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats | token_distance_mean | edge_residual_mean | 2222 | 0.1504 | 0.1584 | 1.1382 | 3/4 |

## Quantile Bins

### bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats: gate_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.7834 | 0.0000-0.8727 | 0.0353 |
| 2 | 604 | 0.9028 | 0.8731-0.9286 | 0.0395 |
| 3 | 603 | 0.9502 | 0.9286-0.9684 | 0.0411 |
| 4 | 604 | 0.9812 | 0.9684-0.9905 | 0.0417 |
| 5 | 604 | 0.9958 | 0.9905-1.0000 | 0.0430 |

### bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats: gate_max -> scale_min

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.7834 | 0.0000-0.8727 | 0.9971 |
| 2 | 604 | 0.9028 | 0.8731-0.9286 | 0.9966 |
| 3 | 603 | 0.9502 | 0.9286-0.9684 | 0.9964 |
| 4 | 604 | 0.9812 | 0.9684-0.9905 | 0.9963 |
| 5 | 604 | 0.9958 | 0.9905-1.0000 | 0.9961 |

### bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats: gate_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.2550 | 0.0000-0.2718 | 0.0359 |
| 2 | 604 | 0.3298 | 0.2719-0.3687 | 0.0382 |
| 3 | 603 | 0.3741 | 0.3687-0.3790 | 0.0390 |
| 4 | 604 | 0.3837 | 0.3790-0.3894 | 0.0415 |
| 5 | 604 | 0.3973 | 0.3894-0.4215 | 0.0460 |

### bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats: gate_mean -> scale_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.2550 | 0.0000-0.2718 | 0.9963 |
| 2 | 604 | 0.3298 | 0.2719-0.3687 | 0.9995 |
| 3 | 603 | 0.3741 | 0.3687-0.3790 | 0.9994 |
| 4 | 604 | 0.3837 | 0.3790-0.3894 | 0.9994 |
| 5 | 604 | 0.3973 | 0.3894-0.4215 | 0.9994 |

### bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats: risk_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.7077 | 0.0000-0.8093 | 0.0352 |
| 2 | 604 | 0.8474 | 0.8093-0.8775 | 0.0392 |
| 3 | 603 | 0.9030 | 0.8775-0.9272 | 0.0407 |
| 4 | 604 | 0.9484 | 0.9272-0.9684 | 0.0423 |
| 5 | 604 | 0.9862 | 0.9685-1.0000 | 0.0431 |

### bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats: risk_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.0913 | 0.0000-0.1035 | 0.0367 |
| 2 | 604 | 0.1200 | 0.1035-0.1331 | 0.0389 |
| 3 | 603 | 0.1387 | 0.1331-0.1438 | 0.0404 |
| 4 | 604 | 0.1489 | 0.1438-0.1538 | 0.0410 |
| 5 | 604 | 0.1619 | 0.1539-0.1950 | 0.0435 |

### bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats: token_distance_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 1.2776 | 0.0000-1.3221 | 0.0402 |
| 2 | 604 | 1.3461 | 1.3221-1.3666 | 0.0398 |
| 3 | 603 | 1.3863 | 1.3668-1.4073 | 0.0402 |
| 4 | 604 | 1.4323 | 1.4074-1.4587 | 0.0402 |
| 5 | 604 | 1.5098 | 1.4588-1.6741 | 0.0401 |

### bonn_crowd2_omega_patch_token_uncertainty_v11_debug_stats: token_distance_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.2530 | 0.0000-0.2705 | 0.0396 |
| 2 | 604 | 0.2776 | 0.2705-0.2846 | 0.0404 |
| 3 | 603 | 0.2912 | 0.2847-0.2974 | 0.0401 |
| 4 | 604 | 0.3041 | 0.2974-0.3109 | 0.0405 |
| 5 | 604 | 0.3241 | 0.3109-0.3878 | 0.0399 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats: gate_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.6337 | 0.4026-0.7560 | 0.0324 |
| 2 | 444 | 0.7939 | 0.7561-0.8283 | 0.0324 |
| 3 | 445 | 0.8545 | 0.8283-0.8758 | 0.0320 |
| 4 | 444 | 0.8997 | 0.8759-0.9304 | 0.0342 |
| 5 | 445 | 0.9609 | 0.9305-0.9992 | 0.0327 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats: gate_max -> scale_min

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.6337 | 0.4026-0.7560 | 0.9977 |
| 2 | 444 | 0.7939 | 0.7561-0.8283 | 0.9970 |
| 3 | 445 | 0.8545 | 0.8283-0.8758 | 0.9967 |
| 4 | 444 | 0.8997 | 0.8759-0.9304 | 0.9966 |
| 5 | 445 | 0.9609 | 0.9305-0.9992 | 0.9964 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats: gate_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.2399 | 0.2209-0.2480 | 0.0334 |
| 2 | 444 | 0.2536 | 0.2480-0.2588 | 0.0316 |
| 3 | 445 | 0.2646 | 0.2588-0.2713 | 0.0326 |
| 4 | 444 | 0.3174 | 0.2713-0.3665 | 0.0323 |
| 5 | 445 | 0.3747 | 0.3665-0.4007 | 0.0339 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats: gate_mean -> scale_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.2399 | 0.2209-0.2480 | 0.9997 |
| 2 | 444 | 0.2536 | 0.2480-0.2588 | 0.9996 |
| 3 | 445 | 0.2646 | 0.2588-0.2713 | 0.9996 |
| 4 | 444 | 0.3174 | 0.2713-0.3665 | 0.9995 |
| 5 | 445 | 0.3747 | 0.3665-0.4007 | 0.9994 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats: risk_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.5580 | 0.3721-0.6753 | 0.0319 |
| 2 | 444 | 0.7357 | 0.6755-0.7777 | 0.0317 |
| 3 | 445 | 0.8060 | 0.7778-0.8329 | 0.0332 |
| 4 | 444 | 0.8563 | 0.8330-0.8808 | 0.0331 |
| 5 | 445 | 0.9269 | 0.8810-0.9960 | 0.0338 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats: risk_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.0796 | 0.0600-0.0888 | 0.0324 |
| 2 | 444 | 0.0934 | 0.0888-0.0972 | 0.0322 |
| 3 | 445 | 0.1011 | 0.0972-0.1056 | 0.0329 |
| 4 | 444 | 0.1163 | 0.1056-0.1357 | 0.0326 |
| 5 | 445 | 0.1442 | 0.1358-0.1635 | 0.0336 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats: token_distance_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 1.3277 | 1.1882-1.3679 | 0.0305 |
| 2 | 444 | 1.3922 | 1.3681-1.4140 | 0.0309 |
| 3 | 445 | 1.4354 | 1.4140-1.4568 | 0.0327 |
| 4 | 444 | 1.4797 | 1.4568-1.5058 | 0.0343 |
| 5 | 445 | 1.5527 | 1.5059-1.6985 | 0.0354 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v11_debug_stats: token_distance_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.2413 | 0.2009-0.2664 | 0.0314 |
| 2 | 444 | 0.2787 | 0.2665-0.2873 | 0.0321 |
| 3 | 445 | 0.2943 | 0.2873-0.3004 | 0.0317 |
| 4 | 444 | 0.3073 | 0.3004-0.3152 | 0.0327 |
| 5 | 445 | 0.3287 | 0.3152-0.3735 | 0.0358 |

## Notes

- `edge_residual_mean` is derived as `source_edge_mean * residual_dtf_mean` from the debug CSV; add a pixelwise product column later for a stricter calibration plot.
- `gate_*` may include residual-based evidence depending on the config, so gate-to-residual correlation is a sanity check rather than an independent prior test.
- `token_distance_*` and `risk_*` are the more useful independent signals for paper-facing calibration evidence.
