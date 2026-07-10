# Omega Patch-Token Uncertainty Calibration

Date: 2026-07-09

Goal: test whether the Omega patch-token uncertainty/gate is calibrated with Edge-DTF dynamic evidence at the per-edge level. Higher token/risk/gate should correspond to higher Edge-DTF residual; higher gate should also correspond to lower final BA scale.

Input CSVs:

- `/data1/czy/Output/DROID-omega/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats/debug/patch_token_uncertainty_stats.csv`
- `/data1/czy/Output/DROID-omega/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats/debug/patch_token_uncertainty_stats.csv`

## Key Correlations

| Scene | Score | Target | Rows | Pearson | Spearman | Top20/Bottom20 | Monotonic bins |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats | gate_max | edge_residual_mean | 3018 | 0.5270 | 0.4980 | 2.1520 | 4/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats | gate_max | scale_min | 3018 | -0.8721 | -0.8226 | 0.9991 | 4/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats | gate_mean | edge_residual_mean | 3018 | 0.6016 | 0.8143 | 3.0531 | 4/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats | gate_mean | scale_mean | 3018 | 0.1582 | -0.8867 | 1.0031 | 3/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats | risk_max | edge_residual_mean | 3018 | 0.5495 | 0.5060 | 2.3386 | 4/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats | risk_mean | edge_residual_mean | 3018 | 0.6613 | 0.6888 | 2.6778 | 4/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats | token_distance_max | edge_residual_mean | 3018 | -0.0164 | -0.0303 | 0.9460 | 2/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats | token_distance_mean | edge_residual_mean | 3018 | 0.3017 | 0.2811 | 1.5028 | 3/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats | gate_max | edge_residual_mean | 2222 | 0.5778 | 0.5509 | 2.9986 | 4/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats | gate_max | scale_min | 2222 | -0.8925 | -0.8520 | 0.9987 | 4/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats | gate_mean | edge_residual_mean | 2222 | 0.4526 | 0.5966 | 3.1551 | 3/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats | gate_mean | scale_mean | 2222 | -0.9012 | -0.8357 | 0.9998 | 4/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats | risk_max | edge_residual_mean | 2222 | 0.6004 | 0.5737 | 3.0765 | 4/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats | risk_mean | edge_residual_mean | 2222 | 0.4853 | 0.5168 | 2.2900 | 4/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats | token_distance_max | edge_residual_mean | 2222 | -0.2009 | -0.1815 | 0.7186 | 1/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats | token_distance_mean | edge_residual_mean | 2222 | 0.0604 | 0.0162 | 1.0921 | 1/4 |

## Quantile Bins

### bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats: gate_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.7835 | 0.0000-0.8733 | 0.0159 |
| 2 | 604 | 0.9030 | 0.8735-0.9288 | 0.0254 |
| 3 | 603 | 0.9503 | 0.9288-0.9684 | 0.0299 |
| 4 | 604 | 0.9812 | 0.9685-0.9905 | 0.0338 |
| 5 | 604 | 0.9958 | 0.9905-1.0000 | 0.0342 |

### bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats: gate_max -> scale_min

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.7835 | 0.0000-0.8733 | 0.9971 |
| 2 | 604 | 0.9030 | 0.8735-0.9288 | 0.9966 |
| 3 | 603 | 0.9503 | 0.9288-0.9684 | 0.9964 |
| 4 | 604 | 0.9812 | 0.9685-0.9905 | 0.9963 |
| 5 | 604 | 0.9958 | 0.9905-1.0000 | 0.9961 |

### bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats: gate_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.2551 | 0.0000-0.2721 | 0.0149 |
| 2 | 604 | 0.3302 | 0.2721-0.3687 | 0.0203 |
| 3 | 603 | 0.3741 | 0.3687-0.3790 | 0.0247 |
| 4 | 604 | 0.3837 | 0.3790-0.3893 | 0.0341 |
| 5 | 604 | 0.3972 | 0.3893-0.4215 | 0.0453 |

### bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats: gate_mean -> scale_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.2551 | 0.0000-0.2721 | 0.9963 |
| 2 | 604 | 0.3302 | 0.2721-0.3687 | 0.9995 |
| 3 | 603 | 0.3741 | 0.3687-0.3790 | 0.9994 |
| 4 | 604 | 0.3837 | 0.3790-0.3893 | 0.9994 |
| 5 | 604 | 0.3972 | 0.3893-0.4215 | 0.9994 |

### bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats: risk_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.7078 | 0.0000-0.8098 | 0.0148 |
| 2 | 604 | 0.8478 | 0.8098-0.8779 | 0.0264 |
| 3 | 603 | 0.9028 | 0.8779-0.9271 | 0.0303 |
| 4 | 604 | 0.9484 | 0.9272-0.9683 | 0.0330 |
| 5 | 604 | 0.9861 | 0.9683-1.0000 | 0.0346 |

### bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats: risk_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.0913 | 0.0000-0.1035 | 0.0155 |
| 2 | 604 | 0.1201 | 0.1036-0.1331 | 0.0222 |
| 3 | 603 | 0.1387 | 0.1331-0.1438 | 0.0271 |
| 4 | 604 | 0.1490 | 0.1438-0.1538 | 0.0328 |
| 5 | 604 | 0.1619 | 0.1538-0.1950 | 0.0416 |

### bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats: token_distance_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 1.2776 | 0.0000-1.3215 | 0.0282 |
| 2 | 604 | 1.3459 | 1.3218-1.3665 | 0.0278 |
| 3 | 603 | 1.3859 | 1.3665-1.4074 | 0.0278 |
| 4 | 604 | 1.4321 | 1.4075-1.4591 | 0.0286 |
| 5 | 604 | 1.5100 | 1.4593-1.6741 | 0.0267 |

### bonn_crowd2_omega_patch_token_uncertainty_v12_pixel_stats: token_distance_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.2530 | 0.0000-0.2706 | 0.0209 |
| 2 | 604 | 0.2776 | 0.2706-0.2847 | 0.0271 |
| 3 | 603 | 0.2912 | 0.2847-0.2974 | 0.0282 |
| 4 | 604 | 0.3042 | 0.2974-0.3109 | 0.0317 |
| 5 | 604 | 0.3241 | 0.3109-0.3878 | 0.0314 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats: gate_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.6336 | 0.4026-0.7560 | 0.0085 |
| 2 | 444 | 0.7939 | 0.7561-0.8283 | 0.0174 |
| 3 | 445 | 0.8545 | 0.8283-0.8758 | 0.0215 |
| 4 | 444 | 0.8997 | 0.8759-0.9304 | 0.0234 |
| 5 | 445 | 0.9609 | 0.9305-0.9992 | 0.0256 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats: gate_max -> scale_min

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.6336 | 0.4026-0.7560 | 0.9977 |
| 2 | 444 | 0.7939 | 0.7561-0.8283 | 0.9970 |
| 3 | 445 | 0.8545 | 0.8283-0.8758 | 0.9967 |
| 4 | 444 | 0.8997 | 0.8759-0.9304 | 0.9966 |
| 5 | 445 | 0.9609 | 0.9305-0.9992 | 0.9964 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats: gate_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.2399 | 0.2209-0.2480 | 0.0092 |
| 2 | 444 | 0.2536 | 0.2480-0.2588 | 0.0170 |
| 3 | 445 | 0.2646 | 0.2588-0.2713 | 0.0210 |
| 4 | 444 | 0.3172 | 0.2713-0.3665 | 0.0201 |
| 5 | 445 | 0.3747 | 0.3665-0.4007 | 0.0291 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats: gate_mean -> scale_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.2399 | 0.2209-0.2480 | 0.9997 |
| 2 | 444 | 0.2536 | 0.2480-0.2588 | 0.9996 |
| 3 | 445 | 0.2646 | 0.2588-0.2713 | 0.9996 |
| 4 | 444 | 0.3172 | 0.2713-0.3665 | 0.9995 |
| 5 | 445 | 0.3747 | 0.3665-0.4007 | 0.9994 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats: risk_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.5579 | 0.3721-0.6753 | 0.0082 |
| 2 | 444 | 0.7357 | 0.6755-0.7777 | 0.0171 |
| 3 | 445 | 0.8060 | 0.7778-0.8329 | 0.0221 |
| 4 | 444 | 0.8563 | 0.8330-0.8808 | 0.0237 |
| 5 | 445 | 0.9269 | 0.8810-0.9960 | 0.0252 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats: risk_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.0796 | 0.0600-0.0888 | 0.0115 |
| 2 | 444 | 0.0934 | 0.0888-0.0971 | 0.0158 |
| 3 | 445 | 0.1011 | 0.0972-0.1055 | 0.0192 |
| 4 | 444 | 0.1163 | 0.1056-0.1357 | 0.0236 |
| 5 | 445 | 0.1442 | 0.1357-0.1635 | 0.0263 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats: token_distance_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 1.3277 | 1.1883-1.3679 | 0.0225 |
| 2 | 444 | 1.3922 | 1.3681-1.4140 | 0.0207 |
| 3 | 445 | 1.4354 | 1.4140-1.4568 | 0.0184 |
| 4 | 444 | 1.4797 | 1.4568-1.5058 | 0.0187 |
| 5 | 445 | 1.5527 | 1.5059-1.6985 | 0.0162 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v12_pixel_stats: token_distance_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.2413 | 0.2009-0.2664 | 0.0157 |
| 2 | 444 | 0.2787 | 0.2665-0.2873 | 0.0227 |
| 3 | 445 | 0.2943 | 0.2873-0.3004 | 0.0208 |
| 4 | 444 | 0.3073 | 0.3004-0.3152 | 0.0202 |
| 5 | 445 | 0.3287 | 0.3153-0.3735 | 0.0171 |

## Notes

- `edge_residual_mean` uses the exact pixelwise `edge_residual_pixel_mean` column when available, and falls back to `source_edge_mean * residual_dtf_mean` for old CSVs.
- `gate_*` may include residual-based evidence depending on the config, so gate-to-residual correlation is a sanity check rather than an independent prior test.
- `token_distance_*` and `risk_*` are the more useful independent signals for paper-facing calibration evidence.
