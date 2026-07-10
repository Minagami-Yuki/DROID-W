# Omega Patch-Token Uncertainty Calibration

Date: 2026-07-09

Goal: test whether the Omega patch-token uncertainty/gate is calibrated with Edge-DTF dynamic evidence at the per-edge level. Higher token/risk/gate should correspond to higher Edge-DTF residual; higher gate should also correspond to lower final BA scale.

Input CSVs:

- `/data1/czy/Output/DROID-omega/Bonn/bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats/debug/patch_token_uncertainty_stats.csv`
- `/data1/czy/Output/DROID-omega/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats/debug/patch_token_uncertainty_stats.csv`

## Key Correlations

| Scene | Score | Target | Rows | Pearson | Spearman | Top20/Bottom20 | Monotonic bins |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats | gate_max | edge_residual_mean | 3016 | 0.3595 | 0.2758 | 1.2169 | 4/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats | gate_max | scale_min | 3016 | -0.8681 | -0.8236 | 0.9991 | 4/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats | gate_mean | edge_residual_mean | 3016 | 0.4505 | 0.3952 | 1.2820 | 4/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats | gate_mean | scale_mean | 3016 | -0.7167 | -0.7321 | 0.9999 | 4/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats | risk_max | edge_residual_mean | 3016 | 0.3506 | 0.2869 | 1.2211 | 4/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats | risk_mean | edge_residual_mean | 3016 | 0.1646 | 0.1443 | 1.1064 | 4/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats | token_distance_max | edge_residual_mean | 3016 | 0.0146 | -0.0163 | 1.0002 | 2/4 |
| bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats | token_distance_mean | edge_residual_mean | 3016 | -0.0062 | -0.0241 | 1.0050 | 2/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats | gate_max | edge_residual_mean | 2222 | 0.0135 | 0.0086 | 1.0080 | 1/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats | gate_max | scale_min | 2222 | -0.8922 | -0.8519 | 0.9987 | 4/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats | gate_mean | edge_residual_mean | 2222 | 0.1496 | 0.0985 | 1.1171 | 4/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats | gate_mean | scale_mean | 2222 | -0.3509 | -0.3536 | 0.9999 | 4/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats | risk_max | edge_residual_mean | 2222 | 0.0494 | 0.0521 | 1.0610 | 2/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats | risk_mean | edge_residual_mean | 2222 | 0.1241 | 0.1361 | 1.1078 | 4/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats | token_distance_max | edge_residual_mean | 2222 | 0.1877 | 0.2073 | 1.1572 | 4/4 |
| bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats | token_distance_mean | edge_residual_mean | 2222 | 0.1498 | 0.1576 | 1.1394 | 3/4 |

## Quantile Bins

### bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats: gate_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.7850 | 0.4173-0.8733 | 0.0354 |
| 2 | 603 | 0.9031 | 0.8735-0.9288 | 0.0394 |
| 3 | 603 | 0.9504 | 0.9288-0.9684 | 0.0411 |
| 4 | 603 | 0.9812 | 0.9685-0.9905 | 0.0417 |
| 5 | 604 | 0.9958 | 0.9905-1.0000 | 0.0430 |

### bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats: gate_max -> scale_min

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.7850 | 0.4173-0.8733 | 0.9971 |
| 2 | 603 | 0.9031 | 0.8735-0.9288 | 0.9966 |
| 3 | 603 | 0.9504 | 0.9288-0.9684 | 0.9964 |
| 4 | 603 | 0.9812 | 0.9685-0.9905 | 0.9962 |
| 5 | 604 | 0.9958 | 0.9905-1.0000 | 0.9961 |

### bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats: gate_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.3592 | 0.3517-0.3641 | 0.0359 |
| 2 | 603 | 0.3684 | 0.3641-0.3726 | 0.0380 |
| 3 | 603 | 0.3761 | 0.3726-0.3799 | 0.0394 |
| 4 | 603 | 0.3843 | 0.3799-0.3896 | 0.0414 |
| 5 | 604 | 0.3974 | 0.3896-0.4338 | 0.0460 |

### bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats: gate_mean -> scale_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.3592 | 0.3517-0.3641 | 0.9995 |
| 2 | 603 | 0.3684 | 0.3641-0.3726 | 0.9995 |
| 3 | 603 | 0.3761 | 0.3726-0.3799 | 0.9994 |
| 4 | 603 | 0.3843 | 0.3799-0.3896 | 0.9994 |
| 5 | 604 | 0.3974 | 0.3896-0.4338 | 0.9994 |

### bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats: risk_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.7089 | 0.4030-0.8092 | 0.0353 |
| 2 | 603 | 0.8481 | 0.8093-0.8775 | 0.0393 |
| 3 | 603 | 0.9031 | 0.8777-0.9273 | 0.0406 |
| 4 | 603 | 0.9487 | 0.9274-0.9687 | 0.0423 |
| 5 | 604 | 0.9863 | 0.9687-1.0000 | 0.0431 |

### bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats: risk_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.1223 | 0.0817-0.1310 | 0.0388 |
| 2 | 603 | 0.1352 | 0.1311-0.1388 | 0.0388 |
| 3 | 603 | 0.1425 | 0.1388-0.1462 | 0.0397 |
| 4 | 603 | 0.1507 | 0.1462-0.1552 | 0.0404 |
| 5 | 604 | 0.1627 | 0.1552-0.1949 | 0.0429 |

### bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats: token_distance_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 1.2801 | 0.9978-1.3209 | 0.0401 |
| 2 | 603 | 1.3453 | 1.3209-1.3664 | 0.0398 |
| 3 | 603 | 1.3859 | 1.3665-1.4073 | 0.0401 |
| 4 | 603 | 1.4323 | 1.4073-1.4590 | 0.0405 |
| 5 | 604 | 1.5096 | 1.4590-1.6741 | 0.0402 |

### bonn_crowd2_omega_patch_token_uncertainty_v7_debug_stats: token_distance_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 603 | 0.2539 | 0.1779-0.2706 | 0.0397 |
| 2 | 603 | 0.2776 | 0.2706-0.2845 | 0.0404 |
| 3 | 603 | 0.2912 | 0.2845-0.2974 | 0.0401 |
| 4 | 603 | 0.3041 | 0.2974-0.3109 | 0.0405 |
| 5 | 604 | 0.3241 | 0.3109-0.3863 | 0.0399 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats: gate_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.6339 | 0.4026-0.7561 | 0.0325 |
| 2 | 444 | 0.7940 | 0.7561-0.8283 | 0.0324 |
| 3 | 445 | 0.8545 | 0.8283-0.8758 | 0.0320 |
| 4 | 444 | 0.8997 | 0.8759-0.9304 | 0.0342 |
| 5 | 445 | 0.9609 | 0.9305-0.9992 | 0.0327 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats: gate_max -> scale_min

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.6339 | 0.4026-0.7561 | 0.9977 |
| 2 | 444 | 0.7940 | 0.7561-0.8283 | 0.9970 |
| 3 | 445 | 0.8545 | 0.8283-0.8758 | 0.9967 |
| 4 | 444 | 0.8997 | 0.8759-0.9304 | 0.9966 |
| 5 | 445 | 0.9609 | 0.9305-0.9992 | 0.9964 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats: gate_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.3534 | 0.3506-0.3563 | 0.0313 |
| 2 | 444 | 0.3589 | 0.3563-0.3613 | 0.0320 |
| 3 | 445 | 0.3638 | 0.3613-0.3661 | 0.0325 |
| 4 | 444 | 0.3686 | 0.3661-0.3714 | 0.0330 |
| 5 | 445 | 0.3770 | 0.3714-0.4007 | 0.0350 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats: gate_mean -> scale_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.3534 | 0.3506-0.3563 | 0.9995 |
| 2 | 444 | 0.3589 | 0.3563-0.3613 | 0.9995 |
| 3 | 445 | 0.3638 | 0.3613-0.3661 | 0.9995 |
| 4 | 444 | 0.3686 | 0.3661-0.3714 | 0.9994 |
| 5 | 445 | 0.3770 | 0.3714-0.4007 | 0.9994 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats: risk_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.5579 | 0.3721-0.6752 | 0.0319 |
| 2 | 444 | 0.7357 | 0.6753-0.7777 | 0.0318 |
| 3 | 445 | 0.8060 | 0.7778-0.8329 | 0.0332 |
| 4 | 444 | 0.8563 | 0.8331-0.8808 | 0.0331 |
| 5 | 445 | 0.9269 | 0.8810-0.9960 | 0.0338 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats: risk_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.1133 | 0.0940-0.1255 | 0.0314 |
| 2 | 444 | 0.1309 | 0.1255-0.1350 | 0.0317 |
| 3 | 445 | 0.1377 | 0.1350-0.1403 | 0.0321 |
| 4 | 444 | 0.1427 | 0.1403-0.1452 | 0.0338 |
| 5 | 445 | 0.1509 | 0.1453-0.1666 | 0.0348 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats: token_distance_max -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 1.3276 | 1.1883-1.3679 | 0.0306 |
| 2 | 444 | 1.3922 | 1.3680-1.4140 | 0.0308 |
| 3 | 445 | 1.4355 | 1.4140-1.4568 | 0.0328 |
| 4 | 444 | 1.4797 | 1.4568-1.5057 | 0.0343 |
| 5 | 445 | 1.5527 | 1.5059-1.6985 | 0.0354 |

### bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_debug_stats: token_distance_mean -> edge_residual_mean

| Bin | Count | Score mean | Score range | Target mean |
| ---: | ---: | ---: | --- | ---: |
| 1 | 444 | 0.2413 | 0.2009-0.2664 | 0.0315 |
| 2 | 444 | 0.2787 | 0.2665-0.2873 | 0.0321 |
| 3 | 445 | 0.2943 | 0.2873-0.3004 | 0.0317 |
| 4 | 444 | 0.3073 | 0.3004-0.3152 | 0.0327 |
| 5 | 445 | 0.3287 | 0.3153-0.3735 | 0.0358 |

## Notes

- `edge_residual_mean` is derived as `source_edge_mean * residual_dtf_mean` from the debug CSV.
- `gate_*` includes residual-based evidence in the current v7 debug configs, so gate-to-residual correlation is a sanity check rather than an independent prior test.
- `token_distance_*` and `risk_*` are the more useful independent signals for paper-facing calibration evidence.
