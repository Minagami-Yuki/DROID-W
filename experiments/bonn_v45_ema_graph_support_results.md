# V45 EMA Graph-Support Low-Parallax Calibration

Date: 2026-07-12

V45 calibrates the Omega patch-token uncertainty strength using online graph-level
support. Per-edge low-parallax evidence combines bidirectional DROID frame distance
with Edge-DTF residual evidence. The mean evidence is temporally stabilized with an
EMA and mapped through a smooth support gate before changing the BA uncertainty
strength. All new behavior is default-off.

Main settings:

```yaml
low_parallax_adaptive:
  enable: True
  risk_gain: 0.0
  strength_boost: 0.35
  graph_support:
    enable: True
    signal: mean_alpha_ema
    ema_decay: 0.80
    min_mean_alpha: 0.006
    max_mean_alpha: 0.016
    mode: smoothstep
```

Initial full-sequence validation (ATE RMSE in meters):

| Sequence | DROID-W Full | v25 Full | v45 Full | v45/DROID-W | Tracking FPS | Full FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_crowd2 | 0.018004 | 0.018261 current run | 0.017394 | 96.61% | 11.59 | 6.18 |
| bonn_person_tracking | 0.034278 | 0.032727 | 0.033478 | 97.67% | 14.89 | 7.28 |

The remaining Bonn sequences are pending. Evaluation configs disable only the
expensive final uncertainty figure export; tracking, priors, BA, trajectory filling,
and timer accounting are unchanged.
