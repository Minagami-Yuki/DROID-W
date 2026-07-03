# Bonn VGGT-Omega Online Results

Date: 2026-07-03

Command:

```bash
bash scripts_eval/run_bonn_omega_online.sh
```

Environment:

- Conda env: `droid-w`
- VGGT-Omega repo: `thirdparty/vggt-omega`
- VGGT-Omega checkpoint: `/data1/czy/Output/DROID-W/vggt_omega_1b_512.pt`
- Config pattern: `configs/Dynamic/Bonn/*_omega_online.yaml`
- Output root: `Outputs/Bonn/*_omega_online`

All 8 sequences finished with `status=0`.

| Scene | KF RMSE | Full RMSE | KF Mean | Full Mean | Tracking FPS | Total Time (s) | Keyframes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_balloon | 0.027889 | 0.027002 | 0.025534 | 0.024653 | 9.91 | 66.52 | 53 |
| bonn_balloon2 | 0.027102 | 0.024507 | 0.024014 | 0.021671 | 10.08 | 70.32 | 60 |
| bonn_crowd | 0.016084 | 0.015322 | 0.014157 | 0.013599 | 10.82 | 134.17 | 108 |
| bonn_crowd2 | 0.023307 | 0.020798 | 0.020001 | 0.018365 | 10.12 | 136.73 | 118 |
| bonn_moving_nonobstructing_box | 0.017102 | 0.016927 | 0.014604 | 0.014632 | 12.80 | 99.74 | 93 |
| bonn_moving_nonobstructing_box2 | 0.030801 | 0.027373 | 0.026689 | 0.023614 | 12.51 | 121.55 | 113 |
| bonn_person_tracking | 0.048816 | 0.049401 | 0.043352 | 0.043557 | 11.34 | 79.83 | 78 |
| bonn_person_tracking2 | 0.032210 | 0.032581 | 0.029079 | 0.029746 | 11.81 | 75.47 | 72 |

Average:

- KF RMSE: `0.027914`
- Full RMSE: `0.026739`

Metric files are under each scene directory:

- `Outputs/Bonn/<scene>_omega_online/traj/metrics_kf_traj.txt`
- `Outputs/Bonn/<scene>_omega_online/traj/metrics_full_traj.txt`
- `Outputs/Bonn/<scene>_omega_online/timer_summary.csv`

Raw run logs are under:

```text
Outputs/Bonn/omega_online_logs/
```
