# TUM Dynamic Unknown-K Four-Way Protocol

Date: 2026-07-17. ATE is RMSE in metres. The GT-K DROID-W reference was not rerun; its existing `*_droidw_eval` output is read only for comparison.

All three evaluated modes use `cam.fx/fy/cx/cy` from a per-sequence VGGT-Omega bundle generated from 12 early RGB frames (stride 5). The BA modes update focal length only, with fixed aspect ratio and fixed principal point.

## Full ATE

| Sequence | DROID-W GT-K | Omega fixed-K | Delta | Omega + stability BA | Delta | Omega + confidence BA | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `walking_halfsphere` | 0.0157119 | 0.0167444 | +6.57% | 0.0165979 | +5.64% | 0.0170200 | +8.33% |
| `walking_rpy` | 0.0303817 | 0.0363391 | +19.61% | 0.0361166 | +18.88% | 0.0371730 | +22.35% |
| `walking_static` | 0.0047341 | 0.0052686 | +11.29% | 0.0052679 | +11.28% | 0.0052658 | +11.23% |
| `walking_xyz` | 0.0121554 | 0.0142856 | +17.53% | 0.0137008 | +12.71% | 0.0137913 | +13.46% |
| `sitting_halfsphere` | 0.0138364 | 0.0164906 | +19.18% | 0.0145978 | +5.50% | 0.0149174 | +7.81% |
| `sitting_rpy` | 0.0211892 | 0.0237837 | +12.24% | 0.0223028 | +5.26% | 0.0219413 | +3.55% |
| `sitting_static` | 0.0047721 | 0.0048697 | +2.04% | 0.0048650 | +1.95% | 0.0048680 | +2.01% |
| `sitting_xyz` | 0.0080418 | 0.0102861 | +27.91% | 0.0102818 | +27.85% | 0.0097605 | +21.37% |

## Keyframe ATE

| Sequence | DROID-W GT-K | Omega fixed-K | Omega + stability BA | Omega + confidence BA |
|---|---:|---:|---:|---:|
| `walking_halfsphere` | 0.0160323 | 0.0173985 | 0.0173205 | 0.0175920 |
| `walking_rpy` | 0.0406552 | 0.0460045 | 0.0457563 | 0.0469485 |
| `walking_static` | 0.0051510 | 0.0056372 | 0.0056359 | 0.0056324 |
| `walking_xyz` | 0.0122502 | 0.0146483 | 0.0139081 | 0.0138899 |
| `sitting_halfsphere` | 0.0144630 | 0.0164094 | 0.0147546 | 0.0155487 |
| `sitting_rpy` | 0.0249264 | 0.0290320 | 0.0270453 | 0.0264112 |
| `sitting_static` | 0.0050884 | 0.0052952 | 0.0052936 | 0.0053001 |
| `sitting_xyz` | 0.0079784 | 0.0101396 | 0.0101357 | 0.0096440 |

## Focal BA Diagnostics

| Sequence | Stability accepted/attempted | Confidence accepted/attempted |
|---|---:|---:|
| `walking_halfsphere` | 59/443 | 373/433 |
| `walking_rpy` | 83/426 | 383/429 |
| `walking_static` | 0/293 | 0/293 |
| `walking_xyz` | 9/360 | 331/360 |
| `sitting_halfsphere` | 37/390 | 245/386 |
| `sitting_rpy` | 20/403 | 182/403 |
| `sitting_static` | 0/276 | 0/276 |
| `sitting_xyz` | 0/319 | 2/319 |

## Mean Full ATE

| DROID-W GT-K | Omega fixed-K | Omega + stability BA | Omega + confidence BA |
|---:|---:|---:|---:|
| 0.0138528 | 0.0160085 | 0.0154663 | 0.0155922 |
