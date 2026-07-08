# YouTube Output Root and Final Point Cloud Run

Date: 2026-07-05

Code/config changes:
- All explicit config outputs now point under `/data1/czy/Output/DROID-omega`.
- `point_cloud.save_final` is enabled by default in `configs/droid_w.yaml`.
- Final keyframe point cloud is saved as `final_point_cloud.ply` after tracking/evaluation.

Sanity checks:

```bash
python -m py_compile src/utils/point_cloud_export.py src/slam.py
conda run -n droid-w python -c "from src import config; from src.utils.datasets import get_dataset; c=config.load_config('configs/Dynamic/YouTube/tokyo_walking1.yaml'); d=get_dataset(c); print(d.input_folder); print(len(d))"
```

Run commands:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/YouTube/tokyo_walking1.yaml
conda run -n droid-w python run.py --config configs/Dynamic/YouTube/tokyo_walking2.yaml
conda run -n droid-w python run.py --config configs/Dynamic/YouTube/tokyo_walking3.yaml
```

Results:

| Sequence | Frames | Output dir | Full system time | FPS | Final point cloud | Full trajectory |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| tokyo_walking1 | 3001 | `/data1/czy/Output/DROID-omega/YouTube/tokyo_walking1` | 376.861s | 7.96 | 844,163 points, 13M | 3001 poses |
| tokyo_walking2 | 1321 | `/data1/czy/Output/DROID-omega/YouTube/tokyo_walking2` | 181.688s | 7.27 | 473,068 points, 6.8M | 1321 poses |
| tokyo_walking3 | 2400 | `/data1/czy/Output/DROID-omega/YouTube/tokyo_walking3` | 349.315s | 6.87 | 1,063,596 points, 16M | 2400 poses |

Notes:
- YouTube sequences are `RGB_NoPose`, so there is no GT ATE/RMSE metric.
- All three runs exited with status 0 and produced `video.npz`, `traj/est_poses_full.txt`, and `final_point_cloud.ply`.
