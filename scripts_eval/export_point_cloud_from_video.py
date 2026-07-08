import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.point_cloud_export import save_point_cloud_from_video_npz


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export DROID and mono point clouds from an existing DROID-W video.npz."
    )
    parser.add_argument("--video", required=True, help="Path to video.npz.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported PLY files.")
    parser.add_argument("--filename", default="final_point_cloud.ply")
    parser.add_argument(
        "--depth-source",
        default="both",
        choices=["droid", "mono", "both"],
        help="Depth source to export.",
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-points-per-frame", type=int, default=12000)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=20.0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = {
        "point_cloud": {
            "save_final": True,
            "filename": args.filename,
            "depth_source": args.depth_source,
            "stride": args.stride,
            "max_points_per_frame": args.max_points_per_frame,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            # video.npz does not store valid_depth_mask, so this script exports
            # comparable raw point clouds from the two saved disparity sources.
            "use_valid_depth_mask": False,
        }
    }
    outputs = save_point_cloud_from_video_npz(args.video, args.output_dir, cfg)
    for output in outputs:
        status = "skipped" if output["path"] is None else output["path"]
        print(
            f"{output['source']}: {status} "
            f"points={output['num_points']} frames={output['num_frames']}"
        )


if __name__ == "__main__":
    main()
