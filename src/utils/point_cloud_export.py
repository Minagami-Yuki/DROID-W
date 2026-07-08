import os

import numpy as np
import torch
from lietorch import SE3

from src.utils.Printer import FontColor


def _cfg_value(cfg, key, default):
    if not isinstance(cfg, dict):
        return default
    return cfg.get(key, default)


def _write_binary_ply(path, points, colors):
    vertex = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = points[:, 0].astype(np.float32)
    vertex["y"] = points[:, 1].astype(np.float32)
    vertex["z"] = points[:, 2].astype(np.float32)
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        vertex.tofile(f)


@torch.no_grad()
def save_final_point_cloud(video, save_dir, cfg, printer=None):
    pc_cfg = cfg.get("point_cloud", {}) or {}
    if not pc_cfg.get("save_final", True):
        return None

    num_frames = int(video.counter.value)
    if num_frames <= 0:
        if printer is not None:
            printer.print("Skipped final point cloud: no keyframes available.", FontColor.INFO)
        return None

    filename = pc_cfg.get("filename", "final_point_cloud.ply")
    path = filename if os.path.isabs(filename) else os.path.join(save_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    stride = max(1, int(_cfg_value(pc_cfg, "stride", 4)))
    max_points_per_frame = int(_cfg_value(pc_cfg, "max_points_per_frame", 12000))
    min_depth = float(_cfg_value(pc_cfg, "min_depth", 0.05))
    max_depth = _cfg_value(pc_cfg, "max_depth", 20.0)
    max_depth = float(max_depth) if max_depth is not None else None
    depth_source = str(_cfg_value(pc_cfg, "depth_source", "droid")).lower()
    use_valid_mask = bool(_cfg_value(pc_cfg, "use_valid_depth_mask", True))

    with video.get_lock():
        poses = video.poses[:num_frames].detach().clone()
        intrinsics = video.intrinsics[:num_frames].detach().clone()
        images = video.images[:num_frames].detach().clone()
        if depth_source == "mono":
            disps = video.mono_disps_up[:num_frames].detach().clone()
            valid_masks = None
        else:
            disps = video.disps_up[:num_frames].detach().clone()
            valid_masks = (
                video.valid_depth_mask[:num_frames].detach().clone()
                if use_valid_mask
                else None
            )

    c2w = SE3(poses).inv().matrix().detach().cpu().numpy()
    intrinsics = intrinsics.cpu().numpy()
    disps = disps.cpu().numpy()
    images = images.cpu().numpy()
    if valid_masks is not None:
        valid_masks = valid_masks.cpu().numpy()

    all_points = []
    all_colors = []

    for frame_idx in range(num_frames):
        disp = disps[frame_idx]
        h, w = disp.shape
        ys = np.arange(0, h, stride, dtype=np.float32)
        xs = np.arange(0, w, stride, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        xi = grid_x.astype(np.int64)
        yi = grid_y.astype(np.int64)

        sampled_disp = disp[yi, xi]
        valid = np.isfinite(sampled_disp) & (sampled_disp > 0.0)
        depth = np.zeros_like(sampled_disp, dtype=np.float32)
        depth[valid] = 1.0 / sampled_disp[valid]
        valid &= np.isfinite(depth) & (depth > min_depth)
        if max_depth is not None:
            valid &= depth < max_depth

        if valid_masks is not None and valid_masks[frame_idx].any():
            valid &= valid_masks[frame_idx][yi, xi]

        if not valid.any():
            continue

        fx, fy, cx, cy = intrinsics[frame_idx]
        z = depth[valid]
        u = grid_x[valid]
        v = grid_y[valid]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        cam_points = np.stack([x, y, z, np.ones_like(z)], axis=1)
        world_points = (cam_points @ c2w[frame_idx].T)[:, :3]

        rgb = images[frame_idx].transpose(1, 2, 0)
        colors = np.clip(rgb[yi[valid], xi[valid]] * 255.0, 0, 255).astype(np.uint8)

        if max_points_per_frame > 0 and world_points.shape[0] > max_points_per_frame:
            keep = np.linspace(
                0, world_points.shape[0] - 1, max_points_per_frame, dtype=np.int64
            )
            world_points = world_points[keep]
            colors = colors[keep]

        all_points.append(world_points.astype(np.float32))
        all_colors.append(colors)

    if not all_points:
        if printer is not None:
            printer.print("Skipped final point cloud: no valid depth samples.", FontColor.INFO)
        return None

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    _write_binary_ply(path, points, colors)

    if printer is not None:
        printer.print(
            f"Saved final point cloud: {path} ({points.shape[0]} points from {num_frames} keyframes)",
            FontColor.INFO,
        )
    return path
