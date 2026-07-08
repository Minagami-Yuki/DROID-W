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


def _source_filename(filename, source, multi_source):
    if not multi_source:
        return filename
    root, ext = os.path.splitext(filename)
    if not ext:
        ext = ".ply"
    return f"{root}_{source}{ext}"


def _parse_depth_sources(pc_cfg):
    if "depth_sources" in pc_cfg and pc_cfg["depth_sources"] is not None:
        raw_sources = pc_cfg["depth_sources"]
    else:
        raw_sources = pc_cfg.get("depth_source", "droid")

    if isinstance(raw_sources, str):
        raw_sources = [raw_sources]

    sources = []
    for source in raw_sources:
        source = str(source).lower()
        if source == "both":
            sources.extend(["droid", "mono"])
        elif source in ("droid", "mono"):
            sources.append(source)
        else:
            raise ValueError(
                f"point_cloud depth source must be 'droid', 'mono', or 'both', got {source}"
            )

    deduped = []
    for source in sources:
        if source not in deduped:
            deduped.append(source)
    return deduped


def _select_depth_arrays(source, droid_disps, mono_disps, valid_masks, use_valid_mask):
    if source == "mono":
        return mono_disps, None
    masks = valid_masks if use_valid_mask else None
    return droid_disps, masks


def _export_point_cloud_arrays(
    path,
    poses_c2w,
    intrinsics,
    images,
    disps,
    valid_masks=None,
    stride=4,
    max_points_per_frame=12000,
    min_depth=0.05,
    max_depth=20.0,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    all_points = []
    all_colors = []
    num_frames = int(disps.shape[0])

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
        world_points = (cam_points @ poses_c2w[frame_idx].T)[:, :3]

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
        return None, 0

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    _write_binary_ply(path, points, colors)
    return path, int(points.shape[0])


def _export_sources(
    save_dir,
    filename,
    pc_cfg,
    poses_c2w,
    intrinsics,
    images,
    droid_disps,
    mono_disps,
    valid_masks=None,
):
    stride = max(1, int(_cfg_value(pc_cfg, "stride", 4)))
    max_points_per_frame = int(_cfg_value(pc_cfg, "max_points_per_frame", 12000))
    min_depth = float(_cfg_value(pc_cfg, "min_depth", 0.05))
    max_depth = _cfg_value(pc_cfg, "max_depth", 20.0)
    max_depth = float(max_depth) if max_depth is not None else None
    use_valid_mask = bool(_cfg_value(pc_cfg, "use_valid_depth_mask", True))
    sources = _parse_depth_sources(pc_cfg)
    multi_source = len(sources) > 1

    outputs = []
    for source in sources:
        source_filename = _source_filename(filename, source, multi_source)
        path = source_filename if os.path.isabs(source_filename) else os.path.join(save_dir, source_filename)
        disps, masks = _select_depth_arrays(
            source, droid_disps, mono_disps, valid_masks, use_valid_mask
        )
        output_path, num_points = _export_point_cloud_arrays(
            path,
            poses_c2w,
            intrinsics,
            images,
            disps,
            valid_masks=masks,
            stride=stride,
            max_points_per_frame=max_points_per_frame,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        outputs.append(
            {
                "source": source,
                "path": output_path,
                "num_points": num_points,
                "num_frames": int(disps.shape[0]),
            }
        )
    return outputs


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

    with video.get_lock():
        poses = video.poses[:num_frames].detach().clone()
        intrinsics = video.intrinsics[:num_frames].detach().clone()
        images = video.images[:num_frames].detach().clone()
        droid_disps = video.disps_up[:num_frames].detach().clone()
        mono_disps = video.mono_disps_up[:num_frames].detach().clone()
        valid_masks = video.valid_depth_mask[:num_frames].detach().clone()

    c2w = SE3(poses).inv().matrix().detach().cpu().numpy()
    intrinsics = intrinsics.cpu().numpy()
    images = images.cpu().numpy()
    droid_disps = droid_disps.cpu().numpy()
    mono_disps = mono_disps.cpu().numpy()
    valid_masks = valid_masks.cpu().numpy()

    outputs = _export_sources(
        save_dir,
        filename,
        pc_cfg,
        c2w,
        intrinsics,
        images,
        droid_disps,
        mono_disps,
        valid_masks=valid_masks,
    )

    if printer is not None:
        for output in outputs:
            if output["path"] is None:
                printer.print(
                    f"Skipped final {output['source']} point cloud: no valid depth samples.",
                    FontColor.INFO,
                )
            else:
                printer.print(
                    f"Saved final {output['source']} point cloud: {output['path']} "
                    f"({output['num_points']} points from {output['num_frames']} keyframes)",
                    FontColor.INFO,
                )
    valid_outputs = [output["path"] for output in outputs if output["path"] is not None]
    if len(valid_outputs) == 1:
        return valid_outputs[0]
    return valid_outputs


def save_point_cloud_from_video_npz(npz_path, output_dir, cfg):
    pc_cfg = cfg.get("point_cloud", {}) or {}
    filename = pc_cfg.get("filename", "final_point_cloud.ply")

    data = np.load(npz_path)
    required = ["poses", "intrinsics", "images", "droid_disps_up", "mono_disps"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{npz_path} is missing required arrays: {missing}")

    return _export_sources(
        output_dir,
        filename,
        pc_cfg,
        data["poses"],
        data["intrinsics"],
        data["images"],
        data["droid_disps_up"],
        data["mono_disps"],
        valid_masks=None,
    )
