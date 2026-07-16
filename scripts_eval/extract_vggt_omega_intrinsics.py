#!/usr/bin/env python3
"""Extract a shared pinhole prior from a short VGGT-Omega bootstrap window."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
OMEGA_REPO = ROOT / "thirdparty" / "vggt-omega"
DEFAULT_CHECKPOINT = Path("/data1/czy/Output/DROID-omega/vggt_omega_1b_512.pt")


def target_shape(height: int, width: int, resolution: int, patch_size: int) -> tuple[int, int]:
    """Match the existing balanced Omega preprocessing without changing aspect ratio."""
    aspect = height / max(width, 1)
    token_count = (resolution // patch_size) ** 2
    width_patches = max(1, int(round((token_count / aspect) ** 0.5)))
    height_patches = max(1, int(round(token_count / width_patches)))
    return height_patches * patch_size, width_patches * patch_size


def select_paths(paths: list[Path], count: int, stride: int) -> list[Path]:
    selected = paths[::stride]
    if len(selected) < count:
        raise ValueError(
            f"Only {len(selected)} frames after stride={stride}; need {count}."
        )
    return selected[:count]


def load_images(paths: list[Path], height: int, width: int) -> torch.Tensor:
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        images.append(tensor)
    batch = torch.stack(images)
    return F.interpolate(batch, size=(height, width), mode="bicubic", align_corners=False).clamp_(0.0, 1.0)


def raw_intrinsics(model_intrinsics: torch.Tensor, source_hw: tuple[int, int], model_hw: tuple[int, int]) -> torch.Tensor:
    source_h, source_w = source_hw
    model_h, model_w = model_hw
    raw = model_intrinsics.clone()
    raw[..., 0, 0] *= source_w / model_w
    raw[..., 0, 2] *= source_w / model_w
    raw[..., 1, 1] *= source_h / model_h
    raw[..., 1, 2] *= source_h / model_h
    return raw


def summary(intrinsics: torch.Tensor) -> dict[str, float | list[float]]:
    params = torch.stack(
        [intrinsics[:, 0, 0], intrinsics[:, 1, 1], intrinsics[:, 0, 2], intrinsics[:, 1, 2]],
        dim=1,
    )
    median = params.median(dim=0).values
    mad = (params - median).abs().median(dim=0).values
    return {
        "fx": float(median[0]),
        "fy": float(median[1]),
        "cx": float(median[2]),
        "cy": float(median[3]),
        "median_absolute_deviation": [float(value) for value in mad],
        "per_frame": [[float(value) for value in row] for row in params],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--frame-glob", default="frame-*.color.png")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--reuse-bundle", action="store_true", help="write the JSON summary from an existing output bundle")
    args = parser.parse_args()

    if args.reuse_bundle:
        bundle = torch.load(args.output, map_location="cpu", weights_only=False)
        report = {
            "input_dir": bundle["input_dir"],
            "frame_paths": bundle["frame_paths"],
            "source_hw": list(bundle["source_hw"]),
            "model_hw": list(bundle["model_hw"]),
            "intrinsics": summary(bundle["raw_intrinsics"]),
        }
        report_path = args.output.with_suffix(".json")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega extraction requires CUDA.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    paths = sorted(args.input_dir.glob(args.frame_glob))
    if not paths:
        raise FileNotFoundError(f"No frames matching {args.frame_glob} in {args.input_dir}")
    paths = select_paths(paths, args.count, args.stride)

    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(paths[0])
    source_hw = tuple(first.shape[:2])
    model_hw = target_shape(*source_hw, args.resolution, args.patch_size)

    if str(OMEGA_REPO) not in sys.path:
        sys.path.insert(0, str(OMEGA_REPO))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.pose_enc import encoding_to_camera

    images = load_images(paths, *model_hw).cuda()
    model = VGGTOmega().eval().cuda()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    with torch.inference_mode():
        predictions = model(images)
        poses_cw, model_k = encoding_to_camera(predictions["pose_enc"], model_hw)

    model_k = model_k[0].float().cpu()
    raw_k = raw_intrinsics(model_k, source_hw, model_hw)
    depth = predictions["depth"][0, :, ..., 0].float().cpu()
    confidence = predictions["depth_conf"][0].float().cpu()
    poses_cw = poses_cw[0].float().cpu()
    images_cpu = images.float().cpu()

    result = {
        "version": 1,
        "input_dir": str(args.input_dir),
        "checkpoint": str(args.checkpoint),
        "source_hw": source_hw,
        "model_hw": model_hw,
        "frame_paths": [str(path) for path in paths],
        "images": images_cpu,
        "depth": depth,
        "confidence": confidence,
        "poses_cw": poses_cw,
        "model_intrinsics": model_k,
        "raw_intrinsics": raw_k,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)

    report = {
        "input_dir": str(args.input_dir),
        "frame_paths": [str(path) for path in paths],
        "source_hw": list(source_hw),
        "model_hw": list(model_hw),
        "intrinsics": summary(raw_k),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
