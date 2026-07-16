#!/usr/bin/env python3
"""Refine a VGGT-Omega focal prior with EPO while freezing pose and depth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def raw_camera(focal: float, source_hw: tuple[int, int], model_hw: tuple[int, int], raw_k: torch.Tensor) -> list[float]:
    source_h, source_w = source_hw
    model_h, model_w = model_hw
    median = raw_k.median(dim=0).values
    return [
        focal * source_w / model_w,
        focal * source_h / model_h,
        float(median[0, 2]),
        float(median[1, 2]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--omega-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epo-root", type=Path, default=Path("/tmp/EPO"))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--max-edges", type=int, default=4096)
    parser.add_argument("--backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.epo_root.is_dir():
        raise FileNotFoundError(f"EPO checkout not found: {args.epo_root}")
    if str(args.epo_root) not in sys.path:
        sys.path.insert(0, str(args.epo_root))
    from epo import EPO

    bundle = torch.load(args.omega_bundle, map_location="cpu", weights_only=False)
    images = bundle["images"]
    depths = bundle["depth"]
    confidence = bundle["confidence"]
    poses = bundle["poses_cw"]
    model_k = bundle["model_intrinsics"]
    raw_k = bundle["raw_intrinsics"]
    names = bundle["frame_paths"]
    source_hw = tuple(bundle["source_hw"])
    model_hw = tuple(bundle["model_hw"])

    ff_data = {}
    for index, path in enumerate(names):
        name = f"cam0/{Path(path).name}"
        ff_data[name] = {
            "image": images[index],
            "depth": depths[index],
            "confidence": confidence[index],
            "pose": poses[index],
            "intrinsic": model_k[index],
        }

    epo = EPO.from_ff(
        ff_data,
        device=args.device,
        detector="canny",
        backend=args.backend,
        fuse_reduction=False,
        matcher_type="sequential",
        sequential_matcher_window=4,
        min_points=100,
        sampling_factor=4,
        max_edges_points=args.max_edges,
        max_num_iterations=args.iterations,
        grad_k=True,
        grad_R=False,
        grad_t=False,
        grad_t_offset=False,
        grad_z=False,
        use_mlp_pose_refinement=False,
        use_depth_confidence=True,
        verbose=False,
        log_granular_time=False,
    )
    _, initial_params = epo.intrinsics.get_camera_parameters("cam0")
    epo(early_stop="none", batch_size=128, drop_last=False)
    _, refined_params = epo.intrinsics.get_camera_parameters("cam0")

    initial_focal = float(initial_params[0].detach().cpu())
    refined_focal = float(refined_params[0].detach().cpu())
    result = {
        "omega_bundle": str(args.omega_bundle),
        "iterations": args.iterations,
        "backend": args.backend,
        "model_initial_focal": initial_focal,
        "model_refined_focal": refined_focal,
        "raw_intrinsics": raw_camera(refined_focal, source_hw, model_hw, raw_k),
        "epo_total_loading_s": float(epo.timings.get("total_loading", 0.0)),
        "epo_total_optimization_s": float(epo.timings.get("total_optimization", 0.0)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
