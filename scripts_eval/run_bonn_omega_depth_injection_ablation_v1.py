#!/usr/bin/env python3
"""Run strict Bonn Omega-depth injection ablations for the main method table."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import re
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/czy/anaconda3/envs/droid-w/bin/python")
DATA_ROOT = Path("/data1/czy/datasets/Bonn")
CACHE_ROOT = Path("/data1/czy/Output/DROID-omega/cache/Bonn")
K_MANIFEST = ROOT / "experiments/bonn_omega_k_manifest_v1.json"
BASE = ROOT / "configs/Dynamic/Bonn/bonn_dynamic.yaml"
PRETRAINED = Path("/home/czy/FanZhu/DROID-Splat/pretrained/droid.pth")
OUTPUT_ROOT = Path("/data1/czy/Output/DROID-omega/Bonn_omega_depth_ablation_v1")
WORK_ROOT = ROOT / "Outputs/Bonn/omega_depth_ablation_v1"
REPORT = ROOT / "experiments/method/bonn_omega_depth_injection_ablation_v1_results.md"
SEED = 43
SEQUENCES = (
    "balloon",
    "balloon2",
    "crowd",
    "crowd2",
    "moving_nonobstructing_box",
    "moving_nonobstructing_box2",
    "person_tracking",
    "person_tracking2",
)
METHODS = ("omega_depth_only", "omega_depth_k_fixed")


def scene(sequence: str, method: str) -> str:
    return f"bonn_{sequence}_{method}_s{SEED}"


def omega_k(sequence: str) -> dict[str, float]:
    values = json.loads(K_MANIFEST.read_text(encoding="utf-8"))["intrinsics"][sequence]
    return {key: float(values[key]) for key in ("fx", "fy", "cx", "cy")}


def config_path(sequence: str, method: str) -> Path:
    return WORK_ROOT / "configs" / f"{sequence}_{method}.json"


def write_config(sequence: str, method: str) -> Path:
    if method not in METHODS:
        raise ValueError(method)
    payload = {
        "inherit_from": str(BASE),
        "scene": scene(sequence, method),
        "setup_seed": SEED,
        "max_frames": -1,
        "data": {
            "input_folder": str(DATA_ROOT / f"rgbd_bonn_{sequence}"),
            "output": str(OUTPUT_ROOT),
        },
        "omega_prior": {
            "enable": True,
            "source": "cache",
            "cache_dir": str(CACHE_ROOT / f"bonn_{sequence}"),
            "missing_policy": "warn",
            "depth": {
                "enable": True,
                "mode": "blend",
                "blend_alpha": 0.10,
                "align_to_mono": "scale",
                "align_trim": 0.05,
                "align_min_pixels": 256,
                "fallback_to_mono": True,
            },
            "uncertainty": {"enable": False},
            "model": {"patch_tokens": {"enable": False}},
        },
        "edge_dtf_prior": {"enable": False},
        "tracking": {
            "pretrained": str(PRETRAINED),
            "uncertainty_params": {"visualize": False},
            "focal_calibration": {"enable": False},
        },
    }
    if method == "omega_depth_k_fixed":
        payload["cam"] = omega_k(sequence)
    path = config_path(sequence, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def trajectory(sequence: str, method: str) -> Path:
    return OUTPUT_ROOT / scene(sequence, method) / "traj"


def complete(sequence: str, method: str) -> bool:
    directory = trajectory(sequence, method)
    return (directory / "metrics_full_traj.txt").is_file() and (directory / "metrics_kf_traj.txt").is_file()


def run_one(sequence: str, method: str, pool: queue.Queue[str]) -> tuple[str, str, bool]:
    config = write_config(sequence, method)
    if complete(sequence, method):
        return sequence, method, True
    gpu = pool.get()
    try:
        logs = WORK_ROOT / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        with (logs / f"{sequence}_{method}.log").open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                [str(PYTHON), "run.py", "--config", str(config)],
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return sequence, method, result.returncode == 0 and complete(sequence, method)
    finally:
        pool.put(gpu)


def run_method(method: str, sequences: list[str], gpus: list[str]) -> bool:
    pool: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        pool.put(gpu)
    success = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_one, sequence, method, pool) for sequence in sequences]
        for future in concurrent.futures.as_completed(futures):
            sequence, name, ok = future.result()
            print(f"{'DONE' if ok else 'FAILED'} {name} {sequence}", flush=True)
            success = success and ok
    return success


def read_rmse(path: Path) -> float | None:
    if not path.is_file():
        return None
    match = re.search(r"'rmse':\s*([0-9.eE+-]+)", path.read_text(encoding="utf-8"))
    return float(match.group(1)) if match else None


def metrics(sequence: str, method: str) -> tuple[float | None, float | None]:
    directory = trajectory(sequence, method)
    return read_rmse(directory / "metrics_full_traj.txt"), read_rmse(directory / "metrics_kf_traj.txt")


def report(sequences: list[str]) -> bool:
    lines = [
        "# Bonn-8 Omega Depth Injection Ablation V1",
        "",
        "Protocol: eight complete Bonn dynamic sequences, seed 43, native Bonn evaluation, and the original DROID-W Metric3D depth prior. Both methods enable only cached Omega depth injection using scale-aligned blend (`alpha=0.10`, trim `0.05`); Omega uncertainty, tokens, Edge-DTF, focal BA, confidence routing and all v25 reliability are disabled.",
        "",
        "`Omega depth only` retains the Bonn native/GT camera K. `Omega depth + Omega K fixed` replaces it with the canonical 12-frame, stride-5 Omega K from `bonn_omega_k_manifest_v1.json` and performs no camera optimization.",
        "",
        "| Sequence | Omega depth only Full (m) | Omega depth + Omega K fixed Full (m) | Depth only KF (m) | Depth + K fixed KF (m) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    depth_values, depth_k_values = [], []
    for sequence in sequences:
        depth, depth_kf = metrics(sequence, "omega_depth_only")
        depth_k, depth_k_kf = metrics(sequence, "omega_depth_k_fixed")
        if depth is None or depth_k is None:
            lines.append(
                f"| {sequence} | {'missing' if depth is None else f'{depth:.9f}'} | "
                f"{'missing' if depth_k is None else f'{depth_k:.9f}'} | - | - |"
            )
            continue
        depth_values.append(depth)
        depth_k_values.append(depth_k)
        lines.append(
            f"| {sequence} | {depth:.9f} | {depth_k:.9f} | {depth_kf:.9f} | {depth_k_kf:.9f} |"
        )
    if len(depth_values) == len(sequences):
        lines += [
            "",
            f"Macro mean Full ATE: Omega depth only `{np.mean(depth_values):.9f} m`; "
            f"Omega depth + Omega K fixed `{np.mean(depth_k_values):.9f} m`.",
        ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return len(depth_values) == len(sequences)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=(*METHODS, "all", "report"), default="all")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--sequences", default=",".join(SEQUENCES))
    args = parser.parse_args()
    sequences = [value.strip() for value in args.sequences.split(",") if value.strip()]
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not sequences or set(sequences) - set(SEQUENCES):
        raise ValueError("Unknown Bonn sequence")
    if args.phase != "report" and not gpus:
        raise ValueError("At least one GPU is required")
    for sequence in sequences:
        for path in (
            DATA_ROOT / f"rgbd_bonn_{sequence}" / "rgb",
            CACHE_ROOT / f"bonn_{sequence}" / "depths",
        ):
            if not path.is_dir():
                raise FileNotFoundError(path)
    for path in (K_MANIFEST, BASE, PRETRAINED):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.phase in ("omega_depth_only", "all"):
        if not run_method("omega_depth_only", sequences, gpus):
            return 1
    if args.phase in ("omega_depth_k_fixed", "all"):
        if not run_method("omega_depth_k_fixed", sequences, gpus):
            return 1
    return 0 if args.phase in METHODS else int(not report(sequences))


if __name__ == "__main__":
    raise SystemExit(main())
