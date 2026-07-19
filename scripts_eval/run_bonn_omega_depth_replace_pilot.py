#!/usr/bin/env python3
"""Test pure online Omega-depth replacement on two Bonn dynamic sequences."""

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
BASE = ROOT / "configs/Dynamic/Bonn/bonn_dynamic.yaml"
PRETRAINED = Path("/home/czy/FanZhu/DROID-Splat/pretrained/droid.pth")
OMEGA_CHECKPOINT = Path("/data1/czy/Output/DROID-omega/vggt_omega_1b_512.pt")
OUTPUT_ROOT = Path("/data1/czy/Output/DROID-omega/Bonn_omega_depth_replace_pilot_v1")
WORK_ROOT = ROOT / "Outputs/Bonn/omega_depth_replace_pilot_v1"
REPORT = ROOT / "experiments/method/bonn_omega_depth_replace_pilot_v1_results.md"
SEED = 43
SEQUENCES = ("crowd2", "moving_nonobstructing_box2")
METRIC3D_BASELINE = {
    "crowd2": 0.018004461,
    "moving_nonobstructing_box2": 0.023466373,
}


def scene(sequence: str) -> str:
    return f"bonn_{sequence}_omega_depth_replace_gt_k_s{SEED}"


def config_path(sequence: str) -> Path:
    return WORK_ROOT / "configs" / f"{sequence}.json"


def write_config(sequence: str) -> Path:
    payload = {
        "inherit_from": str(BASE),
        "scene": scene(sequence),
        "setup_seed": SEED,
        "max_frames": -1,
        "data": {
            "input_folder": str(DATA_ROOT / f"rgbd_bonn_{sequence}"),
            "output": str(OUTPUT_ROOT),
        },
        "omega_prior": {
            "enable": True,
            "source": "model",
            "cache": {"write": False},
            "depth": {
                "enable": True,
                "mode": "replace",
                "fallback_to_mono": False,
            },
            "uncertainty": {"enable": False},
            "model": {
                "repo_path": "thirdparty/vggt-omega",
                "checkpoint": str(OMEGA_CHECKPOINT),
                "image_resolution": 512,
                "preprocess_mode": "balanced",
                "patch_tokens": {"enable": False},
            },
        },
        "edge_dtf_prior": {"enable": False},
        "tracking": {
            "pretrained": str(PRETRAINED),
            "uncertainty_params": {"visualize": False},
            "focal_calibration": {"enable": False},
        },
    }
    path = config_path(sequence)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def trajectory(sequence: str) -> Path:
    return OUTPUT_ROOT / scene(sequence) / "traj"


def complete(sequence: str) -> bool:
    directory = trajectory(sequence)
    return (directory / "metrics_full_traj.txt").is_file() and (directory / "metrics_kf_traj.txt").is_file()


def run_one(sequence: str, pool: queue.Queue[str]) -> tuple[str, bool]:
    config = write_config(sequence)
    if complete(sequence):
        return sequence, True
    gpu = pool.get()
    try:
        logs = WORK_ROOT / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        with (logs / f"{sequence}.log").open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                [str(PYTHON), "run.py", "--config", str(config)],
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return sequence, result.returncode == 0 and complete(sequence)
    finally:
        pool.put(gpu)


def read_rmse(path: Path) -> float | None:
    if not path.is_file():
        return None
    match = re.search(r"'rmse':\s*([0-9.eE+-]+)", path.read_text(encoding="utf-8"))
    return float(match.group(1)) if match else None


def report(sequences: list[str]) -> bool:
    lines = [
        "# Bonn Pure Omega-Depth Replacement Pilot V1",
        "",
        "Protocol: seed 43; native/GT Bonn K; online VGGT-Omega depth replaces Metric3D depth at every candidate keyframe (`mode: replace`, `fallback_to_mono: false`). Omega uncertainty, tokens, Edge-DTF, focal BA, confidence routing and v25 reliability are disabled. The Metric3D estimator is not instantiated in this configuration.",
        "",
        "| Sequence | Metric3D baseline Full (m) | Omega depth replace Full (m) | Delta (m) | Ratio | Omega depth replace KF (m) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    values = []
    for sequence in sequences:
        full = read_rmse(trajectory(sequence) / "metrics_full_traj.txt")
        kf = read_rmse(trajectory(sequence) / "metrics_kf_traj.txt")
        baseline = METRIC3D_BASELINE[sequence]
        if full is None:
            lines.append(f"| {sequence} | {baseline:.9f} | missing | - | - | - |")
            continue
        values.append(full)
        lines.append(
            f"| {sequence} | {baseline:.9f} | {full:.9f} | {full-baseline:+.9f} | "
            f"{full / baseline:.2%} | {kf:.9f} |"
        )
    if len(values) == len(sequences):
        lines += ["", f"Macro mean replacement Full ATE: `{np.mean(values):.9f} m`."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return len(values) == len(sequences)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("run", "report", "all"), default="all")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--sequences", default=",".join(SEQUENCES))
    args = parser.parse_args()
    sequences = [value.strip() for value in args.sequences.split(",") if value.strip()]
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not sequences or set(sequences) - set(SEQUENCES):
        raise ValueError("Unknown Bonn sequence")
    if args.phase != "report" and not gpus:
        raise ValueError("At least one GPU is required")
    for path in (BASE, PRETRAINED, OMEGA_CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(path)
    for sequence in sequences:
        if not (DATA_ROOT / f"rgbd_bonn_{sequence}" / "rgb").is_dir():
            raise FileNotFoundError(DATA_ROOT / f"rgbd_bonn_{sequence}" / "rgb")
    if args.phase in ("run", "all"):
        pool: queue.Queue[str] = queue.Queue()
        for gpu in gpus:
            pool.put(gpu)
        success = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [executor.submit(run_one, sequence, pool) for sequence in sequences]
            for future in concurrent.futures.as_completed(futures):
                sequence, ok = future.result()
                print(f"{'DONE' if ok else 'FAILED'} {sequence}", flush=True)
                success = success and ok
        if not success:
            return 1
    return 0 if args.phase == "run" else int(not report(sequences))


if __name__ == "__main__":
    raise SystemExit(main())
