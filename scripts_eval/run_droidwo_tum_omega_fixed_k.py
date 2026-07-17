#!/usr/bin/env python3
"""Run clean DROID-W-O on TUM Dynamic with fixed Omega intrinsics only."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DROIDWO_ROOT = Path("/home/czy/FanZhu/DROID-W-O-baseline-clean")
DROIDWO_BINARY_ROOT = Path("/home/czy/FanZhu/DROID-W-O")
PYTHON = Path("/home/czy/anaconda3/envs/droid-w/bin/python")
EXPECTED_COMMIT = "c3414af"
DATA_ROOT = Path("/data1/czy/datasets/TUM_RGBD/dynamic")
BUNDLE_ROOT = ROOT / "Outputs/TUM/unknownk_v1/omega_intrinsics"
OUTPUT_ROOT = Path("/data1/czy/Output/DROID-W-O/TUM_omega_fixed_k_baseline_v1")
WORK_ROOT = ROOT / "Outputs/TUM/droidwo_omega_fixed_k_baseline_v1"
CONFIG_ROOT = WORK_ROOT / "configs"
LOG_ROOT = WORK_ROOT / "logs"
PRETRAINED = Path("/home/czy/FanZhu/DROID-Splat/pretrained/droid.pth")
BASE_CONFIG = DROIDWO_ROOT / "configs/Dynamic/TUM_RGBD/tum_dynamic.yaml"
SEQUENCES = (
    "freiburg3_walking_halfsphere", "freiburg3_walking_rpy",
    "freiburg3_walking_static", "freiburg3_walking_xyz",
    "freiburg3_sitting_halfsphere", "freiburg3_sitting_rpy",
    "freiburg3_sitting_static", "freiburg3_sitting_xyz",
)


def scene_name(sequence: str) -> str:
    return f"{sequence}_droidwo_omega_fixed_k_s43"


def intrinsics(sequence: str) -> dict[str, float]:
    data = json.loads((BUNDLE_ROOT / f"{sequence}.json").read_text(encoding="utf-8"))["intrinsics"]
    return {key: float(data[key]) for key in ("fx", "fy", "cx", "cy")}


def validate_source() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=DROIDWO_ROOT, text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Expected clean DROID-W-O {EXPECTED_COMMIT}, got {commit}")
    status = subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=DROIDWO_ROOT, text=True).strip()
    if status:
        raise RuntimeError(f"Clean worktree changed:\n{status}")
    if not (DROIDWO_BINARY_ROOT / "droid_backends.cpython-310-x86_64-linux-gnu.so").is_file():
        raise FileNotFoundError("Original DROID-W-O CUDA backend is missing")


def write_config(sequence: str) -> Path:
    payload = {
        "inherit_from": str(BASE_CONFIG),
        "scene": scene_name(sequence),
        "setup_seed": 43,
        "max_frames": -1,
        "data": {
            "input_folder": str(DATA_ROOT / f"rgbd_dataset_{sequence}"),
            "output": str(OUTPUT_ROOT),
        },
        "cam": intrinsics(sequence),
        "tracking": {
            "pretrained": str(PRETRAINED),
            "uncertainty_params": {"visualize": False},
        },
    }
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    path = CONFIG_ROOT / f"{sequence}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def complete(sequence: str) -> bool:
    traj = OUTPUT_ROOT / scene_name(sequence) / "traj"
    return (traj / "metrics_full_traj.txt").is_file() and (traj / "metrics_kf_traj.txt").is_file()


def run_one(sequence: str, gpu_pool: queue.Queue[str]) -> tuple[str, bool]:
    config = write_config(sequence)
    if complete(sequence):
        return f"SKIP {sequence}: complete", True
    gpu = gpu_pool.get()
    try:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONPATH"] = os.pathsep.join([str(DROIDWO_ROOT), str(DROIDWO_BINARY_ROOT), env.get("PYTHONPATH", "")])
        log_path = LOG_ROOT / f"{sequence}.log"
        print(f"START {sequence} gpu={gpu}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run([str(PYTHON), "run.py", "--config", str(config)], cwd=DROIDWO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        ok = result.returncode == 0 and complete(sequence)
        return f"DONE {sequence} status={result.returncode} complete={ok}", ok
    finally:
        gpu_pool.put(gpu)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--sequences", default=",".join(SEQUENCES))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sequences = [item.strip() for item in args.sequences.split(",") if item.strip()]
    if set(sequences) - set(SEQUENCES):
        raise ValueError("Unknown TUM sequence")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    validate_source()
    for sequence in sequences:
        if not (DATA_ROOT / f"rgbd_dataset_{sequence}/rgb").is_dir():
            raise FileNotFoundError(sequence)
        if not (BUNDLE_ROOT / f"{sequence}.json").is_file():
            raise FileNotFoundError(sequence)
        write_config(sequence)
    if args.dry_run:
        print(f"source={DROIDWO_ROOT} commit={EXPECTED_COMMIT} output={OUTPUT_ROOT} sequences={len(sequences)}")
        return 0
    pool: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        pool.put(gpu)
    success = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_one, sequence, pool) for sequence in sequences]
        for future in concurrent.futures.as_completed(futures):
            message, ok = future.result()
            print(message, flush=True)
            success = success and ok
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
