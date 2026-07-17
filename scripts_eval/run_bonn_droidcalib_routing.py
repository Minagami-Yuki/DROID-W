#!/usr/bin/env python3
"""Run focal-recovery routing controls, then the seed-43 Bonn sweep."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/czy/anaconda3/envs/droid-w/bin/python")
LEGACY_OUTPUT_ROOT = Path("/data1/czy/Output/DROID-omega/Bonn")
OUTPUT_ROOT = ROOT / "Outputs/Bonn/droidcalib_routing_results"
CONFIG_ROOT = ROOT / "Outputs/Bonn/droidcalib_routing_configs"
LOG_ROOT = ROOT / "Outputs/Bonn/droidcalib_routing_logs"
CONTROL_SEQUENCES = (
    "bonn_moving_nonobstructing_box2",
    "bonn_person_tracking2",
    "bonn_crowd2",
)
ALL_SEQUENCES = (
    "bonn_balloon",
    "bonn_balloon2",
    "bonn_crowd",
    "bonn_crowd2",
    "bonn_moving_nonobstructing_box",
    "bonn_moving_nonobstructing_box2",
    "bonn_person_tracking",
    "bonn_person_tracking2",
)
SEEDS = (41, 42, 43)
METHODS = ("stability", "omega_fixed_k", "k_recovery", "confidence_recovery")


def scene_name(sequence: str, method: str, seed: int) -> str:
    return f"{sequence}_droidcalib_routing_{method}_s{seed}"


def override(method: str) -> dict:
    if method == "stability":
        return {}
    if method == "omega_fixed_k":
        return {"tracking": {"focal_calibration": {"enable": False}}}
    if method == "k_recovery":
        return {
            "tracking": {
                "focal_calibration": {
                    "schur_bootstrap": {
                        "initial_k_recovery": {
                            "enable": True,
                            "focal_min_px": 456.0,
                            "focal_max_px": 520.0,
                        }
                    }
                }
            }
        }
    if method == "confidence_recovery":
        return {
            "tracking": {
                "focal_calibration": {
                    "schur_bootstrap": {
                        "omega_confidence_recovery": {
                            "enable": True,
                            "min_samples": 30,
                            "min_mean_median_ratio": 1.05,
                        }
                    }
                }
            }
        }
    raise ValueError(method)


def write_config(sequence: str, method: str, seed: int) -> Path:
    base = ROOT / "configs/Experiments" / f"{sequence}_omega_droidcalib_schur_stability_v2.yaml"
    if not base.is_file():
        raise FileNotFoundError(base)
    config = {
        "inherit_from": str(base),
        "scene": scene_name(sequence, method, seed),
        "setup_seed": seed,
        "data": {"output": str(OUTPUT_ROOT)},
        "tracking": {"uncertainty_params": {"visualize": False}},
    }
    patch = override(method)
    if patch:
        config["tracking"].update(patch["tracking"])
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    path = CONFIG_ROOT / f"{sequence}_{method}_s{seed}.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def completed(sequence: str, method: str, seed: int) -> bool:
    for root in (OUTPUT_ROOT, LEGACY_OUTPUT_ROOT):
        traj = root / scene_name(sequence, method, seed) / "traj"
        if (traj / "metrics_full_traj.txt").is_file() and (traj / "metrics_kf_traj.txt").is_file():
            return True
    return False


def run_one(sequence: str, method: str, seed: int, gpu_pool: queue.Queue[str]) -> tuple[str, bool]:
    config = write_config(sequence, method, seed)
    if completed(sequence, method, seed):
        return f"SKIP {sequence} {method} s{seed}", True
    gpu = gpu_pool.get()
    try:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        log_path = LOG_ROOT / f"{sequence}_{method}_s{seed}.log"
        env = os.environ.copy()
        env["DROIDCALIB_SCHUR_SOURCE"] = "/tmp/droidcalib_schur_src"
        # Some managed CUDA environments expose a usable default device but
        # fail NVML/device remapping once CUDA_VISIBLE_DEVICES is set.  The
        # special pool entry "default" deliberately preserves that state.
        if gpu != "default":
            env["CUDA_VISIBLE_DEVICES"] = gpu
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        command = [
            str(PYTHON),
            "scripts_eval/run_tracking_until_ate.py",
            "--config", str(config),
            "--scene", scene_name(sequence, method, seed),
            "--gpu", gpu,
            "--poll-seconds", "15",
            "--log", str(log_path),
            "--output-root", str(OUTPUT_ROOT),
        ]
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        return f"DONE {sequence} {method} s{seed}: status={result.returncode}", result.returncode == 0 and completed(sequence, method, seed)
    finally:
        gpu_pool.put(gpu)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("controls", "all"), required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    if args.stage == "controls":
        tasks = [(sequence, method, seed) for sequence in CONTROL_SEQUENCES for method in METHODS for seed in SEEDS]
    else:
        tasks = [(sequence, "confidence_recovery", 43) for sequence in ALL_SEQUENCES]
    for task in tasks:
        write_config(*task)
    if args.dry_run:
        print(f"stage={args.stage} tasks={len(tasks)} gpus={','.join(gpus)}")
        return 0
    gpu_pool: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        gpu_pool.put(gpu)
    ok = True
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_one, *task, gpu_pool) for task in tasks]
        for future in as_completed(futures):
            message, success = future.result()
            print(message, flush=True)
            ok = ok and success
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
