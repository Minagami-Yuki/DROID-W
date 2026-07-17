#!/usr/bin/env python3
"""Extend the TUM soft full-pinhole calibration pilot to all Bonn sequences."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/czy/anaconda3/envs/droid-w/bin/python")
DATA_ROOT = Path("/data1/czy/datasets/Bonn")
CACHE_ROOT = Path("/data1/czy/Output/DROID-omega/cache/Bonn")
K_MANIFEST = ROOT / "experiments/bonn_omega_k_manifest_v1.json"
BASE_CONFIG = ROOT / "configs/Dynamic/Bonn/bonn_dynamic.yaml"
OUTPUT_ROOT = Path("/data1/czy/Output/DROID-omega/Bonn_full_pinhole_softcalib_v1")
WORK_ROOT = ROOT / "Outputs/Bonn/full_pinhole_softcalib_v1"
CONFIG_ROOT = WORK_ROOT / "configs"
LOG_ROOT = WORK_ROOT / "logs"
SEQUENCES = (
    "balloon", "balloon2", "crowd", "crowd2", "moving_nonobstructing_box",
    "moving_nonobstructing_box2", "person_tracking", "person_tracking2",
)


def scene_name(sequence: str) -> str:
    return f"bonn_{sequence}_omega_full_pinhole_softcalib_s43"


def intrinsics(sequence: str) -> dict[str, float]:
    data = json.loads(K_MANIFEST.read_text(encoding="utf-8"))["intrinsics"][sequence]
    return {key: float(data[key]) for key in ("fx", "fy", "cx", "cy")}


def focal_config() -> dict:
    # This is the clean TUM soft-calibration preset, carried over unchanged.
    return {
        "enable": True,
        "solver": "droidcalib_schur",
        "intrinsics_mode": "full_pinhole",
        "prior_weight": 5.0,
        "schur_diagnostics": True,
        "schur_min_hessian": 25.0,
        "full_focal_min_hessian": 1.0,
        "intrinsics_prior_weight": [0.01, 0.01, 0.005, 0.005],
        "warmup_keyframes": 20,
        "every_n_ba": 8,
        "min_edges": 12,
        "max_log_step": 0.002,
        "max_log_deviation": 0.15,
        "principal_point": {
            "enable": True,
            "max_step": 0.10,
            "max_deviation": 1.50,
            "prior_weight": 0.01,
            "min_hessian": 0.001,
        },
        "schur_bootstrap": {
            "enable": True,
            "start_keyframes": 30,
            "end_keyframes": 80,
            "every_n_ba": 1,
            "min_hessian": 25.0,
            "max_log_step": 0.004,
            "max_log_deviation": 0.10,
            "trajectory_stability": {
                "enable": True,
                "max_translation_update": 0.020,
                "max_rotation_update_rad": 0.0125,
                "min_consecutive_ba": 3,
                "reset_after_accept": True,
            },
            "omega_confidence_recovery": {"enable": False},
        },
        "calibration_observation": {
            "enable": True,
            "mode": "soft",
            "require_omega_uncertainty": True,
            "soft_omega_certain": 0.85,
            "soft_omega_uncertain": 1.00,
            "soft_min_pixel_scale": 0.60,
            "soft_span_start": 1.0,
            "soft_span_full": 4.0,
            "soft_min_edge_scale": 0.75,
        },
    }


def write_config(sequence: str) -> Path:
    scene = scene_name(sequence)
    payload = {
        "inherit_from": str(BASE_CONFIG),
        "scene": scene,
        "setup_seed": 43,
        "max_frames": -1,
        "data": {
            "input_folder": str(DATA_ROOT / f"rgbd_bonn_{sequence}"),
            "output": str(OUTPUT_ROOT),
        },
        "cam": intrinsics(sequence),
        # Maps are loaded exclusively for the auxiliary soft K calibration
        # objective; normal DROID tracking weights remain unmodified.
        "omega_prior": {
            "enable": True,
            "source": "cache",
            "cache_dir": str(CACHE_ROOT / f"bonn_{sequence}"),
            "uncertainty": {"enable": True, "apply_to": "none"},
        },
        "tracking": {
            "focal_calibration": focal_config(),
            "uncertainty_params": {"visualize": False},
        },
    }
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    path = CONFIG_ROOT / f"bonn_{sequence}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
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
        env["DROIDCALIB_SCHUR_SOURCE"] = "/tmp/droidcalib_schur_src"
        command = [
            str(PYTHON), "scripts_eval/run_tracking_until_ate.py",
            "--config", str(config), "--scene", scene_name(sequence), "--gpu", gpu,
            "--poll-seconds", "15", "--max-retries", "2", "--retry-seconds", "20",
            "--output-root", str(OUTPUT_ROOT), "--log", str(LOG_ROOT / f"bonn_{sequence}.log"),
        ]
        print(f"START {sequence} gpu={gpu}", flush=True)
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
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
        raise ValueError("Unknown Bonn sequence")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    if not K_MANIFEST.is_file() or not BASE_CONFIG.is_file():
        raise FileNotFoundError("Missing manifest or base config")
    for sequence in sequences:
        if not (DATA_ROOT / f"rgbd_bonn_{sequence}/rgb").is_dir():
            raise FileNotFoundError(sequence)
        if not (CACHE_ROOT / f"bonn_{sequence}/uncertainties").is_dir():
            raise FileNotFoundError(f"Omega uncertainty cache missing for {sequence}")
        write_config(sequence)
    if args.dry_run:
        print(f"output={OUTPUT_ROOT} sequences={len(sequences)}")
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
