#!/usr/bin/env python3
"""Run clean DROID-W-O on Bonn with only Omega-predicted fixed intrinsics."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DROIDWO_ROOT = Path("/home/czy/FanZhu/DROID-W-O-baseline-clean")
DROIDWO_BINARY_ROOT = Path("/home/czy/FanZhu/DROID-W-O")
EXPECTED_COMMIT = "c3414af"
PYTHON = Path("/home/czy/anaconda3/envs/droid-w/bin/python")
DATA_ROOT = Path("/data1/czy/datasets/Bonn")
INTRINSICS_MANIFEST = ROOT / "experiments/bonn_omega_k_manifest_v1.json"
OUTPUT_ROOT = Path("/data1/czy/Output/DROID-W-O/Bonn_omega_fixed_k_baseline_v2")
WORK_ROOT = ROOT / "Outputs/DROID-W-O/bonn_omega_fixed_k_baseline_v2"
CONFIG_ROOT = WORK_ROOT / "configs"
LOG_ROOT = WORK_ROOT / "logs"
PRETRAINED = Path("/home/czy/FanZhu/DROID-Splat/pretrained/droid.pth")
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


def scene_name(sequence: str) -> str:
    return f"bonn_{sequence}_droidwo_omega_fixed_k_s43"


def omega_intrinsics(sequence: str) -> dict[str, float]:
    data = json.loads(INTRINSICS_MANIFEST.read_text(encoding="utf-8"))["intrinsics"][sequence]
    return {key: float(data[key]) for key in ("fx", "fy", "cx", "cy")}


def write_config(sequence: str, max_frames: int) -> Path:
    scene = scene_name(sequence)
    payload = {
        "inherit_from": str(DROIDWO_ROOT / f"configs/Dynamic/Bonn/bonn_{sequence}.yaml"),
        "scene": scene,
        "setup_seed": 43,
        "max_frames": max_frames,
        "data": {
            "input_folder": str(DATA_ROOT / f"rgbd_bonn_{sequence}"),
            "output": str(OUTPUT_ROOT),
        },
        "cam": omega_intrinsics(sequence),
        # The clean repository expects a relative checkpoint that is not tracked.
        # This absolute path changes only checkpoint location, not model weights.
        "tracking": {
            "pretrained": str(PRETRAINED),
            # Diagnostic image export is not part of tracking or ATE.
            "uncertainty_params": {"visualize": False},
        },
    }
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    path = CONFIG_ROOT / f"bonn_{sequence}.yaml"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def validate_source() -> None:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=DROIDWO_ROOT, text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Expected DROID-W-O {EXPECTED_COMMIT}, found {commit}")
    status = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"], cwd=DROIDWO_ROOT, text=True
    ).strip()
    if status:
        raise RuntimeError(f"Clean DROID-W-O worktree has tracked changes:\n{status}")
    if not (DROIDWO_BINARY_ROOT / "droid_backends.cpython-310-x86_64-linux-gnu.so").is_file():
        raise FileNotFoundError("Original DROID-W-O backend binary is missing")
    if not PRETRAINED.is_file():
        raise FileNotFoundError(PRETRAINED)


def validate_config(sequence: str, path: Path) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw["cam"] != omega_intrinsics(sequence):
        raise RuntimeError(f"{sequence}: generated K differs from Omega bundle")
    forbidden = {"omega_prior", "focal_calibration", "calibration_observation"}
    serialized = json.dumps(raw)
    if any(field in serialized for field in forbidden):
        raise RuntimeError(f"{sequence}: generated config enables a non-baseline feature")


def complete(sequence: str) -> bool:
    trajectory = OUTPUT_ROOT / scene_name(sequence) / "traj"
    return (trajectory / "metrics_full_traj.txt").is_file() and (
        trajectory / "metrics_kf_traj.txt"
    ).is_file()


def read_rmse(path: Path) -> float:
    match = re.search(r"'rmse': ([0-9.eE+-]+)", path.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"No RMSE in {path}")
    return float(match.group(1))


def run_one(sequence: str, gpu_pool: queue.Queue[str], max_frames: int) -> tuple[str, bool]:
    config = write_config(sequence, max_frames)
    validate_config(sequence, config)
    if max_frames < 0 and complete(sequence):
        return f"SKIP {sequence}: complete", True

    gpu = gpu_pool.get()
    try:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        log_path = LOG_ROOT / f"bonn_{sequence}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONPATH"] = os.pathsep.join(
            [str(DROIDWO_ROOT), str(DROIDWO_BINARY_ROOT), env.get("PYTHONPATH", "")]
        )
        print(f"START {sequence} gpu={gpu}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [str(PYTHON), "run.py", "--config", str(config)],
                cwd=DROIDWO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        ok = result.returncode == 0 and (max_frames >= 0 or complete(sequence))
        return f"DONE {sequence} status={result.returncode} complete={ok}", ok
    finally:
        gpu_pool.put(gpu)


def print_results(sequences: list[str]) -> None:
    print("\n| Sequence | Full ATE (m) | KF ATE (m) |")
    print("|---|---:|---:|")
    for sequence in sequences:
        trajectory = OUTPUT_ROOT / scene_name(sequence) / "traj"
        if complete(sequence):
            full = read_rmse(trajectory / "metrics_full_traj.txt")
            keyframe = read_rmse(trajectory / "metrics_kf_traj.txt")
            print(f"| bonn_{sequence} | {full:.9f} | {keyframe:.9f} |")
        else:
            print(f"| bonn_{sequence} | NA | NA |")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--sequences", default=",".join(SEQUENCES))
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sequences = [item.strip() for item in args.sequences.split(",") if item.strip()]
    unknown = sorted(set(sequences) - set(SEQUENCES))
    if unknown:
        raise ValueError(f"Unknown Bonn sequences: {unknown}")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")

    validate_source()
    if not INTRINSICS_MANIFEST.is_file():
        raise FileNotFoundError(INTRINSICS_MANIFEST)
    for sequence in sequences:
        dataset = DATA_ROOT / f"rgbd_bonn_{sequence}/rgb"
        if not dataset.is_dir():
            raise FileNotFoundError(dataset)
        config = write_config(sequence, args.max_frames)
        validate_config(sequence, config)

    if args.dry_run:
        print(f"source={DROIDWO_ROOT} commit={EXPECTED_COMMIT}")
        print(f"output={OUTPUT_ROOT}")
        print(f"sequences={len(sequences)} gpus={','.join(gpus)}")
        return 0

    gpu_pool: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        gpu_pool.put(gpu)
    ok = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(run_one, sequence, gpu_pool, args.max_frames)
            for sequence in sequences
        ]
        for future in concurrent.futures.as_completed(futures):
            message, task_ok = future.result()
            print(message, flush=True)
            ok = ok and task_ok
    if args.max_frames < 0:
        print_results(sequences)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
