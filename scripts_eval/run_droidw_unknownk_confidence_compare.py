#!/usr/bin/env python3
"""Compare clean DROID-W-O + fixed Omega K with confidence-main v1 on DROID-W."""

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
import yaml
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D


ROOT = Path(__file__).resolve().parents[1]
DROIDWO_ROOT = Path("/home/czy/FanZhu/DROID-W-O-baseline-clean")
DROIDWO_BINARY_ROOT = Path("/home/czy/FanZhu/DROID-W-O")
EXPECTED_DROIDWO_COMMIT = "c3414af"
PYTHON = Path("/home/czy/anaconda3/envs/droid-w/bin/python")
DATA_ROOT = Path("/data1/czy/datasets/DROID-W")
CACHE_ROOT = Path("/data1/czy/Output/DROID-omega/cache/DROID-W")
OMEGA_BUNDLE_ROOT = ROOT / "Outputs/DROID-W/unknownk_confidence_v1/omega_intrinsics"
WORK_ROOT = ROOT / "Outputs/DROID-W/unknownk_confidence_v1"
CONFIG_ROOT = WORK_ROOT / "configs"
LOG_ROOT = WORK_ROOT / "logs"
BASELINE_OUTPUT_ROOT = Path("/data1/czy/Output/DROID-W-O/DROID-W_omega_fixed_k_baseline_v1")
CONFIDENCE_OUTPUT_ROOT = Path("/data1/czy/Output/DROID-omega/DROID-W_unknownk_confidence_v1")
PRETRAINED = Path("/home/czy/FanZhu/DROID-Splat/pretrained/droid.pth")
PROFILE = ROOT / "configs/Experiments/confidence_main_v1.yaml"
REPORT = ROOT / "experiments/droidw_unknownk_confidence_v1_results.md"
SEQUENCES = tuple(f"downtown{index}" for index in range(1, 8))


def baseline_scene(sequence: str) -> str:
    return f"{sequence}_droidwo_omega_fixed_k_s43"


def confidence_scene(sequence: str) -> str:
    return f"{sequence}_omega_confidence_main_v1_s43"


def bundle_path(sequence: str) -> Path:
    return OMEGA_BUNDLE_ROOT / f"{sequence}.pt"


def omega_intrinsics(sequence: str) -> dict[str, float]:
    payload = json.loads(bundle_path(sequence).with_suffix(".json").read_text(encoding="utf-8"))
    values = payload["intrinsics"]
    return {key: float(values[key]) for key in ("fx", "fy", "cx", "cy")}


def profile() -> dict:
    return yaml.safe_load(PROFILE.read_text(encoding="utf-8"))["tracking"]["focal_calibration"]


def write_config(sequence: str, method: str) -> Path:
    if method == "baseline":
        payload = {
            "inherit_from": str(DROIDWO_ROOT / "configs/Dynamic/DROIDW/droidw.yaml"),
            "scene": baseline_scene(sequence),
            "setup_seed": 43,
            "max_frames": -1,
            "data": {"input_folder": str(DATA_ROOT / sequence), "output": str(BASELINE_OUTPUT_ROOT)},
            "cam": omega_intrinsics(sequence),
            "tracking": {"pretrained": str(PRETRAINED), "uncertainty_params": {"visualize": False}},
        }
    elif method == "confidence":
        payload = {
            "inherit_from": str(ROOT / "configs/Dynamic/DROIDW/droidw.yaml"),
            "scene": confidence_scene(sequence),
            "setup_seed": 43,
            "max_frames": -1,
            "data": {"input_folder": str(DATA_ROOT / sequence), "output": str(CONFIDENCE_OUTPUT_ROOT)},
            "cam": omega_intrinsics(sequence),
            "omega_prior": {
                "enable": True,
                "source": "cache",
                "cache_dir": str(CACHE_ROOT / sequence),
                "missing_policy": "error",
                "depth": {"enable": False},
                "uncertainty": {"enable": False},
            },
            "tracking": {
                "pretrained": str(PRETRAINED),
                "uncertainty_params": {"visualize": False},
                "focal_calibration": profile(),
            },
        }
    else:
        raise ValueError(method)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    path = CONFIG_ROOT / f"{sequence}_{method}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def validate_source() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=DROIDWO_ROOT, text=True).strip()
    if commit != EXPECTED_DROIDWO_COMMIT:
        raise RuntimeError(f"Expected clean DROID-W-O {EXPECTED_DROIDWO_COMMIT}, found {commit}")
    status = subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=DROIDWO_ROOT, text=True).strip()
    if status:
        raise RuntimeError(f"Clean DROID-W-O worktree has tracked changes:\n{status}")
    if not (DROIDWO_BINARY_ROOT / "droid_backends.cpython-310-x86_64-linux-gnu.so").is_file():
        raise FileNotFoundError("Original DROID-W-O CUDA backend is missing")
    if not PRETRAINED.is_file():
        raise FileNotFoundError(PRETRAINED)


def extract_intrinsics(sequence: str, gpu: str) -> None:
    output = bundle_path(sequence)
    if output.is_file() and output.with_suffix(".json").is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    subprocess.run(
        [str(PYTHON), "scripts_eval/extract_vggt_omega_intrinsics.py", "--input-dir", str(DATA_ROOT / sequence / "images_anonymized"), "--frame-glob", "*.jpg", "--count", "12", "--stride", "5", "--output", str(output)],
        cwd=ROOT, env=env, check=True,
    )


def traj_dir(sequence: str, method: str) -> Path:
    if method == "baseline":
        return BASELINE_OUTPUT_ROOT / baseline_scene(sequence) / "traj"
    return CONFIDENCE_OUTPUT_ROOT / confidence_scene(sequence) / "traj"


def complete(sequence: str, method: str) -> bool:
    return (traj_dir(sequence, method) / "est_poses_full.txt").is_file() and (
        (traj_dir(sequence, method) / "metrics_full_traj.txt").is_file()
    )


def run_one(sequence: str, method: str, gpu_pool: queue.Queue[str]) -> tuple[str, bool]:
    config = write_config(sequence, method)
    if complete(sequence, method):
        return f"SKIP {method} {sequence}: complete", True
    gpu = gpu_pool.get()
    try:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        log_path = LOG_ROOT / f"{method}_{sequence}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        if method == "baseline":
            cwd = DROIDWO_ROOT
            env["PYTHONPATH"] = os.pathsep.join([str(DROIDWO_ROOT), str(DROIDWO_BINARY_ROOT), env.get("PYTHONPATH", "")])
        else:
            cwd = ROOT
            env["DROIDCALIB_SCHUR_SOURCE"] = "/tmp/droidcalib_schur_src"
        print(f"START {method} {sequence} gpu={gpu}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run([str(PYTHON), "run.py", "--config", str(config)], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        ok = result.returncode == 0 and complete(sequence, method)
        return f"DONE {method} {sequence} status={result.returncode} complete={ok}", ok
    finally:
        gpu_pool.put(gpu)


def run_phase(method: str, sequences: list[str], gpus: list[str]) -> bool:
    pool: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        pool.put(gpu)
    success = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_one, sequence, method, pool) for sequence in sequences]
        for future in concurrent.futures.as_completed(futures):
            message, ok = future.result()
            print(message, flush=True)
            success = success and ok
    return success


def trajectory(path: Path) -> PoseTrajectory3D:
    rows = np.loadtxt(path, comments="#", ndmin=2)
    return PoseTrajectory3D(rows[:, 1:4], np.column_stack((rows[:, 7], rows[:, 4:7])), rows[:, 0])


def evaluate(sequence: str, method: str) -> dict[str, float | int | str]:
    directory = traj_dir(sequence, method)
    estimate_path = directory / "est_poses_full.txt"
    timestamps_path = directory.parent / "timestamps.txt"
    if not estimate_path.is_file() or not timestamps_path.is_file():
        return {"status": "missing"}
    estimate = np.loadtxt(estimate_path, comments="#", ndmin=2)
    timestamps = np.loadtxt(timestamps_path, comments="#", ndmin=1)
    if len(estimate) != len(timestamps):
        return {"status": "timestamp-mismatch"}
    estimate[:, 0] = timestamps
    associated = directory / "est_poses_full_timestamped.txt"
    np.savetxt(associated, estimate, fmt="%.9f")
    gt_file = DATA_ROOT / sequence / "traj_gt_fastlivo.txt"
    if not gt_file.is_file():
        gt_file = DATA_ROOT / sequence / "traj_gt.txt"
    reference, estimated = sync.associate_trajectories(trajectory(gt_file), trajectory(associated), max_diff=0.1)
    _, _, scale = estimated.align(reference, correct_scale=True)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((reference, estimated))
    stats = ape.get_all_statistics()
    (directory / "metrics_full_traj.txt").write_text(
        "##########Full traj##########\n" f"scale: {scale}\nstatistics:\n{stats}\n", encoding="utf-8"
    )
    return {"status": "completed", "rmse": float(stats["rmse"]), "frames": int(reference.num_poses)}


def report(sequences: list[str]) -> bool:
    rows = [(sequence, evaluate(sequence, "baseline"), evaluate(sequence, "confidence")) for sequence in sequences]
    lines = [
        "# DROID-W Unknown-K: DROID-W-O + Omega K vs Confidence Main v1", "",
        "Protocol: one 12-frame, stride-5 VGGT-Omega pinhole K bootstrap per sequence; source timestamps; evo association `max_diff=0.1 s`; Sim(3) alignment; translation-part Full ATE RMSE in metres. DROID-W downtown has no separate keyframe trajectory output.", "",
        "| Sequence | DROID-W-O + Omega K (m) | Confidence Main v1 (m) | Delta (m) | Ratio | Frames |", "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    base, ours = [], []
    for sequence, baseline, confidence in rows:
        if baseline["status"] != "completed" or confidence["status"] != "completed":
            lines.append(f"| {sequence} | {baseline['status']} | {confidence['status']} | - | - | - |")
            continue
        b, c = float(baseline["rmse"]), float(confidence["rmse"])
        base.append(b); ours.append(c)
        lines.append(f"| {sequence} | {b:.9f} | {c:.9f} | {c-b:+.9f} | {c/b:.2%} | {int(baseline['frames'])} |")
    if base and len(base) == len(sequences):
        b, c = float(np.mean(base)), float(np.mean(ours))
        wins = sum(o < x for x, o in zip(base, ours))
        lines += ["", f"Macro mean: DROID-W-O + Omega K `{b:.9f} m`; Confidence Main v1 `{c:.9f} m`; delta `{c-b:+.9f} m` ({c/b-1:+.2%}); wins `{wins}/{len(base)}`."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return len(base) == len(sequences)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "extract", "baseline", "confidence", "report"), default="all")
    parser.add_argument("--sequences", default=",".join(SEQUENCES))
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    sequences = [item.strip() for item in args.sequences.split(",") if item.strip()]
    if set(sequences) - set(SEQUENCES):
        raise ValueError("Unknown DROID-W sequence")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    validate_source()
    if not PROFILE.is_file():
        raise FileNotFoundError(PROFILE)
    for sequence in sequences:
        if not (DATA_ROOT / sequence / "images_anonymized").is_dir():
            raise FileNotFoundError(sequence)
        if not (CACHE_ROOT / sequence / "confidences").is_dir():
            raise FileNotFoundError(f"Omega confidence cache missing for {sequence}")
    if args.phase in {"all", "extract"}:
        for index, sequence in enumerate(sequences):
            extract_intrinsics(sequence, gpus[index % len(gpus)])
    if args.phase in {"all", "baseline"} and not run_phase("baseline", sequences, gpus):
        return 1
    if args.phase in {"all", "confidence"} and not run_phase("confidence", sequences, gpus):
        return 1
    return 0 if args.phase in {"extract", "baseline", "confidence"} else int(not report(sequences))


if __name__ == "__main__":
    raise SystemExit(main())
