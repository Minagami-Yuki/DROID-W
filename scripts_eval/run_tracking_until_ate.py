#!/usr/bin/env python3
"""Run one tracking config until both trajectory ATE files are durable."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/czy/anaconda3/envs/droid-w/bin/python")
OUTPUT_ROOT = Path("/data1/czy/Output/DROID-omega/Bonn")


def complete(scene: str) -> bool:
    traj = OUTPUT_ROOT / scene / "traj"
    return (traj / "metrics_full_traj.txt").is_file() and (traj / "metrics_kf_traj.txt").is_file()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()

    if complete(args.scene):
        print(f"ATE already complete: {args.scene}")
        return 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    log_path = args.log or ROOT / "Outputs" / "tracking_logs" / f"{args.scene}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(PYTHON), "run.py", "--config", str(args.config)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None and not complete(args.scene):
            time.sleep(args.poll_seconds)
        if complete(args.scene):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            print(f"ATE complete: {args.scene}")
            return 0
        raise RuntimeError(f"Tracking exited with status {process.returncode} before both ATE files were written; log: {log_path}")


if __name__ == "__main__":
    raise SystemExit(main())
