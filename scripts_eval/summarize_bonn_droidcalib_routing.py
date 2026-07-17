#!/usr/bin/env python3
"""Summarize the focal-recovery routing controls and Bonn sweep."""

from __future__ import annotations

import ast
import re
import statistics
from pathlib import Path


LEGACY_OUTPUT_ROOT = Path("/data1/czy/Output/DROID-omega/Bonn")
OUTPUT_ROOT = Path("Outputs/Bonn/droidcalib_routing_results")
REPORT_PATH = Path("experiments/bonn_droidcalib_routing_results.md")
CONTROL_SEQUENCES = ("bonn_moving_nonobstructing_box2", "bonn_person_tracking2", "bonn_crowd2")
ALL_SEQUENCES = (
    "bonn_balloon", "bonn_balloon2", "bonn_crowd", "bonn_crowd2",
    "bonn_moving_nonobstructing_box", "bonn_moving_nonobstructing_box2",
    "bonn_person_tracking", "bonn_person_tracking2",
)
SEEDS = (41, 42, 43)
METHODS = ("stability", "omega_fixed_k", "k_recovery", "confidence_recovery")


def scene_name(sequence: str, method: str, seed: int) -> str:
    return f"{sequence}_droidcalib_routing_{method}_s{seed}"


def read_rmse(path: Path) -> float | None:
    if not path.is_file():
        return None
    match = re.search(r"statistics:\n(\{.*?\})", path.read_text(encoding="utf-8"))
    return None if match is None else float(ast.literal_eval(match.group(1))["rmse"])


def metric(sequence: str, method: str, seed: int, kind: str) -> float | None:
    for root in (OUTPUT_ROOT, LEGACY_OUTPUT_ROOT):
        value = read_rmse(root / scene_name(sequence, method, seed) / "traj" / f"metrics_{kind}_traj.txt")
        if value is not None:
            return value
    return None


def existing_stability(sequence: str, kind: str) -> float | None:
    return read_rmse(LEGACY_OUTPUT_ROOT / f"{sequence}_omega_droidcalib_schur_stability_v2" / "traj" / f"metrics_{kind}_traj.txt")


def fmt(value: float | None) -> str:
    return "missing" if value is None else f"{value:.6f}"


def main() -> int:
    lines = [
        "# Bonn DROIDCalib Routing Results", "",
        "ATE values are RMSE in metres. Controls use seeds 41, 42, and 43.",
        "The full sweep uses seed 43 and compares confidence recovery with the existing stability-v2 run.",
        "", "## Multi-Seed Controls", "",
        "| Sequence | Method | Full mean | Full std | KF mean | Completed |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    control_complete = True
    for sequence in CONTROL_SEQUENCES:
        for method in METHODS:
            full = [metric(sequence, method, seed, "full") for seed in SEEDS]
            kf = [metric(sequence, method, seed, "kf") for seed in SEEDS]
            present_full = [value for value in full if value is not None]
            present_kf = [value for value in kf if value is not None]
            control_complete &= len(present_full) == len(SEEDS) and len(present_kf) == len(SEEDS)
            mean = statistics.mean(present_full) if present_full else None
            std = statistics.stdev(present_full) if len(present_full) > 1 else (0.0 if present_full else None)
            kf_mean = statistics.mean(present_kf) if present_kf else None
            lines.append(f"| {sequence} | {method} | {fmt(mean)} | {fmt(std)} | {fmt(kf_mean)} | {len(present_full)}/3 |")
    lines.extend(["", "## Full Bonn Sweep", "", "| Sequence | Stability v2 Full | Confidence recovery Full | Delta |", "| --- | ---: | ---: | ---: |"])
    sweep_complete = True
    for sequence in ALL_SEQUENCES:
        stability = existing_stability(sequence, "full")
        confidence = metric(sequence, "confidence_recovery", 43, "full")
        sweep_complete &= stability is not None and confidence is not None
        delta = "missing" if stability is None or confidence is None else f"{100.0 * (confidence / stability - 1.0):+.2f}%"
        lines.append(f"| {sequence} | {fmt(stability)} | {fmt(confidence)} | {delta} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if control_complete and sweep_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
