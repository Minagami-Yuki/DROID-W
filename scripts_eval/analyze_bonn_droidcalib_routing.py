#!/usr/bin/env python3
"""Diagnose repeatability of confidence-recovery focal calibration."""

from __future__ import annotations

import ast
import csv
import re
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "Outputs/Bonn/droidcalib_routing_results"
LEGACY = Path("/data1/czy/Output/DROID-omega/Bonn")
REPORT = ROOT / "experiments/bonn_droidcalib_routing_diagnostics.md"

RUNS = {
    "balloon2 first": LOCAL / "bonn_balloon2_droidcalib_routing_confidence_recovery_s43",
    "balloon2 repeat01": LOCAL / "bonn_balloon2_droidcalib_routing_confidence_recovery_s43_repeat01",
    "person_tracking2 first": LEGACY / "bonn_person_tracking2_droidcalib_routing_confidence_recovery_s43",
    "person_tracking2 repeat01": LOCAL / "bonn_person_tracking2_droidcalib_routing_confidence_recovery_s43_repeat01",
}
BASELINES = {
    "balloon2": LEGACY / "bonn_balloon2_omega_droidcalib_schur_stability_v2",
    "person_tracking2": LEGACY / "bonn_person_tracking2_omega_droidcalib_schur_stability_v2",
}


def ate(run: Path, kind: str) -> float:
    text = (run / "traj" / f"metrics_{kind}_traj.txt").read_text(encoding="utf-8")
    return float(ast.literal_eval(re.search(r"statistics:\n(\{.*?\})", text).group(1))["rmse"])


def number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return None if value in ("", None) else float(value)


def stats(run: Path) -> dict[str, object]:
    with (run / "focal_calibration.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    accepted = [row for row in rows if row["accepted"] == "1"]
    recovery = [row for row in rows if row["omega_confidence_recovery"] == "1"]
    recovery_accepted = [row for row in accepted if row["omega_confidence_recovery"] == "1"]
    abs_steps = [abs(number(row, "step_log_applied") or 0.0) for row in accepted]
    pose_t = [number(row, "base_pose_translation_max") for row in rows]
    pose_r = [number(row, "base_pose_rotation_max") for row in rows]
    pose_t = [value for value in pose_t if value is not None]
    pose_r = [value for value in pose_r if value is not None]
    scores = [number(row, "omega_confidence_shape_score") for row in rows]
    scores = [value for value in scores if value is not None]
    return {
        "rows": len(rows),
        "accepted": len(accepted),
        "recovery": len(recovery),
        "recovery_accepted": len(recovery_accepted),
        "reasons": Counter(row["reason"] for row in rows),
        "fx_initial": number(rows[0], "fx_before"),
        "fx_final": number(rows[-1], "fx_after"),
        "step_sum": sum(abs_steps),
        "step_max": max(abs_steps, default=0.0),
        "score_median": statistics.median(scores),
        "score_min": min(scores),
        "score_max": max(scores),
        "pose_t_max": max(pose_t, default=0.0),
        "pose_r_max": max(pose_r, default=0.0),
    }


def main() -> int:
    all_stats = {name: stats(path) for name, path in RUNS.items()}
    lines = [
        "# Bonn Confidence-Recovery Repeatability and Mechanism Diagnostics",
        "",
        "All runs use the same seed (43), the same stability-v2 base configuration, and the same",
        "Omega confidence threshold (30 samples, mean/median ratio >= 1.05). ATE is RMSE in metres.",
        "",
        "## ATE Repeatability",
        "",
        "| Sequence | Stability v2 Full/KF | First confidence Full/KF | Repeat01 Full/KF |",
        "| --- | ---: | ---: | ---: |",
    ]
    for sequence in ("balloon2", "person_tracking2"):
        baseline = BASELINES[sequence]
        first = RUNS[f"{sequence} first"]
        repeat = RUNS[f"{sequence} repeat01"]
        lines.append(
            f"| {sequence} | {ate(baseline, 'full'):.6f} / {ate(baseline, 'kf'):.6f} | "
            f"{ate(first, 'full'):.6f} / {ate(first, 'kf'):.6f} | "
            f"{ate(repeat, 'full'):.6f} / {ate(repeat, 'kf'):.6f} |"
        )

    lines.extend([
        "", "## Calibration-Path Statistics", "",
        "| Run | Accepted / rows | Recovery / accepted | fx initial → final | |Δfx| | Σ|log step| | max |log step| | confidence median [min,max] | max pose t / r |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name, item in all_stats.items():
        shift = abs(float(item["fx_final"]) / float(item["fx_initial"]) - 1.0) * 100.0
        lines.append(
            f"| {name} | {item['accepted']} / {item['rows']} | {item['recovery']} / {item['recovery_accepted']} | "
            f"{item['fx_initial']:.3f} → {item['fx_final']:.3f} | {shift:.2f}% | "
            f"{item['step_sum']:.6f} | {item['step_max']:.6f} | "
            f"{item['score_median']:.4f} [{item['score_min']:.4f}, {item['score_max']:.4f}] | "
            f"{item['pose_t_max']:.4f} / {item['pose_r_max']:.4f} |"
        )

    lines.extend([
        "", "## Rejection Reasons", "",
        "| Run | accepted | unstable trajectory | loss increase | low observability |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for name, item in all_stats.items():
        reasons = item["reasons"]
        lines.append(
            f"| {name} | {reasons['accepted']} | {reasons['unstable_trajectory']} | "
            f"{reasons['loss_increase']} | {reasons['low_observability']} |"
        )

    lines.extend([
        "", "## Interpretation", "",
        "The balloon2 failure is not repeatable with the fixed seed: its repeat01 ATE returns near the",
        "stability-v2 baseline even though the confidence statistic, recovery duration, final focal, and",
        "rejection counts are nearly identical.  The remaining difference is the accumulated accepted",
        "Schur focal-step path, indicating numerical/multithreaded BA sensitivity rather than a robust",
        "sequence-level routing signal.  Person_tracking2 is repeatable in ATE and focal evolution,",
        "confirming that the same branch can be stable on another sequence while still not providing a",
        "safe universal trigger.",
    ])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
