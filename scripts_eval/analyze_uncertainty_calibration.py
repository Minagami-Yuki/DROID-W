import argparse
import csv
import math
import os
from collections import defaultdict


SCORE_COLUMNS = [
    "token_distance_mean",
    "token_distance_max",
    "risk_mean",
    "risk_max",
    "gate_mean",
    "gate_max",
]

TARGET_COLUMNS = [
    "residual_dtf_mean",
    "edge_residual_mean",
    "scale_mean",
    "scale_min",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze patch-token uncertainty calibration CSVs."
    )
    parser.add_argument("csv", nargs="+", help="patch_token_uncertainty_stats.csv files.")
    parser.add_argument(
        "--output",
        default="experiments/uncertainty_calibration_results.md",
        help="Markdown file to write.",
    )
    parser.add_argument("--bins", type=int, default=5, help="Quantile bins per score.")
    return parser.parse_args()


def to_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def scene_name_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 3 and parts[-2] == "debug":
        return parts[-3]
    return os.path.splitext(os.path.basename(path))[0]


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {}
            for key, value in raw.items():
                parsed = to_float(value)
                if parsed is not None:
                    row[key] = parsed
            if "edge_residual_pixel_mean" in row:
                row["edge_residual_mean"] = row["edge_residual_pixel_mean"]
            elif "source_edge_mean" in row and "residual_dtf_mean" in row:
                row["edge_residual_mean"] = row["source_edge_mean"] * row["residual_dtf_mean"]
            rows.append(row)
    return rows


def paired_values(rows, x_col, y_col):
    xs = []
    ys = []
    for row in rows:
        x = row.get(x_col)
        y = row.get(y_col)
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    vx = sum(x * x for x in dx)
    vy = sum(y * y for y in dy)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return sum(x * y for x, y in zip(dx, dy)) / math.sqrt(vx * vy)


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def spearman(xs, ys):
    if len(xs) < 2:
        return float("nan")
    return pearson(ranks(xs), ranks(ys))


def quantile_bins(xs, ys, num_bins):
    pairs = sorted(zip(xs, ys), key=lambda item: item[0])
    if not pairs:
        return []
    num_bins = max(1, min(num_bins, len(pairs)))
    bins = []
    for idx in range(num_bins):
        start = idx * len(pairs) // num_bins
        end = (idx + 1) * len(pairs) // num_bins
        chunk = pairs[start:end]
        bx = [item[0] for item in chunk]
        by = [item[1] for item in chunk]
        bins.append(
            {
                "idx": idx + 1,
                "count": len(chunk),
                "score_min": min(bx),
                "score_max": max(bx),
                "score_mean": mean(bx),
                "target_mean": mean(by),
            }
        )
    return bins


def high_low_ratio(xs, ys, fraction=0.2):
    pairs = sorted(zip(xs, ys), key=lambda item: item[0])
    if len(pairs) < 5:
        return float("nan")
    count = max(1, int(round(len(pairs) * fraction)))
    low = mean([y for _, y in pairs[:count]])
    high = mean([y for _, y in pairs[-count:]])
    if abs(low) < 1e-12:
        return float("nan")
    return high / low


def monotonic_steps(bins, higher_is_better=True):
    if len(bins) < 2:
        return 0, 0
    values = [b["target_mean"] for b in bins]
    good = 0
    for a, b in zip(values[:-1], values[1:]):
        if higher_is_better and b >= a:
            good += 1
        if not higher_is_better and b <= a:
            good += 1
    return good, len(values) - 1


def fmt(value):
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def summarize_scene(scene, rows, num_bins):
    summaries = []
    for score_col in SCORE_COLUMNS:
        for target_col in TARGET_COLUMNS:
            xs, ys = paired_values(rows, score_col, target_col)
            if len(xs) < 2:
                continue
            bins = quantile_bins(xs, ys, num_bins)
            higher_target = not target_col.startswith("scale")
            mono_good, mono_total = monotonic_steps(bins, higher_is_better=higher_target)
            summaries.append(
                {
                    "scene": scene,
                    "score": score_col,
                    "target": target_col,
                    "rows": len(xs),
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                    "high_low_ratio": high_low_ratio(xs, ys),
                    "monotonic": f"{mono_good}/{mono_total}",
                    "bins": bins,
                }
            )
    return summaries


def write_markdown(output, csv_paths, all_summaries):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    key_pairs = [
        ("token_distance_mean", "edge_residual_mean"),
        ("token_distance_max", "edge_residual_mean"),
        ("risk_mean", "edge_residual_mean"),
        ("risk_max", "edge_residual_mean"),
        ("gate_mean", "edge_residual_mean"),
        ("gate_max", "edge_residual_mean"),
        ("gate_mean", "scale_mean"),
        ("gate_max", "scale_min"),
    ]
    summaries_by_key = {
        (item["scene"], item["score"], item["target"]): item for item in all_summaries
    }

    with open(output, "w") as f:
        f.write("# Omega Patch-Token Uncertainty Calibration\n\n")
        f.write("Date: 2026-07-09\n\n")
        f.write(
            "Goal: test whether the Omega patch-token uncertainty/gate is calibrated with "
            "Edge-DTF dynamic evidence at the per-edge level. Higher token/risk/gate should "
            "correspond to higher Edge-DTF residual; higher gate should also correspond to "
            "lower final BA scale.\n\n"
        )
        f.write("Input CSVs:\n\n")
        for path in csv_paths:
            f.write(f"- `{path}`\n")
        f.write("\n")

        f.write("## Key Correlations\n\n")
        f.write(
            "| Scene | Score | Target | Rows | Pearson | Spearman | Top20/Bottom20 | Monotonic bins |\n"
        )
        f.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for (scene, score, target), item in sorted(summaries_by_key.items()):
            if (score, target) not in key_pairs:
                continue
            f.write(
                f"| {scene} | {score} | {target} | {item['rows']} | "
                f"{fmt(item['pearson'])} | {fmt(item['spearman'])} | "
                f"{fmt(item['high_low_ratio'])} | {item['monotonic']} |\n"
            )
        f.write("\n")

        f.write("## Quantile Bins\n\n")
        for (scene, score, target), item in sorted(summaries_by_key.items()):
            if (score, target) not in key_pairs:
                continue
            f.write(f"### {scene}: {score} -> {target}\n\n")
            f.write("| Bin | Count | Score mean | Score range | Target mean |\n")
            f.write("| ---: | ---: | ---: | --- | ---: |\n")
            for b in item["bins"]:
                f.write(
                    f"| {b['idx']} | {b['count']} | {fmt(b['score_mean'])} | "
                    f"{fmt(b['score_min'])}-{fmt(b['score_max'])} | {fmt(b['target_mean'])} |\n"
                )
            f.write("\n")

        f.write("## Notes\n\n")
        f.write(
            "- `edge_residual_mean` uses the exact pixelwise `edge_residual_pixel_mean` column when available, and falls back to `source_edge_mean * residual_dtf_mean` for old CSVs.\n"
        )
        f.write(
            "- `gate_*` may include residual-based evidence depending on the config, so gate-to-residual correlation is a sanity check rather than an independent prior test.\n"
        )
        f.write(
            "- `token_distance_*` and `risk_*` are the more useful independent signals for paper-facing calibration evidence.\n"
        )


def main():
    args = parse_args()
    grouped = defaultdict(list)
    for path in args.csv:
        scene = scene_name_from_path(path)
        grouped[scene].extend(load_rows(path))

    all_summaries = []
    for scene, rows in grouped.items():
        all_summaries.extend(summarize_scene(scene, rows, args.bins))

    write_markdown(args.output, args.csv, all_summaries)
    print(f"Wrote {args.output}")
    for scene, rows in sorted(grouped.items()):
        print(f"{scene}: rows={len(rows)}")


if __name__ == "__main__":
    main()
