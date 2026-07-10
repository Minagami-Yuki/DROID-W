import argparse
import csv
import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Omega patch-token uncertainty calibration curves."
    )
    parser.add_argument("csv", nargs="+", help="patch_token_uncertainty_stats.csv files.")
    parser.add_argument(
        "--output",
        default="experiments/figures/omega_patch_token_calibration.png",
        help="PNG output path.",
    )
    parser.add_argument("--pdf", default=None, help="Optional PDF output path.")
    parser.add_argument(
        "--scores",
        nargs="+",
        default=["risk_mean", "gate_mean"],
        help="Score columns to plot against the target.",
    )
    parser.add_argument(
        "--target",
        default="edge_residual_mean",
        help="Target residual column. edge_residual_mean is synthesized when possible.",
    )
    parser.add_argument("--bins", type=int, default=8, help="Quantile bins per scene.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=800,
        help="Maximum scatter points per scene and score.",
    )
    return parser.parse_args()


def scene_name_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 3 and parts[-2] == "debug":
        return parts[-3]
    return os.path.splitext(os.path.basename(path))[0]


def display_scene_name(scene):
    if scene.startswith("bonn_crowd2"):
        return "Bonn crowd2"
    if scene.startswith("bonn_moving_nonobstructing_box"):
        return "Bonn moving box"
    if "_omega_" in scene:
        scene = scene.split("_omega_", 1)[0]
    return scene.replace("_", " ")


def to_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


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


def paired_values(rows, score_col, target_col):
    xs = []
    ys = []
    for row in rows:
        x = row.get(score_col)
        y = row.get(target_col)
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


def quantile_bins(xs, ys, num_bins):
    pairs = sorted(zip(xs, ys), key=lambda item: item[0])
    if not pairs:
        return []
    num_bins = max(1, min(num_bins, len(pairs)))
    out = []
    for idx in range(num_bins):
        start = idx * len(pairs) // num_bins
        end = (idx + 1) * len(pairs) // num_bins
        chunk = pairs[start:end]
        bx = [x for x, _ in chunk]
        by = [y for _, y in chunk]
        out.append((mean(bx), mean(by), len(chunk)))
    return out


def sampled_pairs(xs, ys, max_points):
    if max_points <= 0 or len(xs) <= max_points:
        return xs, ys
    step = max(1, len(xs) // max_points)
    return xs[::step][:max_points], ys[::step][:max_points]


def plot_calibration(csv_paths, scores, target, bins, max_points, output, pdf=None):
    scenes = []
    for path in csv_paths:
        rows = load_rows(path)
        scenes.append((scene_name_from_path(path), rows))

    fig, axes = plt.subplots(
        1,
        len(scores),
        figsize=(5.0 * len(scores), 4.0),
        squeeze=False,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plotted = False

    for ax, score_col in zip(axes[0], scores):
        for scene_idx, (scene, rows) in enumerate(scenes):
            xs, ys = paired_values(rows, score_col, target)
            if len(xs) < 2:
                continue
            plotted = True
            display_name = display_scene_name(scene)
            color = colors[scene_idx % len(colors)]
            sx, sy = sampled_pairs(xs, ys, max_points)
            ax.scatter(sx, sy, s=8, alpha=0.14, color=color, edgecolors="none")

            binned = quantile_bins(xs, ys, bins)
            bx = [item[0] for item in binned]
            by = [item[1] for item in binned]
            corr = pearson(xs, ys)
            ax.plot(
                bx,
                by,
                marker="o",
                linewidth=2.0,
                color=color,
                label=f"{display_name} (r={corr:.2f}, n={len(xs)})",
            )

        ax.set_xlabel(score_col)
        ax.set_ylabel(target)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{score_col} calibration")
        ax.legend(fontsize=7)

    if not plotted:
        raise RuntimeError("No plottable score/target pairs were found.")

    fig.suptitle("Omega Patch-Token Uncertainty Calibration", fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=220)
    if pdf:
        os.makedirs(os.path.dirname(pdf) or ".", exist_ok=True)
        fig.savefig(pdf)
    plt.close(fig)


def main():
    args = parse_args()
    plot_calibration(
        csv_paths=args.csv,
        scores=args.scores,
        target=args.target,
        bins=args.bins,
        max_points=args.max_points,
        output=args.output,
        pdf=args.pdf,
    )
    print(f"wrote {args.output}")
    if args.pdf:
        print(f"wrote {args.pdf}")


if __name__ == "__main__":
    main()
