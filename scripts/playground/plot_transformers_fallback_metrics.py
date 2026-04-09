#!/usr/bin/env python3
"""Plot transformers fallback metrics from per-commit CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_output_dir = repo_root / "tmp" / "transformers_regression"

    parser = argparse.ArgumentParser(
        description="Plot transformers fallback metrics from CSV files."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="Per-commit CSV files produced by run_transformers_fallback_metrics.sh",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory for summary files and figure output.",
    )
    parser.add_argument(
        "--title",
        default="Transformers Fallback TorchAO Regression",
        help="Figure title.",
    )
    return parser.parse_args()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_rows(csv_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["source_csv"] = str(csv_path)
                row["commit_label"] = row.get("commit_label") or csv_path.stem
                row["benchmark_bound"] = (
                    float(row["benchmark_bound"])
                    if row.get("benchmark_bound")
                    else None
                )
                if row.get("score"):
                    row["score"] = float(row["score"])
                else:
                    row["score"] = None

                # Backward compatibility with the previous wide CSV schema.
                if row["score"] is None:
                    if row.get("benchmark") == "gsm8k" and row.get("gsm8k_score"):
                        row["score"] = float(row["gsm8k_score"])
                    elif row.get("benchmark") == "mmlu" and row.get("mmlu_score"):
                        row["score"] = float(row["mmlu_score"])
                rows.append(row)
    return rows


def benchmark_scores(rows: list[dict[str, Any]], commit_label: str, benchmark: str) -> list[float]:
    return [
        row["score"]
        for row in rows
        if row["commit_label"] == commit_label
        and row["benchmark"] == benchmark
        and row["score"] is not None
    ]


def benchmark_bound(rows: list[dict[str, Any]], commit_label: str, benchmark: str) -> float | None:
    for row in rows:
        if row["commit_label"] == commit_label and row["benchmark"] == benchmark:
            return row["benchmark_bound"]
    return None


def compute_summary(rows: list[dict[str, Any]], commit_labels: list[str]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for commit_label in commit_labels:
        for benchmark in ("gsm8k", "mmlu"):
            scores = benchmark_scores(rows, commit_label, benchmark)
            if not scores:
                continue
            if len(scores) >= 2:
                sample_variance = statistics.variance(scores)
            else:
                sample_variance = 0.0
            bound = benchmark_bound(rows, commit_label, benchmark)
            below_bound_count = (
                sum(score < bound for score in scores) if bound is not None else 0
            )
            summary_rows.append(
                {
                    "commit_label": commit_label,
                    "benchmark": benchmark,
                    "trials": len(scores),
                    "mean": statistics.mean(scores),
                    "sample_variance": sample_variance,
                    "sample_std": math.sqrt(sample_variance),
                    "min": min(scores),
                    "max": max(scores),
                    "bound_used": bound,
                    "below_bound_count": below_bound_count,
                }
            )
    return summary_rows


def write_summary(output_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    ensure_directory(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps({"summary_rows": summary_rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    fieldnames = [
        "commit_label",
        "benchmark",
        "trials",
        "mean",
        "sample_variance",
        "sample_std",
        "min",
        "max",
        "bound_used",
        "below_bound_count",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def plot(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], commit_labels: list[str], output_dir: Path, title: str) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to plot metrics. Install it before running this script."
        ) from exc

    summary_map = {
        (row["commit_label"], row["benchmark"]): row for row in summary_rows
    }
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b", "#17becf"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    for column, benchmark in enumerate(("mmlu", "gsm8k")):
        top_ax = axes[0][column]
        bottom_ax = axes[1][column]
        all_scores: list[float] = []
        all_bounds: list[float] = []

        for index, commit_label in enumerate(commit_labels):
            color = colors[index % len(colors)]
            scores = benchmark_scores(rows, commit_label, benchmark)
            if not scores:
                continue
            all_scores.extend(scores)

            rng = random.Random(f"{benchmark}:{commit_label}")
            xs = [index + rng.uniform(-0.12, 0.12) for _ in scores]
            top_ax.scatter(xs, scores, color=color, alpha=0.75, s=28)

            summary = summary_map[(commit_label, benchmark)]
            mean = summary["mean"]
            std = summary["sample_std"]
            top_ax.errorbar(
                [index],
                [mean],
                yerr=[std],
                fmt="o",
                color="black",
                ecolor=color,
                elinewidth=2,
                capsize=5,
                markersize=7,
            )

            bound = summary["bound_used"]
            if bound is not None:
                all_bounds.append(bound)
                top_ax.hlines(
                    bound,
                    index - 0.24,
                    index + 0.24,
                    color=color,
                    linestyle="dashed",
                    linewidth=1.4,
                    alpha=0.9,
                )
                top_ax.text(
                    index,
                    max(scores + [mean, bound]) + 0.02,
                    f"breaches: {summary['below_bound_count']}/{len(scores)} @ {bound:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        if all_scores:
            y_min = min(all_scores + all_bounds) - 0.05 if all_bounds else min(all_scores) - 0.05
            y_max = max(all_scores + all_bounds) + 0.12 if all_bounds else max(all_scores) + 0.12
            top_ax.set_ylim(max(0.0, y_min), min(1.0, y_max))

        top_ax.set_title(f"{benchmark.upper()} Raw Trial Scores")
        top_ax.set_ylabel("Score")
        top_ax.set_xticks(range(len(commit_labels)))
        top_ax.set_xticklabels(commit_labels)
        top_ax.grid(axis="y", linestyle=":", alpha=0.45)

        variance_positions = [
            index
            for index, commit_label in enumerate(commit_labels)
            if (commit_label, benchmark) in summary_map
        ]
        variances = [
            summary_map[(commit_label, benchmark)]["sample_variance"]
            for commit_label in commit_labels
            if (commit_label, benchmark) in summary_map
        ]
        bars = bottom_ax.bar(
            variance_positions,
            variances,
            color=[colors[i % len(colors)] for i in range(len(variances))],
            alpha=0.85,
        )
        for bar, variance in zip(bars, variances):
            bottom_ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(variances + [0.0]) * 0.03 + 1e-9,
                f"{variance:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        bottom_ax.set_title(f"{benchmark.upper()} Sample Variance")
        bottom_ax.set_ylabel("Variance")
        bottom_ax.set_xticks(range(len(commit_labels)))
        bottom_ax.set_xticklabels(commit_labels)
        bottom_ax.grid(axis="y", linestyle=":", alpha=0.45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = output_dir / "transformers_fallback_regression.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv_files)
    if not rows:
        raise RuntimeError("No metric rows were found in the provided CSV files.")

    commit_labels = []
    for row in rows:
        label = row["commit_label"]
        if label not in commit_labels:
            commit_labels.append(label)

    summary_rows = compute_summary(rows, commit_labels)
    write_summary(args.output_dir, summary_rows)
    output_path = plot(rows, summary_rows, commit_labels, args.output_dir, args.title)
    print(f"Wrote summary and figure outputs to {args.output_dir}")
    print(f"Figure: {output_path}")


if __name__ == "__main__":
    main()
