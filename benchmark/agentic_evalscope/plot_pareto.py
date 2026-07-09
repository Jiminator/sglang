#!/usr/bin/env python3
"""Plot the OpenHands agentic-sweep Pareto curve for the GLM-5.2-NVFP4 run.

Only the WS=4 (``tp4``) configs are plotted, one panel per config, with axes
auto-scaled and each point annotated with its concurrency.

Axis math:
  X = User Token/s        = 1000 / TPOT(ms)
  Y = kTokens / GPU-min   = Total Throughput(tok/s) / n_gpus * 60 / 1000

Usage:
    python3 plot_pareto.py \
        --sweep <evalscope outputs dir, e.g. outputs/20260628_091132> \
        --output <png path> \
        --label "SGLang (GLM-5.2 NVFP4, EAGLE) — openhands"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Only the WS=4 configs the default 4-GPU sweep runs.
CONFIG_TITLES = {
    "attn_tp4_moe_tp4": "Attn TP4, MoE TP",
    "attn_tp4_moe_ep4": "Attn TP4, MoE EP",
}
PANEL_ORDER = ["attn_tp4_moe_tp4", "attn_tp4_moe_ep4"]


def num_gpus_from_config(config: str) -> int:
    match = re.search(r"attn_(?:tp|dp)(\d+)", config)
    if not match:
        raise ValueError(f"Cannot infer GPU count from config name: {config}")
    return int(match.group(1))


def read_sweep(path: Path) -> dict[str, list[tuple[float, float, int]]]:
    """Return {config: [(user_tps, k_tok_per_gpu_min, concurrency), ...]}."""
    by_config: dict[str, list[tuple[float, float, int]]] = {}
    if not path.is_dir():
        return by_config

    for config_dir in sorted(p for p in path.iterdir() if p.is_dir()):
        config = config_dir.name
        if config not in CONFIG_TITLES:
            continue
        n_gpus = num_gpus_from_config(config)
        for summary_path in sorted(config_dir.glob("*/benchmark_summary.json")):
            summary = json.loads(summary_path.read_text())
            tpot_ms = summary["TPOT (ms)"]
            user_tps = 1000.0 / tpot_ms if tpot_ms else 0.0
            tps_gpu = summary["Total Throughput (tok/s)"] / n_gpus
            k_tok_gpu_min = tps_gpu * 60.0 / 1000.0
            by_config.setdefault(config, []).append(
                (user_tps, k_tok_gpu_min, int(summary["Concurrency"]))
            )

    for points in by_config.values():
        points.sort(key=lambda item: item[2])
    return by_config


def plot_line(ax, points, *, label: str) -> None:
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    concs = [p[2] for p in points]
    ax.plot(
        xs,
        ys,
        label=label,
        color="#ff8c00",
        linestyle="-",
        marker="o",
        linewidth=2.6,
        markersize=5,
    )
    # Annotate each point with its concurrency so the chart reads at a glance.
    for x, y, c in zip(xs, ys, concs):
        ax.annotate(
            str(c),
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color="#536173",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        type=Path,
        required=True,
        help="evalscope sweep dir containing <config>/parallel_*/benchmark_summary.json",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument(
        "--label",
        type=str,
        default="SGLang (GLM-5.2 NVFP4, EAGLE) — openhands",
        help="Legend label for the line.",
    )
    args = parser.parse_args()

    sglang = read_sweep(args.sweep)
    present = [c for c in PANEL_ORDER if sglang.get(c)]
    if not present:
        parser.error(f"no tp4 config data found under {args.sweep}")

    ncols = len(present)
    fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols, 4.8), dpi=200, squeeze=False)
    for ax, config in zip(axes[0], present):
        plot_line(ax, sglang.get(config, []), label=args.label)
        ax.set_title(CONFIG_TITLES[config], fontsize=12, fontweight="bold", pad=10)
        ax.grid(True, color="#e8ecf0", linewidth=0.9)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#e8ecf0")
        ax.spines["bottom"].set_color("#e8ecf0")
        ax.tick_params(colors="#536173", labelsize=10)
        ax.set_xlabel("User Token/s", fontsize=11)

    axes[0][0].set_ylabel("Thousand Tokens / GPU mins", fontsize=11)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.20, wspace=0.24)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
