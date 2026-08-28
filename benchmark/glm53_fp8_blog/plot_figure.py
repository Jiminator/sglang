#!/usr/bin/env python3
"""Plot the GLM-5.3 FP8 Pareto figure from results/ produced by the run scripts.

Usage:
    python3 plot_figure.py b300  [--out figure.png]
    python3 plot_figure.py gb300
    python3 plot_figure.py all          # both platforms side by side

One panel per platform with two curves — TP and TEP — read from
results/<platform>/glm53/<panel>/parallel_*/benchmark_{summary,percentile}.json.

X = p90 interactivity = 1000 / p90 TPOT(ms), where per-request
TPOT = (latency - TTFT) / (output_tokens - 1); override with --x-tpot.
Y = total throughput (prompt+completion tok/s, as evalscope reports) / n_gpus;
one point per concurrency 1,2,4,8. The y-axis is truncated (break glyph,
'0' at the origin).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

BASE = Path(__file__).resolve().parent
CONCS = (1, 2, 4, 8)
X_TPOT = "p90"  # avg | p50 | p90 | p99

# platform -> ((curve label, panel dir, color, linestyle, annotate-above), ...),
#             n_gpus, y-axis floor
PLATFORMS = {
    "gb300": ((("TP=4", "tp4", "#A43223", "-", True),
               ("TEP=4", "tep4", "#2E6E8E", "-", False)), 4, 10000),
    "b300": ((("TP=8", "tp8", "#A43223", "-", True),
              ("TEP=8", "tep8", "#2E6E8E", "-", False)), 8, 5000),
}


def tpot_ms(summary_file, summary):
    """TPOT for the x-axis: the summary average, or a percentile row of the
    benchmark_percentile.json sitting next to the summary."""
    if X_TPOT == "avg":
        return summary["TPOT (ms)"]
    rows = json.loads((summary_file.parent / "benchmark_percentile.json").read_text())
    want = X_TPOT.replace("p", "") + "%"
    return next(r["TPOT (ms)"] for r in rows if r["Percentiles"] == want)


def read_points(platform, panel_dir, n_gpus):
    base = BASE / "results" / platform / "glm53" / panel_dir
    pts = []
    for f in sorted(base.glob("parallel_*/benchmark_summary.json")):
        s = json.loads(f.read_text())
        conc = int(s["Concurrency"])
        if conc not in CONCS:
            continue
        tpot = tpot_ms(f, s)
        pts.append((1000.0 / tpot if tpot else 0.0,
                    s["Total Throughput (tok/s)"] / n_gpus, conc))
    pts.sort(key=lambda p: p[2])
    return pts


def axis_break(ax, work_min):
    ax.spines["left"].set_bounds(work_min, ax.get_ylim()[1])
    y0, y1 = ax.get_ylim()
    wf = (work_min - y0) / (y1 - y0)
    seg_x = [0, 0, 0.020, -0.020, 0, 0]
    seg_y = [0.0, 0.30 * wf, 0.42 * wf, 0.58 * wf, 0.69 * wf, wf]
    ax.plot(seg_x, seg_y, transform=ax.transAxes, clip_on=False,
            color="black", linewidth=1.4, solid_capstyle="round", zorder=5)
    ax.text(-0.018, 0.0, "0", transform=ax.transAxes, ha="right",
            va="center", fontsize=10, color="#222222")


def plot_curve(ax, pts, *, color, linestyle, above, label):
    if not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, label=label, color=color, linestyle=linestyle,
            marker="o", linewidth=1.5, markersize=6,
            markeredgecolor="white", markeredgewidth=0.8)
    off = (5, 7) if above else (-3, -15)
    for x, y, c in zip(xs, ys, (p[2] for p in pts)):
        ax.annotate(str(c), (x, y), textcoords="offset points",
                    xytext=off, fontsize=8, color="#8a8a8a")


def style_panel(ax, title, xmax, ymax, work_min):
    ytop = ymax * 1.07
    wf = 0.12
    ax.set_xlim(0, xmax * 1.07)
    ax.set_ylim((work_min - wf * ytop) / (1 - wf), ytop)
    ax.set_xticks(list(range(50, int(xmax * 1.07) + 1, 50)))
    ax.yaxis.set_major_locator(MultipleLocator(5000))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:g}k"))
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", color="#9aa0a6", alpha=0.4, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("black")
        ax.spines[sp].set_linewidth(1.4)
    ax.tick_params(colors="#222222", labelsize=10, width=1.2)
    axis_break(ax, work_min)


def render_panel(ax, platform):
    curves, n_gpus, work_min = PLATFORMS[platform]
    series = [(c, read_points(platform, c[1], n_gpus)) for c in curves]
    series = [(c, p) for c, p in series if p]
    if not series:
        raise SystemExit(f"no data under results/{platform}/glm53/ — run the sweeps first")
    all_pts = [p for _, pts in series for p in pts]
    xmax = max(p[0] for p in all_pts)
    ymax = max(p[1] for p in all_pts)
    for (label, pdir, color, ls, above), pts in series:
        plot_curve(ax, pts, color=color, linestyle=ls, above=above, label=label)
    style_panel(ax, platform.upper(), xmax, ymax, work_min)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("platform", choices=["gb300", "b300", "all"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--x-tpot", choices=["avg", "p50", "p90", "p99"], default="p90",
                    help="TPOT used for the x-axis: a percentile row of "
                         "benchmark_percentile.json (default p90) or the summary average")
    ap.add_argument("--title", default="GLM 5.3 FP8 Perf on SGLang")
    args = ap.parse_args()

    global X_TPOT
    X_TPOT = args.x_tpot
    platforms = ["b300", "gb300"] if args.platform == "all" else [args.platform]

    fig, axes = plt.subplots(1, len(platforms),
                             figsize=(6.1 * len(platforms), 4.7), dpi=200,
                             squeeze=False)
    for ax, platform in zip(axes[0], platforms):
        render_panel(ax, platform)

    title = args.title
    if len(platforms) == 1 and platforms[0].lower() not in title.lower():
        title += f" — {platforms[0].upper()}"
    fig.suptitle(title, fontsize=17, fontweight="bold")
    xlabel = ("Interactivity (tok/s/user)" if args.x_tpot == "avg" else
              f"{args.x_tpot} Interactivity (tok/s/user)")
    fig.supxlabel(xlabel, fontsize=12, y=0.11)
    fig.supylabel("Token Throughput per GPU (tok/s/gpu)", fontsize=12, x=0.015)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=max(len(labels), 1), frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.82, bottom=0.24, wspace=0.16)

    out = Path(args.out) if args.out else BASE / f"figure_{args.platform}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
