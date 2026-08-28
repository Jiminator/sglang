#!/usr/bin/env python3
"""Plot the GLM-5.3 B300 Pareto figure from results/ produced by the run scripts.

Usage:
    python3 plot_figure.py b300 [--curves glm53_fp8,glm53_nvfp4] [--out figure.png]

One figure, two subplots — TP=8 and TEP=8 — each with one curve per model
config (FP8 and NVFP4), read from
results/b300/<curve>/<panel>/parallel_*/benchmark_{summary,percentile}.json.

X = p90 interactivity = 1000 / p90 TPOT(ms), where per-request
TPOT = (latency - TTFT) / (output_tokens - 1); override with --x-tpot.
Y = total throughput (prompt+completion tok/s, as evalscope reports) / 8 GPUs;
one point per concurrency 1,2,4,8. The y-axis is truncated (break glyph,
'0' at the origin).

The built-in curves carry the figure's styling; any additional
results/b300/<name>/ directory is auto-discovered and drawn with a fallback
colour, so a new config appears just by adding its results.
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

CURVES = [
    ("FP8", "glm53_fp8", "#A43223", "-", True),
    ("NVFP4", "glm53_nvfp4", "#2E6E8E", "-", False),
]

PANELS = (("TP=8", "tp8"), ("TEP=8", "tep8"))
N_GPUS = 8
Y_FLOOR = 5000

FALLBACK_COLORS = ["#4C8C4A", "#7E5AA2", "#B58900", "#5A5A5A"]


def curve_style(name, k):
    for c in CURVES:
        if c[1] == name:
            return c
    return (name.replace("_", " "), name,
            FALLBACK_COLORS[k % len(FALLBACK_COLORS)], "-", k % 2 == 0)


def discover_curves():
    known = [c[1] for c in CURVES]
    extra = set()
    base = BASE / "results" / "b300"
    if base.is_dir():
        for d in base.iterdir():
            if (d.is_dir() and d.name not in known and any(
                    any((d / pdir).glob("parallel_*/benchmark_summary.json"))
                    for _, pdir in PANELS)):
                extra.add(d.name)
    return known + sorted(extra)


def tpot_ms(summary_file, summary):
    if X_TPOT == "avg":
        return summary["TPOT (ms)"]
    rows = json.loads((summary_file.parent / "benchmark_percentile.json").read_text())
    want = X_TPOT.replace("p", "") + "%"
    return next(r["TPOT (ms)"] for r in rows if r["Percentiles"] == want)


def read_points(curve_dir, panel_dir):
    base = BASE / "results" / "b300" / curve_dir / panel_dir
    pts = []
    for f in sorted(base.glob("parallel_*/benchmark_summary.json")):
        s = json.loads(f.read_text())
        conc = int(s["Concurrency"])
        if conc not in CONCS:
            continue
        tpot = tpot_ms(f, s)
        pts.append((1000.0 / tpot if tpot else 0.0,
                    s["Total Throughput (tok/s)"] / N_GPUS, conc))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("platform", choices=["b300"])
    ap.add_argument("--curves", default=None,
                    help="comma-separated curve dir names (default: the built-in "
                         "curves plus any extra results/b300/<name>/ dirs)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--x-tpot", choices=["avg", "p50", "p90", "p99"], default="p90")
    ap.add_argument("--title", default="GLM 5.3 Perf on SGLang, 8xB300")
    args = ap.parse_args()

    global X_TPOT
    X_TPOT = args.x_tpot
    names = args.curves.split(",") if args.curves else discover_curves()
    curves = [curve_style(n, k) for k, n in enumerate(names)]

    panels = [(t, p) for t, p in PANELS
              if any(read_points(cdir, p) for _, cdir, *_ in curves)]
    if not panels:
        raise SystemExit("no data under results/b300/ — run the sweeps first")

    fig, axes = plt.subplots(1, len(panels), figsize=(6.1 * len(panels), 4.7),
                             dpi=200, squeeze=False)
    row_pts = [pt for _, cdir, *_ in curves for _, pdir in panels
               for pt in read_points(cdir, pdir)]
    xmax = max(p[0] for p in row_pts)
    ymax = max(p[1] for p in row_pts)
    for ax, (title, pdir) in zip(axes[0], panels):
        for label, cdir, color, ls, above in curves:
            plot_curve(ax, read_points(cdir, pdir),
                       color=color, linestyle=ls, above=above, label=label)
        style_panel(ax, title, xmax, ymax, Y_FLOOR)

    fig.suptitle(args.title, fontsize=17, fontweight="bold")
    xlabel = ("Interactivity (tok/s/user)" if args.x_tpot == "avg" else
              f"{args.x_tpot} Interactivity (tok/s/user)")
    fig.supxlabel(xlabel, fontsize=12, y=0.11)
    fig.supylabel("Token Throughput per GPU (tok/s/gpu)", fontsize=12, x=0.015)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=max(len(labels), 1), frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.24, wspace=0.16)

    out = Path(args.out) if args.out else BASE / "figure_b300.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
