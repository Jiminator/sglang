#!/usr/bin/env python3
"""Plot blog-style Pareto curves from results/ produced by the run scripts.

Usage:
    python3 plot_figure.py gb300 [--curves day0,glm52_v0515] [--out figure.png]
    python3 plot_figure.py b300
    python3 plot_figure.py all          # 2x2 (both platforms) when both were run

Reads results/<platform>/<curve>/<panel>/parallel_*/benchmark_summary.json and
plots whatever exists. X = interactivity = 1000/TPOT(ms); Y = total throughput
(prompt+completion tok/s, as evalscope reports) / n_gpus; one point per
concurrency 1,2,4,8. The y-axis is truncated (break glyph, '0' at the origin).
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

CURVES = [
    ("GLM-5.2 day-0", "day0", "#9C948A", "--", False),
    ("GLM-5.2 v0.5.15", "glm52_v0515", "#A43223", "-", True),
    ("GLM-5.1 v0.5.15", "glm51_v0515", "#D45927", "-", False),
]

# platform -> ((panel title, panel dir), ...), n_gpus, y-axis floor
PLATFORMS = {
    "gb300": ((("TP=4", "tp4"), ("TEP=4", "tep4")), 4, 10000),
    "b300": ((("TP=8", "tp8"), ("TEP=8", "tep8")), 8, 5000),
}


def read_points(platform, curve_dir, panel_dir, n_gpus):
    base = BASE / "results" / platform / curve_dir / panel_dir
    pts = []
    for f in sorted(base.glob("parallel_*/benchmark_summary.json")):
        s = json.loads(f.read_text())
        conc = int(s["Concurrency"])
        if conc not in CONCS:
            continue
        tpot = s["TPOT (ms)"]
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


def render_row(axes_row, platform, curves):
    panels, n_gpus, work_min = PLATFORMS[platform]
    row_pts = [p for label, cdir, *_ in curves
               for _, pdir in panels
               for p in read_points(platform, cdir, pdir, n_gpus)]
    if not row_pts:
        raise SystemExit(f"no data under results/{platform}/ for curves "
                         f"{[c[1] for c in curves]} — run the sweeps first")
    xmax = max(p[0] for p in row_pts)
    ymax = max(p[1] for p in row_pts)
    for ax, (title, pdir) in zip(axes_row, panels):
        for label, cdir, color, ls, above in curves:
            plot_curve(ax, read_points(platform, cdir, pdir, n_gpus),
                       color=color, linestyle=ls, above=above, label=label)
        style_panel(ax, title, xmax, ymax, work_min)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("platform", choices=["gb300", "b300", "all"])
    ap.add_argument("--curves", default="day0,glm52_v0515,glm51_v0515",
                    help="comma-separated curve names to plot")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    wanted = args.curves.split(",")
    curves = [c for c in CURVES if c[1] in wanted]
    platforms = ["gb300", "b300"] if args.platform == "all" else [args.platform]

    nrows = len(platforms)
    fig, axes = plt.subplots(nrows, 2, figsize=(12.2, 4.7 * nrows), dpi=200,
                             squeeze=False)
    for row, platform in enumerate(platforms):
        render_row(axes[row], platform, curves)

    title = "GLM NVFP4 Perf on SGLang"
    if nrows == 1:
        title += f" — {platforms[0].upper()}"
    fig.suptitle(title, fontsize=17, fontweight="bold")
    fig.supxlabel("Interactivity (tok/s/user)", fontsize=12, y=0.11 / nrows)
    fig.supylabel("Token Throughput per GPU (tok/s/gpu)", fontsize=12, x=0.015)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=len(curves), frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.85 if nrows == 1 else 0.91,
                        bottom=0.24 / nrows, wspace=0.16, hspace=0.40)
    if nrows > 1:
        for row, platform in enumerate(platforms):
            y = axes[row][0].get_position().y1
            fig.text(0.5, y + 0.038 / nrows, platform.upper(),
                     ha="center", va="bottom", fontsize=14, fontweight="bold",
                     color="#333333")

    out = Path(args.out) if args.out else BASE / f"figure_{args.platform}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
