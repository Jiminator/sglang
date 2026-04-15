"""
Plot cumulative accuracy curves for each RMSNorm kernel variant.

Reads the per-sample JSONs from exps/mmlu_curves/ and produces:
- exps/mmlu_curves/failure_curves.png
- exps/mmlu_curves/failure_curves_divergence.png  (where variants differ)

Run: python exps/plot_mmlu_failure_curves.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURVES_DIR = Path(__file__).parent / "mmlu_curves"

# Variant display order / color / style
VARIANTS = [
    ("baseline_wrong",  "Baseline (unpatched sgl_kernel.rmsnorm)",      "tab:red",    "-"),
    ("jit_cta",         "JIT RMSNormHFKernel (CTA)",                    "tab:purple", "--"),
    ("jit_half",        "JIT RMSNormHFHalfKernel (half-block vectorized)", "tab:orange", "-."),
    ("forward_native",  "forward_native (Python HF semantics)",         "tab:green",  ":"),
    ("sgl_kernel_hf",   "sgl_kernel.rmsnorm_hf (AOT CUDA)",            "tab:blue",   "-"),
    ("jit_scalar",      "JIT RMSNormHFScalarKernel (scalar+reg cache)", "black",      "-"),
]


def load_variants():
    data = {}
    for v, _, _, _ in VARIANTS:
        p = CURVES_DIR / f"{v}.json"
        if not p.exists():
            print(f"[warn] missing {p}")
            continue
        with open(p) as f:
            data[v] = json.load(f)
    return data


def cumulative_accuracy(per_sample):
    arr = np.asarray(per_sample, dtype=np.float64)
    k = np.arange(1, len(arr) + 1)
    return np.cumsum(arr) / k


def plot_main(data):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for v, label, color, linestyle in VARIANTS:
        if v not in data:
            continue
        cum = cumulative_accuracy(data[v]["per_sample"])
        n = len(cum)
        ax.plot(
            np.arange(1, n + 1), cum,
            label=f"{label}  (final={data[v]['score']:.4f} = {int(round(data[v]['score']*n))}/{n})",
            color=color, linestyle=linestyle, linewidth=2.0, alpha=0.9,
        )

    ax.set_xlabel("Sample index (MMLU question, in submission order)", fontsize=11)
    ax.set_ylabel("Cumulative accuracy", fontsize=11)
    ax.set_title("MMLU Cumulative-Accuracy Failure Curves by RMSNorm Kernel Variant\n"
                 "(meta-llama/Llama-3.1-8B-Instruct + int4wo-128, 64 samples, seed=0)", fontsize=11)
    ax.set_xlim(1, len(cum))
    ax.set_ylim(0.55, 0.85)
    ax.grid(True, alpha=0.25)
    ax.axhline(y=0.7031, color="gray", linestyle=":", alpha=0.5, linewidth=1.0)
    ax.text(
        2, 0.7055, "0.7031 (45/64) — HF-correct ceiling",
        fontsize=9, color="gray",
    )
    ax.axhline(y=0.6562, color="gray", linestyle=":", alpha=0.5, linewidth=1.0)
    ax.text(
        2, 0.6586, "0.6562 (42/64) — unpatched baseline",
        fontsize=9, color="gray",
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    plt.tight_layout()
    out = CURVES_DIR / "failure_curves.png"
    plt.savefig(out, dpi=130)
    print(f"wrote {out}")
    plt.close(fig)


def plot_divergence(data):
    """Show only the indices where variants disagree on correctness."""
    if not data:
        return
    per_sample_matrix = np.array([data[v]["per_sample"] for v, _, _, _ in VARIANTS if v in data])
    # sample is a "divergence point" if not all variants agree
    divergence_mask = per_sample_matrix.std(axis=0) > 0
    div_indices = np.where(divergence_mask)[0] + 1  # 1-indexed for plotting

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))

    # Panel 1: running score difference vs jit_scalar reference
    if "jit_scalar" in data:
        ref = np.array(data["jit_scalar"]["per_sample"], dtype=np.float64)
        ref_cum = np.cumsum(ref) / np.arange(1, len(ref) + 1)
        for v, label, color, linestyle in VARIANTS:
            if v not in data:
                continue
            arr = np.array(data[v]["per_sample"], dtype=np.float64)
            cum = np.cumsum(arr) / np.arange(1, len(arr) + 1)
            delta = cum - ref_cum
            ax1.plot(
                np.arange(1, len(cum) + 1), delta,
                label=f"{label}", color=color, linestyle=linestyle, linewidth=2.0, alpha=0.9,
            )
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
        ax1.set_xlabel("Sample index", fontsize=11)
        ax1.set_ylabel(r"$\Delta$ cumulative accuracy (variant − jit_scalar)", fontsize=11)
        ax1.set_title("Cumulative accuracy deviation from jit_scalar (reference)", fontsize=11)
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # Panel 2: per-sample correctness markers for divergent samples only
    labels = [v for v, _, _, _ in VARIANTS if v in data]
    y_offsets = np.arange(len(labels)) * 0.9
    colors_map = {v: c for v, _, c, _ in VARIANTS}
    for i, v in enumerate(labels):
        arr = np.array(data[v]["per_sample"])
        correct_idx = div_indices[arr[div_indices - 1] == 1]
        wrong_idx = div_indices[arr[div_indices - 1] == 0]
        ax2.scatter(correct_idx, np.full_like(correct_idx, y_offsets[i], dtype=float),
                    marker="o", s=55, color=colors_map[v], label=None, edgecolors="black", linewidths=0.4)
        ax2.scatter(wrong_idx, np.full_like(wrong_idx, y_offsets[i], dtype=float),
                    marker="x", s=55, color=colors_map[v], label=None, linewidths=2)

    ax2.set_yticks(y_offsets)
    display_labels = [dict((v, label) for v, label, _, _ in VARIANTS)[v] for v in labels]
    ax2.set_yticklabels(display_labels, fontsize=9)
    ax2.set_xlabel("Sample index (only divergent questions shown)", fontsize=11)
    ax2.set_title(f"Divergent samples: {len(div_indices)} of {per_sample_matrix.shape[1]} questions\n"
                  f"(○ correct, ✕ wrong)", fontsize=11)
    ax2.grid(True, axis="x", alpha=0.25)
    ax2.set_xlim(0, per_sample_matrix.shape[1] + 1)

    plt.tight_layout()
    out = CURVES_DIR / "failure_curves_divergence.png"
    plt.savefig(out, dpi=130)
    print(f"wrote {out}")
    plt.close(fig)


def print_summary(data):
    print("\n=== Summary ===")
    for v, label, _, _ in VARIANTS:
        if v not in data:
            continue
        d = data[v]
        n = d["num_examples"]
        n_correct = int(round(d["score"] * n))
        print(f"  {v:20s}  {n_correct}/{n}  ({d['score']:.4f})")

    # Find divergent sample indices
    if len(data) >= 2:
        per_sample_matrix = np.array([data[v]["per_sample"] for v, _, _, _ in VARIANTS if v in data])
        divergence_mask = per_sample_matrix.std(axis=0) > 0
        n_div = int(divergence_mask.sum())
        print(f"\n  Divergent samples (at least one variant disagrees): {n_div}")
        for idx in np.where(divergence_mask)[0]:
            row = per_sample_matrix[:, idx]
            labels = [v for v, _, _, _ in VARIANTS if v in data]
            per_var = ", ".join(f"{l}={int(s)}" for l, s in zip(labels, row))
            print(f"    sample #{idx+1}: {per_var}")


if __name__ == "__main__":
    data = load_variants()
    if not data:
        raise SystemExit("No data loaded")
    plot_main(data)
    plot_divergence(data)
    print_summary(data)
