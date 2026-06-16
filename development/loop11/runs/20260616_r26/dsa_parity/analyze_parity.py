"""DSA-native radix-reuse parity analyzer — FALLBACK metric.

Pairs reuse vs fresh by needle idx. Reports:
  * per-arm answer-recall (mean %), cached_tokens distribution (engagement proof),
  * paired answer-recall delta (reuse - fresh) in pp + 95% bootstrap CI over needles,
  * exact greedy-output divergence rate (needles whose reuse text != fresh text),
  * the parity verdict vs DS's +1.57pp [1.28,1.87] and the +-0.5pp edge bar.

Statistical unit = the NEEDLE. Cache engagement is PROVEN from raw cached_tokens:
fresh arm must be all-0, reuse arm must be >0 at the heavy (~4288) level.
"""

from __future__ import annotations

import argparse
import json
import os
import random


def load_arm(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["idx"]] = r
    return out


def dist(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return {}
    n = len(vals)
    return {
        "n": n, "min": vals[0], "max": vals[-1],
        "mean": round(sum(vals) / n, 1), "median": vals[n // 2],
        "n_zero": sum(1 for v in vals if v == 0),
        "n_gt0": sum(1 for v in vals if v > 0),
        "n_ge4000": sum(1 for v in vals if v >= 4000),
    }


def bootstrap_ci(deltas, iters=20000, seed=12345):
    if not deltas:
        return None, None
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--max-delta-pp", type=float, default=0.5)
    ap.add_argument("--min-needles", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table-out", required=True)
    args = ap.parse_args()

    fresh = load_arm(os.path.join(args.dir, "index_fresh.jsonl"))
    reuse = load_arm(os.path.join(args.dir, "index_reuse.jsonl"))

    fresh_cached = dist([r.get("cached_tokens") for r in fresh.values()])
    reuse_cached = dist([r.get("cached_tokens") for r in reuse.values()])

    problems = []
    if fresh_cached.get("n_gt0", 0) != 0:
        problems.append(f"fresh control NOT clean: cached_tokens>0 on {fresh_cached['n_gt0']} reqs")
    if reuse_cached.get("n_gt0", 0) == 0:
        problems.append("reuse: NO cache hit (cached_tokens all 0) — heavy reuse not exercised")
    # HEAVY level: must match DS R26 partial arm (~4288). Require all reuse reqs >=4000.
    elif reuse_cached.get("n_ge4000", 0) != reuse_cached.get("n", 0):
        problems.append(
            f"reuse: only {reuse_cached.get('n_ge4000', 0)}/{reuse_cached.get('n', 0)} reqs at "
            f"the HEAVY ~4288 level (cached>=4000); reuse not at DS's heavy-reuse level"
        )

    idxs = sorted(set(fresh) & set(reuse))
    rows = []
    deltas = []          # per-needle answer_hit delta (reuse - fresh), in {-1,0,+1}*100
    diverge = 0
    n_both_present = 0
    fresh_hits = reuse_hits = 0
    for i in idxs:
        f, r = fresh[i], reuse[i]
        fh_hit = 1 if f.get("answer_hit") else 0
        rh_hit = 1 if r.get("answer_hit") else 0
        fresh_hits += fh_hit
        reuse_hits += rh_hit
        d_pp = (rh_hit - fh_hit) * 100.0
        deltas.append(d_pp)
        text_match = (f.get("output_text") == r.get("output_text"))
        if not text_match:
            diverge += 1
        n_both_present += 1
        rows.append({
            "idx": i,
            "needle": f.get("needle"),
            "fresh_hit": bool(fh_hit), "reuse_hit": bool(rh_hit),
            "answer_delta_pp": d_pp,
            "exact_output_match": text_match,
            "fresh_cached": f.get("cached_tokens"),
            "reuse_cached": r.get("cached_tokens"),
            "fresh_text": f.get("output_text"),
            "reuse_text": r.get("output_text"),
        })

    n = len(deltas)
    mean = sum(deltas) / n if n else None
    if n > 1:
        var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
        std = var ** 0.5
    else:
        std = 0.0 if n == 1 else None
    lo, hi = bootstrap_ci(deltas)

    fresh_recall = round(100.0 * fresh_hits / n, 4) if n else None
    reuse_recall = round(100.0 * reuse_hits / n, 4) if n else None
    diverge_rate = round(100.0 * diverge / n_both_present, 4) if n_both_present else None

    if n < args.min_needles:
        problems.append(f"only {n} matched needles (< robust n {args.min_needles})")

    saturated = (fresh_recall == 100.0 and reuse_recall == 100.0)

    within_bar = (mean is not None and abs(mean) <= args.max_delta_pp)
    # DS reference: +1.57pp, 95% CI [1.28, 1.87]; edge bar +-0.5pp.
    ds_delta_pp = 1.57
    ds_ci = [1.28, 1.87]

    if within_bar and not saturated:
        verdict = ("DSA stays within +-0.5pp under heavy reuse (NOT saturated) "
                   "while DS is +1.57pp -> deviation is DS-SPECIFIC")
    elif within_bar and saturated:
        verdict = ("DSA answer-recall delta within +-0.5pp BUT the metric SATURATED "
                   "(both arms 100%) -> answer-recall cannot discriminate; "
                   "see exact_output_divergence_rate as the sharper selection signal")
    else:
        verdict = ("DSA answer-recall deviates beyond +-0.5pp under heavy reuse "
                   "(comparable to DS) -> GENERAL radix-reuse property")

    out = {
        "probe": "DSA_native_radix_reuse_parity_fallback",
        "metric": "FALLBACK end-to-end greedy NIAH answer-recall (reuse - fresh), paired by needle",
        "primary_metric_infeasible_reason": (
            "enable_return_indexer_topk returns PHYSICAL kv-slot indices not logical "
            "positions; cross-arm logical needle selection-recall not computable client-"
            "only under radix reuse (arbitrary logical->physical remap). 77/144 needles "
            "exceed the row-2047 reconstruction ceiling. No new production code allowed."
        ),
        "length": 4096, "page_size": 64, "index_topk": 2048,
        "max_delta_pp": args.max_delta_pp, "min_needles": args.min_needles,
        "n_needles_matched": n,
        "cache_engagement": {
            "fresh_cached_dist": fresh_cached,
            "reuse_cached_dist": reuse_cached,
            "fresh_is_clean_control": fresh_cached.get("n_gt0", 0) == 0,
            "reuse_engaged_heavy": reuse_cached.get("n_gt0", 0) > 0,
        },
        "answer_recall": {
            "fresh_recall_pct": fresh_recall,
            "reuse_recall_pct": reuse_recall,
            "paired_delta_reuse_minus_fresh_pp": {
                "mean": round(mean, 4) if mean is not None else None,
                "std": round(std, 4) if std is not None else None,
                "min": round(min(deltas), 4) if deltas else None,
                "max": round(max(deltas), 4) if deltas else None,
                "ci95_bootstrap": [round(lo, 4), round(hi, 4)] if lo is not None else None,
                "ci_iters": 20000,
            },
            "saturated_both_100pct": saturated,
            "within_pm0_5pp_bar": within_bar,
        },
        "exact_output_divergence": {
            "n_needles": n_both_present,
            "n_divergent": diverge,
            "divergence_rate_pct": diverge_rate,
            "note": ("temp=0 greedy: divergent text = selection/numeric difference "
                     "fresh vs reuse on the SAME needle. 0% = bit-identical selection."),
        },
        "ds_reference": {
            "ds_selection_recall_delta_pp": ds_delta_pp,
            "ds_ci95": ds_ci,
            "edge_bar_pp": args.max_delta_pp,
        },
        "verdict": verdict,
        "status": "VALID" if not problems else "INVALID",
        "problems": problems,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    with open(args.table_out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"[analyze_parity] -> {args.out}")
    print(f"  fresh cached dist : {fresh_cached}")
    print(f"  reuse cached dist : {reuse_cached}")
    print(f"  n_matched={n}")
    print(f"  fresh_recall={fresh_recall}%  reuse_recall={reuse_recall}%")
    d = out["answer_recall"]["paired_delta_reuse_minus_fresh_pp"]
    print(f"  paired answer-recall delta mean={d['mean']}pp CI95={d['ci95_bootstrap']} "
          f"within+-0.5pp={within_bar} saturated={saturated}")
    print(f"  exact output divergence: {diverge}/{n_both_present} = {diverge_rate}%")
    print(f"  VERDICT: {verdict}")
    print(f"  STATUS: {out['status']}")
    for p in problems:
        print(f"    - {p}")
    return 0 if not problems else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
