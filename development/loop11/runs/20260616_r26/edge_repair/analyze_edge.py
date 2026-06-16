"""R26 GATE C edge analyzer — ROBUST-N paired per-needle recall@2048 for each edge case.

Statistical unit = the NEEDLE. For each arm (cold / boundary / partial / evict) the
per-needle recall@2048 is the fraction of that needle's oracle records with
recall_at_k["2048"] true. Each on-arm case is paired by needle index to the cold/fresh
control. Reports per case: paired delta = case - cold (pp), mean/std/min/max, 95%
bootstrap CI on the mean delta, AND the cached_tokens distribution (proof of engagement).

Contract (the original ±0.5pp-clean edge contract, robust n):
  * boundary (full page-aligned reuse): |mean paired delta vs cold| <= 0.5pp,
    cached>0 AND a page multiple (engaged page-aligned reuse).
  * partial (mid-page divergence): |mean paired delta vs cold| <= 0.5pp, cached>0
    (aligned pages reused + partial page recomputed).
  * eviction/recompute: cached fell to 0 (prefix actually evicted), recall == cold
    (mean paired delta ~0; deterministic recompute == cold = no stale slot). We require
    |mean delta| <= 0.5pp AND that the cached distribution is all-zero.
Cache engagement is PROVEN from raw per-request cached_tokens (reuse arm >0, cold/evict
arm ==0). If any on-arm did NOT engage, the run is INVALID for that arm (not a pass).

status = PASS iff all three within their bound AND eviction cached fell to 0 AND no
arm INVALID; problems lists the breaching case(s). The FAIL-CLOSED writer reads this
top-level status/problems verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict


def _recall_2048(payload) -> bool:
    rk = payload.get("recall_at_k", {})
    if "2048" in rk:
        return bool(rk["2048"])
    if 2048 in rk:
        return bool(rk[2048])
    return int(payload.get("needle_worst_rank", 1 << 30)) < 2048


def _idx_of(request_id):
    # request_id like "L4096-bnd-i37" -> 37
    try:
        return int(str(request_id).rsplit("-i", 1)[1])
    except (ValueError, IndexError):
        return None


def per_needle_recall(sink_path, prefix):
    """For request_ids starting with `prefix` (e.g. 'L4096-bnd-i'), return
    {idx -> {records, hits, pct}} plus hard-failure markers."""
    by_idx = defaultdict(lambda: [0, 0])
    failures = defaultdict(int)
    with open(sink_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "failure" in rec:
                failures[str(rec["failure"]).split(":")[0]] += 1
                continue
            rid = rec.get("request_id", "")
            if not str(rid).startswith(prefix):
                continue
            i = _idx_of(rid)
            if i is None:
                continue
            by_idx[i][0] += 1
            if _recall_2048(rec):
                by_idx[i][1] += 1
    out = {}
    for i, (n, hits) in by_idx.items():
        out[i] = {"records": n, "hits": hits, "pct": (100.0 * hits / n) if n else None}
    return out, dict(failures)


def load_cached(index_path):
    """request_id -> cached_tokens (server-reported), keyed by idx."""
    out = {}
    if not os.path.exists(index_path):
        return out
    with open(index_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            i = _idx_of(r.get("request_id", ""))
            if i is not None:
                out[i] = r.get("cached_tokens")
    return out


def dist(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return {}
    n = len(vals)
    mean = sum(vals) / n
    return {
        "n": n, "min": vals[0], "max": vals[-1], "mean": round(mean, 2),
        "median": vals[n // 2],
        "n_zero": sum(1 for v in vals if v == 0),
        "n_gt0": sum(1 for v in vals if v > 0),
        "n_page_multiple_gt0": sum(1 for v in vals if v > 0 and v % 64 == 0),
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


def paired(cold, case, cached_case, max_delta_pp):
    """Per-needle paired delta = case - cold over needles present in BOTH."""
    idxs = sorted(set(cold) & set(case))
    rows, deltas = [], []
    for i in idxs:
        c, k = cold[i], case[i]
        d = None
        if c["pct"] is not None and k["pct"] is not None:
            d = k["pct"] - c["pct"]
            deltas.append(d)
        rows.append({
            "idx": i,
            "cold_pct": round(c["pct"], 4) if c["pct"] is not None else None,
            "case_pct": round(k["pct"], 4) if k["pct"] is not None else None,
            "delta_case_minus_cold_pp": round(d, 4) if d is not None else None,
            "cached_case": cached_case.get(i),
        })
    n = len(deltas)
    mean = sum(deltas) / n if n else None
    if n > 1:
        var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
        std = var ** 0.5
    else:
        std = 0.0 if n == 1 else None
    lo, hi = bootstrap_ci(deltas)
    mean_case = round(sum(case[i]["pct"] for i in idxs) / n, 4) if n else None
    mean_cold = round(sum(cold[i]["pct"] for i in idxs) / n, 4) if n else None
    return {
        "n_needles_matched": n,
        "recall_cold_pct_mean": mean_cold,
        "recall_case_pct_mean": mean_case,
        "paired_delta_case_minus_cold_pp": {
            "mean": round(mean, 4) if mean is not None else None,
            "std": round(std, 4) if std is not None else None,
            "min": round(min(deltas), 4) if deltas else None,
            "max": round(max(deltas), 4) if deltas else None,
            "ci95_bootstrap": [round(lo, 4), round(hi, 4)] if lo is not None else None,
            "ci_iters": 20000,
        },
        "within_bound_on_mean": (mean is not None and abs(mean) <= max_delta_pp),
    }, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="dir holding sink_<arm>.jsonl + index_<arm>.jsonl")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--max-delta-pp", type=float, default=0.5)
    ap.add_argument("--min-needles", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table-out", required=True)
    args = ap.parse_args()

    L = args.length
    D = args.dir
    arms = {
        "cold": f"L{L}-cold-i",
        "boundary": f"L{L}-bnd-i",
        "partial": f"L{L}-prt-i",
        "evict": f"L{L}-evc-i",
    }
    recall, fails, cached = {}, {}, {}
    for a, pref in arms.items():
        recall[a], fails[a] = per_needle_recall(os.path.join(D, f"sink_{a}.jsonl"), pref)
        cached[a] = load_cached(os.path.join(D, f"index_{a}.jsonl"))

    problems = []

    # Cache-engagement proof per arm (raw cached_tokens).
    cold_dist = dist(list(cached["cold"].values()))
    bnd_dist = dist(list(cached["boundary"].values()))
    prt_dist = dist(list(cached["partial"].values()))
    evc_dist = dist(list(cached["evict"].values()))

    # cold control MUST be clean (radix-OFF, all 0).
    if cold_dist.get("n_gt0", 0) != 0:
        problems.append(f"cold control not clean: cached_tokens>0 on {cold_dist['n_gt0']} reqs")
    # boundary MUST engage page-aligned reuse.
    if bnd_dist.get("n_gt0", 0) == 0:
        problems.append("boundary: NO cache hit (cached_tokens all 0) — page-aligned reuse not exercised")
    elif bnd_dist.get("n_gt0", 0) != bnd_dist.get("n_page_multiple_gt0", 0):
        problems.append(
            f"boundary: {bnd_dist['n_gt0'] - bnd_dist.get('n_page_multiple_gt0', 0)} reuse hits "
            "not a page multiple (expected full page-aligned reuse)"
        )
    # partial MUST engage (aligned pages reused).
    if prt_dist.get("n_gt0", 0) == 0:
        problems.append("partial: NO cache hit (cached_tokens all 0) — partial-page reuse not exercised")
    # eviction MUST have fallen to 0 (prefix actually evicted, recompute).
    if evc_dist.get("n_gt0", 0) != 0:
        problems.append(
            f"eviction: cache did NOT fall — cached_tokens>0 on {evc_dist['n_gt0']} reqs after flush_cache"
        )

    # Hard-failure markers.
    for a in arms:
        hard = sum(fails[a].get(k, 0) for k in ("span_out_of_range", "exception"))
        if hard:
            problems.append(f"{a}: {hard} hard-failure oracle marker(s)")

    # Robust-n: each on-arm must have >= min_needles paired with cold.
    cases = {}
    tables = {}
    for a in ("boundary", "partial", "evict"):
        stats, rows = paired(recall["cold"], recall[a], cached[a], args.max_delta_pp)
        cases[a] = stats
        tables[a] = rows
        if stats["n_needles_matched"] < args.min_needles:
            problems.append(
                f"{a}: only {stats['n_needles_matched']} matched needles (< robust n {args.min_needles})"
            )
        # boundary/partial: |mean delta| <= 0.5pp. eviction: same bound (recompute==cold).
        md = stats["paired_delta_case_minus_cold_pp"]["mean"]
        if md is None:
            problems.append(f"{a}: no paired recall data (mean delta None)")
        elif abs(md) > args.max_delta_pp:
            label = "recompute != cold (stale slot?)" if a == "evict" else "possible stale-slot corruption"
            problems.append(
                f"{a}: mean paired recall delta {md:+.4f}pp from cold (>±{args.max_delta_pp}pp) — {label}"
            )

    verdict = {
        "probe": "GATE_C_edge_robust_n_page64",
        "length": L,
        "page_size": args.page_size,
        "top_k": 2048,
        "max_delta_pp": args.max_delta_pp,
        "min_needles": args.min_needles,
        "rule": ("PASS iff boundary & partial mean paired recall delta vs cold within "
                 "±0.5pp AND eviction recompute == cold (mean delta within ±0.5pp) with "
                 "cached fallen to 0; cache engagement PROVEN per arm from raw cached_tokens. "
                 "Statistical unit = needle; robust n >= 128 matched."),
        "cache_engagement": {
            "cold_control_cached_dist": cold_dist,
            "boundary_cached_dist": bnd_dist,
            "partial_cached_dist": prt_dist,
            "eviction_cached_dist": evc_dist,
            "cold_is_clean_control": cold_dist.get("n_gt0", 0) == 0,
            "boundary_engaged_page_aligned": (
                bnd_dist.get("n_gt0", 0) > 0
                and bnd_dist.get("n_gt0", 0) == bnd_dist.get("n_page_multiple_gt0", 0)
            ),
            "partial_engaged": prt_dist.get("n_gt0", 0) > 0,
            "eviction_fell_to_zero": evc_dist.get("n_gt0", 0) == 0,
        },
        "cases": {
            "a_boundary_aligned_full_reuse": cases["boundary"],
            "b_partial_page_hit": cases["partial"],
            "c_eviction_recompute": cases["evict"],
        },
        "failures": {a: dict(fails[a]) for a in arms},
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(verdict, fh, indent=2)
    with open(args.table_out, "w") as fh:
        for a in ("boundary", "partial", "evict"):
            for r in tables[a]:
                r2 = dict(r)
                r2["case"] = a
                fh.write(json.dumps(r2) + "\n")

    print(f"[analyze_edge] -> {args.out}")
    print(f"  cold cached dist : {cold_dist}")
    print(f"  bnd  cached dist : {bnd_dist}")
    print(f"  prt  cached dist : {prt_dist}")
    print(f"  evc  cached dist : {evc_dist}")
    for a in ("boundary", "partial", "evict"):
        s = cases[a]
        d = s["paired_delta_case_minus_cold_pp"]
        print(f"  {a:9s}: n={s['n_needles_matched']} cold={s['recall_cold_pct_mean']}% "
              f"case={s['recall_case_pct_mean']}% mean_delta={d['mean']}pp "
              f"CI95={d['ci95_bootstrap']} within={s['within_bound_on_mean']}")
    print(f"  STATUS: {verdict['status']}")
    for p in problems:
        print(f"    - {p}")
    return 0 if not problems else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
