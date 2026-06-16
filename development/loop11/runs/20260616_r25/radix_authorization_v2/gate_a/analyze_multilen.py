"""GATE A multi-length analysis — PAIRED per-needle recall@2048, radix OFF vs ON.

Generalizes the proven single-length analyzer
(development/loop11/runs/20260616_r25/l4096_highn/analyze.py) to several lengths.
Statistical unit = the NEEDLE (one NIAH trial). Per-needle recall@2048 = fraction of
that needle's oracle (layer, decode-step) records with recall_at_k["2048"] true. Off
and on are matched by request_id (paired) within each length.

Per length reports: recall OFF/ON (mean over matched needles), mean paired delta
(off - on, pp), std, 95% bootstrap CI on the mean, the cached_tokens distribution ON
(proof of engagement) and OFF (proof control == 0), and matched-population counts.
Also an OVERALL (pooled-needle) delta + CI.

Verdict (owner GATE A rule): PASS iff |mean delta| <= 0.5pp OVERALL and for EVERY
length, applied on the MEAN (CI reported for transparency). A length whose radix-ON
arm did NOT hit cache (no cached_tokens>0) is INVALID — not a pass; the overall
verdict is FAIL if any measured length is INVALID or breaches the mean bar.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

_HARD_FAILURES = ("span_out_of_range", "exception")


def _length_of(request_id):
    try:
        return int(str(request_id).split("-")[0].lstrip("L"))
    except (ValueError, AttributeError):
        return None


def _recall_2048(payload) -> bool:
    rk = payload.get("recall_at_k", {})
    if "2048" in rk:
        return bool(rk["2048"])
    if 2048 in rk:
        return bool(rk[2048])
    return int(payload.get("needle_worst_rank", 1 << 30)) < 2048


def per_needle_recall_all(sink_path):
    """request_id -> {records, hits, pct}; plus failure markers (all lengths)."""
    by_id = defaultdict(lambda: [0, 0])
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
            by_id[rid][0] += 1
            if _recall_2048(rec):
                by_id[rid][1] += 1
    out = {}
    for rid, (n, hits) in by_id.items():
        out[rid] = {"records": n, "hits": hits, "pct": (100.0 * hits / n) if n else None}
    return out, dict(failures)


def load_cached(index_path):
    """request_id -> cached_tokens (MEASURED requests only; warmup excluded)."""
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
            if r.get("warmup"):
                continue  # warmup pre-seeds the prefix; not a measured request
            out[r["request_id"]] = r.get("cached_tokens")
    return out


def dist(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return {}
    n = len(vals)
    return {
        "n": n,
        "min": vals[0],
        "max": vals[-1],
        "mean": round(sum(vals) / n, 2),
        "median": vals[n // 2],
        "n_zero": sum(1 for v in vals if v == 0),
        "n_gt0": sum(1 for v in vals if v > 0),
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


def analyze_one_length(off, on, cached_off, cached_on, length, max_delta):
    off_ids = {r for r in off if _length_of(r) == length}
    on_ids = {r for r in on if _length_of(r) == length}
    matched = sorted(off_ids & on_ids, key=lambda r: int(r.split("-i")[1]))

    coff = {k: v for k, v in cached_off.items() if _length_of(k) == length}
    con = {k: v for k, v in cached_on.items() if _length_of(k) == length}
    on_dist = dist(list(con.values()))
    off_dist = dist(list(coff.values()))
    on_engaged = on_dist.get("n_gt0", 0) > 0
    off_is_control = off_dist.get("n_gt0", 0) == 0

    deltas = []
    for rid in matched:
        o, n_ = off[rid]["pct"], on[rid]["pct"]
        if o is not None and n_ is not None:
            deltas.append(o - n_)
    n = len(deltas)
    mean = (sum(deltas) / n) if n else None
    std = ((sum((d - mean) ** 2 for d in deltas) / (n - 1)) ** 0.5) if n > 1 else (0.0 if n == 1 else None)
    lo, hi = bootstrap_ci(deltas)
    mean_off = round(sum(off[r]["pct"] for r in matched) / n, 4) if n else None
    mean_on = round(sum(on[r]["pct"] for r in matched) / n, 4) if n else None

    invalid = []
    if not on_engaged:
        invalid.append("radix-ON arm did NOT hit cache for this length (no cached_tokens>0)")
    if not off_is_control:
        invalid.append("radix-OFF control had cached_tokens>0 (control not clean)")
    if n == 0:
        invalid.append("no matched needles with recall data")

    within = (mean is not None and abs(mean) <= max_delta)
    if invalid:
        verdict = "INVALID"
    else:
        verdict = "WITHIN" if within else "BREACH"

    return {
        "length": length,
        "n_needles_matched": n,
        "matched_off_ids": len(off_ids),
        "matched_on_ids": len(on_ids),
        "recall_off_pct": mean_off,
        "recall_on_pct": mean_on,
        "paired_delta_off_minus_on_pp": {
            "mean": round(mean, 4) if mean is not None else None,
            "std": round(std, 4) if std is not None else None,
            "min": round(min(deltas), 4) if deltas else None,
            "max": round(max(deltas), 4) if deltas else None,
            "ci95_bootstrap": [round(lo, 4), round(hi, 4)] if lo is not None else None,
        },
        "cache_engagement": {
            "radix_on_cached_tokens_dist": on_dist,
            "radix_off_cached_tokens_dist": off_dist,
            "on_engaged": on_engaged,
            "off_is_clean_control": off_is_control,
        },
        "within_bound_on_mean": within,
        "verdict": verdict,
        "invalid_reasons": invalid,
    }, deltas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sink-off", required=True)
    ap.add_argument("--sink-on", required=True)
    ap.add_argument("--index-off", required=True)
    ap.add_argument("--index-on", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384])
    ap.add_argument("--max-delta-pp", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    off, off_fail = per_needle_recall_all(args.sink_off)
    on, on_fail = per_needle_recall_all(args.sink_on)
    cached_off = load_cached(args.index_off)
    cached_on = load_cached(args.index_on)

    per_length = {}
    all_deltas = []
    any_invalid = False
    any_breach = False
    for L in args.lengths:
        res, deltas = analyze_one_length(off, on, cached_off, cached_on, L, args.max_delta_pp)
        per_length[str(L)] = res
        all_deltas.extend(deltas)
        if res["verdict"] == "INVALID":
            any_invalid = True
        if res["verdict"] == "BREACH":
            any_breach = True

    # OVERALL pooled-needle delta.
    n = len(all_deltas)
    omean = (sum(all_deltas) / n) if n else None
    ostd = ((sum((d - omean) ** 2 for d in all_deltas) / (n - 1)) ** 0.5) if n > 1 else None
    olo, ohi = bootstrap_ci(all_deltas)
    overall_within = (omean is not None and abs(omean) <= args.max_delta_pp)

    off_hard = sum(off_fail.get(k, 0) for k in _HARD_FAILURES)
    on_hard = sum(on_fail.get(k, 0) for k in _HARD_FAILURES)

    if any_invalid or off_hard or on_hard or not all_deltas:
        verdict = "FAIL"
    elif any_breach or not overall_within:
        verdict = "FAIL"
    else:
        verdict = "PASS"

    summary = {
        "probe": "GATE_A_recall_equivalence_multilen",
        "lengths": args.lengths,
        "max_delta_pp": args.max_delta_pp,
        "rule": "PASS iff |mean paired delta| <= max_delta_pp OVERALL and for EVERY "
                "length (on the MEAN); every length's radix-ON arm must hit cache "
                "(else INVALID). CI reported for transparency.",
        "per_length": per_length,
        "overall": {
            "n_needles": n,
            "mean_delta_off_minus_on_pp": round(omean, 4) if omean is not None else None,
            "std": round(ostd, 4) if ostd is not None else None,
            "ci95_bootstrap": [round(olo, 4), round(ohi, 4)] if olo is not None else None,
            "within_bound_on_mean": overall_within,
        },
        "hard_failures": {"off": dict(off_fail), "on": dict(on_fail)},
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[gateA] -> {args.out}")
    for L in args.lengths:
        r = per_length[str(L)]
        d = r["paired_delta_off_minus_on_pp"]
        cd = r["cache_engagement"]
        print(f"  L{L:>5}: off={r['recall_off_pct']}% on={r['recall_on_pct']}% "
              f"delta(off-on)={d['mean']:+}pp std={d['std']} CI95={d['ci95_bootstrap']} "
              f"n={r['n_needles_matched']} cachedON={cd['radix_on_cached_tokens_dist']} "
              f"cachedOFF_n_gt0={cd['radix_off_cached_tokens_dist'].get('n_gt0')} "
              f"-> {r['verdict']}")
    print(f"  OVERALL: delta={summary['overall']['mean_delta_off_minus_on_pp']:+}pp "
          f"CI95={summary['overall']['ci95_bootstrap']} n={n}")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
