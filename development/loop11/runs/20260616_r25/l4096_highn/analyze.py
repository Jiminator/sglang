"""L4096 high-n analysis — PAIRED per-needle recall@2048, radix OFF vs ON.

Statistical unit = the NEEDLE (one NIAH trial). Layer/decode-step records within a
needle are NOT independent, so per-needle recall@2048 = fraction of that needle's
oracle records with recall_at_k["2048"] true. Off and on are matched by
request_id (paired). Reports:
  * per-needle recall OFF/ON and paired delta = off - on (pp),
  * mean / std / min / max of the paired delta,
  * 95% bootstrap CI on the mean delta (resampling needles),
  * cached_tokens distribution ON (proof of engagement) + OFF (proof control==0),
  * matched-population confirmation (id pairing).

Verdict (owner rule): INSIDE if |mean delta| <= 0.5pp (also report whether the CI
lies within +/-0.5pp); OUTSIDE otherwise. If the ON arm did not hit cache, the run
is INVALID (not a verdict).
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


def per_needle_recall(sink_path, length):
    """request_id -> {records, hits, pct} for the given length, plus failure markers."""
    by_id = defaultdict(lambda: [0, 0])  # rid -> [records, hits]
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
            if _length_of(rid) != length:
                continue
            by_id[rid][0] += 1
            if _recall_2048(rec):
                by_id[rid][1] += 1
    out = {}
    for rid, (n, hits) in by_id.items():
        out[rid] = {"records": n, "hits": hits, "pct": (100.0 * hits / n) if n else None}
    return out, dict(failures)


def load_cached(index_path):
    """request_id -> cached_tokens (server-reported)."""
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
            out[r["request_id"]] = r.get("cached_tokens")
    return out


def dist(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return {}
    n = len(vals)
    mean = sum(vals) / n
    return {
        "n": n,
        "min": vals[0],
        "max": vals[-1],
        "mean": round(mean, 2),
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
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sink-off", required=True)
    ap.add_argument("--sink-on", required=True)
    ap.add_argument("--index-off", required=True)
    ap.add_argument("--index-on", required=True)
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--max-delta-pp", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table-out", required=True)
    args = ap.parse_args()

    off, off_fail = per_needle_recall(args.sink_off, args.length)
    on, on_fail = per_needle_recall(args.sink_on, args.length)
    cached_off = load_cached(args.index_off)
    cached_on = load_cached(args.index_on)

    off_ids = set(off)
    on_ids = set(on)
    matched = sorted(off_ids & on_ids, key=lambda r: int(r.split("-i")[1]))

    # Cache engagement gates.
    cached_on_eligible = {k: v for k, v in cached_on.items() if _length_of(k) == args.length}
    cached_off_eligible = {k: v for k, v in cached_off.items() if _length_of(k) == args.length}
    on_dist = dist(list(cached_on_eligible.values()))
    off_dist = dist(list(cached_off_eligible.values()))
    on_engaged = on_dist.get("n_gt0", 0) > 0
    off_is_control = (off_dist.get("n_gt0", 0) == 0)

    # Paired per-needle deltas (off - on).
    rows = []
    deltas = []
    off_hits_tot = off_n_tot = on_hits_tot = on_n_tot = 0
    for rid in matched:
        o = off[rid]
        n_ = on[rid]
        d = None
        if o["pct"] is not None and n_["pct"] is not None:
            d = o["pct"] - n_["pct"]
            deltas.append(d)
        off_hits_tot += o["hits"]; off_n_tot += o["records"]
        on_hits_tot += n_["hits"]; on_n_tot += n_["records"]
        rows.append({
            "request_id": rid,
            "off_records": o["records"], "off_hits": o["hits"], "off_pct": round(o["pct"], 4) if o["pct"] is not None else None,
            "on_records": n_["records"], "on_hits": n_["hits"], "on_pct": round(n_["pct"], 4) if n_["pct"] is not None else None,
            "delta_off_minus_on_pp": round(d, 4) if d is not None else None,
            "cached_off": cached_off.get(rid),
            "cached_on": cached_on.get(rid),
        })

    n = len(deltas)
    mean = sum(deltas) / n if n else None
    if n > 1:
        var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
        std = var ** 0.5
    else:
        std = 0.0 if n == 1 else None
    lo, hi = bootstrap_ci(deltas)

    # Record-pooled recall (for cross-check with the prior n=20 verdict's 6240-sample pooling).
    pooled_off_pct = round(100.0 * off_hits_tot / off_n_tot, 4) if off_n_tot else None
    pooled_on_pct = round(100.0 * on_hits_tot / on_n_tot, 4) if on_n_tot else None
    pooled_delta = (round(pooled_off_pct - pooled_on_pct, 4)
                    if (pooled_off_pct is not None and pooled_on_pct is not None) else None)

    # Mean per-needle recall (unit = needle).
    mean_off_needle = round(sum(off[r]["pct"] for r in matched) / n, 4) if n else None
    mean_on_needle = round(sum(on[r]["pct"] for r in matched) / n, 4) if n else None

    off_hard = sum(off_fail.get(k, 0) for k in _HARD_FAILURES)
    on_hard = sum(on_fail.get(k, 0) for k in _HARD_FAILURES)

    invalid_reasons = []
    if not on_engaged:
        invalid_reasons.append("radix-ON arm did NOT hit cache (no cached_tokens>0 on eligible requests)")
    if not off_is_control:
        invalid_reasons.append("radix-OFF control had cached_tokens>0 (control not clean)")
    if off_hard or on_hard:
        invalid_reasons.append(f"hard-failure markers off={off_hard} on={on_hard}")
    if n == 0:
        invalid_reasons.append("no matched needles with recall data")

    if invalid_reasons:
        verdict = "INVALID"
    else:
        mean_inside = abs(mean) <= args.max_delta_pp
        ci_inside = (lo is not None and hi is not None and
                     abs(lo) <= args.max_delta_pp and abs(hi) <= args.max_delta_pp)
        verdict = "INSIDE" if mean_inside else "OUTSIDE"

    summary = {
        "probe": "L4096_highn_paired_recall_characterization",
        "length_words": args.length,
        "max_delta_pp": args.max_delta_pp,
        "n_needles_matched": n,
        "matched_population": {
            "off_ids": len(off_ids), "on_ids": len(on_ids), "matched": len(matched),
            "off_only": sorted(off_ids - on_ids), "on_only": sorted(on_ids - off_ids),
            "paired_by_request_id": True,
        },
        "cache_engagement": {
            "radix_on_cached_tokens_dist": on_dist,
            "radix_off_cached_tokens_dist": off_dist,
            "on_engaged": on_engaged,
            "off_is_clean_control": off_is_control,
        },
        "per_needle_recall_pct": {
            "mean_off": mean_off_needle,
            "mean_on": mean_on_needle,
            "unit": "needle (mean over matched needles of per-needle recall@2048)",
        },
        "paired_delta_off_minus_on_pp": {
            "mean": round(mean, 4) if mean is not None else None,
            "std": round(std, 4) if std is not None else None,
            "min": round(min(deltas), 4) if deltas else None,
            "max": round(max(deltas), 4) if deltas else None,
            "ci95_bootstrap": [round(lo, 4), round(hi, 4)] if lo is not None else None,
            "ci_iters": 20000,
        },
        "record_pooled_recall_pct": {
            "off": pooled_off_pct, "on": pooled_on_pct, "delta_off_minus_on_pp": pooled_delta,
            "off_records": off_n_tot, "on_records": on_n_tot,
            "note": "layer/decode-step pooling (matches prior n=20 verdict method); NOT the paired unit",
        },
        "failures": {"off": dict(off_fail), "on": dict(on_fail)},
        "verdict": verdict,
        "invalid_reasons": invalid_reasons,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(args.table_out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"[analyze] -> {args.out}")
    print(f"  matched needles: {n} (off_ids={len(off_ids)} on_ids={len(on_ids)})")
    print(f"  cached_tokens ON  dist: {on_dist}")
    print(f"  cached_tokens OFF dist: {off_dist}")
    print(f"  per-needle mean recall: off={mean_off_needle}%  on={mean_on_needle}%")
    if mean is not None:
        print(f"  paired delta (off-on): mean={mean:+.4f}pp std={std:.4f} "
              f"min={min(deltas):+.4f} max={max(deltas):+.4f} "
              f"CI95=[{lo:+.4f},{hi:+.4f}]")
    print(f"  record-pooled: off={pooled_off_pct}% on={pooled_on_pct}% delta={pooled_delta}pp")
    print(f"  VERDICT: {verdict}")
    for r in invalid_reasons:
        print(f"    - {r}")


if __name__ == "__main__":
    main()
