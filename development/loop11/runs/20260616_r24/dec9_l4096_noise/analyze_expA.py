"""ExpA analysis — per-seed off/on recall@2048 per length, paired (off-on) delta.

Reuses the SAME recall extraction as R24 P2 (p2_recall_compare.py): the top-level
recall_at_k["2048"] of each oracle record IS the table-free absorbed selection's
recall@2048. Each (seed, radix) sink is aggregated per length; for each seed we
compute the paired off-on delta pp; then mean/min/max/std across seeds, plus the
higher-n point estimate. NOISE-vs-REAL verdict per the spec, fail-closed on any
missing sink / hard-failure marker.
"""
from __future__ import annotations
import json, os, sys, statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "recall")
SEEDS = [20260607, 20260608, 20260609, 20260610]
_HARD = ("span_out_of_range", "exception")


def _length_of(rid):
    try:
        return int(str(rid).split("-")[0].lstrip("L"))
    except (ValueError, AttributeError):
        return None


def _recall_2048(p):
    rk = p.get("recall_at_k", {})
    if "2048" in rk:
        return bool(rk["2048"])
    if 2048 in rk:
        return bool(rk[2048])
    return int(p.get("needle_worst_rank", 1 << 30)) < 2048


def aggregate(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing sink: {path}")
    by_len = defaultdict(list)
    failures = defaultdict(int)
    with open(path) as fh:
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
            L = _length_of(rec.get("request_id", ""))
            if L is None:
                continue
            by_len[L].append(rec)
    per = {}
    for L in sorted(by_len):
        recs = by_len[L]
        hits = sum(1 for r in recs if _recall_2048(r))
        n = len(recs)
        per[L] = {"samples": n, "hits": hits, "pct": round(100.0 * hits / n, 4) if n else None}
    hard = sum(failures.get(k, 0) for k in _HARD)
    return per, dict(failures), hard


def main():
    result = {"probe": "expA_l4096_radix_noise", "seeds": SEEDS, "max_delta_pp": 0.5}
    # per-seed
    per_seed = {}
    deltas = {4096: [], 16384: []}
    problems = []
    for S in SEEDS:
        off_p, off_f, off_h = aggregate(os.path.join(OUT, f"sink_s{S}_off.jsonl"))
        on_p, on_f, on_h = aggregate(os.path.join(OUT, f"sink_s{S}_on.jsonl"))
        if off_h:
            problems.append(f"seed {S} OFF hard-failures {off_f}")
        if on_h:
            problems.append(f"seed {S} ON hard-failures {on_f}")
        seed_rec = {}
        for L in (4096, 16384):
            o = off_p.get(L, {}).get("pct")
            n = on_p.get(L, {}).get("pct")
            on_samp = on_p.get(L, {}).get("samples")
            off_samp = off_p.get(L, {}).get("samples")
            d = None if (o is None or n is None) else round(o - n, 4)  # off - on
            seed_rec[L] = {
                "off_pct": o, "on_pct": n, "off_minus_on_pp": d,
                "samples_off": off_samp, "samples_on": on_samp,
            }
            if d is not None:
                deltas[L].append(d)
            if off_samp != on_samp:
                problems.append(f"seed {S} L{L}: off/on sample mismatch {off_samp}/{on_samp}")
        per_seed[str(S)] = seed_rec
    result["per_seed"] = per_seed

    # cross-seed stats of off-minus-on delta
    stats = {}
    for L in (4096, 16384):
        ds = deltas[L]
        stats[str(L)] = {
            "n_seeds": len(ds),
            "deltas_off_minus_on_pp": ds,
            "mean": round(statistics.mean(ds), 4) if ds else None,
            "min": round(min(ds), 4) if ds else None,
            "max": round(max(ds), 4) if ds else None,
            "std": round(statistics.pstdev(ds), 4) if len(ds) > 1 else 0.0,
            "straddles_zero": (min(ds) < 0 < max(ds)) if ds else None,
        }
    result["cross_seed_delta_stats"] = stats

    # higher-n point estimate (L4096 num=80, seed 20260607)
    hi = {}
    hoff_p, _, hoff_h = aggregate(os.path.join(OUT, "sink_hi_off.jsonl"))
    hon_p, _, hon_h = aggregate(os.path.join(OUT, "sink_hi_on.jsonl"))
    o = hoff_p.get(4096, {}).get("pct")
    n = hon_p.get(4096, {}).get("pct")
    hi = {
        "length": 4096,
        "off_pct": o, "on_pct": n,
        "off_minus_on_pp": None if (o is None or n is None) else round(o - n, 4),
        "samples_off": hoff_p.get(4096, {}).get("samples"),
        "samples_on": hon_p.get(4096, {}).get("samples"),
    }
    if hoff_h or hon_h:
        problems.append("higher-n run hard-failures")
    result["higher_n_L4096"] = hi

    # verdicts
    def verdict_for(L, hi_delta=None):
        s = stats[str(L)]
        if s["mean"] is None:
            return "INCONCLUSIVE", ["no deltas"]
        notes = []
        mean_ok = abs(s["mean"]) <= 0.5
        straddle = s["straddles_zero"]
        consistently_neg = all(d > 0 for d in s["deltas_off_minus_on_pp"])  # off>on i.e. on lower
        # NOISE if delta straddles 0 / |mean| well within bound AND (for L4096) higher-n within bound
        hi_ok = True
        if hi_delta is not None:
            hi_ok = abs(hi_delta) <= 0.5
            notes.append(f"higher-n off-minus-on={hi_delta:+.4f}pp within={hi_ok}")
        notes.append(f"mean={s['mean']:+.4f}pp |mean|<=0.5={mean_ok} straddles0={straddle} all-on-lower={consistently_neg}")
        if (straddle or mean_ok) and hi_ok and not consistently_neg:
            return "NOISE", notes
        if consistently_neg and not straddle and (hi_delta is not None and hi_delta > 0.5):
            return "REAL", notes
        # ambiguous middle
        if mean_ok and hi_ok:
            return "NOISE", notes
        return "REAL", notes

    v4096, n4096 = verdict_for(4096, hi.get("off_minus_on_pp"))
    v16384, n16384 = verdict_for(16384)
    result["verdict_L4096"] = {"verdict": v4096, "notes": n4096}
    result["verdict_L16384"] = {"verdict": v16384, "notes": n16384}
    result["problems"] = problems

    out_path = os.path.join(HERE, "expA_verdict.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    # console table
    print(f"\n=== ExpA per-seed off/on recall@2048 (off-minus-on pp) ===")
    for L in (4096, 16384):
        print(f"\nLength {L}:")
        print(f"  {'seed':>10} {'off%':>9} {'on%':>9} {'off-on(pp)':>11} {'n(off/on)':>11}")
        for S in SEEDS:
            r = per_seed[str(S)][L]
            print(f"  {S:>10} {str(r['off_pct']):>9} {str(r['on_pct']):>9} "
                  f"{str(r['off_minus_on_pp']):>11} {r['samples_off']}/{r['samples_on']}")
        s = stats[str(L)]
        print(f"  cross-seed off-on: mean={s['mean']} min={s['min']} max={s['max']} "
              f"std={s['std']} straddles0={s['straddles_zero']}")
    print(f"\nHigher-n L4096 num=80 seed 20260607: off={hi['off_pct']}% on={hi['on_pct']}% "
          f"off-on={hi['off_minus_on_pp']}pp n(off/on)={hi['samples_off']}/{hi['samples_on']}")
    print(f"\nVERDICT L4096: {v4096}  {n4096}")
    print(f"VERDICT L16384: {v16384}  {n16384}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
