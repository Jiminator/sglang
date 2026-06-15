"""P2 (CORRECTED) — fail-closed per-length recall@2048: radix-OFF vs radix-ON.

Apples-to-apples: both sinks are produced by the SAME NIAH oracle sweep
(same seed / same trial set), once with radix-OFF and once with radix-ON. If
cold/warm selection is identical (P1), recall must be identical here.

This helper reads each sink's TOP-LEVEL recall payload (on this post-deletion HEAD
the top-level `recall_at_k` IS the table-free absorbed selection's recall — there is
NO separate rec["absorbed"]). It enforces, FAIL-CLOSED (exit non-zero on ANY breach):
  * length-set equality between off and on;
  * per-length sample-count equality;
  * ZERO hard-failure markers (span_out_of_range / exception) in either sink;
  * OVERALL recall@2048 |on-off| <= max-delta-pp;
  * EVERY per-length bin |on-off| <= max-delta-pp.

The verdict is the REAL computed result. A per-length bin that genuinely diverges
beyond the bound is reported as FAIL with the numbers — never relabeled "noise".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_HARD_FAILURES = ("span_out_of_range", "exception")


def _length_of(request_id):
    try:
        return int(str(request_id).split("-")[0].lstrip("L"))
    except (ValueError, AttributeError):
        return None


def _recall_2048(payload) -> bool:
    """Top-level recall@2048 for one oracle record (the table-free absorbed selection)."""
    rk = payload.get("recall_at_k", {})
    if "2048" in rk:
        return bool(rk["2048"])
    if 2048 in rk:
        return bool(rk[2048])
    return int(payload.get("needle_worst_rank", 1 << 30)) < 2048


def _aggregate(sink_path):
    """Return (per_length: {L: {samples, hits, pct}}, overall: {...}, failures: {marker: n})."""
    if not os.path.exists(sink_path):
        raise SystemExit(f"[recall-compare] sink not found: {sink_path}")
    by_len = defaultdict(list)
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
            L = _length_of(rec.get("request_id", ""))
            if L is None:
                continue
            by_len[L].append(rec)
    per_length = {}
    all_hits = all_n = 0
    for L in sorted(by_len):
        recs = by_len[L]
        hits = sum(1 for r in recs if _recall_2048(r))
        n = len(recs)
        per_length[L] = {
            "samples": n,
            "hits": hits,
            "pct": round(100.0 * hits / n, 3) if n else None,
        }
        all_hits += hits
        all_n += n
    overall = {
        "samples": all_n,
        "hits": all_hits,
        "pct": round(100.0 * all_hits / all_n, 3) if all_n else None,
    }
    return per_length, overall, dict(failures)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sink-off", required=True, help="radix-OFF oracle sink jsonl")
    ap.add_argument("--sink-on", required=True, help="radix-ON oracle sink jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-delta-pp", type=float, default=0.5)
    args = ap.parse_args()

    off_len, off_overall, off_fail = _aggregate(args.sink_off)
    on_len, on_overall, on_fail = _aggregate(args.sink_on)

    problems = []

    # Hard-failure markers in either sink.
    off_hard = sum(off_fail.get(k, 0) for k in _HARD_FAILURES)
    on_hard = sum(on_fail.get(k, 0) for k in _HARD_FAILURES)
    if off_hard:
        problems.append(f"radix-OFF has {off_hard} hard-failure marker(s): "
                        f"{ {k: off_fail[k] for k in _HARD_FAILURES if off_fail.get(k)} }")
    if on_hard:
        problems.append(f"radix-ON has {on_hard} hard-failure marker(s): "
                        f"{ {k: on_fail[k] for k in _HARD_FAILURES if on_fail.get(k)} }")

    # Length-set equality.
    if sorted(off_len) != sorted(on_len):
        problems.append(
            f"length sets differ: off={sorted(off_len)} on={sorted(on_len)}"
        )

    # Per-length sample-count equality + per-length delta.
    table = {}
    common = sorted(set(off_len) & set(on_len))
    for L in common:
        off_n = off_len[L]["samples"]
        on_n = on_len[L]["samples"]
        off_pct = off_len[L]["pct"]
        on_pct = on_len[L]["pct"]
        delta = (None if (off_pct is None or on_pct is None)
                 else round(on_pct - off_pct, 3))
        bin_ok = (delta is not None and abs(delta) <= args.max_delta_pp)
        table[str(L)] = {
            "samples_off": off_n,
            "samples_on": on_n,
            "radix_off_pct": off_pct,
            "radix_on_pct": on_pct,
            "delta_pp": delta,
            "within_bound": bool(bin_ok),
        }
        if off_n != on_n:
            problems.append(
                f"length {L}: sample counts differ (off={off_n} on={on_n}) — not comparable"
            )
        if delta is None:
            problems.append(f"length {L}: recall pct is None on one side")
        elif abs(delta) > args.max_delta_pp:
            problems.append(
                f"length {L}: radix on-off delta {delta:+.3f}pp exceeds ±{args.max_delta_pp}pp "
                f"(off={off_pct}% on={on_pct}%)"
            )

    # Overall delta.
    overall_delta = None
    if off_overall["pct"] is not None and on_overall["pct"] is not None:
        overall_delta = round(on_overall["pct"] - off_overall["pct"], 3)
        if abs(overall_delta) > args.max_delta_pp:
            problems.append(
                f"OVERALL radix on-off delta {overall_delta:+.3f}pp exceeds ±{args.max_delta_pp}pp "
                f"(off={off_overall['pct']}% on={on_overall['pct']}%)"
            )
    else:
        problems.append("overall recall pct is None on one side")

    # Empty-sink guard (fail-closed: no evidence is not a pass).
    if off_overall["samples"] == 0 or on_overall["samples"] == 0:
        problems.append(
            f"empty sink(s): off_samples={off_overall['samples']} on_samples={on_overall['samples']}"
        )

    verdict = "PASS" if not problems else "FAIL"
    summary = {
        "probe": "P2_recall_radix_off_vs_on_corrected",
        "max_delta_pp": args.max_delta_pp,
        "sink_off": args.sink_off,
        "sink_on": args.sink_on,
        "radix_off_failures": off_fail,
        "radix_on_failures": on_fail,
        "per_length": table,
        "overall": {
            "radix_off_pct": off_overall["pct"],
            "radix_on_pct": on_overall["pct"],
            "delta_pp": overall_delta,
            "samples_off": off_overall["samples"],
            "samples_on": on_overall["samples"],
        },
        "verdict": verdict,
        "problems": problems,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[recall-compare] -> {args.out}")
    print(f"  OVERALL  off={off_overall['pct']}%  on={on_overall['pct']}%  delta={overall_delta}pp")
    for L in common:
        t = table[str(L)]
        print(f"    L{L:>6}: off={t['radix_off_pct']}%  on={t['radix_on_pct']}%  "
              f"delta={t['delta_pp']}pp  samples(off/on)={t['samples_off']}/{t['samples_on']}  "
              f"within={t['within_bound']}")
    print(f"  VERDICT: {verdict}")
    for p in problems:
        print(f"    - {p}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
