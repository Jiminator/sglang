"""Loop-11 M2 — absorbed-latent recall summary + gate.

Post-processes the NIAH oracle sink for BOTH the table-path payload (top-level
``recall_at_k`` / ``needle_worst_rank``) and the side-by-side absorbed-latent
payload (``rec["absorbed"]``), aggregating recall@2048 per length + overall.
The table summary reproduces oracle_recall_summary.py exactly (sanity check vs
the frozen baseline); the absorbed summary is the new artifact this loop validates.

Gate (fail-closed): the ABSORBED overall + every per-length recall@2048 must be
within --max-delta-pp (default 0.5) of the frozen table-path baseline. Missing
sink, zero samples, hard failure markers, or any record missing rec["absorbed"]
(when present at all) all surface explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_HARD_FAILURES = ("span_out_of_range", "exception")


def _length_of(request_id: str):
    try:
        return int(request_id.split("-")[0].lstrip("L"))
    except (ValueError, AttributeError):
        return None


def _recall_2048(payload) -> bool:
    rk = payload.get("recall_at_k", {})
    if "2048" in rk:
        return bool(rk["2048"])
    if 2048 in rk:
        return bool(rk[2048])
    return int(payload.get("needle_worst_rank", 1 << 30)) < 2048


def _summarize(by_len, key) -> dict:
    """Aggregate recall@2048 for either the top-level payload (key=None) or the
    rec["absorbed"] sub-dict (key="absorbed")."""
    out = {"lengths": {}}
    all_hits = all_n = 0
    for length in sorted(by_len):
        recs = by_len[length]
        payloads = [r if key is None else r.get(key) for r in recs]
        payloads = [p for p in payloads if p is not None]
        hits = sum(1 for p in payloads if _recall_2048(p))
        out["lengths"][str(length)] = {
            "samples": len(payloads),
            "recall_at_2048_hits": hits,
            "recall_at_2048_pct": (
                round(100.0 * hits / len(payloads), 3) if payloads else None
            ),
        }
        all_hits += hits
        all_n += len(payloads)
    if all_n == 0:
        raise SystemExit(f"[absorbed-gate] FAIL-CLOSED: zero samples for key={key}")
    out["overall"] = {
        "samples": all_n,
        "recall_at_2048_hits": all_hits,
        "recall_at_2048_pct": round(100.0 * all_hits / all_n, 3),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sink", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", required=True, help="frozen table-path baseline JSON")
    ap.add_argument("--max-delta-pp", type=float, default=0.5)
    args = ap.parse_args()

    if not os.path.exists(args.sink):
        raise SystemExit(f"[absorbed-gate] sink not found: {args.sink}")

    by_len = defaultdict(list)
    failures = defaultdict(int)
    n_records = n_absorbed = 0
    with open(args.sink, "r", encoding="utf-8") as fh:
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
            length = _length_of(rec.get("request_id", ""))
            if length is None:
                continue
            n_records += 1
            if "absorbed" in rec:
                n_absorbed += 1
            by_len[length].append(rec)

    hard = sum(failures[k] for k in _HARD_FAILURES)
    table = _summarize(by_len, key=None)
    absorbed = _summarize(by_len, key="absorbed")

    with open(args.baseline, "r", encoding="utf-8") as fh:
        baseline = json.load(fh)

    problems = []
    if hard:
        problems.append(
            f"{hard} hard failure marker(s): "
            f"{ {k: failures[k] for k in _HARD_FAILURES if failures[k]} }"
        )
    if n_absorbed != n_records:
        problems.append(
            f"{n_records - n_absorbed}/{n_records} records MISSING rec['absorbed'] "
            "(absorbed wiring did not emit on every sample)"
        )

    b_overall = baseline["overall"]["recall_at_2048_pct"]
    a_overall = absorbed["overall"]["recall_at_2048_pct"]
    if abs(a_overall - b_overall) > args.max_delta_pp:
        problems.append(
            f"absorbed overall recall@2048 {a_overall} vs baseline {b_overall} "
            f"= {a_overall - b_overall:+.3f}pp, bound ±{args.max_delta_pp}pp"
        )
    b_lengths = baseline.get("lengths", {})
    for length in sorted(b_lengths):
        if length not in absorbed["lengths"]:
            problems.append(f"length {length} missing from absorbed summary")
            continue
        b = b_lengths[length]["recall_at_2048_pct"]
        a = absorbed["lengths"][length]["recall_at_2048_pct"]
        if a is None or abs(a - b) > args.max_delta_pp:
            problems.append(
                f"length {length}: absorbed {a} vs baseline {b} "
                f"= {(a - b):+.3f}pp, bound ±{args.max_delta_pp}pp"
                if a is not None
                else f"length {length}: absorbed recall None"
            )

    summary = {
        "sink": args.sink,
        "n_records": n_records,
        "n_with_absorbed": n_absorbed,
        "failure_markers": dict(failures),
        "table": table,
        "absorbed": absorbed,
        "baseline": {
            "path": args.baseline,
            "overall_pct": b_overall,
            "lengths_pct": {k: v["recall_at_2048_pct"] for k, v in b_lengths.items()},
        },
        "gate": {
            "max_delta_pp": args.max_delta_pp,
            "verdict": "PASS" if not problems else "FAIL",
            "problems": problems,
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[absorbed-gate] summary -> {args.out}")
    print(
        f"  records={n_records}  with_absorbed={n_absorbed}  failures={dict(failures)}"
    )
    print(f"  TABLE    overall recall@2048 = {table['overall']['recall_at_2048_pct']}%")
    print(f"  ABSORBED overall recall@2048 = {a_overall}%  (baseline {b_overall}%)")
    for length in sorted(b_lengths):
        a = absorbed["lengths"].get(length, {}).get("recall_at_2048_pct")
        t = table["lengths"].get(length, {}).get("recall_at_2048_pct")
        b = b_lengths[length]["recall_at_2048_pct"]
        print(f"    L{length:>6}: table={t}  absorbed={a}  baseline={b}")
    print(f"  VERDICT: {summary['gate']['verdict']}")
    for p in problems:
        print(f"    - {p}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
