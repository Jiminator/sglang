"""Loop-9 recall gate — aggregate the NIAH oracle sink into recall@2048 and gate it.

Reads the JSONL sink written by the server-side recall oracle during a
``niah_oracle_sweep.py`` run (per-(trial, layer, decode-step) SUCCESS records),
aggregates ``recall_at_k[2048]`` per length and overall, and either freezes the
result (baseline mode) or gates a candidate against a frozen baseline:
the overall recall@2048 and every per-length recall@2048 must move by at most
``--max-delta-pp`` (default 0.5) percentage points. Fail-closed: missing sink,
zero samples for an issued length, hard failure markers, or a gate breach all
exit non-zero.

Each sample is one (trial, layer, decode-step) score-only measurement on the
live all-reduced score tensor; ``selected_contains_needle`` (the decode-true
selection outcome) is aggregated alongside as a cross-check.
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


def _recall_2048(rec) -> bool:
    rk = rec.get("recall_at_k", {})
    if "2048" in rk:
        return bool(rk["2048"])
    if 2048 in rk:
        return bool(rk[2048])
    # Fall back to the rank rule the payload derives recall from.
    return int(rec.get("needle_worst_rank", 1 << 30)) < 2048


def summarize(sink_path: str) -> dict:
    if not os.path.exists(sink_path):
        raise SystemExit(f"[recall-gate] sink not found: {sink_path}")
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
            length = _length_of(rec.get("request_id", ""))
            if length is not None:
                by_len[length].append(rec)
    out = {
        "sink": sink_path,
        "failure_markers": dict(failures),
        "lengths": {},
    }
    all_hits = 0
    all_n = 0
    sel_hits = 0
    sel_n = 0
    for length in sorted(by_len):
        recs = by_len[length]
        hits = sum(1 for r in recs if _recall_2048(r))
        trials = sorted({r.get("request_id") for r in recs})
        entry = {
            "samples": len(recs),
            "trials": len(trials),
            "recall_at_2048_hits": hits,
            "recall_at_2048_pct": round(100.0 * hits / len(recs), 3),
        }
        sel = [r for r in recs if "selected_contains_needle" in r]
        if sel:
            sh = sum(1 for r in sel if r["selected_contains_needle"])
            entry["selected_contains_needle_pct"] = round(100.0 * sh / len(sel), 3)
            sel_hits += sh
            sel_n += len(sel)
        out["lengths"][str(length)] = entry
        all_hits += hits
        all_n += len(recs)
    if all_n == 0:
        raise SystemExit("[recall-gate] FAIL-CLOSED: zero success samples in sink")
    out["overall"] = {
        "samples": all_n,
        "recall_at_2048_hits": all_hits,
        "recall_at_2048_pct": round(100.0 * all_hits / all_n, 3),
    }
    if sel_n:
        out["overall"]["selected_contains_needle_pct"] = round(
            100.0 * sel_hits / sel_n, 3
        )
    hard = sum(failures[k] for k in _HARD_FAILURES)
    if hard:
        raise SystemExit(
            f"[recall-gate] FAIL-CLOSED: {hard} hard failure marker(s): "
            f"{ {k: failures[k] for k in _HARD_FAILURES if failures[k]} }"
        )
    return out


def gate(candidate: dict, baseline: dict, max_delta_pp: float) -> list:
    problems = []
    c_overall = candidate["overall"]["recall_at_2048_pct"]
    b_overall = baseline["overall"]["recall_at_2048_pct"]
    if abs(c_overall - b_overall) > max_delta_pp:
        problems.append(
            f"overall recall@2048 moved {c_overall - b_overall:+.3f}pp "
            f"({b_overall} -> {c_overall}), bound ±{max_delta_pp}pp"
        )
    b_lengths = baseline.get("lengths", {})
    c_lengths = candidate.get("lengths", {})
    if sorted(b_lengths) != sorted(c_lengths):
        problems.append(
            f"length sets differ: baseline {sorted(b_lengths)} vs "
            f"candidate {sorted(c_lengths)} (workload not fixed)"
        )
    for length in sorted(b_lengths):
        if length not in c_lengths:
            continue
        b = b_lengths[length]["recall_at_2048_pct"]
        c = c_lengths[length]["recall_at_2048_pct"]
        if abs(c - b) > max_delta_pp:
            problems.append(
                f"length {length}: recall@2048 moved {c - b:+.3f}pp "
                f"({b} -> {c}), bound ±{max_delta_pp}pp"
            )
        if b_lengths[length]["samples"] != c_lengths[length]["samples"]:
            problems.append(
                f"length {length}: sample counts differ "
                f"({b_lengths[length]['samples']} vs {c_lengths[length]['samples']}) "
                "— workload/measurement not comparable"
            )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sink", default=None, help="oracle sink JSONL (default: module path)")
    ap.add_argument("--out", required=True, help="write the summary JSON here")
    ap.add_argument(
        "--baseline",
        default=None,
        help="frozen baseline summary JSON; when set, enforce the recall gate",
    )
    ap.add_argument("--max-delta-pp", type=float, default=0.5)
    args = ap.parse_args()

    sink_path = args.sink
    if not sink_path:
        from sglang.srt.layers.attention.double_sparsity import (
            oracle_artifact_sink as s,
        )

        sink_path = s.default_sink_path()

    summary = summarize(sink_path)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.baseline:
        with open(args.baseline, "r", encoding="utf-8") as fh:
            baseline = json.load(fh)
        problems = gate(summary, baseline, float(args.max_delta_pp))
        summary["gate"] = {
            "baseline": args.baseline,
            "max_delta_pp": float(args.max_delta_pp),
            "verdict": "PASS" if not problems else "FAIL",
            "problems": problems,
        }
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[recall-gate] summary -> {args.out}")
        if problems:
            print(f"[recall-gate] FAIL ({len(problems)} problem(s)):")
            for p in problems:
                print(f"  - {p}")
            return 1
        print(
            f"[recall-gate] PASS: overall recall@2048 "
            f"{summary['overall']['recall_at_2048_pct']}% within "
            f"±{args.max_delta_pp}pp of baseline"
        )
        return 0

    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(
        f"[recall-gate] baseline frozen -> {args.out} "
        f"(overall recall@2048 {summary['overall']['recall_at_2048_pct']}% over "
        f"{summary['overall']['samples']} samples)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
