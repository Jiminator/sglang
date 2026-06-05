#!/usr/bin/env python3
"""Rebuild sweep_table.md with the re-baselined official metric (client TPS).

Preserves each row's tag / changed-knobs / max_total_num_tokens / rationale from the
existing table, and recomputes the metrics (client_TPS, median_itl, mean_tpot, etc.)
and target_met (client_TPS >= 30 AND p99_ttft < 22000) freshly from the result JSONLs.
"""
import json
import os
import re

import parse_result as P  # reuse client_tps_from_record + constants

DIR = os.path.dirname(__file__)
TABLE = os.path.join(DIR, "sweep_table.md")
RESULTS = os.path.join(DIR, "results")


def load(tag):
    f = os.path.join(RESULTS, f"{tag}_isl4096_osl512_c{P.CONCURRENCY}.jsonl")
    if not os.path.exists(f):
        return None
    return json.loads([l for l in open(f) if l.strip()][-1])


def main():
    rows = []
    for line in open(TABLE):
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cells[0] in ("tag", "") or set(cells[0]) <= set("-: "):
            continue  # header / separator
        # old format col0=tag col1=knobs ... col13=max_total col15=rationale
        tag, knobs = cells[0], cells[1]
        max_total = cells[13] if len(cells) > 13 else "?"
        rationale = cells[-1]
        rows.append((tag, knobs, max_total, rationale))

    out = ["# GLM-5.1-FP8 flags-only hill-climb — sweep table\n",
           "\n",
           "OFFICIAL metric (owner re-baseline R2): **client_TPS = Sigma tokens / Sigma decode_time** "
           "(= total_output_tokens / (total_latency - TTFT)); target >= 30. "
           "median_itl_ms is a speculation-inflated cross-check only. target_met = client_TPS>=30 AND p99_ttft<22000.\n",
           "\n",
           P.HEADER]
    for tag, knobs, max_total, rationale in rows:
        r = load(tag)
        if r is None:
            continue
        tps = P.client_tps_from_record(r)
        mitl = P.fnum(r.get("median_itl_ms"))
        mtpot = P.fnum(r.get("mean_tpot_ms"))
        ot = P.fnum(r.get("output_throughput"))
        thr = ot / P.CONCURRENCY if ot == ot else float("nan")
        p99t = P.fnum(r.get("p99_ttft_ms"))
        p99i = P.fnum(r.get("p99_itl_ms"))
        acc = r.get("accept_length")
        acc_s = f"{acc:.3f}" if isinstance(acc, (int, float)) else str(acc)
        conc = P.fnum(r.get("concurrency"))
        mc = r.get("max_concurrent_requests")
        comp = r.get("completed")
        errs = r.get("errors", [])
        ec = sum(1 for e in errs if e) if isinstance(errs, list) else "n/a"
        met = (tps >= P.TPS_TARGET) and (p99t < P.TTFT_TARGET_MS)
        out.append(
            f"| {tag} | {knobs} | {tps:.2f} | {mitl:.2f} | {mtpot:.2f} | {thr:.2f} | "
            f"{p99t:.1f} | {p99i:.2f} | {acc_s} | {conc:.1f} | {mc} | {comp} | {ec} | "
            f"{max_total} | {met} | {rationale} |\n"
        )
    with open(TABLE, "w") as f:
        f.writelines(out)
    print(f"Regenerated {TABLE} with {len(rows)} rows (official metric = client_TPS).")


if __name__ == "__main__":
    main()
