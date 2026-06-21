#!/usr/bin/env python3
"""AC-8 self-check — refuse AC-8 completion unless the final evidence package is complete and cited.

Fail-closed gate over the committed package: the AC-8 root-cause writeup (`ROOT_CAUSE.md`) may only stand if
(1) every AC-4 core arm has a non-blank batched AND serial cell in its per-arm ledger JSON, (2)
`run_meta.json` records the artifact-backed selected-vs-total + the recall-oracle + the captured-row
materialized-K corroboration summaries, and (3) `ROOT_CAUSE.md` actually CITES each required artifact and
contains the final serial table + the verdict + the no-fix boundary. Nonzero exit listing every miss; this
is the script the loop runs to assert the diagnosis is fully packaged (NOT a fix).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVID = os.path.join(HERE, "evidence")
ROOT_CAUSE = os.path.join(HERE, "ROOT_CAUSE.md")
RUN_META = os.path.join(EVID, "meta", "run_meta.json")

CORE_ARMS = ("dsa", "dsa_noradix", "production_ds", "ref_faithful", "ref_cosine")
REQUIRED_CITATIONS = (
    "forced_all_assertions.json",                       # AC-2.1 (H3 on the bitmap)
    "ac2_4_recall_oracle.json",                         # AC-2.4 (recall-oracle)
    "ac3_1_materialized_k_selected_index_equality.json",  # AC-3.1 (captured-row materialized-K)
    "ac4_garbage_counters.json",                        # AC-4 (garbage counters)
    "ac4_selected_vs_total.json",                       # AC-4 (selected-vs-total)
    "evidence_table.md",                                # AC-4 (per-arm table)
    "ac6_bisection_matrix.json",                        # AC-6 (bisection matrix)
    "gate_ac5.md",                                      # AC-5 (gate)
)


def main():
    problems = []

    # (1) every AC-4 core arm: non-blank batched AND serial cells.
    for arm in CORE_ARMS:
        p = os.path.join(EVID, "meta", "arms", f"{arm}.json")
        if not os.path.exists(p):
            problems.append(f"per-arm JSON missing: {arm}")
            continue
        s = json.load(open(p)).get("scores", {})
        for cell in ("dense_batched", "sparse_batched", "dense_serial", "sparse_serial"):
            if s.get(cell) is None:
                problems.append(f"{arm}: {cell} is blank (AC-4 table incomplete)")

    # (2) run_meta records the artifact-backed corroboration summaries.
    if not os.path.exists(RUN_META):
        problems.append("run_meta.json missing")
    else:
        rm = json.load(open(RUN_META))
        svt = rm.get("selected_vs_total", {})
        if svt.get("artifact") != "evidence/ac4_selected_vs_total.json":
            problems.append(f"run_meta.selected_vs_total.artifact={svt.get('artifact')!r}, "
                            f"expected 'evidence/ac4_selected_vs_total.json'")
        for arm in ("production_ds", "ref_faithful", "ref_cosine"):
            if arm not in (svt.get("core_arms") or {}):
                problems.append(f"run_meta.selected_vs_total missing core arm {arm}")
        for key in ("recall_oracle_corroboration", "materialized_k_captured_row_equality"):
            v = rm.get(key) or {}
            if not v.get("artifact") or "dense" not in v or "sparse" not in v:
                problems.append(f"run_meta.{key} missing/incomplete")

    # (3) ROOT_CAUSE.md cites every required artifact + has the serial table + verdict + no-fix boundary.
    if not os.path.exists(ROOT_CAUSE):
        problems.append("ROOT_CAUSE.md missing")
    else:
        txt = open(ROOT_CAUSE).read()
        for cite in REQUIRED_CITATIONS:
            if cite not in txt:
                problems.append(f"ROOT_CAUSE.md does not cite required artifact: {cite}")
        low = txt.lower()
        if "serial" not in low or "dense (serial)" not in low:
            problems.append("ROOT_CAUSE.md missing the final serial+batched per-arm table")
        if "h3" not in low or "scorer" not in low:
            problems.append("ROOT_CAUSE.md missing the H3 / scorer verdict")
        if "no selection/adapter fix is landed" not in low:
            problems.append("ROOT_CAUSE.md missing the explicit 'no fix landed' diagnosis-loop boundary")
        if "gate = good" not in low:
            problems.append("ROOT_CAUSE.md missing the GOOD gate result")

    report = {"ac": "AC-8 self-check (final evidence package complete + cited)",
              "core_arms_checked": list(CORE_ARMS),
              "required_citations": list(REQUIRED_CITATIONS),
              "problems": problems,
              "verdict": "AC-8 PACKAGE COMPLETE" if not problems else "AC-8 INCOMPLETE"}
    print(json.dumps(report, indent=2))
    if problems:
        print("FAIL (fail-closed): AC-8 package incomplete — " + " | ".join(problems), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
