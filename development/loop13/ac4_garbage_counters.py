#!/usr/bin/env python3
"""AC-4 length-cap garbage counters for the production SCORED DS selection.

Reuses the R14 forced_all_assert capture (dumps post-adapter physical slots + `_ds_slot_written` bits +
KV capacity + decode step per (rank,req,layer,step)) — but run WITHOUT `forced_all_dense_control`, so the
captured selection is the REAL production scored top-k, not the forced sweep. This reduces those records
to the AC-4 per-arm length-cap garbage-rate on the actual selected physical slots:
  - duplicate physical slot, live-lane -1, out-of-range (vs the true KV-slot CAPACITY), adapter error,
  - unwritten live slots, split into: the CURRENT decode slot (logical seq_len-1, if it was selected =
    the H3 marker) vs a NON-current slot (a real reused/stale-slot garbage event).
No req_to_token / sweep equality is checked — the selection is scored, not [0..seq_len-1]. Rows are split
by regime (dense seq_len<=top_k, sparse seq_len>top_k) and reported separately.

Usage: ac4_garbage_counters.py [--arm NAME] [CAPDIR]. `--arm` (default `production_ds`) sets the report
`arm` and the per-arm output filename (`ac4_garbage_counters.json` for production_ds, `..._{arm}.json`
otherwise) — used to reduce the served REFERENCE arms (`ref_faithful`, `ref_cosine`), which run the same
`forced_all_assert` hook but INCLUDE the current decode slot (so their current_slot_unwritten is nonzero).
For production_ds the input dir defaults to the production SCORED capture `evidence/.sglang_ds_garbage`
(the `ds_garbage` run), NOT the forced-all control `.sglang_ds_forcedall`; for any other arm pass its
capture dir explicitly. The report stamps `source_dir_basename` so downstream consumers (build_ledger) can
verify the artifact came from the expected capture.

Fail-closed: nonzero exit on (a) a missing/empty regime — this production scored reducer REQUIRES BOTH
dense and sparse with rows>0, so a dense-only forced-all dir cannot pass as scored evidence (and the JSON
is NOT written in that case, so a wrong-dir run can't clobber the canonical artifact); (b) any missing
required field; (c) any REAL (non-current) garbage — the current-slot-unwritten H3 marker is NOT a failure.
Writes evidence/ac4_garbage_counters.json. CPU-only.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# The PRODUCTION SCORED capture dir (ds_garbage run). NOT `.sglang_ds_forcedall` — that holds the forced-all
# control (dense-only, current-slot force-INCLUDED), a different dataset; reducing it here and committing it
# as "production scored" is exactly the R15 provenance regression. The reducer below fails closed unless
# BOTH regimes are present, so a forced-all dir (dense-only) can no longer pass as scored evidence.
DEFAULT_DIR = os.path.join(HERE, "evidence", ".sglang_ds_garbage")
TOP_K = 2048
_REQUIRED = ("physical_slots", "logical_positions", "slot_written_bits", "kv_capacity",
             "valid_length", "seq_len", "decode_step", "adapter_error_count",
             "tp_rank", "req_pool_index", "layer_id")


def _regime_acc():
    return {"rows": 0, "dup": 0, "neg": 0, "oor": 0, "err": 0,
            "current_slot_unwritten": 0, "noncurrent_unwritten": 0, "mismatches": []}


def main():
    # Usage: ac4_garbage_counters.py [--arm NAME] [CAPDIR]
    #   --arm production_ds (default) | ref_faithful | ref_cosine | ... — sets the report `arm` and the
    #   per-arm output filename. The capture dir defaults to DEFAULT_DIR only for production_ds; for any
    #   other arm pass the arm's capture dir explicitly (its records live in a distinct .sglang_* dir).
    args = sys.argv[1:]
    arm = "production_ds"
    positionals = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--arm":
            arm = args[i + 1]; i += 2
        elif a.startswith("--arm="):
            arm = a.split("=", 1)[1]; i += 1
        else:
            positionals.append(a); i += 1
    capdir = positionals[0] if positionals else DEFAULT_DIR
    files = sorted(glob.glob(os.path.join(capdir, "*.pt")))
    if not files:
        print(f"FAIL: no capture records in {capdir}", file=sys.stderr)
        raise SystemExit(2)

    acc = defaultdict(_regime_acc)   # "dense" / "sparse"
    seen = set()
    for p in files:
        r = torch.load(p, map_location="cpu", weights_only=False)
        missing = [k for k in _REQUIRED if k not in r]
        if missing:
            print(f"FAIL: record {os.path.basename(p)} missing fields {missing}", file=sys.stderr)
            raise SystemExit(2)
        key = (int(r["tp_rank"]), int(r["req_pool_index"]), int(r["layer_id"]), int(r["decode_step"]))
        if key in seen:
            continue
        seen.add(key)
        seq_len = int(r["seq_len"]); vlen = int(r["valid_length"]); kv_cap = int(r["kv_capacity"])
        if vlen <= 0:
            continue
        regime = "sparse" if seq_len > TOP_K else "dense"
        a = acc[regime]
        a["rows"] += 1
        a["err"] += int(r["adapter_error_count"])
        phys = r["physical_slots"].to(torch.long).reshape(-1)[:vlen]
        logi = r["logical_positions"].to(torch.long).reshape(-1)[:vlen]
        wbits = r["slot_written_bits"].reshape(-1).bool()
        if wbits.numel() < vlen:
            a["noncurrent_unwritten"] += 1
            a["mismatches"].append({"key": key, "reason": "slot_written_bits shorter than valid_length"})
            continue
        wbits = wbits[:vlen]
        if int(phys.numel()) != int(torch.unique(phys).numel()):
            a["dup"] += 1
        if bool((phys < 0).any()):
            a["neg"] += 1
        if bool(((phys < 0) | (phys >= kv_cap)).any()):
            a["oor"] += 1
        # unwritten live lanes -> is the unwritten lane the CURRENT decode slot (logical seq_len-1)?
        unwritten_lanes = (~wbits).nonzero(as_tuple=True)[0]
        if unwritten_lanes.numel():
            cur_pos = seq_len - 1
            unwritten_logical = logi[unwritten_lanes]
            has_current = bool((unwritten_logical == cur_pos).any())
            has_noncurrent = bool((unwritten_logical != cur_pos).any())
            a["current_slot_unwritten"] += int(has_current)
            a["noncurrent_unwritten"] += int(has_noncurrent)
            if has_noncurrent:
                a["mismatches"].append({"key": key, "seq_len": seq_len,
                                        "unwritten_noncurrent_logical": [int(x) for x in
                                                                        unwritten_logical[unwritten_logical != cur_pos].tolist()[:8]]})

    regimes = {}
    real_garbage = 0
    total_rows = 0
    for reg in ("dense", "sparse"):
        if reg not in acc:
            continue
        a = acc[reg]
        total_rows += a["rows"]
        real = a["dup"] + a["neg"] + a["oor"] + a["err"] + a["noncurrent_unwritten"]
        real_garbage += real
        regimes[reg] = {
            "rows": a["rows"],
            "duplicate_physical": a["dup"],
            "live_lane_-1": a["neg"],
            "out_of_range_vs_kv_capacity": a["oor"],
            "adapter_errors": a["err"],
            "noncurrent_unwritten (REAL garbage)": a["noncurrent_unwritten"],
            "current_slot_unwritten (H3 marker; not garbage)": a["current_slot_unwritten"],
            "real_garbage_total": real,
            "mismatches": a["mismatches"][:10],
        }
    # FAIL CLOSED on a structurally-wrong capture BEFORE writing, so a forced-all (dense-only) dir or a
    # no-arg run over the wrong dir can never overwrite the canonical committed scored artifact. A scored
    # capture run captures BOTH regimes (5-shot dense + 24-shot sparse); a single-regime dir is the
    # forced-all control or a partial capture and is NOT valid scored evidence for this arm.
    basename = os.path.basename(os.path.normpath(capdir))
    present = {r for r in ("dense", "sparse") if regimes.get(r, {}).get("rows", 0) > 0}
    missing = {"dense", "sparse"} - present
    if missing:
        print(f"FAIL: {arm} scored reducer requires BOTH dense and sparse regimes with rows>0; "
              f"missing/empty: {sorted(missing)}. Capture dir {capdir} (basename '{basename}') — this looks "
              f"like the forced-all control or a partial capture, not a full scored run. "
              f"NOT writing the artifact (refusing to clobber the canonical one).", file=sys.stderr)
        raise SystemExit(2)
    report = {
        "ac": f"AC-4 length-cap garbage counters — {arm} SCORED DS selection",
        "source": f"{capdir} (forced_all_assert eager run; scored top-k, no forced-all override)",
        "source_dir_basename": basename,
        "arm": arm,
        "regimes": regimes,
        "verdict": (f"CLEAN — the {arm} scored selection has zero real garbage (no duplicate / live-`-1` "
                    "/ out-of-range / adapter-error / non-current unwritten slot) on the captured rows; any "
                    "unwritten slot is the current decode slot (the H3 marker — present in the selected set "
                    "for current-slot-INCLUDED reference arms, absent for the current-slot-EXCLUDED "
                    "production arm)." if real_garbage == 0
                    else f"DIRTY — {real_garbage} real garbage events found; see per-regime mismatches."),
    }
    out_name = "ac4_garbage_counters.json" if arm == "production_ds" else f"ac4_garbage_counters_{arm}.json"
    out = os.path.join(HERE, "evidence", out_name)
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps({"arm": arm, "source_dir_basename": basename,
                      "regimes": {k: {kk: vv for kk, vv in v.items() if kk != "mismatches"}
                                  for k, v in regimes.items()}, "verdict": report["verdict"]}, indent=2))
    print("wrote", out)
    if real_garbage > 0:
        print(f"FAIL: {real_garbage} real (non-current) garbage events in the scored selection", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
