#!/usr/bin/env python3
"""AC-2.1 forced-all dense downstream-isolation assertions (reduce captured adapter records).

Consumes the per-(rank,req,layer) dumps from forced_all_assert_capture (a `ds_forced_all_assert` run):
each record has the post-adapter PHYSICAL selected slots, the forced LOGICAL positions, the request's
`req_to_token[req, 0:seq_len]` slice, and the `logical_to_physical` error count. For the forced-all dense
control (seq_len <= top_k) the selector emits logical [0..seq_len-1], so this verifies the adapter maps
that sweep to EXACTLY the request's own KV slots:
  - logical positions == [0..seq_len-1] (the forced sweep; valid_length == seq_len),
  - physical slots == req_to_token[req, 0:seq_len] (element-wise),
  - no duplicate physical slot, no live-lane -1, no out-of-range physical slot,
  - adapter error_count == 0.

PASS ⇒ the dense forced-all selection is a PROVABLE no-op (the same KV slots DSA would feed), so the
residual dense degradation (0.620 vs 0.975) is DOWNSTREAM of selection — the H3 fork. The same counters
are the AC-4 length-cap garbage-rate (duplicate / unwritten-via-equality / -1 / out-of-range / error).

Fail-closed: nonzero exit on zero records, any missing field, or any failing assertion. Writes
evidence/forced_all_assertions.json. CPU-only.
"""
import glob
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "evidence", ".sglang_ds_forcedall")
TOP_K = 2048
_REQUIRED = ("physical_slots", "logical_positions", "expected_physical", "valid_length", "seq_len",
             "adapter_error_count", "req_to_token_width", "tp_rank", "req_pool_index", "layer_id")


def main():
    capdir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    files = sorted(glob.glob(os.path.join(capdir, "*.pt")))
    if not files:
        print(f"FAIL: no forced-all assertion records in {capdir}", file=sys.stderr)
        raise SystemExit(2)

    n = 0
    eq = 0                 # physical == req_to_token[0:seq_len]
    logical_sweep_ok = 0   # logical == [0..seq_len-1]
    dup = 0                # rows with a duplicate physical slot
    neg = 0                # rows with a live-lane -1
    oor = 0                # rows with an out-of-range physical slot
    err = 0                # sum of adapter error_count
    dense_rows = 0
    seen = set()
    mismatches = []
    for p in files:
        r = torch.load(p, map_location="cpu", weights_only=False)
        missing = [k for k in _REQUIRED if k not in r]
        if missing:
            print(f"FAIL: record {os.path.basename(p)} missing fields {missing}", file=sys.stderr)
            raise SystemExit(2)
        key = (int(r["tp_rank"]), int(r["req_pool_index"]), int(r["layer_id"]))
        if key in seen:
            continue
        seen.add(key)
        seq_len = int(r["seq_len"])
        vlen = int(r["valid_length"])
        width = int(r["req_to_token_width"])
        phys = r["physical_slots"].to(torch.long).reshape(-1)
        logi = r["logical_positions"].to(torch.long).reshape(-1)
        exp = r["expected_physical"].to(torch.long).reshape(-1)
        err += int(r["adapter_error_count"])
        n += 1
        if seq_len > TOP_K:
            continue            # forced-all only overrides dense rows (seq_len <= top_k)
        dense_rows += 1
        live_phys = phys[:vlen]
        live_logi = logi[:vlen]
        # (a) forced logical sweep [0..seq_len-1]
        is_sweep = (vlen == seq_len) and torch.equal(live_logi, torch.arange(seq_len, dtype=torch.long))
        logical_sweep_ok += int(is_sweep)
        # (b) physical == req_to_token[0:seq_len] element-wise
        is_eq = (exp.numel() == seq_len) and (live_phys.numel() == seq_len) and torch.equal(live_phys, exp[:seq_len])
        eq += int(is_eq)
        # (c) no duplicate live physical slot
        has_dup = int(live_phys.numel()) != int(torch.unique(live_phys).numel())
        dup += int(has_dup)
        # (d) no live-lane -1
        has_neg = bool((live_phys < 0).any())
        neg += int(has_neg)
        # (e) no out-of-range physical slot
        has_oor = bool(((live_phys < 0) | (live_phys >= width)).any())
        oor += int(has_oor)
        if not (is_sweep and is_eq) or has_dup or has_neg or has_oor:
            mismatches.append({"key": key, "seq_len": seq_len, "valid_length": vlen,
                               "sweep": is_sweep, "phys_eq_reqtok": is_eq, "dup": has_dup,
                               "neg": has_neg, "oor": has_oor})

    all_pass = (dense_rows > 0 and eq == dense_rows and logical_sweep_ok == dense_rows
                and dup == 0 and neg == 0 and oor == 0 and err == 0)
    report = {
        "ac": "AC-2.1 forced-all dense downstream-isolation assertions",
        "source": f"{capdir} (ds_forced_all_assert eager run)",
        "records": n,
        "dense_forced_all_rows": dense_rows,
        "assertions": {
            "logical_sweep_[0..seq_len-1]": f"{logical_sweep_ok}/{dense_rows}",
            "physical==req_to_token[0:seq_len]": f"{eq}/{dense_rows}",
            "rows_with_duplicate_physical": dup,
            "rows_with_live_lane_-1": neg,
            "rows_with_out_of_range_physical": oor,
            "adapter_error_count_total": err,
        },
        # AC-4 length-cap garbage-rate (same counters; unwritten is subsumed by the physical==req_to_token
        # equality — the request's own token slots [0:seq_len] are written by construction).
        "ac4_garbage_counters": {"duplicate": dup, "live_-1": neg, "out_of_range": oor,
                                 "adapter_errors": err, "unwritten": "0 (subsumed by physical==req_to_token equality)"},
        "verdict": ("PASS — forced-all dense selection maps to exactly the request's own KV slots with zero "
                    "duplicate/-1/out-of-range/adapter errors, so dense selection is a PROVABLE no-op and "
                    "the residual dense degradation is DOWNSTREAM of selection (H3)." if all_pass
                    else "FAIL — see mismatches; the forced-all dense override is not a clean no-op."),
        "mismatches": mismatches[:10],
    }
    out = os.path.join(HERE, "evidence", "forced_all_assertions.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps({k: report[k] for k in ("records", "dense_forced_all_rows", "assertions", "verdict")}, indent=2))
    print("wrote", out)
    if dense_rows == 0:
        print(f"FAIL: zero dense forced-all rows (seq_len<=top_k) observed", file=sys.stderr)
        raise SystemExit(2)
    if not all_pass:
        print("FAIL: forced-all dense assertions did not all pass", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
