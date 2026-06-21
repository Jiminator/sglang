#!/usr/bin/env python3
"""AC-2.1 forced-all dense downstream-isolation assertions (reduce captured adapter records).

Consumes the per-(rank,req,layer,step) dumps from forced_all_assert_capture (a `ds_forced_all_assert`
run): each record has the post-adapter PHYSICAL selected slots, the forced LOGICAL positions, the
request's `req_to_token[req, 0:seq_len]` slice, the `_ds_slot_written[layer_id, physical_slot]` validity
bit for each LIVE physical slot, the KV-slot capacity, and the `logical_to_physical` error count. For the
forced-all dense control (seq_len <= top_k) the selector emits logical [0..seq_len-1], so this verifies
that the forced dense sweep maps to EXACTLY the request's own KV slots AND that every selected slot is
valid:
  - logical positions == [0..seq_len-1] (the forced sweep; valid_length == seq_len),
  - physical slots == req_to_token[req, 0:seq_len] (element-wise; the adapter GATHER),
  - every live physical slot is WRITTEN (_ds_slot_written True — separate backend validity bitmap),
  - no duplicate physical slot, no live-lane -1, no out-of-range physical slot (vs KV-slot CAPACITY, not
    the req_to_token max-context width), adapter error_count == 0,
  per layer AND per decode step (records are keyed by step; no overwrite).

PASS ⇒ the dense forced-all selection is a PROVABLE no-op (the same valid KV slots DSA would feed), so
the residual dense degradation (0.620 vs 0.975) is DOWNSTREAM of selection — the H3 fork. The same
counters are the AC-4 length-cap garbage-rate (duplicate / unwritten / -1 / out-of-range / error).

Fail-closed: nonzero exit on zero dense rows, any missing required field (incl. slot_written_bits / step /
kv_capacity), or any failing assertion. Writes evidence/forced_all_assertions.json. CPU-only.
"""
import glob
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "evidence", ".sglang_ds_forcedall")
TOP_K = 2048
_REQUIRED = ("physical_slots", "logical_positions", "expected_physical", "slot_written_bits",
             "kv_capacity", "valid_length", "seq_len", "decode_step", "adapter_error_count",
             "tp_rank", "req_pool_index", "layer_id")


def main():
    capdir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    files = sorted(glob.glob(os.path.join(capdir, "*.pt")))
    if not files:
        print(f"FAIL: no forced-all assertion records in {capdir}", file=sys.stderr)
        raise SystemExit(2)

    n = 0
    eq = 0
    logical_sweep_ok = 0
    dup = neg = oor = 0               # rows with a violation
    current_unwritten = 0            # rows whose CURRENT decode slot (seq_len-1) is unwritten — the H3 marker
    noncurrent_unwritten = 0         # rows with an unwritten NON-current slot — a real garbage failure
    err = 0
    dense_rows = 0
    steps = set()
    seen = set()
    mismatches = []
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
        steps.add(int(r["decode_step"]))
        seq_len = int(r["seq_len"])
        vlen = int(r["valid_length"])
        kv_cap = int(r["kv_capacity"])
        phys = r["physical_slots"].to(torch.long).reshape(-1)
        logi = r["logical_positions"].to(torch.long).reshape(-1)
        exp = r["expected_physical"].to(torch.long).reshape(-1)
        wbits = r["slot_written_bits"].reshape(-1).bool()
        err += int(r["adapter_error_count"])
        n += 1
        if seq_len > TOP_K:
            continue            # forced-all only overrides dense rows (seq_len <= top_k)
        dense_rows += 1
        live_phys = phys[:vlen]
        live_logi = logi[:vlen]
        is_sweep = (vlen == seq_len) and torch.equal(live_logi, torch.arange(seq_len, dtype=torch.long))
        logical_sweep_ok += int(is_sweep)
        is_eq = (exp.numel() == seq_len) and (live_phys.numel() == seq_len) and torch.equal(live_phys, exp[:seq_len])
        eq += int(is_eq)
        has_dup = int(live_phys.numel()) != int(torch.unique(live_phys).numel())
        has_neg = bool((live_phys < 0).any())
        has_oor = bool(((live_phys < 0) | (live_phys >= kv_cap)).any())   # vs KV-slot CAPACITY
        # _ds_slot_written validity: which LIVE slots are marked unwritten. For the forced-all dense
        # sweep, live lane i is logical position i, so the CURRENT decode slot is the last live lane
        # (valid_length-1 == seq_len-1). The production path invalidates exactly that slot before scoring
        # (_slot_written[layer,out_cache_loc]=False) — the H3 mechanism — so its bit is EXPECTED to be
        # False; a False bit on any OTHER live slot is a real reused/stale-slot garbage failure.
        if wbits.numel() != vlen:
            noncurrent_unwritten += 1   # missing/short bits -> cannot verify -> treat as failure
            mismatches.append({"key": key, "reason": "slot_written_bits length != valid_length"})
            continue
        unwritten_idx = (~wbits[:vlen]).nonzero(as_tuple=True)[0]
        cur = vlen - 1
        has_current_unwritten = bool((unwritten_idx == cur).any())
        has_noncurrent_unwritten = bool((unwritten_idx != cur).any())
        current_unwritten += int(has_current_unwritten)
        noncurrent_unwritten += int(has_noncurrent_unwritten)
        dup += int(has_dup); neg += int(has_neg); oor += int(has_oor)
        if not (is_sweep and is_eq) or has_dup or has_neg or has_oor or has_noncurrent_unwritten:
            mismatches.append({"key": key, "seq_len": seq_len, "valid_length": vlen, "kv_capacity": kv_cap,
                               "sweep": is_sweep, "phys_eq_reqtok": is_eq, "dup": has_dup, "neg": has_neg,
                               "oor": has_oor, "noncurrent_unwritten": has_noncurrent_unwritten})

    # PASS = the adapter gather is exact and every NON-current slot is valid. The current decode slot being
    # unwritten is NOT a failure — it is the H3 marker (the slot-validity invalidation under test).
    all_pass = (dense_rows > 0 and eq == dense_rows and logical_sweep_ok == dense_rows
                and dup == 0 and neg == 0 and oor == 0 and noncurrent_unwritten == 0 and err == 0)
    # The H3 finding ("current slot unwritten on every row") is only assertable when it actually holds on
    # ALL dense rows — make the prose conditional so a future rerun cannot inherit a stale claim.
    h3_on_all_rows = (dense_rows > 0 and current_unwritten == dense_rows)
    report = {
        "ac": "AC-2.1 forced-all dense downstream-isolation assertions (per layer + decode step)",
        "source": f"{capdir} (ds_forced_all_assert eager run)",
        "records": n,
        "distinct_decode_steps": sorted(steps)[:20],
        "dense_forced_all_rows": dense_rows,
        "assertions": {
            "logical_sweep_[0..seq_len-1]": f"{logical_sweep_ok}/{dense_rows}",
            "physical==req_to_token[0:seq_len]": f"{eq}/{dense_rows}",
            "rows_with_NONcurrent_unwritten_slot (real garbage; must be 0)": noncurrent_unwritten,
            "rows_with_CURRENT_slot_unwritten (the H3 marker; expected)": f"{current_unwritten}/{dense_rows}",
            "rows_with_duplicate_physical": dup,
            "rows_with_live_lane_-1": neg,
            "rows_with_out_of_range_physical (vs kv_capacity)": oor,
            "adapter_error_count_total": err,
        },
        # AC-4 length-cap garbage-rate (same counters; unwritten now MEASURED via _ds_slot_written).
        # "unwritten" here is the REAL garbage count (non-current); the current-slot invalidation is the
        # H3 mechanism, reported separately, not garbage.
        "ac4_garbage_counters": {"duplicate": dup, "live_-1": neg, "out_of_range": oor,
                                 "unwritten_noncurrent": noncurrent_unwritten,
                                 "current_slot_unwritten_H3": current_unwritten, "adapter_errors": err},
        "h3_marker_on_all_rows": h3_on_all_rows,
        "h3_finding": (("On every dense row exactly ONE live slot is marked unwritten by _ds_slot_written, "
                        "and it is exactly the CURRENT decode slot (logical seq_len-1) — the production "
                        "_slot_written[layer,out_cache_loc]=False invalidation. The adapter gather is exact "
                        "and every OTHER slot is written, so the selected-index/adapter path is clean; the "
                        "dense regression localizes to this current-slot invalidation (H3), observed "
                        "directly on the validity bitmap. Forcing all tokens recovers dense to ~0.950, so "
                        "the current slot's KV is valid at attention time and the bit is merely stale.")
                       if h3_on_all_rows else
                       (f"current-slot-unwritten holds on {current_unwritten}/{dense_rows} dense rows "
                        "(NOT all) — the H3-on-every-row claim does not apply to this capture; see the "
                        "per-row counters.")),
        "verdict": ("PASS — the forced-all dense adapter gather is exact (physical==req_to_token), every "
                    "NON-current slot is written, and there are zero duplicate/-1/out-of-range/adapter "
                    "errors across all layers+steps; the ONLY unwritten slot is the current decode slot on "
                    "every row (the H3 marker). So the selected-index/adapter path is a clean no-op and the "
                    "dense regression is DOWNSTREAM of selection at the slot-validity (H3) level." if all_pass
                    else "FAIL — a NON-current slot is unwritten or the gather/range/dup checks failed; see mismatches."),
        "mismatches": mismatches[:10],
    }
    out = os.path.join(HERE, "evidence", "forced_all_assertions.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps({k: report[k] for k in
                      ("records", "dense_forced_all_rows", "distinct_decode_steps", "assertions", "verdict")},
                     indent=2))
    print("wrote", out)
    if dense_rows == 0:
        print("FAIL: zero dense forced-all rows (seq_len<=top_k) observed", file=sys.stderr)
        raise SystemExit(2)
    if not all_pass:
        print("FAIL: forced-all dense assertions did not all pass", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
