#!/usr/bin/env python3
"""AC-3.1 captured-row materialized fp32 K_label selected-index equality (CPU reducer).

Consumes the per-(rank,layer,regime,step) minimal reconstructions from materialized_k_capture (a guarded
``ref_faithful_matk`` run). For each captured decode row it rebuilds a bs=1 call over the captured live fp8
latent (``req_to_token = [[0..seq_len-1]]``) and runs the SAME two served functions the proof rests on:
  - absorbed raw-dot:  absorbed_latent_score_logical_fp8(...)           (the cheap served reference score),
  - materialized:      absorbed_latent_cosine_logical_fp8(..., normalize=False)  (raw dot on the MATERIALIZED
                       per-head K_label signature),
then takes select_topk_sequence_order(..., 2048) of each and asserts the selected-index SETS are equal. This
is the CAPTURED-row form of the algebra identity proven synthetically in
test_reference_selectors.py::test_materialized_raw_equals_absorbed_raw — but on REAL served decode rows.

Split by regime (dense seq_len<=top_k, sparse seq_len>top_k); REQUIRES both regimes with rows>0. Fail-closed:
the canonical artifact is written ONLY via atomic rename after every row's selected-index sets match in BOTH
regimes; a missing field / missing regime / ANY mismatch → exit 2 with the canonical artifact UNTOUCHED.
CPU-only (the absorbed functions run on CPU, like the unit test).
"""
import glob
import json
import os
import sys
from collections import defaultdict

import torch

from sglang.srt.layers.attention.double_sparsity.absorbed_latent import (
    absorbed_latent_cosine_logical_fp8,
    absorbed_latent_score_logical_fp8,
)
from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
    select_topk_sequence_order,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "evidence", ".sglang_ds_matk")
TOP_K = 2048
_REQUIRED = ("queries", "live_latent_fp8", "live_latent_scales", "live_written", "w_sel",
             "channel_selection", "channel_weights", "seq_len", "max_top_k", "head_agg",
             "regime", "tp_rank", "layer_id", "decode_step")


def _sel_set(sel):
    return set(int(i) for i in sel[0].tolist() if int(i) >= 0)


def main():
    capdir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    files = sorted(glob.glob(os.path.join(capdir, "*.pt")))
    if not files:
        print(f"FAIL: no materialized-K capture records in {capdir}", file=sys.stderr)
        raise SystemExit(2)

    acc = defaultdict(lambda: {"rows": 0, "equal": 0, "max_score_diff": 0.0, "mismatches": []})
    seen = set()
    for p in files:
        r = torch.load(p, map_location="cpu", weights_only=False)
        missing = [k for k in _REQUIRED if k not in r]
        if missing:
            print(f"FAIL: record {os.path.basename(p)} missing fields {missing}", file=sys.stderr)
            raise SystemExit(2)
        key = (int(r["tp_rank"]), int(r["layer_id"]), str(r["regime"]), int(r["decode_step"]))
        if key in seen:
            continue
        seen.add(key)
        seq_len = int(r["seq_len"])
        regime = str(r["regime"])
        # Minimal bs=1 reconstruction over the captured live latent: req_to_token maps logical [0..seq_len-1]
        # to the gathered live slots [0..seq_len-1], so the functions gather exactly the captured latent.
        common = dict(
            queries=r["queries"],
            latent_fp8=r["live_latent_fp8"],
            latent_scales=r["live_latent_scales"],
            w_sel=r["w_sel"],
            channel_selection=r["channel_selection"],
            channel_weights=r["channel_weights"],
            req_pool_indices=torch.zeros(1, dtype=torch.long),
            req_to_token=torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
            seq_lens=torch.tensor([seq_len], dtype=torch.long),
            max_seq_len=seq_len,
            written=r["live_written"].bool(),
            head_agg=str(r["head_agg"]),
        )
        raw = absorbed_latent_score_logical_fp8(**common)
        mat = absorbed_latent_cosine_logical_fp8(**common, normalize=False)
        raw_sel, _ = select_topk_sequence_order(raw, TOP_K)
        mat_sel, _ = select_topk_sequence_order(mat, TOP_K)

        a = acc[regime]
        a["rows"] += 1
        fin = torch.isfinite(raw) & torch.isfinite(mat)
        a["max_score_diff"] = max(a["max_score_diff"],
                                  (raw[fin] - mat[fin]).abs().max().item() if bool(fin.any()) else 0.0)
        rs, ms = _sel_set(raw_sel), _sel_set(mat_sel)
        if rs == ms:
            a["equal"] += 1
        else:
            a["mismatches"].append({"key": list(key), "raw_only": sorted(rs - ms)[:8],
                                    "mat_only": sorted(ms - rs)[:8]})

    regimes = {}
    total = 0
    all_equal = True
    for reg in ("dense", "sparse"):
        if reg not in acc:
            all_equal = False
            continue
        a = acc[reg]
        total += a["rows"]
        if a["equal"] != a["rows"] or a["rows"] == 0:
            all_equal = False
        regimes[reg] = {"rows": a["rows"], "selected_index_equal_rows": a["equal"],
                        "max_abs_score_diff": round(a["max_score_diff"], 9),
                        "mismatches": a["mismatches"][:5]}

    present = {reg for reg in ("dense", "sparse") if acc.get(reg, {}).get("rows", 0) > 0}
    ok = (present == {"dense", "sparse"}) and all_equal and total > 0
    report = {
        "ac": "AC-3.1 captured-row materialized fp32 K_label selected-index equality @2048",
        "source": f"{capdir} (ref_faithful_matk eager run; reference_rawdot_select inputs)",
        "source_dir_basename": os.path.basename(os.path.normpath(capdir)),
        "method": ("rebuild bs=1 over the captured live fp8 latent; absorbed_latent_score_logical_fp8 (raw) "
                   "vs absorbed_latent_cosine_logical_fp8(normalize=False) (materialized per-head K_label "
                   "numerator); select_topk_sequence_order @2048; assert selected-index set equality per row."),
        "index_topk": TOP_K,
        "total_rows": total,
        "regimes": regimes,
        "all_selected_index_equal": ok,
        "verdict": ("EQUAL — on every captured decode row (dense + sparse) the absorbed raw-dot reference and "
                    "the materialized fp32 K_label score select the IDENTICAL top-2048 indices, so the served "
                    "raw-dot ceiling is the materialized-K_label ceiling on real data (captured-row form of "
                    "the exact-algebra identity)." if ok
                    else "NOT EQUAL / INCOMPLETE — a selected-index mismatch or a missing regime; see mismatches."),
        "supersedes": "evidence/ac3_1_materialized_k.json (synthetic CPU algebra proof)",
    }
    out = os.path.join(HERE, "evidence", "ac3_1_materialized_k_selected_index_equality.json")
    print(json.dumps({k: report[k] for k in ("total_rows", "regimes", "all_selected_index_equal")}, indent=2))
    if not ok:
        print("FAIL (fail-closed): " + report["verdict"] + " — NOT writing the canonical artifact.",
              file=sys.stderr)
        raise SystemExit(2)
    tmp = out + ".tmp"
    json.dump(report, open(tmp, "w"), indent=2)
    os.replace(tmp, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
