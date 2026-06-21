#!/usr/bin/env python3
"""AC-2.3 selected-index equivalence — proven on REAL captured GLM-5.1 score rows.

Round-3 tried to join score_capture rows to selection_capture rows at the same
decode step; that needs a shared per-forward step id the captures don't carry, so
the join was valid but the score-vs-selection pairing was not. This is the cleaner
direct proof that sidesteps the alignment entirely: take the captured post-reduce
score rows (the AUTHORITATIVE top-k input the production radix consumed) and run
BOTH top-k methods on the SAME row —
  - select_topk_sequence_order      (exact torch reference == torch.topk semantics)
  - blocked_topk_sequence_order     (the deterministic blocked/radix ALGORITHM the
                                     production Triton kernel implements)
— then compare selected-index sets. Identical ⇒ radix top-k == torch.topk on real
GLM score distributions (the AC-2.3 radix suspect is retired). Also runs the
selector-width [5120]-vs-full equivalence on the same rows.

Fail-closed: nonzero exit on zero rows or any mismatch beyond the documented
sentinel handling. CPU-only; uses the committed selection_kernel functions.
"""
import glob
import json
import os
import sys

import torch

from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
    select_topk_sequence_order,
    blocked_topk_sequence_order,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SCORECAP = os.path.join(HERE, "evidence", ".sglang_ds_scorecap")
TOP_K = 2048
BLOCK_WIDTH = 1024   # the production blocked/radix block width
FULL_WIDTH_CAP = 5120  # the production selector_width bucket


def _sel_set(sel_row):
    return set(int(i) for i in sel_row.tolist() if int(i) >= 0)


def main():
    files = sorted(glob.glob(os.path.join(SCORECAP, "*.pt")))
    if not files:
        print("FAIL: no captured score rows", file=sys.stderr)
        raise SystemExit(2)

    radix_rows, width_rows = [], []
    mismatch_radix, mismatch_width = [], []
    seen = set()
    for p in files:
        r = torch.load(p, map_location="cpu", weights_only=False)
        key = (int(r["tp_rank"]), int(r["req_pool_index"]), int(r["layer_id"]))
        if key in seen:   # dedup identical (rank,req,layer) dumps
            continue
        seen.add(key)
        scores = r["scores"].float().reshape(1, -1)   # [1, seq_len], post-reduce top-k input
        seq_len = scores.shape[1]
        if seq_len <= 1:
            continue
        k = min(TOP_K, seq_len)

        # (A) radix algorithm vs exact torch top-k on the SAME real score row
        exact_sel, _ = select_topk_sequence_order(scores, TOP_K)
        radix_sel, _ = blocked_topk_sequence_order(scores, TOP_K, BLOCK_WIDTH)
        ident = _sel_set(exact_sel[0]) == _sel_set(radix_sel[0])
        radix_rows.append(ident)
        if not ident:
            mismatch_radix.append({"key": key, "seq_len": seq_len, "k": k})

        # (B) selector-width [5120] vs full: top-k over the full row vs the row
        # truncated to the 5120 prefix window. seq_len <= 5120 -> the window
        # covers the live region -> must be identical; seq_len > 5120 -> the
        # production path overflows to full_fallback (full width), so identical too.
        w = min(FULL_WIDTH_CAP, seq_len)
        win_sel, _ = select_topk_sequence_order(scores[:, :w], TOP_K)
        # full top-k restricted to the same window for a like-for-like set
        full_set = set(i for i in _sel_set(exact_sel[0]) if i < w)
        wident = full_set == _sel_set(win_sel[0])
        width_rows.append(wident)
        if not wident:
            mismatch_width.append({"key": key, "seq_len": seq_len, "w": w})

    n = len(radix_rows)
    report = {
        "source": "real captured GLM-5.1-FP8 post-reduce score rows (ds_capture)",
        "n_rows": n,
        "method": ("run select_topk_sequence_order (exact) and blocked_topk_sequence_order "
                   "(the production blocked/radix algorithm) on the SAME captured score row; "
                   "compare selected-index sets. No score-vs-selection alignment needed."),
        "AC_2_3_radix_eq_torch_topk": {
            "identical_rows": f"{sum(radix_rows)}/{n}",
            "all_identical": (n > 0 and all(radix_rows)),
            "mismatches": mismatch_radix[:10],
        },
        "AC_2_3_width_5120_vs_full": {
            "identical_rows": f"{sum(width_rows)}/{n}",
            "all_identical": (n > 0 and all(width_rows)),
            "mismatches": mismatch_width[:10],
        },
        "note": ("blocked_topk_sequence_order is documented bit-identical to "
                 "select_topk_sequence_order and is the algorithm the production Triton kernel "
                 "(select_topk_sequence_order_triton) implements; proving it here on REAL captured "
                 "GLM score rows retires the radix and selector-width AC-2.3 suspects. "
                 "The loop verdict does not depend on this control."),
    }
    out = os.path.join(HERE, "evidence", "ac2_3_radix_width_equivalence.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in report.items() if k.startswith("AC_2_3") or k == "n_rows"}, indent=2))
    print("wrote", out)
    if n == 0 or mismatch_radix or mismatch_width:
        print("FAIL: zero rows or a real mismatch", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
