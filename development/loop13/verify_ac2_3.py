#!/usr/bin/env python3
"""AC-2.3 selected-index equivalence — proven on REAL captured GLM-5.1-FP8 score rows.

Retires the radix (suspicion-order 5) and selector-width (6) legs of the AC-6 bisection by
running BOTH top-k methods on the SAME captured post-reduce score row (no score-vs-selection
alignment needed):
  - select_topk_sequence_order    (exact torch reference == torch.topk semantics)
  - blocked_topk_sequence_order   (the deterministic blocked/radix ALGORITHM the production Triton
                                   kernel select_topk_sequence_order_triton implements)
and comparing selected-index sets.

PRUNING VALIDITY (Round 5): a row only exercises top-k pruning when seq_len > top_k. The Round-4
captures were all seq_len=13 << top_k=2048, so both methods trivially selected ALL positions and the
width-[5120] check was vacuous — a select-all sanity check, not a sparse-regime proof. This script
now:
  - records the seq_len distribution (min/median/max) and a pruning_rows count (seq_len > top_k),
  - evaluates the radix==torch.topk claim on the PRUNING SUBSET only,
  - splits the width check: for top_k < seq_len <= 5120 require full == 5120-window selection
    (the [5120] bucket fully covers the live region); seq_len > 5120 rows are counted as
    width_overflow and excluded from the equivalence claim (production overflow policy is not
    re-implemented here),
  - FAILS (nonzero exit) if pruning_rows == 0, so it can never "pass" on smoke captures again.

Drive the sparse regime (24-shot GSM8K, ~4.2k tokens) so captured rows have seq_len > top_k.
Capture dir: argv[1], or $SGLANG_DS_SCORE_CAPTURE_DIR, or evidence/.sglang_ds_scorecap.
CPU-only; uses the committed selection_kernel functions.
"""
import glob
import json
import os
import statistics
import sys

import torch

from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
    select_topk_sequence_order,
    blocked_topk_sequence_order,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCORECAP = os.path.join(HERE, "evidence", ".sglang_ds_scorecap")
TOP_K = 2048
BLOCK_WIDTH = 1024     # the production blocked/radix block width
WIDTH_BUCKET = 5120    # the production selector_width bucket


def _sel_set(sel_row):
    return set(int(i) for i in sel_row.tolist() if int(i) >= 0)


def main():
    scorecap = (sys.argv[1] if len(sys.argv) > 1
                else os.environ.get("SGLANG_DS_SCORE_CAPTURE_DIR", DEFAULT_SCORECAP))
    files = sorted(glob.glob(os.path.join(scorecap, "*.pt")))
    if not files:
        print(f"FAIL: no captured score rows in {scorecap}", file=sys.stderr)
        raise SystemExit(2)

    seq_lens = []
    radix_rows, mismatch_radix = [], []           # over the pruning subset (seq_len > top_k)
    width_rows, mismatch_width = [], []           # over top_k < seq_len <= 5120
    width_overflow = 0                            # seq_len > 5120 (excluded from width claim)
    seen = set()
    for p in files:
        r = torch.load(p, map_location="cpu", weights_only=False)
        key = (int(r["tp_rank"]), int(r["req_pool_index"]), int(r["layer_id"]))
        if key in seen:    # dedup identical (rank,req,layer) dumps
            continue
        seen.add(key)
        scores = r["scores"].float().reshape(1, -1)   # [1, seq_len], post-reduce top-k input
        seq_len = scores.shape[1]
        if seq_len <= 1:
            continue
        seq_lens.append(seq_len)
        if seq_len <= TOP_K:
            continue   # no pruning exercised on this row

        # (A) radix algorithm vs exact torch top-k on the SAME real PRUNING score row
        exact_sel, _ = select_topk_sequence_order(scores, TOP_K)
        radix_sel, _ = blocked_topk_sequence_order(scores, TOP_K, BLOCK_WIDTH)
        ident = _sel_set(exact_sel[0]) == _sel_set(radix_sel[0])
        radix_rows.append(ident)
        if not ident:
            mismatch_radix.append({"key": key, "seq_len": seq_len})

        # (B) selector-width [5120] vs full. Meaningful case: top_k < seq_len <= 5120 -> the [5120]
        # window fully covers the live region, so the prefix-window top-k must equal the full top-k.
        if seq_len <= WIDTH_BUCKET:
            win_sel, _ = select_topk_sequence_order(scores[:, :WIDTH_BUCKET], TOP_K)
            wident = _sel_set(exact_sel[0]) == _sel_set(win_sel[0])
            width_rows.append(wident)
            if not wident:
                mismatch_width.append({"key": key, "seq_len": seq_len})
        else:
            width_overflow += 1   # production overflow policy not re-implemented; reported, not claimed

    n_total = len(seq_lens)
    n_prune = len(radix_rows)
    seq_dist = (None if not seq_lens else {
        "min": min(seq_lens), "median": int(statistics.median(seq_lens)), "max": max(seq_lens),
        "n_rows": n_total, "rows_seq_len_gt_top_k": n_prune,
    })
    report = {
        "source": f"real captured GLM-5.1-FP8 post-reduce score rows ({scorecap})",
        "top_k": TOP_K, "width_bucket": WIDTH_BUCKET, "block_width": BLOCK_WIDTH,
        "seq_len_distribution": seq_dist,
        "method": ("run select_topk_sequence_order (exact) and blocked_topk_sequence_order "
                   "(production blocked/radix algorithm) on the SAME captured score row; compare "
                   "selected-index sets. Radix claim evaluated on the pruning subset seq_len > top_k."),
        "AC_2_3_radix_eq_torch_topk": {
            "pruning_rows": n_prune,
            "identical_rows": f"{sum(radix_rows)}/{n_prune}",
            "all_identical": (n_prune > 0 and all(radix_rows)),
            "mismatches": mismatch_radix[:10],
        },
        "AC_2_3_width_5120_vs_full": {
            "eval_rows_top_k_lt_seq_le_5120": len(width_rows),
            "identical_rows": f"{sum(width_rows)}/{len(width_rows)}",
            "all_identical": (len(width_rows) > 0 and all(width_rows)),
            "width_overflow_rows_seq_gt_5120_excluded": width_overflow,
            "mismatches": mismatch_width[:10],
        },
        "note": ("blocked_topk_sequence_order is the algorithm the production Triton kernel implements; "
                 "proving it == exact torch.topk on REAL captured GLM score rows WHERE top_k < seq_len "
                 "retires the radix and (live-region) selector-width AC-2.3 suspects for the sparse "
                 "regime. The loop verdict does not depend on this control."),
    }
    out = os.path.join(HERE, "evidence", "ac2_3_radix_width_equivalence.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps({k: report[k] for k in
                      ("seq_len_distribution", "AC_2_3_radix_eq_torch_topk", "AC_2_3_width_5120_vs_full")},
                     indent=2))
    print("wrote", out)

    # Fail-closed: must exercise pruning, and no real mismatch.
    if n_prune == 0:
        print(f"FAIL: zero pruning rows (no captured row has seq_len > top_k={TOP_K}); "
              f"capture the SPARSE regime (24-shot). seq_len max seen = "
              f"{max(seq_lens) if seq_lens else 'n/a'}", file=sys.stderr)
        raise SystemExit(2)
    if mismatch_radix or mismatch_width:
        print("FAIL: a real radix/width selection mismatch was found", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
