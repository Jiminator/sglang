"""R25 PROBE 4 — no_dense_fallback. Across the selection captures gathered by the
other probes, assert DS selection always returns a valid SPARSE top-k:
valid_length == top_k for every (layer, batch-row) whose seq_len >= top_k, and
never a full-dense degrade (valid_length == seq_len > top_k, or valid_length == 0).

Reads every <capdir>/rank*_step*.pt selection-capture file under the given dirs.
Each record carries:
  indices  [num_layers, bs, max_top_k]   (selected logical positions, -1 padded)
  lengths  [num_layers, bs]              (valid_length per (layer, row))
  seq_lens [bs]                          (per-row sequence length at this step)

A row with seq_len >= top_k MUST have valid_length == top_k (full sparse budget).
A row with seq_len < top_k may have valid_length == seq_len (the whole short
context is selected — still sparse-correct, not a dense degrade of a long context).
FAIL-CLOSED: any (layer,row) with seq_len >= top_k and valid_length != top_k, or any
valid_length > top_k (over-budget), or valid_length == 0 with seq_len > 0.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capdirs", nargs="+", required=True)
    ap.add_argument("--top-k", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    top_k = args.top_k
    files = []
    for d in args.capdirs:
        files += sorted(glob.glob(os.path.join(d, "rank*_step*.pt")))

    summary = {
        "probe": "P4_no_dense_fallback",
        "top_k": top_k,
        "capdirs": args.capdirs,
        "files_scanned": len(files),
    }
    if not files:
        summary["status"] = "FAIL"
        summary["reason"] = "no selection-capture files found (no evidence is not a pass)"
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(json.dumps(summary, indent=2))
        return 1

    violations = []
    rows_checked = 0
    rows_full_budget = 0     # seq_len >= top_k and valid_length == top_k
    rows_short_ctx = 0       # seq_len < top_k (whole context selected)
    min_seq = 1 << 60
    max_seq = 0
    for f in files:
        rec = torch.load(f, weights_only=False)
        lens = rec["lengths"]      # [L, bs]
        seq_lens = rec.get("seq_lens")  # list[bs] or None
        L = int(lens.shape[0])
        B = int(lens.shape[1])
        for b in range(B):
            sl = int(seq_lens[b]) if seq_lens is not None else None
            if sl is not None:
                min_seq = min(min_seq, sl)
                max_seq = max(max_seq, sl)
            for layer in range(L):
                vl = int(lens[layer, b].item())
                rows_checked += 1
                if vl > top_k:
                    violations.append(
                        f"{os.path.basename(f)} layer={layer} row={b}: valid_length={vl} > top_k={top_k} (over-budget)"
                    )
                    continue
                if sl is None:
                    # Without seq_len we can still flag over-budget/zero; require >0.
                    if vl == 0:
                        violations.append(
                            f"{os.path.basename(f)} layer={layer} row={b}: valid_length=0 (no selection)"
                        )
                    continue
                if sl >= top_k:
                    if vl != top_k:
                        violations.append(
                            f"{os.path.basename(f)} layer={layer} row={b}: seq_len={sl}>=top_k but valid_length={vl}!=top_k (dense fallback or short select)"
                        )
                    else:
                        rows_full_budget += 1
                else:
                    # short context: valid_length should equal the full (sparse) context,
                    # i.e. <= sl and > 0. A full-dense degrade is not possible here since
                    # the whole context is < top_k.
                    if vl == 0 and sl > 0:
                        violations.append(
                            f"{os.path.basename(f)} layer={layer} row={b}: seq_len={sl}<top_k but valid_length=0"
                        )
                    else:
                        rows_short_ctx += 1

    summary["rows_checked"] = rows_checked
    summary["rows_full_sparse_budget(seq>=top_k,vl==top_k)"] = rows_full_budget
    summary["rows_short_context(seq<top_k)"] = rows_short_ctx
    summary["seq_len_range"] = [None if min_seq == (1 << 60) else min_seq, max_seq]
    summary["num_violations"] = len(violations)
    summary["violations_sample"] = violations[:20]
    summary["status"] = "PASS" if not violations else "FAIL"
    if violations:
        summary["reason"] = f"{len(violations)} dense-fallback/over-budget/zero-selection violation(s)"

    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
