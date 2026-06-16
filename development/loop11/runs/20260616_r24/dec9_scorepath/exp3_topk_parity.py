"""DEC-9 SCORE-PATH exp-3 — radix top-k vs fallback top-k cutoff ordering.

Question (owner): for the SAME score tensor, do the radix/graph-safe top-k path
and the eager/fallback top-k produce the SAME top-2048 cutoff ordering and
membership? If they diverge, the top-k path itself is a culprit (a DS-side fix).

Three top-k implementations are compared on the SAME [bs, width] fp32 score
tensor at the production max_top_k=2048:

  RADIX     select_topk_sequence_order_triton — the Triton radix kernel the
            graph-safe path uses when radix_topk_scratch is present (production).
  FALLBACK  the inline torch.topk pipeline retrieve_topk_graph_safe runs when
            radix_topk_scratch is None (the eager/no-radix-scratch branch).
            Reconstructed here verbatim from selection_kernel.
  REFERENCE select_topk_sequence_order — the canonical torch (score desc, pos
            asc) selection.

Inputs: the REAL post-reduce score rows captured during the bf16 cold pass
(rank0 + rank4, one row per DS layer), upcast to fp32 — exactly the tensors the
production top-k consumed. Plus a synthetic tie-stress tensor (many exact ties
at the cutoff) so the comparison is not blind to tie-break divergence even if
the real rows happen to have few ties.

Compares: per (rank, layer) the set of selected logical positions (membership)
and, for the positions both select, the ascending-order sequence. Reports any
divergence with counts and example positions. Raw numbers only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

REPO = "/sgl-workspace/sglang"
sys.path.insert(0, os.path.join(REPO, "python"))

from sglang.srt.layers.attention.double_sparsity.selection_kernel import (  # noqa: E402
    select_topk_sequence_order,
)
from sglang.srt.layers.attention.double_sparsity.topk_kernel import (  # noqa: E402
    allocate_topk_scratch,
    select_topk_sequence_order_triton,
)

MAX_TOP_K = 2048


def _radix_select(scores: torch.Tensor, seq_lens: torch.Tensor, block: int = 1024):
    """Production radix path. Returns ascending int32 [bs, MAX_TOP_K], lengths."""
    bs, width = scores.shape
    scratch = allocate_topk_scratch(
        max_bs=bs, width=width, block=block, device=scores.device
    )
    out_indices = torch.full(
        (bs, MAX_TOP_K), -1, dtype=torch.int32, device=scores.device
    )
    out_lengths = torch.zeros(bs, dtype=torch.int32, device=scores.device)
    select_topk_sequence_order_triton(
        scores,
        seq_lens,
        MAX_TOP_K,
        out_indices=out_indices,
        out_lengths=out_lengths,
        block=block,
        **scratch,
    )
    return out_indices, out_lengths


def _fallback_select(scores: torch.Tensor, max_seq_len: int):
    """The inline torch.topk pipeline from retrieve_topk_graph_safe (radix None).

    Verbatim reconstruction of steps 1-5 of the else-branch: top-k largest
    unsorted, sentinel -inf -> max_seq_len, ascending sort, sentinel -> -1,
    searchsorted for valid_lengths.
    """
    bs = scores.shape[0]
    device = scores.device
    effective_k = min(MAX_TOP_K, max_seq_len)

    topk_vals, topk_idx = torch.topk(
        scores, effective_k, dim=-1, largest=True, sorted=False
    )
    invalid = torch.isneginf(topk_vals)
    topk_idx = topk_idx.masked_fill(invalid, max_seq_len)
    sorted_vals, _ = torch.topk(
        topk_idx, effective_k, dim=-1, largest=False, sorted=True
    )
    out_indices = torch.full((bs, MAX_TOP_K), -1, dtype=torch.int32, device=device)
    out_indices[:, :effective_k] = sorted_vals.to(torch.int32)
    ge = sorted_vals >= max_seq_len
    out_indices[:, :effective_k] = out_indices[:, :effective_k].masked_fill(ge, -1)
    boundary = torch.full((bs, 1), max_seq_len, dtype=sorted_vals.dtype, device=device)
    valid_i64 = torch.searchsorted(sorted_vals, boundary, right=False)
    out_lengths = valid_i64.squeeze(-1).to(torch.int32)
    return out_indices, out_lengths


def _members(out_indices: torch.Tensor, out_lengths: torch.Tensor, b: int):
    n = int(out_lengths[b].item())
    row = out_indices[b].to(torch.int64).tolist()
    return [x for x in row if x != -1][:n]


def _compare(name_a, idx_a, len_a, name_b, idx_b, len_b, bs):
    """Per-row membership + ascending-order comparison. Returns a dict."""
    rep = {
        "pair": f"{name_a}_vs_{name_b}",
        "rows": bs,
        "rows_membership_identical": 0,
        "rows_order_identical": 0,
        "total_member_symdiff": 0,
        "first_divergence": None,
    }
    for b in range(bs):
        ma = _members(idx_a, len_a, b)
        mb = _members(idx_b, len_b, b)
        sa, sb = set(ma), set(mb)
        symdiff = len(sa ^ sb)
        rep["total_member_symdiff"] += symdiff
        if symdiff == 0:
            rep["rows_membership_identical"] += 1
            # ascending order is the emitted sequence (both should be ascending)
            if ma == mb:
                rep["rows_order_identical"] += 1
            elif rep["first_divergence"] is None:
                rep["first_divergence"] = {
                    "row": b,
                    "kind": "order_differs_same_members",
                    "len_a": len(ma),
                    "len_b": len(mb),
                }
        elif rep["first_divergence"] is None:
            only_a = sorted(sa - sb)[:5]
            only_b = sorted(sb - sa)[:5]
            rep["first_divergence"] = {
                "row": b,
                "kind": "membership_differs",
                "symdiff": symdiff,
                "len_a": len(ma),
                "len_b": len(mb),
                f"only_in_{name_a}": only_a,
                f"only_in_{name_b}": only_b,
            }
    return rep


def _load_real_rows(score_dir, ranks):
    """Stack the captured post-reduce fp32 score rows into one padded batch.

    Returns (scores [N, W] fp32, seq_lens [N] int32, meta [N]) on cuda, where
    each row is one (rank, layer) captured row padded with -inf to the common
    width. seq_len is the live window so the radix path bounds correctly.
    """
    rows = []
    seqs = []
    meta = []
    for rk in ranks:
        files = sorted(glob.glob(os.path.join(score_dir, f"rank{rk}_req*_layer*.pt")))
        for f in files:
            rec = torch.load(f, weights_only=False)
            sc = rec["scores"].to(torch.float32)  # [seq_len]
            rows.append(sc)
            seqs.append(int(rec["seq_len"]))
            meta.append({"rank": rk, "layer": int(rec["layer_id"]), "seq_len": int(rec["seq_len"])})
    if not rows:
        return None, None, None
    width = max(r.numel() for r in rows)
    N = len(rows)
    scores = torch.full((N, width), float("-inf"), dtype=torch.float32)
    for i, r in enumerate(rows):
        scores[i, : r.numel()] = r
    seq_lens = torch.tensor(seqs, dtype=torch.int32)
    return scores.cuda(), seq_lens.cuda(), meta


def _synthetic_tie_stress():
    """Tensor engineered so the k=2048 cutoff lands INSIDE a tie plateau.

    Layout (width=4096): the first 1000 positions hold distinct DESCENDING high
    scores (always kept). Positions [1000, 4096) are ALL the SAME value (a 3096-
    wide plateau). With k=2048, exactly 1048 of the 3096 tied positions must be
    chosen — the cutoff falls squarely inside the plateau, so the choice is
    decided purely by the tie-break. A (score desc, pos asc) selector takes
    positions 1000..2047; torch.topk(sorted=False) is free to take any 1048 of
    the tied positions. If radix and fallback disagree on WHICH tied positions,
    membership diverges here.
    """
    width = 4096
    bs = 4
    base = torch.full((bs, width), 0.5, dtype=torch.float32)
    head = torch.linspace(100.0, 10.0, 1000)
    base[:, :1000] = head.unsqueeze(0)
    seq_lens = torch.full((bs,), width, dtype=torch.int32)
    return base.contiguous().cuda(), seq_lens.cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold-score-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ranks", default="0,4")
    args = ap.parse_args()
    ranks = [int(x) for x in args.ranks.split(",")]
    os.makedirs(args.outdir, exist_ok=True)

    result = {
        "probe": "DEC9_scorepath_exp3_radix_vs_fallback_topk",
        "max_top_k": MAX_TOP_K,
        "comparisons": [],
    }

    # ---- real captured score rows ----
    scores, seq_lens, meta = _load_real_rows(args.cold_score_dir, ranks)
    if scores is not None:
        bs, width = scores.shape
        result["real_rows"] = {"n_rows": bs, "width": width, "ranks": ranks}
        max_seq_len = width
        r_idx, r_len = _radix_select(scores, seq_lens)
        f_idx, f_len = _fallback_select(scores, max_seq_len)
        ref_idx, ref_len = select_topk_sequence_order(scores, MAX_TOP_K)
        result["comparisons"].append(
            {"dataset": "real", **_compare("radix", r_idx, r_len, "fallback", f_idx, f_len, bs)}
        )
        result["comparisons"].append(
            {"dataset": "real", **_compare("radix", r_idx, r_len, "reference", ref_idx, ref_len, bs)}
        )
        # also report per-row valid_length agreement
        result["real_lengths"] = {
            "radix_vs_fallback_len_mismatches": int(
                (r_len != f_len).sum().item()
            ),
            "radix_vs_reference_len_mismatches": int(
                (r_len != ref_len).sum().item()
            ),
        }
    else:
        result["real_rows"] = "NONE_FOUND"

    # ---- synthetic tie-stress ----
    s_scores, s_seq = _synthetic_tie_stress()
    sb = s_scores.shape[0]
    s_w = s_scores.shape[1]
    sr_idx, sr_len = _radix_select(s_scores, s_seq)
    sf_idx, sf_len = _fallback_select(s_scores, s_w)
    sref_idx, sref_len = select_topk_sequence_order(s_scores, MAX_TOP_K)
    result["synthetic_tie_stress"] = {
        "width": s_w,
        "bs": sb,
        "note": "plateau of equal scores across the k=2048 cutoff forces tie-break",
    }
    result["comparisons"].append(
        {"dataset": "synthetic_tie", **_compare("radix", sr_idx, sr_len, "fallback", sf_idx, sf_len, sb)}
    )
    result["comparisons"].append(
        {"dataset": "synthetic_tie", **_compare("radix", sr_idx, sr_len, "reference", sref_idx, sref_len, sb)}
    )

    # verdict
    any_membership_div = any(
        c["total_member_symdiff"] > 0 for c in result["comparisons"]
    )
    any_order_div = any(
        c["rows_membership_identical"] != c["rows_order_identical"]
        for c in result["comparisons"]
    )
    if any_membership_div:
        result["verdict"] = "DIVERGE_MEMBERSHIP — top-k paths select different members on the same scores"
    elif any_order_div:
        result["verdict"] = "DIVERGE_ORDER — same members, different ascending order"
    else:
        result["verdict"] = "IDENTICAL — radix, fallback, and reference top-k agree on membership and order"

    with open(os.path.join(args.outdir, "exp3_topk_parity.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
