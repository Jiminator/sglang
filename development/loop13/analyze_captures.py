#!/usr/bin/env python3
"""Offline analysis of Double Sparsity capture artifacts (loop13 cheap controls).

Consumes the .pt dumps emitted by the DS capture instruments:
  - score_capture     -> .sglang_ds_scorecap/rank{tp}_req{req:04d}_layer{layer:03d}.pt
  - selection_capture -> .sglang_ds_selcap/rank{tp}_step{step:05d}.pt

Computes the two decisive cheap micro-tests:
  (A) TP head-aggregation equivalence: the served head_agg="max" does local-max
      over each rank's heads (= pre_reduce_scores) then SUM across the TP group.
      We compare the resulting top-k index set under SUM (served) vs MAX
      (= a true global max over all heads) vs MEAN, on the captured score rows.
  (B) Radix-topk vs torch.topk selected-index equivalence: the production
      selection (selection_capture indices) vs an exact torch.topk over the same
      final post-reduce score row (score_capture "scores"). Identity retires the
      "approximate radix top-k" regression suspect by inspection.

Designed for single-request (bs=1) captures so the score-row req maps
unambiguously to selection row b=0. Read-only; prints a structured report and
writes JSON to evidence/.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import torch

_SCORECAP_RE = re.compile(r"rank(\d+)_req(\d+)_layer(\d+)\.pt$")
_SELCAP_RE = re.compile(r"rank(\d+)_step(\d+)\.pt$")


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _topk_set(scores, k):
    """Indices of the top-k scores as a Python set (k clamped to len)."""
    k = int(min(k, scores.numel()))
    if k <= 0:
        return set()
    vals, idx = torch.topk(scores.float(), k)
    return set(int(i) for i in idx.tolist())


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def load_score_captures(scorecap_dir):
    """-> dict[(req,layer)] = {rank: record}."""
    out = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(scorecap_dir, "rank*_req*_layer*.pt"))):
        m = _SCORECAP_RE.search(os.path.basename(p))
        if not m:
            continue
        rank, req, layer = int(m.group(1)), int(m.group(2)), int(m.group(3))
        out[(req, layer)][rank] = _load(p)
    return out


def load_selection_captures(selcap_dir):
    """-> list of records (per rank/step). We use rank0 records."""
    recs = []
    for p in sorted(glob.glob(os.path.join(selcap_dir, "rank*_step*.pt"))):
        m = _SELCAP_RE.search(os.path.basename(p))
        if not m:
            continue
        rank, step = int(m.group(1)), int(m.group(2))
        if rank != 0:
            continue
        r = _load(p)
        r["_rank"], r["_step"] = rank, step
        recs.append(r)
    return recs


def head_agg_test(score_caps, top_k):
    """(A) SUM (served) vs MAX vs MEAN top-k index-set agreement on captured rows."""
    rows = []
    for (req, layer), per_rank in sorted(score_caps.items()):
        ranks = sorted(per_rank)
        pre = [per_rank[r].get("pre_reduce_scores") for r in ranks]
        if any(x is None for x in pre):
            continue  # pre-reduce row not captured for this (req,layer)
        pre = torch.stack([p.float().reshape(-1) for p in pre], dim=0)  # [R, S]
        seq_len = pre.shape[1]
        k = int(min(top_k, seq_len))
        served = pre.sum(dim=0)            # cross-TP SUM of local maxima = served score
        gmax = pre.max(dim=0).values       # max of local maxima = global max over heads
        gmean = pre.mean(dim=0)
        s_set, m_set, mean_set = _topk_set(served, k), _topk_set(gmax, k), _topk_set(gmean, k)
        # sanity: served SUM row should match the post-reduce "scores" if present
        post = per_rank[ranks[0]].get("scores")
        sum_eq_post = None
        if post is not None and post.numel() == served.numel():
            sum_eq_post = bool(torch.allclose(served, post.float(), rtol=1e-3, atol=1e-3))
        rows.append({
            "req": req, "layer": layer, "seq_len": seq_len, "k": k, "ranks": len(ranks),
            "sum_vs_max_jaccard": round(_jaccard(s_set, m_set), 4),
            "sum_vs_mean_jaccard": round(_jaccard(s_set, mean_set), 4),
            "sum_eq_max_topk": s_set == m_set,
            "served_sum_matches_post_reduce": sum_eq_post,
        })
    return rows


def selected_index_equivalence(score_caps, sel_caps, top_k):
    """(B) production radix top-k (selection_capture) vs exact torch.topk on the
    same post-reduce score row (score_capture), joined EXACTLY on
    ``(req_pool_index, layer)`` via the selection record's ``req_pool_indices``.

    Returns ``(rows, unmatched)``; ``unmatched`` counts selected rows with no
    matching score row (a non-zero count means the join is incomplete and the
    result must NOT be trusted)."""
    rows = []
    unmatched = 0
    for rec in sel_caps:
        idx = rec.get("indices")        # [num_layers, bs, max_top_k]
        lens = rec.get("lengths")       # [num_layers, bs]
        rpi = rec.get("req_pool_indices")  # [bs]
        if idx is None or rpi is None:
            continue
        num_layers, bs = idx.shape[0], idx.shape[1]
        for layer in range(num_layers):
            for b in range(bs):
                pool_idx = int(rpi[b])
                per_rank = score_caps.get((pool_idx, layer))
                if not per_rank:
                    unmatched += 1
                    continue
                scores = per_rank[min(per_rank)].get("scores")
                if scores is None:
                    unmatched += 1
                    continue
                scores = scores.float().reshape(-1)
                prod = idx[layer, b]
                length = int(lens[layer, b]) if lens is not None else int((prod >= 0).sum())
                prod_set = set(int(i) for i in prod[:length].tolist() if int(i) >= 0)
                # exact torch.topk reference at the same k on the same score row
                ref_set = _topk_set(scores, min(top_k, length if length > 0 else top_k))
                rows.append({
                    "req_pool_index": pool_idx, "layer": layer, "step": rec.get("_step"),
                    "prod_k": len(prod_set), "ref_k": len(ref_set),
                    "jaccard": round(_jaccard(prod_set, ref_set), 4),
                    "identical": prod_set == ref_set,
                })
    return rows, unmatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorecap-dir", default=".sglang_ds_scorecap")
    ap.add_argument("--selcap-dir", default=".sglang_ds_selcap")
    ap.add_argument("--top-k", type=int, default=2048)
    ap.add_argument("--out", default="evidence/cheap_controls.json")
    args = ap.parse_args()

    score_caps = load_score_captures(args.scorecap_dir)
    sel_caps = load_selection_captures(args.selcap_dir)
    print(f"loaded score-capture groups: {len(score_caps)} ; selection-capture rank0 records: {len(sel_caps)}")

    head = head_agg_test(score_caps, args.top_k)
    equiv, unmatched = selected_index_equivalence(score_caps, sel_caps, args.top_k)

    n_ident = sum(1 for r in equiv if r["identical"])
    min_jac = min((r["jaccard"] for r in equiv), default=None)
    report = {
        "top_k": args.top_k,
        "n_score_groups": len(score_caps),
        "n_selection_records": len(sel_caps),
        "join": "exact (req_pool_index, layer) via selection_capture.req_pool_indices",
        "head_agg_test": head,
        "selected_index_equivalence": equiv,
        "summary": {
            "AC_2_3_equiv_rows": len(equiv),
            "AC_2_3_unmatched_rows": unmatched,
            "AC_2_3_radix_eq_torch_topk_all": (n_ident == len(equiv) and len(equiv) > 0),
            "AC_2_3_identical_rows": f"{n_ident}/{len(equiv)}",
            "AC_2_3_min_jaccard": min_jac,
            "AC_2_2_head_agg_rows_with_pre_reduce": len(head),
            "AC_2_2_served_sum_matches_post_reduce_all": (
                all(r["served_sum_matches_post_reduce"] for r in head
                    if r["served_sum_matches_post_reduce"] is not None) if head else None),
            "AC_2_2_note": ("head-agg interpretation depends on pre_reduce_scores semantics; "
                            "trust only if served_sum_matches_post_reduce is True"),
        },
    }
    # Fail-closed: a successful-looking artifact must NOT be produced from an
    # incomplete/stale/empty capture set. These are hard errors, not warnings.
    errors = []
    if len(score_caps) == 0:
        errors.append("zero score-capture groups loaded (capture dir empty/stale)")
    if len(equiv) == 0:
        errors.append("zero equivalence rows produced (no joinable selected rows)")
    if unmatched:
        errors.append(f"{unmatched} selected rows had NO matching score row (incomplete join)")
    if errors:
        report["FAILED"] = errors
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["summary"], indent=2))
    if errors:
        print("ANALYZER FAILED (fail-closed):", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        raise SystemExit(2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
