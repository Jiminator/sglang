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
    same post-reduce score row (score_capture). Assumes bs=1 captures."""
    # final post-reduce score row per (req, layer): identical across ranks; take rank0
    rows = []
    # Build a layer->(req, scores) lookup from score caps (rank0).
    score_by_layer = {}
    for (req, layer), per_rank in score_caps.items():
        r0 = per_rank.get(min(per_rank))
        sc = r0.get("scores")
        if sc is not None:
            score_by_layer.setdefault(layer, []).append((req, sc.float().reshape(-1)))
    for rec in sel_caps:
        idx = rec.get("indices")        # [num_layers, bs, max_top_k]
        lens = rec.get("lengths")       # [num_layers, bs]
        if idx is None:
            continue
        num_layers = idx.shape[0]
        for layer in range(num_layers):
            if layer not in score_by_layer:
                continue
            # bs=1 assumption: production selected set for row 0
            prod = idx[layer, 0]
            length = int(lens[layer, 0]) if lens is not None else int((prod >= 0).sum())
            prod_set = set(int(i) for i in prod[:length].tolist() if int(i) >= 0)
            for (req, scores) in score_by_layer[layer]:
                ref_set = _topk_set(scores, min(top_k, length if length > 0 else top_k))
                rows.append({
                    "layer": layer, "req": req, "step": rec.get("_step"),
                    "prod_k": len(prod_set), "ref_k": len(ref_set),
                    "jaccard": round(_jaccard(prod_set, ref_set), 4),
                    "identical": prod_set == ref_set,
                })
    return rows


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
    equiv = selected_index_equivalence(score_caps, sel_caps, args.top_k)

    report = {
        "top_k": args.top_k,
        "n_score_groups": len(score_caps),
        "n_selection_records": len(sel_caps),
        "head_agg_test": head,
        "selected_index_equivalence": equiv,
        "summary": {
            "head_agg_rows_with_pre_reduce": len(head),
            "head_agg_sum_eq_max_all": (all(r["sum_eq_max_topk"] for r in head) if head else None),
            "equiv_rows": len(equiv),
            "equiv_all_identical": (all(r["identical"] for r in equiv) if equiv else None),
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
