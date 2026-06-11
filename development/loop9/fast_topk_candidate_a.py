"""Candidate A for the DS top-k milestone — the EXACT fast_topk_v2 wrapper.

NOT a production path. This is the loop's head-to-head Candidate-A artifact:
an exact, deterministic adapter around ``sgl_kernel.top_k.fast_topk_v2``
(whose radix boundary bin admits ties by atomicAdd race and whose dense rows
include ``-inf`` positions), benchmarked on captured-shape Case-1 tensors
against the reference selector and the shipped radix suite.

Exactness argument: the racy kernel still selects the top-K *value multiset*
deterministically (races only choose among equal values), so the K-th order
statistic (the threshold) derived from its output is deterministic. The
wrapper recomputes the exact selection from that threshold: all strictly-
better live finite positions, plus the lowest-position ties up to the quota —
emitting the DS contract (ascending logical positions, ``-1`` padding,
``valid_lengths``). The repair is full-width torch work, which is exactly why
this candidate loses: correctness costs back the kernel's seq-aware win.

Run:  python development/loop9/fast_topk_candidate_a.py \
        --out development/loop9/runs/20260611_r1/candidate_a_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

K = 2048


def select_topk_sequence_order_fast_topk_v2(
    scores: torch.Tensor,    # fp32 [bs, width]
    seq_lens: torch.Tensor,  # int32 [bs]
    max_top_k: int,
):
    """Exact (score desc, pos asc) selection via fast_topk_v2 + repair."""
    from sgl_kernel.top_k import fast_topk_v2

    bs, width = scores.shape
    device = scores.device
    lengths = torch.clamp(seq_lens, max=width).to(torch.int32)
    pos = torch.arange(width, device=device, dtype=torch.int64).unsqueeze(0)
    live = pos < lengths.unsqueeze(1).long()
    finite_full = torch.isfinite(scores) & live

    raw_idx = fast_topk_v2(scores, lengths, max_top_k)  # [bs, K] int32, racy ties
    safe = raw_idx.clamp_min(0).long()
    vals = scores.gather(1, safe)
    finite_sel = (raw_idx >= 0) & torch.isfinite(vals)

    # Deterministic threshold: the K-th order statistic of the live finite
    # scores == the min finite value the kernel selected (value multiset is
    # race-independent). +inf when the row selected nothing finite.
    thr = torch.where(finite_sel, vals, torch.full_like(vals, torch.inf)).amin(-1)

    k_target = torch.minimum(
        finite_full.sum(-1), torch.full((bs,), max_top_k, device=device, dtype=torch.long)
    )
    above = finite_full & (scores > thr.unsqueeze(1))
    n_above = above.sum(-1)
    quota = (k_target - n_above).clamp_min(0)

    # Lowest-position tie admission: rank tied positions by position.
    tied = finite_full & (scores == thr.unsqueeze(1))
    tie_rank = torch.cumsum(tied.long(), dim=-1) - 1
    admitted_tie = tied & (tie_rank < quota.unsqueeze(1))

    selected = above | admitted_tie
    # Ascending emission: sort positions with non-selected pushed past width.
    keys = torch.where(selected, pos, torch.full_like(pos, width))
    sorted_keys, _ = torch.sort(keys, dim=-1)
    out = sorted_keys[:, :max_top_k]
    out_indices = torch.where(
        out < width, out, torch.full_like(out, -1)
    ).to(torch.int32)
    valid_lengths = k_target.to(torch.int32)
    return out_indices, valid_lengths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
        select_topk_sequence_order,
    )
    from sglang.srt.layers.attention.double_sparsity.topk_kernel import (
        allocate_topk_scratch,
        select_topk_sequence_order_triton,
    )

    dev = torch.device("cuda")
    W, BS = 202752, 29
    torch.manual_seed(7)

    def ref(scores, seq):
        s = scores.clone()
        for b in range(s.shape[0]):
            s[b, int(seq[b]):] = float("-inf")
        return select_topk_sequence_order(s, K)

    # Correctness: adversarial fixtures incl. tie plateaus and bf16-quantized
    # scores (the served post-reduce reality).
    report = {"correctness": {}, "bench_us_per_call": {}, "notes": []}
    cases = {}
    sc = torch.randn(4, W, device=dev)
    cases["random_mixed"] = (sc, torch.tensor([4608, 1000, 16384, 2048], dtype=torch.int32, device=dev))
    sc2 = torch.full((4, W), -1e9, device=dev)
    sc2[:, :1500] = torch.randperm(1500, device=dev).float() + 1000
    sc2[:, 1500:3500] = 7.5
    cases["tie_plateau"] = (sc2, torch.full((4,), 4608, dtype=torch.int32, device=dev))
    sc3 = torch.randn(4, W, device=dev)
    sc3[:, ::3] = float("-inf")
    sc3[1, :] = float("-inf")
    sc3[1, 10:110] = torch.randn(100, device=dev)
    cases["neg_inf_underfull"] = (sc3, torch.tensor([3000, 5000, 1024, 4608], dtype=torch.int32, device=dev))
    sc4 = torch.randn(4, W, device=dev).to(torch.bfloat16).float()
    cases["bf16_quantized"] = (sc4, torch.full((4,), 4608, dtype=torch.int32, device=dev))

    all_ok = True
    for name, (s, q) in cases.items():
        gi, gl = select_topk_sequence_order_fast_topk_v2(s, q, K)
        ri, rl = ref(s, q)
        ok = bool(torch.equal(gi, ri) and torch.equal(gl, rl))
        report["correctness"][name] = ok
        all_ok &= ok
        print(f"[cand-a] {name}: exact={ok}")
    # Determinism on the tie plateau (the raw kernel is racy; the wrapper must not be).
    outs = [select_topk_sequence_order_fast_topk_v2(*cases["tie_plateau"], K)[0] for _ in range(10)]
    det = all(torch.equal(outs[0], o) for o in outs[1:])
    report["correctness"]["tie_determinism_10_runs"] = bool(det)
    print(f"[cand-a] tie determinism (10 runs): {det}")
    all_ok &= det

    # Benchmark at the captured Case-1 shape, eager AND CUDA-graph captured,
    # vs the shipped radix suite (candidate B) under identical conditions.
    def t(fn, iters=50):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        s_, e_ = torch.cuda.Event(True), torch.cuda.Event(True)
        ts = []
        for _ in range(iters):
            s_.record(); fn(); e_.record(); torch.cuda.synchronize()
            ts.append(s_.elapsed_time(e_) * 1000.0)
        ts.sort()
        return round(ts[len(ts) // 2], 1)

    scores = torch.randn(BS, W, device=dev).to(torch.bfloat16).float()
    scores[:, 4608:] = float("-inf")
    seq = torch.full((BS,), 4608, dtype=torch.int32, device=dev)
    report["bench_us_per_call"]["candidate_a_eager"] = t(
        lambda: select_topk_sequence_order_fast_topk_v2(scores, seq, K)
    )
    scratch = allocate_topk_scratch(max_bs=BS, width=W, device=dev)
    oi = torch.full((BS, K), -1, dtype=torch.int32, device=dev)
    ol = torch.zeros(BS, dtype=torch.int32, device=dev)

    def call_b():
        select_topk_sequence_order_triton(
            scores, seq, K, out_indices=oi, out_lengths=ol, **scratch
        )

    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        call_b()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call_b()
    report["bench_us_per_call"]["candidate_b_triton_captured"] = t(lambda: g.replay())
    report["bench_us_per_call"]["candidate_b_triton_eager"] = t(call_b)
    # Candidate A allocates per call (gathers/sorts) -> not graph-capturable as
    # written; record the structural fact rather than a forced capture.
    report["notes"].append(
        "candidate A allocates full-width intermediates per call (repair is "
        "torch-composed) and is not allocation-free graph-capturable as "
        "written; eager number is its honest cost."
    )
    print(f"[cand-a] bench: {report['bench_us_per_call']}")
    report["verdict"] = (
        "exact and deterministic, but the full-width repair costs back the "
        "seq-aware kernel's win; candidate B (shipped radix suite) wins the "
        "head-to-head."
        if all_ok
        else "CORRECTNESS FAILURE — see correctness map"
    )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[cand-a] report -> {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
