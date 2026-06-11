"""Loop-10 AC-1.2 exact candidate — head-split logical score, prototype.

The exact-floor harness showed the landed kernel at ~20-21 us/call isolated
(at the budget) with the real miss living in cold-cache effective bandwidth.
More memory-level parallelism is the one structural lever left in the exact
regime: split the per-position HEAD loop across programs — grid
(bs, workers * HEAD_SPLIT) — and combine with tl.atomic_max, which is
ORDER-INDEPENDENT and therefore bit-exact for the served head_agg="max"
(mean would need order-dependent atomic_add and stays on the landed path).
The output buffer is -inf-initialized by a tiny fill kernel captured into
the same graph; invalid/unwritten positions contribute -inf (a no-op).

This prototype measures, in ONE process with interleaved replay rounds:
  production tb512  vs  fill + head-split variant (HS in {2, 4})
at the op-point shapes, both layouts, and bitwise-verifies the head-split
output against production on mixed rows first. Landing only happens if the
isolated win is decisive (>=10%); the real profile remains the binding
verdict.

Run: python development/loop10/task11_headsplit_prototype.py \
       --out development/loop10/runs/20260611_task11/headsplit_proto
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import triton
import triton.language as tl

WIDTH, BS, H, LABEL_DIM, HEAD_DIM, TABLE_T = 5120, 29, 8, 32, 192, 142272
ROUNDS = 100
REPLAYS_PER_ROUND = 4
WARMUP_ROUNDS = 10


@triton.jit
def _fill_neg_inf_kernel(out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.full((BLOCK,), float("-inf"), tl.float32),
             mask=offs < n)


@triton.jit
def _headsplit_logical_score_kernel(
    q_ptr, ch_sel_ptr, ch_w_ptr, sig_ptr, written_ptr, rpi_ptr, rtt_ptr,
    sl_ptr, out_ptr,
    num_heads: tl.constexpr,
    max_seq_len: tl.constexpr,
    label_dim: tl.constexpr,
    max_pool_len: tl.constexpr,
    max_tokens: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    ch_sel_stride_h: tl.constexpr,
    ch_w_stride_h: tl.constexpr,
    sig_stride_t: tl.constexpr,
    sig_stride_h: tl.constexpr,
    rtt_stride_p: tl.constexpr,
    out_stride_b: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    LABEL_DIM_POW2: tl.constexpr,
    WORKERS: tl.constexpr,
    HEAD_SPLIT: tl.constexpr,
):
    # Production structure with the head loop sharded across HEAD_SPLIT
    # program groups; per-position partial max combined via atomic_max
    # (order-independent => bit-exact for max aggregation). Scoring math per
    # (position, head) is identical to the production kernel's raw
    # channel-dot path.
    batch_id = tl.program_id(0)
    lane = tl.program_id(1)
    worker = lane // HEAD_SPLIT
    split = lane % HEAD_SPLIT
    heads_per_split: tl.constexpr = num_heads // HEAD_SPLIT

    seq_len_i = tl.load(sl_ptr + batch_id).to(tl.int32)
    n_live = tl.minimum(seq_len_i, max_seq_len)
    live_blocks = (n_live + TOKEN_BLOCK - 1) // TOKEN_BLOCK
    pool_idx = tl.load(rpi_ptr + batch_id).to(tl.int64)
    d_offs = tl.arange(0, LABEL_DIM_POW2)
    d_mask = d_offs < label_dim

    for tok_blk in range(worker, live_blocks, WORKERS):
        tok_offs = tok_blk * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
        in_range = tok_offs < max_seq_len
        pos_valid = in_range & (tok_offs < seq_len_i)
        safe_tok = tl.minimum(tok_offs, max_pool_len - 1)
        phys = tl.load(
            rtt_ptr + pool_idx * rtt_stride_p + safe_tok, mask=in_range, other=0
        ).to(tl.int64)
        safe_phys = tl.minimum(tl.maximum(phys, 0), max_tokens - 1)
        written = tl.load(written_ptr + safe_phys, mask=in_range, other=0).to(
            tl.int1
        )
        valid = pos_valid & written

        acc = tl.full((TOKEN_BLOCK,), float("-inf"), dtype=tl.float32)
        for hi in range(heads_per_split):
            h = split * heads_per_split + hi
            sel_h = tl.load(
                ch_sel_ptr + h * ch_sel_stride_h + d_offs, mask=d_mask, other=0
            ).to(tl.int64)
            w_h = tl.load(
                ch_w_ptr + h * ch_w_stride_h + d_offs, mask=d_mask, other=0.0
            ).to(tl.float32)
            q_base = q_ptr + batch_id * q_stride_b + h * q_stride_h
            q_h = tl.load(q_base + sel_h, mask=d_mask, other=0.0).to(tl.float32)
            q_proj_h = q_h * w_h
            sig_offs = (
                safe_phys[:, None] * sig_stride_t
                + h * sig_stride_h
                + d_offs[None, :]
            )
            sig_block = tl.load(
                sig_ptr + sig_offs,
                mask=in_range[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            dot = tl.sum(q_proj_h[None, :] * sig_block, axis=1)
            acc = tl.where(dot > acc, dot, acc)

        contrib = tl.where(valid, acc, tl.full(acc.shape, float("-inf"),
                                               dtype=tl.float32))
        tl.atomic_max(
            out_ptr + batch_id * out_stride_b + tok_offs, contrib, mask=in_range
        )


def _graph_for(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def _interleaved(pairs):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    times = {k: [] for k, _ in pairs}
    for _ in range(WARMUP_ROUNDS):
        for _, g in pairs:
            g.replay()
    torch.cuda.synchronize()
    for _ in range(ROUNDS):
        for key, g in pairs:
            s.record()
            for _ in range(REPLAYS_PER_ROUND):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            times[key].append(s.elapsed_time(e) * 1000.0 / REPLAYS_PER_ROUND)
    out = {}
    for k in times:
        times[k].sort()
        out[k] = times[k][len(times[k]) // 2]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    torch.manual_seed(20260611)

    from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
        _logical_score_triton,
    )

    q = torch.randn(BS, H, HEAD_DIM, dtype=torch.float32, device=device)
    ch_sel = torch.randint(0, HEAD_DIM, (H, LABEL_DIM), dtype=torch.int32, device=device)
    ch_w = torch.randn(H, LABEL_DIM, dtype=torch.float32, device=device)
    sig = torch.randn(TABLE_T, H, LABEL_DIM, dtype=torch.float16, device=device)
    written = torch.ones(TABLE_T, dtype=torch.bool, device=device)
    rpi = torch.arange(BS, dtype=torch.int32, device=device)
    out_prod = torch.empty(BS, WIDTH, dtype=torch.float32, device=device)
    out_hs = torch.empty(BS, WIDTH, dtype=torch.float32, device=device)
    nblocks = (WIDTH + 511) // 512
    workers = min(128, nblocks)
    n_elem = BS * WIDTH
    fill_grid = ((n_elem + 1023) // 1024,)

    def hs_call(rtt, seq, hs):
        _fill_neg_inf_kernel[fill_grid](out_hs, n=n_elem, BLOCK=1024)
        _headsplit_logical_score_kernel[(BS, workers * hs)](
            q, ch_sel, ch_w, sig, written, rpi, rtt, seq, out_hs,
            num_heads=H, max_seq_len=WIDTH, label_dim=LABEL_DIM,
            max_pool_len=WIDTH, max_tokens=TABLE_T,
            q_stride_b=q.stride(0), q_stride_h=q.stride(1),
            ch_sel_stride_h=ch_sel.stride(0), ch_w_stride_h=ch_w.stride(0),
            sig_stride_t=sig.stride(0), sig_stride_h=sig.stride(1),
            rtt_stride_p=rtt.stride(0), out_stride_b=out_hs.stride(0),
            TOKEN_BLOCK=512, LABEL_DIM_POW2=32, WORKERS=workers, HEAD_SPLIT=hs,
        )

    def prod_call(rtt, seq):
        _logical_score_triton(
            q, ch_sel, ch_w, sig, written, rpi, rtt, seq, out_prod, WIDTH,
            scale_layer=None, token_block=512, workers=128,
            store_dead_neg_inf=False, scorer_norm="off", head_agg="max",
            hybrid_threshold=8192,
        )

    # Bitwise verification on mixed rows (random layout).
    rtt_v = torch.randint(0, TABLE_T, (BS, WIDTH), dtype=torch.int32, device=device)
    mixed = ([4096, 4608, 5120, 547, 2886] * 6)[:BS]
    seq_v = torch.tensor(mixed, dtype=torch.int32, device=device)
    report = {"bitwise_vs_production": {}, "replay_us_per_call": {}}
    prod_call(rtt_v, seq_v)
    torch.cuda.synchronize()
    for hs in (2, 4):
        hs_call(rtt_v, seq_v, hs)
        torch.cuda.synchronize()
        # Compare the LIVE region only: production with store_dead=False
        # leaves dead positions stale, head-split leaves them -inf — both
        # unread by the radix selector; live rows must match bit-exactly.
        ok = True
        for b in range(BS):
            n = mixed[b]
            if not torch.equal(out_prod[b, :n], out_hs[b, :n]):
                ok = False
                break
        report["bitwise_vs_production"][f"hs{hs}"] = bool(ok)
        print(f"[headsplit] bitwise hs={hs}: {'OK' if ok else 'DIFF'}", flush=True)

    seq_op = torch.full((BS,), 4608, dtype=torch.int32, device=device)
    for layout in ("random", "page64"):
        if layout == "page64":
            pages = WIDTH // 64
            rtt = torch.empty(BS, WIDTH, dtype=torch.int32, device=device)
            for b in range(BS):
                st = torch.randint(0, TABLE_T // 64, (pages,), device=device) * 64
                rtt[b] = (st[:, None] + torch.arange(64, device=device)[None, :]
                          ).reshape(-1)[:WIDTH]
        else:
            rtt = rtt_v
        pairs = [("production", _graph_for(lambda: prod_call(rtt, seq_op)))]
        for hs in (2, 4):
            pairs.append((f"headsplit{hs}",
                          _graph_for(lambda hs=hs: hs_call(rtt, seq_op, hs))))
        med = _interleaved(pairs)
        report["replay_us_per_call"][layout] = {
            k: round(v, 2) for k, v in med.items()
        }
        print(f"[headsplit] {layout}: " + ", ".join(
            f"{k}={v:.2f}" for k, v in med.items()), flush=True)

    if args.out:
        path = f"{args.out}.json"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[headsplit] report -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
