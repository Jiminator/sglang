"""Loop-10 AC-1.2 exact candidate — head/dim-major signature layout, prototype.

The measured frontier closed every same-layout kernel-structure lever; the
remaining exact direction (round-5 review) is the TABLE LAYOUT: today's
signatures are token-major `[T, H, D]`, so each (head, dim) lane reads
512-byte-strided token gathers; a head/dim-major `[H, D, T]` table makes
page-64-contiguous logical windows read CONTIGUOUS token runs per
(head, dim) lane — same fp16 values, different addressing.

This prototype measures, in ONE process with interleaved replay rounds:
  production tb512 (token-major)  vs  transposed-layout variants
at the op-point shapes, both slot layouts, WARM and COLD (an L2-thrashing
flush captured before the kernel in a second graph; cold kernel cost =
median(flush+kernel) − median(flush alone)) — because the binding miss is
cold-cache bandwidth, warm-isolated wins alone are not decision-grade.

Bitwise check FIRST on mixed rows: if the transposed reduction
(tl.sum over axis 0 of a [D_TILE, TB] tile, possibly chunk-accumulated)
is not bit-identical to production's [TB, D] axis-1 sum, the layout lever
is summation-order-changing and would fall in the VALUE-AFFECTING regime
(recall-gated) rather than exact — recorded either way.

Run: python development/loop10/task11_layout_prototype.py \
       --out development/loop10/runs/20260611_task11/layout_proto
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import triton
import triton.language as tl

WIDTH, BS, H, LABEL_DIM, HEAD_DIM, TABLE_T = 5120, 29, 8, 32, 192, 142272
ROUNDS = 80
REPLAYS_PER_ROUND = 4
WARMUP_ROUNDS = 10
FLUSH_MB = 96


@triton.jit
def _flush_l2_kernel(buf_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(buf_ptr + offs, mask=offs < n)
    tl.store(buf_ptr + offs, x + 1.0, mask=offs < n)


@triton.jit
def _transposed_logical_score_kernel(
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
    sig_stride_h: tl.constexpr,   # = D * T
    sig_stride_d: tl.constexpr,   # = T
    rtt_stride_p: tl.constexpr,
    out_stride_b: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    D_TILE: tl.constexpr,
    LABEL_DIM_POW2: tl.constexpr,
    WORKERS: tl.constexpr,
):
    # Production structure (persistent workers, live-block loop, identical
    # rtt/written indirection and per-position raw channel-dot + head max)
    # with the signature loads addressed head/dim-major: [D_TILE, TB] tiles
    # whose token axis is CONTIGUOUS for page-contiguous physical runs.
    batch_id = tl.program_id(0)
    worker = tl.program_id(1)
    seq_len_i = tl.load(sl_ptr + batch_id).to(tl.int32)
    n_live = tl.minimum(seq_len_i, max_seq_len)
    live_blocks = (n_live + TOKEN_BLOCK - 1) // TOKEN_BLOCK
    pool_idx = tl.load(rpi_ptr + batch_id).to(tl.int64)

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
        for h in range(num_heads):
            q_base = q_ptr + batch_id * q_stride_b + h * q_stride_h
            # Transposed [D, T] addressing forces the reduction into D_TILE
            # chunks, so q_proj is reloaded per chunk below (production projects
            # the full head width once and reduces over axis 1 in a single sum).
            dot = tl.zeros((TOKEN_BLOCK,), dtype=tl.float32)
            for d0 in range(0, label_dim, D_TILE):
                d_offs = d0 + tl.arange(0, D_TILE)
                # q_proj slice for this chunk (gathered q is small; reload).
                qp = tl.load(
                    q_base
                    + tl.load(
                        ch_sel_ptr + h * ch_sel_stride_h + d_offs
                    ).to(tl.int64),
                ).to(tl.float32) * tl.load(
                    ch_w_ptr + h * ch_w_stride_h + d_offs
                ).to(tl.float32)
                tile = tl.load(
                    sig_ptr
                    + h * sig_stride_h
                    + d_offs[:, None] * sig_stride_d
                    + safe_phys[None, :],
                    mask=in_range[None, :],
                    other=0.0,
                ).to(tl.float32)
                dot += tl.sum(qp[:, None] * tile, axis=0)
            score_h = dot
            acc = tl.where(score_h > acc, score_h, acc)

        out_score = tl.where(
            valid, acc, tl.full(acc.shape, float("-inf"), dtype=tl.float32)
        )
        tl.store(
            out_ptr + batch_id * out_stride_b + tok_offs, out_score, mask=in_range
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
    sig_t = sig.permute(1, 2, 0).contiguous()  # [H, D, T], same values
    written = torch.ones(TABLE_T, dtype=torch.bool, device=device)
    rpi = torch.arange(BS, dtype=torch.int32, device=device)
    out_prod = torch.empty(BS, WIDTH, dtype=torch.float32, device=device)
    out_tr = torch.empty(BS, WIDTH, dtype=torch.float32, device=device)
    flush = torch.zeros(FLUSH_MB * 1024 * 1024 // 4, dtype=torch.float32,
                        device=device)
    flush_grid = ((flush.numel() + 4095) // 4096,)

    def flush_call():
        _flush_l2_kernel[flush_grid](flush, n=flush.numel(), BLOCK=4096)

    nblocks512 = (WIDTH + 511) // 512

    def prod_call(rtt, seq):
        _logical_score_triton(
            q, ch_sel, ch_w, sig, written, rpi, rtt, seq, out_prod, WIDTH,
            scale_layer=None, token_block=512, workers=128,
            store_dead_neg_inf=False, scorer_norm="off", head_agg="max",
            hybrid_threshold=8192,
        )

    def tr_call(rtt, seq, tb, dt):
        nblocks = (WIDTH + tb - 1) // tb
        workers = min(128, nblocks)
        _transposed_logical_score_kernel[(BS, workers)](
            q, ch_sel, ch_w, sig_t, written, rpi, rtt, seq, out_tr,
            num_heads=H, max_seq_len=WIDTH, label_dim=LABEL_DIM,
            max_pool_len=WIDTH, max_tokens=TABLE_T,
            q_stride_b=q.stride(0), q_stride_h=q.stride(1),
            ch_sel_stride_h=ch_sel.stride(0), ch_w_stride_h=ch_w.stride(0),
            sig_stride_h=sig_t.stride(0), sig_stride_d=sig_t.stride(1),
            rtt_stride_p=rtt.stride(0), out_stride_b=out_tr.stride(0),
            TOKEN_BLOCK=tb, D_TILE=dt, LABEL_DIM_POW2=32, WORKERS=workers,
        )

    variants = [(128, 32), (256, 16), (512, 8), (64, 32)]

    # Bitwise + value-equality verification on mixed rows (random layout).
    rtt_v = torch.randint(0, TABLE_T, (BS, WIDTH), dtype=torch.int32, device=device)
    mixed = ([4096, 4608, 5120, 547, 2886] * 6)[:BS]
    seq_v = torch.tensor(mixed, dtype=torch.int32, device=device)
    report = {"bitwise_vs_production": {}, "max_abs_delta": {},
              "replay_us_per_call": {}}
    prod_call(rtt_v, seq_v)
    torch.cuda.synchronize()
    for tb, dt in variants:
        tr_call(rtt_v, seq_v, tb, dt)
        torch.cuda.synchronize()
        bit_ok, max_d = True, 0.0
        for b in range(BS):
            n = mixed[b]
            a_, b_ = out_prod[b, :n], out_tr[b, :n]
            if not torch.equal(a_, b_):
                bit_ok = False
                max_d = max(max_d, float((a_ - b_).abs().max()))
        report["bitwise_vs_production"][f"tb{tb}_dt{dt}"] = bool(bit_ok)
        report["max_abs_delta"][f"tb{tb}_dt{dt}"] = max_d
        print(f"[layout] bitwise tb={tb} dt={dt}: "
              f"{'OK' if bit_ok else f'DIFF (max {max_d:.3e})'}", flush=True)

    seq_op = torch.full((BS,), 4608, dtype=torch.int32, device=device)
    for layout in ("page64", "random"):
        if layout == "page64":
            pages = WIDTH // 64
            rtt = torch.empty(BS, WIDTH, dtype=torch.int32, device=device)
            for b in range(BS):
                st = torch.randint(0, TABLE_T // 64, (pages,), device=device) * 64
                rtt[b] = (st[:, None] + torch.arange(64, device=device)[None, :]
                          ).reshape(-1)[:WIDTH]
        else:
            rtt = rtt_v
        # WARM graphs
        pairs = [("production_warm", _graph_for(lambda: prod_call(rtt, seq_op)))]
        for tb, dt in variants:
            pairs.append((f"tr{tb}_{dt}_warm",
                          _graph_for(lambda tb=tb, dt=dt: tr_call(rtt, seq_op, tb, dt))))
        # COLD graphs: flush + kernel for production AND every transposed
        # variant, plus flush alone for net subtraction.
        pairs.append(("flush_only", _graph_for(flush_call)))
        pairs.append(("production_cold", _graph_for(
            lambda: (flush_call(), prod_call(rtt, seq_op)))))
        for tb, dt in variants:
            pairs.append((f"tr{tb}_{dt}_cold", _graph_for(
                lambda tb=tb, dt=dt: (flush_call(), tr_call(rtt, seq_op, tb, dt)))))
        med = _interleaved(pairs)
        flush_us = med["flush_only"]
        out_l = {k: round(v, 2) for k, v in med.items()}
        out_l["production_cold_net"] = round(med["production_cold"] - flush_us, 2)
        for tb, dt in variants:
            out_l[f"tr{tb}_{dt}_cold_net"] = round(
                med[f"tr{tb}_{dt}_cold"] - flush_us, 2)
        report["replay_us_per_call"][layout] = out_l
        print(f"[layout] {layout}: " + ", ".join(
            f"{k}={v}" for k, v in out_l.items()), flush=True)

    if args.out:
        path = f"{args.out}.json"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[layout] report -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
