"""Loop-10 AC-1.2 exact-floor harness — apples-to-apples, ONE process.

The round-4 probe compared a stand-alone gather kernel against the production
kernel's number from a DIFFERENT process — and measured "floor" 25.47 µs/call
vs the production kernel's own 23.58: cross-process clock states made the
claimed lower bound invalid (a bound slower than a real implementation is no
bound). This harness repairs that:

  - ONE process, interleaved captured-replay timing (median of alternating
    rounds so clock drift hits both sides equally);
  - (a) the LANDED production `_logical_score_triton` (tb=512, full scoring
    math) and (b) a STRIPPED variant of the SAME kernel structure (identical
    persistent-worker grid, identical tile shapes, identical signature/rtt
    loads kept live via a trivial sum, all scoring math deleted) on identical
    inputs at the op-point shapes;
  - both slot layouts (worst-case random scatter and the allocator's
    page-64-contiguous runs), each emitted as its own artifact when --out is
    a directory-style prefix.

Validity check: stripped must measure <= production (it does strictly less
work on the same memory). Decision rule (owner ladder, DEC-L10-2): if
stripped > the isolated budget (20,000/780 − measured ~6 µs in-context
interference = 19.64 µs/call), every same-gather exact implementation is
bounded above the budget by the fastest-known gather structure and the exact
rung is exhausted with a valid bound; if stripped <= 19.64, the gap between
stripped and production is reachable compute/overhead headroom and exact
work continues.

Run: python development/loop10/task11_exact_floor_harness.py \
       --out development/loop10/runs/20260611_task11/exact_floor
   (writes <out>_random.json and <out>_page64.json)
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import triton
import triton.language as tl

WIDTH, BS, H, LABEL_DIM, HEAD_DIM, TABLE_T = 5120, 29, 8, 32, 192, 142272
ROUNDS = 100          # interleaved timing rounds per pair
REPLAYS_PER_ROUND = 4
WARMUP_ROUNDS = 10
BUDGET_REAL_US = 20000.0 / 780.0
INTERFERENCE_US = 6.0


@triton.jit
def _stripped_logical_score_kernel(
    sig_ptr, written_ptr, rpi_ptr, rtt_ptr, sl_ptr, out_ptr,
    max_seq_len: tl.constexpr,
    max_pool_len: tl.constexpr,
    max_tokens: tl.constexpr,
    sig_stride_t: tl.constexpr,
    sig_stride_h: tl.constexpr,
    rtt_stride_p: tl.constexpr,
    out_stride_b: tl.constexpr,
    num_heads: tl.constexpr,
    LABEL_DIM_POW2: tl.constexpr,
    label_dim: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    WORKERS: tl.constexpr,
):
    # Mirrors _logical_score_kernel's structure exactly — persistent-worker
    # grid (bs, WORKERS), device-side live-block loop, identical rtt/written/
    # signature loads — with every scoring FLOP deleted. The trivial
    # per-position sum keeps the loads alive against dead-code elimination.
    batch_id = tl.program_id(0)
    worker = tl.program_id(1)
    seq_len_i = tl.load(sl_ptr + batch_id).to(tl.int32)
    n_live = tl.minimum(seq_len_i, max_seq_len)
    live_blocks = (n_live + TOKEN_BLOCK - 1) // TOKEN_BLOCK
    pool_idx = tl.load(rpi_ptr + batch_id).to(tl.int64)
    d_offs = tl.arange(0, LABEL_DIM_POW2)
    d_mask = d_offs < label_dim
    for tok_blk in range(worker, live_blocks, WORKERS):
        tok_offs = tok_blk * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
        in_range = tok_offs < max_seq_len
        safe_tok = tl.minimum(tok_offs, max_pool_len - 1)
        phys = tl.load(
            rtt_ptr + pool_idx * rtt_stride_p + safe_tok, mask=in_range, other=0
        ).to(tl.int64)
        safe_phys = tl.minimum(tl.maximum(phys, 0), max_tokens - 1)
        written = tl.load(written_ptr + safe_phys, mask=in_range, other=0).to(
            tl.int1
        )
        acc = tl.zeros((TOKEN_BLOCK,), dtype=tl.float32)
        for h in range(num_heads):
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
            acc += tl.sum(sig_block, axis=1)
        acc = tl.where(written, acc, 0.0)
        tl.store(out_ptr + batch_id * out_stride_b + tok_offs, acc, mask=in_range)


def _graph_for(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def _interleaved_medians(g_a, g_b):
    """Median replay µs for two graphs timed in ALTERNATING rounds, so clock
    state and thermal drift hit both equally."""
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    times = {"a": [], "b": []}
    for _ in range(WARMUP_ROUNDS):
        g_a.replay()
        g_b.replay()
    torch.cuda.synchronize()
    for _ in range(ROUNDS):
        for key, g in (("a", g_a), ("b", g_b)):
            s.record()
            for _ in range(REPLAYS_PER_ROUND):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            times[key].append(s.elapsed_time(e) * 1000.0 / REPLAYS_PER_ROUND)
    for key in times:
        times[key].sort()
    return (
        times["a"][len(times["a"]) // 2],
        times["b"][len(times["b"]) // 2],
    )


def _build_rtt(layout: str, device) -> torch.Tensor:
    if layout == "page64":
        pages_per_row = WIDTH // 64
        rtt = torch.empty(BS, WIDTH, dtype=torch.int32, device=device)
        for b in range(BS):
            starts = (
                torch.randint(0, TABLE_T // 64, (pages_per_row,), device=device)
                * 64
            )
            rtt[b] = (
                starts[:, None] + torch.arange(64, device=device)[None, :]
            ).reshape(-1)[:WIDTH]
        return rtt
    return torch.randint(0, TABLE_T, (BS, WIDTH), dtype=torch.int32, device=device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="artifact path prefix; writes <out>_<layout>.json")
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
    seq = torch.full((BS,), 4608, dtype=torch.int32, device=device)
    out_prod = torch.empty(BS, WIDTH, dtype=torch.float32, device=device)
    out_strip = torch.empty(BS, WIDTH, dtype=torch.float32, device=device)
    budget_iso = BUDGET_REAL_US - INTERFERENCE_US

    for layout in ("random", "page64"):
        rtt = _build_rtt(layout, device)
        nblocks = (WIDTH + 511) // 512
        workers = min(128, nblocks)

        def production():
            _logical_score_triton(
                q, ch_sel, ch_w, sig, written, rpi, rtt, seq, out_prod, WIDTH,
                scale_layer=None, token_block=512, workers=128,
                store_dead_neg_inf=False, scorer_norm="off", head_agg="max",
                hybrid_threshold=8192,
            )

        def stripped():
            _stripped_logical_score_kernel[(BS, workers)](
                sig, written, rpi, rtt, seq, out_strip,
                max_seq_len=WIDTH, max_pool_len=WIDTH, max_tokens=TABLE_T,
                sig_stride_t=sig.stride(0), sig_stride_h=sig.stride(1),
                rtt_stride_p=rtt.stride(0), out_stride_b=out_strip.stride(0),
                num_heads=H, LABEL_DIM_POW2=32, label_dim=LABEL_DIM,
                TOKEN_BLOCK=512, WORKERS=workers,
            )

        g_prod = _graph_for(production)
        g_strip = _graph_for(stripped)
        prod_us, strip_us = _interleaved_medians(g_prod, g_strip)

        report = {
            "layout": layout,
            "shapes": {"bs": BS, "width": WIDTH, "H": H, "label_dim": LABEL_DIM,
                       "table_t": TABLE_T, "live_seq": 4608},
            "method": "one process; interleaved captured-replay rounds "
                      f"({ROUNDS}x{REPLAYS_PER_ROUND} per side); medians",
            "production_tb512_us_per_call": round(prod_us, 2),
            "stripped_same_structure_us_per_call": round(strip_us, 2),
            "budget_us_per_call_real": round(BUDGET_REAL_US, 2),
            "in_context_interference_us": INTERFERENCE_US,
            "budget_us_per_call_isolated": round(budget_iso, 2),
            "bound_valid_stripped_le_production": bool(strip_us <= prod_us + 0.3),
            "exact_rung_exhausted_at_this_layout": bool(strip_us > budget_iso),
            "compute_overhead_headroom_us": round(prod_us - strip_us, 2),
        }
        print(f"[exact-floor] {layout}: production {prod_us:.2f} vs stripped "
              f"{strip_us:.2f} us/call (budget {budget_iso:.2f}; "
              f"bound_valid={report['bound_valid_stripped_le_production']}, "
              f"exhausted={report['exact_rung_exhausted_at_this_layout']})",
              flush=True)
        if args.out:
            path = f"{args.out}_{layout}.json"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as fh:
                json.dump(report, fh, indent=2)
            print(f"[exact-floor] report -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
