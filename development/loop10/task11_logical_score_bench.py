"""Loop-10 task11 measure — CAPTURED-REPLAY cost of `_logical_score_triton`
at the compact W=5120 op-point shapes, swept over TOKEN_BLOCK / WORKERS.

Eager CUDA-event timing of this kernel measures host JIT dispatch (~40 µs
floor — BL-20260531 addendum), so every variant is captured into its own CUDA
graph and the REPLAY is timed: that is the number the frozen Case-1 profile
pays. Target: ≤20,000 µs / 780 calls = 25.6 µs/call (current landed bucket
23,080 µs ≈ 29.6 µs/call at tb=256).

The sweep also bitwise-verifies every variant against the tb=256 production
default on identical random inputs (mixed live rows incl. 4096/4608/5120) —
per-position math is block-partition-independent, so any inequality is an
implementation bug, not noise.

Run: python development/loop10/task11_logical_score_bench.py \
       --out development/loop10/runs/<dir>/task11_bench.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch

WIDTH = 5120        # compact selector width (the landed op-point variant)
BS = 29
H = 8               # num_local_heads at TP=8
LABEL_DIM = 32
HEAD_DIM = 192
TABLE_T = 142272    # token_label_table tokens/rank at mem 0.7
ITERS = 200
WARMUP = 20


def _time_replay_us(graph) -> float:
    for _ in range(WARMUP):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(ITERS):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
        _logical_score_triton,
    )

    device = torch.device("cuda:0")
    torch.manual_seed(20260611)

    q = torch.randn(BS, H, HEAD_DIM, dtype=torch.float32, device=device)
    ch_sel = torch.randint(
        0, HEAD_DIM, (H, LABEL_DIM), dtype=torch.int32, device=device
    )
    ch_w = torch.randn(H, LABEL_DIM, dtype=torch.float32, device=device)
    sig = torch.randn(TABLE_T, H, LABEL_DIM, dtype=torch.float16, device=device)
    written = torch.ones(TABLE_T, dtype=torch.bool, device=device)
    rpi = torch.arange(BS, dtype=torch.int32, device=device)
    rtt = torch.randint(
        0, TABLE_T, (BS, WIDTH), dtype=torch.int32, device=device
    )
    # Mixed live rows incl. the boundary shapes; op-point realism for timing.
    mixed = [4096, 4608, 5120, 547, 2886] * 6
    seq_mixed = torch.tensor(mixed[:BS], dtype=torch.int32, device=device)
    seq_op = torch.full((BS,), 4608, dtype=torch.int32, device=device)

    def call(out, seq_lens, tb, workers):
        _logical_score_triton(
            q, ch_sel, ch_w, sig, written, rpi, rtt, seq_lens, out,
            WIDTH, scale_layer=None, token_block=tb, workers=workers,
            store_dead_neg_inf=True, scorer_norm="off", head_agg="max",
            hybrid_threshold=8192,
        )

    variants = [
        (256, 128),   # production default (workers caps at nblocks=20)
        (256, 10),
        (512, 128),   # caps at nblocks=10
        (512, 5),
        (1024, 128),  # caps at nblocks=5
    ]

    # Bitwise verification on mixed rows vs the production default.
    ref = torch.empty(BS, WIDTH, dtype=torch.float32, device=device)
    call(ref, seq_mixed, 256, 128)
    torch.cuda.synchronize()
    report = {"shapes": {"bs": BS, "width": WIDTH, "H": H,
                         "label_dim": LABEL_DIM, "table_t": TABLE_T},
              "bitwise_vs_tb256": {}, "replay_us_per_call": {}}
    for tb, wk in variants:
        out = torch.empty_like(ref)
        call(out, seq_mixed, tb, wk)
        torch.cuda.synchronize()
        same = bool(torch.equal(out, ref))
        report["bitwise_vs_tb256"][f"tb{tb}_w{wk}"] = same
        print(f"[task11-bench] bitwise tb={tb} w={wk}: {'OK' if same else 'DIFF'}",
              flush=True)

    # Captured-replay timing at the op-point seq (4608) per variant.
    for tb, wk in variants:
        out = torch.empty_like(ref)
        # warm the JIT before capture
        call(out, seq_op, tb, wk)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call(out, seq_op, tb, wk)
        us = _time_replay_us(g)
        report["replay_us_per_call"][f"tb{tb}_w{wk}"] = round(us, 2)
        window = us * 780.0
        print(f"[task11-bench] replay seq=4608 tb={tb} w={wk}: {us:.2f} us/call "
              f"(~{window/1000:.1f}k us/window)", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[task11-bench] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
