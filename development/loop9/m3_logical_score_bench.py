"""Loop-9 M3 measure — per-kernel profile of `_logical_score_triton` at the
served GLM-5.1 Case-1 shapes (bs 29, width 202752, H_local 8, label_dim 32,
head_dim 192, fp16 signatures, fp16 scales=None, head_agg max, scorer off).

Frozen reference cost: 63,107 µs / 780 calls ≈ 81 µs/call.

Decomposition strategy: sweep seq_len at fixed width (the early-exit path
stores -inf and returns for token blocks past seq_len, so seq_len ≈ 0 prices
the dead-width store + launch floor, seq_len == width prices the all-live
compute), and sweep TOKEN_BLOCK for block-size sensitivity. One GPU (the
kernel is rank-local).

Run: python development/loop9/m3_logical_score_bench.py \
       --out development/loop9/runs/20260610_m0/m3_logical_score_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

WIDTH = 202752      # req_to_token width == context_len (served boot log)
BS = 29
H = 8               # num_local_heads at TP=8 (64 / 8)
LABEL_DIM = 32
HEAD_DIM = 192      # qk_nope_head_dim
TABLE_T = 142272    # token_label_table tokens/rank at mem 0.7
ITERS = 50
WARMUP = 10


def _time_us(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(ITERS):
        start.record()
        fn()
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

    device = torch.device("cuda")
    torch.manual_seed(0)
    sig = torch.randn(TABLE_T, H, LABEL_DIM, dtype=torch.float16, device=device)
    written = torch.ones(TABLE_T, dtype=torch.bool, device=device)
    ch_sel = torch.randint(
        0, HEAD_DIM, (H, LABEL_DIM), dtype=torch.int32, device=device
    )
    ch_w = torch.rand(H, LABEL_DIM, dtype=torch.float32, device=device)
    q = torch.randn(BS, H, HEAD_DIM, dtype=torch.bfloat16, device=device)
    rpi = torch.arange(BS, dtype=torch.int32, device=device)
    # Identity-ish mapping bounded by the table size.
    rtt = (
        torch.arange(WIDTH, dtype=torch.int32, device=device) % TABLE_T
    ).unsqueeze(0).expand(BS, -1).contiguous()
    out = torch.zeros(BS, WIDTH, dtype=torch.float32, device=device)

    def run(seq, token_block):
        seq_lens = torch.full((BS,), seq, dtype=torch.int32, device=device)
        return _time_us(
            lambda: _logical_score_triton(
                q_proj_input=q,
                channel_selection_layer=ch_sel,
                channel_weights_layer=ch_w,
                sig_layer=sig,
                written_layer=written,
                req_pool_indices=rpi,
                req_to_token=rtt,
                seq_lens=seq_lens,
                out=out,
                max_seq_len=WIDTH,
                scale_layer=None,
                token_block=token_block,
                scorer_norm="off",
                head_agg="max",
                hybrid_threshold=8192,
            )
        )

    report = {
        "shape": {
            "bs": BS, "width": WIDTH, "H": H, "label_dim": LABEL_DIM,
            "head_dim": HEAD_DIM, "table_T": TABLE_T, "sig_dtype": "fp16",
        },
        "frozen_reference_us_per_call": 81,
        "seq_sweep_tb64": {},
        "block_sweep_seq4608": {},
        "block_sweep_seq202752": {},
    }
    # seq sweep at the default TOKEN_BLOCK=64: decomposes dead-width floor vs
    # live compute. 4608 = the Case-1 bench op point (ISL 4096 + OSL 512).
    for seq in (64, 1024, 4608, 16384, 65536, WIDTH):
        us = run(seq, 64)
        report["seq_sweep_tb64"][str(seq)] = round(us, 1)
        print(f"[m3-bench] seq={seq:>7} tb=64: {us:.1f} us/call", flush=True)
    # block-size sensitivity at the op-point seq and at all-live.
    for tb in (32, 64, 128, 256, 512):
        us = run(4608, tb)
        report["block_sweep_seq4608"][str(tb)] = round(us, 1)
        print(f"[m3-bench] seq=4608 tb={tb}: {us:.1f} us/call", flush=True)
    for tb in (64, 128, 256):
        us = run(WIDTH, tb)
        report["block_sweep_seq202752"][str(tb)] = round(us, 1)
        print(f"[m3-bench] seq=ALL tb={tb}: {us:.1f} us/call", flush=True)

    # Dead-width store floor: a pure fill of the same dead region for scale.
    dead = out[:, 4608:]
    report["pure_dead_fill_us"] = round(
        _time_us(lambda: dead.fill_(float("-inf"))), 1
    )
    print(f"[m3-bench] pure fill_(-inf) of dead region: {report['pure_dead_fill_us']} us")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[m3-bench] report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
