"""task5 GPU paged absorbed-latent score kernel — CAPTURED-REPLAY budget.

Mirrors development/loop10/task11_logical_score_bench.py (BS=29, H=8, WIDTH=5120,
seq=4608, captured-replay median x 780 calls/window) so the absorbed kernel's
per-window cost is directly comparable to the landed logical-score bucket
(~23,080 us/window, target <=20,000 us/window = 25.6 us/call).

R14: the kernel uses block-scale reassociation with tl.dot tensor-core tiles
(per 128-channel block: partial = tl.dot(fp8_latent_blk, v_blk) then *block_scale,
fp32-accumulate per head, head-max) — no [TOKEN_BLOCK,512] dequant tile. The harness
calls the PUBLIC wrapper absorbed_score_paged_fp8() (not the private Triton symbol),
so the measured number is the default callable path. Eager timing measures host JIT
dispatch, so each token_block is captured into its own CUDA graph and the REPLAY is
timed.

Run: python development/loop11/runs/20260614_m2/absorbed_kernel_budget.py \
       --out development/loop11/runs/20260614_m2/absorbed_kernel_budget.json
"""

from __future__ import annotations

import argparse
import json

import torch

WIDTH = 5120
BS = 29
H = 8
LORA = 512
BLOCK = 128
SEQ = 4608
ITERS = 200
WARMUP = 20


def _time_replay_us(graph) -> float:
    for _ in range(WARMUP):
        graph.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(ITERS):
        s.record()
        graph.replay()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from sglang.srt.layers.attention.double_sparsity.absorbed_latent_kernel import (
        absorbed_score_paged_fp8,
    )

    dev = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0)
    v = torch.randn(BS, H, LORA, generator=g).to(dev)
    fp8 = torch.randn(WIDTH, LORA, generator=g).to(dev).to(torch.float8_e4m3fn)
    scales = (torch.rand(WIDTH, LORA // BLOCK, generator=g) + 0.5).to(dev)
    written = torch.ones(WIDTH, dtype=torch.bool, device=dev)
    rpi = torch.arange(BS, dtype=torch.int32, device=dev)
    rtt = (
        torch.arange(WIDTH, dtype=torch.int32, device=dev)
        .unsqueeze(0)
        .expand(BS, -1)
        .contiguous()
    )
    seq_lens = torch.full((BS,), SEQ, dtype=torch.int32, device=dev)

    def call(tb):
        return absorbed_score_paged_fp8(
            v, fp8, scales, rpi, rtt, seq_lens, WIDTH, written=written, token_block=tb
        )

    results = []
    for tb in [16, 32, 64, 128]:
        for _ in range(3):  # warmup compile outside capture
            call(tb)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            call(tb)
        us = _time_replay_us(gr)
        window = us * 780.0
        results.append({"token_block": tb, "us_per_call": us, "us_per_window": window})
        print(
            f"[absorbed-budget] seq={SEQ} tb={tb}: {us:.2f} us/call "
            f"(~{window/1000:.1f}k us/window)",
            flush=True,
        )

    best = min(results, key=lambda r: r["us_per_call"])
    summary = {
        "op_point": {
            "bs": BS,
            "H": H,
            "width": WIDTH,
            "lora": LORA,
            "seq": SEQ,
            "block": BLOCK,
        },
        "kernel": "tl.dot block-scale reassociation (tf32), public wrapper path",
        "logical_score_landed_bucket_us_window": 23080.0,
        "budget_target_us_window": 20000.0,
        "results": results,
        "best": best,
        "verdict": "PASS" if best["us_per_window"] <= 20000.0 else "OVER_BUDGET",
    }
    print(
        f"[absorbed-budget] BEST tb={best['token_block']}: "
        f"{best['us_per_call']:.2f} us/call ~{best['us_per_window']/1000:.1f}k us/window "
        f"-> {summary['verdict']} (vs logical-score ~23.1k landed / 20k target)",
        flush=True,
    )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[absorbed-budget] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
