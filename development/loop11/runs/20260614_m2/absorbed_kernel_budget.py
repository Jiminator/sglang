"""task5 GPU paged absorbed-latent score kernel — CAPTURED-REPLAY budget.

Mirrors development/loop10/task11_logical_score_bench.py exactly (BS=29, H=8,
WIDTH=5120, seq=4608, captured-replay median x 780 calls/window) so the absorbed
kernel's per-window cost is directly comparable to the landed logical-score
bucket (~23,080 us/window, target <=20,000 us/window = 25.6 us/call).

The absorbed kernel reads the full 512-dim fp8 latent (vs the label kernel's
label_dim=32 signature), so per token it does num_heads x 512 MACs vs num_heads x
32 — a known ~16x compute risk the plan flagged as MEASURED, not assumed. One
latent read serves all heads. Eager timing measures host JIT dispatch, so every
variant is captured into its own CUDA graph and the REPLAY is timed.

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
        _absorbed_score_kernel,
        _next_pow2,
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
    out = torch.empty((BS, WIDTH), dtype=torch.float32, device=dev)

    def launch(tb, workers):
        tbp = _next_pow2(min(tb, WIDTH))
        lora_pow2 = _next_pow2(LORA)
        ntb = (WIDTH + tbp - 1) // tbp
        nw = max(1, min(workers, ntb))
        grid = (BS, nw)
        _absorbed_score_kernel[grid](
            v,
            fp8,
            scales,
            written,
            rpi,
            rtt,
            seq_lens,
            out,
            num_heads=H,
            max_seq_len=WIDTH,
            lora=LORA,
            block_size=BLOCK,
            max_pool_len=WIDTH,
            max_tokens=WIDTH,
            v_stride_b=v.stride(0),
            v_stride_h=v.stride(1),
            fp8_stride_t=fp8.stride(0),
            scale_stride_t=scales.stride(0),
            rtt_stride_p=rtt.stride(0),
            out_stride_b=out.stride(0),
            TOKEN_BLOCK=tbp,
            LORA_POW2=lora_pow2,
            HEAD_AGG_MEAN=False,
            STORE_DEAD_NEG_INF=True,
            WORKERS=nw,
        )

    results = []
    for tb in [16, 32, 64, 128, 256]:
        for wk in [128]:
            # warmup compile outside capture
            for _ in range(3):
                launch(tb, wk)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                launch(tb, wk)
            us = _time_replay_us(gr)
            window = us * 780.0
            results.append(
                {
                    "token_block": tb,
                    "workers": wk,
                    "us_per_call": us,
                    "us_per_window": window,
                }
            )
            print(
                f"[absorbed-budget] seq={SEQ} tb={tb} w={wk}: {us:.2f} us/call "
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
