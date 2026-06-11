"""Loop-10 AC-1.2 exhaustion probe — the irreducible signature-gather floor.

Any exact logical-score implementation must read, per call, every live
position's per-head signature block through the req_to_token indirection
(29 rows x 4,608 positions x 8 heads x 32 labels). This stripped kernel does
ONLY that gather plus a trivial per-position reduction and store — an upper
bound on achievable bandwidth, hence a LOWER bound on any exact kernel's
captured-replay time. If the floor exceeds the AC-1.2 budget
(20,000 us / 780 calls = 25.6 us/call real, ~19.6 us/call isolated after the
measured ~6 us/call in-context interference), the exact regime is exhausted
by measurement.

The int8 variant (signatures int8 + per-(slot, head) fp16 scales) projects
the owner-authorized fallback's floor under the same access pattern.

Run: python development/loop10/task11_roofline_probe.py \
       --out development/loop10/runs/<dir>/roofline_probe.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import triton
import triton.language as tl

WIDTH, BS, H, LABEL_DIM, TABLE_T = 5120, 29, 8, 32, 142272
ITERS = 200
WARMUP = 20


@triton.jit
def _gather_floor_kernel(
    sig_ptr, rtt_ptr, sl_ptr, out_ptr,
    sig_stride_t: tl.constexpr,
    sig_stride_h: tl.constexpr,
    rtt_stride_b: tl.constexpr,
    out_stride_b: tl.constexpr,
    num_heads: tl.constexpr,
    label_dim: tl.constexpr,
    width: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    IS_INT8: tl.constexpr,
    scale_ptr=None,
    scale_stride_t: tl.constexpr = 0,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    seq_len = tl.load(sl_ptr + row).to(tl.int32)
    start = blk * TOKEN_BLOCK
    if start >= seq_len:
        return
    offs = start + tl.arange(0, TOKEN_BLOCK)
    in_win = offs < seq_len
    phys = tl.load(rtt_ptr + row * rtt_stride_b + offs, mask=in_win, other=0).to(tl.int64)
    d_offs = tl.arange(0, label_dim)
    acc = tl.zeros((TOKEN_BLOCK,), dtype=tl.float32)
    for h in range(num_heads):
        sig = tl.load(
            sig_ptr + phys[:, None] * sig_stride_t + h * sig_stride_h + d_offs[None, :],
            mask=in_win[:, None],
            other=0.0,
        ).to(tl.float32)
        part = tl.sum(sig, axis=1)
        if IS_INT8:
            sc = tl.load(
                scale_ptr + phys * scale_stride_t + h, mask=in_win, other=0.0
            ).to(tl.float32)
            part = part * sc
        acc += part
    tl.store(out_ptr + row * out_stride_b + offs, acc, mask=in_win)


def _replay_us(fn) -> float:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(WARMUP):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    times = []
    for _ in range(ITERS):
        s.record(); g.replay(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--layout",
        choices=("random", "page64"),
        default="random",
        help="slot layout: worst-case random scatter, or page-64-contiguous "
        "runs at random page starts (the real allocator pattern; produced "
        "the artifact's page_contiguous_floors block)",
    )
    args = ap.parse_args()
    device = torch.device("cuda:0")
    torch.manual_seed(20260611)

    if args.layout == "page64":
        pages_per_row = WIDTH // 64
        rtt = torch.empty(BS, WIDTH, dtype=torch.int32, device=device)
        for b in range(BS):
            starts = (
                torch.randint(0, TABLE_T // 64, (pages_per_row,), device=device) * 64
            )
            rtt[b] = (
                starts[:, None] + torch.arange(64, device=device)[None, :]
            ).reshape(-1)[:WIDTH]
    else:
        rtt = torch.randint(0, TABLE_T, (BS, WIDTH), dtype=torch.int32, device=device)
    seq = torch.full((BS,), 4608, dtype=torch.int32, device=device)
    out = torch.empty(BS, WIDTH, dtype=torch.float32, device=device)
    budget_us = 20000.0 / 780.0

    report = {
        "shapes": {"bs": BS, "width": WIDTH, "H": H, "label_dim": LABEL_DIM,
                   "table_t": TABLE_T, "live_seq": 4608},
        "budget_us_per_call_real": round(budget_us, 2),
        "measured_in_context_overhead_us": 6.0,
        "budget_us_per_call_isolated": round(budget_us - 6.0, 2),
        "floors": {},
    }

    for name, dtype in (("fp16", torch.float16), ("int8", torch.int8)):
        if dtype is torch.int8:
            sig = torch.randint(-127, 127, (TABLE_T, H, LABEL_DIM),
                                dtype=torch.int8, device=device)
            scale = torch.rand(TABLE_T, H, dtype=torch.float16, device=device)
        else:
            sig = torch.randn(TABLE_T, H, LABEL_DIM, dtype=dtype, device=device)
            scale = torch.zeros(1, 1, dtype=torch.float16, device=device)
        nblocks = (WIDTH + 511) // 512

        def call(sig=sig, scale=scale, is_int8=(dtype is torch.int8)):
            _gather_floor_kernel[(BS, nblocks)](
                sig, rtt, seq, out,
                sig_stride_t=sig.stride(0), sig_stride_h=sig.stride(1),
                rtt_stride_b=rtt.stride(0), out_stride_b=out.stride(0),
                num_heads=H, label_dim=LABEL_DIM, width=WIDTH,
                TOKEN_BLOCK=512, IS_INT8=is_int8,
                scale_ptr=scale, scale_stride_t=scale.stride(0),
            )

        us = _replay_us(call)
        gb = BS * 4608 * H * LABEL_DIM * sig.element_size() / 1e9
        report["floors"][name] = {
            "replay_us_per_call": round(us, 2),
            "gather_gb_per_call": round(gb, 4),
            "effective_tbps": round(gb / (us / 1e6) / 1000.0, 2),
            "window_us_at_780": round(us * 780),
        }
        print(f"[roofline] {name}: floor {us:.2f} us/call "
              f"({report['floors'][name]['effective_tbps']} TB/s effective; "
              f"isolated budget {report['budget_us_per_call_isolated']:.2f})",
              flush=True)

    f16 = report["floors"]["fp16"]["replay_us_per_call"]
    report["exact_regime_exhausted"] = bool(
        f16 > report["budget_us_per_call_isolated"]
    )
    print(f"[roofline] exact regime exhausted: {report['exact_regime_exhausted']}",
          flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[roofline] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
