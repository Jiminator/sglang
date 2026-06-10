"""Loop-9 M1 spike — measure the DS score-reduce options on 8xH200.

The production DS score reduce is `torch.distributed.all_reduce` (NCCL ring) on
`scratch_scores[:bs, :max_seq_len]` where max_seq_len == req_to_token width ==
context_len (202752 for served GLM-5.1) — i.e. ~23.5 MB fp32 at bs 29, NOT the
~534 KB live-width tensor the plan draft assumed. This bench produces the GO /
NO-GO evidence:

  1. group facts: is the attention-TP group the TP GroupCoordinator (custom-AR
     capable); which custom-AR (v1/v2) is active; its size caps/thresholds.
  2. eligibility + measured cost per (bs, width, dtype): NCCL ring on the raw
     device_group vs GroupCoordinator.all_reduce (dispatched), with the
     selected backend recorded.
  3. a v2 instance with a raised pull cap (32 MB) to price custom-AR at the
     real 23.5 MB width.
  4. a CUDA-graph micro-capture of the coordinator reduce at an eligible size:
     replay correctness + zero replay allocations + the named kernel from a
     torch-profiler pass over replay.

Run:  torchrun --nproc-per-node 8 development/loop9/m1_spike_allreduce_bench.py \
        --out development/loop9/runs/20260610_m0/m1_spike.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

CONTEXT_LEN = 202752
ITERS = 50
WARMUP = 10


def _rank0(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def _time_op(fn) -> float:
    """Median wall time of fn() in microseconds via CUDA events (all-rank barrier
    before each timing region so collectives start together)."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(ITERS):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # ms -> us
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-capture", action="store_true")
    args = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    from sglang.srt.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.distributed.parallel_state import get_attn_tp_group, graph_capture

    init_distributed_environment(
        world_size=world,
        rank=local_rank,
        distributed_init_method="env://",
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world)

    tp = get_tp_group()
    attn_tp = get_attn_tp_group()
    ca = tp.ca_comm
    report = {
        "world_size": world,
        "attn_tp_is_tp": attn_tp is tp,
        "ca_comm_type": type(ca).__name__ if ca is not None else None,
        "ca_disabled": getattr(ca, "disabled", None),
        "ca_max_size": getattr(ca, "max_size", None),
        "ca_max_pull_size": getattr(ca, "max_pull_size", None),
        "ca_max_push_size": getattr(ca, "max_push_size", None),
        "ca_config": str(getattr(ca, "config", None)),
        "cases": [],
    }
    _rank0(
        f"[m1-spike] attn_tp is tp: {report['attn_tp_is_tp']}; "
        f"ca: {report['ca_comm_type']} disabled={report['ca_disabled']} "
        f"max_size={report['ca_max_size']}"
    )

    # A second v2 instance with the pull cap raised to cover the real 23.5MB
    # width — prices custom-AR at the production size.
    ca_wide = None
    try:
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        ca_wide = CustomAllReduceV2(
            group=tp.device_group,
            device=torch.device(f"cuda:{local_rank}"),
            max_pull_size=32 * 1024 * 1024,
        )
        report["ca_wide_disabled"] = ca_wide.disabled
        if ca_wide.disabled:
            ca_wide = None
    except Exception as e:
        report["ca_wide_error"] = f"{type(e).__name__}: {e}"
        ca_wide = None

    cases = [
        # (bs, width, dtype) — width 202752 is the served req_to_token width;
        # 4608 is the live SLO-workload width the draft assumed; 16384/65536
        # chart the curve. bs 1/29/64 are the graph-bucket extremes.
        (29, CONTEXT_LEN, torch.float32),
        (29, CONTEXT_LEN, torch.bfloat16),
        (64, CONTEXT_LEN, torch.float32),
        (1, CONTEXT_LEN, torch.float32),
        (29, 65536, torch.float32),
        (29, 16384, torch.float32),
        (29, 4608, torch.float32),
        (29, 4608, torch.bfloat16),
    ]
    for bs, width, dtype in cases:
        t = torch.randn(bs, width, dtype=dtype, device="cuda")
        nbytes = t.numel() * t.element_size()
        entry = {
            "bs": bs,
            "width": width,
            "dtype": str(dtype).replace("torch.", ""),
            "mbytes": round(nbytes / 1e6, 2),
        }
        entry["should_custom_ar"] = bool(ca.should_custom_ar(t)) if ca else False
        # NCCL ring on the raw process group (today's DS path).
        ref = t.clone()
        entry["nccl_ring_us"] = round(
            _time_op(lambda: dist.all_reduce(ref, group=tp.device_group)), 1
        )
        # Coordinator dispatch (custom-AR when eligible, else its ladder).
        src = t.clone()
        entry["coordinator_us"] = round(_time_op(lambda: tp.all_reduce(src)), 1)
        # Wide-cap v2 at the real width.
        if ca_wide is not None and ca_wide.should_custom_ar(t):
            src2 = t.clone()
            entry["ca_wide_us"] = round(
                _time_op(lambda: ca_wide.custom_all_reduce(src2)), 1
            )
            entry["ca_wide_eligible"] = True
        else:
            entry["ca_wide_eligible"] = bool(
                ca_wide is not None and ca_wide.should_custom_ar(t)
            )
        report["cases"].append(entry)
        _rank0(f"[m1-spike] {entry}")

    # Correctness spot-check: coordinator reduce == NCCL ring reduce (bitwise
    # equality is NOT expected — summation order differs; check tolerance).
    t = torch.randn(29, 4608, dtype=torch.float32, device="cuda")
    a = t.clone()
    b = t.clone()
    dist.all_reduce(a, group=tp.device_group)
    b_out = tp.all_reduce(b)
    max_rel = float(
        ((a - b_out).abs() / a.abs().clamp_min(1e-6)).max()
    )
    report["coordinator_vs_ring_max_rel_err"] = max_rel
    _rank0(f"[m1-spike] coordinator vs ring max rel err: {max_rel:.3e}")

    # CUDA-graph micro-capture of the coordinator reduce at an eligible size,
    # inside the coordinator's graph-capture context (custom-AR registration).
    if not args.skip_capture:
        try:
            n = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
            static_in = torch.randn(29, 4608, dtype=torch.float32, device="cuda")
            static_out = None
            with graph_capture() as gc_ctx:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=gc_ctx.stream):
                    static_out = tp.all_reduce(static_in)
            # Replay with mutated input; eager reference on a copy.
            static_in.copy_(torch.full_like(static_in, 2.0))
            before = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
            g.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
            expect = 2.0 * world
            ok = bool(torch.allclose(static_out, torch.full_like(static_out, expect)))
            report["graph_capture"] = {
                "captured": True,
                "replay_correct": ok,
                "replay_new_allocations": int(after - before),
            }
            # Named kernel: profile one more replay.
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                g.replay()
                torch.cuda.synchronize()
            kernels = sorted(
                {
                    e.key
                    for e in prof.key_averages()
                    if getattr(e, "self_device_time_total", 0) > 0
                }
            )
            report["graph_capture"]["replay_kernels"] = kernels[:12]
            _rank0(f"[m1-spike] graph capture: {report['graph_capture']}")
        except Exception as e:
            report["graph_capture"] = {
                "captured": False,
                "error": f"{type(e).__name__}: {e}",
            }
            _rank0(f"[m1-spike] graph capture FAILED: {e}")

    if dist.get_rank() == 0 and args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[m1-spike] report -> {args.out}")
    dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
