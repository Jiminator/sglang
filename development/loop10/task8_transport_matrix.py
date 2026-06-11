"""Loop-10 task8 measure — SAME-SHAPE transport matrix for the DS compact
score reduce at the real op-point shape [32, 5120] bf16 (320 KiB), 8 ranks.

Three non-crashing candidates under identical conditions:
  - pinned TWO_SHOT_PULL  (the landed incumbent, per-call override)
  - forced ONE_SHOT_PULL  (the only viable declared one-shot; PUSH hard-errors
    above ~160 KiB — cited from ar_algo_probe.json, not re-crashed here)
  - NCCL bf16             (torch.distributed all_reduce on the raw group)

Each candidate is timed BOTH ways and both numbers are recorded:
  - captured-replay (the binding mode for graph-served kernels): the call is
    captured into a CUDA graph and the replay is timed
  - eager CUDA-event median (lower-confidence; recorded for comparability
    with the loop-9 spike bench)

Per-candidate fields: dtype, bytes, weak_contiguous, custom-AR eligibility,
actual algorithm, execution mode, us/call, and the projected us/window at
780 calls.

Run: torchrun --nproc-per-node 8 development/loop10/task8_transport_matrix.py \
       --out development/loop10/runs/<dir>/task8_matrix.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

BS, WIDTH = 32, 5120
ITERS = 200
WARMUP = 20
CALLS_PER_WINDOW = 780


def _median_us(fn) -> float:
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
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    from sglang.srt.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=world,
        rank=local_rank,
        distributed_init_method="env://",
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world)

    from sglang.jit_kernel.all_reduce import AllReduceAlgo
    from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
        is_weak_contiguous,
    )

    tp = get_tp_group()
    ca = tp.ca_comm
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(20260611 + rank)

    buf = torch.randn(BS, WIDTH, dtype=torch.float32, device=device).to(
        torch.bfloat16
    )
    nbytes = buf.numel() * buf.element_size()
    report = {
        "shape": [BS, WIDTH],
        "dtype": "torch.bfloat16",
        "bytes": nbytes,
        "weak_contiguous": bool(is_weak_contiguous(buf)),
        "custom_ar_eligible": bool(ca is not None and ca.should_custom_ar(buf)),
        "world_size": world,
        "calls_per_window": CALLS_PER_WINDOW,
        "one_shot_push": "not viable at this size — hard-errors above ~160 KiB "
        "(cited: runs/20260611_task6_gates/ar_algo_probe.json)",
        "candidates": {},
    }

    candidates = [
        ("pinned TWO_SHOT_PULL (incumbent)", "custom_ar",
         AllReduceAlgo.TWO_SHOT_PULL),
        ("forced ONE_SHOT_PULL (declared one-shot)", "custom_ar",
         AllReduceAlgo.ONE_SHOT_PULL),
        ("NCCL bf16", "nccl", None),
    ]

    for name, kind, algo in candidates:
        if kind == "custom_ar":
            def call(inp=buf, a=algo):
                ca.custom_all_reduce(inp, override_algo=a)
        else:
            def call(inp=buf):
                dist.all_reduce(inp, op=dist.ReduceOp.SUM, group=tp.device_group)

        entry = {"algorithm": name}
        eager_us = _median_us(call)
        entry["eager_us_per_call"] = round(eager_us, 2)

        # Captured-replay (the binding mode). NCCL and custom-AR are both
        # graph-capturable in this build; record an honest error otherwise.
        try:
            if kind == "custom_ar":
                ctx = ca.capture()
            else:
                import contextlib

                ctx = contextlib.nullcontext()
            with ctx:
                call()
                torch.cuda.synchronize()
                dist.barrier()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    call()

            def replay():
                g.replay()

            replay_us = _median_us(replay)
            entry["replay_us_per_call"] = round(replay_us, 2)
            entry["replay_us_per_window"] = round(replay_us * CALLS_PER_WINDOW)
            entry["mode"] = "captured_replay+eager"
        except Exception as exc:
            entry["replay_error"] = f"{type(exc).__name__}: {exc}"
            entry["mode"] = "eager_only"
        report["candidates"][name] = entry
        if rank == 0:
            print(f"[task8-matrix] {name}: "
                  f"eager {entry['eager_us_per_call']} us/call, "
                  f"replay {entry.get('replay_us_per_call', 'n/a')} us/call",
                  flush=True)

    if args.out and rank == 0:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[task8-matrix] report -> {args.out}")
    dist.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
