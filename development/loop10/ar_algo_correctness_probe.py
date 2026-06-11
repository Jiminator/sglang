"""Loop-10 probe — is custom-AR v2 CORRECT (not merely reordered) per algorithm
at compact DS score-reduce sizes?

Motivation: the task6 bs-1 selcap gate showed a near-total selection reshuffle
(~90% of selected positions moved) on compact-width steps. A summation-ORDER
change explains only boundary-tie movement, so the suspect is algorithm
correctness at sizes the size-based selection never serves in production
(two-shot was pinned at 10 KiB; the default would have run one-shot there).

Method: every rank fills the buffer with (rank + 1), so the exact per-element
sum is 36 for 8 ranks IN ANY SUMMATION ORDER (all partial sums are exactly
representable in bf16). Any deviation is a correctness bug, not reordering.
A second pass with non-exact random values quantifies legitimate order noise
(two-shot vs NCCL max |delta|).

Run: torchrun --nproc-per-node 8 development/loop10/ar_algo_correctness_probe.py \
       --out development/loop10/runs/<dir>/ar_algo_probe.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist


def _rank0(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


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

    tp = get_tp_group()
    ca = tp.ca_comm
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    expected = float(world * (world + 1) // 2)  # 36 for 8 ranks

    report = {
        "world_size": world,
        "ca_type": type(ca).__name__ if ca is not None else None,
        "ca_disabled": getattr(ca, "disabled", None),
        "expected_exact_sum": expected,
        "cases": [],
    }

    shapes = [
        (1, 5120),
        (2, 5120),
        (4, 5120),
        (16, 5120),
        (32, 5120),
        (29, 4608),
        (32, 202756),  # full-width op-point reference (two-shot by size)
    ]
    algos = {
        "TWO_SHOT_PULL": AllReduceAlgo.TWO_SHOT_PULL,
        "ONE_SHOT_PUSH": AllReduceAlgo.ONE_SHOT_PUSH,
        "ONE_SHOT_PULL": AllReduceAlgo.ONE_SHOT_PULL,
        "size_based": None,
    }

    for bs, width in shapes:
        case = {"shape": [bs, width], "bytes": bs * width * 2, "algos": {}}
        base = torch.full(
            (bs, width), float(rank + 1), dtype=torch.bfloat16, device=device
        )
        eligible = bool(ca is not None and ca.should_custom_ar(base))
        case["should_custom_ar"] = eligible

        # Exactness check per algorithm: result must be exactly `expected`.
        for name, algo in algos.items():
            if not eligible and name != "nccl":
                case["algos"][name] = {"skipped": "not custom-AR eligible"}
                continue
            inp = base.clone()
            dist.barrier()
            try:
                out = ca.custom_all_reduce(inp, override_algo=algo)
                bad = (out != expected).sum().item()
                case["algos"][name] = {
                    "mismatched_elements": int(bad),
                    "total": out.numel(),
                    "min": float(out.min()),
                    "max": float(out.max()),
                }
            except Exception as exc:
                case["algos"][name] = {"error": f"{type(exc).__name__}: {exc}"}

        # NCCL reference on the raw group.
        inp = base.clone()
        dist.barrier()
        dist.all_reduce(inp, op=dist.ReduceOp.SUM, group=tp.device_group)
        bad = (inp != expected).sum().item()
        case["algos"]["nccl"] = {
            "mismatched_elements": int(bad),
            "total": inp.numel(),
        }

        # Order-noise quantification on non-exact values: two-shot vs NCCL.
        if eligible:
            torch.manual_seed(1234 + rank)
            rnd = torch.randn(bs, width, dtype=torch.float32, device=device).to(
                torch.bfloat16
            )
            ref = rnd.clone()
            dist.barrier()
            dist.all_reduce(ref, op=dist.ReduceOp.SUM, group=tp.device_group)
            two = ca.custom_all_reduce(
                rnd.clone(), override_algo=AllReduceAlgo.TWO_SHOT_PULL
            )
            delta = (two.float() - ref.float()).abs()
            case["order_noise_two_shot_vs_nccl"] = {
                "max_abs_delta": float(delta.max()),
                "frac_nonzero": float((delta > 0).float().mean()),
            }

        report["cases"].append(case)
        _rank0(f"[ar-probe] {case['shape']} bytes={case['bytes']} -> "
               + json.dumps(case["algos"]))

    if args.out and rank == 0:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[ar-probe] report -> {args.out}")
    dist.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
