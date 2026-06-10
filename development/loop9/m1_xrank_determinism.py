"""Loop-9 M1 pre-landing check — cross-rank determinism of the bf16 score reduce.

Verifies, at the real DS shape ([29, 202752] with -inf masked positions and
near-tie plateaus finer than bf16 resolution), that every rank receives
bit-identical reduced bytes and derives bit-identical selections from them —
eager AND under CUDA-graph replay of the captured reduce (zero replay
allocations). Also reports the bf16-vs-fp32 selection delta as a lossiness
preview (diagnostic; the binding gate is the served recall run).

Run: torchrun --nproc-per-node 8 development/loop9/m1_xrank_determinism.py
Exit 0 only if every cross-rank check passes.
"""

from __future__ import annotations

import hashlib
import os
import sys

import torch
import torch.distributed as dist

BS = 29
WIDTH = 202752
SEQ = 4608
TOP_K = 2048


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()


def _all_ranks_equal(tag: str, t: torch.Tensor, group) -> bool:
    h = torch.frombuffer(
        bytes.fromhex(_sha(t)), dtype=torch.uint8
    ).clone().cuda()
    gathered = [torch.zeros_like(h) for _ in range(dist.get_world_size(group=group))]
    dist.all_gather(gathered, h, group=group)
    ok = all(torch.equal(g, gathered[0]) for g in gathered)
    if dist.get_rank() == 0:
        print(f"[m1-xrank] {tag}: cross-rank bit-identical = {ok}", flush=True)
    return ok


def main() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    from sglang.srt.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.distributed.parallel_state import graph_capture
    from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
        reduce_token_scores,
        select_topk_sequence_order,
    )

    init_distributed_environment(
        world_size=world,
        rank=local_rank,
        distributed_init_method="env://",
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world)
    tp = get_tp_group()
    ca = tp.ca_comm
    assert ca is not None and not getattr(ca, "disabled", True), "custom-AR inactive"

    # Per-rank PARTIAL scores: rank-dependent values (head-sharded semantics),
    # identical -inf mask, and a near-tie plateau whose per-rank deltas are
    # below bf16 resolution so the reduced plateau TIES exactly in bf16.
    torch.manual_seed(1234 + local_rank)
    partial = torch.randn(BS, WIDTH, dtype=torch.float32, device="cuda")
    plateau = slice(2000, 2100)  # straddles the top_k boundary after masking
    partial[:, plateau] = 5.0 + local_rank * 1e-7  # < bf16 ulp at 5.0 (~0.03)
    mask_invalid = torch.zeros(BS, WIDTH, dtype=torch.bool, device="cuda")
    mask_invalid[:, SEQ:] = True
    mask_invalid[:, 100:140] = True  # unwritten slots inside the live window
    partial.masked_fill_(mask_invalid, float("-inf"))

    failures = 0
    bf16_scratch = torch.zeros(BS, WIDTH, dtype=torch.bfloat16, device="cuda")

    # --- eager bf16 reduce ---
    scores_bf16 = partial.clone()
    reduce_token_scores(
        scores_bf16,
        process_group=tp.device_group,
        reduce_ca=ca,
        bf16_scratch=bf16_scratch,
        use_bf16=True,
    )
    torch.cuda.synchronize()
    failures += not _all_ranks_equal("eager bf16 reduced bytes", scores_bf16, tp.device_group)
    idx_b, len_b = select_topk_sequence_order(scores_bf16, TOP_K)
    failures += not _all_ranks_equal("eager bf16 selected_indices", idx_b, tp.device_group)
    failures += not _all_ranks_equal("eager bf16 valid_lengths", len_b, tp.device_group)

    # --- fp32 reference reduce (today's path) for the lossiness preview ---
    scores_fp32 = partial.clone()
    reduce_token_scores(scores_fp32, process_group=tp.device_group, use_bf16=False)
    torch.cuda.synchronize()
    idx_f, len_f = select_topk_sequence_order(scores_fp32, TOP_K)
    rows_diff = int((~((idx_b == idx_f).all(dim=-1) & (len_b == len_f))).sum())
    pos_diff = int((idx_b != idx_f).sum())
    if local_rank == 0:
        print(
            f"[m1-xrank] bf16-vs-fp32 selection delta (diagnostic): "
            f"{rows_diff}/{BS} rows differ, {pos_diff} of {BS * TOP_K} positions",
            flush=True,
        )

    # --- graph-captured bf16 reduce, replay with mutated input ---
    static_scores = partial.clone()
    before = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    with graph_capture() as gc_ctx:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=gc_ctx.stream):
            reduce_token_scores(
                static_scores,
                process_group=tp.device_group,
                reduce_ca=ca,
                bf16_scratch=bf16_scratch,
                use_bf16=True,
            )
    # Mutate the static input via copy_ (graphs capture addresses) and replay.
    static_scores.copy_(partial * 0.5)
    static_scores.masked_fill_(mask_invalid, float("-inf"))
    alloc_before = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    g.replay()
    torch.cuda.synchronize()
    alloc_after = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    new_allocs = alloc_after - alloc_before
    if local_rank == 0:
        print(f"[m1-xrank] graph replay new allocations: {new_allocs}", flush=True)
    failures += new_allocs > 0
    failures += not _all_ranks_equal(
        "replayed bf16 reduced bytes", static_scores, tp.device_group
    )
    idx_r, len_r = select_topk_sequence_order(static_scores, TOP_K)
    failures += not _all_ranks_equal("replayed selection", idx_r, tp.device_group)
    del before

    dist.barrier()
    if local_rank == 0:
        print(f"[m1-xrank] {'PASS' if failures == 0 else f'FAIL ({failures})'}", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
