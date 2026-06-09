"""Async invariant probes — fire torch._assert_async without CPU sync.

All probes are gated on SGLANG_ENABLE_ASYNC_ASSERT (default off in prod).
When the gate is on, a violation surfaces as an assertion at the next CUDA
sync point instead of as a silent NaN cascade or illegal-address crash.
"""

from typing import Optional

import torch

from sglang.srt.environ import envs


def maybe_detect_nan(tensor: Optional[torch.Tensor], msg: str = ""):
    """Async NaN check — no GPU-CPU sync, error surfaces at next sync point."""
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    # A None tensor means there is nothing to probe, e.g. hidden_states on
    # capture_hidden_mode=NULL paths (STANDALONE speculative decoding).
    if tensor is None:
        return
    torch._assert_async(~torch.any(torch.isnan(tensor)), f"NaN detected! {msg}")


def maybe_detect_inf(tensor: Optional[torch.Tensor], msg: str = ""):
    """Async Inf check — fp16 overflow surfaces as Inf before NaN."""
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if tensor is None:
        return
    torch._assert_async(~torch.any(torch.isinf(tensor)), f"Inf detected! {msg}")


def maybe_detect_oob(indices: Optional[torch.Tensor], low: int, high: int, msg: str):
    """Async OOB check — no GPU-CPU sync, error surfaces at next sync point.

    Low/high asserted separately so the message names which failed (low =
    negative/sentinel, high = out of range).
    """
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if indices is None or indices.numel() == 0:
        return
    torch._assert_async(
        indices.min() >= low,
        f"index < {low} (negative / unmasked sentinel?): {msg}",
    )
    torch._assert_async(
        indices.max() < high,
        f"index >= {high} (out of range): {msg}",
    )


def maybe_assert_page_table_in_range(
    page_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    num_kv_slots: int,
    page_size: int = 1,
    msg: str = "",
):
    """Async read-side guard for the FA3 paged-KV gather (read-side complement to
    ``maybe_detect_oob``).

    The FA3 paged kernel uses each page_table value it reads as a KV slot index
    with no bounds check. This asserts that every slot the kernel will dereference
    -- ``page_table[i, :cdiv(cache_seqlens[i], page_size)]`` -- is in
    ``[0, num_kv_slots)``; positions past a row's covered length are masked (the
    kernel never reads them), matching the kernel's own ``row_idx < seqlen_k`` mask.

    Call once per forward where page_table/cache_seqlens are finalized, NOT per layer.
    """
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if (
        page_table is None
        or cache_seqlens is None
        or page_table.numel() == 0
        or cache_seqlens.numel() == 0
    ):
        return
    cols = torch.arange(page_table.shape[1], device=page_table.device)
    covered_len = (cache_seqlens.to(torch.int64) + (page_size - 1)) // page_size
    covered = cols.unsqueeze(0) < covered_len.unsqueeze(1)  # (B, L) bool
    slots = page_table.to(torch.int64)
    # neutralize masked (uncovered) positions to a known-valid slot so only the
    # positions the kernel actually dereferences are asserted.
    safe = torch.where(covered, slots, torch.zeros_like(slots))
    torch._assert_async(
        safe.min() >= 0,
        f"FA3 page_table slot id < 0 (cache_seqlens/req_to_token mismatch): {msg}",
    )
    torch._assert_async(
        safe.max() < num_kv_slots,
        f"FA3 page_table slot id >= {num_kv_slots} (cache_seqlens/req_to_token mismatch): {msg}",
    )


def maybe_assert_seqlens_within_context(
    cache_seqlens: Optional[torch.Tensor],
    max_context_len: int,
    msg: str = "",
):
    """Async fail-fast: a ``cache_seqlen > max_context_len`` would index
    ``req_to_token`` past its columns (OOB read -> garbage slot), so assert it here
    and surface an attributable error naming the context limit. Complements
    ``maybe_assert_page_table_in_range``, which guards the slot *values*.
    """
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if cache_seqlens is None or cache_seqlens.numel() == 0:
        return
    torch._assert_async(
        cache_seqlens.max() <= max_context_len,
        f"cache_seqlen exceeds context_len={max_context_len} "
        f"(seq_len/req_to_token boundary overshoot): {msg}",
    )


def maybe_detect_page_aligned(
    indices: Optional[torch.Tensor], page_size: int, msg: str
):
    """Async page-alignment check on slot ids."""
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if indices is None or indices.numel() == 0 or page_size <= 1:
        return
    torch._assert_async(
        (indices % page_size == 0).all(),
        f"page-misaligned indices (page_size={page_size}): {msg}",
    )
