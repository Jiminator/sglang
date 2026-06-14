"""GPU paged absorbed-latent score kernel (score-only diagnostic).

Companion to ``absorbed_latent.py``: the same identity
``score[b, t] = agg_h ( v_h[b] · c_kv[t] )`` (``scorer_norm="off"``), but the key
side is read from the RESIDENT paged fp8 MLA latent instead of a materialized
signature table, with per-128-channel-block dequantization done in-register. The
query-side projection ``v_h`` is built once per step on the host
(``absorbed_latent.absorbed_latent_v``) and handed to the kernel, so the kernel
is a paged ``max_h Σ_l v_h[b,h,l] · dequant(latent[slot,l])`` reduction.

The kernel mirrors the persistent-worker topology of the production logical-score
kernel (``selection_kernel._logical_score_kernel``): a static ``(bs, WORKERS)``
grid, each worker striding over the token blocks it owns, loop bound = the LIVE
block count, written-then-``seq_len`` masking in the production order. The
per-element dequant-then-dot matches the CPU reference value-for-value (only fp32
summation order reassociates), so the CPU ``absorbed_latent_score_logical`` is its
exact oracle. This is a diagnostic only — it does NOT change the selector ABI or
delete the table; that integration is a later step. Value-affecting (the fp8
latent vs the bf16-pre-quant label), declared, recall-gated — not a bit-identity
claim.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only import path
    _HAS_TRITON = False


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def quantize_latent_fp8(
    c_kv: torch.Tensor, *, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-128-channel-block fp8 quantization of the MLA nope latent.

    Matches the pool's ``quantize_k_cache_separate`` scheme: per block
    ``s = max|tile| / FP8_MAX; q = clamp(tile / s, ±FP8_MAX).to(fp8)``.

    Args:
        c_kv: ``[T, lora]`` fp32/bf16 latent. ``lora % block_size == 0``.

    Returns:
        ``(fp8 [T, lora] float8_e4m3fn, scales [T, lora//block_size] fp32)`` —
        the two tensors ``get_mla_kv_buffer`` exposes after unpacking the
        ``[512 fp8 | 4 fp32 scales]`` pool bytes.
    """
    T, lora = c_kv.shape
    assert lora % block_size == 0, f"lora {lora} not a multiple of block {block_size}"
    nblk = lora // block_size
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    src = c_kv.to(torch.float32)
    tiles = src.view(T, nblk, block_size)
    scales = tiles.abs().amax(dim=2) / fp8_max  # [T, nblk]
    safe = scales.clamp_min(torch.finfo(torch.float32).tiny)
    q = torch.clamp(tiles / safe.unsqueeze(2), -fp8_max, fp8_max).to(
        torch.float8_e4m3fn
    )
    return q.view(T, lora).contiguous(), scales.contiguous()


def dequantize_latent_fp8(
    fp8: torch.Tensor, scales: torch.Tensor, *, block_size: int = 128
) -> torch.Tensor:
    """Inverse of :func:`quantize_latent_fp8` — the value-for-value latent the
    kernel scores against; feed this to the CPU reference as the oracle's input."""
    T, lora = fp8.shape
    nblk = lora // block_size
    deq = fp8.to(torch.float32).view(T, nblk, block_size) * scales.unsqueeze(2)
    return deq.view(T, lora)


if _HAS_TRITON:

    @triton.jit
    def _absorbed_score_kernel(
        v_ptr,  # [bs, H, lora] fp32 (precomputed v_h)
        fp8_ptr,  # [max_tokens, lora] float8_e4m3fn (paged nope latent)
        scale_ptr,  # [max_tokens, nblk] fp32 (per-128-block scales)
        written_ptr,  # [max_tokens] bool
        rpi_ptr,  # [bs] int32
        rtt_ptr,  # [num_pools, max_pool_len] int32
        sl_ptr,  # [bs] int32
        out_ptr,  # [bs, max_seq_len] fp32 (pre-allocated)
        num_heads: tl.constexpr,
        max_seq_len: tl.constexpr,
        lora: tl.constexpr,
        block_size: tl.constexpr,
        max_pool_len: tl.constexpr,
        max_tokens: tl.constexpr,
        v_stride_b: tl.constexpr,
        v_stride_h: tl.constexpr,
        fp8_stride_t: tl.constexpr,
        scale_stride_t: tl.constexpr,
        rtt_stride_p: tl.constexpr,
        out_stride_b: tl.constexpr,
        TOKEN_BLOCK: tl.constexpr,
        LORA_POW2: tl.constexpr,
        HEAD_AGG_MEAN: tl.constexpr,
        STORE_DEAD_NEG_INF: tl.constexpr,
        WORKERS: tl.constexpr,
    ):
        # Persistent-worker layout, identical in spirit to _logical_score_kernel:
        # static (bs, WORKERS) grid; each worker strides its token blocks; the
        # loop bound is the LIVE block count (device-computed from seq_len). The
        # full-width torch.topk consumer scans the whole scratch, so dead blocks
        # are filled with -inf when STORE_DEAD_NEG_INF is set.
        batch_id = tl.program_id(0)
        worker = tl.program_id(1)

        seq_len_i = tl.load(sl_ptr + batch_id).to(tl.int32)
        n_live = tl.minimum(seq_len_i, max_seq_len)
        live_blocks = (n_live + TOKEN_BLOCK - 1) // TOKEN_BLOCK
        if STORE_DEAD_NEG_INF:
            nblk = (max_seq_len + TOKEN_BLOCK - 1) // TOKEN_BLOCK
        else:
            nblk = live_blocks

        pool_idx = tl.load(rpi_ptr + batch_id).to(tl.int64)
        l_offs = tl.arange(0, LORA_POW2)
        l_mask = l_offs < lora
        blk_of = (
            l_offs // block_size
        )  # which 128-block each channel reads its scale from

        for tok_blk in range(worker, nblk, WORKERS):
            tok_offs = tok_blk * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
            in_range = tok_offs < max_seq_len
            if tok_blk * TOKEN_BLOCK >= seq_len_i:
                tl.store(
                    out_ptr + batch_id * out_stride_b + tok_offs,
                    tl.full((TOKEN_BLOCK,), float("-inf"), dtype=tl.float32),
                    mask=in_range,
                )
            else:
                pos_valid = in_range & (tok_offs < seq_len_i)

                safe_tok = tl.minimum(tok_offs, max_pool_len - 1)
                phys = tl.load(
                    rtt_ptr + pool_idx * rtt_stride_p + safe_tok,
                    mask=in_range,
                    other=0,
                ).to(tl.int64)
                safe_phys = tl.minimum(tl.maximum(phys, 0), max_tokens - 1)

                written = tl.load(written_ptr + safe_phys, mask=in_range, other=0).to(
                    tl.int1
                )
                valid = pos_valid & written

                # Load the fp8 latent tile and dequant per element with its
                # per-128-block scale (dequant-then-dot == the CPU reference).
                lat_offs = safe_phys[:, None] * fp8_stride_t + l_offs[None, :]
                tile_mask = in_range[:, None] & l_mask[None, :]
                lat = tl.load(fp8_ptr + lat_offs, mask=tile_mask, other=0.0).to(
                    tl.float32
                )
                sc = tl.load(
                    scale_ptr + safe_phys[:, None] * scale_stride_t + blk_of[None, :],
                    mask=tile_mask,
                    other=0.0,
                ).to(tl.float32)
                lat_deq = lat * sc  # [TOKEN_BLOCK, LORA_POW2]

                if HEAD_AGG_MEAN:
                    acc = tl.zeros((TOKEN_BLOCK,), dtype=tl.float32)
                else:
                    acc = tl.full((TOKEN_BLOCK,), float("-inf"), dtype=tl.float32)

                for h in range(num_heads):
                    v_h = tl.load(
                        v_ptr + batch_id * v_stride_b + h * v_stride_h + l_offs,
                        mask=l_mask,
                        other=0.0,
                    ).to(tl.float32)
                    dot = tl.sum(lat_deq * v_h[None, :], axis=1)  # [TOKEN_BLOCK]
                    if HEAD_AGG_MEAN:
                        acc += dot
                    else:
                        acc = tl.where(dot > acc, dot, acc)

                if HEAD_AGG_MEAN:
                    acc = acc / num_heads

                out_score = tl.where(
                    valid,
                    acc,
                    tl.full(acc.shape, float("-inf"), dtype=tl.float32),
                )
                tl.store(
                    out_ptr + batch_id * out_stride_b + tok_offs,
                    out_score,
                    mask=in_range,
                )


def absorbed_score_paged_fp8(
    v: torch.Tensor,
    latent_fp8: torch.Tensor,
    latent_scales: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    written: Optional[torch.Tensor] = None,
    *,
    block_size: int = 128,
    token_block: int = 64,
    workers: int = 128,
    head_agg: str = "max",
) -> torch.Tensor:
    """Paged absorbed score from the resident fp8 latent — GPU.

    Args:
        v: ``[bs, H, lora]`` fp32 — the per-head query projection ``v_h`` (from
            ``absorbed_latent.absorbed_latent_v``).
        latent_fp8: ``[max_tokens, lora]`` float8_e4m3fn — the resident nope latent.
        latent_scales: ``[max_tokens, lora//block_size]`` fp32 — per-block scales.
        req_pool_indices / req_to_token / seq_lens: paging, as in the production
            logical scorer.
        written: optional ``[max_tokens]`` bool; ``None`` treats all slots written.

    Returns:
        ``[bs, max_seq_len]`` fp32 scores (``-inf`` on unwritten / out-of-range).
    """
    if not _HAS_TRITON:
        raise RuntimeError("absorbed_score_paged_fp8 requires Triton/CUDA")
    bs, num_heads, lora = v.shape
    max_tokens = latent_fp8.shape[0]
    assert lora % block_size == 0, f"lora {lora} not a multiple of block {block_size}"
    nblk = lora // block_size
    assert latent_scales.shape == (
        max_tokens,
        nblk,
    ), f"scales {tuple(latent_scales.shape)} != {(max_tokens, nblk)}"
    device = v.device
    if max_seq_len <= 0:
        return torch.full((bs, 1), float("-inf"), dtype=torch.float32, device=device)
    if written is None:
        written = torch.ones(max_tokens, dtype=torch.bool, device=device)

    out = torch.empty((bs, max_seq_len), dtype=torch.float32, device=device)
    max_pool_len = int(req_to_token.shape[1])
    desired_block = min(token_block, max(max_seq_len, 1))
    token_block_pow2 = _next_pow2(desired_block)
    lora_pow2 = _next_pow2(lora)
    num_token_blocks = (max_seq_len + token_block_pow2 - 1) // token_block_pow2
    num_workers = max(1, min(int(workers), num_token_blocks))
    grid = (bs, num_workers)

    _absorbed_score_kernel[grid](
        v,
        latent_fp8,
        latent_scales,
        written,
        req_pool_indices,
        req_to_token,
        seq_lens,
        out,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        lora=lora,
        block_size=block_size,
        max_pool_len=max_pool_len,
        max_tokens=max_tokens,
        v_stride_b=v.stride(0),
        v_stride_h=v.stride(1),
        fp8_stride_t=latent_fp8.stride(0),
        scale_stride_t=latent_scales.stride(0),
        rtt_stride_p=req_to_token.stride(0),
        out_stride_b=out.stride(0),
        TOKEN_BLOCK=token_block_pow2,
        LORA_POW2=lora_pow2,
        HEAD_AGG_MEAN=head_agg == "mean",
        STORE_DEAD_NEG_INF=True,
        WORKERS=num_workers,
    )
    return out


def absorbed_latent_score_logical_paged(
    queries: torch.Tensor,
    latent_fp8: torch.Tensor,
    latent_scales: torch.Tensor,
    w_sel: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    written: Optional[torch.Tensor] = None,
    *,
    block_size: int = 128,
    head_agg: str = "max",
) -> torch.Tensor:
    """GPU equivalent of ``absorbed_latent.absorbed_latent_score_logical`` reading
    the paged fp8 latent. Builds ``v_h`` host-side then launches the paged kernel.
    """
    from .absorbed_latent import absorbed_latent_v

    v = absorbed_latent_v(queries, w_sel, channel_selection, channel_weights)
    return absorbed_score_paged_fp8(
        v,
        latent_fp8,
        latent_scales,
        req_pool_indices,
        req_to_token,
        seq_lens,
        max_seq_len,
        written=written,
        block_size=block_size,
        head_agg=head_agg,
    )
