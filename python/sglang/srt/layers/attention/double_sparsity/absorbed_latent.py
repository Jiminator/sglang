"""Absorbed-latent Double Sparsity scoring.

Double Sparsity selects tokens by ``score = query · signature`` where the
signature is ``channel_select(W_UK · c_kv)`` — the per-head K_nope
(``k_nope[h] = W_UK[h] · c_kv``) sliced to the offline-mask channels ``S_h``,
with the channel weights ``w_c`` applied on the query side (query =
``w_c · q_{S_h}``).

Substituting ``k_nope[t,h,c] = Σ_l W_UK[h][c,l] · c_kv[t,l]`` collapses the score
to an inner product against the resident latent::

    score[b,t] = max_h Σ_{c∈S_h} (w_c[h,c] · q[b,h,c]) · k_nope[t,h,c]
               = max_h Σ_l ( Σ_{c∈S_h} w_c[h,c]·q[b,h,c]·W_UK[h][c,l] ) · c_kv[t,l]
               = max_h ( v_h[b] · c_kv[t] )                         # scorer_norm="off"

so the per-token signature IS the latent: the ``v_h`` projection (a few MACs per
head per step) is built query-side from the bind-time-selected ``W_UK`` rows, and
the key side is the fp8 latent the KV pool already stores. No separate signature
store and no prefill label-write hook are needed; the score reads the resident
latent directly (``scorer_norm="off"``).

This module owns the production absorbed-latent scoring math (``scorer_norm="off"``
only): the bind-time ``build_absorbed_projection`` and the per-step query-side
``absorbed_latent_v`` build, plus CPU reference scorers used as the exact oracle
for the Triton kernel.
"""

from __future__ import annotations

import torch


def build_absorbed_projection(
    kv_b_proj_weight: torch.Tensor,
    *,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    channel_selection: torch.Tensor,
    weight_scale_inv: torch.Tensor = None,
    weight_block_size=None,
) -> torch.Tensor:
    """Bind-time absorbed projection: dequantize the real ``kv_b_proj`` weight and
    return the SELECTED ``W_UK`` rows ``[H, label_dim, kv_lora_rank]`` (fp32).

    Mirrors the model's own ``w_kc`` extraction (deepseek_weight_loader): the
    block-fp8 ``[out, in]`` weight (``out = H·(qk_nope+v_head)``, ``in =
    kv_lora_rank``) is dequantized with the SAME block-fp8 semantics attention
    uses, reshaped to ``[H, qk_nope+v_head, lora]``, sliced to the K-noPE rows
    ``[:qk_nope_head_dim]`` (rope dims excluded by construction), then GATHERED at
    each head's mask channels ``S_h``. Built once at bind; the result is exactly
    the per-mask-channel ``W_UK`` rows the absorbed score consumes, so the
    per-token signature then IS the resident latent.

    Args:
        kv_b_proj_weight: ``[H·(qk_nope+v_head), kv_lora_rank]``.
        channel_selection: ``[H, label_dim]`` int — mask channel indices into
            ``qk_nope_head_dim``.
        weight_scale_inv / weight_block_size: block-fp8 dequant inputs; when both
            present, dequantize via ``block_quant_dequant``, else ``.float()``.

    Returns:
        ``[H, label_dim, kv_lora_rank]`` fp32 — ``W_UK[h][S_h[d], :]``.
    """
    if weight_scale_inv is not None and weight_block_size is not None:
        from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

        w = block_quant_dequant(
            kv_b_proj_weight, weight_scale_inv, list(weight_block_size), torch.float32
        )
    else:
        w = kv_b_proj_weight.to(torch.float32)
    out, lora = w.shape
    head_width = qk_nope_head_dim + v_head_dim
    assert (
        out == num_heads * head_width
    ), f"kv_b_proj out {out} != num_heads*(qk_nope+v_head) {num_heads * head_width}"
    # [H, qk_nope+v_head, lora] -> K-noPE rows -> W_UK [H, qk_nope, lora]
    w_kc = w.view(num_heads, head_width, lora)[:, :qk_nope_head_dim, :]
    # gather the mask channels per head -> [H, label_dim, lora]
    sel = channel_selection.long().to(w_kc.device)
    return torch.gather(w_kc, 1, sel.unsqueeze(-1).expand(-1, -1, lora)).contiguous()


def absorbed_latent_v(
    queries: torch.Tensor,
    w_sel: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
) -> torch.Tensor:
    """Per-head query-side latent projection ``v_h`` (the absorbed score query),
    fp32 to match the production ``_logical_score`` accumulation.

    Args:
        queries: ``[bs, H, qk_nope_head_dim]`` — no-PE query, BEFORE channel
            projection (same input as ``project_query_onto_channels``).
        w_sel: ``[H, label_dim, kv_lora_rank]`` — the bind-time-selected ``W_UK``
            rows from :func:`build_absorbed_projection`.
        channel_selection: ``[H, label_dim]`` int — the query channels ``S_h``.
        channel_weights: ``[H, label_dim]`` float — the per-channel weights ``w_c``.

    Returns:
        ``[bs, H, kv_lora_rank]`` fp32 — ``v_h[b] = Σ_{c∈S_h} w_c·q_c · W_UK[h][c,:]``.
    """
    bs = queries.shape[0]
    sel = channel_selection.long()
    # weighted query at the selected channels: w_c · q_{S_h}  -> [bs, H, label_dim]
    q_sel = torch.gather(
        queries.to(torch.float32), 2, sel.unsqueeze(0).expand(bs, -1, -1)
    ) * channel_weights.to(torch.float32).unsqueeze(0)
    # v_h = Σ_d (w_c·q_c) · w_sel[h, d, :]  -> [bs, H, lora]
    return torch.einsum("bhd,hdl->bhl", q_sel, w_sel.to(torch.float32))


def absorbed_latent_v_into(
    out: torch.Tensor,
    queries: torch.Tensor,
    w_sel: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    *,
    scratch_qsel: torch.Tensor,
    scratch_q: torch.Tensor,
) -> torch.Tensor:
    """Allocation-free :func:`absorbed_latent_v` — writes ``v_h`` into a caller-owned
    ``out[:bs]`` (fp32 ``[bs_buf, H, kv_lora_rank]``) for the graph-safe path.

    Same value as :func:`absorbed_latent_v`: ``v_h[b,h,l] = Σ_d (w_c·q_c)·w_sel[h,d,l]``.
    Every step writes into caller-owned scratch (an in-place fp32 cast of ``queries``,
    ``gather(out=)``, ``mul_``, then a head-batched ``bmm(out=)`` over a transposed
    view of ``out``), so after warmup the caching-allocator counter does not grow —
    CUDA-graph safe. ``scratch_qsel`` is fp32 ``[bs_buf, H, label_dim]`` for the
    weighted-gather; ``scratch_q`` is fp32 ``[bs_buf, H, qk_nope_head_dim]`` that the
    bf16/fp16 served ``queries`` are cast into in place (``copy_`` does the dtype cast
    with no allocation), so the hot path never calls ``queries.to(torch.float32)``.
    ``channel_selection`` must be int64 here (the caller pre-copies the int32 layer
    mask into an int64 scratch so ``gather`` does no per-step ``.long()`` allocation).
    """
    bs = queries.shape[0]
    out_v = out[:bs]  # [bs, H, lora]
    label_dim = int(channel_selection.shape[1])
    q_sel = scratch_qsel[:bs, :, :label_dim]  # [bs, H, label_dim]
    # Cast the served bf16/fp16 query into the fp32 scratch in place (copy_ does
    # the dtype conversion without allocating), so the gather reads fp32 without a
    # per-step queries.to(torch.float32).
    q_f32 = scratch_q[:bs, :, : queries.shape[2]]  # [bs, H, qk_nope_head_dim]
    q_f32.copy_(queries)
    # weighted query at the selected channels: w_c · q_{S_h}  -> [bs, H, label_dim].
    torch.gather(
        q_f32,
        2,
        channel_selection.unsqueeze(0).expand(bs, -1, -1),
        out=q_sel,
    )
    q_sel.mul_(channel_weights.to(torch.float32).unsqueeze(0))
    # v_h = Σ_d (w_c·q_c) · w_sel[h, d, :] as a per-head bmm: batch the head axis so
    # the contraction is [H, bs, label_dim] @ [H, label_dim, lora] -> [H, bs, lora],
    # written straight into a head-major (transposed) view of out_v with out=.
    q_hbd = q_sel.transpose(0, 1)  # [H, bs, label_dim] (view)
    w_sel_f = w_sel.to(torch.float32)  # [H, label_dim, lora]
    v_hbl = out_v.transpose(0, 1)  # [H, bs, lora] (view of out_v)
    torch.bmm(q_hbd, w_sel_f, out=v_hbl)
    return out_v


def absorbed_latent_score(
    queries: torch.Tensor,
    c_kv: torch.Tensor,
    w_sel: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    head_agg: str = "max",
) -> torch.Tensor:
    """Per-(query, token) selection score from the latent, no table.

    ``score[b, t] = agg_h ( v_h[b] · c_kv[t] )`` for ``scorer_norm="off"``.

    Args:
        queries: ``[bs, H, qk_nope_head_dim]``.
        c_kv: ``[T, kv_lora_rank]`` — the resident MLA KV latent (dequantized).
        w_sel, channel_selection, channel_weights: see :func:`absorbed_latent_v`.
        head_agg: ``"max"`` (default) or ``"mean"``.

    Returns:
        ``[bs, T]`` fp32 scores.
    """
    v = absorbed_latent_v(queries, w_sel, channel_selection, channel_weights)
    dots = torch.einsum("bhl,tl->bht", v, c_kv.to(torch.float32))  # [bs, H, T]
    if head_agg == "mean":
        return dots.mean(dim=1)
    return dots.amax(dim=1)


def absorbed_latent_score_logical(
    queries: torch.Tensor,
    c_kv: torch.Tensor,
    w_sel: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    written: torch.Tensor = None,
    head_agg: str = "max",
) -> torch.Tensor:
    """Logical-domain paged absorbed score — scores the resident latent
    directly (no materialized signatures).

    For each request, walks logical positions ``0..max_seq_len`` through
    ``req_to_token[pool, pos] -> physical slot``, gathers ``c_kv`` at the physical
    slot, scores ``agg_h (v_h · c_kv[slot])``, and masks unwritten slots (if
    ``written`` given) then positions ``>= seq_len`` to ``-inf`` — same order as
    the production logical scorer. Returns ``[bs, max_seq_len]`` fp32 (feed to
    ``select_topk_sequence_order``).

    Args:
        queries: ``[bs, H, qk_nope_head_dim]``.
        c_kv: ``[max_tokens, kv_lora_rank]`` physical-slot latent (dequantized).
        w_sel: ``[H, label_dim, kv_lora_rank]`` (from :func:`build_absorbed_projection`).
        req_pool_indices: ``[bs]`` int; req_to_token: ``[num_pools, max_seqlen]`` int;
        seq_lens: ``[bs]`` int; written: optional ``[max_tokens]`` bool.
    """
    bs = queries.shape[0]
    device = queries.device
    if max_seq_len <= 0:
        return torch.full((bs, 1), float("-inf"), dtype=torch.float32, device=device)
    v = absorbed_latent_v(queries, w_sel, channel_selection, channel_weights)
    safe_pool = req_pool_indices.clamp(0, max(req_to_token.shape[0] - 1, 0)).long()
    logical_positions = (
        torch.arange(max_seq_len, device=device).unsqueeze(0).expand(bs, -1)
    )
    safe_positions = logical_positions.clamp(0, max(req_to_token.shape[1] - 1, 0))
    pool_expanded = safe_pool.unsqueeze(1).expand(-1, max_seq_len)
    physical_slots = req_to_token[pool_expanded, safe_positions.long()]
    max_tokens = c_kv.shape[0]
    safe_phys = physical_slots.long().clamp(0, max(max_tokens - 1, 0))
    gathered = c_kv[safe_phys].to(torch.float32)  # [bs, max_seq_len, lora]
    dots = torch.einsum("bhl,bil->bih", v, gathered)  # [bs, max_seq_len, H]
    scores = dots.mean(dim=-1) if head_agg == "mean" else dots.amax(dim=-1)
    if written is not None:
        scores = scores.masked_fill(~written[safe_phys], float("-inf"))
    seq_len_mask = logical_positions < seq_lens.unsqueeze(1).to(device)
    return scores.masked_fill(~seq_len_mask, float("-inf"))


def dequantize_resident_latent(
    latent_fp8: torch.Tensor, latent_scales: torch.Tensor
) -> torch.Tensor:
    """fp8-e4m3 resident latent + per-block fp32 scales -> fp32 ``c_kv`` ``[T, lora]``.

    The KV pool stores the MLA noPE latent as fp8 with one fp32 scale per
    128-channel block. This reverses that exactly in fp32 — the
    performance-naive dequant the reference selector scores against (no
    in-register fp8 dequant, no tf32). ``latent_fp8`` is ``[T, lora]`` viewed as
    ``float8_e4m3fn``; ``latent_scales`` is ``[T, nblk]`` fp32.
    """
    t, lora = latent_fp8.shape
    nblk = latent_scales.shape[-1]
    block = lora // nblk
    deq = latent_fp8.to(torch.float32).view(t, nblk, block)
    deq = deq * latent_scales.to(torch.float32).view(t, nblk, 1)
    return deq.view(t, lora)


def reference_rawdot_select(
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
    max_top_k: int,
    written: torch.Tensor = None,
    head_agg: str = "max",
):
    """Performance-naive fp32 raw-dot reference selection.

    Dequantizes the resident latent to fp32, scores the exact absorbed
    channel-dot (``absorbed_latent_score_logical``), and takes an exact
    full-width ``torch.topk`` (``select_topk_sequence_order``) — no fp8-in-register
    dequant, no bf16 reduce, no radix approximation, no selector-width bucketing.
    Returns ``(selected_indices int32 [bs, max_top_k] ascending, -1 padded;
    valid_lengths int32 [bs])`` — byte-identical in shape/semantics to the
    production graph-safe selector output.
    """
    from .selection_kernel import select_topk_sequence_order

    c_kv = dequantize_resident_latent(latent_fp8, latent_scales)
    scores = absorbed_latent_score_logical(
        queries=queries,
        c_kv=c_kv,
        w_sel=w_sel,
        channel_selection=channel_selection,
        channel_weights=channel_weights,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        written=written,
        head_agg=head_agg,
    )
    return select_topk_sequence_order(scores, max_top_k)


def apply_forced_all_dense(
    selected_indices: torch.Tensor,
    valid_lengths: torch.Tensor,
    seq_lens: torch.Tensor,
    max_top_k: int,
):
    """Downstream-isolation control: for rows where ``seq_len <= max_top_k``,
    replace the scored selection with the logical sweep ``[0 .. seq_len-1]``
    (ascending, ``-1`` padded) so selection is provably a no-op and any residual
    degradation is downstream of selection (the adapter / slot-validity / kernel
    feed). Rows with ``seq_len > max_top_k`` are left unchanged (keeping all
    tokens is undefined when they do not fit). Returns new
    ``(selected_indices, valid_lengths)`` in the production format.
    """
    bs, width = selected_indices.shape
    device = selected_indices.device
    out_idx = selected_indices.clone()
    out_len = valid_lengths.clone()
    positions = torch.arange(width, device=device, dtype=out_idx.dtype)
    for b in range(bs):
        s = int(seq_lens[b])
        if s <= max_top_k:
            n = min(s, width)
            out_idx[b].fill_(-1)
            out_idx[b, :n] = positions[:n]
            out_len[b] = n
    return out_idx, out_len.to(valid_lengths.dtype)
