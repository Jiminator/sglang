"""Absorbed-latent Double Sparsity scoring (score-only prototype).

Double Sparsity selects tokens by ``score = query · signature`` where the
materialized signature is ``channel_select(W_UK · c_kv)`` — the per-head K_nope
(``k_nope[h] = W_UK[h] · c_kv``) sliced to the offline-mask channels ``S_h``,
with the channel weights ``w_c`` applied on the query side. See
``token_label_write.py`` (write = ``k_nope[:, :, S_h]``) and
``selection_kernel.project_query_onto_channels`` (query = ``w_c · q_{S_h}``).

Substituting ``k_nope[t,h,c] = Σ_l W_UK[h][c,l] · c_kv[t,l]`` collapses the score
to an inner product against the resident latent::

    score[b,t] = max_h Σ_{c∈S_h} (w_c[h,c] · q[b,h,c]) · k_nope[t,h,c]
               = max_h Σ_l ( Σ_{c∈S_h} w_c[h,c]·q[b,h,c]·W_UK[h][c,l] ) · c_kv[t,l]
               = max_h ( v_h[b] · c_kv[t] )                         # scorer_norm="off"

so the per-token signature IS the latent: the ``v_h`` projection (a few MACs per
head per step) is built query-side and the key side is the fp8 latent the KV pool
already stores. The TokenLabelTable, the prefill label-write hook, and its
GB/rank disappear exactly — only fp32 reassociation and (in serving) the
fp8-quantized latent vs the bf16-pre-quant label distinguish the two, which is a
declared value-affecting change gated by recall@2048 (±0.5pp) + selection
equivalence, NOT a bit-identity claim.

This module is the score-only DIAGNOSTIC reference (``scorer_norm="off"`` only).
It does not change the selector ABI or delete the table — that is task6. It backs
the live-path equivalence + recall gates that must pass before the swap.
"""

from __future__ import annotations

import torch


def absorbed_latent_v(
    queries: torch.Tensor,
    w_kc: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
) -> torch.Tensor:
    """Build the per-head query-side latent projection ``v_h`` (the absorbed score
    query). All math in fp32 to match the production ``_logical_score`` accumulation.

    Args:
        queries: ``[bs, H, qk_nope_head_dim]`` — the no-PE query, BEFORE channel
            projection (same input as ``project_query_onto_channels``).
        w_kc: ``[H, qk_nope_head_dim, kv_lora_rank]`` — the K-noPE up-projection
            (``W_UK``; ``k_nope[h] = w_kc[h] · c_kv``), per local head.
        channel_selection: ``[H, label_dim]`` int — mask channel indices into
            ``qk_nope_head_dim``.
        channel_weights: ``[H, label_dim]`` float — the per-channel weights ``w_c``.

    Returns:
        ``[bs, H, kv_lora_rank]`` fp32 — ``v_h[b] = Σ_{c∈S_h} w_c·q_c · W_UK[h][c,:]``.
    """
    bs = queries.shape[0]
    lora = w_kc.shape[-1]
    sel = channel_selection.long()  # [H, label_dim]
    # weighted query at the selected channels: w_c · q_{S_h}  -> [bs, H, label_dim]
    q_sel = torch.gather(
        queries.to(torch.float32), 2, sel.unsqueeze(0).expand(bs, -1, -1)
    ) * channel_weights.to(torch.float32).unsqueeze(0)
    # the W_UK rows at the selected channels: [H, label_dim, lora]
    w_sel = torch.gather(
        w_kc.to(torch.float32), 1, sel.unsqueeze(-1).expand(-1, -1, lora)
    )
    # v_h = Σ_{c∈S_h} (w_c·q_c) · W_UK[h][c, :]  -> [bs, H, lora]
    return torch.einsum("bhd,hdl->bhl", q_sel, w_sel)


def absorbed_latent_score(
    queries: torch.Tensor,
    c_kv: torch.Tensor,
    w_kc: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    head_agg: str = "max",
) -> torch.Tensor:
    """Per-(query, token) selection score computed from the latent, no table.

    ``score[b, t] = agg_h ( v_h[b] · c_kv[t] )`` for ``scorer_norm="off"``.

    Args:
        queries: ``[bs, H, qk_nope_head_dim]``.
        c_kv: ``[T, kv_lora_rank]`` — the resident MLA KV latent (dequantized).
        w_kc, channel_selection, channel_weights: see :func:`absorbed_latent_v`.
        head_agg: ``"max"`` (default) or ``"mean"`` — matches the scorer's
            cross-head aggregation.

    Returns:
        ``[bs, T]`` fp32 scores.
    """
    v = absorbed_latent_v(queries, w_kc, channel_selection, channel_weights)
    dots = torch.einsum("bhl,tl->bht", v, c_kv.to(torch.float32))  # [bs, H, T]
    if head_agg == "mean":
        return dots.mean(dim=1)
    return dots.amax(dim=1)
