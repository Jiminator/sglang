"""Diagnostic capture for the AC-3.1 captured-row materialized-K selected-index equality.

When ``materialized_k_capture`` is on (eager only, reference selector path), the model dumps — per
(tp_rank, layer_id, regime, decode_step), capped per bucket — a SELF-CONTAINED minimal reconstruction of the
inputs the served reference scorer (``reference_rawdot_select`` →
:func:`absorbed_latent_score_logical_fp8`) consumed for the FIRST request in the batch: the per-request
query, the GATHERED live fp8 latent + scales (only ``req_to_token[req, :seq_len]`` slots, not the whole KV
pool), the live ``_ds_slot_written`` bits, and the per-DS-layer absorbed projection ``w_sel`` /
``channel_selection`` / ``channel_weights``. An offline reducer rebuilds a bs=1 call (``req_to_token =
[[0..seq_len-1]]`` over the captured live latent) and runs BOTH the absorbed raw-dot
(:func:`absorbed_latent_score_logical_fp8`) and the materialized-K cosine-numerator
(:func:`absorbed_latent_cosine_logical_fp8` with ``normalize=False``) on the SAME captured inputs, then
asserts top-2048 selected-index equality — proving the absorbed raw-dot ceiling equals the materialized
fp32 ``K_label`` score on REAL decode rows (the captured-row form of the synthetic algebra identity).

Host-side copy only; NO mutation of the selected set; production byte-identical when the flag is off.
Mirrors ``latent_capture`` / ``forced_all_assert_capture`` (config-borne, default-off, eager only).
"""
import os
from collections import defaultdict

import torch

_DUMP_DIR_ENV = "SGLANG_DS_MATK_CAPTURE_DIR"
# Bound the capture: it dumps live latent tensors (a few MB each), so cap per (rank, layer, regime) and only
# the first few DS layers. A handful of rows is conclusive — the raw-vs-materialized identity is exact
# algebra, so it holds on every row.
_MAX_PER_BUCKET = 2
_MAX_LAYER = 6
_BUCKET_COUNT: dict = defaultdict(int)  # (rank, layer_id, regime) -> count


def materialized_k_capture_dir() -> str:
    return os.environ.get(_DUMP_DIR_ENV) or os.path.join(os.getcwd(), ".sglang_ds_matk")


def _resolve_tp_rank() -> int:
    try:
        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
        )

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001
        return 0


def maybe_dump_materialized_k(
    *,
    queries: torch.Tensor,
    latent_fp8: torch.Tensor,
    latent_scales: torch.Tensor,
    w_sel: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    max_top_k: int,
    written: torch.Tensor,
    head_agg: str,
    layer_id: int,
) -> None:
    """Dump a self-contained minimal (bs=1) reconstruction for the first request, regime-aware-capped."""
    if layer_id >= _MAX_LAYER or queries is None or queries.shape[0] == 0:
        return
    b = 0
    seq_len = int(seq_lens[b].item())
    if seq_len <= 0:
        return
    regime = "sparse" if seq_len > int(max_top_k) else "dense"
    rank = _resolve_tp_rank()
    key = (rank, int(layer_id), regime)
    step = _BUCKET_COUNT[key]
    if step >= _MAX_PER_BUCKET:
        return
    _BUCKET_COUNT[key] += 1

    rpi = int(req_pool_indices[b].item())
    live = req_to_token[rpi, :seq_len].long()  # physical slots for this request's live positions
    rec = {
        # the absorbed query for this single request (bs=1 kept)
        "queries": queries[b : b + 1].detach().to("cpu").clone(),
        # gathered live fp8 latent + scales (only the live slots), so the reducer needs no KV pool
        "live_latent_fp8": latent_fp8[live].detach().to("cpu").clone(),      # [seq_len, lora] fp8
        "live_latent_scales": latent_scales[live].detach().to("cpu").clone(),  # [seq_len, nblk]
        "live_written": written[live].detach().to("cpu").clone(),            # [seq_len] bool
        # per-DS-layer static projection + channel mask (already this-layer-indexed at the call site)
        "w_sel": w_sel.detach().to("cpu").clone(),
        "channel_selection": channel_selection.detach().to("cpu").clone(),
        "channel_weights": channel_weights.detach().to("cpu").clone(),
        "seq_len": seq_len,
        "max_top_k": int(max_top_k),
        "head_agg": str(head_agg),
        "regime": regime,
        "tp_rank": rank,
        "req_pool_index": rpi,
        "layer_id": int(layer_id),
        "decode_step": step,
    }
    dump_dir = materialized_k_capture_dir()
    os.makedirs(dump_dir, exist_ok=True)
    fname = f"rank{rank}_layer{int(layer_id):03d}_{regime}_step{step}.pt"
    torch.save(rec, os.path.join(dump_dir, fname))
