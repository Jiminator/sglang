"""Diagnostic capture for the AC-2.1 forced-all dense downstream-isolation assertions.

When ``forced_all_dense_control`` AND ``forced_all_assert`` are both on (eager only), the model dumps —
per (tp_rank, req_pool_index, layer_id, decode_step) — the post-adapter physical selected slots, the
request's ``req_to_token[req, 0:seq_len]`` slice, the ``_ds_slot_written[layer_id, physical_slot]``
validity bit for each live physical slot, the KV-slot capacity, and the ``logical_to_physical`` error
count. An offline reducer then verifies that the forced dense sweep ``[0..seq_len-1]`` maps to exactly
the request's own KV slots, that every selected physical slot is WRITTEN (not a reused/unwritten slot),
and that there is no duplicate / live-lane ``-1`` / out-of-range (vs KV capacity) / adapter error — per
layer AND per decode step. That makes the dense forced-all selection a PROVABLE no-op, so any residual
dense degradation is downstream of selection (H3).

`_ds_slot_written` is the SEPARATE backend validity bitmap (resolved by the caller the same way the
production/reference selector does) — physical==req_to_token proves the gather, NOT slot validity.

Host-side copy only; NO mutation of the selected set; production byte-identical when the flag is off.
Mirrors ``score_capture`` / ``selection_capture`` (config-borne, default-off, ``--disable-cuda-graph``).
"""
import os
from collections import defaultdict

import torch

_DUMP_DIR_ENV = "SGLANG_DS_FORCEDALL_ASSERT_DIR"
# Per (rank, req, layer) monotonic decode-step counter — there is no decode-step id at the seam, so
# this distinguishes successive decode steps for the same request/layer (no overwrite). Eager runs only.
_STEP_COUNTER: dict = defaultdict(int)


def forced_all_assert_dir() -> str:
    return os.environ.get(_DUMP_DIR_ENV) or os.path.join(os.getcwd(), ".sglang_ds_forcedall")


def _resolve_tp_rank() -> int:
    try:
        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
        )

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001
        return 0


def maybe_dump_forced_all_assert(
    *,
    ds_out: torch.Tensor,             # int32 [bs, max_top_k] PHYSICAL slots (post-adapter), -1 padded
    selected_indices: torch.Tensor,   # int32 [bs, max_top_k] LOGICAL positions, -1 padded
    valid_lengths: torch.Tensor,      # int32 [bs]
    req_pool_indices: torch.Tensor,   # [bs]
    req_to_token: torch.Tensor,       # int32 [num_pools, max_seqlen]
    seq_lens: torch.Tensor,           # [bs]
    slot_written: torch.Tensor,       # bool [num_ds_layers, kv_slots] — the validity bitmap
    error_count: int,
    layer_id: int,
) -> None:
    """Dump one record per request row, per decode step. No-op on any missing/empty input."""
    if ds_out is None or req_pool_indices is None or req_to_token is None or seq_lens is None:
        return
    if slot_written is None:
        return
    bs = int(ds_out.shape[0])
    if bs == 0:
        return
    dump_dir = forced_all_assert_dir()
    os.makedirs(dump_dir, exist_ok=True)
    tp = _resolve_tp_rank()
    kv_capacity = int(slot_written.shape[1])
    sw_row = slot_written[int(layer_id)]                 # [kv_slots] bool for this DS layer
    rpi_c = req_pool_indices[:bs].detach().to("cpu").tolist()
    sl_c = seq_lens[:bs].detach().to("cpu").tolist()
    vl_c = valid_lengths[:bs].detach().to("cpu").tolist()
    phys = ds_out.detach()
    logi = selected_indices.detach()
    for b in range(bs):
        seq_len = int(sl_c[b])
        if seq_len <= 0:
            continue
        req = int(rpi_c[b])
        vlen = int(vl_c[b])
        live = phys[b, :vlen].to(torch.long)
        # validity bit for each LIVE physical slot (clamp so an out-of-range index can't IndexError;
        # the reducer flags out-of-range separately from unwritten).
        live_clamped = live.clamp(min=0, max=kv_capacity - 1)
        written_bits = sw_row[live_clamped].detach().to("cpu").contiguous()
        step = _STEP_COUNTER[(tp, req, int(layer_id))]
        _STEP_COUNTER[(tp, req, int(layer_id))] += 1
        record = {
            "tp_rank": tp,
            "req_pool_index": req,
            "layer_id": int(layer_id),
            "decode_step": int(step),
            "seq_len": seq_len,
            "valid_length": vlen,
            "physical_slots": phys[b].to("cpu").contiguous(),    # [max_top_k] -1 padded
            "logical_positions": logi[b].to("cpu").contiguous(), # [max_top_k] -1 padded
            "expected_physical": req_to_token[req, :seq_len].detach().to("cpu").contiguous(),
            "slot_written_bits": written_bits,                   # [valid_length] bool — live-slot validity
            "kv_capacity": kv_capacity,                          # _ds_slot_written.shape[1]
            "adapter_error_count": int(error_count),
            "req_to_token_width": int(req_to_token.shape[1]),    # logical-position bound only
        }
        fname = f"rank{tp}_req{req:04d}_layer{int(layer_id):03d}_step{int(step):05d}.pt"
        torch.save(record, os.path.join(dump_dir, fname))
