"""Diagnostic capture for the AC-2.1 forced-all dense downstream-isolation assertions.

When ``forced_all_dense_control`` AND ``forced_all_assert`` are both on (eager only), the model dumps —
per (tp_rank, req_pool_index, layer_id) — the post-adapter physical selected slots together with the
request's ``req_to_token[req_pool, 0:seq_len]`` slice and the ``logical_to_physical`` error count, so an
offline reducer can verify that the forced dense sweep ``[0..seq_len-1]`` maps to exactly the request's
own KV slots (no duplicate / live-lane ``-1`` / out-of-range / adapter error). That makes the dense
forced-all selection a PROVABLE no-op, so any residual dense degradation is downstream of selection (H3).

Host-side copy only; NO mutation of the selected set; production byte-identical when the flag is off.
Mirrors ``score_capture`` / ``selection_capture`` (config-borne, default-off, ``--disable-cuda-graph``).
"""
import os

import torch

_DUMP_DIR_ENV = "SGLANG_DS_FORCEDALL_ASSERT_DIR"


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
    error_count: int,
    layer_id: int,
) -> None:
    """Dump one record per request row. No-op on any missing/empty input."""
    if ds_out is None or req_pool_indices is None or req_to_token is None or seq_lens is None:
        return
    bs = int(ds_out.shape[0])
    if bs == 0:
        return
    dump_dir = forced_all_assert_dir()
    os.makedirs(dump_dir, exist_ok=True)
    tp = _resolve_tp_rank()
    rpi_c = req_pool_indices[:bs].detach().to("cpu").tolist()
    sl_c = seq_lens[:bs].detach().to("cpu").tolist()
    vl_c = valid_lengths[:bs].detach().to("cpu").tolist()
    phys = ds_out.detach().to("cpu")
    logi = selected_indices.detach().to("cpu")
    for b in range(bs):
        seq_len = int(sl_c[b])
        if seq_len <= 0:
            continue
        req = int(rpi_c[b])
        # the request's own physical KV slots for logical positions [0, seq_len)
        expected = req_to_token[req, :seq_len].detach().to("cpu").contiguous()
        record = {
            "tp_rank": tp,
            "req_pool_index": req,
            "layer_id": int(layer_id),
            "seq_len": seq_len,
            "valid_length": int(vl_c[b]),
            "physical_slots": phys[b].contiguous(),       # [max_top_k] -1 padded
            "logical_positions": logi[b].contiguous(),    # [max_top_k] -1 padded
            "expected_physical": expected,                # req_to_token[req, 0:seq_len]
            "adapter_error_count": int(error_count),
            "req_to_token_width": int(req_to_token.shape[1]),
        }
        fname = f"rank{tp}_req{req:04d}_layer{int(layer_id):03d}.pt"
        torch.save(record, os.path.join(dump_dir, fname))
