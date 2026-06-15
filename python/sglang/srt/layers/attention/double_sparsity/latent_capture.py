"""Per-request dump of the resident fp8-latent identity at the prompt slots.

Config-borne diagnostic (``latent_capture`` in the DS config JSON — env vars do
not reach the TP worker subprocesses). It is the table-free radix fixture's
*root identity* capture: the absorbed-latent selection is a deterministic
function of (query, resident fp8 latent bytes), so a cold request and a warm
request that reuse the SAME radix-cached prefix slots read bit-identical latent
at those slots, hence identical absorbed score, hence identical top-k. There are
no materialized label signatures to hash (the TokenLabelTable is deleted), so
the latent bytes ARE the identity.

When the flag is on, :func:`maybe_dump_latent_capture` is called once per local
DS layer from ``deepseek_v2._select_topk_indices`` on the EAGER decode path
(``--disable-cuda-graph``). For each request in the batch it SHAs the resident
fp8-latent bytes (the [lora fp8 | nblk fp32 scales] unpack the absorbed kernel
consumes) of every prompt-prefix slot ``req_to_token[req_pool_indices[b],
:seq_len[b]]``, and appends one record per (tp_rank, request, layer) under the
dump dir. The offline comparator assembles ``per_layer_per_token_latent_sha``
(per layer, the per-cached-prefix-token latent SHA list) for
``compare_tablefree_selection``.

Capture-safety: a host copy + SHA is illegal during CUDA-graph capture, and the
prompt-prefix slots only fully exist on the eager forward; the dump no-ops while
a graph is capturing. The dump dir defaults under ``os.getcwd()`` (the worker's
CWD is the repo root) — ``/dev/shm`` is NOT shared across sandboxed processes —
mirroring the selection-capture / recall-oracle sink convention.
"""

from __future__ import annotations

import hashlib
import os
import threading

import torch

_DUMP_DIR_ENV = "SGLANG_DS_LATENT_CAPTURE_DIR"

_lock = threading.Lock()


def latent_capture_dir() -> str:
    return os.environ.get(_DUMP_DIR_ENV) or os.path.join(
        os.getcwd(), ".sglang_ds_latentcap"
    )


def _sha256_bytes(buf: bytes) -> str:
    return hashlib.sha256(buf).hexdigest()


def _resolve_tp_rank() -> int:
    try:
        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
        )

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001
        return 0


def maybe_dump_latent_capture(
    *,
    forward_batch,
    latent_fp8: torch.Tensor,
    latent_scales: torch.Tensor,
    req_to_token: torch.Tensor,
    layer_id: int,
) -> None:
    """Dump per-request, per-prompt-slot latent SHA for one DS layer.

    No-op unless the forward is a decode step (the prompt prefix is resident),
    a graph is not being captured, and the latent / metadata are present. One
    file per (tp_rank, request, layer); the request id is the physical
    req_pool_indices row so cold and warm passes that reuse the same row align.
    """
    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None or not forward_mode.is_decode():
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    if latent_fp8 is None or latent_scales is None or req_to_token is None:
        return

    tp_rank = _resolve_tp_rank()

    rpi = getattr(forward_batch, "req_pool_indices", None)
    seq_lens = getattr(forward_batch, "seq_lens", None)
    if rpi is None or seq_lens is None:
        return
    bs = int(forward_batch.batch_size)
    rpi_c = rpi[:bs].detach().to("cpu").tolist()
    sl_c = seq_lens[:bs].detach().to("cpu").tolist()

    # fp8 latent bytes as uint8 (dtype-agnostic, byte-exact) + the per-128-block
    # scales as fp32 bytes — the absorbed kernel scores `fp8 * scales`, so a slot
    # that quantized to equal fp8 bytes under a different scale must NOT collide.
    fp8_u8 = latent_fp8.view(torch.uint8)
    scale_u8 = latent_scales.view(torch.uint8)

    dump_dir = latent_capture_dir()
    os.makedirs(dump_dir, exist_ok=True)

    for b in range(bs):
        prompt_len = int(sl_c[b])
        if prompt_len <= 1:
            continue
        # The current decode token's slot is the last position; the prompt
        # prefix is [0, prompt_len-1). Capture the whole resident sequence; the
        # comparator only inspects the first `cached_tokens` of it.
        row = int(rpi_c[b])
        slots = req_to_token[row, :prompt_len].long().to(fp8_u8.device)
        fp8_rows = fp8_u8[slots].detach().to("cpu").contiguous()
        scale_rows = scale_u8[slots].detach().to("cpu").contiguous()
        per_token_latent_sha = [
            _sha256_bytes(
                fp8_rows[t].numpy().tobytes() + scale_rows[t].numpy().tobytes()
            )
            for t in range(prompt_len)
        ]
        record = {
            "tp_rank": int(tp_rank),
            "req_pool_index": row,
            "layer_id": int(layer_id),
            "prompt_len": prompt_len,
            "per_token_latent_sha": per_token_latent_sha,
        }
        fname = f"rank{int(tp_rank)}_req{row:04d}_layer{int(layer_id):03d}.pt"
        with _lock:
            torch.save(record, os.path.join(dump_dir, fname))
