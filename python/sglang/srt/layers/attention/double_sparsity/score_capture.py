"""Per-request dump of the absorbed fp32 SCORE row fed to top-k.

Config-borne diagnostic (``score_capture`` in the DS config JSON — env vars do
not reach the TP worker subprocesses). The companion of ``latent_capture``: where
latent_capture SHAs the resident fp8-latent *input* bytes, score_capture dumps the
absorbed *score* the selection top-k actually consumes — i.e. the post-reduce,
post-per-request-mask score row ``[bs, max_seq_len]`` (logical-position indexed,
the same logical domain as ``selection_capture`` indices).

This is the DEC-9 Q2 instrument: at the logical positions that FLIP between a
cold and a warm decode (cold's top-2048 XOR warm's), compare the cold-vs-warm
score. Bit-identical scores at flips => the membership change is a pure top-k
tie-break at exact ties (a deterministic tie-break is a viable DS-side fix).
Epsilon-different scores => the membership genuinely changed (a tie-break will
not fix it; bit-identical scores are required).

Capture-safety: a host copy is illegal during CUDA-graph capture, and the score
row only exists fully on the eager decode forward; the dump no-ops while a graph
is capturing and on non-decode forwards. Off by default — when off the only hot
path cost is a single ``getattr`` on the config at the call site. The dump dir
defaults under ``os.getcwd()`` (the worker CWD is the repo root) — ``/dev/shm`` is
NOT shared across sandboxed processes — mirroring the latent-capture /
selection-capture sink convention.
"""

from __future__ import annotations

import os
import threading

import torch

_DUMP_DIR_ENV = "SGLANG_DS_SCORE_CAPTURE_DIR"

_lock = threading.Lock()


def score_capture_dir() -> str:
    return os.environ.get(_DUMP_DIR_ENV) or os.path.join(
        os.getcwd(), ".sglang_ds_scorecap"
    )


def _resolve_tp_rank() -> int:
    try:
        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
        )

        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001
        return 0


def maybe_dump_score_capture(
    *,
    scores: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    layer_id: int,
    pre_reduce_scores: torch.Tensor | None = None,
) -> None:
    """Dump the per-request absorbed score row (logical-position indexed).

    ``scores`` is the ``[bs, max_seq_len]`` post-reduce / post-mask score tensor
    the selection top-k consumes; column ``t`` is logical position ``t`` (the
    same logical domain as the selection-capture indices). One file per
    ``(tp_rank, req_pool_index, layer)``; the request id is the physical
    req_pool_indices row so cold and warm passes that reuse the same row align.

    ``pre_reduce_scores`` (DEC-9 R24-scorepath exp-1): the per-rank PRE-reduce
    absorbed score row, captured right after the absorbed paged kernel and
    BEFORE the cross-TP all-reduce. When provided, it is stored alongside the
    post-reduce row in the same record so the comparator can ask whether the
    cold/warm score difference is already present pre-reduce (upstream in v_h /
    the per-rank score) or only appears post-reduce (the bf16 reduce introduces
    it). The pre-reduce row is the raw kernel output before the per-request mask
    (the mask is applied to ``topk_scores`` after the reduce); dead/invalid
    positions are -inf as the kernel stores them.

    No-op while a graph is capturing (host copy illegal there). The caller (the
    eager-only / capture-guarded site in ``retrieve_topk_graph_safe``) already
    gates on capture + the config flag; this is a defensive second guard.
    """
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    if scores is None or req_pool_indices is None or seq_lens is None:
        return

    tp_rank = _resolve_tp_rank()
    bs = int(scores.shape[0])
    rpi_c = req_pool_indices[:bs].detach().to("cpu").tolist()
    sl_c = seq_lens[:bs].detach().to("cpu").tolist()

    dump_dir = score_capture_dir()
    os.makedirs(dump_dir, exist_ok=True)

    for b in range(bs):
        seq_len = int(sl_c[b])
        if seq_len <= 1:
            continue
        row = int(rpi_c[b])
        # Capture the live logical window [0, seq_len) at fp32 (the top-k input
        # dtype is upcast to fp32 in-register; bf16->fp32 is exact, so the fp32
        # snapshot is the authoritative compared value). The comparator only
        # inspects the flipped logical positions.
        score_row = (
            scores[b, :seq_len].detach().to(torch.float32).to("cpu").contiguous()
        )
        record = {
            "tp_rank": int(tp_rank),
            "req_pool_index": row,
            "layer_id": int(layer_id),
            "seq_len": seq_len,
            "score_dtype": str(scores.dtype),
            # fp32 score row, column t == logical position t (POST-reduce)
            "scores": score_row,
        }
        if pre_reduce_scores is not None and b < int(pre_reduce_scores.shape[0]):
            # PRE-reduce per-rank score row (before the cross-TP all-reduce).
            # Same logical window/domain; fp32 snapshot (the kernel writes fp32).
            record["pre_reduce_scores"] = (
                pre_reduce_scores[b, :seq_len]
                .detach()
                .to(torch.float32)
                .to("cpu")
                .contiguous()
            )
            record["pre_reduce_score_dtype"] = str(pre_reduce_scores.dtype)
        fname = f"rank{int(tp_rank)}_req{row:04d}_layer{int(layer_id):03d}.pt"
        with _lock:
            torch.save(record, os.path.join(dump_dir, fname))
