"""Per-decode-step dump of the DS selection mirrors.

Config-borne diagnostic (``selection_capture`` in the DS config JSON — env vars
do not reach the TP worker subprocesses). When the flag is on, the graph state
carries per-layer mirrors of ``(selected_indices, valid_lengths)`` that the
selector fills with a captured device copy, so CUDA-graph replay keeps them
current. After each decode forward the model runner calls
:func:`maybe_dump_selection_capture`, which reads the mirrors to host and
writes one file per ``(tp_rank, decode_step)`` under the shared dump directory.
Every TP rank dumps, so cross-rank selection agreement can be verified
bit-exactly offline.

The dump directory must live on a mount the driver and every TP worker resolve
identically. The repository working tree is that mount (the worker's CWD is the
repo root) — ``/dev/shm`` is NOT shared across sandboxed processes — so the
default anchors at ``os.getcwd()``, mirroring the recall-oracle sink
convention. ``SGLANG_DS_SELECTION_CAPTURE_DIR`` overrides (driver-side / test
use; worker-side gating stays config-borne).
"""

from __future__ import annotations

import os
import threading

import torch

_DUMP_DIR_ENV = "SGLANG_DS_SELECTION_CAPTURE_DIR"

_step_lock = threading.Lock()
_step_counter = 0


def selection_capture_dir() -> str:
    return os.environ.get(_DUMP_DIR_ENV) or os.path.join(
        os.getcwd(), ".sglang_ds_selcap"
    )


def reset_step_counter() -> None:
    """Reset the per-process decode-step counter (unit tests only)."""
    global _step_counter
    with _step_lock:
        _step_counter = 0


def _resolve_graph_state(forward_batch, attn_backend):
    gs = getattr(forward_batch, "ds_graph_state", None)
    if gs is not None:
        return gs
    backend = attn_backend
    # TBO wrapper exposes the real backend as `.primary`.
    primary = getattr(backend, "primary", None)
    if primary is not None:
        backend = primary
    fm = getattr(backend, "forward_metadata", None)
    if fm is None:
        return None
    return getattr(fm, "ds_graph_state", None)


def maybe_dump_selection_capture(forward_batch, attn_backend, tp_rank: int) -> None:
    """Dump this decode step's per-layer selection mirrors for one rank.

    No-op unless the forward is a decode step and the capture mirrors exist
    (i.e. the config-borne ``selection_capture`` flag sized them). Runs on the
    host after the forward returns — never inside graph capture.
    """
    global _step_counter

    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None or not forward_mode.is_decode():
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    gs = _resolve_graph_state(forward_batch, attn_backend)
    if gs is None or getattr(gs, "capture_indices", None) is None:
        return

    bs = int(forward_batch.batch_size)
    seq_lens = getattr(forward_batch, "seq_lens", None)
    seq_lens_list = (
        seq_lens[:bs].detach().to("cpu").tolist() if seq_lens is not None else None
    )
    # Bucket identity: which captured variant served this step, at what
    # allocated row count and selector width. `graph_key` comes from the
    # backend's pre-replay stamp (None == eager path, whose graph state is
    # freshly allocated per forward and never stamped).
    graph_key = getattr(gs, "last_replay_graph_key", None)
    record = {
        "bs": bs,
        "seq_lens": seq_lens_list,
        # [num_layers, bs, max_top_k] int32 / [num_layers, bs] int32
        "indices": gs.capture_indices[:, :bs].detach().to("cpu").clone(),
        "lengths": gs.capture_lengths[:, :bs].detach().to("cpu").clone(),
        "raw_bs": bs,
        "padded_bs": int(gs.capture_indices.shape[1]),
        "selector_width": int(getattr(gs, "max_seq_len", 0) or 0),
        "graph_key": graph_key,
        "replay_path": graph_key is not None,
        "max_real_seq_len": (max(seq_lens_list) if seq_lens_list else None),
    }
    with _step_lock:
        step = _step_counter
        _step_counter += 1
    record["step"] = step

    dump_dir = selection_capture_dir()
    os.makedirs(dump_dir, exist_ok=True)
    torch.save(
        record, os.path.join(dump_dir, f"rank{int(tp_rank)}_step{step:05d}.pt")
    )
