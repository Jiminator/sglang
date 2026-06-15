"""Priority 1 — cold/warm SELECTION equivalence on a REAL radix cache hit.

Boots-agnostic driver (the server is booted radix-ON by p1_coldwarm.sh). Issues:
  * a COLD request with a long (>=2000 token) prompt, then
  * a WARM request whose prompt SHARES that long prefix + a short different suffix,
so the radix cache reuses the prefix. Confirms a POSITIVE hit from the warm
request's meta_info (cached_tokens > 0).

Identity captured: the resident fp8-latent SHA per cached-prefix slot, per DS
layer, via the config-borne `latent_capture` hook (latent_capture.py). The
absorbed-latent selection is a deterministic function of (query, resident fp8
latent bytes), so identical cached latent => identical absorbed score => identical
top-k. This is the table-free root identity the fixture proves.

Mechanics: the dump dir is cleared before each pass; the cold pass dumps to
<capdir>/.sglang_ds_latentcap (driver moves it to cold/), the warm pass to warm/.
Each (rank, layer) file carries per_token_latent_sha over the resident prefix.
The driver assembles, for tp_rank 0 (selection is bit-identical across ranks; we
prove the per-rank latent root here), per_layer_per_token_latent_sha sorted by
layer, then runs compare_tablefree_selection(cold, warm, cached_tokens=<hit>).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time

import requests
import torch

sys.path.insert(0, "/sgl-workspace/sglang")
from sglang.srt.layers.attention.double_sparsity.radix_fixture_capture import (  # noqa: E402
    compare_tablefree_selection,
)

PORT = int(os.environ.get("PORT", "30000"))
BASE = f"http://127.0.0.1:{PORT}"
CAPDIR = os.environ.get(
    "SGLANG_DS_LATENT_CAPTURE_DIR", "/sgl-workspace/sglang/.sglang_ds_latentcap"
)


def _gen(prompt: str, max_new_tokens: int = 4):
    """Force >=1 decode step (ignore_eos) so the latent-capture decode hook fires."""
    r = requests.post(
        f"{BASE}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def _clear_capdir():
    if os.path.isdir(CAPDIR):
        shutil.rmtree(CAPDIR)
    os.makedirs(CAPDIR, exist_ok=True)


def _move_capdir(dst: str):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.move(CAPDIR, dst)


def _assemble_per_layer(capdir: str, tp_rank: int = 0):
    """Build per_layer_per_token_latent_sha for one tp_rank from the dump files.

    Returns (per_layer_list, n_layers, max_prompt_len). The capture overwrites
    one file per (rank, req, layer) each decode step, so each file holds the
    last decode step's resident-prefix SHA. There is exactly one request per
    pass here; pick the single req row present.
    """
    files = sorted(glob.glob(os.path.join(capdir, f"rank{tp_rank}_req*_layer*.pt")))
    if not files:
        return None, 0, 0
    by_layer = {}
    req_rows = set()
    for f in files:
        rec = torch.load(f, weights_only=False)
        req_rows.add(rec["req_pool_index"])
        by_layer[rec["layer_id"]] = rec["per_token_latent_sha"]
    # Single request per pass.
    layers = sorted(by_layer)
    per_layer = [by_layer[l] for l in layers]
    max_len = max((len(v) for v in per_layer), default=0)
    return per_layer, len(layers), max_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-tokens", type=int, default=2400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-file", default=None)
    args = ap.parse_args()

    # Build a long, deterministic prompt whose token length comfortably exceeds
    # 2000. Use a repeated distinctive sentence; a short different suffix for warm.
    base_sentence = (
        "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
    )
    # ~14 tokens/sentence -> ~200 sentences for ~2800 tokens; oversize to be safe.
    n_sent = max(1, args.prefix_tokens // 12)
    long_prefix = "".join(f"[{i}] {base_sentence}" for i in range(n_sent))
    cold_prompt = long_prefix + " COLD-TAIL question: summarize the above."
    warm_prompt = long_prefix + " WARM-TAIL different question: list three facts."

    result = {
        "probe": "P1_cold_warm_selection_equivalence",
        "capdir": CAPDIR,
        "prefix_request_chars": len(long_prefix),
    }

    # --- COLD pass ---
    _clear_capdir()
    cold_resp = _gen(cold_prompt)
    cold_meta = cold_resp.get("meta_info", {})
    cold_prompt_tokens = cold_meta.get("prompt_tokens")
    cold_cached = cold_meta.get("cached_tokens", 0)
    time.sleep(1.0)
    _move_capdir(os.path.join(os.path.dirname(args.out), "cold"))
    cold_dir = os.path.join(os.path.dirname(args.out), "cold")

    # --- WARM pass (shares the long prefix) ---
    _clear_capdir()
    warm_resp = _gen(warm_prompt)
    warm_meta = warm_resp.get("meta_info", {})
    warm_prompt_tokens = warm_meta.get("prompt_tokens")
    warm_cached = warm_meta.get("cached_tokens", 0)
    time.sleep(1.0)
    _move_capdir(os.path.join(os.path.dirname(args.out), "warm"))
    warm_dir = os.path.join(os.path.dirname(args.out), "warm")

    result["cold_prompt_tokens"] = cold_prompt_tokens
    result["cold_cached_tokens"] = cold_cached
    result["warm_prompt_tokens"] = warm_prompt_tokens
    result["warm_cached_tokens"] = warm_cached

    # POSITIVE hit gate: the warm request must reuse the prefix.
    if not warm_cached or warm_cached <= 0:
        result["status"] = "FAIL"
        result["reason"] = (
            f"no positive radix cache hit on warm request "
            f"(cached_tokens={warm_cached}); probe is meaningless without a hit."
        )
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(json.dumps(result, indent=2))
        return 1

    cold_pl, cold_nl, cold_ml = _assemble_per_layer(cold_dir)
    warm_pl, warm_nl, warm_ml = _assemble_per_layer(warm_dir)
    result["cold_capture_layers"] = cold_nl
    result["warm_capture_layers"] = warm_nl
    result["cold_capture_max_prefix_len"] = cold_ml
    result["warm_capture_max_prefix_len"] = warm_ml

    if cold_pl is None or warm_pl is None:
        result["status"] = "FAIL"
        result["reason"] = "latent_capture produced no dump files for one/both passes"
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(json.dumps(result, indent=2))
        return 1

    # The radix cache reused `warm_cached` prefix tokens. Compare ONLY that
    # cached prefix. The comparator fail-closes on short/empty/divergent.
    cmp = compare_tablefree_selection(
        cold={"per_layer_per_token_latent_sha": cold_pl},
        warm={"per_layer_per_token_latent_sha": warm_pl},
        cached_tokens=int(warm_cached),
    )
    result["comparator"] = cmp
    result["status"] = "PASS" if cmp.get("ok") else "FAIL"

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if cmp.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
