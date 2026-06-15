"""P1 (CORRECTED) — cold/warm SELECTED-INDEX equivalence on a REAL radix hit.

Fixes the prior round's overclaims:
  * compares the SELECTED INDICES (per_layer_selected_indices), not just latent SHA;
  * uses the IDENTICAL prompt P for cold and warm (same query => the equality test
    is meaningful — identical latent alone does not prove identical top-k);
  * reads ALL 8 TP ranks (rank0..rank7) and requires every one present;
  * also asserts cross-rank bit-identity (rank0 selection == every other rank).

Mechanism: the config-borne `selection_capture` diagnostic dumps, per (tp_rank,
decode_step), the per-layer selected indices [num_layers, bs, max_top_k] +
valid_lengths [num_layers, bs] to <capdir>/rank{r}_step{s}.pt. We clear the dir
before each pass, run the request, then move the dir aside. The FIRST decode step
of each pass is the file with the smallest step index in that pass's dir.

For each rank we build per_layer_selected_indices = the top `valid_length` selected
positions per DS layer at the first decode step, then run compare_tablefree_selection
on per_layer_selected_indices, cold vs warm, per rank. cached_tokens (the compare
length) is the warm meta_info cached_tokens — but because the prompt is IDENTICAL,
the whole selection lives in the cached-prefix domain, so the comparator compares
min(cached_tokens, valid_length) prefix positions of the (ascending) selection.

A divergence at ANY rank/layer => FAIL. Cross-rank divergence => FAIL.
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
    "SGLANG_DS_SELECTION_CAPTURE_DIR",
    "/sgl-workspace/sglang/.sglang_ds_selcap",
)


def _gen(prompt: str, max_new_tokens: int = 4):
    """Force >=1 decode step (ignore_eos) so the selection-capture decode hook fires."""
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


def _ranks_present(capdir: str):
    """Discover the tp ranks that dumped in this pass dir."""
    ranks = set()
    for f in glob.glob(os.path.join(capdir, "rank*_step*.pt")):
        base = os.path.basename(f)
        # rank{R}_step{S}.pt
        r = int(base.split("rank", 1)[1].split("_", 1)[0])
        ranks.add(r)
    return sorted(ranks)


def _first_step_file(capdir: str, rank: int):
    """File for rank's FIRST decode step in this pass (smallest step index)."""
    files = sorted(glob.glob(os.path.join(capdir, f"rank{rank}_step*.pt")))
    if not files:
        return None
    # Sort by the numeric step embedded in the name.
    def _step(f):
        return int(os.path.basename(f).split("_step", 1)[1].split(".pt", 1)[0])

    files.sort(key=_step)
    return files[0]


def _per_layer_selected(capdir: str, rank: int):
    """Build per_layer_selected_indices (List[List[int]]) for rank's first decode step.

    Reads the record's `indices` [num_layers, bs, max_top_k] and `lengths`
    [num_layers, bs]. Single request per pass => bs row 0. For each layer, the
    selected positions are the first valid_length entries (already ascending,
    -1 padded). Returns (per_layer, n_layers, valid_lengths_list, bs, step).
    """
    f = _first_step_file(capdir, rank)
    if f is None:
        return None, 0, [], 0, None
    rec = torch.load(f, weights_only=False)
    idx = rec["indices"]  # [L, bs, max_top_k]
    lens = rec["lengths"]  # [L, bs]
    bs = int(idx.shape[1])
    L = int(idx.shape[0])
    row = 0  # single request per pass
    per_layer = []
    vlens = []
    for layer_id in range(L):
        vl = int(lens[layer_id, row].item())
        vlens.append(vl)
        sel = idx[layer_id, row, :vl].to(torch.int64).tolist()
        per_layer.append(sel)
    step = int(os.path.basename(f).split("_step", 1)[1].split(".pt", 1)[0])
    return per_layer, L, vlens, bs, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-tokens", type=int, default=2400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-ranks", type=int, default=8)
    args = ap.parse_args()
    outdir = os.path.dirname(args.out)

    # Long, deterministic prompt > 2000 tokens. IDENTICAL for cold and warm so
    # the query is identical and the selected-index equality test is meaningful.
    base_sentence = (
        "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
    )
    n_sent = max(1, args.prefix_tokens // 12)
    long_prefix = "".join(f"[{i}] {base_sentence}" for i in range(n_sent))
    prompt = long_prefix + " Summarize the passage above in one sentence."

    result = {
        "probe": "P1_cold_warm_SELECTION_equivalence_corrected",
        "capdir": CAPDIR,
        "prompt_chars": len(prompt),
        "identical_prompt": True,
        "expect_ranks": args.expect_ranks,
    }

    # --- COLD pass: populates the radix cache ---
    _clear_capdir()
    cold_resp = _gen(prompt)
    cold_meta = cold_resp.get("meta_info", {})
    time.sleep(1.0)
    cold_dir = os.path.join(outdir, "cold")
    _move_capdir(cold_dir)

    # --- WARM pass: IDENTICAL prompt -> must reuse the cached prefix ---
    _clear_capdir()
    warm_resp = _gen(prompt)
    warm_meta = warm_resp.get("meta_info", {})
    time.sleep(1.0)
    warm_dir = os.path.join(outdir, "warm")
    _move_capdir(warm_dir)

    result["cold_prompt_tokens"] = cold_meta.get("prompt_tokens")
    result["cold_cached_tokens"] = cold_meta.get("cached_tokens", 0)
    result["warm_prompt_tokens"] = warm_meta.get("prompt_tokens")
    warm_cached = warm_meta.get("cached_tokens", 0)
    result["warm_cached_tokens"] = warm_cached

    # POSITIVE hit gate: warm must reuse the prefix.
    if not warm_cached or warm_cached <= 0:
        result["status"] = "FAIL"
        result["reason"] = (
            f"no positive radix cache hit on warm request (cached_tokens={warm_cached}); "
            "probe meaningless without a real hit."
        )
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(json.dumps(result, indent=2))
        return 1

    cold_ranks = _ranks_present(cold_dir)
    warm_ranks = _ranks_present(warm_dir)
    result["cold_ranks_present"] = cold_ranks
    result["warm_ranks_present"] = warm_ranks

    expect = list(range(args.expect_ranks))
    if cold_ranks != expect or warm_ranks != expect:
        result["status"] = "FAIL"
        result["reason"] = (
            f"not all {args.expect_ranks} ranks present: cold={cold_ranks} warm={warm_ranks} "
            f"(expected {expect})"
        )
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(json.dumps(result, indent=2))
        return 1

    # Per-rank cold==warm selected-index comparison + cross-rank identity.
    per_rank = {}
    all_ok = True
    cross_rank_ok = True
    rank0_cold = None
    layers_per_rank = None
    for r in expect:
        cold_pl, cold_L, cold_vl, cold_bs, cold_step = _per_layer_selected(cold_dir, r)
        warm_pl, warm_L, warm_vl, warm_bs, warm_step = _per_layer_selected(warm_dir, r)
        if cold_pl is None or warm_pl is None:
            per_rank[r] = {"cold_eq_warm": False, "reason": "missing capture file"}
            all_ok = False
            continue
        if layers_per_rank is None:
            layers_per_rank = cold_L
        # Compare ONLY the cached-prefix domain. With the IDENTICAL prompt the
        # whole selection is over the cached prefix, but cap the comparator at
        # min(cached_tokens, valid_length) so we never claim coverage beyond the
        # proven hit or beyond the actual selected count.
        min_vl = min(min(cold_vl), min(warm_vl)) if cold_vl and warm_vl else 0
        cmp_len = min(int(warm_cached), min_vl)
        if cmp_len <= 0:
            per_rank[r] = {
                "cold_eq_warm": False,
                "reason": f"no comparable prefix (cached={warm_cached}, min_valid_len={min_vl})",
                "cold_step": cold_step,
                "warm_step": warm_step,
            }
            all_ok = False
            continue
        cmp = compare_tablefree_selection(
            cold={"per_layer_selected_indices": cold_pl},
            warm={"per_layer_selected_indices": warm_pl},
            cached_tokens=int(cmp_len),
        )
        ok = bool(cmp.get("ok"))
        per_rank[r] = {
            "cold_eq_warm": ok,
            "compare_len": int(cmp_len),
            "layers": cold_L,
            "cold_first_step": cold_step,
            "warm_first_step": warm_step,
            "cold_valid_len_range": [min(cold_vl), max(cold_vl)],
            "warm_valid_len_range": [min(warm_vl), max(warm_vl)],
            "comparator": cmp,
        }
        if not ok:
            all_ok = False
        # Cross-rank: rank0 cold selection must equal every other rank's cold.
        if r == 0:
            rank0_cold = cold_pl
        else:
            if rank0_cold is not None:
                xr = compare_tablefree_selection(
                    cold={"per_layer_selected_indices": rank0_cold},
                    warm={"per_layer_selected_indices": cold_pl},
                    cached_tokens=int(cmp_len),
                )
                per_rank[r]["matches_rank0"] = bool(xr.get("ok"))
                if not xr.get("ok"):
                    cross_rank_ok = False
                    per_rank[r]["cross_rank_reason"] = xr.get("reason")
            else:
                per_rank[r]["matches_rank0"] = None

    result["ranks_compared"] = expect
    result["layers_per_rank"] = layers_per_rank
    result["per_rank"] = per_rank
    result["all_cold_eq_warm"] = all_ok
    result["cross_rank_identical"] = cross_rank_ok
    result["all_ok"] = bool(all_ok and cross_rank_ok)
    result["status"] = "PASS" if result["all_ok"] else "FAIL"
    if not result["all_ok"]:
        result["reason"] = (
            f"all_cold_eq_warm={all_ok} cross_rank_identical={cross_rank_ok}"
        )

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
