"""R24 rootcause — correlate cold/warm SELECTION FLIPS against latent BYTE diffs.

Boots-agnostic (server booted radix-ON, eager, BOTH latent_capture AND
selection_capture on by rootcause.sh). Issues:
  * a COLD request with a long IDENTICAL prompt P (~3800 tokens) -> populates radix,
  * a WARM request with the IDENTICAL prompt P -> radix cache hit (cached_tokens C>0).

Captures, per pass, for tp_rank 0 (and one spot-check rank):
  * latent_capture: per-(rank,req,layer) per-LOGICAL-position fp8-latent SHA
    (req_to_token[row, :prompt_len], so index t == logical position t).
  * selection_capture: per-(rank,step) per-layer SELECTED logical indices
    (indices[layer,row,:valid_len], ascending logical positions in the top-k).

The DECISIVE correlation (per layer):
  flipped = symmetric difference of cold-selected-set vs warm-selected-set
            (logical positions in cold's top-k but not warm's, and vice-versa).
  For each flipped logical position p, look up cold_latent_sha[p] vs warm_latent_sha[p]:
    * SHA differ  => cause (a): the radix-cached fp8 latent BYTES at p differ.
    * SHA identical => cause (b): bytes identical, yet the borderline selection flips
      (near-tie at the hard top-k cutoff / score-reduce non-determinism).

Also reports, over the cached prefix [0,C): #positions where cold!=warm latent SHA
(and whether boundary-clustered vs spread); and over the tail [C,len): #differing.

Writes a SMALL summary JSON + a per-position comparison table (gz). It does NOT keep
the raw multi-MB .pt dumps — rootcause.sh deletes them after this runs.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import shutil
import sys
import time

import requests
import torch

PORT = int(os.environ.get("PORT", "30000"))
BASE = f"http://127.0.0.1:{PORT}"
LATENT_DIR = os.environ["SGLANG_DS_LATENT_CAPTURE_DIR"]
SEL_DIR = os.environ["SGLANG_DS_SELECTION_CAPTURE_DIR"]


def _gen(prompt: str, max_new_tokens: int = 4):
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


def _clear(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)


def _move(src, dst):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)


# ---- latent capture: per-rank, per-layer, per-LOGICAL-position SHA -----------
def _latent_per_layer(capdir, rank):
    """Return dict layer_id -> list[sha] (index == logical position), and prompt_len."""
    files = sorted(glob.glob(os.path.join(capdir, f"rank{rank}_req*_layer*.pt")))
    by_layer = {}
    prompt_len = 0
    for f in files:
        rec = torch.load(f, weights_only=False)
        by_layer[int(rec["layer_id"])] = rec["per_token_latent_sha"]
        prompt_len = max(prompt_len, int(rec["prompt_len"]))
    return by_layer, prompt_len


# ---- selection capture: first decode step, per-layer selected logical idxs ----
def _sel_first_step_file(capdir, rank):
    files = glob.glob(os.path.join(capdir, f"rank{rank}_step*.pt"))
    if not files:
        return None
    files.sort(key=lambda f: int(os.path.basename(f).split("_step", 1)[1].split(".pt", 1)[0]))
    return files[0]


def _sel_per_layer(capdir, rank):
    """Return dict layer_id -> sorted list[selected logical idx], valid_lengths, step."""
    f = _sel_first_step_file(capdir, rank)
    if f is None:
        return None, None, None
    rec = torch.load(f, weights_only=False)
    idx = rec["indices"]  # [L, bs, max_top_k]
    lens = rec["lengths"]  # [L, bs]
    L = int(idx.shape[0])
    row = 0
    per_layer = {}
    vlens = {}
    for layer_id in range(L):
        vl = int(lens[layer_id, row].item())
        sel = idx[layer_id, row, :vl].to(torch.int64).tolist()
        per_layer[layer_id] = sel
        vlens[layer_id] = vl
    step = int(os.path.basename(f).split("_step", 1)[1].split(".pt", 1)[0])
    return per_layer, vlens, step


def _classify_position(p, C, prompt_len):
    if p < C:
        return "cached_prefix"
    return "tail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-tokens", type=int, default=3800)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--spot-rank", type=int, default=4)
    args = ap.parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    base_sentence = (
        "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
    )
    n_sent = max(1, args.prefix_tokens // 12)
    long_prefix = "".join(f"[{i}] {base_sentence}" for i in range(n_sent))
    prompt = long_prefix + " Summarize the passage above in one sentence."

    result = {
        "probe": "R24_rootcause_selection_flip_vs_latent_byte_diff",
        "identical_prompt": True,
        "prompt_chars": len(prompt),
        "ranks": [args.rank, args.spot_rank],
        "runtime_base_commit": "3de57d32b",
    }

    # ---- COLD pass (populate radix) ----
    _clear(LATENT_DIR)
    _clear(SEL_DIR)
    cold_resp = _gen(prompt)
    cold_meta = cold_resp.get("meta_info", {})
    time.sleep(1.0)
    cold_lat = os.path.join(outdir, "cold_latent")
    cold_sel = os.path.join(outdir, "cold_sel")
    _move(LATENT_DIR, cold_lat)
    _move(SEL_DIR, cold_sel)

    # ---- WARM pass (IDENTICAL prompt -> cache hit) ----
    _clear(LATENT_DIR)
    _clear(SEL_DIR)
    warm_resp = _gen(prompt)
    warm_meta = warm_resp.get("meta_info", {})
    time.sleep(1.0)
    warm_lat = os.path.join(outdir, "warm_latent")
    warm_sel = os.path.join(outdir, "warm_sel")
    _move(LATENT_DIR, warm_lat)
    _move(SEL_DIR, warm_sel)

    result["cold_prompt_tokens"] = cold_meta.get("prompt_tokens")
    result["cold_cached_tokens"] = cold_meta.get("cached_tokens", 0)
    result["warm_prompt_tokens"] = warm_meta.get("prompt_tokens")
    C = int(warm_meta.get("cached_tokens", 0) or 0)
    result["warm_cached_tokens"] = C

    if C <= 0:
        result["status"] = "FAIL"
        result["reason"] = f"no positive radix hit (cached_tokens={C})"
        with open(os.path.join(outdir, "summary.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        print(json.dumps(result, indent=2))
        return 1

    per_rank_report = {}
    comparison_rows = []  # small per-position table for the flipped positions

    for rk in [args.rank, args.spot_rank]:
        cold_lat_pl, cold_plen = _latent_per_layer(cold_lat, rk)
        warm_lat_pl, warm_plen = _latent_per_layer(warm_lat, rk)
        cold_sel_pl, cold_vl, cold_step = _sel_per_layer(cold_sel, rk)
        warm_sel_pl, warm_vl, warm_step = _sel_per_layer(warm_sel, rk)

        rk_rep = {
            "cold_prompt_len_captured": cold_plen,
            "warm_prompt_len_captured": warm_plen,
            "cold_first_step": cold_step,
            "warm_first_step": warm_step,
            "n_latent_layers_cold": len(cold_lat_pl),
            "n_latent_layers_warm": len(warm_lat_pl),
            "n_sel_layers_cold": len(cold_sel_pl) if cold_sel_pl else 0,
            "n_sel_layers_warm": len(warm_sel_pl) if warm_sel_pl else 0,
        }

        # --- (A) latent byte comparison over cached prefix [0,C) and tail [C,len) ---
        # Use min over common layers; report per-position cold!=warm count, by domain.
        common_layers = sorted(set(cold_lat_pl) & set(warm_lat_pl))
        n_prefix_diff_positions = 0  # union over layers: positions w/ any layer SHA diff in [0,C)
        n_tail_diff_positions = 0
        prefix_diff_layer_count = 0  # total (layer,pos) prefix mismatches
        tail_diff_layer_count = 0
        prefix_diff_positions_set = set()
        tail_diff_positions_set = set()
        boundary_window = 64  # cache boundary cluster check (page_size)
        for layer_id in common_layers:
            c = cold_lat_pl[layer_id]
            w = warm_lat_pl[layer_id]
            n = min(len(c), len(w))
            for t in range(n):
                if c[t] != w[t]:
                    if t < C:
                        prefix_diff_layer_count += 1
                        prefix_diff_positions_set.add(t)
                    else:
                        tail_diff_layer_count += 1
                        tail_diff_positions_set.add(t)
        n_prefix_diff_positions = len(prefix_diff_positions_set)
        n_tail_diff_positions = len(tail_diff_positions_set)
        near_boundary = sorted(
            p for p in prefix_diff_positions_set if abs(p - C) <= boundary_window
        )
        rk_rep["latent_bytes"] = {
            "cached_tokens_C": C,
            "n_cached_prefix_positions_with_any_layer_sha_diff": n_prefix_diff_positions,
            "n_cached_prefix_layerpos_mismatches_total": prefix_diff_layer_count,
            "n_tail_positions_with_any_layer_sha_diff": n_tail_diff_positions,
            "n_tail_layerpos_mismatches_total": tail_diff_layer_count,
            "example_prefix_diff_positions": sorted(prefix_diff_positions_set)[:20],
            "example_tail_diff_positions": sorted(tail_diff_positions_set)[:20],
            "prefix_diff_positions_within_64_of_boundary_C": near_boundary[:20],
        }

        # --- (B) selection flips per layer + correlate against latent bytes ---
        flip_summary = []
        total_flips = 0
        layers_with_flip = 0
        flips_at_bytediff = 0
        flips_at_byteidentical = 0
        flips_outside_prefix = 0  # flipped logical pos >= C (shouldn't happen if all selected in prefix)
        if cold_sel_pl and warm_sel_pl:
            sel_layers = sorted(set(cold_sel_pl) & set(warm_sel_pl))
            for layer_id in sel_layers:
                cset = set(cold_sel_pl[layer_id])
                wset = set(warm_sel_pl[layer_id])
                cold_only = sorted(cset - wset)  # in cold top-k, not warm
                warm_only = sorted(wset - cset)  # in warm top-k, not cold
                flipped = sorted(set(cold_only) | set(warm_only))
                if not flipped:
                    continue
                layers_with_flip += 1
                total_flips += len(flipped)
                # correlate each flipped logical position against latent bytes
                lat_c = cold_lat_pl.get(layer_id)
                lat_w = warm_lat_pl.get(layer_id)
                per_pos = []
                for p in flipped:
                    side = "cold_only" if p in cset else "warm_only"
                    domain = _classify_position(p, C, cold_plen)
                    csha = lat_c[p] if (lat_c is not None and p < len(lat_c)) else None
                    wsha = lat_w[p] if (lat_w is not None and p < len(lat_w)) else None
                    if csha is None or wsha is None:
                        byte_state = "no_latent_sha"
                    elif csha == wsha:
                        byte_state = "bytes_identical"
                        flips_at_byteidentical += 1
                    else:
                        byte_state = "bytes_differ"
                        flips_at_bytediff += 1
                    if domain != "cached_prefix":
                        flips_outside_prefix += 1
                    rowrec = {
                        "rank": rk,
                        "layer": layer_id,
                        "logical_pos": p,
                        "side": side,
                        "domain": domain,
                        "byte_state": byte_state,
                        "cold_sha8": (csha[:8] if csha else None),
                        "warm_sha8": (wsha[:8] if wsha else None),
                    }
                    per_pos.append(rowrec)
                    comparison_rows.append(rowrec)
                flip_summary.append({
                    "layer": layer_id,
                    "n_flipped": len(flipped),
                    "cold_only": cold_only[:10],
                    "warm_only": warm_only[:10],
                    "per_pos": per_pos[:12],
                })
        rk_rep["selection_flips"] = {
            "n_sel_layers_compared": len(set(cold_sel_pl) & set(warm_sel_pl)) if (cold_sel_pl and warm_sel_pl) else 0,
            "n_layers_with_flip": layers_with_flip,
            "total_flipped_positions": total_flips,
            "flips_where_latent_bytes_DIFFER": flips_at_bytediff,
            "flips_where_latent_bytes_IDENTICAL": flips_at_byteidentical,
            "flipped_positions_outside_cached_prefix": flips_outside_prefix,
            "first_layers_with_flip_detail": flip_summary[:6],
        }
        per_rank_report[str(rk)] = rk_rep

    result["per_rank"] = per_rank_report

    # ---- CONCLUSION (rank0, the decisive rank) ----
    r0 = per_rank_report.get(str(args.rank), {})
    sf = r0.get("selection_flips", {})
    lb = r0.get("latent_bytes", {})
    diff = sf.get("flips_where_latent_bytes_DIFFER", 0)
    ident = sf.get("flips_where_latent_bytes_IDENTICAL", 0)
    if (diff + ident) == 0:
        conclusion = "NO_FLIPS"
    elif diff == 0 and ident > 0 and lb.get("n_cached_prefix_positions_with_any_layer_sha_diff", 0) == 0:
        conclusion = "cause_b_identical_bytes_near_tie"
    elif diff > 0 and ident == 0:
        conclusion = "cause_a_byte_difference"
    else:
        conclusion = "mixed"
    result["conclusion"] = conclusion
    result["conclusion_detail"] = {
        "rank0_flips_at_byte_DIFFER": diff,
        "rank0_flips_at_byte_IDENTICAL": ident,
        "rank0_cached_prefix_positions_with_latent_byte_diff": lb.get(
            "n_cached_prefix_positions_with_any_layer_sha_diff"
        ),
        "rank0_tail_positions_with_latent_byte_diff": lb.get(
            "n_tail_positions_with_any_layer_sha_diff"
        ),
    }
    result["status"] = "DONE"

    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    # small comparison table (gzip JSONL)
    with gzip.open(os.path.join(outdir, "flip_position_table.jsonl.gz"), "wt") as fh:
        for rrow in comparison_rows:
            fh.write(json.dumps(rrow) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
