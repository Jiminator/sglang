"""DEC-9 Q1 — DETERMINISM CONTROL (radix OFF, selection_capture only, no new code).

Boots DS radix-OFF (--disable-radix-cache), EAGER, selection_capture=true.
Sends the SAME prompt as TWO separate FRESH requests (no cache reuse possible —
radix is off). Compares the first-decode-step selected logical indices across
ALL ranks and ALL DS layers between the two fresh requests.

  * BIT-IDENTICAL  => DS selection is deterministic given inputs; the R24
    cold/warm flip is RADIX-SPECIFIC (cached-decode path differs from fresh).
  * DIFFER         => DS selection is run-to-run non-deterministic in general
    (radix-independent).

Selection_capture writes per-(rank,decode_step) files; the step counter is a
per-process monotonic global, so request 2's first decode step has a higher
step index than request 1's. We snapshot the dump dir after each request into
a per-request subdir and take each request's FIRST step file per rank.

Writes a SMALL summary JSON. Deletes the raw .pt snapshots after computing.
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

PORT = int(os.environ.get("PORT", "30000"))
BASE = f"http://127.0.0.1:{PORT}"
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


def _ranks_present(capdir):
    ranks = set()
    for f in glob.glob(os.path.join(capdir, "rank*_step*.pt")):
        rk = int(os.path.basename(f).split("rank", 1)[1].split("_step", 1)[0])
        ranks.add(rk)
    return sorted(ranks)


def _first_step_file(capdir, rank):
    files = glob.glob(os.path.join(capdir, f"rank{rank}_step*.pt"))
    if not files:
        return None
    files.sort(
        key=lambda f: int(
            os.path.basename(f).split("_step", 1)[1].split(".pt", 1)[0]
        )
    )
    return files[0]


def _sel_per_layer(capdir, rank):
    """layer_id -> sorted list[selected logical idx] for first decode step, + step."""
    f = _first_step_file(capdir, rank)
    if f is None:
        return None, None
    rec = torch.load(f, weights_only=False)
    idx = rec["indices"]  # [L, bs, max_top_k]
    lens = rec["lengths"]  # [L, bs]
    L = int(idx.shape[0])
    row = 0
    per_layer = {}
    for layer_id in range(L):
        vl = int(lens[layer_id, row].item())
        per_layer[layer_id] = idx[layer_id, row, :vl].to(torch.int64).tolist()
    step = int(os.path.basename(f).split("_step", 1)[1].split(".pt", 1)[0])
    return per_layer, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-tokens", type=int, default=3800)
    ap.add_argument("--outdir", required=True)
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
        "probe": "DEC9_Q1_determinism_control_two_fresh_radix_OFF",
        "identical_prompt": True,
        "prompt_chars": len(prompt),
    }

    # ---- FRESH request 1 ----
    _clear(SEL_DIR)
    r1 = _gen(prompt)
    m1 = r1.get("meta_info", {})
    time.sleep(1.0)
    sel1 = os.path.join(outdir, "fresh1_sel")
    _move(SEL_DIR, sel1)

    # ---- FRESH request 2 (identical prompt; radix OFF -> no cache reuse) ----
    _clear(SEL_DIR)
    r2 = _gen(prompt)
    m2 = r2.get("meta_info", {})
    time.sleep(1.0)
    sel2 = os.path.join(outdir, "fresh2_sel")
    _move(SEL_DIR, sel2)

    result["fresh1_prompt_tokens"] = m1.get("prompt_tokens")
    result["fresh1_cached_tokens"] = m1.get("cached_tokens", 0)
    result["fresh2_prompt_tokens"] = m2.get("prompt_tokens")
    result["fresh2_cached_tokens"] = m2.get("cached_tokens", 0)

    ranks = sorted(set(_ranks_present(sel1)) & set(_ranks_present(sel2)))
    result["ranks_compared"] = ranks

    per_rank = {}
    grand_total_layers = 0
    grand_identical_layers = 0
    grand_differ_layers = 0
    grand_total_flips = 0
    example_diffs = []
    for rk in ranks:
        pl1, step1 = _sel_per_layer(sel1, rk)
        pl2, step2 = _sel_per_layer(sel2, rk)
        if pl1 is None or pl2 is None:
            per_rank[str(rk)] = {"status": "missing_capture"}
            continue
        common = sorted(set(pl1) & set(pl2))
        identical_layers = 0
        differ_layers = 0
        total_flips = 0
        layer_diff_detail = []
        for layer_id in common:
            s1 = set(pl1[layer_id])
            s2 = set(pl2[layer_id])
            flipped = (s1 - s2) | (s2 - s1)
            if not flipped:
                identical_layers += 1
            else:
                differ_layers += 1
                total_flips += len(flipped)
                layer_diff_detail.append(
                    {
                        "layer": layer_id,
                        "n_flipped": len(flipped),
                        "len1": len(s1),
                        "len2": len(s2),
                        "fresh1_only": sorted(s1 - s2)[:8],
                        "fresh2_only": sorted(s2 - s1)[:8],
                    }
                )
        per_rank[str(rk)] = {
            "fresh1_first_step": step1,
            "fresh2_first_step": step2,
            "n_layers_compared": len(common),
            "n_layers_bit_identical": identical_layers,
            "n_layers_differ": differ_layers,
            "total_flipped_positions": total_flips,
            "layer_diff_detail": layer_diff_detail[:6],
        }
        grand_total_layers += len(common)
        grand_identical_layers += identical_layers
        grand_differ_layers += differ_layers
        grand_total_flips += total_flips
        if layer_diff_detail and len(example_diffs) < 4:
            example_diffs.extend(layer_diff_detail[: 4 - len(example_diffs)])

    result["per_rank"] = per_rank
    result["aggregate"] = {
        "n_ranks": len(ranks),
        "total_rank_layer_pairs_compared": grand_total_layers,
        "total_rank_layer_pairs_bit_identical": grand_identical_layers,
        "total_rank_layer_pairs_differ": grand_differ_layers,
        "total_flipped_positions": grand_total_flips,
    }
    if grand_total_layers == 0:
        result["verdict"] = "NO_DATA"
    elif grand_differ_layers == 0:
        result["verdict"] = (
            "BIT_IDENTICAL — two fresh requests select identical top-k on every "
            "rank+layer; DS selection deterministic given inputs; R24 flip is "
            "RADIX-SPECIFIC"
        )
    else:
        result["verdict"] = (
            "DIFFER — two fresh requests select different top-k on "
            f"{grand_differ_layers} rank+layer pairs; DS selection is run-to-run "
            "non-deterministic (radix-independent)"
        )
        result["example_diffs"] = example_diffs

    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
