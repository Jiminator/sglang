"""DEC-9 Q2 — SCORE TRACE (fix-feasibility). Radix-ON, score_capture +
selection_capture. COLD (fresh, populates radix) then WARM (IDENTICAL prompt,
cache hit). At the logical positions that FLIP in the selection (cold top-2048
XOR warm top-2048, per layer), compare cold-vs-warm absorbed SCORES.

  * scores BIT-IDENTICAL at flips => flip is pure top-k tie-break at exact ties
    => a deterministic tie-break (by logical index) is a viable DS-side fix.
  * scores DIFFER (epsilon) => top-k membership genuinely changed => a tie-break
    will NOT fix it; bit-identical scores are required. We report the magnitude
    of the score delta + whether the flipped positions sit at the top-k cutoff
    boundary (the 2048th-vs-2049th score gap).

Cutoff-boundary analysis: for each layer we take the per-position score row of
EACH pass, find the value of the kth-largest score (k = valid_len, the cutoff),
and report (a) the cold/warm score delta at each flipped position and (b) the
flipped position's score relative to the cutoff (how close to the membership
boundary). A flip with bit-identical scores AND those scores exactly equal to
the cutoff is a tie; a flip with a non-zero delta that straddles the cutoff is a
genuine membership change.

Writes a SMALL summary JSON + a per-flip table (gz). Deletes raw .pt after.
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
SCORE_DIR = os.environ["SGLANG_DS_SCORE_CAPTURE_DIR"]
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


# ---- score capture: per-rank, per-layer fp32 score row (col t == logical pos) -
def _score_per_layer(capdir, rank):
    """layer_id -> (score_tensor[seq_len] fp32, seq_len, dtype_str)."""
    files = sorted(glob.glob(os.path.join(capdir, f"rank{rank}_req*_layer*.pt")))
    by_layer = {}
    for f in files:
        rec = torch.load(f, weights_only=False)
        by_layer[int(rec["layer_id"])] = (
            rec["scores"],
            int(rec["seq_len"]),
            rec.get("score_dtype"),
        )
    return by_layer


# ---- selection capture: first decode step, per-layer selected logical idxs ----
def _sel_first_step_file(capdir, rank):
    files = glob.glob(os.path.join(capdir, f"rank{rank}_step*.pt"))
    if not files:
        return None
    files.sort(
        key=lambda f: int(os.path.basename(f).split("_step", 1)[1].split(".pt", 1)[0])
    )
    return files[0]


def _sel_per_layer(capdir, rank):
    f = _sel_first_step_file(capdir, rank)
    if f is None:
        return None, None
    rec = torch.load(f, weights_only=False)
    idx = rec["indices"]
    lens = rec["lengths"]
    L = int(idx.shape[0])
    row = 0
    per_layer = {}
    vlens = {}
    for layer_id in range(L):
        vl = int(lens[layer_id, row].item())
        per_layer[layer_id] = idx[layer_id, row, :vl].to(torch.int64).tolist()
        vlens[layer_id] = vl
    return per_layer, vlens


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
        "probe": "DEC9_Q2_score_trace_cold_vs_warm_at_flips",
        "identical_prompt": True,
        "prompt_chars": len(prompt),
        "ranks": [args.rank, args.spot_rank],
    }

    # ---- COLD pass (populate radix) ----
    _clear(SCORE_DIR)
    _clear(SEL_DIR)
    cold_resp = _gen(prompt)
    cold_meta = cold_resp.get("meta_info", {})
    time.sleep(1.0)
    cold_score = os.path.join(outdir, "cold_score")
    cold_sel = os.path.join(outdir, "cold_sel")
    _move(SCORE_DIR, cold_score)
    _move(SEL_DIR, cold_sel)

    # ---- WARM pass (IDENTICAL prompt -> cache hit) ----
    _clear(SCORE_DIR)
    _clear(SEL_DIR)
    warm_resp = _gen(prompt)
    warm_meta = warm_resp.get("meta_info", {})
    time.sleep(1.0)
    warm_score = os.path.join(outdir, "warm_score")
    warm_sel = os.path.join(outdir, "warm_sel")
    _move(SCORE_DIR, warm_score)
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
    flip_rows = []

    for rk in [args.rank, args.spot_rank]:
        cold_sc = _score_per_layer(cold_score, rk)
        warm_sc = _score_per_layer(warm_score, rk)
        cold_sel_pl, cold_vl = _sel_per_layer(cold_sel, rk)
        warm_sel_pl, warm_vl = _sel_per_layer(warm_sel, rk)

        rk_rep = {
            "n_score_layers_cold": len(cold_sc),
            "n_score_layers_warm": len(warm_sc),
            "n_sel_layers_cold": len(cold_sel_pl) if cold_sel_pl else 0,
            "n_sel_layers_warm": len(warm_sel_pl) if warm_sel_pl else 0,
            "score_dtype_cold": (
                next(iter(cold_sc.values()))[2] if cold_sc else None
            ),
            "score_dtype_warm": (
                next(iter(warm_sc.values()))[2] if warm_sc else None
            ),
        }

        total_flips = 0
        layers_with_flip = 0
        flips_in_prefix = 0
        flips_in_tail = 0
        # cached-prefix-domain flip score classification
        prefix_flips_bit_identical = 0
        prefix_flips_score_differ = 0
        prefix_flip_abs_deltas = []  # |cold_score - warm_score| at prefix flips
        prefix_flips_at_cutoff = 0  # flip score within tiny eps of the kth cutoff
        max_prefix_abs_delta = 0.0
        layer_detail = []

        if cold_sel_pl and warm_sel_pl:
            sel_layers = sorted(set(cold_sel_pl) & set(warm_sel_pl))
            for layer_id in sel_layers:
                cset = set(cold_sel_pl[layer_id])
                wset = set(warm_sel_pl[layer_id])
                flipped = sorted((cset - wset) | (wset - cset))
                if not flipped:
                    continue
                layers_with_flip += 1
                total_flips += len(flipped)

                c_entry = cold_sc.get(layer_id)
                w_entry = warm_sc.get(layer_id)
                if c_entry is None or w_entry is None:
                    continue
                c_scores, c_len, _ = c_entry
                w_scores, w_len, _ = w_entry
                # kth-largest (the membership cutoff) for each pass
                c_cut = (
                    torch.topk(c_scores, min(len(cold_sel_pl[layer_id]), c_scores.numel())).values[-1].item()
                    if cold_sel_pl[layer_id]
                    else float("-inf")
                )
                w_cut = (
                    torch.topk(w_scores, min(len(warm_sel_pl[layer_id]), w_scores.numel())).values[-1].item()
                    if warm_sel_pl[layer_id]
                    else float("-inf")
                )

                per_pos = []
                for p in flipped:
                    in_prefix = p < C
                    if in_prefix:
                        flips_in_prefix += 1
                    else:
                        flips_in_tail += 1
                    cs = (
                        float(c_scores[p].item())
                        if p < c_scores.numel()
                        else None
                    )
                    ws = (
                        float(w_scores[p].item())
                        if p < w_scores.numel()
                        else None
                    )
                    if cs is None or ws is None:
                        state = "no_score"
                        delta = None
                    else:
                        # bit-identical compare on the fp32 score bits
                        bit_identical = (
                            c_scores[p].view(torch.int32).item()
                            == w_scores[p].view(torch.int32).item()
                        )
                        delta = abs(cs - ws)
                        state = "bit_identical" if bit_identical else "score_differ"
                        if in_prefix:
                            if bit_identical:
                                prefix_flips_bit_identical += 1
                            else:
                                prefix_flips_score_differ += 1
                                prefix_flip_abs_deltas.append(delta)
                                max_prefix_abs_delta = max(
                                    max_prefix_abs_delta, delta
                                )
                            # at-cutoff: is the flip score essentially equal to
                            # the relevant pass's kth cutoff? (tie at boundary)
                            side_cut = c_cut if p in cset else w_cut
                            side_score = cs if p in cset else ws
                            if abs(side_score - side_cut) <= 1e-12:
                                prefix_flips_at_cutoff += 1
                    rowrec = {
                        "rank": rk,
                        "layer": layer_id,
                        "logical_pos": p,
                        "domain": "cached_prefix" if in_prefix else "tail",
                        "side": "cold_only" if p in cset else "warm_only",
                        "cold_score": cs,
                        "warm_score": ws,
                        "abs_delta": delta,
                        "cold_cutoff_kth": c_cut,
                        "warm_cutoff_kth": w_cut,
                        "state": state,
                    }
                    per_pos.append(rowrec)
                    flip_rows.append(rowrec)
                layer_detail.append(
                    {
                        "layer": layer_id,
                        "n_flipped": len(flipped),
                        "cold_valid_len": len(cold_sel_pl[layer_id]),
                        "warm_valid_len": len(warm_sel_pl[layer_id]),
                        "cold_cutoff_kth": c_cut,
                        "warm_cutoff_kth": w_cut,
                        "per_pos": per_pos[:10],
                    }
                )

        deltas_sorted = sorted(prefix_flip_abs_deltas)
        n_d = len(deltas_sorted)
        rk_rep["score_flips"] = {
            "n_sel_layers_compared": (
                len(set(cold_sel_pl) & set(warm_sel_pl))
                if (cold_sel_pl and warm_sel_pl)
                else 0
            ),
            "n_layers_with_flip": layers_with_flip,
            "total_flipped_positions": total_flips,
            "flips_in_cached_prefix": flips_in_prefix,
            "flips_in_tail": flips_in_tail,
            "cached_prefix_flips_score_BIT_IDENTICAL": prefix_flips_bit_identical,
            "cached_prefix_flips_score_DIFFER": prefix_flips_score_differ,
            "cached_prefix_flips_at_cutoff_boundary_tie": prefix_flips_at_cutoff,
            "max_abs_score_delta_prefix_flips": max_prefix_abs_delta,
            "median_abs_score_delta_prefix_flips": (
                deltas_sorted[n_d // 2] if n_d else 0.0
            ),
            "p99_abs_score_delta_prefix_flips": (
                deltas_sorted[min(n_d - 1, int(0.99 * n_d))] if n_d else 0.0
            ),
            "first_layers_with_flip_detail": layer_detail[:6],
        }
        per_rank_report[str(rk)] = rk_rep

    result["per_rank"] = per_rank_report

    # ---- CONCLUSION (rank0) restricted to the cached-prefix domain ----
    r0 = per_rank_report.get(str(args.rank), {}).get("score_flips", {})
    bit_id = r0.get("cached_prefix_flips_score_BIT_IDENTICAL", 0)
    differ = r0.get("cached_prefix_flips_score_DIFFER", 0)
    if (bit_id + differ) == 0:
        verdict = "NO_PREFIX_FLIPS"
    elif differ == 0 and bit_id > 0:
        verdict = (
            "SCORES_BIT_IDENTICAL_AT_FLIPS — pure top-k tie-break at exact ties; "
            "a deterministic tie-break (by logical index) is a viable DS-side fix"
        )
    elif bit_id == 0 and differ > 0:
        verdict = (
            "SCORES_DIFFER_AT_FLIPS — top-k membership genuinely changed; a "
            "tie-break will NOT fix it; bit-identical scores are required"
        )
    else:
        verdict = (
            f"MIXED — {bit_id} flips bit-identical, {differ} flips score-differ "
            "(in cached prefix)"
        )
    result["verdict"] = verdict
    result["verdict_detail_rank0_cached_prefix"] = {
        "flips_bit_identical": bit_id,
        "flips_score_differ": differ,
        "max_abs_score_delta": r0.get("max_abs_score_delta_prefix_flips"),
        "median_abs_score_delta": r0.get("median_abs_score_delta_prefix_flips"),
        "flips_at_cutoff_boundary_tie": r0.get(
            "cached_prefix_flips_at_cutoff_boundary_tie"
        ),
    }
    result["status"] = "DONE"

    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    with gzip.open(os.path.join(outdir, "flip_score_table.jsonl.gz"), "wt") as fh:
        for rrow in flip_rows:
            fh.write(json.dumps(rrow) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
