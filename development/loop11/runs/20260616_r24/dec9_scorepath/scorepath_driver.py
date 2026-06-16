"""DEC-9 SCORE-PATH parity driver (R24 scorepath, owner directive).

One server, one reduce-dtype config (bf16 or fp32 — set by the launcher). COLD
(fresh, populates radix) then WARM (IDENTICAL prompt, radix cache hit). The
score_capture instrument now records BOTH the per-rank PRE-reduce absorbed score
(snapshot before the cross-TP all-reduce) and the POST-reduce score the top-k
consumed. The selection_capture instrument records the selected top-2048 logical
positions per layer.

Reports, per the owner's experiment plan:

 EXP-1 (pre vs post reduce, around the cutoff): at the positions that FLIP in the
   selection (cold top-2048 XOR warm, per layer), restricted to the cached prefix
   domain, report the cold-vs-warm PRE-reduce score delta and the POST-reduce
   delta. If cold/warm already differ PRE-reduce => the difference is upstream in
   v_h / the per-rank score, not the reduce. If pre-reduce is bit-identical and
   the difference appears only POST-reduce => the bf16 reduce introduces it.

 EXP-2 (flip count bf16 vs fp32): this driver emits the cold/warm SELECTION flip
   count for whatever reduce dtype the server runs. The launcher runs it twice
   (bf16 then fp32); a separate combine step compares the two flip counts.

Writes a SMALL summary.json + a per-flip table (gz). Raw .pt deleted by the
launcher. Absolute honesty: raw numbers only, no inference beyond the data.
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


def _score_per_layer(capdir, rank):
    """layer_id -> dict with post-reduce + (optional) pre-reduce fp32 score rows."""
    files = sorted(glob.glob(os.path.join(capdir, f"rank{rank}_req*_layer*.pt")))
    by_layer = {}
    for f in files:
        rec = torch.load(f, weights_only=False)
        by_layer[int(rec["layer_id"])] = {
            "post": rec["scores"],
            "pre": rec.get("pre_reduce_scores"),
            "seq_len": int(rec["seq_len"]),
            "post_dtype": rec.get("score_dtype"),
            "pre_dtype": rec.get("pre_reduce_score_dtype"),
        }
    return by_layer


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
        return None
    rec = torch.load(f, weights_only=False)
    idx = rec["indices"]
    lens = rec["lengths"]
    L = int(idx.shape[0])
    row = 0
    per_layer = {}
    for layer_id in range(L):
        vl = int(lens[layer_id, row].item())
        per_layer[layer_id] = idx[layer_id, row, :vl].to(torch.int64).tolist()
    return per_layer


def _bit_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.view(torch.int32).item() == b.view(torch.int32).item()


def _pct(sorted_list, q):
    n = len(sorted_list)
    if n == 0:
        return 0.0
    return sorted_list[min(n - 1, int(q * n))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-tokens", type=int, default=3800)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--reduce-dtype", required=True, help="bf16 or fp32 (label only)")
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
        "probe": "DEC9_scorepath_pre_vs_post_reduce_at_flips",
        "reduce_dtype": args.reduce_dtype,
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
        cold_sel_pl = _sel_per_layer(cold_sel, rk)
        warm_sel_pl = _sel_per_layer(warm_sel, rk)

        # Whether the pre-reduce row was captured at all (instrument liveness).
        any_pre = any(v.get("pre") is not None for v in cold_sc.values()) and any(
            v.get("pre") is not None for v in warm_sc.values()
        )

        rk_rep = {
            "n_score_layers_cold": len(cold_sc),
            "n_score_layers_warm": len(warm_sc),
            "pre_reduce_captured": bool(any_pre),
            "post_dtype": (
                next(iter(cold_sc.values()))["post_dtype"] if cold_sc else None
            ),
            "pre_dtype": (
                next(iter(cold_sc.values()))["pre_dtype"] if cold_sc else None
            ),
        }

        total_flips = 0
        layers_with_flip = 0
        flips_in_prefix = 0
        flips_in_tail = 0

        # POST-reduce classification at prefix flips
        post_bit_identical = 0
        post_differ = 0
        post_deltas = []
        # PRE-reduce classification at prefix flips
        pre_bit_identical = 0
        pre_differ = 0
        pre_deltas = []
        pre_missing = 0

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
                c_post = c_entry["post"]
                w_post = w_entry["post"]
                c_pre = c_entry["pre"]
                w_pre = w_entry["pre"]

                # kth-largest cutoff (membership boundary) for each pass, post-reduce
                c_cut = (
                    torch.topk(
                        c_post, min(len(cold_sel_pl[layer_id]), c_post.numel())
                    ).values[-1].item()
                    if cold_sel_pl[layer_id]
                    else float("-inf")
                )
                w_cut = (
                    torch.topk(
                        w_post, min(len(warm_sel_pl[layer_id]), w_post.numel())
                    ).values[-1].item()
                    if warm_sel_pl[layer_id]
                    else float("-inf")
                )

                lay_pos = []
                for p in flipped:
                    in_prefix = p < C
                    if in_prefix:
                        flips_in_prefix += 1
                    else:
                        flips_in_tail += 1

                    # POST-reduce values
                    cs_post = (
                        float(c_post[p].item()) if p < c_post.numel() else None
                    )
                    ws_post = (
                        float(w_post[p].item()) if p < w_post.numel() else None
                    )
                    post_bid = None
                    post_delta = None
                    if cs_post is not None and ws_post is not None:
                        post_bid = _bit_identical(c_post[p], w_post[p])
                        post_delta = abs(cs_post - ws_post)
                        if in_prefix:
                            if post_bid:
                                post_bit_identical += 1
                            else:
                                post_differ += 1
                                post_deltas.append(post_delta)

                    # PRE-reduce values
                    cs_pre = ws_pre = None
                    pre_bid = None
                    pre_delta = None
                    if c_pre is not None and w_pre is not None:
                        cs_pre = (
                            float(c_pre[p].item()) if p < c_pre.numel() else None
                        )
                        ws_pre = (
                            float(w_pre[p].item()) if p < w_pre.numel() else None
                        )
                        if cs_pre is not None and ws_pre is not None:
                            pre_bid = _bit_identical(c_pre[p], w_pre[p])
                            pre_delta = abs(cs_pre - ws_pre)
                            if in_prefix:
                                if pre_bid:
                                    pre_bit_identical += 1
                                else:
                                    pre_differ += 1
                                    pre_deltas.append(pre_delta)
                    elif in_prefix:
                        pre_missing += 1

                    rowrec = {
                        "rank": rk,
                        "layer": layer_id,
                        "logical_pos": p,
                        "domain": "cached_prefix" if in_prefix else "tail",
                        "side": "cold_only" if p in cset else "warm_only",
                        "post_cold": cs_post,
                        "post_warm": ws_post,
                        "post_abs_delta": post_delta,
                        "post_bit_identical": post_bid,
                        "pre_cold": cs_pre,
                        "pre_warm": ws_pre,
                        "pre_abs_delta": pre_delta,
                        "pre_bit_identical": pre_bid,
                        "post_cold_cutoff_kth": c_cut,
                        "post_warm_cutoff_kth": w_cut,
                    }
                    lay_pos.append(rowrec)
                    flip_rows.append(rowrec)

                layer_detail.append(
                    {
                        "layer": layer_id,
                        "n_flipped": len(flipped),
                        "cold_valid_len": len(cold_sel_pl[layer_id]),
                        "warm_valid_len": len(warm_sel_pl[layer_id]),
                        "post_cold_cutoff_kth": c_cut,
                        "post_warm_cutoff_kth": w_cut,
                        "per_pos": lay_pos[:6],
                    }
                )

        post_sorted = sorted(post_deltas)
        pre_sorted = sorted(pre_deltas)
        rk_rep["flips"] = {
            "n_sel_layers_compared": (
                len(set(cold_sel_pl) & set(warm_sel_pl))
                if (cold_sel_pl and warm_sel_pl)
                else 0
            ),
            "n_layers_with_flip": layers_with_flip,
            "total_flipped_positions": total_flips,
            "flips_in_cached_prefix": flips_in_prefix,
            "flips_in_tail": flips_in_tail,
            # POST-reduce (what the prior round measured)
            "post_reduce_prefix_flips_BIT_IDENTICAL": post_bit_identical,
            "post_reduce_prefix_flips_DIFFER": post_differ,
            "post_reduce_abs_delta_median": (
                post_sorted[len(post_sorted) // 2] if post_sorted else 0.0
            ),
            "post_reduce_abs_delta_p99": _pct(post_sorted, 0.99),
            "post_reduce_abs_delta_max": (post_sorted[-1] if post_sorted else 0.0),
            # PRE-reduce (NEW exp-1 — the decisive measurement)
            "pre_reduce_captured": bool(any_pre),
            "pre_reduce_prefix_flips_BIT_IDENTICAL": pre_bit_identical,
            "pre_reduce_prefix_flips_DIFFER": pre_differ,
            "pre_reduce_prefix_flips_MISSING": pre_missing,
            "pre_reduce_abs_delta_median": (
                pre_sorted[len(pre_sorted) // 2] if pre_sorted else 0.0
            ),
            "pre_reduce_abs_delta_p99": _pct(pre_sorted, 0.99),
            "pre_reduce_abs_delta_max": (pre_sorted[-1] if pre_sorted else 0.0),
            "first_layers_with_flip_detail": layer_detail[:4],
        }
        per_rank_report[str(rk)] = rk_rep

    result["per_rank"] = per_rank_report
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
