"""DEC-9 scorepath combine — assemble the three experiments into one summary
with raw numbers and the feasibility verdict. No new measurement; reads the
per-pass driver summaries + exp-3 output. Absolute honesty: the verdict follows
the raw numbers mechanically.
"""

from __future__ import annotations

import argparse
import json
import sys


def _load(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception as e:  # noqa: BLE001
        return {"_load_error": f"{type(e).__name__}: {e}", "_path": p}


def _flips(summary, rank="0"):
    pr = summary.get("per_rank", {}).get(rank, {})
    return pr.get("flips", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--exp3", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bf16 = _load(args.bf16)
    fp32 = _load(args.fp32)
    exp3 = _load(args.exp3)

    out = {
        "probe": "DEC-9 score-PATH parity (pre/post reduce, fp32 reduce, radix-vs-fallback top-k)",
        "runtime_base_commit": "f82ec2c5d",
        "op_point": "GLM-5.1-FP8 TP=8 fp8_e4m3 page64 mem0.8 max-running64 EAGER radix-ON",
    }

    bf16_r0 = _flips(bf16, "0")
    bf16_r4 = _flips(bf16, "4")
    fp32_r0 = _flips(fp32, "0")
    fp32_r4 = _flips(fp32, "4")

    # ---- EXP-1: pre vs post reduce at flips (from the bf16 pass) ----
    out["exp1_pre_vs_post_reduce"] = {
        "source": "bf16 reduce pass (the production default; the pass that flips)",
        "bf16_cached_tokens_warm": bf16.get("warm_cached_tokens"),
        "rank0": {
            "total_flipped_positions": bf16_r0.get("total_flipped_positions"),
            "flips_in_cached_prefix": bf16_r0.get("flips_in_cached_prefix"),
            "flips_in_tail": bf16_r0.get("flips_in_tail"),
            "pre_reduce_captured": bf16_r0.get("pre_reduce_captured"),
            "POST_reduce_prefix_flips_BIT_IDENTICAL": bf16_r0.get(
                "post_reduce_prefix_flips_BIT_IDENTICAL"
            ),
            "POST_reduce_prefix_flips_DIFFER": bf16_r0.get(
                "post_reduce_prefix_flips_DIFFER"
            ),
            "POST_reduce_abs_delta_median": bf16_r0.get(
                "post_reduce_abs_delta_median"
            ),
            "POST_reduce_abs_delta_p99": bf16_r0.get("post_reduce_abs_delta_p99"),
            "POST_reduce_abs_delta_max": bf16_r0.get("post_reduce_abs_delta_max"),
            "PRE_reduce_prefix_flips_BIT_IDENTICAL": bf16_r0.get(
                "pre_reduce_prefix_flips_BIT_IDENTICAL"
            ),
            "PRE_reduce_prefix_flips_DIFFER": bf16_r0.get(
                "pre_reduce_prefix_flips_DIFFER"
            ),
            "PRE_reduce_prefix_flips_MISSING": bf16_r0.get(
                "pre_reduce_prefix_flips_MISSING"
            ),
            "PRE_reduce_abs_delta_median": bf16_r0.get("pre_reduce_abs_delta_median"),
            "PRE_reduce_abs_delta_p99": bf16_r0.get("pre_reduce_abs_delta_p99"),
            "PRE_reduce_abs_delta_max": bf16_r0.get("pre_reduce_abs_delta_max"),
        },
        "rank4": {
            "total_flipped_positions": bf16_r4.get("total_flipped_positions"),
            "PRE_reduce_prefix_flips_BIT_IDENTICAL": bf16_r4.get(
                "pre_reduce_prefix_flips_BIT_IDENTICAL"
            ),
            "PRE_reduce_prefix_flips_DIFFER": bf16_r4.get(
                "pre_reduce_prefix_flips_DIFFER"
            ),
            "POST_reduce_prefix_flips_DIFFER": bf16_r4.get(
                "post_reduce_prefix_flips_DIFFER"
            ),
        },
    }

    # ---- EXP-2: flip count bf16 vs fp32 ----
    out["exp2_flip_count_bf16_vs_fp32"] = {
        "bf16": {
            "warm_cached_tokens": bf16.get("warm_cached_tokens"),
            "rank0_total_flips": bf16_r0.get("total_flipped_positions"),
            "rank0_prefix_flips": bf16_r0.get("flips_in_cached_prefix"),
            "rank4_total_flips": bf16_r4.get("total_flipped_positions"),
        },
        "fp32": {
            "warm_cached_tokens": fp32.get("warm_cached_tokens"),
            "rank0_total_flips": fp32_r0.get("total_flipped_positions"),
            "rank0_prefix_flips": fp32_r0.get("flips_in_cached_prefix"),
            "rank4_total_flips": fp32_r4.get("total_flipped_positions"),
        },
    }

    # ---- EXP-3: radix vs fallback top-k ----
    out["exp3_radix_vs_fallback_topk"] = {
        "verdict": exp3.get("verdict"),
        "real_rows": exp3.get("real_rows"),
        "real_lengths": exp3.get("real_lengths"),
        "comparisons": exp3.get("comparisons"),
    }

    # ---- FEASIBILITY VERDICT (mechanical from the raw numbers) ----
    bf16_prefix_flips = bf16_r0.get("flips_in_cached_prefix")
    fp32_prefix_flips = fp32_r0.get("flips_in_cached_prefix")
    pre_differ = bf16_r0.get("pre_reduce_prefix_flips_DIFFER")
    pre_bit_id = bf16_r0.get("pre_reduce_prefix_flips_BIT_IDENTICAL")
    post_differ = bf16_r0.get("post_reduce_prefix_flips_DIFFER")
    topk_diverge = any(
        c.get("total_member_symdiff", 0) > 0 for c in exp3.get("comparisons", [])
    )

    verdict = {}

    # exp-3 decision
    if topk_diverge:
        verdict["exp3_topk_path"] = (
            "DIVERGE — radix and fallback top-k select different members on the "
            "SAME score tensor => the top-k path is a culprit; canonical top-k is "
            "a DS-side fix. See exp3 comparisons for counts."
        )
    else:
        verdict["exp3_topk_path"] = (
            "AGREE — radix, fallback, and reference top-k produce identical "
            "membership and order on the same scores; the top-k path is NOT the "
            "culprit."
        )

    # exp-2 decision
    if (
        isinstance(bf16_prefix_flips, int)
        and isinstance(fp32_prefix_flips, int)
    ):
        if fp32_prefix_flips == 0 and bf16_prefix_flips > 0:
            verdict["exp2_fp32_reduce"] = (
                f"FP32 REDUCE ZEROES THE FLIPS (bf16={bf16_prefix_flips} -> "
                f"fp32=0) => the bf16 reduce was the culprit; pinning fp32 reduce "
                f"on the radix-on path is a DS-side fix."
            )
        elif fp32_prefix_flips < bf16_prefix_flips:
            verdict["exp2_fp32_reduce"] = (
                f"FP32 REDUCE REDUCES BUT DOES NOT ZERO THE FLIPS "
                f"(bf16={bf16_prefix_flips} -> fp32={fp32_prefix_flips}); the bf16 "
                f"reduce contributes but is not the sole cause."
            )
        else:
            verdict["exp2_fp32_reduce"] = (
                f"FP32 REDUCE DOES NOT REDUCE THE FLIPS "
                f"(bf16={bf16_prefix_flips}, fp32={fp32_prefix_flips}); the bf16 "
                f"reduce is NOT the culprit."
            )
    else:
        verdict["exp2_fp32_reduce"] = "INCONCLUSIVE — missing flip counts"

    # exp-1 decision
    if isinstance(pre_differ, int) and isinstance(pre_bit_id, int):
        tot = pre_differ + pre_bit_id
        if tot == 0:
            verdict["exp1_pre_reduce"] = "NO_PREFIX_FLIPS_WITH_PRE_SCORE"
        elif pre_differ > 0:
            verdict["exp1_pre_reduce"] = (
                f"PRE-REDUCE SCORES ALREADY DIFFER at {pre_differ}/{tot} prefix "
                f"flips (median |Δ|={bf16_r0.get('pre_reduce_abs_delta_median')}) "
                f"=> the divergence is upstream in the per-rank score (v_h), "
                f"present BEFORE the cross-TP reduce."
            )
        else:
            verdict["exp1_pre_reduce"] = (
                f"PRE-REDUCE SCORES BIT-IDENTICAL at all {tot} prefix flips while "
                f"POST-reduce differs at {post_differ} => the difference appears "
                f"only AFTER the reduce (the bf16 reduce introduces it)."
            )
    else:
        verdict["exp1_pre_reduce"] = "INCONCLUSIVE — pre-reduce not captured"

    # overall feasibility
    if (
        isinstance(fp32_prefix_flips, int)
        and isinstance(bf16_prefix_flips, int)
        and fp32_prefix_flips == 0
        and bf16_prefix_flips > 0
    ):
        verdict["FEASIBILITY"] = (
            "DS-SIDE FIX EXISTS via fp32 reduce (exp-2 zeroed the flips)."
        )
    elif topk_diverge:
        verdict["FEASIBILITY"] = (
            "DS-SIDE FIX EXISTS via canonical top-k (exp-3 top-k paths diverge)."
        )
    elif isinstance(pre_differ, int) and pre_differ > 0:
        verdict["FEASIBILITY"] = (
            "NOT DS-FIXABLE — pre-reduce scores already differ cold-vs-warm "
            "(exp-1) and fp32 reduce does not zero flips (exp-2): the divergence "
            "is genuinely upstream in v_h (the cached vs fresh model forward), "
            "outside DS authority."
        )
    else:
        verdict["FEASIBILITY"] = "SEE RAW NUMBERS — no single experiment was decisive"

    out["FEASIBILITY_VERDICT"] = verdict

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
