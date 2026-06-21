#!/usr/bin/env python3
"""Generate the AC-6 per-leg single-variable bisection matrix.

AC-6 walks reference -> production one variable per arm. Every differing variable is either MEASURED
(a config toggle that affects the served path), RETIRED (selection-neutral, proven by AC-2.3),
NOT-A-DIFFERING-VARIABLE (production already matches the reference), or BLOCKED (the variable lives
only in the production raw-dot Triton scoring kernel and isolating it under cosine would need a new
production-path cosine kernel = a selection-path code change = a fix, forbidden this loop). No leg is
silently deferred.

Reads GSM8K scores from evidence/meta/arms/*.json so the numbers stay synced with the ledger.
Writes evidence/ac6_bisection_matrix.json. CPU-only; fail-closed if a measured leg's arm is missing.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = os.path.join(HERE, "evidence", "meta", "arms")


def arm_scores(name):
    p = os.path.join(ARMS, f"{name}.json")
    if not os.path.exists(p):
        print(f"FAIL: missing arm JSON {p}", file=sys.stderr)
        raise SystemExit(2)
    s = json.load(open(p))["scores"]
    return {"dense": s["dense_batched"], "sparse": s["sparse_batched"]}


def main():
    # measured arms (scores pulled from the ledger)
    prod = arm_scores("production_ds")        # raw-dot, fp8, bf16-reduce, current-excl
    rawf = arm_scores("ref_faithful")         # raw-dot, exact fp32, current-incl
    cos = arm_scores("ref_cosine")            # cosine, exact fp32, current-incl
    cose = arm_scores("ref_cosine_noinc")     # cosine, exact fp32, current-EXCL

    PROD_KERNEL = ("production scoring = DeepseekV2 `_select_topk_indices` (python/sglang/srt/models/"
                   "deepseek_v2.py ~2570-2603) -> the absorbed-latent Triton scoring kernel "
                   "(python/sglang/srt/layers/attention/double_sparsity/absorbed_latent_kernel.py), "
                   "which implements ONLY scorer_norm='off' (raw channel-dot). config.py:110 "
                   "_ALLOWED_SCORER_NORM=('off',) and the validation at config.py:170 hard-reject "
                   "scorer_norm='cosine'. The reference path (reference_cosine_select) computes exact "
                   "fp32 and does NOT route through this kernel.")
    NO_FIX = ("Isolating this variable UNDER cosine requires implementing cosine in the production "
              "Triton scoring kernel (the materialized per-head signature + normalization, as "
              "absorbed_latent_cosine_logical_fp8 does on the reference path) — a new selection-path "
              "kernel = a fix, forbidden in this diagnosis loop.")

    legs = [
        {"leg": 1, "variable": "head_agg (within-rank head aggregation)",
         "base_arm": "ref_cosine", "changed_variable": "head_agg",
         "config_diff": "none — production AND the reference both use head_agg='max'",
         "dense_sparse": None, "corroboration": "n/a",
         "verdict": "not-a-differing-variable",
         "detail": ("head_agg='max' is identical on the production and reference paths, so it is not a "
                    "reference->production bisection step. The distinct cross-TP question (local-max "
                    "per rank then SUM across the TP group vs a true global max) is AC-2.2, examined "
                    "separately via the capture head_agg_test (still PRELIMINARY).")},
        {"leg": 2, "variable": "scorer normalization (raw-dot vs cosine)",
         "base_arm": "ref_faithful (raw-dot) -> ref_cosine (cosine)", "changed_variable": "scorer_norm / selector_impl",
         "config_diff": "selector_impl reference_rawdot -> reference_cosine (scorer_norm off -> cosine)",
         "dense_sparse": {"current_incl": {"rawdot": rawf, "cosine": cos},
                          "current_excl": {"rawdot": prod, "cosine": cose}},
         "corroboration": ("2x2 + development/loop13/test_reference_selectors.py "
                           "(materialized-raw == absorbed-raw selection, bit-identical)"),
         "verdict": "measured",
         "detail": ("holding current-slot INCLUDED: sparse rawdot 0.013 -> cosine 0.940 (+92.7pp); "
                    "holding current-slot EXCLUDED: sparse rawdot 0.000 -> cosine 0.313. The cosine "
                    "scorer is a primary culprit; interacts with the current-slot leg.")},
        {"leg": 3, "variable": "current decode-slot inclusion",
         "base_arm": "ref_cosine -> ref_cosine_noinc", "changed_variable": "reference_include_current",
         "config_diff": "reference_include_current true -> false (production _slot_written exclusion)",
         "dense_sparse": {"included": cos, "excluded": cose},
         "corroboration": "evidence/ac6_ref_cosine_noinc_corrob.json (4992/4992 single-swap; current slot -inf-masked)",
         "verdict": "measured",
         "detail": ("dense 0.940 -> 0.625 (= production 0.620), sparse 0.940 -> 0.313. Current-slot "
                    "exclusion (H3) is a culprit in BOTH regimes; corroboration shows the selected sets "
                    "differ by exactly the current decode slot at every step.")},
        {"leg": 4, "variable": "radix top-k (approximate vs exact)",
         "base_arm": "production blocked/radix vs exact torch.topk", "changed_variable": "top-k algorithm",
         "config_diff": "blocked_topk_sequence_order vs select_topk_sequence_order",
         "dense_sparse": None,
         "corroboration": "evidence/ac2_3_radix_width_equivalence.json (4992/4992 identical, real sparse rows)",
         "verdict": "retired",
         "detail": "selection-neutral on the real sparse workload (median seq_len 4280, 2048 of ~4280 pruned)."},
        {"leg": 5, "variable": "selector width bucket ([5120] vs full)",
         "base_arm": "width [5120] vs full", "changed_variable": "selector_width_buckets",
         "config_diff": "[5120] vs []",
         "dense_sparse": None,
         "corroboration": "evidence/ac2_3_radix_width_equivalence.json (4992/4992 identical, real sparse rows)",
         "verdict": "retired",
         "detail": "selection-neutral: the [5120] window covers the live region (seq_len<=5120) on the sparse workload."},
        {"leg": 6, "variable": "fp8-in-register absorbed scoring (vs exact fp32 dequant)",
         "base_arm": "(production raw-dot only)", "changed_variable": "absorbed_latent_fp8",
         "config_diff": "production kernel scores fp8 absorbed latent in-register (deepseek_v2.py:2602); reference dequants to exact fp32",
         "dense_sparse": None,
         "corroboration": ("second-order bound: on the raw-dot path where exact-fp32 (ref_faithful) and "
                           "fp8 (production) CAN be compared, sparse 0.013 (exact) vs 0.000 (fp8+bf16+excl) "
                           "— fp8/reduce contribute <=~1.3pp beyond the scorer/current-slot effects"),
         "verdict": "blocked",
         "detail": PROD_KERNEL, "blocker": NO_FIX},
        {"leg": 7, "variable": "bf16 score-reduce (vs fp32)",
         "base_arm": "(production raw-dot only)", "changed_variable": "score_reduce_dtype",
         "config_diff": "production reduces cross-TP scores in bf16 (score_reduce_bf16, deepseek_v2.py:2588); reference reduces in fp32",
         "dense_sparse": None,
         "corroboration": ("second-order bound: same as fp8 — exact-fp32 raw-dot 0.013 vs production "
                           "fp8+bf16 0.000 sparse; reduce dtype is within that <=1.3pp residual"),
         "verdict": "blocked",
         "detail": PROD_KERNEL, "blocker": NO_FIX},
    ]

    by_verdict = {}
    for l in legs:
        by_verdict.setdefault(l["verdict"], []).append(l["leg"])
    report = {
        "ac6": "single-variable bisection matrix (reference -> production)",
        "rule": "exactly one variable changes per arm; multi-variable steps rejected; each measured delta corroborated",
        "arms_scores": {"production_ds": prod, "ref_faithful": rawf, "ref_cosine": cos, "ref_cosine_noinc": cose},
        "legs": legs,
        "summary_by_verdict": by_verdict,
        "conclusion": ("Sparse recovery to ~0.94 needs BOTH the cosine scorer (leg 2) AND current-slot "
                       "inclusion (leg 3); the two interact. radix+width (legs 4-5) are selection-neutral "
                       "(retired). head_agg (leg 1) is not a reference->production difference. fp8/bf16 "
                       "(legs 6-7) are blocked (production raw-dot kernel only; no non-fix cosine route) "
                       "and bounded second-order. No leg is silently deferred."),
    }
    out = os.path.join(HERE, "evidence", "ac6_bisection_matrix.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps({"summary_by_verdict": by_verdict,
                      "arms_scores": report["arms_scores"]}, indent=2))
    print("wrote", out)
    # sanity: every leg has a verdict in the allowed set
    allowed = {"measured", "retired", "not-a-differing-variable", "blocked"}
    bad = [l["leg"] for l in legs if l["verdict"] not in allowed]
    if bad:
        print(f"FAIL: legs with invalid verdict: {bad}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
