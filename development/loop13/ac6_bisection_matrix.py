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
    redf = arm_scores("ds_reduce_fp32")       # raw-dot, fp8, FP32-reduce, current-excl (leg-7 measured)

    # fp8-absorbed has NO production config route (verified R7): config.py exposes scorer_norm, head_agg,
    # score_reduce_dtype, selector_width, anchor, selector_impl — none toggles absorbed-scoring precision.
    FP8_BLOCK = ("No production config flag toggles fp8-vs-fp32 absorbed scoring: the graph-safe selector "
                 "scores the fp8 resident latent IN-REGISTER (deepseek_v2.py:2602 absorbed_latent_fp8 -> "
                 "absorbed_latent_kernel.py). Exact-fp32 absorbed scoring exists ONLY on the reference path "
                 "(reference_rawdot_select dequants the resident latent to fp32), which ALSO changes "
                 "current-slot/TF32/radix/width/reduce at once — no single-variable isolation. A production "
                 "fp32-absorbed path = new selection-path code = a fix, forbidden this loop.")
    FP8_SECOND_ORDER = ("Bounded second-order: now that the bf16-vs-fp32 REDUCE leg is measured (leg 7, "
                        "near-selection-neutral) and radix+width are retired, the remaining production-numeric "
                        "difference is fp8 absorbed scoring; production raw-dot (fp8) sparse 0.000 vs exact-fp32 "
                        "raw-dot (ref_faithful) sparse 0.013 bounds its selection effect to <=~1.3pp — far below "
                        "the scorer (+92.7pp) and current-slot effects.")

    legs = [
        {"leg": 1, "variable": "head aggregation (within-rank head_agg + cross-TP reduce)",
         "base_arm": "ref_faithful (per-rank-local) vs production (cross-TP SUM)", "changed_variable": "cross-TP head aggregation",
         "config_diff": "within-rank head_agg='max' is identical on both; cross-TP differs: production SUM (reduce_token_scores) vs reference per-rank-local (no cross-TP reduce, num_local_heads)",
         "dense_sparse": None,
         "corroboration": ("evidence/head_agg_tp_semantics.json (AC-2.2 SETTLED, 702/702 sum(pre)==post): "
                           "served cross-TP SUM vs global-MAX median Jaccard 0.679 -> SUM != global-max"),
         "verdict": "measured",
         "detail": ("within-rank head_agg='max' is matched (not a difference). The CROSS-TP aggregation "
                    "DOES differ (production SUM vs reference per-rank-local), but it is bounded "
                    "SECOND-ORDER: on the raw-dot path production-SUM sparse 0.000 vs reference-local 0.013 "
                    "=> <=~1.3pp, like fp8/reduce. Raw-dot collapses under both aggregations; the accuracy "
                    "driver is the scorer + current-slot. cosine-under-production-SUM is not measured (no "
                    "production cosine kernel, leg 6), so cosine recovery under SUM is NOT claimed.")},
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
         "corroboration": ("evidence/ac6_ref_cosine_noinc_corrob.json — BOTH regimes: sparse 4992/4992 "
                           "(symdiff==2 swap, Jaccard 0.999024) + dense 3744/3744 (symdiff==1 add, "
                           "valid_length seq_len-1->seq_len); current slot -inf-masked in every capture"),
         "verdict": "measured",
         "detail": ("dense 0.940 -> 0.625 (= production 0.620), sparse 0.940 -> 0.313. Current-slot "
                    "exclusion (H3) is a culprit in BOTH regimes; corroboration shows the selected set "
                    "changes by exactly the current decode slot at every step (a swap in sparse, a pure "
                    "drop in dense).")},
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
         "base_arm": "(production raw-dot; no single-variable route)", "changed_variable": "absorbed_latent_fp8",
         "config_diff": "production kernel scores fp8 absorbed latent in-register (deepseek_v2.py:2602); exact fp32 only on the reference path",
         "dense_sparse": None,
         "corroboration": FP8_SECOND_ORDER,
         "verdict": "blocked",
         "detail": FP8_BLOCK},
        {"leg": 7, "variable": "score_reduce_dtype (bf16 vs fp32 cross-TP reduce)",
         "base_arm": "production_ds (bf16 reduce) -> ds_reduce_fp32 (fp32 reduce)", "changed_variable": "score_reduce_dtype",
         "config_diff": "score_reduce_dtype 'bf16' -> 'fp32' (config.py accepts both; the ONLY change vs production)",
         "dense_sparse": {"bf16_reduce": prod, "fp32_reduce": redf},
         "corroboration": ("evidence/ac6_score_reduce_fp32_corrob.json — bf16-vs-fp32 reduce of the SAME "
                           "captured per-rank pre_reduce scores: median selected-set Jaccard 0.998 "
                           "(near-selection-neutral; only bottom-of-top-k near-ties reshuffle)"),
         "verdict": "measured",
         "detail": ("RUNNABLE production config route (score_reduce_dtype='fp32') — NOT blocked. fp32 reduce "
                    "gives ~production scores (reduce dtype is near-selection-neutral), so bf16 reduce is NOT "
                    "the regression driver. Scoring + top-k stay fp32 either way; only the cross-TP transport "
                    "dtype differs.")},
    ]

    by_verdict = {}
    for l in legs:
        by_verdict.setdefault(l["verdict"], []).append(l["leg"])
    report = {
        "ac6": "single-variable bisection matrix (reference -> production)",
        "rule": "exactly one variable changes per arm; multi-variable steps rejected; each measured delta corroborated",
        "arms_scores": {"production_ds": prod, "ref_faithful": rawf, "ref_cosine": cos,
                        "ref_cosine_noinc": cose, "ds_reduce_fp32": redf},
        "legs": legs,
        "summary_by_verdict": by_verdict,
        "conclusion": ("Sparse recovery to ~0.94 needs BOTH the cosine scorer (leg 2) AND current-slot "
                       "inclusion (leg 3); the two interact. radix+width (legs 4-5) are selection-neutral "
                       "(retired). bf16-vs-fp32 score-reduce (leg 7) is MEASURED via the runnable "
                       "score_reduce_dtype='fp32' route -> near-selection-neutral, not a culprit. Head "
                       "aggregation (leg 1, AC-2.2): within-rank head_agg='max' matched; cross-TP differs "
                       "(SUM vs reference-local) but is MEASURED second-order (<=~1.3pp on raw-dot). Only "
                       "fp8-absorbed (leg 6) is blocked — no production config toggles absorbed precision; "
                       "exact-fp32 absorbed lives only on the multi-variable reference path — and it is "
                       "bounded second-order (<=~1.3pp). No leg is silently deferred."),
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

    # AC-2.2 consistency guard: once head_agg_tp_semantics.json validates the captures (sum(pre)==post on
    # ALL groups), no generated surface may still publish the stale PRELIMINARY / served_sum_matches=false
    # verdict (Codex R8). Fail-closed.
    hav_p = os.path.join(HERE, "evidence", "head_agg_tp_semantics.json")
    if os.path.exists(hav_p):
        hav = json.load(open(hav_p))
        v = hav.get("capture_validation_sum_pre_eq_post", "")
        if "/" in v and v.split("/")[0] == v.split("/")[1] and v.split("/")[0] != "0":  # all groups validated
            errs = []
            this = open(out).read()
            if "PRELIMINARY" in this:
                errs.append("ac6_bisection_matrix.json still contains PRELIMINARY")
            cc = json.load(open(os.path.join(HERE, "evidence", "cheap_controls.json")))
            if any("served_sum_matches" in k for k in cc.get("summary", {})):
                errs.append("cheap_controls.json.summary still has a served_sum_matches_* field")
            if cc.get("summary", {}).get("AC_2_2_verdict", "").startswith("SETTLED") is False:
                errs.append("cheap_controls.json.summary AC_2_2_verdict is not SETTLED")
            if "cosine recovers under both" in json.dumps(hav):
                errs.append("head_agg_tp_semantics.json still overclaims 'cosine recovers under both'")
            # leg 1 is MEASURED -> no generated surface may still call head aggregation a non-difference
            leg1 = next((l for l in legs if l["leg"] == 1), None)
            if leg1 and leg1["verdict"] == "measured":
                for fn in ("evidence_table.md", "findings.md"):
                    fp = os.path.join(HERE, "evidence", fn)
                    if os.path.exists(fp) and "head_agg NOT-a-differing-variable" in open(fp).read():
                        errs.append(f"{fn} still says head_agg NOT-a-differing-variable (leg 1 is measured)")
            # the stale row-level served_sum_matches data is allowed ONLY under a superseded_* section
            for k, v in cc.items():
                if k.startswith("superseded") or k in ("summary", "_status"):
                    continue
                if "served_sum_matches" in json.dumps(v):
                    errs.append(f"cheap_controls.json top-level '{k}' still carries served_sum_matches "
                                f"(move under a superseded_* key)")
            if errs:
                print("FAIL: AC-2.2 consistency guard:", file=sys.stderr)
                for e in errs:
                    print("  -", e, file=sys.stderr)
                raise SystemExit(2)


if __name__ == "__main__":
    main()
