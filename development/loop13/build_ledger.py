#!/usr/bin/env python3
"""Generate the AC-4 per-arm evidence ledger from the committed run artifacts.

Emits evidence/meta/arms/<arm>.json for every measured arm and regenerates
evidence/evidence_table.md from those JSONs. Scores are read from the run_eval
.out files (data-driven); fixed config/selected-vs-total come from the recorded
probes. Fields that require harness instrumentation not built this loop
(per-example sample IDs/order, per-step length-cap garbage counters) are listed
under "fields_not_instrumented" rather than faked — fail-honest, not fail-closed-silent.
"""
import glob
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
EVID = os.path.join(HERE, "evidence")
ARMS_DIR = os.path.join(EVID, "meta", "arms")
os.makedirs(ARMS_DIR, exist_ok=True)


def git_sha():
    return subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"]).decode().strip()


def generator_provenance():
    """Unambiguous generator-source identity even when run from a dirty worktree
    (build_ledger.py emits evidence BEFORE the commit that contains it exists, so
    HEAD alone is one commit behind). Returns (head_sha, generator_blob_sha,
    worktree_state)."""
    head = git_sha()
    blob = subprocess.check_output(
        ["git", "-C", HERE, "hash-object", os.path.abspath(__file__)]
    ).decode().strip()  # content hash of THIS generator file (commit-independent)
    dirty = bool(subprocess.check_output(
        ["git", "-C", HERE, "status", "--porcelain", "."]  # "." == this loop13 dir (HERE)
    ).decode().strip())
    return head, blob, ("dirty (+uncommitted evidence/generator)" if dirty else "clean")


def score_from_out(label_tag):
    """Read 'Score: X' from evidence/<label_tag>.out, or None."""
    p = os.path.join(EVID, f"{label_tag}.out")
    if not os.path.exists(p):
        return None
    m = re.findall(r"^Score:\s*([0-9.]+)", open(p).read(), re.M)
    return float(m[-1]) if m else None


GEN_HEAD, GEN_BLOB, GEN_WORKTREE = generator_provenance()
COMMON_ARGS = ("--tp-size 8 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.8 "
               "--max-running-requests 64 --cuda-graph-max-bs 64 --page-size 64 "
               "--dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv "
               "--disable-overlap-schedule --disable-piecewise-cuda-graph --random-seed 42 "
               "--trust-remote-code")

# measured_git_sha: the commit the arm's run actually happened under (NOT the
# generator HEAD). Baselines (dsa/dsa_noradix/production_ds) do not use any new
# diagnostic code, so they are SHA-independent; recorded at the Round-0 base SHA.
BASE_SHA = "180f6dd6decb1577da8e40bf002f79805ece693d"   # Round-0 baselines + plan/harness
DIAG0_SHA = "fc6ac20a7"   # Round-0 diagnostic code (forced_all_dense_control)
HARNESS_SHA = "29ec137bf"  # Round-0 harness with ds_anchor mode (anchor_mode pre-existing)
R1_SHA = "fea920c06"      # Round-1 faithful/cosine reference code
# Round-5 measured-run identity for ref_cosine_noinc: the serve mode was added in R5, so it did NOT
# exist at R1_SHA. The arm ran at worktree HEAD 393966c02 with serve.sh dirty (the ref_cosine_noinc
# mode), later committed as c7b66f04b; the serve.sh blob below pins the exact mode definition.
R5_NOINC_SHA = "393966c02d0d57d0c99c355367f52704c1964581"
R5_NOINC_SOURCE = ("worktree HEAD 393966c02 (dirty: serve.sh ref_cosine_noinc mode added R5; "
                   "serve.sh blob e1c83e22a085f0aa499adfbaca155d9aa0069579; committed c7b66f04b). "
                   "reference_cosine_select selection code unchanged since R1 fea920c06.")
# Round-7 measured-run identity for ds_reduce_fp32 (serve mode added R7).
R7_REDUCE_SHA = "8b55dfba3"
R7_REDUCE_SOURCE = ("worktree HEAD 8b55dfba3 (dirty: serve.sh ds_reduce_fp32 mode added R7; serve.sh "
                    "blob 1324c5a6cf21a1916e34bbad0cbc1c57cf1d518d). Config-only diff vs `ds` (same graph "
                    "mode): score_reduce_dtype=fp32; production raw-dot selection code unchanged.")

# Canonical per-arm DS launch config — the exact --double-sparsity-config serve.sh passes for each DS
# mode (AC-1/AC-4 require the FULL launch config, not abbreviated extras).
MASK_PATH = "/cluster-storage/models/glm51-fp8-channel-mask-loop12.safetensors"
DS_BASE = {"top_k": 2048, "page_size": 64, "channel_mask_path": MASK_PATH, "device_buffer_size": 4096,
           "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0,
           "enable_lifted_budget_decode": False, "lifted_budget_top_k": 0}
DS_OVERRIDES = {  # vs DS_BASE, matching serve.sh exactly
    "production_ds": {},
    "ref_faithful": {"selector_impl": "reference_rawdot", "reference_include_current": True},
    "ref_cosine": {"selector_impl": "reference_cosine", "reference_include_current": True},
    "ref_cosine_noinc": {"selector_impl": "reference_cosine", "reference_include_current": False},
    "ds_reduce_fp32": {"score_reduce_dtype": "fp32"},
    "ds_forced_all": {"forced_all_dense_control": True},
    "ds_anchor_b1": {"anchor_mode": "recency", "anchor_budget": 1},
    "ds_anchor_b64": {"anchor_mode": "recency", "anchor_budget": 64},
}


# Full resolved DoubleSparsityConfig defaults (config.py) — the EFFECTIVE runtime config after defaults.
# channel_mask_path is required (no default) and comes from the launch JSON.
DS_DEFAULTS = {
    "top_k": 2048, "page_size": 64, "device_buffer_size": 4096,
    "scorer_norm": "off", "head_agg": "max", "anchor_mode": "off", "anchor_budget": 0,
    "recall_oracle": False, "selection_capture": False, "latent_capture": False, "score_capture": False,
    "selector_width_buckets": [5120], "selector_width_overflow_policy": "full_fallback",
    "score_reduce_dtype": "bf16", "enable_lifted_budget_decode": False, "lifted_budget_top_k": 0,
    "selector_impl": "production", "forced_all_dense_control": False, "forced_all_assert": False,
    "reference_include_current": False,
}
# AC-4 wants these effective fields per DS arm (selector width / reduce dtype / scorer / head-agg / impl):
DS_EFFECTIVE_REQUIRED = ("selector_width_buckets", "score_reduce_dtype", "selector_impl",
                         "head_agg", "scorer_norm")


def ds_config_for(arm):
    """The literal launch-JSON DS config for a DS arm (DS_BASE + per-arm overrides) — exactly what
    serve.sh passed via --double-sparsity-config. None for non-DS arms."""
    if arm not in DS_OVERRIDES:
        return None
    return {**DS_BASE, **DS_OVERRIDES[arm]}


def effective_ds_config_for(arm):
    """The EFFECTIVE runtime DS config = all DoubleSparsityConfig fields resolved (defaults +
    channel_mask_path + the arm's launch overrides). None for non-DS arms. NOTE: a knob being set in
    this config object does NOT mean the selector path uses it — see ds_selector_behavior_for()."""
    if arm not in DS_OVERRIDES:
        return None
    return {**DS_DEFAULTS, "channel_mask_path": MASK_PATH, **DS_OVERRIDES[arm]}


def ds_selector_behavior_for(arm):
    """What the selector ACTUALLY does (AC-4 behavior surface), keyed on selector_impl. The reference_*
    paths (deepseek_v2.py:_reference_selector_topk) dequantize to fp32 and run the EXACT absorbed
    channel-dot + full-width torch.topk — no fp8-in-register, no bf16 reduce, no radix kernel, no
    selector-width bucketing — so the production width/reduce/radix/fp8 knobs in effective_ds_config are
    DORMANT on those arms. None for non-DS arms."""
    eff = effective_ds_config_for(arm)
    if eff is None:
        return None
    impl = eff["selector_impl"]
    if impl.startswith("reference_"):
        return {
            "path": "reference (eager fp32, perf-naive diagnostic)",
            "selector_width": "full (no bucketing)",
            "score_reduce": "none (per-rank-local fp32; no cross-TP reduce)",
            "topk": "exact torch.topk",
            "scoring": "exact fp32 dequant",
            "scorer": "cosine" if impl == "reference_cosine" else "raw-dot",
            "head_agg": eff["head_agg"],
            "note": "production width/reduce/radix/fp8 knobs are bypassed on the reference path",
        }
    if eff.get("forced_all_dense_control"):
        # apply_forced_all_dense() OVERWRITES the production scored top-k for dense rows
        # (seq_len <= top_k) with the logical sweep [0..seq_len-1] — so the final dense selected
        # set is NOT the production top-k (deepseek_v2.py:2631, absorbed_latent.py:apply_forced_all_dense).
        return {
            "path": "forced-all dense diagnostic (production scoring then dense override)",
            "selector_width": "full live dense rows (seq_len<=top_k)",
            "score_reduce": "not used for the final dense selected set",
            "topk": "forced [0..seq_len-1] after production scoring (dense rows seq_len<=top_k)",
            "scoring": "production pre-override only (overridden for dense)",
            "scorer": "raw-dot (scorer_norm=off) — pre-override",
            "head_agg": eff["head_agg"],
            "note": "downstream-isolation control: the scored selection is replaced by the dense sweep",
        }
    return {  # selector_impl == "production": the graph-safe fp8 selector — configured knobs ARE used
        "path": "production (graph-safe, fp8 absorbed)",
        "selector_width": str(eff["selector_width_buckets"]),
        "score_reduce": eff["score_reduce_dtype"],
        "topk": "blocked/radix",
        "scoring": "fp8 absorbed in-register",
        "scorer": "raw-dot (scorer_norm=off)",
        "head_agg": eff["head_agg"],
    }


def _server_args(arm, extra):
    """Full launch command line: COMMON_ARGS + extra, plus the exact --double-sparsity-config serve.sh
    passes for DS arms (so the ledger reconstructs the real launch, not an abbreviated one)."""
    args = (COMMON_ARGS + " " + extra).strip()
    cfg = ds_config_for(arm)
    if cfg is not None:
        args += " --double-sparsity-config '" + json.dumps(cfg, separators=(",", ":")) + "'"
    return args


# arm -> (serve_mode, extra_args, dsa_by_regime, dense_out, sparse_out, dense_serial_out, note)
ARMS = {
    "dsa": dict(mode="dsa", extra="", ds=None, measured_sha=BASE_SHA,
                dense="dsa_batched_dense", sparse="dsa_batched_sparse",
                dense_serial="dsa_serial_dense", sparse_serial="dsa_serial_sparse",
                note="native DSA indexer (DS off) — accuracy target"),
    "dsa_noradix": dict(mode="dsa_noradix", extra="--disable-radix-cache", ds=None, measured_sha=BASE_SHA,
                        dense="dsa_noradix_batched_dense", sparse="dsa_noradix_batched_sparse",
                        note="DSA + radix-cache disabled — output-neutral control"),
    "production_ds": dict(mode="ds", extra="--disable-radix-cache --enable-double-sparsity", measured_sha=BASE_SHA,
                          ds={"dense": [715, 716], "sparse": [2048, 5620]},
                          dense="ds_batched_dense", sparse="ds_batched_sparse",
                          dense_serial="ds_serial_dense",
                          note="table-free DS (scorer_norm=off,head_agg=max,bf16 reduce,radix,W=5120) — the regression"),
    "ref_faithful": dict(mode="ref_faithful", extra="--disable-radix-cache --disable-cuda-graph --enable-double-sparsity",
                         ds={"dense": [714, 714], "sparse": [2048, 5610]}, measured_sha=R1_SHA,
                         dense="ref_faithful_dense", sparse="ref_faithful_sparse",
                         note="faithful raw-dot ceiling: exact fp32, TF32 off, current slot incl (dense selected==seq_len)"),
    "ref_cosine": dict(mode="ref_cosine", extra="--disable-radix-cache --disable-cuda-graph --enable-double-sparsity",
                       ds={"dense": [714, 714], "sparse": [2048, 5610]}, measured_sha=R1_SHA,
                       dense="ref_cosine_dense", sparse="ref_cosine_sparse",
                       note="faithful COSINE ceiling: materialized per-head signature, normalize after gather"),
    "ref_cosine_noinc": dict(mode="ref_cosine_noinc", extra="--disable-radix-cache --disable-cuda-graph --enable-double-sparsity",
                             ds=None, measured_sha=R5_NOINC_SHA, measured_source=R5_NOINC_SOURCE,
                             ac6_leg="current-slot (reference_include_current true->false)",
                             corroboration="evidence/ac6_ref_cosine_noinc_corrob.json",
                             dense="ref_cosine_noinc_dense", sparse="ref_cosine_noinc_sparse",
                             note="AC-6 single-variable bisection arm (R5): cosine with reference_include_current=FALSE — "
                                  "the ONE variable flipped vs ref_cosine (production current-slot exclusion). head_agg=max, "
                                  "exact-fp32, TF32-off all unchanged. reference_cosine_select code unchanged since R1 fea920c06; "
                                  "serve.sh ref_cosine_noinc mode added R5. RESULT: dense 0.940->0.625 (=production 0.620) AND "
                                  "sparse 0.940->0.313 -> current-slot exclusion (H3) is a major culprit in BOTH regimes, not "
                                  "dense-only. Sparse needs BOTH cosine scorer AND current-slot inclusion (see 2x2 in ROOT_CAUSE)."),
    "ds_reduce_fp32": dict(mode="ds_reduce_fp32", extra="--disable-radix-cache --enable-double-sparsity",
                           ds=None, measured_sha=R7_REDUCE_SHA, measured_source=R7_REDUCE_SOURCE,
                           ac6_leg="score_reduce_dtype (bf16->fp32 cross-TP reduce)",
                           corroboration="evidence/ac6_score_reduce_fp32_corrob.json",
                           dense="ds_reduce_fp32_dense", sparse="ds_reduce_fp32_sparse",
                           note="AC-6 leg 7 (R7): production raw-dot with score_reduce_dtype=fp32 (the ONE variable vs "
                                "production bf16-reduce). Reduce dtype is near-selection-neutral (bf16-vs-fp32 median "
                                "Jaccard 0.998, ac6_score_reduce_fp32_corrob.json) -> expect ~production scores; reduce "
                                "is NOT the regression driver."),
    "ds_forced_all": dict(mode="ds_forced_all", extra="--disable-radix-cache --disable-cuda-graph --enable-double-sparsity",
                          ds={"dense": [716, 716]}, dense="ds_forced_all_dense", sparse=None, measured_sha=DIAG0_SHA,
                          note="dense forced-all [0..seq-1] control (incl current); dense-only"),
    "ds_anchor_b1": dict(mode="ds_anchor (ANCHOR_BUDGET=1)", extra="--disable-radix-cache --enable-double-sparsity",
                         ds=None, dense="ds_anchor1_dense", sparse="ds_anchor1_sparse", measured_sha=HARNESS_SHA,
                         note="recency anchor budget=1 (current slot only) on production top-k"),
    "ds_anchor_b64": dict(mode="ds_anchor (ANCHOR_BUDGET=64)", extra="--disable-radix-cache --enable-double-sparsity",
                          ds=None, dense="ds_anchor64_dense", sparse="ds_anchor64_sparse", measured_sha=HARNESS_SHA,
                          note="recency anchor budget=64 on production top-k"),
}

NOT_INSTRUMENTED = ["per_step_length_cap_garbage_counts for ds_anchor_* only — every PRIMARY served DS arm "
                    "is now instrumented: production_ds (evidence/ac4_garbage_counters.json, R15/R16: scored "
                    "selection, current slot EXCLUDED, real garbage 0), the forced-all control "
                    "(forced_all_assertions.json, R14), and BOTH reference arms ref_faithful + ref_cosine "
                    "(evidence/ac4_garbage_counters_ref_*.json, R17: scored selection, current slot INCLUDED, "
                    "real garbage 0). ds_anchor_* are auxiliary anchor-budget controls, not AC-4 core arms"]
# AC-4 per-example sample IDs/order are now instrumented (deterministic stock loader, re-derived):
SAMPLE_IDS_ARTIFACT = "evidence/gsm8k_sample_ids.json"
# AC-2.1 forced-all dense physical-slot + slot-validity assertions (R14) — also the AC-4 garbage counters
# for the forced-all control (real garbage 0; the only unwritten slot is the current decode slot = H3).
FORCED_ALL_ASSERT_ARTIFACT = "evidence/forced_all_assertions.json"
# AC-2.4 NIAH recall-oracle@2048 corroboration (R18): per-regime recall@2048 for the production DS scorer.
RECALL_ORACLE_ARTIFACT = "evidence/ac2_4_recall_oracle.json"
# AC-3.1 captured-row materialized fp32 K_label selected-index equality (R20): absorbed raw-dot == materialized
# K_label top-2048 on REAL captured decode rows (supersedes the synthetic ac3_1_materialized_k.json).
MATERIALIZED_K_ARTIFACT = "evidence/ac3_1_materialized_k_selected_index_equality.json"


def validate_materialized_k_artifact():
    """Fail closed unless the AC-3.1 captured-row materialized-K equality artifact proves equality on real
    rows in BOTH regimes. The reducer is itself fail-closed (writes only when every captured row's top-2048
    sets match), but the ledger must independently reject an absent / partial / not-all-equal artifact.
    Returns the per-regime summary."""
    p = os.path.join(HERE, MATERIALIZED_K_ARTIFACT)
    assert os.path.exists(p), f"{MATERIALIZED_K_ARTIFACT} missing — run ac3_1_materialized_k_equality.py"
    d = json.load(open(p))
    assert d.get("index_topk") == 2048, f"materialized-K index_topk={d.get('index_topk')!r}, expected 2048"
    assert d.get("source_dir_basename") == ".sglang_ds_matk", (
        f"materialized-K source_dir_basename={d.get('source_dir_basename')!r}, expected '.sglang_ds_matk'")
    assert d.get("all_selected_index_equal") is True, "materialized-K all_selected_index_equal must be true"
    regs = d.get("regimes", {})
    assert set(regs) == {"dense", "sparse"}, f"materialized-K regimes={sorted(regs)}, expected exactly dense+sparse"
    summary = {}
    for reg in ("dense", "sparse"):
        v = regs[reg]
        rows = v.get("rows", 0)
        eq = v.get("selected_index_equal_rows", -1)
        assert rows > 0 and eq == rows, (
            f"materialized-K {reg} selected_index_equal_rows={eq} != rows={rows} (or zero)")
        summary[reg] = {"rows": rows, "selected_index_equal_rows": eq,
                        "max_abs_score_diff": v.get("max_abs_score_diff")}
    return summary


def validate_recall_oracle_artifact():
    """Fail closed unless the AC-2.4 recall-oracle artifact passes the FULL success contract.

    The driver (niah_recall_oracle.py) is itself fail-closed, but the ledger must independently REJECT a
    partial / failure-marker / wrong-source artifact rather than render AC-2.4 present from a nearby JSON
    (the R15/R18 evidence-integrity class). Asserts: arm + corroboration_only label, EXACTLY dense+sparse
    regimes, index_topk==2048, source_oracle_dir_basename==.sglang_ds_oracle, ZERO failure markers, and per
    regime trials_with_records==trials_issued>0, oracle_records==recall_at_2048_records==
    selected_contains_needle_records>0, recall_at_2048==selected_contains_needle_rate (non-null), and a
    non-null server prompt-token sample. Returns the per-regime summary.
    """
    p = os.path.join(HERE, RECALL_ORACLE_ARTIFACT)
    assert os.path.exists(p), f"{RECALL_ORACLE_ARTIFACT} missing — run niah_recall_oracle.py"
    d = json.load(open(p))
    assert d.get("arm") == "production_ds", f"recall-oracle arm={d.get('arm')!r}, expected production_ds"
    assert d.get("corroboration_only") is True, "recall-oracle must be labelled corroboration_only=true"
    assert d.get("index_topk") == 2048, f"recall-oracle index_topk={d.get('index_topk')!r}, expected 2048"
    assert d.get("source_oracle_dir_basename") == ".sglang_ds_oracle", (
        f"recall-oracle source_oracle_dir_basename={d.get('source_oracle_dir_basename')!r}, "
        f"expected '.sglang_ds_oracle'")
    fm = d.get("failure_markers", {}) or {}
    assert sum(fm.values()) == 0, f"recall-oracle has failure markers (ANY is fatal): {fm}"
    regs = d.get("regimes", {})
    assert set(regs) == {"dense", "sparse"}, f"recall-oracle regimes={sorted(regs)}, expected exactly dense+sparse"
    summary = {}
    for reg in ("dense", "sparse"):
        v = regs[reg]
        issuedn = v.get("trials_issued", 0)
        withrec = v.get("trials_with_records", -1)
        recs = v.get("oracle_records", 0)
        rkrec = v.get("recall_at_2048_records", -1)
        selrec = v.get("selected_contains_needle_records", -1)
        r2048 = v.get("recall_at_2048")
        selrate = v.get("selected_contains_needle_rate")
        sample = v.get("server_prompt_tokens_sample", {}) or {}
        assert issuedn > 0 and withrec == issuedn, (
            f"recall-oracle {reg} trials_with_records={withrec} != trials_issued={issuedn} (or zero)")
        assert recs > 0, f"recall-oracle {reg} oracle_records={recs}, expected >0"
        assert rkrec == recs, f"recall-oracle {reg} recall_at_2048_records={rkrec} != oracle_records={recs}"
        assert selrec == recs, f"recall-oracle {reg} selected_contains_needle_records={selrec} != oracle_records={recs}"
        assert r2048 is not None and selrate is not None and r2048 == selrate, (
            f"recall-oracle {reg} recall_at_2048={r2048} != selected_contains_needle_rate={selrate}")
        assert sample and all(t is not None for t in sample.values()), (
            f"recall-oracle {reg} server_prompt_tokens_sample has null/empty entries: {sample}")
        summary[reg] = {"recall_at_2048": r2048, "oracle_records": recs,
                        "selected_contains_needle_rate": selrate}
    return summary
# AC-4 length-cap garbage counters for the SCORED selection of every served DS arm (real garbage 0 in both
# regimes). production_ds EXCLUDES the current decode slot (current_slot_unwritten==0 = the H3 cause from the
# selection side); the reference arms INCLUDE it (reference_include_current=true), so current_slot_unwritten>0.
# Each arm -> (relative artifact path, expected source_dir_basename, current-slot expectation).
#   current_slot: "excluded" => assert ==0 ; "included" => assert >0.
SCORED_GARBAGE_ARTIFACT = "evidence/ac4_garbage_counters.json"
GARBAGE_ARTIFACTS = {
    "production_ds": (SCORED_GARBAGE_ARTIFACT, ".sglang_ds_garbage", "excluded"),
    "ref_faithful": ("evidence/ac4_garbage_counters_ref_faithful.json",
                     ".sglang_ds_ref_faithful_garbage", "included"),
    "ref_cosine": ("evidence/ac4_garbage_counters_ref_cosine.json",
                   ".sglang_ds_ref_cosine_garbage", "included"),
}


def validate_garbage_artifact(arm):
    """Fail closed unless the per-arm SCORED garbage artifact is the REAL scored capture for that arm.

    R15 regressed by committing a forced-all (dense-only, current-slot force-INCLUDED) artifact as if it
    were the production scored result, because the reducer defaulted to the forced-all dir and failed open
    on the missing sparse regime. This guard loads the artifact and refuses to wire it onto the arm unless
    it self-identifies as that arm's scored capture: correct source_dir_basename, both regimes present,
    rows>0, zero real (non-current) garbage in both. The current-slot expectation is arm-specific —
    production EXCLUDES the current slot (count must be 0, the H3 cause from the selection side), the
    reference arms INCLUDE it (count must be >0). Returns the validated dense/sparse summary.
    """
    rel, want_basename, current_slot = GARBAGE_ARTIFACTS[arm]
    p = os.path.join(HERE, rel)
    assert os.path.exists(p), f"{rel} missing — regenerate from {want_basename}"
    d = json.load(open(p))
    assert d.get("arm") == arm, f"garbage artifact arm={d.get('arm')!r}, expected {arm!r}"
    assert d.get("source_dir_basename") == want_basename, (
        f"{arm} garbage source_dir_basename={d.get('source_dir_basename')!r} — must be {want_basename!r} "
        f"(the {arm} scored capture), NOT another arm's / the forced-all control")
    regs = d.get("regimes", {})
    assert set(regs) == {"dense", "sparse"}, f"{arm} garbage regimes={sorted(regs)}, expected dense+sparse"
    summary = {}
    for reg in ("dense", "sparse"):
        v = regs[reg]
        rows = v.get("rows", 0)
        real = v.get("real_garbage_total", -1)
        cur = v.get("current_slot_unwritten (H3 marker; not garbage)", -1)
        assert rows > 0, f"{arm} garbage {reg} rows={rows}, expected >0"
        assert real == 0, f"{arm} garbage {reg} real_garbage_total={real}, expected 0"
        if current_slot == "excluded":
            assert cur == 0, (f"{arm} garbage {reg} current_slot_unwritten={cur}, expected 0 — the production "
                              f"scored selection must EXCLUDE the current slot (H3 from the selection side)")
        else:  # "included"
            assert cur > 0, (f"{arm} garbage {reg} current_slot_unwritten={cur}, expected >0 — the reference "
                             f"selection INCLUDES the current slot (reference_include_current=true)")
        summary[reg] = {"rows": rows, "real_garbage_total": real, "current_slot_unwritten": cur}
    return summary

ledger = []
for arm, a in ARMS.items():
    rec = {
        "arm": arm,
        "measured_git_sha": a["measured_sha"],
        "ledger_generated_from": {
            "head_sha_at_generation": GEN_HEAD,
            "generator_blob_sha": GEN_BLOB,  # content hash of build_ledger.py — commit-independent
            "worktree": GEN_WORKTREE,
            "note": ("generated BEFORE its own commit exists, so head_sha_at_generation is one commit "
                     "behind the commit that contains this file; generator_blob_sha pins the source exactly"),
        },
        "model_path": "/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db",
        "mask_content_sha256": "5c89c516428f379c983461ceb58fb366c0d6cb12733b3f957d98edb5406f7b21",
        "serve_mode": a["mode"],
        "server_args": _server_args(arm, a["extra"]),
        "cuda_graph": "off" if "--disable-cuda-graph" in a["extra"] else "on (piecewise off)",
        "gsm8k": {"temperature": 0, "max_tokens": 512, "api": "completion",
                  "dense_config": "5-shot/200", "sparse_config": "24-shot/150",
                  "sample_ids_artifact": SAMPLE_IDS_ARTIFACT},
        "scores": {
            "dense_batched": score_from_out(a.get("dense")) if a.get("dense") else None,
            "sparse_batched": score_from_out(a.get("sparse")) if a.get("sparse") else None,
            "dense_serial": score_from_out(a.get("dense_serial")) if a.get("dense_serial") else None,
            "sparse_serial": score_from_out(a.get("sparse_serial")) if a.get("sparse_serial") else None,
        },
        "ds_selected_vs_total_by_regime": a["ds"],
        "fields_not_instrumented": NOT_INSTRUMENTED,
        "note": a["note"],
    }
    if a.get("measured_source"):
        rec["measured_source"] = a["measured_source"]
    _cfg = ds_config_for(arm)
    if _cfg is not None:
        rec["ds_config"] = _cfg                              # literal launch JSON (serve.sh)
        rec["effective_ds_config"] = effective_ds_config_for(arm)  # full resolved config OBJECT
        rec["ds_selector_behavior"] = ds_selector_behavior_for(arm)  # what the selector ACTUALLY does (AC-4)
        if effective_ds_config_for(arm).get("forced_all_dense_control"):
            rec["forced_all_assertions_artifact"] = FORCED_ALL_ASSERT_ARTIFACT  # AC-2.1 + AC-4 garbage counters
        if arm in GARBAGE_ARTIFACTS:
            # Load + VALIDATE before wiring — a forced-all/partial/other-arm artifact must not pass as this
            # arm's scored evidence. production_ds R15; reference arms R17 (current slot INCLUDED -> count>0).
            rec["garbage_counters_artifact"] = GARBAGE_ARTIFACTS[arm][0]  # AC-4 scored-selection garbage
            rec["garbage_counters_validated"] = validate_garbage_artifact(arm)  # fail-closed
    if a.get("ac6_leg"):
        rec["ac6_leg"] = a["ac6_leg"]
        rec["corroboration_artifact"] = a.get("corroboration")
    json.dump(rec, open(os.path.join(ARMS_DIR, f"{arm}.json"), "w"), indent=2)
    ledger.append(rec)

# AC-6 corroboration guard: an arm tagged as an AC-6 bisection leg that records a GSM8K score MUST
# point at a corroboration artifact that exists on disk (plan: each measured AC-6 delta is corroborated
# by recall/selected-index/score-rank). Fail loud otherwise — a scores-only AC-6 arm is not AC-6 evidence.
for r in ledger:
    if r.get("ac6_leg") and any(v is not None for v in r["scores"].values()):
        art = r.get("corroboration_artifact")
        assert art and os.path.exists(os.path.join(HERE, art)), (
            f"AC-6 arm {r['arm']} has GSM8K scores but no corroboration artifact on disk "
            f"(corroboration_artifact={art!r})")

# ds_reduce_fp32 is a graph-mode single-variable arm: its recorded metadata must match the actual run
# (Codex R7: the arm JSON wrongly said --disable-cuda-graph). Fail loud if the metadata drifts again.
_rf = next((r for r in ledger if r["arm"] == "ds_reduce_fp32"), None)
if _rf is not None:
    assert "--disable-cuda-graph" not in _rf["server_args"], "ds_reduce_fp32 ran graph-mode; server_args must not contain --disable-cuda-graph"
    assert _rf["cuda_graph"] != "off", f"ds_reduce_fp32 cuda_graph must be graph-enabled, got {_rf['cuda_graph']!r}"
    assert _rf.get("ds_config", {}).get("score_reduce_dtype") == "fp32", "ds_reduce_fp32 ds_config must record score_reduce_dtype=fp32"

# AC-1/AC-4: every DS arm must record its FULL launch config — --double-sparsity-config in server_args
# AND a complete structured ds_config (the abbreviated extras alone cannot reconstruct the DS launch).
_REQUIRED_DS_KEYS = set(DS_BASE)
for r in ledger:
    if "--enable-double-sparsity" in r["server_args"]:
        assert "--double-sparsity-config" in r["server_args"], (
            f"DS arm {r['arm']} server_args missing --double-sparsity-config")
        missing = _REQUIRED_DS_KEYS - set(r.get("ds_config") or {})
        assert not missing, f"DS arm {r['arm']} ds_config missing launch keys: {sorted(missing)}"
        # AC-4 needs the EFFECTIVE config (defaults expanded) with the selector-width/reduce/scorer keys.
        eff = r.get("effective_ds_config") or {}
        eff_missing = [k for k in DS_EFFECTIVE_REQUIRED if k not in eff]
        assert not eff_missing, f"DS arm {r['arm']} effective_ds_config missing AC-4 keys: {eff_missing}"
        # AC-4 behavior surface: a reference_* arm must NOT render the production width/reduce as USED
        # (the reference path bypasses width bucketing + bf16/fp32 cross-TP reduce). Codex R10.
        beh = r.get("ds_selector_behavior") or {}
        if eff.get("selector_impl", "").startswith("reference_"):
            used = f"{beh.get('selector_width','')} {beh.get('score_reduce','')}"
            for bad in ("5120", "bf16"):
                assert bad not in used, (
                    f"reference arm {r['arm']} ds_selector_behavior shows production '{bad}' as used "
                    f"(width/reduce are bypassed on the reference path): {used!r}")
        # AC-4 behavior surface: a forced_all_dense_control arm OVERRIDES the dense scored top-k, so its
        # behavior.topk must reflect the forced sweep, not plain production blocked/radix. Codex R11.
        if eff.get("forced_all_dense_control"):
            topk = beh.get("topk", "")
            assert "forced" in topk and topk != "blocked/radix", (
                f"forced-all arm {r['arm']} ds_selector_behavior.topk must show the forced dense override, "
                f"not plain production top-k: {topk!r}")

# regenerate evidence_table.md from the ledger
lines = ["# Loop 13 — Per-arm GSM8K evidence ledger (AC-1 / AC-4), generated from evidence/meta/arms/*.json",
         "",
         f"ledger generator blob {GEN_BLOB[:12]} (head@gen {GEN_HEAD[:9]}, worktree {GEN_WORKTREE}) · "
         f"per-arm measured_git_sha in each evidence/meta/arms/*.json (baselines @180f6dd6d, R1 ref arms "
         f"@fea920c06) · model GLM-5.1-FP8 · mask sha256 5c89c516… · TP=8 page64 fp8_e4m3 KV seed42 · "
         f"temp0 max_tokens512 completion API",
         "Dense = 5-shot/200 (~716 tok < top_k 2048). Sparse = 24-shot/150 (~5.6k tok > 2048). batched=64 threads.",
         "selected/total: DS selected vs total tokens by regime (— = native DSA / no DS meta).",
         "DS selector behavior: what the selector ACTUALLY uses (ds_selector_behavior; reference_* arms bypass "
         "the production width/reduce/radix/fp8 knobs — full config object in each arm JSON's effective_ds_config).",
         "",
         "| Arm | dense (b) | sparse (b) | dense (serial) | sparse (serial) | DS selected/total (dense; sparse) | DS selector behavior (path·width·reduce·topk·scorer·head-agg) | note |",
         "|---|---|---|---|---|---|---|---|"]
def cell(x): return "—" if x is None else f"{x:.3f}"
def ds_cell(ds):
    if not ds: return "—"
    parts = []
    for k in ("dense", "sparse"):
        if k in ds: parts.append(f"{k} {ds[k][0]}/{ds[k][1]}")
    return "; ".join(parts) if parts else "—"
def beh_cell(r):
    b = r.get("ds_selector_behavior")
    if not b: return "—"
    path = ("prod" if b["path"].startswith("production")
            else "forced-all" if b["path"].startswith("forced-all")
            else "ref")
    return (f"{path} · {b['selector_width']} · {b['score_reduce']} · {b['topk']} · "
            f"{b['scorer']} · {b['head_agg']}")
for r in ledger:
    s = r["scores"]
    lines.append(f"| {r['arm']} | {cell(s['dense_batched'])} | {cell(s['sparse_batched'])} | "
                 f"{cell(s['dense_serial'])} | {cell(s['sparse_serial'])} | {ds_cell(r['ds_selected_vs_total_by_regime'])} | "
                 f"{beh_cell(r)} | {r['note']} |")
lines += ["",
          "Per-example sample IDs/order: evidence/gsm8k_sample_ids.json (deterministic stock loader; all "
          "arms share the identical ordered slice — dense lines [5:205], sparse [24:174]). Per-step "
          "length-cap garbage counters (duplicate/unwritten/-1/out-of-range physical slots + adapter "
          "errors) via _ds_slot_written, per (rank,req,layer,step), on EVERY primary served DS arm — real "
          "(non-current) garbage 0 everywhere: forced-all control evidence/forced_all_assertions.json (R14, "
          "61776 rows; only unwritten = the current decode slot = the H3 marker); PRODUCTION SCORED "
          "evidence/ac4_garbage_counters.json (R15/R16, dense 41808 + sparse 37440, current slot EXCLUDED "
          "from the scored selection = current_slot_unwritten 0 = the H3 cause from the selection side); "
          "REFERENCE arms evidence/ac4_garbage_counters_ref_faithful.json + _ref_cosine.json (R17, dense "
          "41808 + sparse 37440 each, current slot INCLUDED = current_slot_unwritten = rows). The "
          "production-excludes vs reference-includes current-slot contrast pins H3 from both sides; the "
          "adapter+selected-index path is clean (zero dup/-1/out-of-range/adapter garbage) on all arms. "
          "Gate uses the measured batched DSA comparator (0.975/0.973).",
          "",
          "Gate (AC-5, evidence/gate_ac5.md): naive-DS=best(faithful raw-dot, cosine): dense 0.950 (2.5pp), "
          "sparse 0.940 (3.3pp) -> GOOD. Verdict (AC-6 bisection, evidence/ac6_bisection_matrix.json): the "
          "scorer x current-slot 2x2 is measured — sparse 0.94 needs BOTH the cosine scorer AND current-slot "
          "inclusion (cosine+excl=0.313, rawdot+incl=0.013, rawdot+excl=production 0.000); current-slot "
          "exclusion (H3) hurts BOTH regimes (corroborated both regimes, ac6_ref_cosine_noinc_corrob.json). "
          "Per AC-6 leg: scorer+current-slot MEASURED; radix+width RETIRED (AC-2.3); bf16-vs-fp32 score-reduce "
          "MEASURED (ds_reduce_fp32 arm; selection near-neutral, ac6_score_reduce_fp32_corrob.json median "
          "Jaccard 0.998); head aggregation (AC-2.2) MEASURED — within-rank head_agg='max' matched, but "
          "cross-TP (production SUM vs reference per-rank-local) is a second-order <=1.3pp difference "
          "(head_agg_tp_semantics.json); only fp8-absorbed is BLOCKED (no production config for fp32 absorbed "
          "scoring; absorbed_latent_kernel.py scores fp8 in-register; exact-fp32 absorbed only on the "
          "reference path)."]
open(os.path.join(EVID, "evidence_table.md"), "w").write("\n".join(lines) + "\n")

# Single source of truth for provenance: patch run_meta.json's generator fields
# from the SAME GEN_BLOB/GEN_HEAD stamped into the per-arm JSONs, so the per-arm
# JSONs, the table header, and run_meta can never disagree (Codex R4: they did —
# run_meta had a stale blob 1391f0e... while the arms had f8771c7f2...).
# AC-2.4: load + validate the recall-oracle corroboration artifact (fail-closed: both regimes, non-zero
# records) and record its summary at the top level, so the ledger cannot render AC-2.4 present when it isn't.
RECALL_ORACLE_SUMMARY = validate_recall_oracle_artifact()
# AC-3.1: load + validate the captured-row materialized-K equality artifact (fail-closed: both regimes, all
# rows selected-index-equal) before recording it — the synthetic proof alone is not the captured-row claim.
MATERIALIZED_K_SUMMARY = validate_materialized_k_artifact()

RUN_META = os.path.join(EVID, "meta", "run_meta.json")
if os.path.exists(RUN_META):
    rm = json.load(open(RUN_META))
    rm["git_sha_current"] = GEN_HEAD
    rm["ledger_generator_blob_sha"] = GEN_BLOB
    rm["recall_oracle_corroboration"] = {"artifact": RECALL_ORACLE_ARTIFACT, **RECALL_ORACLE_SUMMARY}
    rm["materialized_k_captured_row_equality"] = {"artifact": MATERIALIZED_K_ARTIFACT, **MATERIALIZED_K_SUMMARY}
    json.dump(rm, open(RUN_META, "w"), indent=2)

# Consistency assertion: the generator blob recorded in every per-arm JSON, in the
# table header, and in run_meta.json must be identical. Fail loud otherwise.
table_hdr = open(os.path.join(EVID, "evidence_table.md")).read()
assert GEN_BLOB[:12] in table_hdr, f"table header missing generator blob {GEN_BLOB[:12]}"
for r in ledger:
    arm_blob = json.load(open(os.path.join(ARMS_DIR, f"{r['arm']}.json")))["ledger_generated_from"]["generator_blob_sha"]
    assert arm_blob == GEN_BLOB, f"{r['arm']}.json blob {arm_blob} != generator {GEN_BLOB}"
if os.path.exists(RUN_META):
    rm_blob = json.load(open(RUN_META))["ledger_generator_blob_sha"]
    assert rm_blob == GEN_BLOB, f"run_meta.json blob {rm_blob} != generator {GEN_BLOB}"

print(f"wrote {len(ledger)} per-arm JSONs to evidence/meta/arms/ and regenerated evidence_table.md")
print(f"provenance consistent: generator blob {GEN_BLOB[:12]} in per-arm JSONs + table + run_meta.json")
for r in ledger:
    print(f"  {r['arm']}: dense_b={r['scores']['dense_batched']} sparse_b={r['scores']['sparse_batched']}")
