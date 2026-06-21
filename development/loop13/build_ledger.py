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

NOT_INSTRUMENTED = ["per_example_sample_ids_order (run_eval uses a fixed seed-42 gsm8k slice; "
                    "the per-example id list is not emitted by the stock harness)",
                    "per_step_length_cap_garbage_counts (invalid/unwritten/duplicate/out-of-range "
                    "physical slots — requires logical_to_physical adapter instrumentation not built this loop)"]

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
        "server_args": (COMMON_ARGS + " " + a["extra"]).strip(),
        "cuda_graph": "off" if "--disable-cuda-graph" in a["extra"] else "on (piecewise off)",
        "gsm8k": {"temperature": 0, "max_tokens": 512, "api": "completion",
                  "dense_config": "5-shot/200", "sparse_config": "24-shot/150"},
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
    json.dump(rec, open(os.path.join(ARMS_DIR, f"{arm}.json"), "w"), indent=2)
    ledger.append(rec)

# regenerate evidence_table.md from the ledger
lines = ["# Loop 13 — Per-arm GSM8K evidence ledger (AC-1 / AC-4), generated from evidence/meta/arms/*.json",
         "",
         f"ledger generator blob {GEN_BLOB[:12]} (head@gen {GEN_HEAD[:9]}, worktree {GEN_WORKTREE}) · "
         f"per-arm measured_git_sha in each evidence/meta/arms/*.json (baselines @180f6dd6d, R1 ref arms "
         f"@fea920c06) · model GLM-5.1-FP8 · mask sha256 5c89c516… · TP=8 page64 fp8_e4m3 KV seed42 · "
         f"temp0 max_tokens512 completion API",
         "Dense = 5-shot/200 (~716 tok < top_k 2048). Sparse = 24-shot/150 (~5.6k tok > 2048). batched=64 threads.",
         "selected/total: DS selected vs total tokens by regime (— = native DSA / no DS meta).",
         "",
         "| Arm | dense (b) | sparse (b) | dense (serial) | sparse (serial) | DS selected/total (dense; sparse) | note |",
         "|---|---|---|---|---|---|---|"]
def cell(x): return "—" if x is None else f"{x:.3f}"
def ds_cell(ds):
    if not ds: return "—"
    parts = []
    for k in ("dense", "sparse"):
        if k in ds: parts.append(f"{k} {ds[k][0]}/{ds[k][1]}")
    return "; ".join(parts) if parts else "—"
for r in ledger:
    s = r["scores"]
    lines.append(f"| {r['arm']} | {cell(s['dense_batched'])} | {cell(s['sparse_batched'])} | "
                 f"{cell(s['dense_serial'])} | {cell(s['sparse_serial'])} | {ds_cell(r['ds_selected_vs_total_by_regime'])} | {r['note']} |")
lines += ["",
          "Fields not instrumented this loop (listed in each arm JSON, not faked): per-example sample "
          "IDs/order; per-step length-cap garbage counters (invalid/unwritten/duplicate/out-of-range "
          "physical slots). Gate uses the measured batched DSA comparator (0.975/0.973).",
          "",
          "Gate (AC-5, evidence/gate_ac5.md): naive-DS=best(faithful raw-dot, cosine): dense 0.950 (2.5pp), "
          "sparse 0.940 (3.3pp) -> GOOD. Verdict: dense=H3 current-slot; sparse=raw-dot scorer_norm=off lock "
          "(reference-ceiling; production-path bisection pending)."]
open(os.path.join(EVID, "evidence_table.md"), "w").write("\n".join(lines) + "\n")
print(f"wrote {len(ledger)} per-arm JSONs to evidence/meta/arms/ and regenerated evidence_table.md")
for r in ledger:
    print(f"  {r['arm']}: dense_b={r['scores']['dense_batched']} sparse_b={r['scores']['sparse_batched']}")
