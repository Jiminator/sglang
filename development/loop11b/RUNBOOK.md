# GLM-5.1-FP8 Double Sparsity — Operator Runbook (table-free DS)

Zero-to-serving runbook for a **table-free Double Sparsity (DS)** server on
**GLM-5.1-FP8**, 8×H200, TP=8, fp8_e4m3 KV, page 64. This is the standalone DS
path (`--enable-double-sparsity`); the shipped default for this op-point remains
native DSA.

Op-point (locked): GLM-5.1-FP8, TP=8, fp8_e4m3 KV, page 64, **table-free** DS,
radix-on via a content-hash radix fixture, `mem_fraction_static` **0.8 (DS)** /
**0.85 (DSA)**, `max_running_requests=64`, `cuda_graph_max_bs=64`.

> **Table-free:** the absorbed-latent selector reads the resident MLA latent
> directly. There is **no per-rank TokenLabelTable**, so DS serves at mem 0.8
> (the ~6.8 GB/rank the table used to consume is freed to the KV pool). The
> offline channel mask is still required — table-free removed the per-token
> table, NOT the mask.

---

## 0. Prerequisites

- The GLM-5.1-FP8 snapshot present at
  `/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db`.
- One free TP=8 (8-GPU) node. **One TP=8 server at a time** — two do not fit on
  one 8-GPU node.
- The client bar is `development/SLOS.md` (renamed from the old `CLIENT_SLOS.md`).

---

## (a) Calibrate the channel mask (loop8 DEC-3 recipe)

The GLM-native recipe is **`--dtype fp8_e4m3 --label-dim 32`** (NOT the
DeepSeek-V3.2 values `bfloat16` / `16` — `label_dim 32` is proportionate to
GLM's wider `qk_nope_head_dim` and the recall gate requires it).

Corpus: a deterministic committed Pile-val corpus, reproducible via
`development/loop11b/build_corpus.py` (in-order stream, recorded sha256).

```bash
# Preflight: FP8 native-sharded placement gate (loads model, confirms Q/K hooks
# fire on every layer, writes NO mask).
bash development/loop11b/run_calibrate.sh dryrun

# Full run: writes /cluster-storage/models/glm51-fp8-channel-mask-s256.safetensors
bash development/loop11b/run_calibrate.sh full
```

- `run_calibrate.sh` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to
  avoid the FP8 mid-load fragmentation OOM. **This is CALIBRATION-ONLY.** It must
  **NEVER** be set for serving — it breaks custom-all-reduce IPC at GLM TP=8
  (`cudaIpcGetMemHandle` does not work on VMM memory). Calibration and serving
  are separate processes, so setting it in the calibration runner is safe.
- Expected mask tensor-content sha256: **`35155ac46ad7…`**. Confirm it matches
  the frozen recall baseline before proceeding.

## (b) Land / confirm the DEC-1 content-hash radix authorization

The launch validator authorizes radix-on by pinning the mask's **tensor-content
sha256** (`channel_mask_content_sha256`), NOT the file path or full-file sha
(DEC-1). This makes the fixture portable across mask file paths with identical
tensor content. Confirm the validator's content-hash authorization path is in
place (`validator.radix_fixture_config_fingerprint` /
`apply_radix_fixture_artifact`) before minting.

## (c) Mint radix-on (DEC-12 probes → fixture)

Boot DS radix-on under the **developer-only** override
`SGLANG_DS_RADIX_OVERRIDE=1` (radix-on without a fixture, needed because the
probes require radix reuse), run the DEC-12 correctness probes, then mint the
fixture.

```bash
# DEC-12 probes (one server at a time, fail-closed). Each boots its own DS server.
bash development/loop11b/runs/20260616_ma/mint/gate_a_recall.sh      # recall@2048 off-vs-on within +/-0.5pp overall + per length
bash development/loop11b/runs/20260616_ma/mint/gate_b_crossrank.sh   # cross-rank selection identity (8 ranks) + no dense fallback
bash development/loop11b/runs/20260616_ma/mint/gate_c_edge.sh        # production-representative-reuse edge probe (page 64)

# Mint the fixture IFF gates A+B+C all PASS (fail-closed; writes nothing otherwise).
# (Verdict paths are the gate scripts' own outputs under .../mint/probes/.)
python development/loop11b/runs/20260616_ma/mint/mint_fixture.py \
  --gate-a      development/loop11b/runs/20260616_ma/mint/probes/gate_a_verdict.json \
  --gate-b-p2on development/loop11b/runs/20260616_ma/mint/probes/gate_b_crossrank/radix_on_verdict.json \
  --gate-b-p4   development/loop11b/runs/20260616_ma/mint/probes/gate_b_no_dense_fallback_verdict.json \
  --gate-c-edge development/loop11b/runs/20260616_ma/mint/probes/gate_c_edge_verdict.json
```

`mint_fixture.py` writes `development/serve_double_sparsity_radix_fixture.json`
(pass `--out` to override) and prints the computed config fingerprint so you can
confirm `channel_mask_content_sha256 == 35155ac4…` before the no-override boot.

> **DEC-12 edge contract:** radix-on is authorized for **production-representative
> reuse** (boundary page-aligned reuse + the partial-page hit at the production
> upper bound, ~63% cached). NEAR-FULL reuse (≥~98% cached) is a documented
> out-of-contract value-affecting characterization (+1.57pp), NOT a gate input;
> production workloads operate at ~55% prefix hit.

## (d) Serve WITHOUT the override (the product mechanism)

```bash
bash development/serve_double_sparsity.sh
```

The launcher passes `--double-sparsity-radix-fixture-artifact` (default: the
minted fixture next to the script) and drops `--disable-radix-cache`. The
validator re-verifies the schema + that the fixture matches THIS boot's config
(fail-closed) and only then permits radix-on. **No env override is set for
serving.**

Confirm authorization via **`/server_info`** (preferred; `/get_server_info` is
the deprecated legacy alias):

```bash
curl -s http://127.0.0.1:30000/server_info | python3 -m json.tool | \
  grep -E 'double_sparsity_radix_fixture_artifact|disable_radix_cache'
# expect: artifact set, disable_radix_cache=false
```

## (e) Read capacity from /server_info

```bash
curl -s http://127.0.0.1:30000/server_info | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
print([s['memory_usage']['token_capacity'] for s in d['internal_states']])"
```

Capacity lives at `internal_states[*].memory_usage.token_capacity`. The
validated GLM-5.1-FP8 table-free point serves `max_total_num_tokens` ≈ 504,640
(bs cap 109) at mem 0.8.

---

## Env knobs: PRODUCTION vs DIAGNOSTIC

| Knob | Class | Notes |
|------|-------|-------|
| `MODEL_PATH` | production | GLM-5.1-FP8 snapshot path. |
| `CHANNEL_MASK_PATH` | production | Calibrated mask (sha256 `35155ac4…`). |
| `MEM_FRACTION_STATIC` | production | 0.8 for table-free DS. |
| `TOP_K` | production | 2048 (matches the model's intrinsic index_topk). |
| `RADIX_FIXTURE_ARTIFACT` | production | Minted fixture; `""` serves radix-off. |
| `RECALL_ORACLE` | diagnostic | Fail-closed NIAH recall oracle; forces eager. |
| `LIFTED_BUDGET` (+ `LIFTED_BUDGET_TOP_K`) | diagnostic | Opt-in lifted-budget decode. |
| `SCORER_NORM` | diagnostic | `off` is the only supported production mode. |
| `ANCHOR_MODE` / `ANCHOR_BUDGET` | diagnostic | Non-learned anchor selector variants. |
| `SGLANG_DS_RADIX_OVERRIDE` | diagnostic / dev-only | Radix-on WITHOUT a fixture — fixture-mint runs only. |
| `DISABLE_CUSTOM_ALL_REDUCE` | diagnostic | Degraded NCCL fallback; the AC-11 comparator refuses this op-point for publication. |

**Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving** — it
breaks custom all-reduce at GLM TP=8. (Calibration-only; see step (a).)

---

## Measured SLO posture (loop11b locked sweep, 2026-06-16)

Production-envelope, 2 trials, 600 s window. Client SLO = decode-TPS p50 ≥ 30
**AND** P99 TTFT < 22 s.

| Concurrency | DS decode-TPS p50 | DS P99 TTFT | SLO |
|-------------|-------------------|-------------|-----|
| 16 | 40.75 | 1.59 s | **MEETS** |
| 32 | 34.12 | 3.20 s | **MEETS** |
| 64 | 26.98 | 25.12 s | **FAILS** (TPS < 30 AND TTFT > 22 s) |

Native DSA also misses the 30-TPS floor at conc 64 (26.3) — the throughput floor
is hard for BOTH at high concurrency on this node. The per-step decode tax is
preserved (DS/DSA TPOT ratio 0.974). Full verdict:
`development/loop11b/runs/20260616_mb/DS_absolute_verdict.md`.
