# M-A capacity + "mask serves" evidence — DS radix-on (GRAPH), fresh 8×H200

Boot: `capacity_probe.sh` → DS radix-on under `SGLANG_DS_RADIX_OVERRIDE=1` (graph mode, the
production op-point). One TP=8 server. NOTE: this initial capacity probe used the first
calibration (ld16, content `a4be98c4`, later superseded for recall — see provenance.json); the
token_capacity is KV-pool math, INDEPENDENT of the mask content, and was reconfirmed at the
production ld32 mask (`content_sha256=35155ac4…`) on the no-override authorization boot
(`mint/probes/no_override/server_info.json`): identical `token_capacity=504640`.

## /server_info — locked op-point keys (AC-0.2 key set)

| key | value |
|-----|-------|
| model_path | `…/GLM-5.1-FP8/snapshots/f396cf8…` ✓ |
| tp_size | 8 ✓ |
| page_size | 64 ✓ |
| kv_cache_dtype | fp8_e4m3 ✓ |
| enable_double_sparsity | True ✓ |
| double_sparsity_config | table-free, scorer_norm=off, new mask path ✓ |
| disable_radix_cache | False (radix-on) ✓ |
| disable_cuda_graph | False ✓ |
| disable_custom_all_reduce | False ✓ |
| mem_fraction_static | 0.8 ✓ |
| max_running_requests | 64 ✓ |
| cuda_graph_max_bs | 64 ✓ |
| double_sparsity_radix_fixture_artifact | None (dev-override boot; set after mint) |

## Capacity (AC-0.3)

- `internal_states[0].memory_usage.token_capacity` = **504640** — reproduces loop11's reference
  (504640) exactly on the fresh node → derived decode-bs cap ≈109 ≥ 64.
- `internal_states[0].effective_max_running_requests_per_dp` = **64** (conc-64 running-req cap, ≥61).
- serve log: `max_total_num_tokens=504640, … max_running_requests=64, context_len=202752`.
- **CUDA graph capture succeeded** on all 8 TP ranks (`Capture cuda graph end. Time elapsed: 153.28 s`),
  bs set `[1,2,4,8,12,16,24,32,40,48,56,64]`.
- **No TokenLabelTable** allocated (table deleted in loop11; mem 0.8 with headroom).

## Serves (AC-0.1)

- smoke `/generate` "The capital of France is" → `Paris. The city is located on the River Seine…`
  → the regenerated mask serves coherent output.

## Verdict
AC-0.1 (mask serves) ✓ · AC-0.3 (capacity 504640, bs cap ≥64, capture OK, no table, conc-64 cap 64) ✓ ·
AC-0.2 locked keys ✓ except the fixture artifact (set after the DEC-12 mint). DEC-12 recall/cross-rank/
edge probes + the no-override authorization follow under `mint/`.

Raw: `mint/probes/capacity_ds/server_info_ds.json`, `mint/probes/capacity_ds/stage.log`.
