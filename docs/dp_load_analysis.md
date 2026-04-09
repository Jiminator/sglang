# DP Load Balance Analysis

Issue: https://github.com/sgl-project/sglang/issues/16080

---

## Background

### Data Parallelism & DP-Attention in SGLang

In SGLang, **Data Parallelism (DP)** runs multiple copies of a model on different GPU groups within the same server instance, each handling a subset of requests independently. **DP-Attention** is a variant where DP workers share the MoE (Mixture of Experts) layers via tensor parallelism but have separate attention computations — this is the common setup for large MoE models like DeepSeek.

The **DataParallelController (DPC)** (`data_parallel_controller.py`) sits inside each server instance and is responsible for dispatching incoming requests to individual DP workers (identified by `dp_rank`). It is the **intra-instance** load balancer.

### PD Disaggregation Mode

SGLang supports **Prefill-Decode (PD) disaggregation**: prefill and decode phases run on separate server instances. A prefill instance processes the prompt and produces KV cache, which is then transferred to a decode instance via RDMA/network. This requires coordination:
- The decode side must know **which prefill dp_rank** handled its request, so it can fetch KV cache from the correct worker.
- A **bootstrap server** (lightweight HTTP service) mediates this: it assigns a `bootstrap_room` (a unique ID) per request and coordinates KV transfer connections between prefill and decode workers.

### Request Flow

```
Client
  ↓
Router / Gateway (optional; inter-instance LB: selects which server instance)
  ↓
Tokenizer Manager (tokenization, batching)
  ↓
DPC (intra-instance LB: selects which dp_rank within the instance)
  ↓
Scheduler (per dp_rank: manages batching, KV cache, model execution)
```

The router/gateway layer is optional — clients can send requests directly to a server instance, in which case DPC handles all load balancing internally.

### Routers: Three Flavors

- **sglang built-in Rust gateway** (`sgl-model-gateway/`): ships with SGLang in the repo. Instance-level routing with policies like round_robin, cache_aware, power_of_two. Currently uses the deprecated `/get_load` API.
- **smg** ([lightseekorg/smg](https://github.com/lightseekorg/smg)): an external model gateway (also Rust-based, out of scope). Uses `/v1/loads` for per-dp-rank load data, supports gRPC, has dp-aware routing. Serves as a reference showing the correct usage of SGLang's load API — SGLang's own internal components (DPC, built-in gateway) should follow the same pattern.
- **mini_lb** (`sglang_router/mini_lb.py`): a minimal Python load balancer for debugging PD disaggregation. Not intended for production.

---

## Issue Status Overview

| Task | Progress | PR | Remaining |
|------|----------|-----|-----------|
| Task 1: Fix confusing names | **100%** | #16110, #16195 | — |
| Task 2: Flexible PD LB (approach a + b) | **100%** | #19268, #19762, #19832 | Mechanism complete; see below |
| Task 3: DPC refactor + naming | **~60%** | #16258 | Naming done; `total_tokens` dispatch has bug; default method doesn't match guideline |
| Task 4: External scheduling | **~50%** | #19268, #19832 | Basic routing done; cache-aware not implemented |

### Task 3 Incomplete Details

1. **`total_tokens` dispatch bug**: `DPBudget.dispatch()` only increments `total_requests += 1` after selecting a worker; `total_tokens` remains unchanged. Between piggyback updates, the primary key `total_tokens` goes stale, causing multiple requests to pile onto the same worker. (#21699)
2. **Default method doesn't match issue guideline**:
   - Issue requires: prefill → `total_tokens` or cache aware; decode → `total_requests` (normal) or `total_tokens` (KV limited)
   - Current auto: PD prefill → `follow_bootstrap_room`; others → `round_robin`
   - `total_tokens` / `total_requests` are never used as defaults

### Task 2 Notes

Task 2 is a mechanism problem: how does the decode side learn the prefill dp rank in PD mode, so that the prefill DPC is no longer locked to `follow_bootstrap_room`. Both approaches are complete:
- **Approach a**: Router directly sets `routed_dp_rank` + `disagg_prefill_dp_rank` on both sides (mini_lb already uses this)
- **Approach b**: Prefill DPC assigns rank internally, then registers it to the bootstrap server; decode side resolves via pending queue + `/query_dp_ranks` async query (adds latency but functionally complete)

The pipe is built, but the quality of the LB strategies running through it is a Task 3 problem.

---

## Current DPC Load Balance Methods

`LoadBalanceMethod` enum (`data_parallel_controller.py`):

- `ROUND_ROBIN` — DPC-internal counter, round-robin assignment
- `FOLLOW_BOOTSTRAP_ROOM` — `bootstrap_room % dp_size`, router decides indirectly (PD-mode legacy; effectively redundant now that `routed_dp_rank` exists)
- `TOTAL_REQUESTS` — dispatch to worker with fewest requests
- `TOTAL_TOKENS` — dispatch to worker with fewest tokens (`total_requests` as tie-breaker)

`auto` mode (`server_args.py`): PD prefill → `follow_bootstrap_room`; others → `round_robin`.

All methods first check `maybe_external_dp_rank_routing(req)` — if `routed_dp_rank` is not None, route directly to the specified worker.

---

## PD Mode: Two Scheduling Approaches

### Approach a: Router Directly Assigns DP Rank

Router sets `routed_dp_rank` (separately for prefill and decode) + `disagg_prefill_dp_rank` (tells decode side which prefill rank was used) in `GenerateReqInput`. DPC directly follows.

### Approach b: DPC Assigns Prefill Rank Internally

Router does not set `routed_dp_rank`; prefill DPC uses its own LB strategy. Decode side discovers the prefill dp rank through:

1. During prefill KVSender bootstrap, if `load_balance_method != "follow_bootstrap_room"` and `dp_size > 1`, calls `_register_prefill_dp_rank()` to POST `{bootstrap_room, dp_rank}` to the bootstrap server
2. Bootstrap server maintains a `room_to_dp_rank` mapping
3. Decode side `_resolve_prefill_dp_rank()` tries `disagg_prefill_dp_rank` first, then `follow_bootstrap_room` formula; if neither works, enters pending queue
4. `_resolve_pending_reqs()` batch-queries the bootstrap server via `/query_dp_ranks`

---

## Load Metrics Architecture

### Foundation: Two Functions on the Scheduler

The scheduler has two load computation functions, both in `scheduler_metrics_mixin.py`:

- **`get_load()`** (:865) → returns `GetLoadReqOutput` (5 fields: `dp_rank, num_reqs, num_waiting_reqs, num_tokens, ts_tic`)
- **`get_loads()`** (:894) → returns `GetLoadsReqOutput` (10+ fields: adds `token_usage, gen_throughput, cache_hit_rate, utilization, max_running_requests` + optional sections)

The first half of both functions (computing num_tokens, iterating waiting queues) is **copy-pasted** with no code reuse.

### Four Consumption Paths

```
                          Scheduler
                         /         \
                  get_load()    get_loads()
                  (old,slim)     (new,full)
                  /       \           \
            Piggyback   /get_load    /v1/loads
               |       (deprecated)  (current)
               v           |            |
              DPC      sglang built-in  smg /
           (internal    Rust gateway    external
              LB)                       router
```

**Path 1 — Piggyback (for DPC internal LB)**
1. Scheduler automatically calls `get_load()` on every `process_batch_result` (`scheduler_output_processor_mixin.py:955`)
2. Result is attached to `BatchStrOutput.load` / `BatchTokenIDOutput.load`
3. TokenizerManager receives it (`tokenizer_manager.py:1735-1741`); if `dp_size > 1`, wraps it as `WatchLoadUpdateReq` and sends to DPC
4. DPC `update_budget()` updates `DPBudget` for the corresponding dp_rank's `total_requests` and `total_tokens`
5. Properties: automatic, once per batch, unidirectional, low latency

**Path 2 — `/get_load` HTTP endpoint (deprecated)**
1. External caller hits `GET /get_load`
2. → `tokenizer_manager.get_load()` → sends `GetLoadReqInput` to each scheduler
3. → Each scheduler calls `get_load()`, returns `GetLoadReqOutput`
4. → Returns `List[GetLoadReqOutput]` (per dp rank)
5. sglang's built-in Rust gateway `LoadMonitor` polls this endpoint periodically, sums `num_tokens` across all dp ranks to get instance-level load

**Path 3 — `/v1/loads` HTTP endpoint (current)**
1. External caller hits `GET /v1/loads?include=core`
2. → `tokenizer_manager.get_loads()` → sends `GetLoadsReqInput` to each scheduler
3. → Each scheduler calls `get_loads()`, returns `GetLoadsReqOutput`
4. → Returns JSON with per-dp-rank full data + aggregate
5. smg uses this endpoint to get `token_usage`, `cache_hit_rate`, etc. for LB decisions

**Path 4 — gRPC `GetLoads` RPC (smg only)**
1. smg's `grpc_servicer` starts a gRPC server inside the SGLang process
2. Internally reuses `get_loads()`, returns results via protobuf
3. Same data source as path 3

### Problem: Paths 1 and 2 Use the Old API

| Path | Function Called | Data Structure | Problem |
|---|---|---|---|
| Piggyback → DPC | `get_load()` | `GetLoadReqOutput` | Missing `token_usage`, `cache_hit_rate`, etc. |
| `/get_load` → built-in gateway | `get_load()` | `GetLoadReqOutput` | Deprecated; only extracts `num_tokens` sum |
| `/v1/loads` → smg | `get_loads()` | `GetLoadsReqOutput` | Correct usage |
| gRPC → smg | `get_loads()` | `GetLoadsReqOutput` | Correct usage |

### Goal: Unify on `get_loads()` / `GetLoadsReqOutput`

1. **Remove `get_load()` and `GetLoadReqOutput`**
2. **Piggyback switches to `get_loads(include=["core"])`**; change `BatchStrOutput.load` / `BatchTokenIDOutput.load` field type to `GetLoadsReqOutput`
3. **Change `WatchLoadUpdateReq.loads` type to `List[GetLoadsReqOutput]`**
4. **DPBudget uses `GetLoadsReqOutput` fields** (`token_usage`, `cache_hit_rate`, etc.), laying the groundwork for cache-aware DPC scheduling
5. **Migrate sglang's built-in Rust gateway to `/v1/loads`**
6. **Remove the deprecated `/get_load` endpoint**

---

## Related PRs

### Merged

| PR | Task | Description |
|----|------|-------------|
| #10201 | T3 | Foundation: `WatchLoadUpdateReq`, `DPBudget`, shortest_queue / minimum_tokens strategies |
| #11469 | T3 | Piggyback mechanism: scheduler load piggybacked on batch output |
| #13203 | T3 | Improved `get_load()` accuracy with dynamically maintained `num_waiting_tokens` |
| #13991 | T3 | Fixed `get_load` API, refactored into `scheduler_metrics_mixin.py` |
| #16088 | T4 | Moved PD configuration conflict checks to model gateway |
| #16110 | T1 | Added `FOLLOW_BOOTSTRAP_ROOM` + `auto` mode, removed `decode_round_robin` |
| #16195 | T1 | Deprecated `--prefill-round-robin-balance` argument |
| #16258 | T3 | Unified dispatch logic, renamed → `TOTAL_REQUESTS` / `TOTAL_TOKENS` |
| #19268 | T2 | Fully support external DP dispatch w/ PD-disaggregation mode |
| #19762 | T2 | Fix `routed_dp_rank` boundary validation |
| #19832 | T2 | Support `X-Data-Parallel-Rank` HTTP header for dp rank routing |

### Open

| PR | Description |
|----|-------------|
| #21699 | Fix dispatch imbalance in DP under `total_tokens` load balancing |
| #20435 | Gateway supports dp rank scheduling and scheduling with minimum tokens |

---

## Open Problems

### Load API Not Unified

`get_load()` / `GetLoadReqOutput` and `get_loads()` / `GetLoadsReqOutput` are two independent code paths with duplicated logic. DPC piggyback uses the slim version, missing critical fields like `token_usage`, `cache_hit_rate`. See section above.

### `+1` Heuristic Bug

`DPBudget.dispatch()` only increments `total_requests[target_rank] += 1` after selecting a worker. For `TOTAL_TOKENS` mode this is incorrect — different requests vary widely in token count, so `+= estimated token count` is needed. This is the root cause of the dispatch imbalance reported in #21699.

### Piggyback Latency

Load is only reported on batch output. Under low load or bursty traffic, the interval between piggyback updates can be long, leaving DPC to dispatch blindly using the heuristic.

### No Cache-Aware Scheduling in DPC

DPC does not consider prefix cache hit rate. Once unified on `GetLoadsReqOutput`, DPC will have access to `cache_hit_rate`, enabling cache-aware DPC scheduling.

### `FOLLOW_BOOTSTRAP_ROOM` Redundancy

With `routed_dp_rank` available, the router can compute `bootstrap_room % dp_size` itself and set `routed_dp_rank` directly. `FOLLOW_BOOTSTRAP_ROOM` as a LB method is effectively redundant — it's only kept for backward compatibility with older routers that don't set `routed_dp_rank`. Should be cleaned up long-term.

### Auto Default Doesn't Match Issue Guideline

Issue #16080 Task 3 specifies defaults: prefill → `total_tokens` or cache aware; decode → `total_requests` (normal) or `total_tokens` (KV limited). Current auto: PD prefill → `follow_bootstrap_room`; others → `round_robin`. Neither `total_tokens` nor `total_requests` is used as a default.

### External Routers Still Use Deprecated `data_parallel_rank` Field

Some external routers (e.g., smg) inject `data_parallel_rank` into requests. SGLang has migrated internally to `routed_dp_rank` with auto-migration + DeprecationWarning. SGLang should keep the deprecation shim for backward compatibility; the rename is an external router concern, not ours.

### No Observability or CI Coverage for DPC Load Balancing

There are currently no metrics tracking DPC dispatch distribution, no balancedness metrics, and no CI tests verifying that LB methods produce balanced dispatch. This means LB bugs (like #21699) are only caught in production.

---

## Next Step: DPC Balancedness Metrics & CI Tests

This is the recommended onboarding task for a new contributor. Before improving DPC load balancing, we need the ability to **measure** it.

### 1. DPC Dispatch Metrics (expose via `/v1/loads` or `/metrics`)

DPC should track and expose per-dp-rank dispatch statistics:

- **`dispatched_requests[dp_rank]`** — cumulative requests dispatched to each rank
- **`dispatched_tokens[dp_rank]`** — cumulative estimated tokens dispatched to each rank (requires DPC to know input length at dispatch time)
- **`dispatch_imbalance_ratio`** — `max(load) / avg(load)` across ranks, as a gauge (1.0 = perfect balance)

These should be available via the existing Prometheus `/metrics` endpoint so they can be monitored in production.

### 2. Unit Tests for `DPBudget`

Test `DPBudget.dispatch()` in isolation with simulated piggyback updates:

- **`total_requests` correctness**: simulate N requests with uniform piggyback updates → assert requests are evenly distributed across ranks
- **`total_tokens` correctness**: simulate requests with varying token counts → assert token-based dispatch spreads tokens evenly, not just request counts
- **`+1` heuristic gap**: simulate a burst of requests between two piggyback updates → measure how much the actual distribution deviates from ideal
- **Stale `total_tokens` bug**: simulate `total_tokens` mode with no token increment in heuristic → demonstrate that requests pile onto one rank (reproduces #21699)

### 3. Integration Tests (CI)

Start a real server with `dp_size >= 2`, send a controlled workload, verify balance:

- **Setup**: launch server with `--dp-size 2` (or use DP-Attention with appropriate TP/DP config)
- **Workload**: send N requests with known input lengths (uniform and skewed)
- **Verify**: poll `/v1/loads` after the run, assert per-rank `num_running_reqs` / `num_used_tokens` are within an acceptable imbalance threshold (e.g., max/min < 1.5)
- **Compare methods**: run the same workload with `round_robin`, `total_requests`, `total_tokens` → assert `total_tokens` achieves better token balance than `round_robin` on skewed workloads

### 4. Key Test Scenarios

| Scenario | What it validates |
|----------|-------------------|
| Uniform input length, `round_robin` | Baseline: round_robin should be balanced for uniform load |
| Uniform input length, `total_requests` | Should match round_robin performance |
| Skewed input length (mix of short and long), `round_robin` vs `total_tokens` | `total_tokens` should achieve significantly better token balance |
| Burst of requests (many requests arriving before piggyback update) | Exposes heuristic compensation accuracy |
| External routing (`routed_dp_rank` set) | Verify DPC respects external assignment, bypasses internal LB |

---

## Priority & Caveats

### Recommended Priority Order

| Priority | Item | Rationale |
|----------|------|-----------|
| **P0** | Metrics + CI tests | Can't improve what you can't measure; without this, everything else is blind |
| **P1** | API unification (`get_load` → `get_loads`) | Single code path; DPC and external routers see the same data |
| **P2** | Fix `total_tokens` heuristic | Production workloads are almost always skewed; this bug has high impact on real deployments |
| **P3** | Auto default adjustment | Most users won't manually set LB method; wrong defaults = most users get bad LB out of the box |
| **P4** | Cache-aware DPC scheduling | Nice to have; depends on P1 (need `cache_hit_rate` from unified API) |
| **P4** | `FOLLOW_BOOTSTRAP_ROOM` cleanup | Tech debt; not urgent, only matters for long-term code clarity |

### Caveat: Piggyback Overhead After API Unification

`get_loads()` computes more fields than `get_load()` — `cache_hit_rate`, `gen_throughput`, `utilization`, etc. Piggyback runs on **every batch output** (hot path). After switching piggyback to `get_loads()`, need to profile the additional overhead and confirm it's acceptable. Options if overhead is too high:
- Use a minimal `include` subset for piggyback (e.g., only core fields, skip expensive computations)
- Compute expensive fields (like `gen_throughput`) lazily or at a lower frequency than every batch
- Keep the fast path slim while still using the same data structure (`GetLoadsReqOutput` with some fields left as default/zero)
