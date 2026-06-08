# Community 452

> 13 nodes

## Key Concepts

- **grpc_server.py** (6 connections) — `python/sglang/srt/entrypoints/grpc_server.py`
- **serve_grpc()** (6 connections) — `python/sglang/srt/entrypoints/grpc_server.py`
- **_add_metrics_routes()** (3 connections) — `python/sglang/srt/entrypoints/grpc_server.py`
- **set_prometheus_multiproc_dir()** (3 connections) — `python/sglang/srt/utils/common.py`
- **_start_sidecar_server()** (2 connections) — `python/sglang/srt/entrypoints/grpc_server.py`
- **_check_communicator_results()** (2 connections) — `python/sglang/srt/entrypoints/grpc_server.py`
- **_add_admin_routes()** (2 connections) — `python/sglang/srt/entrypoints/grpc_server.py`
- **Thin gRPC server wrapper — delegates to smg-grpc-servicer package.  A lightweigh** (1 connections) — `python/sglang/srt/entrypoints/grpc_server.py`
- **Start the aiohttp sidecar and return the runner for cleanup.** (1 connections) — `python/sglang/srt/entrypoints/grpc_server.py`
- **Add Prometheus /metrics endpoint to the aiohttp app.** (1 connections) — `python/sglang/srt/entrypoints/grpc_server.py`
- **Return a web.Response error if results indicate failure, else None.** (1 connections) — `python/sglang/srt/entrypoints/grpc_server.py`
- **Add admin endpoints to the aiohttp app.      Endpoints: /start_profile, /stop_pr** (1 connections) — `python/sglang/srt/entrypoints/grpc_server.py`
- **Start the standalone gRPC server with integrated scheduler.** (1 connections) — `python/sglang/srt/entrypoints/grpc_server.py`

## Relationships

- [[Community 33]] (2 shared connections)
- [[Community 47]] (1 shared connections)
- [[Community 42]] (1 shared connections)

## Source Files

- `python/sglang/srt/entrypoints/grpc_server.py`
- `python/sglang/srt/utils/common.py`

## Audit Trail

- EXTRACTED: 25 (83%)
- INFERRED: 5 (17%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*