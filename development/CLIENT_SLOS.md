Here are the immediate client requirements:
- Model: zai-org/GLM-5.1 (FP8)
- Inference SLOs: 30 TPS (Total Latency - TTFT / total tokens) with a P99 TTFT of < 22s
- Workload: 4096 ISL, 512 OSL, max-concurrency: 64, Cache hit: ~55%
- Page size: 64 (technically not explicitly listed as a hard requirement, but significantly preferred and implementation should support different page sizes)