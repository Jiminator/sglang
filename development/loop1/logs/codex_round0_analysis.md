**Bottleneck**

The sustained-speed bottleneck is decode/verify GPU service time, dominated by the 78-layer DSA+MoE path: DSA index/top-k + MLA decode + routed expert GEMMs + TP collectives. Capacity is tight but sufficient; TTFT has ~9s slack, so prefill is secondary except where large prefill chunks starve decode.

EAGLE is not proven net-negative until spec-off returns. With `accept_length=3.106` on a max-4 draft, it is plausible that verify-batch cost is too high at concurrency 64, but not certain. Decision rule:

- If spec-off `mean_tpot_ms < 42.16`, EAGLE is hurting sustained speed.
- If spec-off is `>45ms`, EAGLE helps throughput but distorts median ITL.
- If within ~5%, prefer spec-off unless median ITL fails, because p99 ITL should be smoother.

**Ordered Sweep**

Use fresh servers. Keep `--max-running-requests 64`; client concurrency is 64.

1. Already running: spec-off  
   `--mem-fraction-static 0.85 --max-running-requests 64`  
   Decisive A/B for true TPOT.

2. DP attention, spec-off  
   `--dp-size 8 --enable-dp-attention --mem-fraction-static 0.85 --max-running-requests 64`  
   Highest non-risk throughput knob. Expect better high-concurrency throughput; TTFT may rise. Note DP divides resolved `chunked_prefill_size` by 8.

3. DP attention + baseline EAGLE  
   `SGLANG_ENABLE_SPEC_V2=1 --dp-size 8 --enable-dp-attention --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mem-fraction-static 0.85 --max-running-requests 64`  
   Tests the official GLM-style DP+MTP path.

4. Shallower EAGLE  
   Same incumbent DP/non-DP setting, but:  
   `--speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3`  
   Expected: worse median ITL than baseline EAGLE, possibly better mean TPOT and p99 ITL.

5. Decode-protection chunking  
   If non-DP incumbent: add `--chunked-prefill-size 4096`.  
   If DP incumbent, default `8192` resolves to `1024`, so skip unless TTFT is bad.  
   Expected: lower decode starvation; spend TTFT slack.

6. Scheduler prefix locality  
   Add `--schedule-policy lpm` to the incumbent.  
   Expected: possible TTFT/cache-locality gain; usually modest TPOT effect.

7. BF16 DSA decode backend probe  
   `--kv-cache-dtype bfloat16 --dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_sparse`  
   Compares Hopper default `fa3` decode against sparse decode. Likely neutral/worse, but one clean low-risk probe.

8. Capacity/headroom probe  
   `--mem-fraction-static 0.90 --cuda-graph-max-bs 64` plus incumbent flags.  
   Expected: helps only if tight KV capacity/admission is a hidden limiter. Watch OOM. `cuda_graph_max_bs` is not a main TPOT knob here.

9. Accuracy-risk rung 1: FP8 KV  
   `--kv-cache-dtype fp8_e4m3 --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv`  
   Flag as accuracy-risk. This is the highest-value remaining decode knob after lower-risk runs.

10. Accuracy-risk rung 2: IndexCache  
   Add:  
   `--json-model-override-args '{"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}'`  
   Flag as accuracy-risk. Targets the DSA indexer cost directly.

Do not spend runs on `eagle_topk>1` in this path; page size is 64 and the DSA/spec path is not a good fit for top-k fanout. Also, `flashmla_auto` is prefill-only, not a decode backend. Page size is effectively forced to 64 for CUDA DSA, so it is not a real performance knob unless logs prove otherwise.

**Metric Trap**

Yes: speculative decoding can improve official median ITL while hurting true sustained speed. Deeper EAGLE, larger draft batches, and higher `stream_interval`-style burst behavior can make median ITL look excellent because accepted tokens arrive in bursts. Always report `mean_tpot_ms`, `output_throughput / 64`, and p99 ITL beside median ITL.
