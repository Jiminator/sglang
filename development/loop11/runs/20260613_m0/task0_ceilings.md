# task0 boot/capture/smoke ceilings (12-config unbounded grid)

Boot ceiling = highest mem_fraction that boots + captures graphs + answers the smoke. **Upper bound on the servable fraction, not the sustained-stable served fraction** (task4/M2 ladders confirm under real 4096-ISL load). These rows use the default full_fallback selector-width ladder ({compact, full}); the bounded selector-width rows are in task0_bounded_compare.md. rs16k = separate context-length set.

| variant | indexer | envelope | highest PASS (frac/bs/ready GB) | first FAIL (frac/reason) | bs>=64 cleared at |
|---|---|---|---|---|---|
| fp16 | off | def | 0.75 / bs73 / 11.21 | 0.80 (graph_capture_oom) | 0.75 |
| fp16 | off | rs | 0.80 / bs109 / 2.75 | 0.85 (cuda_oom) | 0.75 |
| fp16 | on | def | 0.80 / bs89 / 1.25 | 0.85 (cuda_oom) | 0.80 |
| fp16 | on | rs | 0.80 / bs89 / 6.71 | 0.85 (cuda_oom) | 0.80 |
| int8 | off | def | 0.80 / bs109 / 5.84 | 0.85 (graph_capture_oom) | 0.75 |
| int8 | off | rs | 0.85 / bs145 / 1.05 | 0.90 (cuda_oom) | 0.75 |
| int8 | on | def | 0.80 / bs89 / 8.17 | 0.85 (graph_capture_oom) | 0.80 |
| int8 | on | rs | 0.85 / bs118 / 3.38 | 0.90 (cuda_oom) | 0.80 |
| tf | off | def | 0.85 / bs145 / 7.87 | 0.90 (graph_capture_oom) | 0.75 |
| tf | off | rs | 0.90 / bs181 / 5.41 | 0.95 (graph_capture_oom) | 0.75 |
| tf | on | def | 0.90 / bs147 / 0.87 | 0.95 (cuda_oom) | 0.80 |
| tf | on | rs | 0.90 / bs147 / 6.32 | 0.95 (graph_capture_oom) | 0.80 |

## bounded right-sized ceilings (fail_closed [4608], rs envelope)

Same boot/capture/smoke ceiling, with the bounded selector-width feature (no full-width DS graph). Compare ready GB to the `rs` rows above (the unbounded control); the bounded gain is ~0.3 GB — see task0_bounded_compare.md.

| variant | indexer | envelope | highest PASS (frac/bs/ready GB) | first FAIL (frac/reason) | bs>=64 cleared at |
|---|---|---|---|---|---|
| fp16 | off | rs(bounded) | 0.80 / bs109 / 3.06 | 0.85 (cuda_oom) | 0.75 |
| fp16 | on | rs(bounded) | 0.80 / bs89 / 7.02 | 0.85 (cuda_oom) | 0.80 |
| int8 | off | rs(bounded) | 0.85 / bs145 / 1.39 | 0.90 (cuda_oom) | 0.75 |
| int8 | on | rs(bounded) | 0.85 / bs118 / 3.72 | 0.90 (cuda_oom) | 0.80 |
| tf | off | rs(bounded) | 0.90 / bs181 / 5.72 | 0.95 (other:Failed to CUDA calloc 10485760 bytes) | 0.75 |
| tf | on | rs(bounded) | 0.95 / bs176 / 1.14 | 0.96 (graph_capture_oom) | 0.80 |
