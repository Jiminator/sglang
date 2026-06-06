# GLM-5.1-FP8 flags-only hill-climb — sweep table

| tag | changed knob(s) | **client_TPS** (select) | tps>=30 | median_itl_ms (xcheck) | mean_tpot_ms | thr/req | p99_ttft_ms | ttft<22s (info) | p99_itl_ms | accept_len | conc | max_conc | completed | errors | max_total_num_tokens | rationale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| combo_baseline | --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mem-fraction-static 0.85 --max-running-requests 64 --chunked-prefill-size 4096 --schedule-policy lpm | 24.08 | False | 15.89 | 41.61 | 19.55 | 15192.3 | True | 304.82 | 3.136 | 60.6 | 84 | 320 | 0 | 300352 | loop2 incumbent combo baseline (loop1 safe winner ~24.3 TPS); fresh-server gate, unprofiled |
