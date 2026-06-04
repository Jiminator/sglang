**1. TTFT Puzzle**
Spec-off has faster honest per-token decode, but much slower slot turnover. At conc 64, no speculation means each request sits through ~512 decode scheduler iterations. With EAGLE accept length 3.106, it is closer to ~165 verify iterations. Even if each verify is expensive, requests finish and free slots earlier, cold prefills get admitted earlier, and burst emission lowers official median ITL. So spec-on can worsen `mean_tpot_ms` while improving p99 TTFT.

**2. Ceiling**
To hit `mean_tpot_ms < 33`, you need ~22% lower than spec-on `42.16`, or ~13% lower than honest spec-off `37.74`. My prior: BF16 KV + TP8 + no EP probably does not reach 33 flags-only. DP attention is the one clean knob that might move enough. If DP attention plus lighter EAGLE still lands above ~35-36 ms, stop chasing scheduler flags; the ceiling is decode/MoE compute. FP8 KV / IndexCache may close the gap only if DSA attention/indexer, not MoE, is a large share.

**3. Ordered Candidates**
1. Keep testing: `--enable-dp-attention --dp-size 8` on spec-on base. Set `SGLANG_ENABLE_SPEC_V2=1` explicitly and keep `--speculative-eagle-topk 1`.

2. Try spec verify on decode backend: `--speculative-attention-mode decode --dsa-decode-backend fa3 --kv-cache-dtype bfloat16`. Current default spec mode is `prefill`, so this may reduce verify overhead.

3. Lighter EAGLE: `--speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3`. This is the best bet to recover mean while still passing median ITL.

4. More aggressive lighter EAGLE: `--speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`. Likely best mean among spec configs, but median ITL may become borderline.

5. Prefix scheduler: `--schedule-policy lpm`. Worth testing with 55% shared prefix; expected gain is from less cold prefill pressure, not faster decode.

6. Chunk retune: non-DP try `--chunked-prefill-size 4096`. With DP attention, note SGLang divides by `dp_size`, so if default DP run has too much prefill overhead, try `--chunked-prefill-size 16384` instead. Smaller chunks protect decode tails but usually do not improve sustained mean.

7. Accuracy-risk ladder, only after above: first `--kv-cache-dtype fp8_e4m3` and leave DSA backends auto, or explicitly `--dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv`. Then IndexCache. Then raised `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`.

Backend calls: on BF16 KV, keep `fa3` decode. Do not test `flashmla_kv` with BF16 KV; the code quantizes cache on the fly. `flashmla_auto` is useful for prefill selection, not decode. Do not chase page-size alternates: DSA/FlashMLA on this path is effectively page size 64. `mem-fraction-static 0.9` is not a speed knob if capacity is already ample.

**4. FP8 KV Rule**
Recommend FP8 KV only as a decode-kernel speed experiment, not for capacity. It is correct here if `--kv-cache-dtype fp8_e4m3` switches Hopper DSA to persistent `flashmla_kv` and measured `mean_tpot_ms` improves while median ITL, p99 TTFT, and accuracy canary are acceptable. It is not correct to force `flashmla_kv` while leaving BF16 KV.
