VERDICT_SOUND: partly

KEY_RISKS
- Dense diagnosis is very strong, but not logically airtight yet. Forced-all bypasses scoring and `_slot_written` filtering for the whole dense row. The saving fact is that production/reference selected 715/716 and the missing one is reported as the current logical slot, so forced-all likely changes exactly one meaningful in-window token.
- H3 as “selection/validity downstream of scorer” is well supported. H3 as “specifically current-slot exclusion” needs a current-only rescue.
- Sparse 0.000 cannot be fully attributed to the same cause from dense-only forced-all. Sparse adds real pruning, so H0 may still coexist there.

MUST_DO_EXPERIMENTS
1. Dense current-only append control: after normal DS selection, append only logical `seq_len - 1` into the first `-1` slot when `seq_len <= top_k`; leave all other scoring, validity, selected positions, `logical_to_physical`, and FlashMLA unchanged. Expected: 0.620 -> ~0.950. This makes the current-slot claim airtight.
2. Sparse current/recency inclusion control: on 24-shot sparse, force-include logical `seq_len - 1` after top-k, evicting lowest selected if width must stay 2048. Better: sweep recency anchor budgets `1/8/32/128`. If accuracy materially recovers, sparse H3 is confirmed; if still 0.000, sparse also has H0/top-k quality issues.
3. Capture invariants for one batch/layer: current logical pos, physical slot, `_slot_written` before score, selected membership, `ds_out` membership, `_slot_written` after KV write. This rules out a mapping/timing artifact.

ALTERNATIVE_EXPLANATIONS
- H1 does not explain dense: fp32 raw-dot exact selector removes fp8 scoring, bf16 reduce, radix top-k, and width bucketing, yet is identical at 0.620.
- H0 does not explain dense: dense `seq_len < top_k` should be a no-op selection regime; forced-all reaches ~DSA. H0 remains possible for sparse after current is fixed.
- H2 broad mask/kernel bug is unlikely: forced-all still uses `logical_to_physical`, `transform_index_page_table_decode`, FlashMLA, and `-1` padding after `seq_len`. A very specific “missing in-window current slot” mask symptom collapses back into H3.
- “Current slot benignly excluded” is weak. Decode causal attention should include the current input token; there is no evident separate current-token attention path, and forced inclusion recovering accuracy is direct evidence that omission is not benign.
