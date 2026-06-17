# results_r2/ is SUPERSEDED by ../results_r3/

R2's per-trial DS evidence carried a mislabeled `total_tokens_mean` (the aggregate was derived as
`selected_tokens / sparsity_rate` instead of the true sequence length; reported ~3588 vs the correct
~4770). The SLO verdict numbers (decode-TPS / P99 TTFT) and the comparator rc=3 were unaffected, but the
no-op aggregate was wrong, so this directory is retired.

The PUBLISHED verdict evidence is **`../results_r3/`** (HEAD `8df44a59c`): explicit `total_tokens` field,
all 6 DS trials `trial_evidence.py` PASS with a strengthened consistency gate, both comparators rc=3,
verdict reproduced (DS PASS@16/32, FAIL@64). See `../results_r3/REPRODUCE.md`.

The R2 raw `.gz`/sidecars were removed from the tree to keep one current evidence state; they remain in git
history if ever needed.
