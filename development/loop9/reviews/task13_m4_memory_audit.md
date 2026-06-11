BUDGET_CHECK:
- fp16 signature `scales` sidecar is not actually allocated. Code leaves `scales=None` unless `dtype == int8`; the log text `scales=fp16` is misleading. Recoverable here: `0 GB`, not `0.17 GB`.
- TokenLabelTable 64-token page pad is ~`0.0024 GiB`, negligible.
- `48.72 -> 31.31 GB` is a `17.41 GB` drop. Subtracting KV `8.14` + table `5.29` leaves ~`3.98 GB`, not `6.9 GB`.
- bs64 math correction: `64 * 4608 = 294,912` tokens. KV needs ~`16.88 GB`; table needs ~`10.97 GB`; combined target ~`27.85 GB`. Increment over today’s `13.43 GB` combined is ~`14.42 GB`, not `19.5 GB`.
- Graph scratch is material in the measured M2 boot: capture used the full `1..512` ladder and `17.68 GB`. DS graph state is roughly `14.7 GiB` of that; `scratch_scores + bf16` alone is ~`11.9 GiB`. Capping capture at bs64 would leave ~`0.43 GiB` DS graph state, saving ~`14.2 GiB`.

VERDICT: re-tune

REASONING:
1. Frozen mem `0.7` itself remains KV-pool-capped: current pool is `142,208 / 4608 = 30.9`, so observed admission `29-30` is expected.
2. Steady free memory alone is not enough as a clean answer. If all `13.54 GB` steady avail were magically converted into proportional KV+table, cap is only about bs61, and that would leave no runtime headroom.
3. The recoverable memory is the over-captured decode graph set, not signatures. The served graph captured up to bs512 even though the workload is capped by KV around bs30.
4. A separate measured retune can convert that graph waste into KV+table: set `cuda_graph_max_bs`/explicit `cuda_graph_bs` to cap at `64`, then raise `mem_fraction_static` to about `0.765` (`0.77` as a practical CLI value). Expected admitted batch for 4608 tokens/req: `64` target, possibly `66-67` raw token capacity at `0.77`.
5. This is still effectively a mem~0.81 resident-state point once the DS table is counted: weights `89.24` + target KV `16.88` + target table `10.97` = ~`117.1 GB/rank`. The retune is justified only because graph trimming recovers almost the same amount.

FOLLOW_ONS:
- Record the fp16 `scales=fp16` log wording as misleading; no sidecar exists in fp16 mode.
- Record full-ladder DS graph state as the real memory item for later loops.
- Measure retune as its own op point: `cuda_graph_max_bs=64`, `mem_fraction_static≈0.765-0.77`, verify boot headroom and admitted bs64.
