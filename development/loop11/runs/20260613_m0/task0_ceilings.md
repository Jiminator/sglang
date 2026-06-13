# task0 boot/capture/smoke ceilings (12-config grid)

Boot ceiling = highest mem_fraction that boots + captures graphs + answers the smoke. **Upper bound on the servable fraction, not the sustained-stable served fraction** (the latter is established on the task4/M2 ladders under real 4096-ISL load). bounded-selector-width axis (q2) UNMEASURED (needs code, not a config knob) — kept queued. rs16k rows are a separate context-length set.

| variant | indexer | envelope | highest PASS (frac/bs/ready GB) | first FAIL (frac) | bs>=64 cleared at |
|---|---|---|---|---|---|
| fp16 | off | def | 0.75 / bs73 / 11.21 | 0.80 | 0.75 |
| fp16 | off | rs | 0.80 / bs109 / 2.75 | 0.85 | 0.75 |
| fp16 | on | def | 0.80 / bs89 / 1.25 | 0.85 | 0.80 |
| fp16 | on | rs | 0.80 / bs89 / 6.71 | 0.85 | 0.80 |
| int8 | off | def | 0.80 / bs109 / 5.84 | 0.85 | 0.75 |
| int8 | off | rs | 0.85 / bs145 / 1.05 | 0.90 | 0.75 |
| int8 | on | def | 0.80 / bs89 / 8.17 | 0.85 | 0.80 |
| int8 | on | rs | 0.85 / bs118 / 3.38 | 0.90 | 0.80 |
| tf | off | def | 0.85 / bs145 / 7.87 | 0.90 | 0.75 |
| tf | off | rs | 0.90 / bs181 / 5.41 | 0.95 | 0.75 |
| tf | on | def | 0.90 / bs147 / 0.87 | 0.95 | 0.80 |
| tf | on | rs | 0.90 / bs147 / 6.32 | 0.95 | 0.80 |
