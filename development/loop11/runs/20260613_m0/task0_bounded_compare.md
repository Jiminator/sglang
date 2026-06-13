# task0 bounded selector-width vs unbounded right-sized (R2)

Bounded = `selector_width_overflow_policy=fail_closed`, `selector_width_buckets=[4608]` → the DS graph captures ONLY the 4608 width (no full 202752-width DS scratch). Unbounded = the R1 right-sized row at the same point (default full_fallback ladder {5120, full}). `ready_GB` delta is the reclaimed full-width DS graph scratch. `ctl_` rows are full_fallback with buckets=[4608] ({4608, full}) — the matched control isolating the full-width drop.

| probe | variant | idx | frac | policy | bs_cap | ready GB | unbounded ready GB | delta GB | smoke |
|---|---|---|---|---|---|---|---|---|---|
| bnd_fp16_on_rs_080 | fp16 | on | 0.80 | fail_closed | 89 | 7.02 | 6.71 | 0.31 | OK:Paris. The city is located on the River  |
| bnd_fp16_on_rs_085 | fp16 | on | 0.85 | fail_closed | 0 | None | None | — | - |
| bnd_int8_off_rs_080 | int8 | off | 0.80 | fail_closed | 109 | 11.83 | 11.49 | 0.34 | OK:Paris. The city is located on the River  |
| ctl_int8_off_rs_080 | int8 | off | 0.80 | full_fallback | 109 | 11.49 | 11.49 | 0.0 | OK:Paris. The city is located on the River  |
| bnd_int8_off_rs_085 | int8 | off | 0.85 | fail_closed | 145 | 1.39 | 1.05 | 0.34 | OK:Paris. The city is located on the River  |
| bnd_tf_off_rs_080 | tf | off | 0.80 | fail_closed | 109 | 21.56 | 21.25 | 0.31 | OK:Paris. The city is located on the River  |
| bnd_tf_off_rs_085 | tf | off | 0.85 | fail_closed | 145 | 13.64 | 13.33 | 0.31 | OK:Paris. The city is located on the River  |
| bnd_tf_off_rs_090 | tf | off | 0.90 | fail_closed | 181 | 5.72 | 5.41 | 0.31 | OK:Paris. The city is located on the River  |
