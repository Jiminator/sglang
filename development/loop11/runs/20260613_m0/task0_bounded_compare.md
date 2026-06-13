# task0 bounded selector-width vs unbounded right-sized (R2)

Bounded = `selector_width_overflow_policy=fail_closed`, `selector_width_buckets=[4608]` → the DS graph captures ONLY the 4608 width (no full 202752-width DS scratch). Unbounded = the R1 right-sized row at the same point (default full_fallback ladder {5120, full}). `ready_GB` delta is the reclaimed full-width DS graph scratch. `ctl_` rows are full_fallback with buckets=[4608] ({4608, full}) — the matched control isolating the full-width drop.

| probe | variant | idx | frac | policy | bs_cap | ready GB | unbounded ready GB | delta GB | smoke |
|---|---|---|---|---|---|---|---|---|---|
| bnd_fp16_off_rs_075 | fp16 | off | 0.75 | fail_closed | 73 | 16.97 | 16.66 | 0.31 | OK:Paris. The city is located on the River  |
| bnd_fp16_off_rs_080 | fp16 | off | 0.80 | fail_closed | 109 | 3.06 | 2.75 | 0.31 | OK:Paris. The city is located on the River  |
| bnd_fp16_off_rs_085 | fp16 | off | 0.85 | fail_closed | 0 | None | None | — | - |
| bnd_fp16_on_rs_075 | fp16 | on | 0.75 | fail_closed | 59 | 19.64 | 19.33 | 0.31 | OK:Paris. The city is located on the River  |
| bnd_fp16_on_rs_080 | fp16 | on | 0.80 | fail_closed | 89 | 7.02 | 6.71 | 0.31 | OK:Paris. The city is located on the River  |
| bnd_fp16_on_rs_085 | fp16 | on | 0.85 | fail_closed | 0 | None | None | — | - |
| bnd_int8_off_rs_075 | int8 | off | 0.75 | fail_closed | 73 | 22.87 | 22.53 | 0.34 | OK:Paris. The city is located on the River  |
| bnd_int8_off_rs_080 | int8 | off | 0.80 | fail_closed | 109 | 11.83 | 11.49 | 0.34 | OK:Paris. The city is located on the River  |
| ctl_int8_off_rs_080 | int8 | off | 0.80 | full_fallback | 109 | 11.49 | 11.49 | 0.0 | OK:Paris. The city is located on the River  |
| bnd_int8_off_rs_085 | int8 | off | 0.85 | fail_closed | 145 | 1.39 | 1.05 | 0.34 | OK:Paris. The city is located on the River  |
| bnd_int8_off_rs_090 | int8 | off | 0.90 | fail_closed | 0 | None | None | — | - |
| bnd_int8_on_rs_075 | int8 | on | 0.75 | fail_closed | 59 | 24.43 | 24.09 | 0.34 | OK:Paris. The city is located on the River  |
| bnd_int8_on_rs_080 | int8 | on | 0.80 | fail_closed | 89 | 14.15 | 13.81 | 0.34 | OK:Paris. The city is located on the River  |
| bnd_int8_on_rs_085 | int8 | on | 0.85 | fail_closed | 118 | 3.72 | 3.38 | 0.34 | OK:Paris. The city is located on the River  |
| bnd_int8_on_rs_090 | int8 | on | 0.90 | fail_closed | 0 | None | None | — | - |
| bnd_tf_off_rs_075 | tf | off | 0.75 | fail_closed | 73 | 29.33 | 29.02 | 0.31 | OK:Paris. The city is located on the River  |
| bnd_tf_off_rs_080 | tf | off | 0.80 | fail_closed | 109 | 21.56 | 21.25 | 0.31 | OK:Paris. The city is located on the River  |
| bnd_tf_off_rs_085 | tf | off | 0.85 | fail_closed | 145 | 13.64 | 13.33 | 0.31 | OK:Paris. The city is located on the River  |
| bnd_tf_off_rs_090 | tf | off | 0.90 | fail_closed | 181 | 5.72 | 5.41 | 0.31 | OK:Paris. The city is located on the River  |
| bnd_tf_off_rs_095 | tf | off | 0.95 | fail_closed | 216 | 0.23 | None | — | - |
| bnd_tf_on_rs_075 | tf | on | 0.75 | fail_closed | 59 | 29.64 | 29.33 | 0.31 | OK:Paris. The city is located on the River  |
| bnd_tf_on_rs_080 | tf | on | 0.80 | fail_closed | 89 | 22.02 | 21.71 | 0.31 | OK:Paris. The city is located on the River  |
| bnd_tf_on_rs_085 | tf | on | 0.85 | fail_closed | 118 | 14.25 | 13.94 | 0.31 | OK:Paris. The city is located on the River  |
| bnd_tf_on_rs_090 | tf | on | 0.90 | fail_closed | 147 | 6.63 | 6.32 | 0.31 | OK:Paris. The city is located on the River  |
| bnd_tf_on_rs_095 | tf | on | 0.95 | fail_closed | 176 | 1.14 | None | — | OK:Paris. The city is located on the River  |
| bnd_tf_on_rs_096 | tf | on | 0.96 | fail_closed | 0 | None | None | — | - |
