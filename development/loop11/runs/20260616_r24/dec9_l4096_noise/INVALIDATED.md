# INVALIDATED — radix never engaged the cache

This run reported L4096 radix-on == radix-off (0.0pp, "noise"). **That conclusion is
WRONG and must not be trusted.**

Root cause: the radix-ON sweep here **never hit the cache** — every prefill batch in
`recall/serve_*_on.log` shows `#cached-token: 0` (each NIAH needle has a unique seed →
no shared prefix → no radix reuse). So this run measured radix-OFF *twice* and told us
nothing about radix-on. A byte-identical 0.0pp "equivalence" across a supposedly-different
path was the red flag.

Corrected by:
- `../../20260616_r25/radix_authorization/` — with cache actually engaged, L4096 showed
  −0.849pp at n=20 (reproduced R24).
- `../../20260616_r25/l4096_highn/` — at n=144 with cache PROVEN engaged (per-request
  cached_tokens ∈ [320,640]), L4096 = +0.38pp (within ±0.5pp); the n=20 miss was sampling
  noise.
- `../../20260616_r25/radix_authorization_v2/` — full-sweep (n=128/length, cache proven
  per length) all within ±0.5pp; this is the authorizing recall evidence.

Lesson: prove cache engagement from raw per-request `cached_tokens` before claiming any
radix on/off equivalence; reproducible-on-same-seed is not statistical significance.
