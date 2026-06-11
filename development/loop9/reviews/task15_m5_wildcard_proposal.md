**RANKING**

| Rank | Candidate | Expected 10-step window | Impact vs 512.7k | Confidence |
|---:|---|---:|---:|---|
| 1 | **B + C-lite: width-bucketed DS selector graphs, with compact score buffers per bucket** | **~377-395k us** | **-118 to -136k** | Medium |
| 2 | **C standalone: fixed compact live-window cap with full-width overflow path** | ~380-405k us at served op point | -108 to -133k | Medium-low |
| 3 | **A: persistent/bounded-grid logical-score, optionally top-k passes** | ~461-474k us with top-k; ~480-485k logical-only | -29 to -52k | High for logical, medium for top-k |
| 4 | **D: fused score+select with exact two-round reduce** | Benign case maybe ~360-430k; worst case current-or-worse | Unstable | Low |

Baseline used: current total **512.7k us**, with DS reduce **~113.2k** including cast, top-k **36.3k**, logical-score **43.2k**, leaving non-target work at roughly **320k**.

**PROPOSAL**

Recommend the next wildcard loop target **B as the primary redesign**, with the compact score-buffer shape from **C** treated as an implementation detail of each width bucket rather than a separate fixed-cap fallback design. At the served op point, an 8k bucket covers the ~4608 live-token window and changes the expensive DS selector shapes from `[bs, 202752]` to `[bs, 8192]`. The biggest win is the reduce: full-width bf16 custom-AR plus cast is currently ~113.2k/window; at 8k, the 0.46 MB reduce should use one-shot custom-AR, roughly **31-35k/window** after multiplying the ~40 us per-call floor across the decode window and allowing small cast/copy overhead. Logical-score should fall from **43.2k -> ~10-15k**, and top-k from **36.3k -> ~16-24k**. That puts the total around **377-395k**, or **~1.10-1.15x** the 342.9k DSA floor.

If the near-term goal is only to clear the logical-score hard gate, **A** is the smallest reliable patch: it should take logical-score to **~10-15k** with no numerical contract change. But it leaves the dominant full-width reduce untouched, so total remains roughly **1.35x+** DSA. That makes A a useful de-risking slice, not the structural fix.

**RISKS**

For **B / B+C-lite**:
- Requires selector-width bucketing in the CUDA graph path; current runner is effectively bs-bucketed, so this is a real runner integration change.
- Bucket dispatch must come from host-visible scheduler/request metadata, not a device-computed max that forces sync before graph replay.
- All dead logical positions must remain equivalent to `-inf`, and selected outputs must preserve logical positions exactly.
- Multiple captured graph variants increase memory pressure and replay bookkeeping.
- Custom-AR remains out-of-place, so bucketed reduce needs captured copy-back or a consumer contract that reads the reduced compact buffer directly.

For **C standalone**:
- Fixed cap with overflow fallback preserves correctness but doubles maintained selector surface.
- Compact-to-logical inverse mapping is now part of the correctness contract.
- Any overflow branch must be graph-safe and bit-identical, or it violates the selection-recall/identity constraint.

For **D**:
- Exact global top-k over TP-summed scores cannot be guaranteed from bounded local candidate unions.
- A threshold/detection round can make it exact, but worst-case expansion reaches the full vector.
- A “real distributions usually work” margin is approximate and conflicts with cross-rank bit-identity as a hard requirement.

**DISPOSITION**

**needs-user-decision**: choose whether the next loop should pursue the structural **B / B+C-lite** redesign, or first land the smaller **A** slice purely to close the logical-score bucket.
