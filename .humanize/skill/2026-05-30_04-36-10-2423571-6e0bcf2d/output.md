ANCHOR line 9 — "single done-criterion"
<comment>CRITIQUE: MODIFY #1 — AGREE-WITH-MODIFICATION. Lines 7, 73-80, and 123 use "success" for two different states: absolute client SLO pass and directional progress while still missing the SLO. Keep the directional MVP gate if the client asked for it, but name it "progress accepted, not shippable"; reserve "done-criterion"/"shippable" for strict P99 TTFT < 22.0 s and >= 30 TPS/req at all 16/32/64 conc.</comment>

ANCHOR line 17 — "offline selector"
<comment>CRITIQUE: MODIFY #2 — AGREE-WITH-MODIFICATION. The smell is real, but don't reopen DEC-2 by asking whether DS should exist at all; this plan already says DS must meet the SLO as an opt-in while DSA remains default. The actionable critique is narrower: AC-2 should explicitly justify any TokenLabelTable work as the minimum reversible DS-opt-in fix on a DSA-native model whose trained indexer already wins recall, 75/5/0 vs 100. That stops "compress the inferior selector" from silently growing into more Tier-1 architecture than the client SLO needs.</comment>

ANCHOR line 26 — "int8-symmetric signatures at the same label_dim"
<comment>CRITIQUE: MODIFY #3 — AGREE-WITH-MODIFICATION. Int8 is not automatically bad taste; preserving the selector shape may be the lowest-quality-risk lever. The bad part is that DEC-4, Lower Bound, and task3 still hard-code int8 before AC-2 has proven the fixed-point works, even though the plan budgets only about 1.6-1.8x and says the table grows with the KV pool. Make AC-2 authoritative: if int8 does not predict nominal conc-64 admission with headroom at mem_fraction_static=0.8, skip it and select the structural lever directly.</comment>

ANCHOR line 49 — "don't build it twice"
<comment>CRITIQUE: MODIFY #4 — AGREE. This is the core fix to DEC-4/AC-2: "int8 first" must mean "evaluate int8 first in the budget," not "implement int8 first no matter what." AC-2 must be allowed to choose page-level/two-stage immediately when the fixed-point math says int8 cannot restore nominal admission with headroom; otherwise the plan is explicitly scheduling throwaway kernel work.</comment>

ANCHOR line 55 — "within an explicit tolerance"
<comment>CRITIQUE: MODIFY #5 — AGREE-WITH-MODIFICATION. A tolerance menu is not a test. AC-3.1 must name the primary equivalence metric and numeric fail threshold before implementation, then use any secondary score-error distribution only as diagnostics. Without a concrete metric, threshold, and test shape, the negative test "selection divergence beyond tolerance" cannot fail deterministically.</comment>

ANCHOR line 58 — "measure the hot path"
<comment>CRITIQUE: MODIFY #6 — AGREE-WITH-MODIFICATION. The risk is not just CUDA-graph safety; it is spending the only TPS margin. Loop-5 DS decode is 33.9 tok/s against a 30 TPS/req floor, while AC-3 adds scale reads and dequant/scale math inside scoring. Add an early compact-vs-fp16 scoring/decode microbench with a numeric overhead budget tied to that 33.9-to-30 margin, before the full AC-5 sweep.</comment>

ANCHOR line 109 — "what is this doing in this loop?"
<comment>CRITIQUE: MODIFY #7 — AGREE-WITH-MODIFICATION. AC-9 is a correct harness fix, but the plan gives task10 no dependency while the dependency text admits its re-run needs a live server, and Lower Bound makes it required even though it does not restore admission or prove the client SLO. Mark AC-9 as opportunistic hardening after a hardware artifact, or give task10 a dependency on the lifted-server validation it actually needs. Do not let a code-only Loop-5 cleanup consume the next round before AC-3 through AC-5.</comment>

ANCHOR AC-5 line 73 — "median pass with the worst trial disclosed"
<comment>CRITIQUE: NEW. A strict P99 SLO cannot be accepted by "median pass with the worst trial disclosed." Disclosure is useful for analysis, but if any predeclared trial misses P99 TTFT < 22.0 or 30 TPS/req, the strict SLO did not pass for that run. Split the rule: all trials must pass for a hard SLO claim; median-plus-worst-disclosed is only acceptable for directional characterization.</comment>

ANCHOR AC-5 line 77 — "or an explicit attribution unavailable"
<comment>CRITIQUE: NEW. Line 13 says the SLO claim must separate admission-wait from prefill-compute, but AC-5 allows "attribution unavailable" and still treats the run as directionally useful. For this loop's spine, that is not optional: if TTFT still misses, no attribution means you cannot tell whether TokenLabelTable compaction fixed admission or exposed a prefill bottleneck. Make admission/prefill attribution required for directional success; without it, record the run but do not call the spine validated.</comment>

ANCHOR AC-3.1 line 53 — "on a synthetic shape"
<comment>CRITIQUE: NEW. AC-3.1 protects int8 with only a synthetic selected-token equivalence test, while AC-3.2 requires NIAH non-regression for structural changes. That is inconsistent with the risk: int8 changes the scores used by the already-weak DS selector, and DEC-1 says recall is 75/5/0 vs DSA 100 at the same 2048 budget. Require an int8 quality gate on real V3.2/Loop-5 mask data, or an AC-Q/NIAH non-regression artifact, in addition to the synthetic unit test.</comment>

ANCHOR line 272 — "--- Original Design Draft Start ---"
<comment>CRITIQUE: NEW. This file contains two executable-looking plans. The "Original Design Draft" section repeats old ACs, pending decisions, critical-path commands, and hard-done language that contradict the resolved main plan above. In an agent-run loop, stale instructions in the same plan are not harmless context; move the draft to a separate archive or mark every old section as non-authoritative.</comment>

ANCHOR line 142 — "probe development/loop6/probe_64k.json"
<comment>CRITIQUE: NEW. The plan tells the runner to probe development/loop6/probe_64k.json, but that file is not present in development/loop6/, and line 127 says no new fixture/scaffolding code. AC-8 is supposed to be a hardware-run acceptance check, so its input must either already exist or be named as an acceptance artifact to create. Point AC-8 at an existing 64K payload, or explicitly add the JSON fixture to the plan's deliverables.</comment>
