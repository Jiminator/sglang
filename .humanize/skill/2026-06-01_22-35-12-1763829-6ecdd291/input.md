# Ask Codex Input

## Question

You are adjudicating the central decision of "Loop 7" — DeepSeek-V3.2 FP8 double-sparsity (DS) long-context recall R&D in SGLang. The measure-first M0 milestone has produced its evidence on real 8×H200 hardware. Your job: decide the A-vs-B ordering and the oracle-uplift gate, grounded ONLY in this evidence.

## Background (decided)
- DS = an offline channel-mask selector picking a top-2048 budget of KV tokens for a sparse decode kernel; the budget is kernel-locked at index_topk=2048. DSA = the model's trained indexer (100% NIAH recall at every length within the same 2048).
- Plan (development/loop7/refined_plan_v1.md): measure-first oracle → lead Tier-2.B (better non-learned selector within 2048) → pursue Tier-2.A (opt-in adjustable-budget decode, e.g. 4096/8192) ONLY if the oracle shows a wider budget recovers recall. DEC-2: recall floor = recorded+characterized, strict = stretch; "material" must exceed the baseline binomial CI. DEC-4: production-ready bar for landed code. The strategic gate (ds_on_v32_decision.md) had named Tier-2.A as PRIMARY; M0 evidence may supersede that ordering.

## M0 EVIDENCE (real, 8×H200, mem 0.7, int8 compact table, TP=8)

### Served-recall baseline (N=20/length, Clopper–Pearson 95% CI), all SERVED (0 admission failures):
- 1024 words (~1.1K tok, within-budget): 100% [0.832,1.0]
- 1536 words (~1.65K tok, within-budget): 100% [0.832,1.0]
- 4096 words (~4.4K tok): 75% [0.509,0.913]
- 16384 words (~17.5K tok): 5% [0.001,0.249]
- 65536 words (~70K tok): 5% [0.001,0.249]
Correction vs the old "75/5/0": 64K is now SERVED (not the old admission-failure "0%"); within-budget recall is a clean 100% (DS decode sound, within-budget parity with DSA holds).

### Oracle score-only recall@K (the budget-vs-scorer decider)
Measured on the live all-reduced DS token-score tensor (after all_reduce, before top-K): the needle's score rank, and whether budget K would have selected it (score-only, no decode). Needle logical positions token-match-verified against server prompt_tokens. EAGER. N small (4 trials at 4K, 3 at 16K — recording went flaky after; treat as directional but tight):
- 4K: recall@2048 = 0%, recall@4096 = 100%, recall@8192 = 100%. Needle rank min/median/max = 2105/2208/2580 (i.e. just PAST the 2048 budget).
- 16K: recall@2048 = 0%, recall@4096 = 0%, recall@8192 = 0%. Needle rank = 8832/10218/10306 (~= its sequence position; the scorer barely discriminates the needle).
- 64K: oracle records absent (long-seq bug), but the needle is planted at logical position ~35K–41K of ~70K tokens, so no feasible budget (≤8192) recovers it unless the scorer discriminates it (it does not even at 16K).

## Interpretation offered (challenge it)
The gap is REGIME-DEPENDENT: 4K is budget-limited (needle ranks just past 2048; a 4096 budget recovers it 100%), but 16K/64K are scorer-limited (needle ranks ~10K+/of context; no feasible budget helps; only a better selector ranks it higher). So Tier-2.A is a bounded win for moderate lengths, but Tier-2.B is the only lever for the long-context goal.

## Your task — output EXACTLY these sections:
ORACLE_GATE: Is the Tier-2.A oracle-uplift gate (score-only recall@4096/8192 materially > recall@2048, beyond the baseline binomial CI) met? At which length(s)? Is the small N a blocker for a binding call, or is the signal robust enough to act on directionally?
AB_ORDERING: The recommended A-vs-B ordering for Loop 7, with reasoning grounded in the evidence. Does the regime-dependent reading hold, or is there a simpler correct read?
GATE_SUPERSESSION: Does this evidence supersede the strategic gate's "Tier-2.A primary" ordering? State precisely what changed and what the corrected ordering should say (this becomes task20's decision record).
TIER2B_DIRECTION: Given 16K/64K are scorer-limited (needle rank ≈ position — the channel-mask scorer is near-non-discriminating at length), which concrete NON-learned selector change (channel weighting/normalization, head-aggregation, anchor-budget) is most likely to raise the needle's rank, and how should M1 measure it against the baseline CIs?
RISKS: the top risks / what could make this interpretation wrong (e.g. the small N, the eager-vs-graph score path, per-layer aggregation, the 64K oracle gap, chat-template-vs-raw prompt differences between the baseline and oracle runs).
CONFIDENCE: high/medium/low with one line why.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-06-01_22-35-12
- Tool: codex
