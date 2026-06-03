ORACLE_GATE:
Met directionally at 4K only. In the 4K oracle runs, recall@2048 = 0/4 while recall@4096 and @8192 = 4/4, with ranks 2105-2580, cleanly just beyond the locked 2048 budget. That is a real oracle uplift signal.

Not met at 16K: recall@2048/@4096/@8192 are all 0/3, with ranks 8832-10306. No 64K oracle record exists, so no formal gate call there.

Small N blocks a production-binding statistical claim, but not the directional engineering call. The 4K rank margins are decisive for those trials; the 16K ranks are also far enough beyond 8192 to treat the <=8192 budget path as unlikely to solve the long-context failure.

AB_ORDERING:
Lead Tier-2.B, then pursue Tier-2.A as a bounded secondary/diagnostic win.

The regime-dependent reading holds: 4K is budget-limited, because the needle is ranked just past 2048 and recovered by 4096. 16K and likely 64K are scorer-limited, because the needle rank is roughly its sequence position, not near the top-K frontier.

The simpler read is: DS decode is sound within budget, but the non-learned selector fails to rank the needle at long context. Tier-2.A helps only when the selector already nearly works. Tier-2.B is the only evidence-backed lever for the long-context objective.

GATE_SUPERSESSION:
Yes, this evidence supersedes the strategic gate’s “Tier-2.A primary” ordering.

What changed: the oracle did not show broad budget recoverability. It showed budget recoverability only at 4K, while 16K remained unrecovered even at 8192. The corrected ordering for task20 should say: Tier-2.B is Loop 7 primary for long-context recall; Tier-2.A is justified only as an opt-in moderate-context improvement or measurement aid, not as the main path to 16K/64K recall.

TIER2B_DIRECTION:
Most likely first non-learned selector change: length-normalized/channel-normalized scoring before top-K, with head/layer aggregation audited separately.

Reason: at 16K the needle rank ~= position, which suggests the scorer is dominated by positional/background magnitude rather than needle salience. A normalization pass that reduces length/position/channel scale bias is more likely to move the needle upward than simply adding budget.

M1 should measure served recall and score-rank recall on the same prompt path, reporting recall@2048 plus needle rank distributions at 4K/16K/64K. Success should be judged against the M0 baseline CIs: especially 16K and 64K must exceed the 5% baseline CI upper bound of 24.9% to count as material, with rank movement showing the selector, not decode luck, caused the gain.

RISKS:
Small oracle N may overstate regime separation.

EAGER score path may differ from graph/runtime path.

Per-layer or per-head aggregation may hide a layer where the needle is discriminative.

64K oracle data is absent, so the 64K scorer-limited conclusion is inferred from 16K plus position, not directly measured.

Baseline and oracle prompt paths may differ: chat-template vs raw prompt, tokenization, or needle placement could affect ranks.

Score-only recall does not prove decoded answer recall unless served decode with the wider/changed selector is measured.

CONFIDENCE:
Medium-high: the 4K and 16K rank evidence is internally strong, but N is small and 64K oracle data is missing.
