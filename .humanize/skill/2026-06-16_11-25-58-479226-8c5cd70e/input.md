# Ask Codex Input

## Question

CONVERGENCE CONFIRMATION review. Do NOT write code.

Read the REVISED candidate plan `development/loop11b/.plan_candidate_v2.md` and your prior review
`development/loop11b/.codex_pass2_out.md`. Verify each of your pass-2 REQUIRED_CHANGES was correctly
incorporated:
1. calibration preflight + fixed command (--dtype bfloat16 --kv-cache-dtype fp8_e4m3, --dry-run-blocks, corpus/env/version capture, exclusive long-running)
2. recall comparability (frozen GLM-5.1 fp16 baseline kept WITH length/sample comparability, or new served-fp8 baseline)
3. exact /server_info keys (not vague "get_server_info confirms")
4. Lower-bound radix-OFF fixed (radix-on authorization a precondition; radix-OFF diagnostic only)
5. one-server run order defined (paired-by-trial or labeled-unpaired)
6. no-op refusal hardened (dense_fallback_total==0 AND sparse-selection proof; missing fields refuse)
7. measurement task names preserve loop-11 task8/task9
8. AC-UX includes stale benchmark/comparator/CLIENT_SLOS/"independent trials"/mem-0.6 references

Also confirm the six DEC items (DEC-1..DEC-6) are the correct and complete set of human decisions.

Output EXACTLY:
CONVERGED: yes | no
REMAINING_REQUIRED_CHANGES:
- (empty if none; else specific must-fix)
NEW_RISKS:
- (anything genuinely new and material; else "none")
DEC_SET_OK: yes | no  (+ one line if no)

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 3600s
- Timestamp: 2026-06-16_11-25-58
- Tool: codex
