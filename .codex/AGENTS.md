# AGENTS.md

## Purpose
This repository is a performance-sensitive serving system.
Optimize for:
1. correctness
2. numerical safety
3. no unintended latency / throughput / memory regressions
4. minimal, reviewable diffs
5. clear tests and documentation

## General behavior
- Prefer small, auditable changes over broad refactors.
- Do not modify unrelated code.
- Preserve public APIs and user-facing behavior unless explicitly asked to change them.
- Before editing code, first identify the execution path, affected files, and key invariants.
- Be concrete: name exact files, classes, functions, and symbols.
- Separate confirmed facts from likely inferences.
- If something is unclear, say so explicitly rather than guessing.

## Onboarding / code-understanding mode
When asked to explain the repo or a subsystem:
- Build a concrete architecture map, not a vague summary.
- Trace end-to-end request flow through exact files/functions.
- Distinguish control-plane logic from data-plane / hot-path logic.
- Identify subsystem boundaries, ownership, and coupling.
- Call out hot paths, risky files, invariants, and likely failure modes.
- Connect docs to code when relevant.
- End with:
  1. a short summary
  2. the next 3-5 files/functions to inspect
  3. open questions or uncertainties

## Quantization-specific guidance
Quantization-related code is high risk.
When working on quantization:
- Trace config -> model load -> weight conversion/packing -> backend/kernel dispatch -> runtime execution -> tests/benchmarks.
- Treat dtype assumptions, tensor layout, packing format, fallback logic, and backend compatibility as critical.
- Call out any assumptions about hardware support, numerics, tolerances, and serialization/checkpoint format.
- Explicitly identify where quantization crosses subsystem boundaries.
- Do not silently broaden fallback behavior without explaining the correctness and performance implications.

## Implementation mode
Before making changes:
- First produce a brief plan for non-trivial tasks.
- List affected files and invariants to preserve.
- Note risks to correctness, numerical behavior, latency, throughput, and memory.
- Prefer the smallest patch that solves the requested problem.
- Avoid speculative cleanup unless explicitly requested.

When proposing or making a change, include:
1. what changed
2. why it is safe
3. what could still break
4. what tests/checks should be run

## Review mode
When reviewing a diff:
- Look for correctness bugs, numerical issues, silent fallback changes, API drift, missing validation, missing tests, and likely performance regressions.
- Return:
  1. blocking issues
  2. non-blocking suggestions
  3. confidence gaps / what evidence would increase confidence

## Validation
Before considering work done:
- Explain the affected execution path.
- Verify assumptions against the code, not just comments/docs.
- Identify relevant tests; if none exist, say what should be added.
- For performance-sensitive changes, mention what benchmarks or smoke tests are needed.
- Update docs/comments if behavior or assumptions changed.

## Style for responses
Keep answers structured and concise.
Prefer:
- short summary first
- exact files/functions next
- risks / invariants after that
- concrete next steps or checks at the end

## What not to do
- Do not hand-wave about architecture.
- Do not claim behavior without pointing to code.
- Do not make broad refactors for a narrow task.
- Do not edit the same conceptual area in multiple ways when one minimal fix is enough.
- Do not assume performance is unchanged without stating why.

## File walkthrough mode
When asked to walk through a batch of files:
- explain each file’s role in the subsystem
- name exact classes/functions/symbols
- trace execution flow across the batch
- identify invariants, coupling, hot paths, and risky sections
- suggest a reading order
- end with a short summary and comprehension checks