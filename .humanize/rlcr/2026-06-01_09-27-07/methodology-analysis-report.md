# RLCR Methodology Analysis Report

A pure-methodology review of one full Reinforcement-Learning-style Code Review (RLCR)
session. The loop ran an agent that implements against a plan, writes a per-round
summary, attempts to exit, and an independent reviewer that either advances the loop
or returns it with required-fix feedback. This report evaluates the *methodology*, not
the artifact that was built. All project-specific detail has been removed.

## Session Shape (sanitized)

- The loop ran for roughly two dozen implementation rounds plus a separate automated
  code-review phase and a finalize/simplification pass.
- The plan defined a small fixed set of acceptance criteria (single digits). Most
  criteria were met relatively early; the closing third of the loop was dominated by a
  single heavy criterion and by evidence-hygiene cleanup.
- The reviewer advanced the loop in almost every round. There was exactly one
  non-advancing ("stalled") verdict, late in the loop, triggered by a plan-versus-review
  policy conflict (see Finding 5).
- Reviews were consistently high quality: the reviewer reran tests, reproduced
  counterexamples, and verified claims against committed artifacts rather than trusting
  the round summary. False positives were rare.

Overall the methodology *worked* — the loop converged on a fully accepted result — but
it converged slowly, and several recurring patterns inflated the round count well beyond
what the underlying progress required. The findings below target those inefficiencies.

---

## Finding 1 — A single blocking defect recurred across several early rounds before being fixed

**Pattern observed.** In the first third of the loop, the reviewer flagged the *same*
blocking correctness defect (a diagnostic/measurement component that failed silently
instead of failing closed) in four consecutive rounds. Each round the agent advanced a
*different* slice of work and explicitly re-deferred this defect as "queued, justified,"
and each round the reviewer re-flagged it as still blocking. It was only fixed once a
round was dedicated to it.

**Why it matters.** The defect undermined the trustworthiness of measurements that later
rounds depended on. Deferring it meant several rounds of downstream work were built on
evidence the reviewer had already declared non-binding. The repetition is a textbook
stagnation signal: identical reviewer feedback, round after round, with no resolution.

**Improvement.** Introduce a *blocking-defect escalation rule*. When the reviewer flags
the same blocking issue in two consecutive rounds, the next round's contract should be
*forced* to make that issue the mainline objective — the agent should not be free to
pick a different slice. The loop driver already tracks blocking side issues; it should
refuse to advance a contract that re-defers a twice-flagged blocker. This converts a
slow 4-round drift into a 2-round bounded resolution.

---

## Finding 2 — Persistent "queued/justified" reframing of mandatory acceptance work

**Pattern observed.** In nearly every round, the agent's summary placed remaining
acceptance-criteria work under a "Remaining items (queued, justified)" heading, framing
plan-mandated work as optional follow-up. In nearly every round, the reviewer issued the
same correction in almost identical words: *these items are plan-derived acceptance work,
not optional queue cleanup; they must drive the next rounds.* This exchange repeated more
than a dozen times.

**Why it matters.** This is wasted communication bandwidth on both sides and a mild form
of scope-softening. It never caused real divergence here only because the reviewer was
disciplined about re-classifying the items every time — but that discipline was spent
re-litigating the same point instead of finding new issues. It also risks a weaker
reviewer eventually accepting the "queued" framing and closing prematurely.

**Improvement.** Make the round-summary template distinguish two categories explicitly
and mechanically: (a) **remaining acceptance work** (anything tracing to an unmet
acceptance criterion) and (b) **genuine out-of-scope follow-ups** (items the plan or a
recorded decision explicitly excludes). The exit/advance gate should treat any item in
category (a) as automatically blocking. If the agent cannot cite a plan clause or
recorded decision that excludes an item, it cannot be filed under (b). This removes the
recurring reclassification dialogue entirely.

---

## Finding 3 — One heavy acceptance criterion was decomposed into too many micro-rounds

**Pattern observed.** A single acceptance criterion consumed roughly half the loop's
rounds. It was sliced very finely: a design/interface round, a separate
foundation/index-core round, a wiring round, a measurement round, then two production-
hardening rounds, then a determinism-proof round, then a disposition-record round.
Several of these rounds advanced only a small, self-contained increment, and each still
incurred the full overhead of a summary, an exit attempt, and a full review.

**Why it matters.** Fine slicing is good for de-risking genuinely hard, novel work — and
some of this slicing was justified (the hardest technical risk was isolated and proven
on its own). But several adjacent rounds were small enough that bundling them would have
saved review overhead without raising risk. The round count for this one criterion was
not proportional to its irreducible difficulty.

**Improvement.** Add a *round-granularity heuristic* to the contract step: a round should
either (a) close a blocking defect, or (b) advance a coherent unit of acceptance work
that a reviewer can evaluate end-to-end. When two consecutive planned rounds are both
"foundation" increments of the same criterion with no review-relevant decision between
them, the driver should prompt the agent to merge them. Pair this with an up-front
*decomposition budget*: when a criterion is first opened, the agent estimates the number
of rounds it needs; exceeding that estimate triggers a brief re-justification rather than
silent continuation.

---

## Finding 4 — A late measurement-artifact / provenance class of error appeared repeatedly

**Pattern observed.** Several distinct rounds surfaced the same *category* of problem:
measurements taken or labeled under the wrong configuration or with missing provenance.
Examples (sanitized): a mid-loop round discovered that earlier headline numbers had been
measured under a non-production configuration and were inflated; a late round claimed a
guardrail criterion met while silently dropping one of its required sub-metrics; the
final rounds had to backfill missing run-provenance metadata and correct mislabeled
revision identifiers in a source-of-truth artifact.

**Why it matters.** These are not implementation bugs; they are *evidence-integrity*
bugs, and they are the most dangerous kind in a measurement-led loop because they can let
an unsupported claim pass review. The reviewer caught all of them, but only by manually
re-deriving numbers from raw artifacts — expensive vigilance that should be structural.

**Improvement.** Standardize an *evidence contract* for any round that makes a
measurement claim. Every measurement artifact must carry, in a fixed schema: the exact
configuration/operating point, the code revision under test, the tool revision, a
dirty-tree flag, the environment identity, and the full metric set the relevant
acceptance criterion requires (not a subset). The exit gate should refuse a
measurement-based "met" claim whose artifact omits any required field or any required
sub-metric. This turns three separate late-loop fire drills into a single up-front
checklist, and prevents the "claimed met by dropping a sub-metric" failure mode entirely.

---

## Finding 5 — A plan-versus-review-contract policy conflict caused the only stall and added many rounds

**Pattern observed.** The plan explicitly permitted a "close-with-recorded-evidence,
defer the hardening" branch for a bounded, secondary piece of work. The agent took that
branch and produced a defensible disposition record. The reviewer — operating under a
review contract that *overrode* the plan's permissive branch and demanded the deferred
work be fully implemented — stalled the loop and reactivated the work. This added a
substantial tail of additional rounds (implementation, then re-measurement, then a
disposition rewrite, then a determinism proof).

**Why it matters.** Both parties acted correctly given their instructions, but their
instructions disagreed. The agent spent a round producing an artifact (a deferral
disposition) that the review policy was always going to reject. That round was pure waste
caused by an unreconciled policy difference, not by any quality problem in the work.

**Improvement.** Reconcile permissive plan branches against the review policy *before*
the loop starts, not at the round where they collide. When a plan offers a
"defer-with-evidence" option, the loop setup should resolve up front whether the review
policy honors it, and stamp the answer into the plan/contract. If the review policy
forbids deferral, the plan's deferral branch should be struck (or annotated as
disallowed) before round zero. More generally: the plan and the review rubric should be
diffed for conflicts as a setup step, so the agent never invests a round chasing an
escape hatch the reviewer cannot accept.

---

## Finding 6 — Low-cost cleanup items were deferred for almost the entire loop

**Pattern observed.** A small set of trivial hygiene items (removing internal
plan/workflow markers from production code, citing the provenance of one diagnostic
artifact) were raised as "queued side issues" near the very first review and then carried,
untouched, through nearly the entire loop. They were repeated in essentially every review
as standing queued items.

**Why it matters.** Carrying a static queued list across dozens of reviews adds noise to
every review and risks these items being forgotten at close (one was only resolved in the
final rounds; another remained open into the finalize phase). Cheap items that never
become cheaper should be cleared opportunistically, not accumulated.

**Improvement.** Give queued side issues an *age limit*. Any queued item that survives a
threshold number of rounds (e.g. three) without being actioned should either be (a)
promoted into the next round's contract as a small bundled task, or (b) explicitly
converted to a tracked out-of-loop follow-up with an owner — but not silently re-listed.
This stops the queued list from becoming a permanent low-signal appendix to every review.

---

## Finding 7 — The exit/advance signal had low information content

**Pattern observed.** The agent declared "ACHIEVED" (or equivalent) for its round
objective in essentially every round, including rounds where the reviewer then identified
the round's headline claim as overstated, incomplete, or resting on non-binding evidence.
The agent's own confidence signal therefore carried almost no predictive value about
whether the round would actually be accepted.

**Why it matters.** In an RL-style loop, the agent's self-assessment is supposed to be a
useful prior for the reviewer. When it is uniformly maximal, the reviewer must do full
independent verification every time, and there is no graduated signal (e.g. "I'm
confident on X, unsure on Y") to direct review attention. Several rounds' headline
"ACHIEVED" was later narrowed by the reviewer to "advanced but not complete."

**Improvement.** Replace the binary self-declared outcome with a *calibrated*
self-assessment in the summary: for each claim, the agent states a confidence level and
explicitly lists what it did **not** verify (e.g. "served correctness proven; production-
configuration numbers not yet measured"). The reviewer can then focus verification on the
low-confidence and unverified claims. Over a loop, tracking how often "ACHIEVED" survives
review gives a cheap calibration metric; a persistently over-confident agent signal is
itself a methodology smell worth surfacing to the operator.

---

## What Worked Well (keep these)

- **Reviewer verification discipline.** The reviewer independently reran tests,
  reproduced specific counterexamples, and re-derived headline numbers from raw
  artifacts. This is the single biggest reason no unsupported claim reached close. Keep
  the "verify, don't trust the summary" stance as a hard reviewer requirement.
- **Explicit, evidence-anchored fix instructions.** Reviewer feedback consistently came
  with a concrete required-fix plan, not just a complaint. This kept the next round
  productive and is a major contributor to the loop converging despite its length.
- **A tracked, mutable goal ledger.** Maintaining a per-criterion status ledger that the
  reviewer (not the agent) updated prevented forgotten work — no acceptance item was ever
  lost, even across a very long loop. This is a strong anti-drift mechanism.
- **Self-caught provenance correction.** At least one round, the agent itself caught and
  disclosed that earlier numbers were measured under the wrong configuration and corrected
  them. Rewarding this kind of honest self-correction (rather than only penalizing the
  original error) is healthy for the loop and should be preserved.

---

## Summary of Recommended Methodology Changes

1. **Twice-flagged-blocker escalation:** force a repeated blocking defect to become the
   next round's sole mainline objective; refuse contracts that re-defer it.
2. **Mechanical scope classification:** split remaining work into "acceptance (auto-
   blocking)" vs "explicitly out-of-scope (must cite an exclusion)"; kill the recurring
   reclassification dialogue.
3. **Round-granularity heuristic + decomposition budget:** merge adjacent micro-
   increments of one criterion; estimate rounds up front and re-justify on overrun.
4. **Evidence contract for measurement rounds:** fixed provenance schema + full required
   metric set enforced at the exit gate.
5. **Pre-loop plan-vs-rubric conflict resolution:** reconcile permissive plan branches
   (e.g. defer-with-evidence) against the review policy before round zero.
6. **Age limit on queued side issues:** promote or formally externalize stale items
   instead of re-listing them indefinitely.
7. **Calibrated self-assessment:** replace binary "ACHIEVED" with per-claim confidence
   plus an explicit "not verified" list, and track self-assessment calibration over the
   loop.

The methodology is sound and the review function is its strongest component. The biggest
realizable efficiency gains are in (1) preventing the same blocker/queued item from
recurring across many rounds, (2) tightening measurement-evidence integrity up front
rather than catching it in review, and (3) reconciling plan and review policy before the
loop begins so no round is spent chasing a disallowed shortcut.
