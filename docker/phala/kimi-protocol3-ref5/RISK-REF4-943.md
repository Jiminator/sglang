# Known regression in protocol3-ref4: cached token masks

**Do not deploy protocol3-ref4 or enable production traffic for it.** Its published source tags and artifacts are retained for traceability. Production traffic has not been enabled for this candidate. The earlier CPU and installed-image results did not cover the reuse window described below and do not establish absence of regressions.

## Public synthetic reproduction

The XGrammar p5 dependency has incomplete rule hashes for some recursive grammar states. Reusing a rule-level token-mask cache entry under that incomplete identity can produce a different mask from compiling the same grammar with cache disabled. The affected synthetic controls cover a partial rule's start state, a partial rule referenced by lookahead, and partial hashing with self-reference. They use fabricated grammars and token vocabularies, with no customer schemas, requests or outputs.

Protocol3-ref4's Python lifecycle change permits a completed peer to return before another native compilation finishes. This preserves the intended short-task behavior but exposes an additional interval in which the shared native cache can be reused. In the selected product-path reproduction, the three controls in both compilation orders produced mask mismatches in 6 of 6 ref4 runs, compared with 0 of 6 ref3 runs for that particular interval. This does not establish that ref3 or p5 is generally safe: the underlying native defect also has direct cache-reuse reproductions outside that interval. See [upstream XGrammar issue and fix #943](https://github.com/mlc-ai/xgrammar/pull/943).

## Replacement and serialized caches

Ref5 retains the ref4 product Python repair and replaces p5 with the focused p6 native partial-hash guards. It also changes serialized-cache identity from `v21-phala-unordered-constraints-v2` to `v21-phala-unordered-constraints-v3`. P5 serialized compiled grammars contain precomputed masks; recalculating hash metadata alone cannot repair those masks. Old compiled blobs must be rejected and rebuilt from the original grammar, rather than relabeled or reused.

The replacement requires dedicated cached-versus-uncached mask checks, the actual product lifecycle reproduction, serialization rejection/roundtrip checks, adjacent native controls, and final installed-image qualification. Source preparation or an earlier passing matrix does not substitute for those gates. This notice makes no claim about production payload coverage, GPU performance, complete schema support or worst-case grammar safety.
