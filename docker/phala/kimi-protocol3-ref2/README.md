# Kimi shared-reference rebinding repair

Immutable extension of protocol3-ref1, retaining its length/type intersection
and buffered tool-progress fixes. This adds a visited-node set to argument
reference rebinding. Union type restriction may reuse nested dictionaries from
the same target; deepcopy preserves those aliases. Rebinding a shared node a
second time tried to resolve its new document-local pointer against the old
root and rejected a valid Schema. Each shared node is now rebound once without
removing union branches or weakening value constraints.

Two synthetic repeated-union cases failed on installed ref1 at the same source
location seen in production. Candidate CPU verification passed 112 tests plus
eight subtests, including object/array targets, single-branch controls, native
grammar/independent validation agreement, existing reference and dialect tests.
Exact customer inputs were not exported or replayed. Final-image and node101
acceptance remain distinct; memory stability and all request failures are not
declared resolved. No dependency, native library or token-loop changes.

Only 101 is authorized for rollout. Formal loading excludes fastsafetensors.
