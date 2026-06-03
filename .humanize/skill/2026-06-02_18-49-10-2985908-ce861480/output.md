Findings first:

- [m9_tier2a_disposition.md:100](/sgl-workspace/sglang/development/loop7/m9_tier2a_disposition.md:100) still overclaims “graph-captured (R18, 8-rank NCCL + `assert_no_alloc` replay)” and [line 117](/sgl-workspace/sglang/development/loop7/m9_tier2a_disposition.md:117) says “graph-captured TP=8 8-rank selector equality.” That contradicts item 5’s composed-evidence framing at [line 142](/sgl-workspace/sglang/development/loop7/m9_tier2a_disposition.md:142) and the actual test: [TestLiftedWidthSelectionGraphCaptured](/sgl-workspace/sglang/test/registered/unit/layers/attention/test_ds_scorer_tp_determinism.py:210) is single-rank CUDA graph capture; the 8-rank equality test is eager/gloo. Reword those bullets to the exact `(a)+(b)+(c)` evidence.

- Outside m9, stale comments still say lifted is eager-only/forces graph disable: [serve_double_sparsity.sh:81](/sgl-workspace/sglang/development/serve_double_sparsity.sh:81), [selection_kernel.py:457](/sgl-workspace/sglang/python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:457), [dsa_backend.py:482](/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py:482), [dsa_backend.py:2105](/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py:2105). Not runtime blockers, but they undercut the “all eager-required claims removed” claim.

Answers:

1. m9 is no longer internally contradictory on deferred/eager-required production state, but it is still internally contradictory on the TP=8 graph-captured artifact wording above.

2. Yes, `(a)+(b)+(c)` is an acceptable AC-4 production-readiness close if documented as composed evidence, not as a literal standalone 8-rank graph-captured selector-equality unit. I would not require a raw per-rank `torch.cuda.graph` NCCL harness. The feasible stronger artifact would be a production-runner integration probe that dumps/asserts selected-index equality under the real `cuda_graph_runner`, not a naive standalone NCCL unit.

3. No runtime/design gap found blocking AC-4 after the wording fixes. The speculative fail-closed guard looks sound. I did not run the GPU/TP tests; `git diff --check` is clean.
