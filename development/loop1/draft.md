Task: Hillclimb GLM 5.1 FP8 on a specific workload to meet target using only sglang flags.

Workload and Target: development/CLIENT_SLOS.md

Benchmark Script: development/benchmark.sh
Out-of-scope: Code changes that affect sglang performance, we are testing out-of-box performance

Revelant Skills: .claude/skills/sglang-sota-performance

Starting Point (cookbook):
```
SGLANG_ENABLE_SPEC_V2=1 sglang serve \
  --model-path zai-org/GLM-5.1-FP8 \
  --tp 8 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mem-fraction-static 0.85
```


Relevant and Useful Sources:
- docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx 
- https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.1
- docs/basic_usage/deepseek_v32.md
- docs_new/docs/advanced_features/
- docs_new/docs/advanced_features/hyperparameter_tuning
- https://sgl-project-sglang-93.mintlify.app/optimization/performance-tuning

Notes:
- Assume Fp8 kv cache is on the table.