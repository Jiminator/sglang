#!/usr/bin/env python3
"""Enrich task6 + IndexCache profile markdown (AC-3 fields). Reads
logs/rollup_<tag>.txt (category rollup), _work/<tag>_triage.md (top kernels),
sweep_table.md (gate metrics). Pure file assembly. Run after round1_runs.sh."""
import os, re
ROOT = "/sgl-workspace/sglang/development/loop2"
PROF = os.path.join(ROOT, "profiling"); WORK = os.path.join(PROF, "_work")
BASE_TOTAL = 2506.8

NOTES = {
 "t6_fused_moe_sum_ar": ("`--enable-fused-moe-sum-all-reduce`", "23.33",
   "Comms remains a material slice in the profile and total decode time is unchanged vs incumbent — fusing the MoE-sum into the all-reduce does **not** reduce exposed comms/decode time at conc 64 (gate 23.33 ≈ incumbent, slightly worse). The 16.5% all-reduce is critical-path TP8 cost that no flag removes without expert parallelism."),
 "t6_topk_flashinfer": ("`--dsa-topk-backend flashinfer`", "20.15",
   "The indexer/topk path is slower with the flashinfer backend; total decode time rises and gate drops to 20.15 TPS (regression). The default `sgl-kernel` topk backend is the better choice."),
 "t6_contdecode2": ("`--num-continuous-decode-steps 2`", "24.30",
   "Profile is ~identical to incumbent and gate (24.30) is within noise — consistent with the baseline's <1% exposed idle: there is no scheduling/CPU gap for continuous-decode to reclaim."),
 "indexcache_loop2": ("`--json-model-override-args {index_topk_pattern:...}` (IndexCache, **ACCURACY-RISK**)", None,
   "IndexCache reuses the DSA indexer result across layers, cutting decode-path indexer compute (the only knob that moved the binding metric in loop 1). Gate TPS and the DSA-indexer category share vs incumbent are the evidence. **Accuracy-risk: this latency benchmark cannot verify output quality — an accuracy eval must gate any production use.**"),
}

def short(n):
    if "FlashAttnFwdSm90" in n or "flash::" in n: return "FlashAttnFwdSm90 (FA3 attn)"
    if "fused_moe_kernel" in n: return "fused_moe_kernel (MoE)"
    if "allreduce_fusion" in n: return "trtllm_allreduce_fusion (comms)"
    if "all_reduce_two_shot" in n: return "all_reduce_two_shot (comms)"
    if "paged_mqa_logits" in n: return "sm90_fp8_paged_mqa_logits (DSA indexer)"
    if "per_token_group_quant" in n: return "per_token_group_quant_8bit (quantize)"
    if "gatherTopK" in n: return "sbtopk::gatherTopK (topk)"
    if "quantize_k_cache" in n: return "_quantize_k_cache_fast (KV requantize)"
    if "sm90_fp8_gemm" in n: return "deep_gemm sm90_fp8_gemm (GEMM)"
    return (n[:54] + "…") if len(n) > 55 else n

def parse_sweep():
    rows = {}
    for ln in open(os.path.join(ROOT, "sweep_table.md")):
        if not ln.startswith("| "): continue
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) < 16 or c[0] in ("tag",) or c[0].startswith("---"): continue
        rows[c[0]] = dict(tps=c[2], mean_tpot=c[5], p99_ttft=c[7], ttft_sub22=c[8],
                          accept=c[10], conc=c[11], completed=c[13], errors=c[14], maxtok=c[15])
    return rows

def top3(tag):
    p = os.path.join(WORK, f"{tag}_triage.md")
    if not os.path.exists(p): return []
    out, ink = [], False
    for ln in open(p):
        if ln.startswith("Kernel Table"): ink = True; continue
        if ink:
            if ln.startswith("Overlap"): break
            if ln.startswith("| ") and "GPU time" not in ln and not ln.startswith("| ---"):
                c = [x.strip() for x in ln.strip().strip("|").split("|")]
                if len(c) >= 4 and c[2].endswith("ms"): out.append((short(c[0]), c[1], c[2], c[3]))
    return out[:3]

def rollup(tag):
    p = os.path.join(ROOT, "logs", f"rollup_{tag}.txt")
    return open(p).read().strip() if os.path.exists(p) else ""

def total_ms(r):
    m = re.search(r"\(([\d.]+) ms\)", r); return float(m.group(1)) if m else None

def main():
    sweep = parse_sweep()
    for tag, (flag, _tps, note) in NOTES.items():
        g = sweep.get(tag, {})
        r = rollup(tag); t = total_ms(r)
        tms = f"{t:.0f} ms (~{t/BASE_TOTAL:.2f}× incumbent {BASE_TOTAL:.0f} ms)" if t else "n/a"
        topmd = "\n".join(f"{i+1}. {k} — {cat}, {tt} ({sh})" for i,(k,cat,tt,sh) in enumerate(top3(tag))) or "_(triage unavailable)_"
        md = f"""# Decode-phase profile — `{tag}`

**Knob:** {flag}

## Capture method (AC-3.1/3.3/3.4)
- torch profiler, non-scoring profile-only run (`profile_candidate.sh`) replaying conc-64 workload, identical flags. Window `start-step 150 / num-steps 40`, **no `--profile-by-stage`** (DECODE+TARGET_VERIFY+DRAFT_EXTEND grouped). TP-0 analyzed. Raw traces deleted (DEC-4).

## Paired gate result (unprofiled fresh server)
client TPS **{g.get('tps','?')}** | mean_tpot {g.get('mean_tpot','?')} ms | p99_ttft {g.get('p99_ttft','?')} ms (sub-22s={g.get('ttft_sub22','?')}, info) | accept {g.get('accept','?')} | completed {g.get('completed','?')}/err {g.get('errors','?')} | conc {g.get('conc','?')} | max_total_num_tokens {g.get('maxtok','?')}

## Category rollup (summed kernel GPU time, TP-0; total {tms})
```
{r}
```

## Top-3 kernels by GPU time
{topmd}

## Summed vs exposed
CUDA-graph-ON decode (graph replay); single-trace overlap analysis found no kernels above the 1% bar → summed ≈ exposed/critical-path here. TP-0 only.

## Verdict (delta vs incumbent combo_baseline 24.08 TPS)
{note}
"""
        open(os.path.join(PROF, f"{tag}.md"), "w").write(md)
        print(f"wrote {tag}.md (tps={g.get('tps','?')}, total_ms={t})")

if __name__ == "__main__":
    main()
