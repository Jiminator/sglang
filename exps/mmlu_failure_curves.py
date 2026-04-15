"""
Failure curve harness for RMSNorm HF-semantics kernel variants.

For each kernel variant, launches a sglang server, runs MMLU with 64 questions
(seed=0, matching TestTransformersFallbackTorchAO::test_mmlu), and records
per-sample correctness in submission order.

Output: per-variant JSON files under exps/mmlu_curves/ with shape:
    {
      "variant": "<name>",
      "score": <final_score>,
      "per_sample": [1 or 0, ...],  # in submission order (same across variants)
    }

Run:  python exps/mmlu_failure_curves.py

Optionally:  python exps/mmlu_failure_curves.py --only jit_scalar,sgl_kernel_hf
"""
import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas
import requests

sys.path.insert(0, "python")

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    ANSWER_PATTERN_MULTICHOICE,
    ChatCompletionSampler,
    format_multichoice_question,
)

REPO_ROOT = Path("/sgl-workspace/sglang").resolve()
OUTPUT_DIR = REPO_ROOT / "exps" / "mmlu_curves"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "baseline_wrong",     # unpatched: sgl_kernel.rmsnorm, fp32 weight multiply (wrong semantics)
    "forward_native",     # Python forward_native (B1-Native)
    "sgl_kernel_hf",      # sgl_kernel.rmsnorm_hf (AOT CUDA, Plan B)
    "jit_cta",            # jit_kernel RMSNormHFKernel (CTA tile::Memory)
    "jit_half",           # jit_kernel RMSNormHFHalfKernel (vectorized half-block)
    "jit_scalar",         # jit_kernel RMSNormHFScalarKernel (scalar + register cache)
]

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PORT = 21000
NUM_EXAMPLES = 64
NUM_THREADS = 32
SERVER_TIMEOUT = 600  # seconds


def wait_for_server(port, timeout):
    start = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def load_mmlu_examples():
    """Load the same 64 questions as the test (seed=0)."""
    # Use locally cached file if present, else fetch
    cache = REPO_ROOT / "exps" / "mmlu_curves" / "mmlu.csv"
    if not cache.exists():
        url = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
        cache.parent.mkdir(parents=True, exist_ok=True)
        df = pandas.read_csv(url, storage_options={"timeout": 30})
        df.to_csv(cache, index=False)
    else:
        df = pandas.read_csv(cache)
    examples = [row.to_dict() for _, row in df.iterrows()]
    examples = random.Random(0).sample(examples, NUM_EXAMPLES)
    return examples


def run_variant(variant: str, examples: list) -> dict:
    print(f"\n=== Running variant: {variant} ===", flush=True)

    env = os.environ.copy()
    env["SGLANG_RMSNORM_HF_VARIANT"] = variant
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["PATH"] = "/usr/local/cuda-12.9/bin:" + env.get("PATH", "")

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--model-impl", "transformers",
        "--torchao-config", "int4wo-128",
        "--random-seed", "1062627396",
    ]
    server = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              preexec_fn=os.setsid)
    try:
        if not wait_for_server(PORT, SERVER_TIMEOUT):
            raise RuntimeError(f"Server did not become healthy for {variant}")
        print(f"  server up, running MMLU...", flush=True)

        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        sampler = ChatCompletionSampler(
            model=MODEL, max_tokens=2048, top_p=1.0,
            base_url=f"http://127.0.0.1:{PORT}/v1", temperature=0.0,
        )

        # Copy of MMLUEval.__call__ fn but returning list of (idx, score) in order.
        def eval_one(idx_row):
            idx, row = idx_row
            prompt = [sampler._pack_message(content=format_multichoice_question(row), role="user")]
            resp = sampler(prompt) or ""
            m = re.search(ANSWER_PATTERN_MULTICHOICE, resp)
            extracted = m.group(1) if m else None
            return idx, (1 if extracted == row["Answer"] else 0)

        t0 = time.perf_counter()
        results = common.map_with_progress(eval_one, list(enumerate(examples)), NUM_THREADS)
        latency = time.perf_counter() - t0

        # pool.imap preserves input order, but sort by idx to be extra-safe
        results.sort(key=lambda x: x[0])
        per_sample = [s for _, s in results]
        score = sum(per_sample) / len(per_sample)
        print(f"  variant={variant}  score={score:.4f}  latency={latency:.1f}s", flush=True)

        return {
            "variant": variant,
            "score": score,
            "latency": latency,
            "per_sample": per_sample,
            "num_examples": len(per_sample),
        }
    finally:
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            server.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            except Exception:
                pass
        time.sleep(3)  # let GPU memory release


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated subset of variants to run")
    args = parser.parse_args()

    variants = VARIANTS if args.only is None else args.only.split(",")
    examples = load_mmlu_examples()
    print(f"Loaded {len(examples)} MMLU examples (seed=0)")

    for v in variants:
        out_path = OUTPUT_DIR / f"{v}.json"
        if out_path.exists():
            print(f"[skip] {v} already exists at {out_path}")
            continue
        result = run_variant(v, examples)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
