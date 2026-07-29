"""Sanitized wire assertion: prove separate reasoning from round N is absent
from round N+1's assistant history, using the real harness replay path."""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import sglang.benchmark.serving as serving  # noqa: E402


def main(router_url: str, model: str, out_path: str) -> int:
    serving.args = type(
        "A",
        (),
        {
            "disable_stream": True,
            "disable_ignore_eos": True,
            "dataset_name": "recovery-agent",
            "cache_report": False,
            "print_requests": False,
            "header": None,
        },
    )()

    captured = []
    inner = serving.ASYNC_REQUEST_FUNCS["sglang-oai-chat"]

    async def recording(request_func_input, pbar=None):
        captured.append(json.loads(json.dumps(request_func_input.prompt)))
        return await inner(request_func_input, pbar=pbar)

    wrapped = serving.wrap_multi_turn_request_func(
        recording, backend="sglang-oai-chat"
    )
    request = serving.RequestFuncInput(
        model=model,
        prompt=[
            "Think step by step, then answer: what is 17*23?",
            "Now double your previous answer.",
        ],
        api_url=f"{router_url}/v1/chat/completions",
        prompt_len=0,
        output_len=200,
        lora_name=None,
        image_data=None,
        extra_request_body={},
        routing_key="wire-probe",
        session_index=0,
    )
    outputs = asyncio.run(wrapped(request))

    round1 = outputs[0]
    reasoning = round1.generated_text[: -len(round1.assistant_replay_text)] if (
        round1.assistant_replay_text
        and round1.generated_text.endswith(round1.assistant_replay_text)
    ) else round1.generated_text
    round2_assistant = [
        m for m in captured[1] if m["role"] == "assistant"
    ]
    assistant_content = round2_assistant[0]["content"] if round2_assistant else None

    def clip(text):
        if text is None:
            return None
        return {
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "chars": len(text),
            "head": text[:120],
        }

    evidence = {
        "round1_reasoning_present": bool(reasoning.strip())
        and reasoning != round1.assistant_replay_text,
        "round1_reasoning": clip(reasoning),
        "round1_content_replay": clip(round1.assistant_replay_text),
        "round2_assistant_content": clip(assistant_content),
        "round2_assistant_equals_content_only": (
            assistant_content == round1.assistant_replay_text
        ),
        "reasoning_absent_from_round2": (
            bool(reasoning.strip())
            and assistant_content is not None
            and reasoning.strip() not in assistant_content
        ),
        "rounds_succeeded": [o.success for o in outputs],
    }
    Path(out_path).write_text(json.dumps(evidence, indent=2))
    ok = (
        all(evidence["rounds_succeeded"])
        and evidence["round1_reasoning_present"]
        and evidence["round2_assistant_equals_content_only"]
        and evidence["reasoning_absent_from_round2"]
    )
    print(json.dumps(evidence, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
