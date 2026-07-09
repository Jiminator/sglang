#!/usr/bin/bash
#
# OpenHands agentic multi-turn concurrency sweep for SGLang, driven by the
# evalscope perf client (`--dataset swe_smith --multi-turn`).
#
# Per config in CONFIGS: launch an SGLang server (configs/<config>.sh), wait
# for /health, run the evalscope concurrency sweep, tear the server down.
# Everything for a run lands in one timestamped directory:
#   outputs/<ts>/
#     ├── DATASET.openhands            # workload recipe marker
#     ├── <config>/parallel_N_number_M/benchmark_{summary,percentile,args}.json
#     ├── server_logs/<config>.log     # server log (accept-length source)
#     ├── server_logs/<config>.startup # launch->ready seconds sidecar
#     ├── metrics.txt                  # bench_report.py tables
#     └── pareto.png                   # plot_pareto.py chart
#
set -euo pipefail

# Pin the evalscope client so sweep results stay comparable across days.
# Tip: prefix with `PIP_NO_DEPS=1` (pip picks it up from the env) when you
# don't want evalscope[all]'s dep tree (transformers, datasets, modelscope,
# torch tooling, plotly, ...) to downgrade things the inference image has
# pinned — requires the deps to already be present on the image.
EVALSCOPE_COMMIT=acd09b44384d53174768bb1063f675420f76fae9
pip install "evalscope[all] @ git+https://github.com/modelscope/evalscope.git@${EVALSCOPE_COMMIT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_JSON="${SCRIPT_DIR}/openhand-dataset.json"
DATASET_MODEL=nvidia/GLM-5.2-NVFP4

# Reuse a previously-built dataset to skip the multi-minute build loop.
# Delete openhand-dataset.json (or set REBUILD_DATASET=1) to force a rebuild.
# The script also auto-rebuilds when the cached dataset was tokenised by a
# different model or with a different pad source.
#
# Recipe (target ISL=80k / OSL=220 / cache hit=92%):
#   first_turn_length     = 74160 bare tokens  (system pad + real first user msg)
#   subsequent_turn_length= 753   bare tokens  (real + synthetic pad per turn)
#   num_turns             = 13
#
# Every conversation gets a unique synthetic system prompt + per-turn padding,
# so all 67k OpenHands rows qualify regardless of how short the natural
# trajectory is. This ensures --number 128 yields exactly 128 unique
# conversations and the evalscope offset-rotation mechanism keeps each sweep
# step on fresh conversations (no cycling-induced cache-hit inflation).
#
# Pad source for the synthetic filler: 'openscience' (default) fills the
# context with real R1 reasoning traces from nvidia/OpenScienceReasoning-2 — an
# orthogonal domain to OpenHands coding, so it barely perturbs the cache-hit
# rate while avoiding a context window full of gibberish.  'random' selects the
# random-ASCII padder.  Both sources are sized to the *same* exact bare-token
# targets, so the workload shape is identical either way.  Override with
# PAD_SOURCE=random.
: "${PAD_SOURCE:=openscience}"
needs_rebuild() {
    [[ "${REBUILD_DATASET:-0}" == "1" ]] && return 0
    [[ -s "${DATASET_JSON}" ]] || return 0
    local current current_pad
    current=$(python3 -c "import json; print(json.load(open('${DATASET_JSON}')).get('metadata', {}).get('model_path', ''))" 2>/dev/null || echo "")
    current_pad=$(python3 -c "import json; print(json.load(open('${DATASET_JSON}')).get('metadata', {}).get('pad_source', 'random'))" 2>/dev/null || echo "")
    if [[ "${current}" != "${DATASET_MODEL}" ]]; then
        echo "Cached dataset tokenised with '${current}', bench targets '${DATASET_MODEL}'; rebuilding."
        return 0
    fi
    if [[ "${current_pad}" != "${PAD_SOURCE}" ]]; then
        echo "Cached dataset pad_source='${current_pad}', bench targets '${PAD_SOURCE}'; rebuilding."
        return 0
    fi
    return 1
}
if needs_rebuild; then
    python3 "${SCRIPT_DIR}/build_openhands_padded_dataset.py" \
        --model "${DATASET_MODEL}" \
        --pad-source "${PAD_SOURCE}" \
        --first-turn-length 74160 \
        --subsequent-turn-length 753 \
        --num-turns 13 \
        --number 128 \
        --output-path "${DATASET_JSON}"
else
    echo "Reusing existing ${DATASET_JSON} ($(wc -c < "${DATASET_JSON}") bytes). Set REBUILD_DATASET=1 to force a rebuild."
fi

# Sweep configs: the non-DP 4-GPU layouts.
CONFIGS=(
    attn_tp4_moe_tp4
    attn_tp4_moe_ep4
)

SERVER_PID=
SERVER_LOG=
: "${PORT:=8002}"

launch_server() {
    local config=$1
    SERVER_LOG="${SERVER_LOG_DIR}/${config}.log"
    PORT="${PORT}" setsid "${SCRIPT_DIR}/configs/${config}.sh" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
}

wait_for_ready() {
    # 1 hour — large NVFP4 checkpoints can take 10+ min to download from cold
    # and another 5–15 min to load + capture CUDA graphs.
    local TIMEOUT=3600
    local START=$SECONDS
    until curl -sf -o /dev/null http://localhost:${PORT}/health; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server died early. Last log lines:" >&2
            tail -100 "$SERVER_LOG" >&2
            return 1
        fi
        if grep -qE "CUDA out of memory|OutOfMemory|RuntimeError|Killed|Traceback" "$SERVER_LOG"; then
            echo "Server hit a fatal error:" >&2
            tail -100 "$SERVER_LOG" >&2
            return 1
        fi
        if (( SECONDS - START > TIMEOUT )); then
            echo "Timeout after ${TIMEOUT}s waiting for server" >&2
            return 1
        fi
        sleep 5
    done
    local elapsed=$((SECONDS - START))
    echo "Server ready after ${elapsed}s"
    # Persist launch->ready wall-clock so the report can tabulate server startup
    # time (bench_report reads the .startup sidecar next to the log).
    printf '%s\n' "${elapsed}" > "${SERVER_LOG%.log}.startup"
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping sglang server (pgid $SERVER_PID)..."
        kill -TERM -"$SERVER_PID" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -"$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
}

wait_for_port_free() {
    local port=${1:-8002}
    local timeout=${2:-90}
    local start=$SECONDS
    while ! python3 -c "import socket; s=socket.socket(); s.bind(('localhost', $port)); s.close()" 2>/dev/null; do
        if (( SECONDS - start > timeout )); then
            echo "Port ${port} still in use after ${timeout}s" >&2
            return 1
        fi
        sleep 1
    done
}

trap stop_server EXIT  # safety net for Ctrl-C / errors

# Preflight: bail out if the port is already in use
wait_for_port_free "$PORT"

SWEEP_TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${SCRIPT_DIR}/outputs/${SWEEP_TS}"
# Server logs go straight into the sweep dir; bench_report sources the true
# spec-decode accept length from them ("accept len:" in the Decode batch
# lines) instead of evalscope's chunk-based value, which is biased under
# concurrency.
SERVER_LOG_DIR="${SWEEP_DIR}/server_logs"
mkdir -p "${SERVER_LOG_DIR}"
echo "Sweep outputs: ${SWEEP_DIR}"

# Drop a marker file so the sweep directory self-identifies as an OpenHands run.
cat > "${SWEEP_DIR}/DATASET.openhands" <<EOF
dataset: openhands
builder: build_openhands_padded_dataset.py
pad_source: ${PAD_SOURCE}
first_turn_length: 74160
subsequent_turn_length: 753
min_turns: 13
max_turns: 13
number: 128
EOF

for CONFIG in "${CONFIGS[@]}"; do
    echo "=== Running $CONFIG ==="
    launch_server "$CONFIG"

    if ! wait_for_ready; then
        stop_server
        exit 1
    fi

    evalscope perf \
        --model "${DATASET_MODEL}" \
        --url http://localhost:${PORT}/v1/chat/completions \
        --api openai \
        --dataset swe_smith \
        --dataset-path "${DATASET_JSON}" \
        --max-tokens 220 \
        --multi-turn \
        --number 4 8 8 16 32 \
        --parallel 1 2 4 8 16 \
        --extra-args '{"ignore_eos": true}' \
        --name $CONFIG \
        --outputs-dir $SWEEP_DIR \
        --no-timestamp

    stop_server
    wait_for_port_free "$PORT"
done

# Post-sweep report: metric tables (server-sourced accept length) + Pareto
# chart. The sweep data is already on disk, so a report hiccup (e.g. missing
# matplotlib) only WARNs.
python3 "${SCRIPT_DIR}/bench_report.py" metrics --ts "${SWEEP_TS}" \
    | tee "${SWEEP_DIR}/metrics.txt" \
    || echo "WARNING: bench_report.py failed — run it manually on ${SWEEP_DIR}" >&2
python3 "${SCRIPT_DIR}/plot_pareto.py" \
    --sweep "${SWEEP_DIR}" \
    --output "${SWEEP_DIR}/pareto.png" \
    || echo "WARNING: plot_pareto.py failed — run it manually on ${SWEEP_DIR}" >&2

echo "DONE. Artifacts in ${SWEEP_DIR}"
