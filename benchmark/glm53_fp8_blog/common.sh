# Shared helpers for the run scripts. Sourced, not executed.
# Callers: the per-panel run scripts under gb300/ and b300/.
#
# Serving code is the SGLang installed in the container (the recipe runs inside
# the lmsysorg/sglang:v0.5.18 docker image) - no worktree or PYTHONPATH swap.

PORT="${PORT:-8002}"
SERVER_PID=""
EVALSCOPE_PIN="evalscope[all] @ git+https://github.com/modelscope/evalscope.git@acd09b44384d53174768bb1063f675420f76fae9"
# The OpenHands dataset (nebius/SWE-rebench-openhands-trajectories) declares the
# `List` feature type, which the dataset build can only read with datasets>=4.0.
# evalscope's own modelscope dependency already requires
# datasets>=4.0.0,<=4.8.4, so pin to that window's ceiling.
DATASETS_PIN="datasets==4.8.4"

sweep_already_done() {
    local dir=$1
    if [ "$(find "$dir" -name benchmark_summary.json 2>/dev/null | wc -l)" -ge 4 ]; then
        echo "results already present in $dir — skipping (delete the directory to re-run)"
        return 0
    fi
    # An interrupted panel leaves partial parallel_* artifacts - notably
    # evalscope's benchmark_data.db, which it refuses to overwrite. The ladder
    # always re-runs as one evalscope invocation, so clear partials.
    if [ -d "$dir" ]; then
        echo "clearing partial results in $dir from an interrupted run"
        rm -rf "$dir"
    fi
    return 1
}

ensure_datasets() {
    if python3 -c "import sys, datasets.features.features as f; sys.exit(0 if 'List' in f._FEATURE_TYPES else 1)" 2>/dev/null; then
        return 0
    fi
    echo "datasets too old for the OpenHands dataset (no 'List' feature type) — pinning $DATASETS_PIN"
    pip install --no-deps "$DATASETS_PIN"
}

ensure_evalscope() {
    ensure_datasets
    python3 -c "import evalscope.perf.plugin.datasets.swe_smith" 2>/dev/null && return 0
    echo "evalscope not found — installing the pinned client (one-time):"
    echo "  evalscope-deps/scripts/install_evalscope_deps.sh"
    echo "  PIP_NO_DEPS=1 pip install \"$EVALSCOPE_PIN\""
    evalscope-deps/scripts/install_evalscope_deps.sh
    PIP_NO_DEPS=1 pip install "$EVALSCOPE_PIN"
}

start_server() {
    local log=$1
    shift
    if curl -sf -m 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "ERROR: something is already serving on port $PORT — stop it first" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$log")"
    setsid "$@" > "$log" 2>&1 &
    SERVER_PID=$!
    trap stop_server EXIT
    echo "waiting for the server to become healthy (log: $log) ..."
    for _ in $(seq 1 720); do
        curl -sf -m 3 "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "server exited during startup — tail of $log:" >&2
            tail -20 "$log" >&2
            exit 1
        fi
        sleep 5
    done
    echo "server did not become healthy within 1h" >&2
    exit 1
}

stop_server() {
    [ -n "$SERVER_PID" ] || return 0
    kill -TERM -"$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 2
    done
    kill -KILL -"$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    for _ in $(seq 1 60); do
        python3 -c "import socket; s=socket.socket(); s.bind(('localhost', $PORT)); s.close()" 2>/dev/null && return 0
        sleep 2
    done
}

model_cached() {
    # True when the model's snapshot is fully resolvable from the local HF
    # cache (no network). Used to skip an optional curve gracefully instead of
    # triggering a multi-hundred-GB download from inside run_all.
    python3 - "$1" <<'PY'
import sys
from huggingface_hub import snapshot_download
try:
    snapshot_download(sys.argv[1], local_files_only=True)
except Exception:
    sys.exit(1)
PY
}
