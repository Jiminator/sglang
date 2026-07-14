# Shared helpers for the run scripts. Sourced, not executed.
# Callers: the per-config run scripts under gb300/ and b300/.

PORT="${PORT:-8002}"
SERVER_PID=""
DAY0_COMMIT=22dce572045c277ce46f1a287c4be1112b214368
EVALSCOPE_PIN="evalscope[all] @ git+https://github.com/modelscope/evalscope.git@acd09b44384d53174768bb1063f675420f76fae9"
# The OpenHands dataset (nebius/SWE-rebench-openhands-trajectories) declares the
# `List` feature type, which the dataset build can only read with datasets>=4.0.
# The release image ships datasets 3.6.0; evalscope's own modelscope dependency
# already requires datasets>=4.0.0,<=4.8.4, so pin to that window's ceiling.
DATASETS_PIN="datasets==4.8.4"

sweep_already_done() {
    local dir=$1
    if [ "$(find "$dir" -name benchmark_summary.json 2>/dev/null | wc -l)" -ge 4 ]; then
        echo "results already present in $dir — skipping (delete the directory to re-run)"
        return 0
    fi
    # An interrupted panel leaves partial parallel_* artifacts — notably
    # evalscope's benchmark_data.db, which it refuses to overwrite ("The db
    # file exists, delete it and start again!"). The ladder always re-runs as
    # one evalscope invocation, so a partial panel can't be continued — clear
    # it so the re-run starts clean.
    if [ -d "$dir" ]; then
        echo "clearing partial results in $dir from an interrupted run"
        rm -rf "$dir"
    fi
    return 1
}

ensure_datasets() {
    # `List` is only registered as a feature type from datasets 4.0 on; if it is
    # missing the installed datasets is too old to read the OpenHands dataset.
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

ensure_day0_checkout() {
    if [ -z "${DAY0_SGLANG:-}" ]; then
        DAY0_SGLANG="$(git rev-parse --show-toplevel)/../sglang-day0"
    fi
    if [ ! -d "$DAY0_SGLANG" ]; then
        echo "creating day-0 checkout at $DAY0_SGLANG ($DAY0_COMMIT, public branch glm-opt)"
        git fetch origin glm-opt \
            || git fetch https://github.com/sgl-project/sglang.git glm-opt
        git worktree add "$DAY0_SGLANG" "$DAY0_COMMIT"
    fi
    DAY0_SGLANG="$(cd "$DAY0_SGLANG" && pwd)"
    if [ "$(git -C "$DAY0_SGLANG" rev-parse HEAD)" != "$DAY0_COMMIT" ]; then
        echo "ERROR: $DAY0_SGLANG is not at the day-0 commit $DAY0_COMMIT" >&2
        exit 1
    fi
}

ensure_v0515_checkout() {
    # Serving code for the v0.5.15 curves: the CURRENT TIP of the public
    # release/v0.5.15 branch, re-fetched on every run — a post-release patch on
    # the branch lands in the sweeps automatically, nothing is pinned to a
    # commit. The worktree only swaps the Python layer (via PYTHONPATH);
    # compiled kernels still come from the installed wheels.
    if [ -z "${V0515_SGLANG:-}" ]; then
        V0515_SGLANG="$(git rev-parse --show-toplevel)/../sglang-v0515"
    fi
    if git fetch https://github.com/sgl-project/sglang.git release/v0.5.15 2>/dev/null \
        || git fetch origin release/v0.5.15; then
        local tip
        tip=$(git rev-parse FETCH_HEAD)
        if [ ! -d "$V0515_SGLANG" ]; then
            echo "creating release/v0.5.15 checkout at $V0515_SGLANG ($tip)"
            git worktree add --detach "$V0515_SGLANG" "$tip"
        elif [ "$(git -C "$V0515_SGLANG" rev-parse HEAD)" != "$tip" ]; then
            echo "updating $V0515_SGLANG to the release/v0.5.15 tip ($tip)"
            git -C "$V0515_SGLANG" checkout --detach "$tip"
        fi
    elif [ -d "$V0515_SGLANG" ]; then
        echo "WARNING: cannot fetch release/v0.5.15 — reusing $V0515_SGLANG at" \
             "$(git -C "$V0515_SGLANG" rev-parse --short HEAD)" >&2
    else
        echo "ERROR: cannot fetch release/v0.5.15 and no checkout at $V0515_SGLANG" >&2
        exit 1
    fi
    V0515_SGLANG="$(cd "$V0515_SGLANG" && pwd)"
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
