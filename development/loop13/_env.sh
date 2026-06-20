#!/usr/bin/env bash
# Loop 13 shared env + CLONE-SAFETY GUARD.  ***SOURCE this, do not execute.***
# Callers MUST check it:   source "$HERE/_env.sh" || exit 1
#
# Refuses to proceed unless the default `import sglang` resolves to the DEV clone
# (/sgl-workspace/sglang).  This is the backstop against accidentally serving the
# v2 clone (/sgl-workspace/double-sparisty-v2/sglang) — a DIFFERENT, refactored DS
# codebase that would invalidate every accuracy number.
set -uo pipefail

export DEV_CLONE=/sgl-workspace/sglang
export V2_CLONE=/sgl-workspace/double-sparisty-v2/sglang
export MODEL=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
export MASK=/cluster-storage/models/glm51-fp8-channel-mask-loop12.safetensors
export HOST=127.0.0.1
export PORT=30000
export EVID=/sgl-workspace/sglang/development/loop13/evidence
export PIDFILE=/tmp/loop13_server.pid
mkdir -p "$EVID"

# Accumulate the first failure reason (do NOT use a function with `return` — that
# only exits the function, not this sourced script).
_guard_fail=""

# (1) PYTHONPATH must not drag in the v2 clone.
case "${PYTHONPATH:-}" in
  *double-sparisty-v2*|*"$V2_CLONE"*) _guard_fail="PYTHONPATH points at the v2 clone: ${PYTHONPATH}";;
esac

# (2) The default `import sglang` must BE the dev clone (the real backstop).
if [ -z "$_guard_fail" ]; then
  _got=$(python3 -c 'import sglang,os;print(os.path.realpath(os.path.dirname(sglang.__file__)))' 2>/dev/null)
  _want=$(python3 -c 'import os;print(os.path.realpath(os.path.join("'"$DEV_CLONE"'","python","sglang")))')
  [ "$_got" = "$_want" ] || _guard_fail="import sglang -> ${_got:-<none>} (expected dev clone ${_want})"
fi

# (3) expandable_segments must NEVER be set for serving (breaks custom all-reduce @ TP=8).
if [ -z "$_guard_fail" ]; then
  case "${PYTORCH_CUDA_ALLOC_CONF:-}" in
    *expandable_segments*) _guard_fail="PYTORCH_CUDA_ALLOC_CONF has expandable_segments — not allowed for serving";;
  esac
fi

if [ -n "$_guard_fail" ]; then
  echo "FATAL(guard): $_guard_fail. Refusing." >&2
  # Stop the sourced script (return) or the executed script (exit).
  return 1 2>/dev/null || exit 1
fi

echo "[guard] OK  sglang=$_got  port=$PORT  mask=$( [ -s "$MASK" ] && echo present || echo MISSING )"
