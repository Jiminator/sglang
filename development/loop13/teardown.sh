#!/usr/bin/env bash
# Tear down the tracked server and wait for all 8 GPUs to return to ~idle.
# Kills ONLY the PID in $PIDFILE — never blanket-kills GPU PIDs or pkill -f.
# Run BACKGROUNDED (it sleeps while waiting for GPU memory to release).
set -uo pipefail
PIDFILE=/tmp/loop13_server.pid
PID=$(cat "$PIDFILE" 2>/dev/null || true)
if [ -n "${PID:-}" ]; then
  echo "tearing down PID=$PID"
  kill "$PID" 2>/dev/null || true
  for i in $(seq 1 60); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
  kill -9 "$PID" 2>/dev/null || true
fi
rm -f "$PIDFILE"
for i in $(seq 1 60); do
  t=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
  [ "$t" -lt 8000 ] && { echo "GPU idle (${t} MiB) after ~$((i*2))s"; break; }
  sleep 2
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
