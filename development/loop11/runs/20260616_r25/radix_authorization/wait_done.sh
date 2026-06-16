#!/usr/bin/env bash
# Generic completion waiter: poll a log for a DONE marker or until a PID exits.
# Usage: wait_done.sh <log> <done_marker> [<pid>]
LOG="$1"; MARKER="$2"; PID="${3:-}"
while true; do
  if grep -qa "$MARKER" "$LOG" 2>/dev/null; then echo "MARKER_FOUND"; break; fi
  if [[ -n "$PID" ]] && ! kill -0 "$PID" 2>/dev/null; then echo "PID_EXITED"; break; fi
  sleep 15
done
