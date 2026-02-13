#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/adamcarrington/Documents/kalshi bot 2.10.26"
PID_FILE="$ROOT_DIR/run/kalshi_overnight.pid"
HOURS="${1:-5}"
RUNNER_LOG="$ROOT_DIR/reports/runner_overnight.log"

cd "$ROOT_DIR"
mkdir -p run reports

if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "already_running pid=$OLD_PID"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

if command -v caffeinate >/dev/null 2>&1; then
  nohup env HOURS="$HOURS" caffeinate -dimsu /bin/bash "$ROOT_DIR/mm_overnight_loop.sh" >"$RUNNER_LOG" 2>&1 &
else
  nohup env HOURS="$HOURS" /bin/bash "$ROOT_DIR/mm_overnight_loop.sh" >"$RUNNER_LOG" 2>&1 &
fi
NEW_PID=$!
echo "$NEW_PID" >"$PID_FILE"
sleep 1
if kill -0 "$NEW_PID" 2>/dev/null; then
  echo "started pid=$NEW_PID hours=$HOURS runner_log=$RUNNER_LOG"
else
  echo "failed_to_start pid=$NEW_PID runner_log=$RUNNER_LOG"
  tail -n 60 "$RUNNER_LOG" 2>/dev/null || true
  exit 1
fi
