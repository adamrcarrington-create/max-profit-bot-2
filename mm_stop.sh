#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/adamcarrington/Documents/kalshi bot 2.10.26"
PID_FILE="$ROOT_DIR/run/kalshi_overnight.pid"

cd "$ROOT_DIR"

if [ ! -f "$PID_FILE" ]; then
  echo "not_running no_pidfile"
  exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [ -z "${PID:-}" ]; then
  rm -f "$PID_FILE"
  echo "not_running empty_pidfile"
  exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
  kill "$PID" 2>/dev/null || true
  sleep 1
  if kill -0 "$PID" 2>/dev/null; then
    echo "still_running pid=$PID"
    exit 1
  fi
  echo "stopped pid=$PID"
else
  echo "not_running stale_pid=$PID"
fi

rm -f "$PID_FILE"
