#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/adamcarrington/Documents/kalshi bot 2.10.26"
PID_FILE="$ROOT_DIR/run/kalshi_overnight.pid"
RUNNER_LOG="$ROOT_DIR/reports/runner_overnight.log"

cd "$ROOT_DIR"

if [ -f "$PID_FILE" ]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
else
  PID=""
fi

RUNNING="no"
if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
  echo "running pid=$PID"
  RUNNING="yes"
else
  echo "running=no"
  if [ -n "${PID:-}" ]; then
    echo "pidfile_stale=$PID"
  fi
fi

if [ -f "$RUNNER_LOG" ]; then
  echo "runner_log=$RUNNER_LOG"
  tail -n 20 "$RUNNER_LOG" || true
else
  echo "runner_log=none"
fi

ACTIVE_LOG_REL="$(
  tr -d '\000' < "$RUNNER_LOG" 2>/dev/null \
    | grep -E 'started_at=.* log=' \
    | tail -n 1 \
    | sed -E 's/.* log=([^ ]+).*/\1/' || true
)"
if [ "$RUNNING" = "yes" ] && [ -n "${ACTIVE_LOG_REL:-}" ] && [ -f "$ROOT_DIR/$ACTIVE_LOG_REL" ]; then
  echo "active_log=$ACTIVE_LOG_REL"
  tail -n 20 "$ROOT_DIR/$ACTIVE_LOG_REL" || true
elif [ "$RUNNING" = "yes" ] && [ -n "${ACTIVE_LOG_REL:-}" ]; then
  echo "active_log_pending=$ACTIVE_LOG_REL"
else
  LATEST_LOG="$(ls reports/live_*.log 2>/dev/null | sort -r | head -n 1 || true)"
  if [ -n "${LATEST_LOG:-}" ]; then
    echo "latest_log=$LATEST_LOG"
    tail -n 20 "$LATEST_LOG" || true
  else
    echo "latest_log=none"
  fi
fi
