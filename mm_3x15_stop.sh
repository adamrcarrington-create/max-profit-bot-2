#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/adamcarrington/Documents/3.0 bot/kalshi bot 2.10.26"
RUN_DIR="$ROOT/run/mm_3x15"

mkdir -p "$RUN_DIR"

for market in btc eth sol; do
  pidfile="$RUN_DIR/${market}.pid"
  if [[ ! -f "$pidfile" ]]; then
    echo "market=$market running=no pidfile=missing"
    continue
  fi
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -z "${pid:-}" ]]; then
    rm -f "$pidfile"
    echo "market=$market running=no pidfile=empty"
    continue
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      echo "market=$market stop=failed pid=$pid"
    else
      echo "market=$market stop=ok pid=$pid"
      rm -f "$pidfile"
    fi
  else
    echo "market=$market running=no stale_pid=$pid"
    rm -f "$pidfile"
  fi
done
