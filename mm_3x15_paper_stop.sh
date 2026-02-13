#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/adamcarrington/Documents/3.0 bot/kalshi bot 2.10.26"
RUN_DIR="$ROOT/run/mm_3x15_paper"

for market in btc eth sol; do
  pidfile="$RUN_DIR/${market}.pid"
  if [[ ! -f "$pidfile" ]]; then
    echo "market=$market stopped=yes pidfile=missing"
    continue
  fi

  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -z "${pid:-}" ]]; then
    rm -f "$pidfile"
    echo "market=$market stopped=yes pidfile=empty"
    continue
  fi

  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
    echo "market=$market stopped=yes pid=$pid"
  else
    echo "market=$market stopped=yes pid=$pid already_dead=1"
  fi

  rm -f "$pidfile"
done
