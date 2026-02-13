#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/adamcarrington/Documents/3.0 bot/kalshi bot 2.10.26"
RUN_DIR="$ROOT/run/mm_3x15_paper"

paper_log_for() {
  case "$1" in
    btc) echo "$ROOT/logs/kalshi_hft_mm_paper_3x15_btc.log" ;;
    eth) echo "$ROOT/logs/kalshi_hft_mm_paper_3x15_eth.log" ;;
    sol) echo "$ROOT/logs/kalshi_hft_mm_paper_3x15_sol.log" ;;
    *) return 1 ;;
  esac
}

perf_for() {
  case "$1" in
    btc) echo "$ROOT/logs/performance_mm_paper_3x15_btc.json" ;;
    eth) echo "$ROOT/logs/performance_mm_paper_3x15_eth.json" ;;
    sol) echo "$ROOT/logs/performance_mm_paper_3x15_sol.json" ;;
    *) return 1 ;;
  esac
}

echo "root=$ROOT"
for market in btc eth sol; do
  pidfile="$RUN_DIR/${market}.pid"
  if [[ -f "$pidfile" ]]; then
    pid="$(cat "$pidfile" 2>/dev/null || true)"
  else
    pid=""
  fi

  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "market=$market running=yes pid=$pid"
  else
    if [[ -n "${pid:-}" ]]; then
      echo "market=$market running=no stale_pid=$pid"
    else
      echo "market=$market running=no"
    fi
  fi

  live_log="$(paper_log_for "$market")"
  perf_json="$(perf_for "$market")"
  echo "market=$market log=$live_log"
  echo "market=$market perf=$perf_json"

  if [[ -f "$live_log" ]]; then
    awk -v market="$market" '
      /mm_quote/ {q++}
      /mm_tp_ioc/ {tp_ioc++}
      /mm_tp_resting/ {tp_rest++}
      /mm_skip_pinned/ {skip_pinned++}
      /mm_skip_spread_wide/ {skip_wide++}
      /mm_skip_spread_narrow/ {skip_narrow++}
      /mm_buy_order_error/ {buy_err++}
      /dry_run=1/ {dry++}
      END {
        printf "market=%s mm_quote=%d tp_ioc=%d tp_resting=%d skip_pinned=%d skip_wide=%d skip_narrow=%d buy_err=%d dry_run_lines=%d\n",
          market, q+0, tp_ioc+0, tp_rest+0, skip_pinned+0, skip_wide+0, skip_narrow+0, buy_err+0, dry+0
      }
    ' "$live_log"
    tail -n 8 "$live_log" || true
  else
    echo "market=$market live_log_missing=yes"
  fi

  if [[ -f "$perf_json" ]]; then
    rg -n '"as_of"|"ticker"|"wins"|"losses"|"fill_count"|"order_failures"|"regime"' "$perf_json" || true
  else
    echo "market=$market perf_missing=yes"
  fi

  echo "---"
done
