#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/adamcarrington/Documents/3.0 bot/kalshi bot 2.10.26"
RUN_DIR="$ROOT/run/mm_3x15_paper"
LOG_DIR="$ROOT/logs"
RUNTIME_MINUTES="${1:-30}"
if [[ -x "/usr/bin/python3" ]]; then
  PY_BIN="/usr/bin/python3"
else
  PY_BIN="python3"
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"
cd "$ROOT"

market_config() {
  case "$1" in
    btc) echo "$ROOT/kalshi_hft/config.paper_mm_3x15_btc.json" ;;
    eth) echo "$ROOT/kalshi_hft/config.paper_mm_3x15_eth.json" ;;
    sol) echo "$ROOT/kalshi_hft/config.paper_mm_3x15_sol.json" ;;
    *) return 1 ;;
  esac
}

assert_paper_mm_config() {
  local cfg="$1"
  "$PY_BIN" - "$cfg" <<'PY'
import json
import sys

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

mode = (raw.get("strategy_mode") or raw.get("strategy", {}).get("strategy_mode") or "hft").strip().lower()
dry_run = bool(raw.get("execution", {}).get("dry_run", True))

if mode != "mm":
    raise SystemExit("config_guard_failed reason=non_mm_mode")
if not dry_run:
    raise SystemExit("config_guard_failed reason=live_mode")

print(f"config_guard_ok mode={mode} dry_run={int(dry_run)}")
PY
}

start_one() {
  local market="$1"
  local cfg="$2"
  local pidfile="$RUN_DIR/${market}.pid"
  local out_log="$LOG_DIR/run_mm_3x15_paper_${market}.out"

  assert_paper_mm_config "$cfg"

  if [[ -f "$pidfile" ]]; then
    local old_pid
    old_pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "already_running market=$market pid=$old_pid out=$out_log"
      return 0
    fi
    rm -f "$pidfile"
  fi

  nohup "$PY_BIN" -m kalshi_hft.kalshi_bot --config "$cfg" --runtime-minutes "$RUNTIME_MINUTES" >"$out_log" 2>&1 &
  local pid=$!
  echo "$pid" >"$pidfile"
  sleep 1

  if kill -0 "$pid" 2>/dev/null; then
    echo "started market=$market pid=$pid mode=paper out=$out_log config=$cfg"
  else
    rm -f "$pidfile"
    echo "failed_to_start market=$market out=$out_log"
    tail -n 80 "$out_log" 2>/dev/null || true
    return 1
  fi
}

for market in btc eth sol; do
  cfg="$(market_config "$market")"
  start_one "$market" "$cfg"
done

echo "status_cmd=$ROOT/mm_3x15_paper_status.sh"
echo "stop_cmd=$ROOT/mm_3x15_paper_stop.sh"
