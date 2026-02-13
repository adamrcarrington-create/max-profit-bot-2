#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/adamcarrington/Documents/3.0 bot/kalshi bot 2.10.26"
RUN_DIR="$ROOT/run/mm_3x15"
LOG_DIR="$ROOT/logs"
RUNTIME_MINUTES="${1:-}"

# Safety guard: require explicit env override before any live launch.
if [[ "${MM_ALLOW_LIVE:-}" != "YES" ]]; then
  echo "live_launch_blocked reason=paper_only_lock hint='set MM_ALLOW_LIVE=YES to override intentionally'"
  exit 1
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"
cd "$ROOT"

market_config() {
  case "$1" in
    btc) echo "$ROOT/kalshi_hft/config.live_mm_3x15_btc.json" ;;
    eth) echo "$ROOT/kalshi_hft/config.live_mm_3x15_eth.json" ;;
    sol) echo "$ROOT/kalshi_hft/config.live_mm_3x15_sol.json" ;;
    *) return 1 ;;
  esac
}

assert_mm_config() {
  local cfg="$1"
  local out
  out="$(python3 - "$cfg" <<'PY'
import json
import sys

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

mode = (raw.get("strategy_mode") or raw.get("strategy", {}).get("strategy_mode") or "hft").strip().lower()
dry_run = bool(raw.get("execution", {}).get("dry_run", True))
series = str(raw.get("strategy", {}).get("series_ticker") or "")

print(f"mode={mode} dry_run={int(dry_run)} series={series}")
if mode != "mm":
    raise SystemExit(2)
if dry_run:
    raise SystemExit(3)
PY
  )"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    if [[ $rc -eq 2 ]]; then
      echo "config_guard_failed reason=non_mm_mode config=$cfg"
    elif [[ $rc -eq 3 ]]; then
      echo "config_guard_failed reason=dry_run_enabled config=$cfg"
    else
      echo "config_guard_failed reason=invalid_json config=$cfg"
    fi
    return 1
  fi
  echo "config_guard_ok config=$cfg $out"
}

start_one() {
  local market="$1"
  local cfg="$2"
  local pidfile="$RUN_DIR/${market}.pid"
  local out_log="$LOG_DIR/run_mm_3x15_${market}.out"

  assert_mm_config "$cfg"

  if [[ -f "$pidfile" ]]; then
    local old_pid
    old_pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "already_running market=$market pid=$old_pid out=$out_log"
      return 0
    fi
    rm -f "$pidfile"
  fi

  local -a cmd=(python3 -m kalshi_hft.kalshi_bot --config "$cfg")
  if [[ -n "$RUNTIME_MINUTES" ]]; then
    cmd+=(--runtime-minutes "$RUNTIME_MINUTES")
  fi

  nohup "${cmd[@]}" >"$out_log" 2>&1 &
  local pid=$!
  echo "$pid" >"$pidfile"
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    echo "started market=$market pid=$pid out=$out_log config=$cfg"
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

echo "status_cmd=$ROOT/mm_3x15_status.sh"
echo "stop_cmd=$ROOT/mm_3x15_stop.sh"
