#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/adamcarrington/Documents/kalshi bot 2.10.26"
HOURS="${HOURS:-5}"
API_KEY_ID="${KALSHI_API_KEY_ID:-bc52dea5-5a5b-46a1-afcf-b3433f40680e}"
PRIVATE_KEY_PATH="${KALSHI_PRIVATE_KEY_PATH:-/Users/adamcarrington/Documents/kalshi bot 2.10.26/kalshi_private.txt}"
API_BASE_URL="${KALSHI_API_BASE_URL:-https://api.elections.kalshi.com}"
MARKETS="${KALSHI_MARKETS:-KXBTC15M}"
PROFIT_FLOOR_CENTS="${PROFIT_FLOOR_CENTS:--700}"
PROFIT_LOCK_CENTS="${PROFIT_LOCK_CENTS:-200}"
PASS_RUNTIME_MINUTES="${PASS_RUNTIME_MINUTES:-3}"
PASS_LOOP_SECONDS="${PASS_LOOP_SECONDS:-8}"

cd "$ROOT_DIR"
mkdir -p run reports

END_TS=$(( $(date +%s) + HOURS*3600 ))
LOG_PATH="reports/live_$(date +%Y%m%d_%H%M%S).log"
export API_KEY_ID PRIVATE_KEY_PATH API_BASE_URL

get_balance_cents() {
  python3 - <<'PY'
import os
from kalshi_mm.kalshi_api import KalshiClient, resolve_credentials

creds = resolve_credentials(
    os.environ.get("API_KEY_ID", ""),
    os.environ.get("PRIVATE_KEY_PATH", ""),
)
client = KalshiClient(creds, base_url=os.environ.get("API_BASE_URL", "https://api.elections.kalshi.com"))
print(int(client.get_balance().get("balance", 0)))
PY
}

START_BALANCE_CENTS="$(get_balance_cents)"
if ! [[ "${START_BALANCE_CENTS:-}" =~ ^-?[0-9]+$ ]]; then
  START_BALANCE_CENTS=0
fi
echo "started_at=$(date '+%Y-%m-%d %H:%M:%S') hours=$HOURS log=$LOG_PATH start_balance_cents=$START_BALANCE_CENTS profit_floor_cents=$PROFIT_FLOOR_CENTS profit_lock_cents=$PROFIT_LOCK_CENTS pass_runtime_minutes=$PASS_RUNTIME_MINUTES pass_loop_seconds=$PASS_LOOP_SECONDS" | tee -a "$LOG_PATH"

while [ "$(date +%s)" -lt "$END_TS" ]; do
  echo "pass_started_at=$(date '+%Y-%m-%d %H:%M:%S') pass_runtime_minutes=$PASS_RUNTIME_MINUTES" | tee -a "$LOG_PATH"
  set +e
  OUT="$(python3 "$ROOT_DIR/run_bot.py" \
    --mode sandbox \
    --config "$ROOT_DIR/config.example.json" \
    --api-key-id "$API_KEY_ID" \
    --private-key-path "$PRIVATE_KEY_PATH" \
    --api-base-url "$API_BASE_URL" \
    --pss-salt-mode digest \
    --markets "$MARKETS" \
    --runtime-minutes "$PASS_RUNTIME_MINUTES" \
    --loop-seconds "$PASS_LOOP_SECONDS" \
    --submit-orders 2>&1)"
  RC=$?
  set -e

  printf "%s\n" "$OUT" | tee -a "$LOG_PATH"
  echo "run_exit_code=$RC" | tee -a "$LOG_PATH"

  CURRENT_BALANCE_CENTS="$(get_balance_cents || echo "$START_BALANCE_CENTS")"
  if ! [[ "${CURRENT_BALANCE_CENTS:-}" =~ ^-?[0-9]+$ ]]; then
    CURRENT_BALANCE_CENTS="$START_BALANCE_CENTS"
  fi
  DELTA_CENTS=$(( CURRENT_BALANCE_CENTS - START_BALANCE_CENTS ))
  echo "session_balance_check start=$START_BALANCE_CENTS current=$CURRENT_BALANCE_CENTS delta_cents=$DELTA_CENTS" | tee -a "$LOG_PATH"
  if [ "$DELTA_CENTS" -lt "$PROFIT_FLOOR_CENTS" ]; then
    echo "stop_reason=profit_floor_breached floor_cents=$PROFIT_FLOOR_CENTS delta_cents=$DELTA_CENTS" | tee -a "$LOG_PATH"
    break
  fi
  if [ "$DELTA_CENTS" -ge "$PROFIT_LOCK_CENTS" ]; then
    echo "stop_reason=profit_lock_hit lock_cents=$PROFIT_LOCK_CENTS delta_cents=$DELTA_CENTS" | tee -a "$LOG_PATH"
    break
  fi

  if printf "%s" "$OUT" | grep -q "sandbox_error=no_markets_resolved"; then
    sleep 5
  elif printf "%s" "$OUT" | grep -q "stop_reason=no_trade_window"; then
    sleep 5
  elif [ "$RC" -ne 0 ]; then
    sleep 5
  else
    sleep 3
  fi
done

echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S') log=$LOG_PATH"
