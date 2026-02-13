# Kalshi Market Maker (Risk-First)

This repository contains a new market-making bot framework built for **Kalshi-style binary markets**, with a focus on:

- Aggressive profit-seeking quote logic
- Hard risk limits and drawdown controls
- Tiered exits (take-profit + stop-loss)
- Kill-switch behavior
- Monte Carlo parameter search to target double-digit simulated returns

## Important

No bot can guarantee profits or "safe" returns in live markets.  
This project is designed to maximize expected edge while enforcing strict risk controls and go/no-go checks before live deployment.

## Quick start

```bash
python3 /Users/adamcarrington/Documents/kalshi\ bot\ 2.10.26/run_bot.py --mode paper --optimize
```

Run all three 15-minute crypto markets explicitly:

```bash
python3 /Users/adamcarrington/Documents/kalshi\ bot\ 2.10.26/run_bot.py --mode paper --markets KXBTC15M,KXETH15M,KXSOL15M --episodes 20 --steps 1200 --optimize
```

## Kalshi sandbox mode (no real money)

Dry-run first (connects and computes quotes, but does not submit orders):

```bash
python3 /Users/adamcarrington/Documents/kalshi\ bot\ 2.10.26/run_bot.py \
  --mode sandbox \
  --markets KXBTC15M,KXETH15M,KXSOL15M \
  --api-key-id "YOUR_API_KEY_ID" \
  --private-key-path "/Users/adamcarrington/Documents/kalshi bot 2.10.26/kalshi_private.txt" \
  --runtime-minutes 5
```

Note: sandbox mode now auto-resolves requested symbols (for example `KXBTC15M`) to live tradable market tickers on the connected exchange.
Default sandbox protections are now strict-safe by default:
- Session stop-loss hard kill (`session_stop_loss_cents=2200`)
- Session take-profit lock (`session_take_profit_cents=600`)
- Larger live sizing (`live_size_scale=0.55`, higher base order size)
- Looser market/quote filters (`min_market_spread=0.035`, `min_quote_edge=0.006`)
- No new entries before close (`no_new_entries_before_close_seconds = preclose_guard_seconds + entry_time_buffer_seconds`, default 210s)
- Faster requotes and higher order capacity (`min_requote_seconds=4`, `max_orders_per_market=90`)
- No-trade window fast-exit (stops early and rolls to the next pass when markets are only preclose/finalized)
- Series-first rolling: request only series aliases (`KXBTC15M`, `KXETH15M`, `KXSOL15M`) and resolver rolls to next non-terminal market
- Strict runway-first market resolution before any "next market" fallback (prevents immediate preclose stalls)
- Momentum toxicity guard (dynamic quote-edge boost + asymmetric size throttling during one-way moves)
- Auto-roll when all requested markets are untradable for consecutive cycles (`no_trade_window_cycle_limit`)

Submit sandbox orders (still demo environment):

```bash
python3 /Users/adamcarrington/Documents/kalshi\ bot\ 2.10.26/run_bot.py \
  --mode sandbox \
  --markets KXBTC15M,KXETH15M,KXSOL15M \
  --api-key-id "YOUR_API_KEY_ID" \
  --private-key-path "/Users/adamcarrington/Documents/kalshi bot 2.10.26/kalshi_private.txt" \
  --runtime-minutes 15 \
  --loop-seconds 8 \
  --submit-orders
```

Aggressive profile with new guards enabled:

```bash
python3 /Users/adamcarrington/Documents/3.0\ bot/kalshi\ bot\ 2.10.26/run_bot.py \
  --mode sandbox \
  --config /Users/adamcarrington/Documents/kalshi\ bot\ 2.10.26/config.profit_push.json \
  --markets KXBTC15M,KXETH15M,KXSOL15M \
  --api-key-id "YOUR_API_KEY_ID" \
  --private-key-path "/Users/adamcarrington/Documents/kalshi bot 2.10.26/kalshi_private.txt" \
  --runtime-minutes 20 \
  --loop-seconds 6 \
  --submit-orders
```

Authentication diagnostics (tests key pair against known Kalshi environments; no trading):

```bash
python3 /Users/adamcarrington/Documents/kalshi\ bot\ 2.10.26/run_bot.py \
  --auth-diagnose \
  --api-key-id "YOUR_API_KEY_ID" \
  --private-key-path "/Users/adamcarrington/Documents/kalshi bot 2.10.26/kalshi_private.txt"
```

Run a fixed config (no optimization):

```bash
python3 /Users/adamcarrington/Documents/kalshi\ bot\ 2.10.26/run_bot.py --mode paper --episodes 60 --steps 5000
```

## Outputs

Each run writes a JSON report by default:

- `/Users/adamcarrington/Documents/kalshi bot 2.10.26/reports/latest_report.json`

## Structure

- `/Users/adamcarrington/Documents/kalshi bot 2.10.26/kalshi_mm/config.py`: bot, strategy, risk, and simulation config
- `/Users/adamcarrington/Documents/kalshi bot 2.10.26/kalshi_mm/models.py`: core datatypes and position accounting
- `/Users/adamcarrington/Documents/kalshi bot 2.10.26/kalshi_mm/strategy.py`: adaptive market-making quote engine
- `/Users/adamcarrington/Documents/kalshi bot 2.10.26/kalshi_mm/risk.py`: safeguards, kill-switches, and tiered exits
- `/Users/adamcarrington/Documents/kalshi bot 2.10.26/kalshi_mm/sim.py`: synthetic market + fill model + Monte Carlo
- `/Users/adamcarrington/Documents/kalshi bot 2.10.26/kalshi_mm/optimizer.py`: parameter search constrained by risk
- `/Users/adamcarrington/Documents/kalshi bot 2.10.26/run_bot.py`: CLI entrypoint
- `/Users/adamcarrington/Documents/kalshi bot 2.10.26/tests/`: unit tests

## Live trading integration

This version is intentionally safe-by-default and paper/simulation focused.  
You can wire in your authenticated Kalshi execution adapter once you validate:

- Latency behavior
- Real fill quality
- Slippage and fee impact
- Risk engine kill paths
