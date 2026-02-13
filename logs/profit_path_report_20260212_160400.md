# Profit Path Report (2026-02-12 16:04 local)

## Scope
- Objective: maximize simulated/paper profit with MM mode only.
- Safety: no live orders sent in this run.

## Code/config updates in this pass
- `kalshi_hft/replay_capture.py`
  - Added fallback when `--prefer-quoteable-mm-market` finds none: falls back to active market instead of failing.
  - Added `market_selection=` logging (`quoteable_mm`, `active_fallback`, etc.).
- Tuned paper MM configs for validation:
  - `kalshi_hft/config.paper_mm_3x15_btc.json`
  - `kalshi_hft/config.paper_mm_3x15_eth.json`
  - `kalshi_hft/config.paper_mm_3x15_sol.json`
  - Changes: `strategy.mm_half_spread_cents=1.0`, `execution.max_orders_per_minute=8`, `mm.base_size=20`.

## New replay windows captured
- `logs/replay_mm3x15_20260212_153552/` (BTC/ETH/SOL)
  - Quality: quoteable ratio 1.0 for all three markets.
- `logs/replay_mm3x15_20260212_155321/` (BTC/ETH/SOL)
  - Quality: quoteable ratio 1.0 for all three markets (shorter rows due intermittent external reference timeouts).

## Profit findings
- Single-window bests:
  - `logs/backtest_mm3x15_20260212_153552_grid_v3.json`
    - Best: `order_count=20`, `mm_half_spread_cents=1.0`
    - PnL: **$10.516** (all 3 markets positive)
  - `logs/backtest_mm3x15_20260212_155321_grid_v3.json`
    - Best: `order_count=30`, `mm_half_spread_cents=1.5`
    - PnL: **$14.608** (all 3 markets positive)

- Multi-window robust (quality-gated) result:
  - `logs/backtest_mm3x15_robust_5windows_20260212_155321_v3_quality.json`
  - Filters: `min_window_fills_informative=50`, `min_informative_windows=3`, `min_window_pnl_cents=100`, `min_market_pnl_cents=0`
  - Best robust: `order_count=20`, `mm_half_spread_cents=1.0`
  - Aggregate informative-window PnL: **$35.2618**
  - Informative windows: 3
  - Worst informative window PnL: **$9.954**
  - Worst per-market informative PnL: **+111.8 cents**

## Important interpretation
- The replay scorer is still a simplified simulator (`kalshi_hft/backtest.py`) and not a full faithful execution replay of the MM live loop.
- So the above is strong evidence for *parameter direction*, not a guarantee of realized fill PnL.

## Real runtime check (MM path)
- Ran a short dry-run MM process for BTC with tuned config:
  - Command: `python3 -m kalshi_hft.kalshi_bot --config ...config.paper_mm_3x15_btc.json --runtime-minutes 1`
  - Observed: active `mm_quote` / toxicity regime behavior in live loop.
  - Because `dry_run=true`, no fills/PnL are realized by design.
