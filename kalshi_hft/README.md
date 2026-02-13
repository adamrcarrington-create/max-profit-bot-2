# Kalshi HFT (15m Crypto)

This module is a raw-API (no official SDK) Kalshi trading bot with:

- `client.py`: signed REST wrappers + optional 30-minute token reauth loop
- `monitor.py`: Binance spot polling + Kalshi websocket orderbook tracking
- `engine.py`: edge-based decision + limit-order execution + advanced risk controls
- `lead_operator.py`: background supervision and restart-on-401 behavior
- `backtest.py`: replay simulator for recorded reference + book streams
- `replay_capture.py`: recorder for timestamped orderbook + reference replay data
- `walkforward_optimize.py`: walk-forward parameter search on replay files
- `calibrate_signal.py`: settled-outcome calibration for model signal vs implied
- `ws_smoke.py`: websocket readiness smoke-test for orderbook feed

## Notes

- Production REST base URL in the current Kalshi docs is:
  `https://api.elections.kalshi.com/trade-api/v2`
- Signature timestamps are always generated in **milliseconds**.
- MM entries use post-only `type="limit"`.
- Snipe/exit paths use IOC `type="limit"` (no market orders).
- Reference feed uses Coinbase + Kraken median by default (Binance as fallback).
- If Coinbase vs Kraken diverges by more than `0.1%`, the bot pauses trading for `5s`.
- The engine enforces:
  - stale reference + stale orderbook guards
  - volatility regime controls (calm/trending/spiky)
  - volatility cancel guard (auto-cancel resting orders in flash-spike/spiky states)
  - inventory-aware sizing/skew
  - fee/slippage-adjusted EV gating
  - no-quote near close + force flatten window
  - hourly drawdown + daily loss kill-switches
- REST rate-limit pacing + automatic 429 retry/backoff
- dual strategy modes:
  - `strategy_mode="hft"`: directional edge model (`real_yes` vs implied)
  - `strategy_mode="mm"`: spread-capture market making (two-sided post-only quoting + profit-taking)
- MM market data path is WebSocket-driven (`KalshiOrderbookMonitor` -> `microstructure()`); MM skip checks and quote-from-mid use that feed, not REST orderbook polling.
- Orders/fills/cancels remain REST API calls; only the orderbook feed is WebSocket.

## Run

```bash
cd "/Users/adamcarrington/Documents/3.0 bot/kalshi bot 2.10.26"
python3 -m kalshi_hft.kalshi_bot --config kalshi_hft/config.example.json
```

Run the operator (supervisor):

```bash
cd "/Users/adamcarrington/Documents/3.0 bot/kalshi bot 2.10.26"
python3 -m kalshi_hft.lead_operator --config kalshi_hft/config.example.json
```

Paper-profit probe profile (dry-run):

```bash
cd "/Users/adamcarrington/Documents/3.0 bot/kalshi bot 2.10.26"
python3 -m kalshi_hft.kalshi_bot --config kalshi_hft/config.paper_profit_probe.json
```

Paper-robust profile (dry-run, calibrated for broader replay profitability):

```bash
cd "/Users/adamcarrington/Documents/3.0 bot/kalshi bot 2.10.26"
python3 -m kalshi_hft.kalshi_bot --config kalshi_hft/config.paper_robust.json
```

Enable MM mode in any config by setting:

```json
{
  "strategy_mode": "mm"
}
```

MM-specific controls live under the top-level `mm` config block (`quote_edge_cents`, `quote_refresh_secs`, `max_sum_buy_cents`, `tp_*`, inventory caps, and post-only cross cooldown).

Additional MM controls:

- `post_only_buffer_cents`: keep quote price this many cents under current ask to reduce post-only cross rejects.
- `min_spread_cents`: skip new buy quotes when either side spread is too tight.

Websocket smoke test:

```bash
cd "/Users/adamcarrington/Documents/3.0 bot/kalshi bot 2.10.26"
python3 -m kalshi_hft.ws_smoke --config kalshi_hft/config.example.json --duration-seconds 12
```

Three-market live MM launchers (BTC/ETH/SOL 15m):

```bash
cd "/Users/adamcarrington/Documents/3.0 bot/kalshi bot 2.10.26"
./mm_3x15_start.sh 20
./mm_3x15_status.sh
./mm_3x15_stop.sh
```

Replay backtest (JSONL/CSV):

```bash
cd "/Users/adamcarrington/Documents/3.0 bot/kalshi bot 2.10.26"
python3 -m kalshi_hft.backtest --config kalshi_hft/config.example.json --input /path/to/replay.jsonl --output logs/backtest_report.json
```

Backtest reports now include attribution fields:

- `signal_quality`: edge quality and expected EV before/after costs.
- `mode_attribution`: maker (`mm`) vs taker (`snipe`) raw P&L, fees, slippage, and net P&L.
- `raw_pnl_before_costs_cents` and `cost_drag_cents` to separate model quality from execution costs.

Strategy model controls:

- `distance_weight` and `momentum_weight`: raw model controls where momentum is an additive adjustment to distance probability (no artificial `0.5` anchor).
- `time_to_close_reference_seconds` with `min_distance_scale_multiplier` / `max_distance_scale_multiplier`: compresses/expands strike-distance scaling as expiry approaches.
- `model_edge_multiplier`: shrinks model edge toward implied probability.
  - `1.0` uses full model edge.
  - `0.0` fully trusts implied and disables directional edge.
  - `0.02` is the current robust paper/live-maker default after replay tuning.
- `cross_side_arb_enabled`: optional pair-arb path.
  - `pair_buy`: buy YES + NO when `YES ask + NO ask` is below parity after estimated costs.
  - `pair_sell`: sell YES + NO when `YES bid + NO bid` is above parity after estimated costs (inventory-aware by default).
- `cross_side_arb_allow_sell_pair`: enable/disable the inventory-aware `pair_sell` path.
- `cross_side_arb_min_edge_cents`: minimum estimated edge for pair-arb execution.
- `use_fractional_kelly`: optional fractional Kelly sizing overlay.
- `mm_half_spread_cents`: half-spread around fair value for maker quotes.
- `max_yes_spread_cents_for_mm` / `max_yes_spread_cents_for_snipe`: spread gates (set higher for demo trigger testing).
- `inventory_exit_*`: IOC sell logic that reduces held inventory when exit EV beats hold EV.
- `force_flatten_cooldown_seconds`: throttles repeated force-flatten attempts near market close.

Fair-value quoting used by the engine (YES side):

- `fair_yes_cents = round(real_yes_probability * 100)`
- `skew_cents = net_yes_inventory * inventory_skew_cents_per_contract`
- `yes_bid = fair_yes_cents - mm_half_spread_cents - skew_cents`
- `yes_ask = fair_yes_cents + mm_half_spread_cents - skew_cents`

NO quotes are parity-consistent and inventory-skewed the opposite direction.

Execution notes:

- Taker/snipe decisions use live orderbook ask/bid prices (not last-trade prints).
- 429 errors include retry/backoff details in logs and `request_rate_limit` metrics in performance JSON.

Capture replay data for backtests:

```bash
cd "/Users/adamcarrington/Documents/3.0 bot/kalshi bot 2.10.26"
python3 -m kalshi_hft.replay_capture --config kalshi_hft/config.example.json --duration-seconds 300 --interval-seconds 0.5
```

Walk-forward optimization over replay captures:

```bash
cd "/Users/adamcarrington/Documents/3.0 bot/kalshi bot 2.10.26"
python3 -m kalshi_hft.walkforward_optimize --inputs-glob "logs/replay_capture*.jsonl" --base-config kalshi_hft/config.example.json --output logs/walkforward_report.json --write-best-config logs/config.walkforward_best.json
```

Calibrate signal quality against settled outcomes:

```bash
cd "/Users/adamcarrington/Documents/3.0 bot/kalshi bot 2.10.26"
python3 -m kalshi_hft.calibrate_signal --inputs-glob "logs/replay_capture*.jsonl" --config kalshi_hft/config.example.json --output logs/signal_calibration_report.json
```
