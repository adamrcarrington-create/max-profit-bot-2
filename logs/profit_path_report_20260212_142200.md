# Paper-Only Profit Path Report (2026-02-12)

## Safety state
- Live launcher hard-locked: `/Users/adamcarrington/Documents/3.0 bot/kalshi bot 2.10.26/mm_3x15_start.sh` now requires `MM_ALLOW_LIVE=YES`.
- Current workflow stayed paper/replay only.

## Fresh replay windows used
- `logs/replay_mm3x15_20260212_115015` (previous benchmark window)
- `logs/replay_mm3x15_20260212_135829` (fresh run in this session)
- `logs/replay_mm3x15_20260212_141145` (fresh run in this session)

## Per-window top results (backtest harness)
- `logs/backtest_mm3x15_20260212_115015_grid_v3.json`
  - best: `order_count=24`, `mm_half_spread_cents=1.5`
  - net PnL: **$17.7878** (231 fills)
- `logs/backtest_mm3x15_20260212_135829_grid_v3.json`
  - best: `order_count=30`, `mm_half_spread_cents=2.0`
  - net PnL: **$23.4150** (15 fills; highly concentrated)
- `logs/backtest_mm3x15_20260212_141145_grid_v3.json`
  - best: `order_count=30`, `mm_half_spread_cents=0.5`
  - net PnL: **$12.5010** (1 fill; not informative)

## Replay quality diagnostics (new)
- `115015` window had healthy quoteability:
  - BTC quoteable ratio `0.6493`, ETH `0.6842`, SOL `0.6243`.
- `135829` and `141145` were mostly pinned/terminal-state:
  - Example (`135829`): BTC quoteable ratio `0.0087`, ETH `0.0493`, SOL `0.0982`.
  - Example (`141145`): BTC quoteable ratio `0.0637`, ETH `0.2803`, SOL `0.0000`.
- Conclusion: those two windows are weak evidence for strategy edge and must be down-weighted.

## Robustness passes

### Strict-ish filter
- report: `logs/backtest_mm3x15_robust_3windows_20260212_142200_v3_strict.json`
- key filters:
  - informative window requires `sum_fills >= 20`
  - min informative windows: 2
  - min window PnL >= 1 cent
  - min per-market PnL >= -100 cents
  - min total fills >= 120
  - max BTC adverse rate <= 0.35 (only on windows with BTC fills >= 5)
- best robust candidate:
  - `order_count=8`, `mm_half_spread_cents=1.5`
  - total robust PnL: **$8.5440**

### Relaxed downside filter
- report: `logs/backtest_mm3x15_robust_3windows_20260212_142200_v3_relaxed.json`
- change vs strict-ish: min per-market PnL relaxed from `-100` to `-150` cents
- best robust candidate:
  - `order_count=12`, `mm_half_spread_cents=1.5`
  - total robust PnL: **$10.8338**

## Interpretation
- There is a simulated path above $10 under relaxed robustness constraints.
- High headline PnL profiles are mostly concentration artifacts (few fills or one-market dominance) and should not be treated as proven edge.
- The newest windows (`135829`, `141145`) had high pinned-state ratios and low quoteability; they were treated as weak/non-informative.

## New tooling added for repeatable proof workflow
- `kalshi_hft/replay_mm3x15_grid.py`
  - runs parameter grid on `btc/eth/sol` replay bundle and writes ranked report.
- `kalshi_hft/replay_mm3x15_robust.py`
  - aggregates multiple grid reports and applies robustness filters.

## Important caveat
- The replay backtest harness approximates MM behavior through `strategy` parameters and does not fully replay every runtime branch in MM live engine (`mm` block logic).
- Use these results as candidate screening, not final live-proof.
