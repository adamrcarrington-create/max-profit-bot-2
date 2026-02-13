from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional

from kalshi_hft.client import KalshiClient, KalshiRequestError, extract_float, extract_timestamp_ms
from kalshi_hft.engine import BotConfig, parse_ticker_close_epoch_ms
from kalshi_hft.monitor import BinanceSpotMonitor, KalshiOrderbookMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture Kalshi orderbook + reference price replay file for backtesting."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.example.json"),
        help="Path to bot config JSON.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=300.0,
        help="Capture duration in seconds.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=0.5,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="",
        help="Optional explicit market ticker. If omitted, resolves active market from series_ticker.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output JSONL file path.",
    )
    parser.add_argument(
        "--prefer-quoteable-mm-market",
        action="store_true",
        help="When ticker is omitted, resolve a quoteable MM market instead of earliest-close market.",
    )
    return parser.parse_args()


def _unwrap_market(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("market"), dict):
        return dict(payload["market"])
    return dict(payload)


def _infer_direction(market: Dict[str, Any]) -> str:
    title = str(market.get("title") or "").lower()
    yes_sub = str(market.get("yes_sub_title") or "").lower()
    if any(x in title for x in ["down", "below", "under"]):
        return "down"
    if any(x in yes_sub for x in ["below", "under"]):
        return "down"
    return "up"


def _extract_strike(market: Dict[str, Any]) -> Optional[float]:
    return extract_float(
        market.get("strike")
        or market.get("strike_price")
        or market.get("target")
        or market.get("target_price")
        or market.get("floor_strike")
    )


def _extract_close_ms(market: Dict[str, Any]) -> Optional[int]:
    close_ms = extract_timestamp_ms(
        market.get("close_time")
        or market.get("close_date")
        or market.get("expiration_time")
        or market.get("end_time")
        or market.get("close_ts")
    )
    if close_ms is not None:
        return int(close_ms)
    return parse_ticker_close_epoch_ms(str(market.get("ticker") or ""))


def _market_top_of_book(client: KalshiClient, ticker: str) -> Dict[str, Optional[int]]:
    payload = client.get_market(ticker)
    market = _unwrap_market(payload)
    yes_bid = extract_float(market.get("yes_bid"))
    yes_ask = extract_float(market.get("yes_ask"))
    no_bid = extract_float(market.get("no_bid"))
    no_ask = extract_float(market.get("no_ask"))
    return {
        "best_yes_bid": int(round(yes_bid)) if yes_bid is not None else None,
        "best_yes_ask": int(round(yes_ask)) if yes_ask is not None else None,
        "best_no_bid": int(round(no_bid)) if no_bid is not None else None,
        "best_no_ask": int(round(no_ask)) if no_ask is not None else None,
    }


def main() -> int:
    args = parse_args()
    cfg = BotConfig.from_json(args.config)
    client = KalshiClient(cfg.kalshi)

    ticker = args.ticker.strip()
    market_selection = "explicit"
    if not ticker:
        if args.prefer_quoteable_mm_market:
            try:
                market = client.resolve_quoteable_mm_market(
                    series_ticker=cfg.strategy.series_ticker,
                    min_seconds_to_close=float(cfg.risk.min_seconds_to_close),
                    min_spread_cents=int(cfg.mm.min_spread_cents),
                    max_spread_cents=int(cfg.mm.max_spread_cents),
                    target_mid_cents=float(cfg.mm.market_target_mid_cents),
                    max_mid_distance_cents=float(cfg.mm.market_max_mid_distance_cents),
                )
                market_selection = "quoteable_mm"
            except KalshiRequestError as exc:
                print(f"market_select_fallback reason={exc}")
                market = client.resolve_active_market(cfg.strategy.series_ticker)
                market_selection = "active_fallback"
        else:
            market = client.resolve_active_market(cfg.strategy.series_ticker)
            market_selection = "active_market"
        ticker = str(market.get("ticker") or "").strip()
        if not ticker:
            raise RuntimeError("resolve_active_market returned no ticker")

    market_payload = client.get_market(ticker)
    market = _unwrap_market(market_payload)
    strike = _extract_strike(market)
    direction = _infer_direction(market)
    close_ms = _extract_close_ms(market)

    symbol = cfg.strategy.binance_symbol.strip().upper() or "BTCUSDT"
    spot_monitor = BinanceSpotMonitor(
        symbol=symbol,
        window_seconds=cfg.strategy.momentum_window_seconds,
        include_perp_reference=cfg.strategy.include_perp_reference,
        perp_weight=cfg.strategy.perp_weight,
        cross_exchange_max_diff_pct=cfg.strategy.cross_exchange_max_diff_pct,
        flash_spike_pause_seconds=cfg.strategy.flash_spike_pause_seconds,
    )
    book_monitor = KalshiOrderbookMonitor(client=client, ticker=ticker)
    book_monitor.refresh_from_rest()
    book_monitor.start()

    out_path = Path(args.output.strip()) if args.output.strip() else None
    if out_path is None:
        out_path = Path("logs") / f"replay_capture_{ticker}_{int(time.time())}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    duration = max(1.0, float(args.duration_seconds))
    interval = max(0.1, float(args.interval_seconds))
    deadline = time.time() + duration
    rows = 0
    errors = 0
    last_market_fallback_fetch_epoch = 0.0
    market_quote_cache: Dict[str, Optional[int]] = {
        "best_yes_bid": None,
        "best_yes_ask": None,
        "best_no_bid": None,
        "best_no_ask": None,
    }

    print(
        f"replay_capture_start ticker={ticker} market_selection={market_selection} "
        f"strike={strike} direction={direction} "
        f"duration_s={duration} interval_s={interval} out={out_path}"
    )
    try:
        with out_path.open("w", encoding="utf-8") as f:
            while time.time() < deadline:
                ts_ms = int(time.time() * 1000)
                try:
                    ref = float(spot_monitor.poll_once())
                    top = book_monitor.top_of_book()
                    if top.get("best_yes_bid") is None and top.get("best_no_bid") is None:
                        now = time.time()
                        # Throttle fallback REST quote refreshes.
                        if now - last_market_fallback_fetch_epoch >= 1.0:
                            last_market_fallback_fetch_epoch = now
                            market_quote_cache = _market_top_of_book(client, ticker)
                        top = market_quote_cache
                    yes_bid = top.get("best_yes_bid")
                    yes_ask = top.get("best_yes_ask")
                    no_bid = top.get("best_no_bid")
                    no_ask = top.get("best_no_ask")
                    spread_yes = None
                    if yes_bid is not None and yes_ask is not None:
                        spread_yes = int(yes_ask - yes_bid)

                    row = {
                        "timestamp_ms": ts_ms,
                        "ticker": ticker,
                        "ref_price": ref,
                        "best_yes_bid": yes_bid,
                        "best_yes_ask": yes_ask,
                        "best_no_bid": no_bid,
                        "best_no_ask": no_ask,
                        "spread_yes": spread_yes,
                        "direction": direction,
                        "strike": strike,
                        "source": spot_monitor.last_reference_source,
                        "cross_exchange_diff_pct": spot_monitor.last_cross_exchange_diff_pct,
                        "flash_spike_pause_remaining_seconds": spot_monitor.pause_remaining_seconds(),
                        "book_age_seconds": book_monitor.book_age_seconds(),
                        "book_stale_seconds": book_monitor.book_stale_seconds(),
                        "seconds_to_close": ((close_ms - ts_ms) / 1000.0) if close_ms is not None else None,
                    }
                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
                    rows += 1
                    if rows % 20 == 0:
                        print(
                            f"capture_progress rows={rows} ref={ref:.2f} "
                            f"yes_bid={yes_bid} yes_ask={yes_ask} "
                            f"source={spot_monitor.last_reference_source}"
                        )
                except Exception as exc:
                    errors += 1
                    if errors <= 8:
                        print(f"capture_error={exc}")
                time.sleep(interval)
    finally:
        book_monitor.stop()

    print(f"replay_capture_done rows={rows} errors={errors} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
