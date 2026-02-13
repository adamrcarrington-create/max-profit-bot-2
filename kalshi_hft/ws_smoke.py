from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from kalshi_hft.client import KalshiAuthConfig, KalshiClient, KalshiRequestError
from kalshi_hft.monitor import KalshiOrderbookMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kalshi websocket smoke test for orderbook feed.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.example.json"),
        help="Path to bot config JSON.",
    )
    parser.add_argument("--ticker", type=str, default="", help="Specific market ticker.")
    parser.add_argument("--series-ticker", type=str, default="", help="Series ticker for active market resolution.")
    parser.add_argument("--duration-seconds", type=float, default=12.0, help="How long to listen before verdict.")
    return parser.parse_args()


def load_auth(path: str) -> tuple[KalshiAuthConfig, dict]:
    raw = json.loads(Path(path).read_text())
    kalshi_raw = raw.get("kalshi", {})
    strategy_raw = raw.get("strategy", {})
    auth = KalshiAuthConfig(**kalshi_raw)
    return auth, strategy_raw


def main() -> int:
    args = parse_args()
    auth, strategy_cfg = load_auth(args.config)
    key_id = (auth.api_key_id or "").strip()
    if not key_id or "REPLACE_WITH_KALSHI_KEY_ID" in key_id:
        print(
            "config_error=missing_real_api_key_id "
            "detail='Set kalshi.api_key_id in your config to your real Kalshi key UUID.'"
        )
        return 4

    client = KalshiClient(auth)
    preflight_ok = False
    preflight_error = ""

    try:
        client.get_balance()
        preflight_ok = True
        print(f"auth_preflight=ok pss_mode={client.auth.pss_salt_mode}")
    except Exception as exc:
        preflight_error = str(exc)
        print(f"auth_preflight=failed pss_mode={client.auth.pss_salt_mode} err={preflight_error}")

    if not preflight_ok and client.auth.pss_salt_mode != "max":
        alt_auth = KalshiAuthConfig(**{**auth.__dict__, "pss_salt_mode": "max"})
        alt_client = KalshiClient(alt_auth)
        try:
            alt_client.get_balance()
            client = alt_client
            preflight_ok = True
            print("auth_preflight_alt=ok pss_mode=max")
        except KalshiRequestError as exc:
            print(f"auth_preflight_alt=failed pss_mode=max err={exc}")
        except Exception as exc:
            print(f"auth_preflight_alt=failed pss_mode=max err={exc}")

    ticker = args.ticker.strip()
    if not ticker:
        from_cfg = str(strategy_cfg.get("market_ticker") or "").strip()
        if from_cfg:
            ticker = from_cfg
        else:
            series = args.series_ticker.strip() or str(strategy_cfg.get("series_ticker") or "KXBTC15M")
            market = client.resolve_active_market(series)
            ticker = str(market.get("ticker") or "").strip()
    if not ticker:
        print("unable to resolve ticker for websocket smoke test")
        return 2

    print(
        f"ws_smoke_start ticker={ticker} ws_url={client.ws_url} pss_mode={client.auth.pss_salt_mode}"
    )
    monitor = KalshiOrderbookMonitor(client=client, ticker=ticker)
    try:
        monitor.refresh_from_rest()
    except Exception as exc:
        print(f"rest_seed_error={exc}")
    monitor.start()

    start = time.time()
    saw_ws_messages = False
    try:
        while time.time() - start < max(3.0, args.duration_seconds):
            time.sleep(1.0)
            micro = monitor.microstructure()
            err_suffix = f" last_error={monitor.last_error}" if monitor.last_error else ""
            print(
                "micro "
                f"yes_bid={micro['best_yes_bid']} yes_ask={micro['best_yes_ask']} "
                f"no_bid={micro['best_no_bid']} no_ask={micro['best_no_ask']} "
                f"spread_yes={micro['spread_yes']} imbalance={micro['imbalance']:.3f} "
                f"book_age={micro['book_age_seconds']:.2f} book_stale={micro['book_stale_seconds']:.2f}"
                f"{err_suffix}"
            )
            if monitor.last_message_epoch > 0:
                saw_ws_messages = True
    finally:
        monitor.stop()

    if saw_ws_messages:
        print("ws_smoke_status=ok")
        return 0

    err = monitor.last_error or "no websocket messages received"
    print(f"ws_smoke_status=failed detail={err}")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
