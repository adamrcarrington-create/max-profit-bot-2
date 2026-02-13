from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kalshi_hft.client import KalshiClient, KalshiRequestError
from kalshi_hft.engine import BotConfig, compact_json, extract_float


def _to_dollars(row: dict[str, Any], cents_key: str, dollars_key: str) -> float:
    if dollars_key in row:
        value = extract_float(row.get(dollars_key))
        if value is not None:
            return float(value)
    value = extract_float(row.get(cents_key))
    if value is None:
        return 0.0
    return float(value) / 100.0


def capture_snapshot(*, config_path: str, series: list[str]) -> dict[str, Any]:
    cfg = BotConfig.from_json(config_path)
    client = KalshiClient(cfg.kalshi)
    try:
        payload = client.get_positions(limit=1000)
    except KalshiRequestError as exc:
        raise RuntimeError(f"positions_fetch_failed err={exc}") from exc

    rows = payload.get("market_positions")
    if not isinstance(rows, list):
        rows = payload.get("positions")
    if not isinstance(rows, list):
        rows = []
    typed_rows = [row for row in rows if isinstance(row, dict)]

    out: dict[str, Any] = {}
    for series_ticker in series:
        matches = [
            row
            for row in typed_rows
            if str(row.get("ticker") or "").strip().startswith(f"{series_ticker}-")
        ]
        active_ticker = ""
        try:
            market = client.resolve_active_market(series_ticker)
            active_ticker = str((market.get("market") or market).get("ticker") or "").strip()
        except KalshiRequestError:
            active_ticker = ""

        if not matches:
            out[series_ticker] = {
                "ticker": active_ticker,
                "position": 0,
                "market_exposure_dollars": 0.0,
                "realized_pnl_dollars": 0.0,
                "fees_paid_dollars": 0.0,
                "total_traded_dollars": 0.0,
            }
            continue

        out[series_ticker] = {
            "ticker": str(matches[0].get("ticker") or active_ticker),
            "position": int(sum(extract_float(row.get("position")) or 0.0 for row in matches)),
            "market_exposure_dollars": round(
                sum(_to_dollars(row, "market_exposure", "market_exposure_dollars") for row in matches), 6
            ),
            "realized_pnl_dollars": round(
                sum(_to_dollars(row, "realized_pnl", "realized_pnl_dollars") for row in matches), 6
            ),
            "fees_paid_dollars": round(
                sum(_to_dollars(row, "fees_paid", "fees_paid_dollars") for row in matches), 6
            ),
            "total_traded_dollars": round(
                sum(_to_dollars(row, "total_traded", "total_traded_dollars") for row in matches), 6
            ),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture per-series position/PnL snapshot for MM attribution.")
    parser.add_argument("--config", required=True, help="Path to JSON config with Kalshi credentials.")
    parser.add_argument("--output", required=True, help="Path to write JSON snapshot.")
    parser.add_argument(
        "--series",
        nargs="+",
        default=["KXBTC15M", "KXETH15M", "KXSOL15M"],
        help="Series tickers to summarize.",
    )
    args = parser.parse_args()

    snapshot = capture_snapshot(config_path=args.config, series=[str(s).strip().upper() for s in args.series])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot), encoding="utf-8")
    print(f"snapshot_written path={out_path} payload={compact_json(snapshot)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
