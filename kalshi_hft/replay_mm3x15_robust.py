from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple replay_mm3x15_grid reports and rank robust MM candidates."
    )
    parser.add_argument(
        "--grid",
        action="append",
        required=True,
        help="Path to backtest grid JSON produced by replay_mm3x15_grid.",
    )
    parser.add_argument(
        "--min-window-pnl-cents",
        type=float,
        default=1.0,
        help="Minimum required total PnL per window (cents).",
    )
    parser.add_argument(
        "--min-market-pnl-cents",
        type=float,
        default=-50.0,
        help="Minimum allowed per-market PnL in any window (cents).",
    )
    parser.add_argument(
        "--min-total-fills",
        type=int,
        default=100,
        help="Minimum aggregate fill count across all windows.",
    )
    parser.add_argument(
        "--min-eth-fills",
        type=int,
        default=20,
        help="Minimum aggregate ETH fills across all windows.",
    )
    parser.add_argument(
        "--max-btc-adverse-rate",
        type=float,
        default=0.35,
        help="Maximum allowed BTC adverse selection rate in any window.",
    )
    parser.add_argument(
        "--min-btc-fills",
        type=int,
        default=20,
        help="Minimum aggregate BTC fills across all windows.",
    )
    parser.add_argument(
        "--btc-min-fills-for-adv",
        type=int,
        default=5,
        help="Apply BTC adverse-rate filter only on windows with at least this many BTC fills.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of robust candidates to output.",
    )
    parser.add_argument(
        "--min-window-fills-informative",
        type=int,
        default=20,
        help="Treat a window as informative only if sum_fills >= this threshold.",
    )
    parser.add_argument(
        "--min-informative-windows",
        type=int,
        default=2,
        help="Minimum informative windows required for a candidate.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path.",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> dict[tuple[int, float], dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("all_results")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"grid has no all_results rows: {path}")
    out: dict[tuple[int, float], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["order_count"]), float(row["mm_half_spread_cents"]))
        out[key] = row
    return out


def _round_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "order_count": int(row["order_count"]),
        "mm_half_spread_cents": round(float(row["mm_half_spread_cents"]), 4),
        "total_pnl_cents": round(float(row["total_pnl_cents"]), 4),
        "total_pnl_dollars": round(float(row["total_pnl_cents"]) / 100.0, 4),
        "informative_windows": int(row["informative_windows"]),
        "min_window_pnl_cents": round(float(row["min_window_pnl_cents"]), 4),
        "min_market_pnl_cents": round(float(row["min_market_pnl_cents"]), 4),
        "total_fills": int(row["total_fills"]),
        "btc_fills": int(row["btc_fills"]),
        "eth_fills": int(row["eth_fills"]),
        "max_btc_adv_rate": round(float(row["max_btc_adv_rate"]), 4),
        "btc_adv_rate_qualified_windows": int(row["btc_adv_rate_qualified_windows"]),
        "windows": row["windows"],
    }


def main() -> int:
    args = parse_args()
    grid_paths = [Path(p) for p in args.grid]
    if len(grid_paths) < 2:
        raise RuntimeError("at least two --grid files are required for robustness checks")

    windows: list[dict[tuple[int, float], dict[str, Any]]] = []
    for path in grid_paths:
        if not path.exists():
            raise FileNotFoundError(f"missing grid file: {path}")
        windows.append(_read_rows(path))

    keys = set(windows[0].keys())
    for rows in windows[1:]:
        keys &= set(rows.keys())
    if not keys:
        raise RuntimeError("no shared parameter keys across grid files")

    scored: list[dict[str, Any]] = []
    for key in sorted(keys):
        order_count, spread = key
        total_pnl = 0.0
        min_window_pnl = float("inf")
        min_market_pnl = float("inf")
        total_fills = 0
        btc_fills = 0
        eth_fills = 0
        max_btc_adv_rate = 0.0
        btc_adv_rate_qualified_windows = 0
        informative_windows = 0
        window_rows: list[dict[str, Any]] = []

        for rows in windows:
            row = rows[key]
            sum_pnl = float(row.get("sum_pnl_cents") or 0.0)
            sum_fills = int(row.get("sum_fills") or 0)
            is_informative = sum_fills >= int(args.min_window_fills_informative)

            per_market = row.get("per_market") or {}
            market_pnls: list[float] = []
            btc_fill_count = int((per_market.get("btc") or {}).get("fills") or 0)
            btc_adv_rate = float((per_market.get("btc") or {}).get("adv_rate") or 0.0)
            eth_fill_count = int((per_market.get("eth") or {}).get("fills") or 0)

            if is_informative:
                informative_windows += 1
                total_pnl += sum_pnl
                min_window_pnl = min(min_window_pnl, sum_pnl)
                total_fills += sum_fills
                btc_fills += btc_fill_count
                eth_fills += eth_fill_count
                if btc_fill_count >= int(args.btc_min_fills_for_adv):
                    btc_adv_rate_qualified_windows += 1
                    max_btc_adv_rate = max(max_btc_adv_rate, btc_adv_rate)

            for market_name, market_row in per_market.items():
                if not isinstance(market_row, dict):
                    continue
                pnl = float(market_row.get("pnl_cents") or 0.0)
                market_pnls.append(pnl)
            if market_pnls and is_informative:
                min_market_pnl = min(min_market_pnl, min(market_pnls))

            window_rows.append(
                {
                    "sum_pnl_cents": round(sum_pnl, 4),
                    "sum_fills": sum_fills,
                    "informative": bool(is_informative),
                    "min_market_pnl_cents": round(min(market_pnls) if market_pnls else 0.0, 4),
                    "eth_fills": eth_fill_count,
                    "btc_fills": btc_fill_count,
                    "btc_adv_rate": round(btc_adv_rate, 4),
                }
            )

        if informative_windows <= 0:
            min_window_pnl = 0.0
            min_market_pnl = 0.0

        scored.append(
            {
                "order_count": int(order_count),
                "mm_half_spread_cents": float(spread),
                "total_pnl_cents": float(total_pnl),
                "informative_windows": int(informative_windows),
                "min_window_pnl_cents": float(min_window_pnl),
                "min_market_pnl_cents": float(min_market_pnl),
                "total_fills": int(total_fills),
                "btc_fills": int(btc_fills),
                "eth_fills": int(eth_fills),
                "max_btc_adv_rate": float(max_btc_adv_rate),
                "btc_adv_rate_qualified_windows": int(btc_adv_rate_qualified_windows),
                "windows": window_rows,
            }
        )

    robust = [
        row
        for row in scored
        if row["informative_windows"] >= int(args.min_informative_windows)
        and row["min_window_pnl_cents"] >= float(args.min_window_pnl_cents)
        and row["min_market_pnl_cents"] >= float(args.min_market_pnl_cents)
        and row["total_fills"] >= int(args.min_total_fills)
        and row["btc_fills"] >= int(args.min_btc_fills)
        and row["eth_fills"] >= int(args.min_eth_fills)
        and (
            row["btc_adv_rate_qualified_windows"] <= 0
            or row["max_btc_adv_rate"] <= float(args.max_btc_adverse_rate)
        )
    ]
    robust.sort(
        key=lambda row: (
            float(row["total_pnl_cents"]),
            float(row["min_window_pnl_cents"]),
            -float(row["max_btc_adv_rate"]),
        ),
        reverse=True,
    )

    scored.sort(
        key=lambda row: (
            float(row["total_pnl_cents"]),
            float(row["min_window_pnl_cents"]),
            -float(row["max_btc_adv_rate"]),
        ),
        reverse=True,
    )

    payload = {
        "grid_files": [str(p) for p in grid_paths],
        "filters": {
            "min_window_pnl_cents": float(args.min_window_pnl_cents),
            "min_informative_windows": int(args.min_informative_windows),
            "min_window_fills_informative": int(args.min_window_fills_informative),
            "min_market_pnl_cents": float(args.min_market_pnl_cents),
            "min_total_fills": int(args.min_total_fills),
            "min_btc_fills": int(args.min_btc_fills),
            "min_eth_fills": int(args.min_eth_fills),
            "max_btc_adverse_rate": float(args.max_btc_adverse_rate),
            "btc_min_fills_for_adv": int(args.btc_min_fills_for_adv),
        },
        "combos_tested": len(scored),
        "robust_count": len(robust),
        "best_robust": _round_candidate(robust[0]) if robust else None,
        "top_robust": [_round_candidate(row) for row in robust[: max(0, int(args.top_n))]],
        "top_all": [_round_candidate(row) for row in scored[: max(0, int(args.top_n))]],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"report={out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
