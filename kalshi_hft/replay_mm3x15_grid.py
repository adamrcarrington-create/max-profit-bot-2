from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from kalshi_hft.backtest import load_rows, run_backtest
from kalshi_hft.engine import BotConfig


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run MM-style replay grid for 3x15 captures (btc/eth/sol)."
    )
    parser.add_argument(
        "--capture-dir",
        required=True,
        help="Directory containing btc.jsonl, eth.jsonl, sol.jsonl replay files.",
    )
    parser.add_argument(
        "--btc-config",
        default=str(root / "config.paper_mm_3x15_btc.json"),
        help="Config path for BTC replay.",
    )
    parser.add_argument(
        "--eth-config",
        default=str(root / "config.paper_mm_3x15_eth.json"),
        help="Config path for ETH replay.",
    )
    parser.add_argument(
        "--sol-config",
        default=str(root / "config.paper_mm_3x15_sol.json"),
        help="Config path for SOL replay.",
    )
    parser.add_argument(
        "--order-counts",
        default="1,2,4,6,8,10,12,16,20,24,30",
        help="Comma-separated order_count grid.",
    )
    parser.add_argument(
        "--spreads",
        default="0.5,1.0,1.5,2.0",
        help="Comma-separated mm_half_spread_cents grid.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top rows to include.",
    )
    parser.add_argument(
        "--worst-n",
        type=int,
        default=5,
        help="Number of worst rows to include.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON summary path.",
    )
    return parser.parse_args()


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw).split(","):
        text = token.strip()
        if not text:
            continue
        out.append(int(text))
    if not out:
        raise ValueError("order-count list is empty")
    return sorted(set(out))


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for token in str(raw).split(","):
        text = token.strip()
        if not text:
            continue
        out.append(float(text))
    if not out:
        raise ValueError("spread list is empty")
    return sorted(set(out))


def _market_inputs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        "btc": {
            "config": Path(args.btc_config),
            "replay": Path(args.capture_dir) / "btc.jsonl",
        },
        "eth": {
            "config": Path(args.eth_config),
            "replay": Path(args.capture_dir) / "eth.jsonl",
        },
        "sol": {
            "config": Path(args.sol_config),
            "replay": Path(args.capture_dir) / "sol.jsonl",
        },
    }


def _apply_overrides(base_cfg: BotConfig, *, order_count: int, spread: float) -> BotConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.strategy.base_order_count = int(order_count)
    cfg.strategy.max_order_count = int(order_count)
    cfg.strategy.mm_half_spread_cents = float(spread)
    return cfg


def _replay_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total <= 0:
        return {
            "rows": 0,
            "quoteable_rows": 0,
            "quoteable_ratio": 0.0,
            "pinned_rows": 0,
            "pinned_ratio": 0.0,
            "missing_yes_pair_rows": 0,
            "missing_yes_pair_ratio": 0.0,
        }

    quoteable = 0
    pinned = 0
    missing_yes_pair = 0
    for row in rows:
        yes_bid_raw = row.get("best_yes_bid")
        yes_ask_raw = row.get("best_yes_ask")
        no_bid_raw = row.get("best_no_bid")
        no_ask_raw = row.get("best_no_ask")

        yes_bid = int(yes_bid_raw) if yes_bid_raw is not None else None
        yes_ask = int(yes_ask_raw) if yes_ask_raw is not None else None
        no_bid = int(no_bid_raw) if no_bid_raw is not None else None
        no_ask = int(no_ask_raw) if no_ask_raw is not None else None

        if yes_bid is None or yes_ask is None:
            missing_yes_pair += 1
        else:
            spread_yes = yes_ask - yes_bid
            if 1 <= spread_yes <= 20:
                quoteable += 1

        if (
            (yes_bid is not None and yes_bid <= 1)
            or (yes_ask is not None and yes_ask >= 99)
            or (no_bid is not None and no_bid <= 1)
            or (no_ask is not None and no_ask >= 99)
        ):
            pinned += 1

    return {
        "rows": total,
        "quoteable_rows": quoteable,
        "quoteable_ratio": round(quoteable / total, 4),
        "pinned_rows": pinned,
        "pinned_ratio": round(pinned / total, 4),
        "missing_yes_pair_rows": missing_yes_pair,
        "missing_yes_pair_ratio": round(missing_yes_pair / total, 4),
    }


def _round_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "order_count": int(row["order_count"]),
        "mm_half_spread_cents": round(float(row["mm_half_spread_cents"]), 4),
        "sum_pnl_cents": round(float(row["sum_pnl_cents"]), 4),
        "sum_pnl_dollars": round(float(row["sum_pnl_cents"]) / 100.0, 4),
        "sum_fills": int(row["sum_fills"]),
        "sum_cost_drag_cents": round(float(row["sum_cost_drag_cents"]), 4),
        "sum_adv_rate": round(float(row["sum_adv_rate"]), 4),
        "min_market_pnl_cents": round(float(row["min_market_pnl_cents"]), 4),
        "all_markets_positive": bool(row["all_markets_positive"]),
        "per_market": row["per_market"],
    }


def main() -> int:
    args = parse_args()
    market_inputs = _market_inputs(args)
    order_counts = _parse_int_list(args.order_counts)
    spreads = _parse_float_list(args.spreads)

    data: dict[str, Any] = {}
    for market, payload in market_inputs.items():
        replay_path = Path(payload["replay"])
        config_path = Path(payload["config"])
        if not replay_path.exists():
            raise FileNotFoundError(f"missing replay file for {market}: {replay_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"missing config for {market}: {config_path}")
        rows = load_rows(replay_path)
        data[market] = {
            "rows": rows,
            "quality": _replay_quality(rows),
            "base_cfg": BotConfig.from_json(str(config_path)),
            "replay_path": str(replay_path),
            "config_path": str(config_path),
        }

    results: list[dict[str, Any]] = []
    for order_count in order_counts:
        for spread in spreads:
            per_market: dict[str, Any] = {}
            sum_pnl = 0.0
            sum_fills = 0
            sum_cost = 0.0
            sum_adv = 0.0
            min_market_pnl = float("inf")
            all_positive = True

            for market, payload in data.items():
                cfg = _apply_overrides(
                    payload["base_cfg"],
                    order_count=int(order_count),
                    spread=float(spread),
                )
                report = run_backtest(cfg, payload["rows"])
                pnl = float(report.get("pnl_cents_mark_to_market") or 0.0)
                fills = int(report.get("fills") or 0)
                cost = float(report.get("cost_drag_cents") or 0.0)
                adv = float(report.get("adverse_selection_rate") or 0.0)
                ending_gross = int((report.get("ending_inventory") or {}).get("gross") or 0)

                per_market[market] = {
                    "pnl_cents": round(pnl, 4),
                    "fills": fills,
                    "cost_drag_cents": round(cost, 4),
                    "adv_rate": round(adv, 4),
                    "ending_gross": ending_gross,
                }
                sum_pnl += pnl
                sum_fills += fills
                sum_cost += cost
                sum_adv += adv
                min_market_pnl = min(min_market_pnl, pnl)
                if pnl <= 0.0:
                    all_positive = False

            results.append(
                {
                    "order_count": int(order_count),
                    "mm_half_spread_cents": float(spread),
                    "sum_pnl_cents": float(sum_pnl),
                    "sum_fills": int(sum_fills),
                    "sum_cost_drag_cents": float(sum_cost),
                    "sum_adv_rate": float(sum_adv / max(1, len(data))),
                    "min_market_pnl_cents": float(min_market_pnl),
                    "all_markets_positive": bool(all_positive),
                    "per_market": per_market,
                }
            )

    results.sort(
        key=lambda row: (
            float(row["sum_pnl_cents"]),
            float(row["min_market_pnl_cents"]),
            int(row["sum_fills"]),
        ),
        reverse=True,
    )

    best_all_positive = next((row for row in results if row["all_markets_positive"]), None)
    payload: dict[str, Any] = {
        "capture_dir": str(Path(args.capture_dir)),
        "market_inputs": {
            market: {
                "config": payload["config_path"],
                "replay": payload["replay_path"],
                "rows": len(payload["rows"]),
                "quality": payload["quality"],
            }
            for market, payload in data.items()
        },
        "grid": {
            "order_counts": order_counts,
            "spreads": spreads,
            "tested": len(results),
        },
        "best_overall": _round_row(results[0]) if results else None,
        "best_all_markets_positive": _round_row(best_all_positive) if best_all_positive else None,
        "all_results": [_round_row(row) for row in results],
        "top": [_round_row(row) for row in results[: max(0, int(args.top_n))]],
        "worst": [_round_row(row) for row in results[-max(0, int(args.worst_n)) :]],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"report={out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
