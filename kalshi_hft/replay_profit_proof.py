from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from kalshi_hft.backtest import load_rows, run_backtest
from kalshi_hft.engine import BotConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay proof harness for old MM-derived parameter variants."
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.paper_robust.json"),
        help="Base config JSON.",
    )
    parser.add_argument(
        "--replay",
        action="append",
        default=[],
        help="Replay file path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/replay_profit_proof_made_money_latest.json",
        help="Output summary JSON path.",
    )
    return parser.parse_args()


def _apply_overrides(cfg: BotConfig, overrides: dict[str, Any]) -> BotConfig:
    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        head, tail = key.split(".", 1)
        setattr(getattr(out, head), tail, value)
    return out


def _default_replays(root: Path) -> list[Path]:
    return [
        root / "logs/replay_capture_profit_probe_fresh.jsonl",
        root / "logs/replay_capture_profit_probe_run2_frozen.jsonl",
    ]


def _common_overrides() -> dict[str, Any]:
    return {
        "strategy.edge_threshold": 0.0,
        "strategy.model_edge_multiplier": 0.0,
        "strategy.maker_only": True,
        "strategy.snipe_edge_threshold": 9.0,
        "strategy.cross_side_arb_enabled": True,
        "strategy.cross_side_arb_min_edge_cents": 0.6,
        "strategy.cross_side_arb_max_count": 1,
        "strategy.inventory_exit_enabled": True,
        "strategy.inventory_exit_min_edge_cents": -0.5,
        "strategy.max_yes_spread_cents_for_mm": 20.0,
    }


def _variant_overrides() -> dict[str, dict[str, Any]]:
    return {
        "made_money_defaults": {
            "strategy.base_order_count": 2,
            "strategy.max_order_count": 2,
            "strategy.mm_half_spread_cents": 1.0,
            "strategy.min_ev_cents_per_contract": 0.3,
            "execution.max_orders_per_minute": 6,
            "risk.max_gross_position": 18,
            "risk.max_net_exposure": 8,
        },
        "safer_notes_defaults": {
            "strategy.base_order_count": 1,
            "strategy.max_order_count": 1,
            "strategy.mm_half_spread_cents": 1.5,
            "strategy.min_ev_cents_per_contract": 0.4,
            "execution.max_orders_per_minute": 4,
            "risk.max_gross_position": 10,
            "risk.max_net_exposure": 4,
        },
        "scaled_profit_proof": {
            "strategy.base_order_count": 24,
            "strategy.max_order_count": 24,
            "strategy.mm_half_spread_cents": 1.0,
            "strategy.min_ev_cents_per_contract": 0.2,
            "execution.max_orders_per_minute": 60,
            "risk.max_gross_position": 120,
            "risk.max_net_exposure": 120,
        },
        "scaled_extreme_profit_proof": {
            "strategy.base_order_count": 60,
            "strategy.max_order_count": 60,
            "strategy.mm_half_spread_cents": 1.0,
            "strategy.min_ev_cents_per_contract": 0.2,
            "execution.max_orders_per_minute": 6,
            "risk.max_gross_position": 1600,
            "risk.max_net_exposure": 1600,
        },
    }


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    base_cfg = BotConfig.from_json(args.base_config)

    replay_paths = [Path(p) for p in args.replay] if args.replay else _default_replays(root)
    replay_rows: list[tuple[str, list[dict[str, Any]]]] = []
    for replay in replay_paths:
        if not replay.exists():
            continue
        replay_rows.append((replay.name, load_rows(replay)))

    if not replay_rows:
        raise RuntimeError("no replay files found")

    common = _common_overrides()
    results: dict[str, Any] = {}
    for variant_name, overrides in _variant_overrides().items():
        cfg = _apply_overrides(base_cfg, {**common, **overrides})
        pnl_sum = 0.0
        fills_sum = 0
        max_ending_gross = 0
        per_file: dict[str, Any] = {}
        for file_name, rows in replay_rows:
            report = run_backtest(cfg, rows)
            pnl = float(report.get("pnl_cents_mark_to_market") or 0.0)
            fills = int(report.get("fills") or 0)
            ending_gross = int((report.get("ending_inventory") or {}).get("gross") or 0)
            per_file[file_name] = {
                "pnl_cents": round(pnl, 4),
                "fills": fills,
                "cost_drag_cents": round(float(report.get("cost_drag_cents") or 0.0), 4),
                "ending_gross": ending_gross,
                "adverse_selection_rate": round(float(report.get("adverse_selection_rate") or 0.0), 4),
            }
            pnl_sum += pnl
            fills_sum += fills
            max_ending_gross = max(max_ending_gross, ending_gross)
        results[variant_name] = {
            "pnl_sum_cents": round(pnl_sum, 4),
            "pnl_sum_dollars": round(pnl_sum / 100.0, 4),
            "fills_sum": fills_sum,
            "max_ending_gross": max_ending_gross,
            "per_file": per_file,
        }

    payload = {
        "base_config": str(Path(args.base_config).resolve()),
        "replay_files": [name for name, _ in replay_rows],
        "common_overrides": common,
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"report={out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
