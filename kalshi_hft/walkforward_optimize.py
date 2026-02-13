from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from kalshi_hft.backtest import load_rows, run_backtest, row_timestamp_ms
from kalshi_hft.engine import BotConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward optimizer for Kalshi HFT backtest parameters."
    )
    parser.add_argument(
        "--inputs-glob",
        type=str,
        default="logs/replay_capture*.jsonl",
        help="Glob for replay files (JSONL/CSV).",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.live_maker_2h.json"),
        help="Base config JSON used as the starting point.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/walkforward_report.json",
        help="Output report JSON path.",
    )
    parser.add_argument(
        "--write-best-config",
        type=str,
        default="logs/config.walkforward_best.json",
        help="Where to write the best candidate config JSON.",
    )
    parser.add_argument(
        "--min-file-rows",
        type=int,
        default=80,
        help="Skip replay files with fewer usable rows than this.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Chronological train ratio per file.",
    )
    return parser.parse_args()


def _usable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        if row_timestamp_ms(row) is None:
            continue
        kept.append(row)
    kept.sort(key=lambda r: row_timestamp_ms(r) or 0)
    return kept


def _split_train_test(rows: list[dict[str, Any]], train_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []
    n = len(rows)
    cut = int(max(1, min(n - 1, round(n * train_ratio))))
    return rows[:cut], rows[cut:]


def _replay_files(glob_expr: str) -> list[Path]:
    return sorted(Path().glob(glob_expr))


def _candidate_grid() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    edge_thresholds = [0.02, 0.05, 0.08]
    min_evs = [0.0, 0.2, 0.5]
    maker_slips = [0.2, 0.35]
    mm_half_spreads = [0.5, 1.0, 1.5]
    max_yes_spreads_mm = [30.0, 45.0, 60.0]
    model_edge_multipliers = [0.0, 0.1, 0.2]
    max_orders = [1, 2]
    maker_only_vals = [True]
    snipe_thresholds = [9.0]
    max_order_rate_per_min = [2, 4]
    gross_limits = [2, 6, 10]

    for edge in edge_thresholds:
        for min_ev in min_evs:
            for mslip in maker_slips:
                for mm_half in mm_half_spreads:
                    for max_yes_spread in max_yes_spreads_mm:
                        for model_mult in model_edge_multipliers:
                            for max_order in max_orders:
                                for maker_only in maker_only_vals:
                                    if maker_only:
                                        snipe_values = [9.0]
                                    else:
                                        snipe_values = snipe_thresholds
                                    for snipe in snipe_values:
                                        for max_rate in max_order_rate_per_min:
                                            for gross_lim in gross_limits:
                                                candidates.append(
                                                    {
                                                        "strategy.edge_threshold": edge,
                                                        "strategy.min_ev_cents_per_contract": min_ev,
                                                        "strategy.maker_slippage_cents_per_contract": mslip,
                                                        "strategy.mm_half_spread_cents": mm_half,
                                                        "strategy.max_yes_spread_cents_for_mm": max_yes_spread,
                                                        "strategy.max_yes_spread_cents_for_snipe": 25.0,
                                                        "strategy.model_edge_multiplier": model_mult,
                                                        "strategy.snipe_edge_threshold": snipe,
                                                        "strategy.maker_only": maker_only,
                                                        "strategy.base_order_count": 1,
                                                        "strategy.max_order_count": max_order,
                                                        "execution.max_orders_per_minute": max_rate,
                                                        "risk.max_gross_position": gross_lim,
                                                        "risk.max_net_exposure": gross_lim,
                                                    }
                                                )
    return candidates


def _set_nested_attr(cfg: BotConfig, path: str, value: Any) -> None:
    head, tail = path.split(".", 1)
    obj = getattr(cfg, head)
    setattr(obj, tail, value)


def _apply_params(base_cfg: BotConfig, params: dict[str, Any]) -> BotConfig:
    cfg = copy.deepcopy(base_cfg)
    for key, value in params.items():
        _set_nested_attr(cfg, key, value)
    return cfg


def _sum_metric(reports: Iterable[dict[str, Any]], key: str) -> float:
    total = 0.0
    for rep in reports:
        total += float(rep.get(key) or 0.0)
    return total


def _score(test_reports: list[dict[str, Any]]) -> float:
    pnl = _sum_metric(test_reports, "pnl_cents_mark_to_market")
    fills = _sum_metric(test_reports, "fills")
    cost_drag = _sum_metric(test_reports, "cost_drag_cents")
    adverse_total = _sum_metric(test_reports, "adverse_selection_total")
    adverse_count = _sum_metric(test_reports, "adverse_selection_count")
    adverse_rate = (adverse_count / adverse_total) if adverse_total > 0 else 0.0
    active_files = sum(1 for rep in test_reports if float(rep.get("fills") or 0.0) > 0)
    profitable_files = sum(1 for rep in test_reports if float(rep.get("pnl_cents_mark_to_market") or 0.0) > 0)

    # Prefer positive out-of-sample PnL with real activity across multiple files.
    score = pnl - (0.05 * cost_drag)
    score += profitable_files * 5.0
    score += active_files * 2.0
    if fills < 5:
        score -= 100.0
    if active_files < 2:
        score -= 50.0
    score -= adverse_rate * 20.0
    return score


def _evaluate_candidate(
    *,
    cfg: BotConfig,
    split_rows: list[tuple[list[dict[str, Any]], list[dict[str, Any]], str]],
) -> dict[str, Any]:
    train_reports: list[dict[str, Any]] = []
    test_reports: list[dict[str, Any]] = []
    per_file: list[dict[str, Any]] = []

    for train_rows, test_rows, file_name in split_rows:
        train_rep = run_backtest(cfg, train_rows)
        test_rep = run_backtest(cfg, test_rows)
        train_reports.append(train_rep)
        test_reports.append(test_rep)
        per_file.append(
            {
                "file": file_name,
                "train_pnl_cents": train_rep.get("pnl_cents_mark_to_market"),
                "train_fills": train_rep.get("fills"),
                "test_pnl_cents": test_rep.get("pnl_cents_mark_to_market"),
                "test_fills": test_rep.get("fills"),
            }
        )

    agg = {
        "train_pnl_cents": round(_sum_metric(train_reports, "pnl_cents_mark_to_market"), 4),
        "train_fills": int(_sum_metric(train_reports, "fills")),
        "train_cost_drag_cents": round(_sum_metric(train_reports, "cost_drag_cents"), 4),
        "test_pnl_cents": round(_sum_metric(test_reports, "pnl_cents_mark_to_market"), 4),
        "test_fills": int(_sum_metric(test_reports, "fills")),
        "test_cost_drag_cents": round(_sum_metric(test_reports, "cost_drag_cents"), 4),
    }
    agg["score"] = round(_score(test_reports), 4)
    return {
        "aggregate": agg,
        "per_file": per_file,
    }


def _cfg_to_json_dict(cfg: BotConfig) -> dict[str, Any]:
    return asdict(cfg)


def main() -> int:
    args = parse_args()
    replay_files = _replay_files(args.inputs_glob)
    if not replay_files:
        raise RuntimeError(f"no replay files found for glob={args.inputs_glob}")

    base_cfg = BotConfig.from_json(args.base_config)

    split_rows: list[tuple[list[dict[str, Any]], list[dict[str, Any]], str]] = []
    for replay_file in replay_files:
        rows = _usable_rows(load_rows(replay_file))
        if len(rows) < args.min_file_rows:
            continue
        train_rows, test_rows = _split_train_test(rows, train_ratio=args.train_ratio)
        if not train_rows or not test_rows:
            continue
        split_rows.append((train_rows, test_rows, str(replay_file)))

    if not split_rows:
        raise RuntimeError("no replay files with enough rows after filtering")

    results: list[dict[str, Any]] = []

    # Baseline (no param overrides)
    baseline_eval = _evaluate_candidate(cfg=copy.deepcopy(base_cfg), split_rows=split_rows)
    results.append(
        {
            "name": "baseline",
            "params": {},
            **baseline_eval,
        }
    )

    for params in _candidate_grid():
        cfg = _apply_params(base_cfg, params)
        eval_result = _evaluate_candidate(cfg=cfg, split_rows=split_rows)
        results.append(
            {
                "name": "candidate",
                "params": params,
                **eval_result,
            }
        )

    results.sort(key=lambda item: float(item["aggregate"]["score"]), reverse=True)
    best = results[0]
    best_cfg = _apply_params(base_cfg, best["params"]) if best.get("params") else base_cfg

    candidates_with_test_fills = [
        row for row in results if int(row["aggregate"].get("test_fills") or 0) > 0
    ]
    best_with_fills = None
    if candidates_with_test_fills:
        candidates_with_test_fills.sort(
            key=lambda item: (
                float(item["aggregate"].get("test_pnl_cents") or 0.0),
                float(item["aggregate"].get("score") or 0.0),
                int(item["aggregate"].get("test_fills") or 0),
            ),
            reverse=True,
        )
        best_with_fills = candidates_with_test_fills[0]

    out_payload = {
        "base_config": str(args.base_config),
        "inputs_glob": args.inputs_glob,
        "files_used": [name for _, _, name in split_rows],
        "train_ratio": args.train_ratio,
        "candidates_tested": len(results),
        "best": best,
        "best_with_test_fills": best_with_fills,
        "top10": results[:10],
        "top10_with_test_fills": candidates_with_test_fills[:10] if candidates_with_test_fills else [],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2))

    best_cfg_path = Path(args.write_best_config)
    best_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    best_cfg_path.write_text(json.dumps(_cfg_to_json_dict(best_cfg), indent=2))

    print(json.dumps(out_payload["best"], indent=2))
    if out_payload["best_with_test_fills"] is not None:
        print("best_with_test_fills:")
        print(json.dumps(out_payload["best_with_test_fills"], indent=2))
    else:
        print("best_with_test_fills: none")
    print(f"report={out_path.resolve()}")
    print(f"best_config={best_cfg_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
