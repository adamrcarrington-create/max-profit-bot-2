from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .config import BotConfig, StrategyConfig
from .optimizer import OptimizationResult, optimize_strategy
from .sim import EpisodeMetrics, MonteCarloSummary, run_monte_carlo


@dataclass
class MarketPaperResult:
    market: str
    strategy: StrategyConfig
    summary: MonteCarloSummary
    metrics: list[EpisodeMetrics]
    optimization: OptimizationResult | None
    go_no_go: str


def run_paper(
    cfg: BotConfig,
    episodes: int,
    seed: int,
    optimize: bool,
) -> tuple[StrategyConfig, MonteCarloSummary, list[EpisodeMetrics], OptimizationResult | None]:
    optimization: OptimizationResult | None = None
    strategy = cfg.strategy

    if optimize:
        optimization = optimize_strategy(
            base_strategy=cfg.strategy,
            risk_cfg=cfg.risk,
            sim_cfg=cfg.sim,
            episodes=max(10, episodes // 3),
            target_return_pct=cfg.target_return_pct,
            max_drawdown_pct=cfg.max_drawdown_pct,
            seed=seed,
        )
        strategy = optimization.best_strategy

    summary, metrics = run_monte_carlo(
        strategy_cfg=strategy,
        risk_cfg=cfg.risk,
        sim_cfg=cfg.sim,
        episodes=episodes,
        seed=seed + 100_000,
    )
    return strategy, summary, metrics, optimization


def run_paper_multi(
    cfg: BotConfig,
    markets: list[str],
    episodes: int,
    seed: int,
    optimize: bool,
) -> tuple[list[MarketPaperResult], dict[str, float]]:
    if not markets:
        markets = cfg.resolved_markets()

    safe_cfg = copy.deepcopy(cfg)
    per_market_pos = max(1, safe_cfg.risk.max_position // max(1, len(markets)))
    per_market_order = max(1, safe_cfg.risk.max_order_size // max(1, len(markets)))

    results: list[MarketPaperResult] = []
    for idx, market in enumerate(markets):
        market_cfg = copy.deepcopy(safe_cfg)
        market_cfg.market_ticker = market
        market_cfg.risk.max_position = per_market_pos
        market_cfg.risk.max_order_size = per_market_order

        strategy, summary, metrics, optimization = run_paper(
            cfg=market_cfg,
            episodes=episodes,
            seed=seed + (idx * 20_003),
            optimize=optimize,
        )
        go_no_go = verdict(
            summary=summary,
            target_return_pct=market_cfg.target_return_pct,
            max_drawdown_pct=market_cfg.max_drawdown_pct,
        )
        results.append(
            MarketPaperResult(
                market=market,
                strategy=strategy,
                summary=summary,
                metrics=metrics,
                optimization=optimization,
                go_no_go=go_no_go,
            )
        )

    combined = combine_market_summaries(results)
    combined["per_market_max_position"] = float(per_market_pos)
    combined["per_market_max_order_size"] = float(per_market_order)
    return results, combined


def combine_market_summaries(results: list[MarketPaperResult]) -> dict[str, float]:
    if not results:
        return {
            "market_count": 0.0,
            "median_return_pct_avg": 0.0,
            "p10_return_pct_worst": 0.0,
            "p90_return_pct_best": 0.0,
            "max_drawdown_p90_pct_worst": 0.0,
            "kill_rate_worst": 0.0,
            "avg_pnl_total": 0.0,
            "avg_trades_total": 0.0,
            "avg_win_rate_mean": 0.0,
            "pass_ratio": 0.0,
        }

    market_count = float(len(results))
    medians = [r.summary.median_return_pct for r in results]
    p10s = [r.summary.p10_return_pct for r in results]
    p90s = [r.summary.p90_return_pct for r in results]
    drawdowns = [r.summary.max_drawdown_p90_pct for r in results]
    kills = [r.summary.kill_rate for r in results]
    pnls = [r.summary.avg_pnl for r in results]
    trades = [r.summary.avg_trades for r in results]
    win_rates = [r.summary.avg_win_rate for r in results]
    passes = sum(1 for r in results if r.go_no_go == "pass")

    return {
        "market_count": market_count,
        "median_return_pct_avg": sum(medians) / market_count,
        "p10_return_pct_worst": min(p10s),
        "p90_return_pct_best": max(p90s),
        "max_drawdown_p90_pct_worst": max(drawdowns),
        "kill_rate_worst": max(kills),
        "avg_pnl_total": sum(pnls),
        "avg_trades_total": sum(trades),
        "avg_win_rate_mean": sum(win_rates) / market_count,
        "pass_ratio": passes / market_count,
    }


def write_report(
    file_path: str | Path,
    cfg: BotConfig,
    strategy: StrategyConfig,
    summary: MonteCarloSummary,
    metrics: list[EpisodeMetrics],
    optimization: OptimizationResult | None,
) -> None:
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg.to_dict(),
        "active_strategy": asdict(strategy),
        "summary": summary.to_dict(),
        "episodes": [asdict(m) for m in metrics],
        "optimization": optimization.to_dict() if optimization else None,
    }
    p.write_text(json.dumps(payload, indent=2))


def write_multi_report(
    file_path: str | Path,
    cfg: BotConfig,
    market_results: list[MarketPaperResult],
    combined: dict[str, float],
) -> None:
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg.to_dict(),
        "combined_summary": combined,
        "markets": [
            {
                "market": r.market,
                "go_no_go": r.go_no_go,
                "active_strategy": asdict(r.strategy),
                "summary": r.summary.to_dict(),
                "episodes": [asdict(m) for m in r.metrics],
                "optimization": r.optimization.to_dict() if r.optimization else None,
            }
            for r in market_results
        ],
    }
    p.write_text(json.dumps(payload, indent=2))


def verdict(summary: MonteCarloSummary, target_return_pct: float, max_drawdown_pct: float) -> str:
    if (
        summary.median_return_pct >= target_return_pct
        and summary.max_drawdown_p90_pct <= max_drawdown_pct
        and summary.p10_return_pct > 0.0
        and summary.kill_rate <= 0.10
    ):
        return "pass"
    return "fail"

