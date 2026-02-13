from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Optional

from .config import RiskConfig, SimulationConfig, StrategyConfig
from .sim import MonteCarloSummary, run_monte_carlo


@dataclass
class OptimizationResult:
    best_strategy: StrategyConfig
    summary: MonteCarloSummary
    objective_score: float
    met_target: bool

    def to_dict(self) -> dict:
        result = asdict(self)
        result["summary"] = self.summary.to_dict()
        return result


def optimize_strategy(
    base_strategy: StrategyConfig,
    risk_cfg: RiskConfig,
    sim_cfg: SimulationConfig,
    episodes: int,
    target_return_pct: float,
    max_drawdown_pct: float,
    seed: int,
) -> OptimizationResult:
    spreads = [0.006, 0.009, 0.012]
    vol_mults = [1.2, 1.8]
    skews = [0.008, 0.014]
    momentum_alphas = [0.20, 0.45]
    sizes = [8, 12, 16]

    best_cfg: Optional[StrategyConfig] = None
    best_summary: Optional[MonteCarloSummary] = None
    best_score = float("-inf")
    best_met = False

    for idx, (spread, vol_mult, skew, alpha, size) in enumerate(
        product(spreads, vol_mults, skews, momentum_alphas, sizes)
    ):
        cfg = StrategyConfig(
            base_half_spread=spread,
            min_half_spread=max(0.003, spread * 0.5),
            max_half_spread=max(0.020, spread * 3.0),
            volatility_multiplier=vol_mult,
            inventory_skew=skew,
            momentum_alpha=alpha,
            min_edge=max(0.002, spread * 0.8),
            base_order_size=size,
        )

        summary, _ = run_monte_carlo(
            strategy_cfg=cfg,
            risk_cfg=risk_cfg,
            sim_cfg=sim_cfg,
            episodes=episodes,
            seed=seed + (idx * 151),
        )

        met = (
            summary.median_return_pct >= target_return_pct
            and summary.max_drawdown_p90_pct <= max_drawdown_pct
            and summary.p10_return_pct > -0.04
        )

        score = _objective(summary, target_return_pct)
        if met:
            score += 1.0

        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_summary = summary
            best_met = met

    if best_cfg is None or best_summary is None:
        summary, _ = run_monte_carlo(base_strategy, risk_cfg, sim_cfg, episodes=episodes, seed=seed)
        return OptimizationResult(
            best_strategy=base_strategy,
            summary=summary,
            objective_score=_objective(summary, target_return_pct),
            met_target=False,
        )

    return OptimizationResult(
        best_strategy=best_cfg,
        summary=best_summary,
        objective_score=best_score,
        met_target=best_met,
    )


def _objective(summary: MonteCarloSummary, target_return_pct: float) -> float:
    target_bonus = summary.median_return_pct / max(0.01, target_return_pct)
    downside_penalty = max(0.0, -summary.p10_return_pct) * 1.4
    drawdown_penalty = summary.max_drawdown_p90_pct * 1.1
    kill_penalty = summary.kill_rate * 0.5
    return target_bonus - downside_penalty - drawdown_penalty - kill_penalty
