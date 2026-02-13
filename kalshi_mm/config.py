from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict


@dataclass
class StrategyConfig:
    base_half_spread: float = 0.010
    min_half_spread: float = 0.004
    max_half_spread: float = 0.035
    volatility_multiplier: float = 1.80
    inventory_skew: float = 0.010
    momentum_alpha: float = 0.30
    min_edge: float = 0.006
    base_order_size: int = 5


@dataclass
class RiskConfig:
    max_position: int = 150
    max_order_size: int = 30
    live_size_scale: float = 0.45
    max_daily_loss: float = 18.0
    max_drawdown: float = 14.0
    volatility_hard_stop: float = 0.055
    stale_data_seconds: float = 2.5
    session_stop_loss_cents: int = 1200
    session_take_profit_cents: int = 250
    min_requote_seconds: float = 4.0
    max_orders_per_market: int = 60
    min_market_spread: float = 0.035
    min_quote_edge: float = 0.006
    preclose_guard_seconds: float = 180.0
    entry_time_buffer_seconds: float = 0.0
    no_trade_window_cycle_limit: int = 4
    auto_roll_on_no_trade_window: bool = True
    momentum_soft_guard: float = 0.0035
    momentum_hard_guard: float = 0.0085
    adverse_side_size_cut: float = 0.80
    toxicity_edge_boost: float = 0.004
    take_profit_tier1: float = 0.015
    take_profit_tier2: float = 0.030
    stop_loss: float = 0.020
    max_consecutive_losses: int = 10


@dataclass
class SimulationConfig:
    steps: int = 5_000
    starting_mid: float = 0.50
    starting_spread: float = 0.018
    drift: float = 0.0
    volatility: float = 0.008
    mean_reversion: float = 0.10
    maker_fee_per_contract: float = 0.004
    slippage_mean: float = 0.0008
    adverse_selection: float = 0.0025
    seed: int = 7


@dataclass
class BotConfig:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    target_return_pct: float = 0.10
    max_drawdown_pct: float = 0.18
    market_tickers: list[str] = field(
        default_factory=lambda: ["KXBTC15M", "KXETH15M", "KXSOL15M"]
    )
    market_ticker: str = "KXBTC15M"

    @classmethod
    def from_json(cls, file_path: str | Path) -> "BotConfig":
        raw = json.loads(Path(file_path).read_text())
        return cls(
            strategy=StrategyConfig(**raw.get("strategy", {})),
            risk=RiskConfig(**raw.get("risk", {})),
            sim=SimulationConfig(**raw.get("sim", {})),
            target_return_pct=raw.get("target_return_pct", 0.10),
            max_drawdown_pct=raw.get("max_drawdown_pct", 0.18),
            market_tickers=raw.get(
                "market_tickers",
                [raw.get("market_ticker", "KXBTC15M")],
            ),
            market_ticker=raw.get("market_ticker", "KXBTC15M"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def resolved_markets(self) -> list[str]:
        markets = [m.strip().upper() for m in self.market_tickers if m and m.strip()]
        if markets:
            return markets
        if self.market_ticker.strip():
            return [self.market_ticker.strip().upper()]
        return ["KXBTC15M"]
