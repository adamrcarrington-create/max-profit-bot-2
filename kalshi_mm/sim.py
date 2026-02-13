from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import random
import statistics
from typing import Iterable, List

from .config import RiskConfig, SimulationConfig, StrategyConfig
from .models import Fill, MarketState, PositionState, Quote
from .risk import RiskManager
from .strategy import AdaptiveMarketMaker


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class EpisodeMetrics:
    final_pnl: float
    return_pct: float
    max_drawdown_pct: float
    trade_count: int
    win_rate: float
    killed: bool
    kill_reason: str


@dataclass
class MonteCarloSummary:
    episodes: int
    avg_pnl: float
    median_return_pct: float
    p10_return_pct: float
    p90_return_pct: float
    max_drawdown_p90_pct: float
    avg_trades: float
    kill_rate: float
    avg_win_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


class SyntheticMarket:
    def __init__(self, cfg: SimulationConfig, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self.mid = cfg.starting_mid
        self.returns: deque[float] = deque(maxlen=40)
        self.last_mid = cfg.starting_mid

    def next(self, step: int) -> MarketState:
        noise = self.rng.gauss(0.0, self.cfg.volatility)
        reversion = self.cfg.mean_reversion * (0.5 - self.mid)
        jump = 0.0
        if self.rng.random() < 0.002:
            jump = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.02, 0.05)
        move = self.cfg.drift + reversion + noise + jump
        self.mid = _clamp(self.mid + move, 0.02, 0.98)

        ret = self.mid - self.last_mid
        self.last_mid = self.mid
        self.returns.append(ret)

        vol = statistics.pstdev(self.returns) if len(self.returns) > 4 else self.cfg.volatility
        mom = statistics.mean(list(self.returns)[-5:]) if self.returns else 0.0

        dynamic_spread = _clamp(
            self.cfg.starting_spread + (abs(ret) * 1.25) + self.rng.uniform(-0.002, 0.002),
            0.006,
            0.080,
        )
        bid = _clamp(self.mid - dynamic_spread * 0.5, 0.01, 0.98)
        ask = _clamp(self.mid + dynamic_spread * 0.5, 0.02, 0.99)
        if ask <= bid:
            ask = min(0.99, bid + 0.002)

        timestamp = step * 0.25
        return MarketState(
            timestamp=timestamp,
            mid=self.mid,
            best_bid=bid,
            best_ask=ask,
            volatility=vol,
            momentum=mom,
        )


def _simulate_fills(rng: random.Random, state: MarketState, quote: Quote) -> List[Fill]:
    fills: list[Fill] = []
    touch_spread = max(0.002, state.best_ask - state.best_bid)

    if quote.buy_size > 0:
        bid_gap = max(0.0, state.best_bid - quote.bid)
        competitiveness = max(0.0, 1.0 - (bid_gap / (touch_spread * 2.5)))
        prob = min(0.60, 0.015 + (0.18 * competitiveness) + (state.volatility * 1.2))
        if quote.bid >= state.best_ask:
            prob = 1.0
        if rng.random() < prob:
            size = max(1, int(round(quote.buy_size * (0.15 + rng.random() * 0.40))))
            fills.append(Fill(side="buy", price=quote.bid, size=size))

    if quote.sell_size > 0:
        ask_gap = max(0.0, quote.ask - state.best_ask)
        competitiveness = max(0.0, 1.0 - (ask_gap / (touch_spread * 2.5)))
        prob = min(0.60, 0.015 + (0.18 * competitiveness) + (state.volatility * 1.2))
        if quote.ask <= state.best_bid:
            prob = 1.0
        if rng.random() < prob:
            size = max(1, int(round(quote.sell_size * (0.15 + rng.random() * 0.40))))
            fills.append(Fill(side="sell", price=quote.ask, size=size))

    return fills


def run_episode(
    strategy_cfg: StrategyConfig,
    risk_cfg: RiskConfig,
    sim_cfg: SimulationConfig,
    seed: int,
) -> EpisodeMetrics:
    rng = random.Random(seed)
    market = SyntheticMarket(sim_cfg, rng)
    strategy = AdaptiveMarketMaker(strategy_cfg)
    risk = RiskManager(risk_cfg)
    position = PositionState()

    trades = 0
    wins = 0
    losses = 0
    last_mark = sim_cfg.starting_mid

    for step in range(sim_cfg.steps):
        state = market.next(step)
        last_mark = state.mid
        equity_before = position.equity(state.mid)
        risk.observe(state, equity_before)

        if position.qty != 0:
            frac, _ = risk.tiered_exit_fraction(position, state.mid)
            if frac > 0:
                close_size = max(1, int(round(abs(position.qty) * frac)))
                side = "sell" if position.qty > 0 else "buy"
                close_size = risk.capped_size(side, close_size, position.qty)
                if close_size > 0:
                    old_realized = position.realized_pnl
                    position.apply_fill(side=side, price=state.mid, size=close_size)
                    trades += 1
                    delta_realized = position.realized_pnl - old_realized
                    if delta_realized > 0:
                        wins += 1
                    elif delta_realized < 0:
                        losses += 1

        status = risk.status()
        if status.killed:
            if position.qty != 0:
                flatten_side = "sell" if position.qty > 0 else "buy"
                flatten_size = min(abs(position.qty), risk_cfg.max_order_size)
                old_realized = position.realized_pnl
                position.apply_fill(side=flatten_side, price=state.mid, size=flatten_size)
                trades += 1
                delta_realized = position.realized_pnl - old_realized
                if delta_realized > 0:
                    wins += 1
                elif delta_realized < 0:
                    losses += 1
            continue

        quote = strategy.make_quote(state, position.qty, risk_cfg.max_position)
        if quote is None:
            continue

        quote.buy_size = risk.capped_size("buy", quote.buy_size, position.qty)
        quote.sell_size = risk.capped_size("sell", quote.sell_size, position.qty)
        if quote.buy_size == 0 and quote.sell_size == 0:
            continue

        fills = _simulate_fills(rng, state, quote)
        for fill in fills:
            old_realized = position.realized_pnl
            exec_price = _execution_price(rng, fill.side, fill.price, sim_cfg.slippage_mean)
            position.apply_fill(fill.side, exec_price, fill.size)
            position.cash -= sim_cfg.maker_fee_per_contract * fill.size
            _apply_adverse_selection(market, fill.side, sim_cfg.adverse_selection, rng)
            trades += 1
            delta_realized = position.realized_pnl - old_realized
            if delta_realized > 0:
                wins += 1
            elif delta_realized < 0:
                losses += 1

    if position.qty != 0:
        final_side = "sell" if position.qty > 0 else "buy"
        position.apply_fill(final_side, last_mark, abs(position.qty))

    final_pnl = position.equity(last_mark)
    capital_base = float(max(1, risk_cfg.max_position))
    return_pct = final_pnl / capital_base
    max_drawdown_pct = risk.max_drawdown_seen / capital_base
    decision_count = wins + losses
    win_rate = wins / decision_count if decision_count > 0 else 0.0

    return EpisodeMetrics(
        final_pnl=final_pnl,
        return_pct=return_pct,
        max_drawdown_pct=max_drawdown_pct,
        trade_count=trades,
        win_rate=win_rate,
        killed=risk.killed,
        kill_reason=risk.kill_reason,
    )


def _execution_price(rng: random.Random, side: str, quote_price: float, slippage_mean: float) -> float:
    slip = abs(rng.gauss(0.0, slippage_mean))
    if side == "buy":
        return _clamp(quote_price + slip, 0.01, 0.99)
    return _clamp(quote_price - slip, 0.01, 0.99)


def _apply_adverse_selection(
    market: SyntheticMarket,
    fill_side: str,
    adverse_selection: float,
    rng: random.Random,
) -> None:
    if adverse_selection <= 0:
        return
    impact = adverse_selection * rng.uniform(0.25, 1.0)
    if fill_side == "buy":
        market.mid = _clamp(market.mid - impact, 0.02, 0.98)
    else:
        market.mid = _clamp(market.mid + impact, 0.02, 0.98)
    market.last_mid = market.mid


def run_monte_carlo(
    strategy_cfg: StrategyConfig,
    risk_cfg: RiskConfig,
    sim_cfg: SimulationConfig,
    episodes: int,
    seed: int = 7,
) -> tuple[MonteCarloSummary, list[EpisodeMetrics]]:
    metrics: list[EpisodeMetrics] = []
    for idx in range(episodes):
        episode_seed = seed + (idx * 97)
        metrics.append(run_episode(strategy_cfg, risk_cfg, sim_cfg, episode_seed))

    returns = sorted(m.return_pct for m in metrics)
    drawdowns = sorted(m.max_drawdown_pct for m in metrics)
    pnls = [m.final_pnl for m in metrics]
    trades = [m.trade_count for m in metrics]
    win_rates = [m.win_rate for m in metrics]
    kills = sum(1 for m in metrics if m.killed)

    summary = MonteCarloSummary(
        episodes=episodes,
        avg_pnl=statistics.mean(pnls),
        median_return_pct=_percentile(returns, 50),
        p10_return_pct=_percentile(returns, 10),
        p90_return_pct=_percentile(returns, 90),
        max_drawdown_p90_pct=_percentile(drawdowns, 90),
        avg_trades=statistics.mean(trades),
        kill_rate=kills / max(1, episodes),
        avg_win_rate=statistics.mean(win_rates),
    )
    return summary, metrics


def _percentile(values: Iterable[float], p: int) -> float:
    data = list(values)
    if not data:
        return 0.0
    idx = int(round((len(data) - 1) * (p / 100.0)))
    return data[idx]
