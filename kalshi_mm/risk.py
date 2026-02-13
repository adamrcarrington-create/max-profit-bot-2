from __future__ import annotations

from dataclasses import dataclass

from .config import RiskConfig
from .models import MarketState, PositionState


@dataclass
class RiskStatus:
    can_trade: bool
    killed: bool
    reason: str = ""


@dataclass
class RiskManager:
    cfg: RiskConfig
    starting_equity: float = 0.0
    equity_peak: float = 0.0
    last_equity: float = 0.0
    max_drawdown_seen: float = 0.0
    consecutive_losses: int = 0
    killed: bool = False
    kill_reason: str = ""
    _initialized: bool = False
    _last_timestamp: float | None = None

    def observe(self, market: MarketState, equity: float) -> None:
        if self._last_timestamp is not None:
            dt = market.timestamp - self._last_timestamp
            if dt > self.cfg.stale_data_seconds:
                self._kill(f"stale_data_dt={dt:.3f}s")
        self._last_timestamp = market.timestamp

        if not self._initialized:
            self.starting_equity = equity
            self.equity_peak = equity
            self.last_equity = equity
            self._initialized = True
            return

        if equity > self.equity_peak:
            self.equity_peak = equity

        drawdown = self.equity_peak - equity
        if drawdown > self.max_drawdown_seen:
            self.max_drawdown_seen = drawdown

        if (equity - self.starting_equity) <= -self.cfg.max_daily_loss:
            self._kill("max_daily_loss")
        if drawdown >= self.cfg.max_drawdown:
            self._kill("max_drawdown")
        if market.volatility >= self.cfg.volatility_hard_stop:
            self._kill("volatility_hard_stop")

        delta = equity - self.last_equity
        if delta < 0:
            self.consecutive_losses += 1
        elif delta > 0:
            self.consecutive_losses = 0
        if self.consecutive_losses >= self.cfg.max_consecutive_losses:
            self._kill("consecutive_losses")

        self.last_equity = equity

    def status(self) -> RiskStatus:
        if self.killed:
            return RiskStatus(can_trade=False, killed=True, reason=self.kill_reason)
        return RiskStatus(can_trade=True, killed=False, reason="")

    def capped_size(self, side: str, requested_size: int, position_qty: int) -> int:
        if requested_size <= 0:
            return 0
        base = min(requested_size, self.cfg.max_order_size)
        if side == "buy":
            available = self.cfg.max_position - position_qty
        elif side == "sell":
            available = self.cfg.max_position + position_qty
        else:
            raise ValueError(f"Unsupported side: {side}")
        return max(0, min(base, available))

    def tiered_exit_fraction(self, position: PositionState, mark_price: float) -> tuple[float, str]:
        if position.qty == 0 or position.avg_price <= 0:
            return 0.0, ""

        gross_notional = abs(position.qty) * position.avg_price
        if gross_notional <= 0:
            return 0.0, ""

        pnl_pct = position.unrealized_pnl(mark_price) / gross_notional

        if pnl_pct <= -self.cfg.stop_loss:
            return 1.0, "stop_loss"
        if pnl_pct >= self.cfg.take_profit_tier2:
            return 0.70, "take_profit_tier2"
        if pnl_pct >= self.cfg.take_profit_tier1:
            return 0.35, "take_profit_tier1"
        return 0.0, ""

    def _kill(self, reason: str) -> None:
        if self.killed:
            return
        self.killed = True
        self.kill_reason = reason
