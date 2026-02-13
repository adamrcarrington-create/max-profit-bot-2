from __future__ import annotations

from dataclasses import dataclass

from .config import StrategyConfig
from .models import MarketState, Quote


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class AdaptiveMarketMaker:
    cfg: StrategyConfig

    def make_quote(self, market: MarketState, position_qty: int, max_position: int) -> Quote | None:
        if max_position <= 0:
            return None

        inventory_ratio = _clamp(position_qty / max_position, -1.0, 1.0)
        fair = self._fair_value(market)
        half_spread = self._half_spread(market, inventory_ratio)
        skew = inventory_ratio * self.cfg.inventory_skew
        fair -= skew

        bid = _clamp(fair - half_spread, 0.01, 0.98)
        ask = _clamp(fair + half_spread, 0.02, 0.99)
        if ask - bid < self.cfg.min_edge:
            mid = (bid + ask) * 0.5
            bid = _clamp(mid - (self.cfg.min_edge * 0.5), 0.01, 0.98)
            ask = _clamp(mid + (self.cfg.min_edge * 0.5), 0.02, 0.99)
        if ask <= bid:
            return None

        base_size = max(1, self.cfg.base_order_size)
        inv_scale = max(0.1, 1.0 - (abs(inventory_ratio) * 0.9))
        vol_scale = max(0.35, 1.0 - (market.volatility * 8.0))
        size = max(1, int(round(base_size * inv_scale * vol_scale)))

        buy_size = size
        sell_size = size
        if inventory_ratio > 0.95:
            buy_size = 0
        if inventory_ratio < -0.95:
            sell_size = 0

        if buy_size == 0 and sell_size == 0:
            return None
        return Quote(bid=bid, ask=ask, buy_size=buy_size, sell_size=sell_size)

    def _fair_value(self, market: MarketState) -> float:
        fair = market.mid + (self.cfg.momentum_alpha * market.momentum)
        return _clamp(fair, 0.01, 0.99)

    def _half_spread(self, market: MarketState, inventory_ratio: float) -> float:
        spread = (
            self.cfg.base_half_spread
            + (market.volatility * self.cfg.volatility_multiplier)
            + (abs(inventory_ratio) * self.cfg.inventory_skew * 0.8)
        )
        return _clamp(spread, self.cfg.min_half_spread, self.cfg.max_half_spread)

