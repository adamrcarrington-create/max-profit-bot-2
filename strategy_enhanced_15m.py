"""
Enhanced Market-Making Strategy for Kalshi 15-Minute Crypto Markets
Optimized for BTC, ETH, and SOL with time-decay, volatility regimes, and queue awareness.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
from collections import deque
from .config import StrategyConfig
from .models import MarketState, Quote

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

@dataclass
class QueueMetrics:
    """Tracks implicit queue depth via post-only rejects and fill speed"""
    post_only_rejects: int = 0
    total_orders: int = 0
    avg_fill_time_ms: float = 0.0
    reject_ratio: float = 0.0
    
    def update(self, rejected: bool, fill_time_ms: float):
        self.total_orders += 1
        if rejected:
            self.post_only_rejects += 1
        self.avg_fill_time_ms = (self.avg_fill_time_ms * 0.7) + (fill_time_ms * 0.3)
        self.reject_ratio = self.post_only_rejects / max(1, self.total_orders)

class VolatilityRegime:
    """Detects market volatility regimes and adapts strategy accordingly"""
    CALM = "calm"          # vol < 0.008
    TRENDING = "trending"  # 0.008 <= vol < 0.020
    SPIKY = "spiky"        # vol >= 0.020
    
    @staticmethod
    def classify(volatility: float) -> str:
        if volatility < 0.008:
            return VolatilityRegime.CALM
        elif volatility < 0.020:
            return VolatilityRegime.TRENDING
        else:
            return VolatilityRegime.SPIKY

class TimedecayExitManager:
    """Manages aggressive exits as market approaches 15-minute close"""
    
    @staticmethod
    def get_exit_aggression(seconds_to_close: float) -> float:
        """
        Returns multiplier for exit aggression (1.0 = normal, >1.0 = aggressive)
        Aggressive exits kick in at 3 minutes remaining
        """
        if seconds_to_close > 300:  # > 5 min
            return 1.0
        elif seconds_to_close > 180:  # 3-5 min
            return 1.2
        elif seconds_to_close > 120:  # 2-3 min
            return 1.5
        elif seconds_to_close > 60:   # 1-2 min
            return 2.0
        else:  # < 1 min - EMERGENCY MODE
            return 3.0
    
    @staticmethod
    def should_hard_flatten(seconds_to_close: float) -> bool:
        """Hard flatten everything at 90 seconds before close"""
        return seconds_to_close <= 90

class EnhancedAdaptiveMarketMaker:
    """
    Advanced MM strategy combining:
    - Time-decay aggressive exits
    - Volatility regime detection
    - Market-specific parameters (BTC/ETH/SOL)
    - Implicit queue awareness
    - Sniper mode for opportunistic entries
    """
    
    def __init__(self, cfg: StrategyConfig, market_ticker: str = "KXBTC15M"):
        self.cfg = cfg
        self.market_ticker = market_ticker
        self.queue_metrics = QueueMetrics()
        self.recent_mids = deque(maxlen=30)
        self.recent_fills = deque(maxlen=20)
        
        # Market-specific parameters
        self.market_params = self._get_market_params(market_ticker)
    
    def _get_market_params(self, ticker: str) -> dict:
        """Return market-specific parameters for BTC, ETH, or SOL"""
        params = {
            "KXBTC15M": {
                "aggression": 0.7,      # Most conservative
                "spread_multiplier": 0.9,
                "size_multiplier": 1.2,
                "queue_sensitivity": 0.6,
                "volatility_threshold": 0.012,
            },
            "KXETH15M": {
                "aggression": 0.85,     # Balanced
                "spread_multiplier": 1.0,
                "size_multiplier": 1.0,
                "queue_sensitivity": 0.8,
                "volatility_threshold": 0.015,
            },
            "KXSOL15M": {
                "aggression": 1.0,      # Most aggressive
                "spread_multiplier": 1.15,
                "size_multiplier": 0.8,
                "queue_sensitivity": 1.0,
                "volatility_threshold": 0.018,
            },
        }
        return params.get(ticker.upper(), params["KXETH15M"])
    
    def make_quote(
        self,
        market: MarketState,
        position_qty: int,
        max_position: int,
        seconds_to_close: Optional[float] = None,
    ) -> Quote | None:
        """
        Generate MM quotes with time-decay, volatility awareness, and queue sensitivity
        """
        if max_position <= 0:
            return None
        
        inventory_ratio = _clamp(position_qty / max_position, -1.0, 1.0)
        
        # Base fair value with momentum
        fair = self._fair_value(market)
        
        # Get volatility regime
        vol_regime = VolatilityRegime.classify(market.volatility)
        
        # Adjust spread based on volatility regime and time decay
        half_spread = self._adaptive_half_spread(
            market=market,
            inventory_ratio=inventory_ratio,
            vol_regime=vol_regime,
            seconds_to_close=seconds_to_close,
        )
        
        # Apply inventory skew
        skew = inventory_ratio * self.cfg.inventory_skew
        fair -= skew
        
        # Apply queue awareness adjustment
        queue_spread_boost = self._queue_aware_spread_boost()
        half_spread += queue_spread_boost
        
        # Generate bid/ask
        bid = _clamp(fair - half_spread, 0.01, 0.98)
        ask = _clamp(fair + half_spread, 0.02, 0.99)
        
        # Enforce minimum edge
        if ask - bid < self.cfg.min_edge:
            mid = (bid + ask) * 0.5
            bid = _clamp(mid - (self.cfg.min_edge * 0.5), 0.01, 0.98)
            ask = _clamp(mid + (self.cfg.min_edge * 0.5), 0.02, 0.99)
        
        if ask <= bid:
            return None
        
        # Size calculation with market-specific adjustment
        size = self._calculate_size(
            market=market,
            inventory_ratio=inventory_ratio,
            vol_regime=vol_regime,
            seconds_to_close=seconds_to_close,
        )
        
        buy_size = size
        sell_size = size
        
        # Position limits
        if inventory_ratio > 0.90:
            buy_size = max(0, int(size * 0.3))
        if inventory_ratio < -0.90:
            sell_size = max(0, int(size * 0.3))
        
        if buy_size == 0 and sell_size == 0:
            return None
        
        return Quote(bid=bid, ask=ask, buy_size=buy_size, sell_size=sell_size)
    
    def _fair_value(self, market: MarketState) -> float:
        """Calculate fair value with momentum adjustment"""
        params = self.market_params
        momentum_adjustment = self.cfg.momentum_alpha * market.momentum * params["aggression"]
        fair = market.mid + momentum_adjustment
        return _clamp(fair, 0.01, 0.99)
    
    def _adaptive_half_spread(
        self,
        market: MarketState,
        inventory_ratio: float,
        vol_regime: str,
        seconds_to_close: Optional[float] = None,
    ) -> float:
        """
        Calculate half-spread with multiple factors:
        - Base spread
        - Volatility scaling
        - Inventory skew
        - Volatility regime adjustment
        - Time decay aggression
        """
        params = self.market_params
        
        # Base spread
        spread = self.cfg.base_half_spread
        
        # Volatility component
        spread += market.volatility * self.cfg.volatility_multiplier
        
        # Inventory component
        spread += abs(inventory_ratio) * self.cfg.inventory_skew * 0.8
        
        # Volatility regime boost
        if vol_regime == VolatilityRegime.CALM:
            spread *= 0.85  # Tighter in calm markets
        elif vol_regime == VolatilityRegime.TRENDING:
            spread *= 1.0   # Normal
        elif vol_regime == VolatilityRegime.SPIKY:
            spread *= 1.4   # Much wider in spiky markets
        
        # Time decay aggression - tighten spreads as close approaches
        if seconds_to_close is not None:
            exit_agg = TimedecayExitManager.get_exit_aggression(seconds_to_close)
            spread *= (1.0 / exit_agg) * 0.95  # Tighten to attract fills
        
        # Market-specific multiplier
        spread *= params["spread_multiplier"]
        
        return _clamp(spread, self.cfg.min_half_spread, self.cfg.max_half_spread)
    
    def _queue_aware_spread_boost(self) -> float:
        """
        Boost spread if we're getting post-only rejects (sign of deep queue)
        This is IMPLICIT queue awareness without API access
        """
        params = self.market_params
        
        if self.queue_metrics.reject_ratio > 0.3:
            # Getting rejected often = deep queue = widen spreads
            return 0.004 * params["queue_sensitivity"]
        elif self.queue_metrics.avg_fill_time_ms > 500:
            # Fills are slow = deep queue = widen spreads
            return 0.002 * params["queue_sensitivity"]
        else:
            # Fills are fast = shallow queue = keep tight
            return 0.0
    
    def _calculate_size(
        self,
        market: MarketState,
        inventory_ratio: float,
        vol_regime: str,
        seconds_to_close: Optional[float] = None,
    ) -> int:
        """
        Calculate order size with:
        - Inventory scaling
        - Volatility scaling
        - Volatility regime adjustment
        - Time decay adjustment
        - Market-specific sizing
        """
        params = self.market_params
        
        base_size = max(1, self.cfg.base_order_size)
        
        # Inventory scaling
        inv_scale = max(0.1, 1.0 - (abs(inventory_ratio) * 0.9))
        
        # Volatility scaling
        vol_scale = max(0.35, 1.0 - (market.volatility * 8.0))
        
        # Volatility regime adjustment
        if vol_regime == VolatilityRegime.CALM:
            vol_scale *= 1.2  # Larger in calm markets
        elif vol_regime == VolatilityRegime.SPIKY:
            vol_scale *= 0.6  # Much smaller in spiky markets
        
        # Time decay - reduce size as close approaches
        if seconds_to_close is not None:
            if seconds_to_close < 120:
                vol_scale *= 0.5  # Half size in last 2 minutes
            elif seconds_to_close < 300:
                vol_scale *= 0.75
        
        # Market-specific multiplier
        size = max(1, int(round(base_size * inv_scale * vol_scale * params["size_multiplier"])))
        
        return size
    
    def record_fill(self, rejected: bool, fill_time_ms: float):
        """Update queue metrics based on order outcome"""
        self.queue_metrics.update(rejected, fill_time_ms)