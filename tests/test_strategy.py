import unittest

from kalshi_mm.config import StrategyConfig
from kalshi_mm.models import MarketState
from kalshi_mm.strategy import AdaptiveMarketMaker


class StrategyTests(unittest.TestCase):
    def test_quote_is_valid_and_non_crossing(self) -> None:
        cfg = StrategyConfig()
        strat = AdaptiveMarketMaker(cfg)
        state = MarketState(
            timestamp=0.0,
            mid=0.50,
            best_bid=0.49,
            best_ask=0.51,
            volatility=0.008,
            momentum=0.001,
        )
        quote = strat.make_quote(state, position_qty=0, max_position=120)
        self.assertIsNotNone(quote)
        self.assertLess(quote.bid, quote.ask)
        self.assertGreaterEqual(quote.bid, 0.01)
        self.assertLessEqual(quote.ask, 0.99)

    def test_inventory_skew_reduces_bid_when_long(self) -> None:
        cfg = StrategyConfig(inventory_skew=0.02)
        strat = AdaptiveMarketMaker(cfg)
        state = MarketState(
            timestamp=0.0,
            mid=0.50,
            best_bid=0.49,
            best_ask=0.51,
            volatility=0.008,
            momentum=0.0,
        )
        flat = strat.make_quote(state, position_qty=0, max_position=100)
        long_inv = strat.make_quote(state, position_qty=80, max_position=100)
        self.assertIsNotNone(flat)
        self.assertIsNotNone(long_inv)
        self.assertLess(long_inv.bid, flat.bid)


if __name__ == "__main__":
    unittest.main()
