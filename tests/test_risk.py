import unittest

from kalshi_mm.config import RiskConfig
from kalshi_mm.models import PositionState
from kalshi_mm.risk import RiskManager


class RiskTests(unittest.TestCase):
    def test_capped_size_respects_position_limit(self) -> None:
        rm = RiskManager(RiskConfig(max_position=100, max_order_size=30))
        self.assertEqual(rm.capped_size("buy", 50, position_qty=90), 10)
        self.assertEqual(rm.capped_size("sell", 50, position_qty=-90), 10)

    def test_tiered_exit_stop_loss_and_profit(self) -> None:
        rm = RiskManager(
            RiskConfig(
                take_profit_tier1=0.01,
                take_profit_tier2=0.02,
                stop_loss=0.015,
            )
        )
        pos = PositionState(qty=20, avg_price=0.50)

        frac, reason = rm.tiered_exit_fraction(pos, mark_price=0.485)
        self.assertEqual(frac, 1.0)
        self.assertEqual(reason, "stop_loss")

        frac, reason = rm.tiered_exit_fraction(pos, mark_price=0.515)
        self.assertEqual(frac, 0.7)
        self.assertEqual(reason, "take_profit_tier2")


if __name__ == "__main__":
    unittest.main()
