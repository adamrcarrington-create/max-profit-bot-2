import unittest

from kalshi_mm.config import BotConfig


class ConfigTests(unittest.TestCase):
    def test_resolved_markets_prefers_market_tickers(self) -> None:
        cfg = BotConfig(market_tickers=["kxbtc15m", "kXeth15m"], market_ticker="KXSOL15M")
        self.assertEqual(cfg.resolved_markets(), ["KXBTC15M", "KXETH15M"])

    def test_resolved_markets_falls_back_to_market_ticker(self) -> None:
        cfg = BotConfig(market_tickers=[], market_ticker="kxsol15m")
        self.assertEqual(cfg.resolved_markets(), ["KXSOL15M"])


if __name__ == "__main__":
    unittest.main()

