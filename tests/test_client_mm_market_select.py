import unittest
from typing import Optional

from kalshi_hft.client import KalshiAuthConfig, KalshiClient, KalshiRequestError


class _StubClient(KalshiClient):
    def __init__(self, markets: list[dict[str, object]]) -> None:
        super().__init__(KalshiAuthConfig())
        self._markets = markets

    def _now_ms(self) -> int:  # type: ignore[override]
        return 1_700_000_000_000

    def list_markets(  # type: ignore[override]
        self,
        *,
        status: Optional[str] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 200,
    ) -> dict[str, object]:
        _ = (status, event_ticker, series_ticker, cursor, limit)
        return {"markets": list(self._markets)}


class ClientMMMarketSelectTests(unittest.TestCase):
    def test_resolve_quoteable_mm_market_prefers_mid_closest_to_target(self) -> None:
        now_ms = 1_700_000_000_000
        markets = [
            {
                "ticker": "KXBTC15M-26FEB111200-00",
                "status": "open",
                "close_time": now_ms + 900_000,
                "yes_bid": 88,
                "yes_ask": 90,
            },
            {
                "ticker": "KXBTC15M-26FEB111200-10",
                "status": "open",
                "close_time": now_ms + 900_000,
                "yes_bid": 49,
                "yes_ask": 51,
            },
        ]
        client = _StubClient(markets)
        selected = client.resolve_quoteable_mm_market(
            series_ticker="KXBTC15M",
            min_seconds_to_close=60.0,
            min_spread_cents=1,
            max_spread_cents=20,
            target_mid_cents=50.0,
            max_mid_distance_cents=35.0,
        )
        self.assertEqual(str(selected.get("ticker")), "KXBTC15M-26FEB111200-10")

    def test_resolve_quoteable_mm_market_raises_when_all_markets_pinned(self) -> None:
        now_ms = 1_700_000_000_000
        markets = [
            {
                "ticker": "KXBTC15M-26FEB111200-00",
                "status": "open",
                "close_time": now_ms + 900_000,
                "yes_bid": 0,
                "yes_ask": 100,
            },
            {
                "ticker": "KXBTC15M-26FEB111200-10",
                "status": "open",
                "close_time": now_ms + 900_000,
                "yes_bid": 1,
                "yes_ask": 99,
            },
        ]
        client = _StubClient(markets)
        with self.assertRaises(KalshiRequestError):
            client.resolve_quoteable_mm_market(
                series_ticker="KXBTC15M",
                min_seconds_to_close=60.0,
                min_spread_cents=1,
                max_spread_cents=20,
                target_mid_cents=50.0,
                max_mid_distance_cents=35.0,
            )


if __name__ == "__main__":
    unittest.main()
