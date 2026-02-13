import unittest

from kalshi_mm.sandbox_bot import (
    _apply_toxicity_size_controls,
    _hydrate_markets,
    _build_flatten_payloads,
    _build_buy_yes_order_payload,
    _build_sell_no_order_payload,
    _get_single_position_qty,
    _market_seconds_to_close,
    _next_client_order_id,
    _session_stop_loss_hit,
    _session_take_profit_hit,
    _choose_best_market,
    _extract_positions,
    _extract_yes_bid_ask_prob,
    _list_markets_paginated,
    _prob_to_cents,
    _resolve_requested_markets,
    _resolve_by_symbol_search,
    _submit_post_only_with_reprice,
    _ticker_close_epoch_from_suffix,
)
from kalshi_mm.kalshi_api import KalshiApiError


class SandboxBotTests(unittest.TestCase):
    def test_extract_yes_bid_ask_from_dollar_fields(self) -> None:
        bid, ask = _extract_yes_bid_ask_prob({"yes_bid_dollars": "0.44", "yes_ask_dollars": "0.56"})
        self.assertAlmostEqual(bid, 0.44, places=6)
        self.assertAlmostEqual(ask, 0.56, places=6)

    def test_extract_yes_bid_ask_from_cent_fields(self) -> None:
        bid, ask = _extract_yes_bid_ask_prob({"yes_bid": 44, "yes_ask": 56})
        self.assertAlmostEqual(bid, 0.44, places=6)
        self.assertAlmostEqual(ask, 0.56, places=6)

    def test_extract_positions(self) -> None:
        payload = {
            "market_positions": [
                {"ticker": "KXBTC15M", "position": "7"},
                {"ticker": "KXETH15M", "position": -3},
            ]
        }
        positions = _extract_positions(payload)
        self.assertEqual(positions["KXBTC15M"], 7)
        self.assertEqual(positions["KXETH15M"], -3)

    def test_prob_to_cents(self) -> None:
        self.assertEqual(_prob_to_cents(0.512), 51)
        self.assertEqual(_prob_to_cents(0.999), 99)
        self.assertEqual(_prob_to_cents(0.0), 1)

    def test_resolve_by_symbol_search_prefers_15m(self) -> None:
        markets = [
            {
                "ticker": "KXBTCD-1H-FOO",
                "title": "Bitcoin hourly",
                "yes_bid": 48,
                "yes_ask": 52,
                "status": "open",
            },
            {
                "ticker": "KXBTC15M-BAR",
                "title": "Bitcoin 15 minute",
                "yes_bid": 49,
                "yes_ask": 51,
                "status": "open",
            },
        ]
        resolved = _resolve_by_symbol_search("KXBTC15M", markets)
        self.assertEqual(resolved, "KXBTC15M-BAR")

    def test_choose_best_market_prefers_tighter_spread(self) -> None:
        markets = [
            {"ticker": "A", "yes_bid": 30, "yes_ask": 70, "status": "open"},
            {"ticker": "B", "yes_bid": 49, "yes_ask": 51, "status": "open"},
        ]
        chosen = _choose_best_market(markets)
        self.assertEqual(chosen, "B")

    def test_buy_payload_allows_post_only_override(self) -> None:
        payload = _build_buy_yes_order_payload(
            ticker="KXBTC15M-TEST",
            count=3,
            yes_price=48,
            client_order_id="abc",
            post_only=False,
        )
        self.assertEqual(payload["action"], "buy")
        self.assertEqual(payload["side"], "yes")
        self.assertFalse(payload["post_only"])

    def test_sell_payload_shape(self) -> None:
        payload = _build_sell_no_order_payload(
            ticker="KXBTC15M-TEST",
            count=4,
            no_price=52,
            client_order_id="def",
            post_only=False,
        )
        self.assertEqual(payload["action"], "sell")
        self.assertEqual(payload["side"], "no")
        self.assertEqual(payload["count"], 4)
        self.assertEqual(payload["no_price"], 52)

    def test_submit_post_only_with_reprice_success(self) -> None:
        class _FakeClient:
            def __init__(self) -> None:
                self.calls = 0
                self.prices = []

            def create_order(self, payload):
                self.calls += 1
                self.prices.append(payload["yes_price"])
                if self.calls == 1:
                    raise KalshiApiError("http_error ... post only cross")
                return {}

        client = _FakeClient()
        ok, error, reprices = _submit_post_only_with_reprice(
            client=client,
            payload={"yes_price": 50},
            price_key="yes_price",
            max_retries=3,
        )
        self.assertTrue(ok)
        self.assertEqual(error, "")
        self.assertEqual(reprices, 1)
        self.assertEqual(client.prices, [50, 49])

    def test_submit_post_only_with_reprice_non_cross_error(self) -> None:
        class _FakeClient:
            def create_order(self, payload):
                raise KalshiApiError("http_error 400 invalid_order some_other_reason")

        ok, error, reprices = _submit_post_only_with_reprice(
            client=_FakeClient(),
            payload={"no_price": 44},
            price_key="no_price",
            max_retries=3,
        )
        self.assertFalse(ok)
        self.assertIn("some_other_reason", error)
        self.assertEqual(reprices, 0)

    def test_submit_post_only_with_reprice_order_exists_is_success(self) -> None:
        class _FakeClient:
            def create_order(self, payload):
                raise KalshiApiError("http_error 409 order_already_exists")

        ok, error, reprices = _submit_post_only_with_reprice(
            client=_FakeClient(),
            payload={"no_price": 44},
            price_key="no_price",
            max_retries=3,
        )
        self.assertTrue(ok)
        self.assertEqual(error, "")
        self.assertEqual(reprices, 0)

    def test_next_client_order_id_is_short_and_unique(self) -> None:
        oid1, seq1 = _next_client_order_id("mmabc123", "KXBTC15M-foo", "by", 0)
        oid2, seq2 = _next_client_order_id("mmabc123", "KXBTC15M-foo", "bn", seq1)
        self.assertLessEqual(len(oid1), 30)
        self.assertLessEqual(len(oid2), 30)
        self.assertNotEqual(oid1, oid2)
        self.assertEqual(seq2, 2)

    def test_build_flatten_payloads_for_positive_and_negative_qty(self) -> None:
        p1, b1, seq1 = _build_flatten_payloads(
            ticker="KXBTC15M-TEST",
            qty=5,
            count=5,
            run_tag="mm123",
            order_seq=0,
        )
        self.assertEqual(p1["action"], "buy")
        self.assertEqual(p1["side"], "no")
        self.assertEqual(p1["no_price"], 99)
        self.assertEqual(b1["action"], "sell")
        self.assertEqual(b1["side"], "yes")
        self.assertEqual(b1["yes_price"], 99)

        p2, b2, seq2 = _build_flatten_payloads(
            ticker="KXBTC15M-TEST",
            qty=-6,
            count=6,
            run_tag="mm123",
            order_seq=seq1,
        )
        self.assertEqual(p2["action"], "buy")
        self.assertEqual(p2["side"], "yes")
        self.assertEqual(p2["yes_price"], 99)
        self.assertEqual(b2["action"], "sell")
        self.assertEqual(b2["side"], "no")
        self.assertEqual(b2["no_price"], 99)
        self.assertGreater(seq2, seq1)

    def test_get_single_position_qty_from_ticker_filtered_response(self) -> None:
        class _FakeClient:
            def get_positions(self, ticker=None, limit=100):
                if ticker:
                    return {"market_positions": [{"ticker": ticker, "position": 12}]}
                return {"market_positions": []}

        qty = _get_single_position_qty(_FakeClient(), "KXBTC15M-TEST")
        self.assertEqual(qty, 12)

    def test_choose_best_market_skips_near_close_when_required(self) -> None:
        close_near = _ticker_close_epoch_from_suffix("KXBTC15M-26FEB110005-00")
        self.assertIsNotNone(close_near)
        now_epoch = close_near - 300.0
        markets = [
            {"ticker": "KXBTC15M-26FEB110005-00", "status": "open", "yes_bid": 49, "yes_ask": 51},
            {"ticker": "KXBTC15M-26FEB110045-00", "status": "open", "yes_bid": 49, "yes_ask": 51},
        ]
        chosen = _choose_best_market(markets, now_epoch=now_epoch, min_seconds_to_close=600.0)
        self.assertEqual(chosen, "KXBTC15M-26FEB110045-00")

    def test_choose_best_market_non_open_fallback_requires_explicit_status(self) -> None:
        close_epoch = _ticker_close_epoch_from_suffix("KXBTC15M-26FEB110030-30")
        self.assertIsNotNone(close_epoch)
        now_epoch = close_epoch - 600.0
        markets = [
            {"ticker": "KXBTC15M-26FEB110030-30", "status": "", "yes_bid": 49, "yes_ask": 51},
            {
                "ticker": "KXBTC15M-26FEB110045-45",
                "status": "scheduled",
                "yes_bid": 49,
                "yes_ask": 51,
            },
        ]
        chosen = _choose_best_market(
            markets,
            now_epoch=now_epoch,
            min_seconds_to_close=0.0,
            allow_non_open=True,
        )
        self.assertEqual(chosen, "KXBTC15M-26FEB110045-45")

    def test_market_seconds_to_close_from_ticker_suffix(self) -> None:
        close_epoch = _ticker_close_epoch_from_suffix("KXBTC15M-26FEB110015-15")
        self.assertIsNotNone(close_epoch)
        seconds = _market_seconds_to_close({"ticker": "KXBTC15M-26FEB110015-15"}, now_epoch=close_epoch - 900.0)
        self.assertIsNotNone(seconds)
        self.assertGreater(seconds, 890)
        self.assertLess(seconds, 910)

    def test_hydrate_markets_fetches_open_market(self) -> None:
        class _FakeClient:
            def get_market(self, ticker):
                return {"market": {"ticker": ticker, "status": "open", "yes_bid": 49, "yes_ask": 51}}

        markets = [{"ticker": "KXBTC15M-TEST"}]
        hydrated = _hydrate_markets(_FakeClient(), markets)
        self.assertEqual(len(hydrated), 1)
        self.assertEqual(hydrated[0]["status"], "open")

    def test_list_markets_paginated_consumes_cursor(self) -> None:
        class _FakeClient:
            def __init__(self) -> None:
                self.calls = []

            def list_markets(
                self,
                status=None,
                event_ticker=None,
                series_ticker=None,
                limit=200,
                cursor=None,
                auth=False,
            ):
                self.calls.append((status, cursor, auth))
                if cursor is None:
                    return {"markets": [{"ticker": "MKT-A"}, {"ticker": "MKT-B"}], "cursor": "page2"}
                if cursor == "page2":
                    return {"markets": [{"ticker": "MKT-B"}, {"ticker": "MKT-C"}], "cursor": ""}
                return {"markets": []}

        client = _FakeClient()
        markets = _list_markets_paginated(client, status="open", limit=2, max_pages=5)
        self.assertEqual([m["ticker"] for m in markets], ["MKT-A", "MKT-B", "MKT-C"])
        self.assertEqual(len(client.calls), 2)

    def test_resolve_requested_markets_can_disable_next_fallback(self) -> None:
        near_ticker = "KXBTC15M-26DEC312359-00"

        class _FakeClient:
            def list_markets(
                self,
                status=None,
                event_ticker=None,
                series_ticker=None,
                limit=200,
                cursor=None,
                auth=False,
            ):
                return {
                    "markets": [
                        {
                            "ticker": near_ticker,
                            "status": "scheduled",
                            "yes_bid": 49,
                            "yes_ask": 51,
                        }
                    ],
                    "cursor": "",
                }

            def list_events(self, series_ticker=None, status=None, limit=200, cursor=None, auth=False):
                return {"events": [], "cursor": ""}

            def get_market(self, ticker):
                raise KalshiApiError("not_found")

            def get_event(self, event_ticker):
                raise KalshiApiError("not_found")

            def get_series(self, series_ticker):
                return {"series": {"events": []}}

        client = _FakeClient()
        strict_resolved, _ = _resolve_requested_markets(
            client=client,
            requested_markets=["KXBTC15M"],
            min_seconds_to_close=900.0,
            allow_next_fallback=False,
        )
        loose_resolved, _ = _resolve_requested_markets(
            client=client,
            requested_markets=["KXBTC15M"],
            min_seconds_to_close=900.0,
            allow_next_fallback=True,
        )

        self.assertEqual(strict_resolved, [])
        self.assertEqual(loose_resolved, [near_ticker])

    def test_toxicity_controls_pause_when_flat_on_hard_momentum(self) -> None:
        yes_size, no_size, reason = _apply_toxicity_size_controls(
            yes_size=7,
            no_size=7,
            position_qty=0,
            momentum=0.020,
            volatility=0.004,
            soft_guard=0.0035,
            hard_guard=0.0085,
            adverse_side_size_cut=0.8,
        )
        self.assertEqual(yes_size, 0)
        self.assertEqual(no_size, 0)
        self.assertEqual(reason, "momentum_hard_guard")

    def test_toxicity_controls_keep_derisk_side_available(self) -> None:
        yes_size, no_size, reason = _apply_toxicity_size_controls(
            yes_size=5,
            no_size=2,
            position_qty=6,
            momentum=0.009,
            volatility=0.003,
            soft_guard=0.0035,
            hard_guard=0.0085,
            adverse_side_size_cut=0.8,
        )
        self.assertGreaterEqual(no_size, 1)
        self.assertGreaterEqual(yes_size, 0)
        self.assertEqual(reason, "")

    def test_session_stop_loss_hit(self) -> None:
        self.assertFalse(_session_stop_loss_hit(10000, 9500, 600))
        self.assertTrue(_session_stop_loss_hit(10000, 9300, 600))
        self.assertFalse(_session_stop_loss_hit(10000, 9000, 0))

    def test_session_take_profit_hit(self) -> None:
        self.assertFalse(_session_take_profit_hit(10000, 10200, 300))
        self.assertTrue(_session_take_profit_hit(10000, 10300, 300))
        self.assertFalse(_session_take_profit_hit(10000, 11000, 0))


if __name__ == "__main__":
    unittest.main()
