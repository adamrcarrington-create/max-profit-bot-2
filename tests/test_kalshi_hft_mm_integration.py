import json
import tempfile
import unittest

from kalshi_hft.client import KalshiRequestError
from kalshi_hft.engine import BotConfig, KalshiHFTEngine, PositionExposure, parse_ticker_close_epoch_ms


class KalshiHFTMMIntegrationTests(unittest.TestCase):
    def _engine(self) -> KalshiHFTEngine:
        cfg = BotConfig()
        cfg.strategy.strategy_mode = "mm"
        return KalshiHFTEngine(cfg)

    def test_from_json_parses_top_level_strategy_mode_and_mm_block(self) -> None:
        payload = {
            "strategy_mode": "mm",
            "strategy": {"series_ticker": "KXBTC15M"},
            "mm": {"quote_edge_cents": 4, "base_size": 3, "max_sum_buy_cents": 96},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(payload, tmp)
            tmp_path = tmp.name

        cfg = BotConfig.from_json(tmp_path)
        self.assertEqual(cfg.strategy.strategy_mode, "mm")
        self.assertEqual(cfg.mm.quote_edge_cents, 4)
        self.assertEqual(cfg.mm.base_size, 3)
        self.assertEqual(cfg.mm.max_sum_buy_cents, 96)

    def test_parse_ticker_close_epoch_ms(self) -> None:
        ts_ms = parse_ticker_close_epoch_ms("KXBTC15M-26FEB112130-30")
        self.assertIsNotNone(ts_ms)
        # 2026-02-11 21:30 ET == 2026-02-12 02:30 UTC
        self.assertEqual(ts_ms, 1770863400000)

    def test_mm_quote_math_from_mid(self) -> None:
        eng = self._engine()
        eng.cfg.mm.quote_edge_cents = 3
        eng.cfg.mm.improve_cents = 0
        yes_px, no_px = eng._mm_compute_quotes_from_mid(yb=40, ya=44, nb=56, na=60)
        self.assertEqual(yes_px, 40)
        self.assertEqual(no_px, 56)

    def test_mm_quote_math_respects_post_only_buffer(self) -> None:
        eng = self._engine()
        eng.cfg.mm.quote_edge_cents = 0
        eng.cfg.mm.improve_cents = 3
        eng.cfg.mm.post_only_buffer_cents = 2
        yes_px, no_px = eng._mm_compute_quotes_from_mid(yb=40, ya=44, nb=56, na=60)
        self.assertEqual(yes_px, 42)
        self.assertEqual(no_px, 58)

    def test_mm_quote_math_allows_toxicity_edge_widening(self) -> None:
        eng = self._engine()
        eng.cfg.mm.quote_edge_cents = 1
        eng.cfg.mm.improve_cents = 0
        yes_px, no_px = eng._mm_compute_quotes_from_mid(
            yb=40,
            ya=44,
            nb=56,
            na=60,
            yes_edge_add=2,
            no_edge_add=1,
        )
        self.assertEqual(yes_px, 40)
        self.assertEqual(no_px, 56)

    def test_mm_pinned_or_garbage_guard(self) -> None:
        eng = self._engine()
        micro = {"pinned_state": False, "zero_spread_state": False}
        self.assertTrue(eng._mm_book_is_pinned_or_garbage(yb=1, ya=1, nb=99, na=99, micro=micro))
        self.assertFalse(eng._mm_book_is_pinned_or_garbage(yb=45, ya=47, nb=53, na=55, micro=micro))

    def test_mm_buy_sizes_reduce_risk_and_straddle_cap(self) -> None:
        eng = self._engine()
        eng.cfg.mm.base_size = 2
        eng.cfg.mm.inv_scale = 8
        eng.cfg.mm.max_pos_per_side = 12
        eng.cfg.mm.max_gross_pos = 18
        eng.cfg.mm.max_net_exposure = 8
        eng.cfg.mm.max_sum_buy_cents = 97
        eng.cfg.mm.reduce_risk_secs = 90.0
        eng.cfg.mm.skip_if_buy_le_cents = 5

        # In risk-reduction window, new opening sizes should be zero.
        yes_sz, no_sz = eng._mm_buy_sizes(
            yes_pos=0,
            no_pos=0,
            buy_yes_px=45,
            buy_no_px=45,
            seconds_to_close=60.0,
        )
        self.assertEqual((yes_sz, no_sz), (0, 0))

        # Outside reduce-risk window, straddle cap should drop one side if sum is too high.
        yes_sz, no_sz = eng._mm_buy_sizes(
            yes_pos=0,
            no_pos=0,
            buy_yes_px=50,
            buy_no_px=49,
            seconds_to_close=200.0,
        )
        self.assertEqual((yes_sz, no_sz), (0, 2))

        # Avoid ultra-low probability tails where fee drag dominates.
        yes_sz, no_sz = eng._mm_buy_sizes(
            yes_pos=0,
            no_pos=0,
            buy_yes_px=4,
            buy_no_px=50,
            seconds_to_close=200.0,
        )
        self.assertEqual((yes_sz, no_sz), (0, 2))

    def test_mm_mode_ignores_reference_staleness_for_data_guard(self) -> None:
        eng = self._engine()
        micro = {
            "book_age_seconds": 0.1,
            "book_stale_seconds": 0.1,
        }
        # No reference updates yet -> stale for HFT path.
        self.assertTrue(eng._is_data_stale(micro, require_reference=True))
        # MM mode should only require orderbook freshness.
        self.assertFalse(eng._is_data_stale(micro, require_reference=False))

    def test_mm_delta_position_scope_uses_per_ticker_baseline_for_sizing(self) -> None:
        eng = self._engine()
        eng.cfg.risk.position_limit_scope = "delta_from_start"
        ticker = "KXBTC15M-26FEB112300-00"
        eng.current_ticker = ticker
        eng._positions_by_ticker[ticker] = PositionExposure(yes=139, no=0)
        eng._starting_positions_by_ticker[ticker] = PositionExposure(yes=139, no=0)

        effective = eng._effective_position_for_ticker(ticker)
        self.assertEqual((effective.yes, effective.no), (0, 0))
        self.assertEqual(eng._net_exposure_for_current_market(), 0)

        yes_sz, no_sz = eng._mm_buy_sizes(
            yes_pos=effective.yes,
            no_pos=effective.no,
            buy_yes_px=50,
            buy_no_px=49,
            seconds_to_close=500.0,
        )
        self.assertGreater(yes_sz + no_sz, 0)

        eng.cfg.risk.position_limit_scope = "account"
        effective_account = eng._effective_position_for_ticker(ticker)
        self.assertEqual((effective_account.yes, effective_account.no), (139, 0))

    def test_mm_skips_narrow_spread_for_new_buys(self) -> None:
        eng = self._engine()
        eng.cfg.execution.dry_run = True
        eng.cfg.mm.min_spread_cents = 2
        eng.current_ticker = "KXBTC15M-26FEB112300-00"
        eng._positions_by_ticker[eng.current_ticker] = PositionExposure()
        micro = {
            "best_yes_bid": 50,
            "best_yes_ask": 51,
            "best_no_bid": 49,
            "best_no_ask": 50,
            "book_age_seconds": 0.1,
            "book_stale_seconds": 0.1,
            "pinned_state": False,
            "zero_spread_state": False,
        }
        eng._run_mm_cycle(micro=micro, seconds_to_close=500.0)
        self.assertEqual(eng._mm_last_quote_epoch, 0.0)
        self.assertEqual(len(eng._orders_last_minute), 0)

    def test_mm_min_edge_after_fee_filters_quotes(self) -> None:
        eng = self._engine()
        eng.cfg.execution.dry_run = True
        eng.cfg.mm.quote_refresh_secs = 0.0
        eng.cfg.mm.min_spread_cents = 1
        eng.cfg.mm.min_edge_after_fee_cents = 3
        eng.current_ticker = "KXBTC15M-26FEB112300-00"
        eng._positions_by_ticker[eng.current_ticker] = PositionExposure()
        micro = {
            "best_yes_bid": 50,
            "best_yes_ask": 52,
            "best_no_bid": 48,
            "best_no_ask": 50,
            "book_age_seconds": 0.1,
            "book_stale_seconds": 0.1,
            "pinned_state": False,
            "zero_spread_state": False,
            "imbalance": 0.0,
            "yes_depth": 100,
            "no_depth": 100,
        }
        eng._run_mm_cycle(micro=micro, seconds_to_close=500.0)
        self.assertEqual(len(eng._orders_last_minute), 0)

    def test_mm_toxicity_pause_blocks_new_buy_quotes(self) -> None:
        eng = self._engine()
        eng.cfg.execution.dry_run = True
        eng.cfg.mm.quote_refresh_secs = 0.0
        eng.cfg.mm.min_spread_cents = 2
        eng.cfg.mm.toxicity_enabled = True
        eng.cfg.mm.toxic_mid_move_cents = 3.0
        eng.cfg.mm.toxic_mid_move_window_secs = 5.0
        eng.cfg.mm.toxic_pause_secs = 12.0
        eng.cfg.mm.toxic_imbalance_abs = 0.8
        eng.cfg.mm.toxic_depth_ratio = 2.0

        eng.current_ticker = "KXBTC15M-26FEB112300-00"
        eng._positions_by_ticker[eng.current_ticker] = PositionExposure()

        calm = {
            "best_yes_bid": 50,
            "best_yes_ask": 53,
            "best_no_bid": 47,
            "best_no_ask": 50,
            "book_age_seconds": 0.1,
            "book_stale_seconds": 0.1,
            "pinned_state": False,
            "zero_spread_state": False,
            "imbalance": 0.0,
            "yes_depth": 100,
            "no_depth": 100,
        }
        toxic = {
            "best_yes_bid": 56,
            "best_yes_ask": 59,
            "best_no_bid": 41,
            "best_no_ask": 44,
            "book_age_seconds": 0.1,
            "book_stale_seconds": 0.1,
            "pinned_state": False,
            "zero_spread_state": False,
            "imbalance": 0.95,
            "yes_depth": 320,
            "no_depth": 10,
        }

        eng._run_mm_cycle(micro=calm, seconds_to_close=500.0)
        orders_before = len(eng._orders_last_minute)
        self.assertGreaterEqual(orders_before, 1)

        eng._run_mm_cycle(micro=toxic, seconds_to_close=500.0)
        pause_until = eng._mm_toxic_pause_until_epoch
        self.assertGreater(pause_until, 0.0)
        orders_after_toxic = len(eng._orders_last_minute)
        self.assertEqual(orders_after_toxic, orders_before)

        eng._run_mm_cycle(micro=calm, seconds_to_close=500.0)
        self.assertEqual(len(eng._orders_last_minute), orders_after_toxic)

    def test_mm_toxicity_regime_skews_buy_size_by_imbalance(self) -> None:
        eng = self._engine()
        eng.cfg.execution.dry_run = True
        eng.cfg.mm.quote_refresh_secs = 0.0
        eng.cfg.mm.min_spread_cents = 2
        eng.cfg.mm.toxicity_enabled = True
        eng.cfg.mm.toxicity_regime_enabled = True
        eng.cfg.mm.toxic_mid_move_cents = 1000.0
        eng.cfg.mm.toxic_widen_imbalance_abs = 0.3
        eng.cfg.mm.toxic_widen_depth_ratio = 1.2
        eng.cfg.mm.toxic_widen_edge_cents = 1
        eng.cfg.mm.toxic_widen_size_mult = 1.0
        eng.cfg.mm.toxic_skew_imbalance_abs = 0.6
        eng.cfg.mm.toxic_skew_opp_size_mult = 0.0

        eng.current_ticker = "KXBTC15M-26FEB112300-00"
        eng._positions_by_ticker[eng.current_ticker] = PositionExposure()
        skewed = {
            "best_yes_bid": 50,
            "best_yes_ask": 54,
            "best_no_bid": 46,
            "best_no_ask": 50,
            "book_age_seconds": 0.1,
            "book_stale_seconds": 0.1,
            "pinned_state": False,
            "zero_spread_state": False,
            "imbalance": 0.9,
            "yes_depth": 350,
            "no_depth": 10,
        }
        eng._run_mm_cycle(micro=skewed, seconds_to_close=500.0)
        self.assertEqual(len(eng._orders_last_minute), 1)
        self.assertIsNotNone(eng._mm_last_px["buy"]["yes"])
        self.assertIsNone(eng._mm_last_px["buy"]["no"])

    def test_mm_queue_value_blocks_low_value_new_quotes(self) -> None:
        eng = self._engine()
        eng.cfg.execution.dry_run = True
        eng.cfg.mm.quote_refresh_secs = 0.0
        eng.cfg.mm.min_spread_cents = 1
        eng.cfg.mm.queue_value_enabled = True
        eng.cfg.mm.queue_value_new_min_cents = 1.0
        eng.cfg.mm.queue_value_top_qty_scale = 1.0

        eng.current_ticker = "KXBTC15M-26FEB112300-00"
        eng._positions_by_ticker[eng.current_ticker] = PositionExposure()
        micro = {
            "best_yes_bid": 50,
            "best_yes_ask": 52,
            "best_no_bid": 48,
            "best_no_ask": 50,
            "best_yes_bid_qty": 500,
            "best_no_bid_qty": 500,
            "book_age_seconds": 0.1,
            "book_stale_seconds": 0.1,
            "pinned_state": False,
            "zero_spread_state": False,
            "imbalance": 0.0,
            "yes_depth": 500,
            "no_depth": 500,
        }
        eng._run_mm_cycle(micro=micro, seconds_to_close=500.0)
        self.assertEqual(len(eng._orders_last_minute), 0)

    def test_mm_queue_value_holds_existing_quote_when_improve_small(self) -> None:
        eng = self._engine()
        eng.cfg.execution.dry_run = True
        eng.cfg.mm.min_quote_change_cents = 1
        eng.cfg.mm.queue_value_enabled = True
        eng.cfg.mm.queue_value_new_min_cents = 0.0
        eng.cfg.mm.queue_value_hold_min_cents = 0.01
        eng.cfg.mm.queue_value_min_improve_cents = 0.5

        eng.current_ticker = "KXBTC15M-26FEB112300-00"
        micro = {
            "best_yes_bid": 50,
            "best_yes_ask": 53,
            "best_no_bid": 47,
            "best_no_ask": 50,
            "best_yes_bid_qty": 12,
            "best_no_bid_qty": 12,
            "yes_depth": 80,
            "no_depth": 80,
        }
        eng._mm_upsert_buy_order(side="yes", count=1, price_cents=50, ask_cents=53, micro=micro)
        self.assertEqual(len(eng._orders_last_minute), 1)

        eng._mm_order_ids["buy"]["yes"] = "existing-yes"
        eng._mm_last_px["buy"]["yes"] = 50
        eng._mm_upsert_buy_order(side="yes", count=1, price_cents=51, ask_cents=53, micro=micro)
        self.assertEqual(len(eng._orders_last_minute), 1)
        self.assertEqual(eng._mm_last_px["buy"]["yes"], 50)

    def test_mm_queue_value_requotes_when_improve_is_large(self) -> None:
        eng = self._engine()
        eng.cfg.execution.dry_run = True
        eng.cfg.mm.min_quote_change_cents = 1
        eng.cfg.mm.queue_value_enabled = True
        eng.cfg.mm.queue_value_new_min_cents = 0.0
        eng.cfg.mm.queue_value_hold_min_cents = 0.01
        eng.cfg.mm.queue_value_min_improve_cents = 0.2
        eng.cfg.mm.queue_value_top_qty_scale = 1.0

        eng.current_ticker = "KXBTC15M-26FEB112300-00"
        micro = {
            "best_yes_bid": 50,
            "best_yes_ask": 53,
            "best_no_bid": 47,
            "best_no_ask": 50,
            "best_yes_bid_qty": 12,
            "best_no_bid_qty": 12,
            "yes_depth": 80,
            "no_depth": 80,
        }
        eng._mm_upsert_buy_order(side="yes", count=1, price_cents=50, ask_cents=53, micro=micro)
        self.assertEqual(len(eng._orders_last_minute), 1)

        eng._mm_order_ids["buy"]["yes"] = "existing-yes"
        eng._mm_last_px["buy"]["yes"] = 50
        eng._mm_upsert_buy_order(side="yes", count=1, price_cents=51, ask_cents=53, micro=micro)
        self.assertEqual(len(eng._orders_last_minute), 2)
        self.assertEqual(eng._mm_last_px["buy"]["yes"], 51)

    def test_mm_market_selector_prefers_quoteable_market(self) -> None:
        eng = self._engine()
        eng.cfg.mm.market_selection_enabled = True
        eng.cfg.mm.market_select_refresh_secs = 0.0
        eng.cfg.execution.dry_run = True
        eng.cfg.strategy.market_ticker = ""
        eng._reset_orderbook_monitor = lambda ticker: None  # type: ignore[method-assign]

        class StubClient:
            def __init__(self) -> None:
                self.quoteable_calls = 0
                self.active_calls = 0

            def resolve_quoteable_mm_market(self, **_: object) -> dict[str, object]:
                self.quoteable_calls += 1
                return {"ticker": "KXBTC15M-26FEB112300-00", "close_time": "2099-01-01T00:00:00Z"}

            def resolve_active_market(self, _: str) -> dict[str, object]:
                self.active_calls += 1
                return {"ticker": "KXBTC15M-26FEB112315-15", "close_time": "2099-01-01T00:15:00Z"}

            def get_market(self, ticker: str) -> dict[str, object]:
                return {"market": {"ticker": ticker}}

        stub = StubClient()
        eng.client = stub  # type: ignore[assignment]
        eng._switch_to_current_market(force=True)
        self.assertEqual(eng.current_ticker, "KXBTC15M-26FEB112300-00")
        self.assertEqual(stub.quoteable_calls, 1)
        self.assertEqual(stub.active_calls, 0)

    def test_mm_market_selector_falls_back_to_active_when_no_quoteable(self) -> None:
        eng = self._engine()
        eng.cfg.mm.market_selection_enabled = True
        eng.cfg.mm.market_select_refresh_secs = 0.0
        eng.cfg.execution.dry_run = True
        eng.cfg.strategy.market_ticker = ""
        eng._reset_orderbook_monitor = lambda ticker: None  # type: ignore[method-assign]

        class StubClient:
            def __init__(self) -> None:
                self.quoteable_calls = 0
                self.active_calls = 0

            def resolve_quoteable_mm_market(self, **_: object) -> dict[str, object]:
                self.quoteable_calls += 1
                raise KalshiRequestError("no_quoteable_mm_markets_for_series=KXBTC15M")

            def resolve_active_market(self, _: str) -> dict[str, object]:
                self.active_calls += 1
                return {"ticker": "KXBTC15M-26FEB112315-15", "close_time": "2099-01-01T00:15:00Z"}

            def get_market(self, ticker: str) -> dict[str, object]:
                return {"market": {"ticker": ticker}}

        stub = StubClient()
        eng.client = stub  # type: ignore[assignment]
        eng._switch_to_current_market(force=True)
        self.assertEqual(eng.current_ticker, "KXBTC15M-26FEB112315-15")
        self.assertEqual(stub.quoteable_calls, 1)
        self.assertEqual(stub.active_calls, 1)


if __name__ == "__main__":
    unittest.main()
