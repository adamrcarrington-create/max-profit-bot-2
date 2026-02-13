from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, field
import datetime as dt
import json
import logging
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, Optional
import uuid
from zoneinfo import ZoneInfo

from kalshi_hft.client import (
    KalshiAuthConfig,
    KalshiClient,
    KalshiRequestError,
    extract_float,
    extract_timestamp_ms,
    fallback_time_in_force_alias,
    normalize_time_in_force,
    parse_json_file,
)
from kalshi_hft.monitor import BinanceSpotMonitor, KalshiOrderbookMonitor


class KillSwitchTriggered(RuntimeError):
    """Raised when risk controls force an immediate stop."""


@dataclass
class PositionExposure:
    yes: int = 0
    no: int = 0

    @property
    def net(self) -> int:
        return int(self.yes - self.no)

    @property
    def gross(self) -> int:
        return int(self.yes + self.no)


@dataclass
class StrategyConfig:
    strategy_mode: str = "hft"  # hft | mm
    series_ticker: str = "KXBTC15M"
    market_ticker: str = ""
    binance_symbol: str = "BTCUSDT"
    edge_threshold: float = 0.05
    snipe_edge_threshold: float = 0.09
    maker_only: bool = False
    garbage_state_edge_boost: float = 0.01
    max_edge_for_sizing: float = 0.20
    base_order_count: int = 2
    max_order_count: int = 8
    depth_levels: int = 5
    momentum_window_seconds: float = 30.0
    momentum_full_scale_dollars: float = 100.0
    distance_scale_pct: float = 0.0015
    distance_weight: float = 0.7
    momentum_weight: float = 0.3
    time_to_close_reference_seconds: float = 900.0
    min_distance_scale_multiplier: float = 0.35
    max_distance_scale_multiplier: float = 1.25
    model_edge_multiplier: float = 1.0
    min_ev_cents_per_contract: float = 0.20
    mm_half_spread_cents: float = 1.0
    max_yes_spread_cents_for_mm: float = 35.0
    max_yes_spread_cents_for_snipe: float = 20.0
    use_fractional_kelly: bool = False
    kelly_fraction: float = 0.5
    kelly_bankroll_contracts: float = 25.0
    maker_fee_cents_per_contract: float = 0.03
    taker_fee_cents_per_contract: float = 0.07
    maker_slippage_cents_per_contract: float = 0.20
    taker_slippage_cents_per_contract: float = 0.50
    cross_side_arb_enabled: bool = False
    cross_side_arb_allow_sell_pair: bool = True
    cross_side_arb_min_edge_cents: float = 0.20
    cross_side_arb_max_count: int = 1
    inventory_exit_enabled: bool = True
    inventory_exit_min_edge_cents: float = 0.0
    inventory_exit_max_count: int = 2
    inventory_reduce_over_limit_ratio: float = 0.8
    spike_cancel_resting_orders: bool = True
    spike_cancel_cooldown_seconds: float = 3.0
    inventory_skew_cents_per_contract: float = 0.02
    reference_stale_seconds: float = 2.5
    book_stale_seconds: float = 3.0
    book_unchanged_seconds: float = 4.0
    adverse_selection_horizon_seconds: float = 5.0
    adverse_selection_move_threshold: float = 15.0
    vol_window_short_seconds: float = 60.0
    vol_window_long_seconds: float = 300.0
    calm_vol_threshold: float = 0.0006
    spiky_vol_threshold: float = 0.0020
    trending_momentum_threshold: float = 0.0012
    include_perp_reference: bool = True
    perp_weight: float = 0.5
    cross_exchange_max_diff_pct: float = 0.001
    flash_spike_pause_seconds: float = 5.0


@dataclass
class MMConfig:
    quote_edge_cents: int = 3
    improve_cents: int = 0
    post_only_buffer_cents: int = 1
    quote_refresh_secs: float = 1.2
    min_quote_change_cents: int = 2
    min_spread_cents: int = 1
    max_spread_cents: int = 20
    skip_pinned_markets: bool = True
    base_size: int = 2
    max_pos_per_side: int = 12
    max_gross_pos: int = 18
    max_net_exposure: int = 8
    inv_scale: int = 8
    max_sum_buy_cents: int = 97
    min_edge_after_fee_cents: int = 0
    queue_value_enabled: bool = False
    queue_value_new_min_cents: float = 0.15
    queue_value_hold_min_cents: float = 0.15
    queue_value_min_improve_cents: float = 0.25
    queue_value_top_qty_scale: float = 25.0
    queue_value_inside_spread_bonus: float = 0.5
    reduce_risk_secs: float = 90.0
    tp_min_profit_over_cost_cents: int = 3
    fee_buffer_cents: int = 2
    tp_clip: int = 6
    tp_throttle_secs: float = 1.0
    skip_if_buy_le_cents: int = 1
    skip_if_buy_ge_cents: int = 98
    cross_cooldown_secs: float = 2.0
    reconcile_secs: float = 6.0
    fills_refresh_secs: float = 3.0
    toxicity_enabled: bool = True
    toxicity_regime_enabled: bool = False
    toxic_widen_imbalance_abs: float = 0.55
    toxic_widen_depth_ratio: float = 2.0
    toxic_widen_edge_cents: int = 1
    toxic_widen_size_mult: float = 0.7
    toxic_skew_imbalance_abs: float = 0.72
    toxic_skew_opp_size_mult: float = 0.35
    toxic_imbalance_abs: float = 0.78
    toxic_depth_ratio: float = 3.5
    toxic_mid_move_cents: float = 5.0
    toxic_mid_move_window_secs: float = 2.5
    toxic_pause_secs: float = 8.0
    market_selection_enabled: bool = False
    market_select_refresh_secs: float = 12.0
    market_target_mid_cents: float = 50.0
    market_max_mid_distance_cents: float = 35.0


@dataclass
class ExecutionConfig:
    loop_seconds: float = 1.0
    dry_run: bool = True
    post_only: bool = True
    time_in_force: str = "good_till_canceled"
    ioc_time_in_force: str = "immediate_or_cancel"
    max_orders_per_minute: int = 30
    runtime_minutes: Optional[float] = None


@dataclass
class RiskConfig:
    max_hourly_drawdown_pct: float = 0.10
    daily_max_loss_pct: float = 0.10
    min_seconds_to_close: int = 60
    no_quote_before_close_seconds: int = 120
    force_flatten_seconds: int = 30
    force_flatten_cooldown_seconds: float = 2.0
    max_open_orders: int = 50
    max_gross_position: int = 250
    max_net_exposure: int = 150
    position_limit_scope: str = "account"  # account | delta_from_start
    cancel_open_orders_on_kill: bool = True
    cancel_open_orders_on_shutdown: bool = False
    balance_poll_seconds: float = 30.0
    position_poll_seconds: float = 8.0


@dataclass
class PathConfig:
    performance_json: str = "logs/performance.json"
    live_log: str = "logs/kalshi_hft.log"


@dataclass
class BotConfig:
    kalshi: KalshiAuthConfig = field(default_factory=KalshiAuthConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    mm: MMConfig = field(default_factory=MMConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def from_json(cls, file_path: str | Path) -> "BotConfig":
        raw = parse_json_file(file_path)
        strategy_payload = dict(raw.get("strategy", {}))
        if "strategy_mode" in raw and "strategy_mode" not in strategy_payload:
            strategy_payload["strategy_mode"] = raw.get("strategy_mode")
        return cls(
            kalshi=KalshiAuthConfig(**raw.get("kalshi", {})),
            strategy=StrategyConfig(**strategy_payload),
            mm=MMConfig(**raw.get("mm", {})),
            execution=ExecutionConfig(**raw.get("execution", {})),
            risk=RiskConfig(**raw.get("risk", {})),
            paths=PathConfig(**raw.get("paths", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KalshiHFTEngine:
    def __init__(self, cfg: BotConfig) -> None:
        self.cfg = cfg
        self.client = KalshiClient(cfg.kalshi)
        symbol = cfg.strategy.binance_symbol.strip().upper()
        if not symbol:
            symbol = infer_binance_symbol(cfg.strategy.series_ticker)
            cfg.strategy.binance_symbol = symbol
        self.spot_monitor = BinanceSpotMonitor(
            symbol=symbol,
            window_seconds=cfg.strategy.momentum_window_seconds,
            include_perp_reference=cfg.strategy.include_perp_reference,
            perp_weight=cfg.strategy.perp_weight,
            cross_exchange_max_diff_pct=cfg.strategy.cross_exchange_max_diff_pct,
            flash_spike_pause_seconds=cfg.strategy.flash_spike_pause_seconds,
        )

        self.book_monitor: Optional[KalshiOrderbookMonitor] = None
        self.current_market: Dict[str, Any] = {}
        self.current_ticker: str = ""
        self._order_targets: Dict[str, Dict[str, Any]] = {}
        self._orders_last_minute: deque[float] = deque()
        self._balance_history: deque[tuple[float, float]] = deque(maxlen=4000)
        self._positions_by_ticker: Dict[str, PositionExposure] = {}
        self._gross_position: int = 0
        self._net_exposure: int = 0
        self._starting_gross_position: Optional[int] = None
        self._starting_net_exposure: Optional[int] = None
        self._starting_positions_by_ticker: Dict[str, PositionExposure] = {}
        self._last_balance_poll: float = 0.0
        self._last_position_poll: float = 0.0
        self._last_perf_write: float = 0.0
        self._last_market_resolve: float = 0.0
        self._last_market_fetch: float = 0.0
        self._last_guard_cancel_epoch: float = 0.0
        self._last_401_epoch: float = 0.0
        self._last_flash_spike_log_epoch: float = 0.0
        self._last_spike_cancel_epoch: float = 0.0
        self._last_force_flatten_epoch: float = 0.0
        self._order_attempts: int = 0
        self._order_failures: int = 0
        self._last_no_market_log: float = 0.0
        self._daily_start_balance: Optional[float] = None
        self._daily_balance_day: str = ""
        self._latest_reference_returns: Dict[str, float] = {
            "return_30s": 0.0,
            "return_1m": 0.0,
            "return_15m": 0.0,
        }
        self._flat_side_toggle: bool = False
        self._mm_order_ids: Dict[str, Dict[str, Optional[str]]] = {
            "buy": {"yes": None, "no": None},
            "sell": {"yes": None, "no": None},
        }
        self._mm_last_px: Dict[str, Dict[str, Optional[int]]] = {
            "buy": {"yes": None, "no": None},
            "sell": {"yes": None, "no": None},
        }
        self._mm_last_quote_epoch: float = 0.0
        self._mm_last_reconcile_epoch: float = 0.0
        self._mm_last_tp_epoch: float = 0.0
        self._mm_last_cross_epoch: Dict[str, float] = {"yes": 0.0, "no": 0.0}
        self._mm_last_fills_fetch_epoch: float = 0.0
        self._mm_fills_cache_by_ticker: Dict[str, list[dict[str, Any]]] = {}
        self._mm_fill_min_ts_ms: int = int((time.time() - 24.0 * 3600.0) * 1000)
        self._mm_last_stale_reconnect_epoch: float = 0.0
        self._mm_mid_history: deque[tuple[float, float]] = deque(maxlen=240)
        self._mm_toxic_pause_until_epoch: float = 0.0
        self._mm_last_toxic_log_epoch: float = 0.0
        self._mm_last_toxic_regime: str = "normal"
        self._mm_last_market_scan_epoch: float = 0.0
        self._logger = setup_logger(cfg.paths.live_log)
        self.cfg.strategy.strategy_mode = self._strategy_mode()

    def run(self) -> int:
        start_epoch = time.time()
        runtime_seconds = (
            float(self.cfg.execution.runtime_minutes) * 60.0
            if self.cfg.execution.runtime_minutes is not None
            else None
        )

        try:
            self._switch_to_current_market(force=True)
            self._poll_balance(force=True, regime_name="calm")
            self._poll_positions(force=True)
            while True:
                loop_start = time.time()
                if runtime_seconds is not None and loop_start - start_epoch >= runtime_seconds:
                    self._logger.info("runtime limit reached; exiting.")
                    break

                self._switch_to_current_market(force=False)
                if not self.current_ticker:
                    time.sleep(max(0.1, self.cfg.execution.loop_seconds))
                    continue
                self._refresh_market_metadata_if_needed()
                micro = self._get_microstructure(allow_rest_fallback=not self._is_mm_mode())

                seconds_to_close = self._seconds_to_close(self.current_market)
                if seconds_to_close is not None and seconds_to_close <= self.cfg.risk.force_flatten_seconds:
                    self._force_flatten_current_market(micro)
                    time.sleep(max(0.05, self.cfg.execution.loop_seconds))
                    continue

                if self._is_mm_mode():
                    if self._is_data_stale(micro, require_reference=False):
                        self._handle_data_stale_guard(micro)
                        time.sleep(max(0.05, self.cfg.execution.loop_seconds))
                        continue
                    self._poll_positions(force=False)
                    self._enforce_position_hard_limits()
                    self._run_mm_cycle(micro=micro, seconds_to_close=seconds_to_close)
                    implied_yes = self._compute_implied_yes(micro)
                    self._poll_balance(force=False, regime_name="mm")
                    self._write_performance_if_due(
                        real_yes=implied_yes,
                        implied_yes=implied_yes,
                        regime_name="mm",
                        micro=micro,
                    )
                    sleep_seconds = max(0.0, self.cfg.execution.loop_seconds - (time.time() - loop_start))
                    if sleep_seconds:
                        time.sleep(sleep_seconds)
                    continue

                spot_price = self._safe_poll_spot()
                if spot_price is None:
                    time.sleep(max(0.05, self.cfg.execution.loop_seconds))
                    continue
                self._latest_reference_returns = {
                    "return_30s": self.spot_monitor.return_over_window(30.0),
                    "return_1m": self.spot_monitor.return_over_window(60.0),
                    "return_15m": self.spot_monitor.return_over_window(900.0),
                }
                pause_remaining = self.spot_monitor.pause_remaining_seconds()
                if pause_remaining > 0:
                    self._cancel_for_volatility_guard(reason="flash_spike_guard")
                    now = time.time()
                    if now - self._last_flash_spike_log_epoch >= 1.0:
                        self._last_flash_spike_log_epoch = now
                        self._logger.warning(
                            "flash_spike_guard_active pause_remaining=%.2fs reason=%s source=%s diff_pct=%.6f",
                            pause_remaining,
                            self.spot_monitor.last_pause_reason,
                            self.spot_monitor.last_reference_source,
                            self.spot_monitor.last_cross_exchange_diff_pct,
                        )
                    time.sleep(min(5.0, max(0.2, pause_remaining)))
                    continue

                regime = self._classify_regime(spot_price)
                if str(regime.get("name")) == "spiky":
                    self._cancel_for_volatility_guard(reason="spiky_regime")
                if self._is_data_stale(micro, require_reference=True):
                    self._handle_data_stale_guard(micro)
                    time.sleep(max(0.05, self.cfg.execution.loop_seconds))
                    continue

                self._poll_positions(force=False)
                self._enforce_position_hard_limits()

                implied_yes = self._compute_implied_yes(micro)
                strike = self._extract_strike_price(self.current_market, fallback_spot=spot_price)
                direction = self._infer_direction(self.current_market)
                momentum_delta = self.spot_monitor.delta_over_window(self.cfg.strategy.momentum_window_seconds)
                model_yes = self._real_world_probability(
                    spot=spot_price,
                    strike=strike,
                    direction=direction,
                    momentum_delta=momentum_delta,
                    seconds_to_close=seconds_to_close,
                )
                real_yes = self._blend_model_with_implied(
                    model_yes=model_yes,
                    implied_yes=implied_yes,
                )
                edge = real_yes - implied_yes

                edge_threshold = self.cfg.strategy.edge_threshold + regime["edge_boost"]
                if micro["pinned_state"] or micro["zero_spread_state"]:
                    edge_threshold += self.cfg.strategy.garbage_state_edge_boost

                allow_new_quotes = True
                if seconds_to_close is not None and seconds_to_close <= self.cfg.risk.no_quote_before_close_seconds:
                    allow_new_quotes = False

                if allow_new_quotes:
                    self._maybe_place_cross_side_arb(
                        micro=micro,
                        seconds_to_close=seconds_to_close,
                    )
                    self._maybe_reduce_inventory(
                        real_yes=real_yes,
                        micro=micro,
                        seconds_to_close=seconds_to_close,
                    )

                if allow_new_quotes and abs(edge) >= edge_threshold:
                    self._maybe_place_limit_order(
                        edge=edge,
                        real_yes=real_yes,
                        implied_yes=implied_yes,
                        micro=micro,
                        regime=regime,
                        seconds_to_close=seconds_to_close,
                    )

                self._poll_balance(force=False, regime_name=str(regime["name"]))
                self._write_performance_if_due(
                    real_yes=real_yes,
                    implied_yes=implied_yes,
                    regime_name=str(regime["name"]),
                    micro=micro,
                )

                sleep_seconds = max(0.0, self.cfg.execution.loop_seconds - (time.time() - loop_start))
                if sleep_seconds:
                    time.sleep(sleep_seconds)
        except KillSwitchTriggered as exc:
            self._logger.critical("kill_switch_triggered reason=%s", exc)
            return 12
        except KeyboardInterrupt:
            self._logger.info("keyboard interrupt, stopping.")
            return 0
        finally:
            self._shutdown()
        return 0

    def _strategy_mode(self) -> str:
        raw = str(self.cfg.strategy.strategy_mode or "hft").strip().lower()
        aliases = {
            "hft": "hft",
            "edge": "hft",
            "mm": "mm",
            "market_making": "mm",
            "market-making": "mm",
        }
        mode = aliases.get(raw)
        if mode is None:
            raise ValueError(f"invalid strategy_mode={self.cfg.strategy.strategy_mode!r}; expected 'hft' or 'mm'")
        return mode

    def _is_mm_mode(self) -> bool:
        return self.cfg.strategy.strategy_mode == "mm"

    def _mm_reset_state(self) -> None:
        self._mm_order_ids = {
            "buy": {"yes": None, "no": None},
            "sell": {"yes": None, "no": None},
        }
        self._mm_last_px = {
            "buy": {"yes": None, "no": None},
            "sell": {"yes": None, "no": None},
        }
        self._mm_last_quote_epoch = 0.0
        self._mm_last_reconcile_epoch = 0.0
        self._mm_last_tp_epoch = 0.0
        self._mm_last_cross_epoch = {"yes": 0.0, "no": 0.0}
        self._mm_last_fills_fetch_epoch = 0.0
        self._mm_fills_cache_by_ticker.clear()
        self._mm_mid_history.clear()
        self._mm_toxic_pause_until_epoch = 0.0
        self._mm_last_toxic_log_epoch = 0.0
        self._mm_last_toxic_regime = "normal"

    def _mm_cancel_slot(self, *, kind: str, side: str) -> None:
        order_id = self._mm_order_ids.get(kind, {}).get(side)
        if order_id and not self.cfg.execution.dry_run:
            try:
                self.client.cancel_order(order_id)
            except KalshiRequestError as exc:
                err_text = str(exc).lower()
                # Racy cancel-before-replace is expected when the order was already filled/canceled.
                if "status=404" in err_text or "not_found" in err_text or "not found" in err_text:
                    self._logger.info(
                        "mm_cancel_order_already_gone ticker=%s kind=%s side=%s order_id=%s",
                        self.current_ticker or "NONE",
                        kind,
                        side,
                        order_id,
                    )
                else:
                    self._log_kalshi_error(
                        "mm_cancel_order_error",
                        exc,
                        ticker=self.current_ticker or "NONE",
                        kind=kind,
                        side=side,
                        order_id=order_id,
                    )
        self._mm_order_ids[kind][side] = None
        self._mm_last_px[kind][side] = None

    def _mm_cancel_tracked_orders(self, *, cancel_buy: bool, cancel_sell: bool) -> None:
        if cancel_buy:
            self._mm_cancel_slot(kind="buy", side="yes")
            self._mm_cancel_slot(kind="buy", side="no")
        if cancel_sell:
            self._mm_cancel_slot(kind="sell", side="yes")
            self._mm_cancel_slot(kind="sell", side="no")

    def _mm_reconcile_order_ids(self) -> None:
        if self.cfg.execution.dry_run or not self.current_ticker:
            return
        now = time.time()
        if now - self._mm_last_reconcile_epoch < max(1.0, float(self.cfg.mm.reconcile_secs)):
            return
        self._mm_last_reconcile_epoch = now
        try:
            payload = self.client.get_orders(ticker=self.current_ticker, status="resting", limit=200)
        except KalshiRequestError as exc:
            self._log_kalshi_error("mm_reconcile_error", exc, ticker=self.current_ticker)
            return
        resting = payload.get("orders", [])
        if not isinstance(resting, list):
            resting = []
        ids = {
            str(row.get("order_id") or row.get("id") or "").strip()
            for row in resting
            if isinstance(row, dict)
        }
        for kind in ("buy", "sell"):
            for side in ("yes", "no"):
                oid = self._mm_order_ids[kind][side]
                if oid and oid not in ids:
                    self._mm_order_ids[kind][side] = None
                    self._mm_last_px[kind][side] = None

    def _mm_topbook(self, micro: Dict[str, Any]) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        yb = maybe_int(micro.get("best_yes_bid"))
        ya = maybe_int(micro.get("best_yes_ask"))
        nb = maybe_int(micro.get("best_no_bid"))
        na = maybe_int(micro.get("best_no_ask"))
        if ya is None and nb is not None:
            ya = 100 - nb
        if na is None and yb is not None:
            na = 100 - yb
        if yb is None and na is not None:
            yb = 100 - na
        if nb is None and ya is not None:
            nb = 100 - ya

        def _clamp_cents(value: Optional[int]) -> Optional[int]:
            if value is None:
                return None
            return int(clamp(int(value), 1, 99))

        return _clamp_cents(yb), _clamp_cents(ya), _clamp_cents(nb), _clamp_cents(na)

    def _mm_book_is_pinned_or_garbage(self, *, yb: int, ya: int, nb: int, na: int, micro: Dict[str, Any]) -> bool:
        if ya <= yb or na <= nb:
            return True
        if bool(micro.get("pinned_state")) or bool(micro.get("zero_spread_state")):
            return True
        if (yb == 1 and ya == 1) or (nb == 99 and na == 99):
            return True
        if yb <= 1 and nb >= 99:
            return True
        if ya <= 2 and na >= 98:
            return True
        if (yb <= 2 and ya <= 3) or (nb >= 98 and na >= 99):
            return True
        return False

    def _mm_compute_quotes_from_mid(
        self,
        *,
        yb: int,
        ya: int,
        nb: int,
        na: int,
        yes_edge_add: int = 0,
        no_edge_add: int = 0,
    ) -> tuple[int, int]:
        yes_mid = int(round((yb + ya) / 2.0))
        no_mid = int(round((nb + na) / 2.0))
        edge = max(0, int(self.cfg.mm.quote_edge_cents))
        yes_edge = max(0, edge + int(yes_edge_add))
        no_edge = max(0, edge + int(no_edge_add))
        improve = max(0, int(self.cfg.mm.improve_cents))
        post_only_buffer = max(1, int(self.cfg.mm.post_only_buffer_cents))
        yes_px = max(1, yes_mid - yes_edge, yb + improve)
        no_px = max(1, no_mid - no_edge, nb + improve)
        yes_px = min(yes_px, max(1, ya - post_only_buffer))
        no_px = min(no_px, max(1, na - post_only_buffer))
        return int(clamp(yes_px, 1, 99)), int(clamp(no_px, 1, 99))

    def _mm_should_requote(self, prev_px: Optional[int], new_px: int) -> bool:
        if prev_px is None:
            return True
        threshold = max(1, int(self.cfg.mm.min_quote_change_cents))
        return abs(int(new_px) - int(prev_px)) >= threshold

    def _mm_fill_price_cents(self, fill: Dict[str, Any], side: str) -> Optional[int]:
        raw = fill.get("price")
        if raw is None:
            raw = fill.get("yes_price") if side == "yes" else fill.get("no_price")
        if raw is None:
            raw = fill.get("yes_price")
        if raw is None:
            raw = fill.get("no_price")
        value = extract_float(raw)
        if value is None:
            return None
        if 0.0 <= value <= 1.0:
            return int(round(value * 100.0))
        if 1.0 < value <= 99.0:
            return int(round(value))
        return int(round(value / 100.0))

    def _mm_fee_to_cents(self, raw_fee: Any) -> float:
        fee = extract_float(raw_fee)
        if fee is None or fee <= 0:
            return 0.0
        # Kalshi fill fees are commonly returned in dollars for fills endpoints.
        if fee <= 5.0:
            return float(fee) * 100.0
        return float(fee)

    def _mm_avg_cost_from_fills(self, fills: list[dict[str, Any]]) -> Dict[str, Optional[int]]:
        rows = sorted(
            [f for f in fills if isinstance(f, dict)],
            key=lambda f: str(
                f.get("created_time")
                or f.get("created_ts")
                or f.get("ts")
                or f.get("fill_time")
                or f.get("id")
                or ""
            ),
        )
        qty = {"yes": 0.0, "no": 0.0}
        cost = {"yes": 0.0, "no": 0.0}
        for fill in rows:
            side = str(fill.get("side") or "").lower()
            if side not in {"yes", "no"}:
                continue
            action = str(fill.get("action") or "").lower()
            count = extract_float(fill.get("count_fp") or fill.get("count") or fill.get("quantity") or fill.get("size"))
            if count is None or count <= 0:
                continue
            price_cents = self._mm_fill_price_cents(fill, side)
            if price_cents is None:
                continue
            fee_cents = self._mm_fee_to_cents(fill.get("fee_cost") or fill.get("fee"))
            if action == "buy":
                qty[side] += count
                cost[side] += float(price_cents) * count + fee_cents
            elif action == "sell" and qty[side] > 0:
                reduce_q = min(qty[side], count)
                avg_basis = cost[side] / qty[side] if qty[side] > 0 else 0.0
                qty[side] -= reduce_q
                cost[side] -= avg_basis * reduce_q
                if qty[side] <= 1e-9:
                    qty[side] = 0.0
                    cost[side] = 0.0
        out: Dict[str, Optional[int]] = {"yes": None, "no": None}
        for side in ("yes", "no"):
            if qty[side] <= 0:
                out[side] = None
                continue
            out[side] = int(round(max(0.0, cost[side]) / qty[side]))
        return out

    def _mm_fetch_fills_current_ticker(self) -> list[dict[str, Any]]:
        if self.cfg.execution.dry_run or not self.current_ticker:
            return []
        now = time.time()
        refresh = max(0.2, float(self.cfg.mm.fills_refresh_secs))
        if (
            now - self._mm_last_fills_fetch_epoch < refresh
            and self.current_ticker in self._mm_fills_cache_by_ticker
        ):
            return self._mm_fills_cache_by_ticker[self.current_ticker]
        try:
            payload = self.client.get_fills(
                ticker=self.current_ticker,
                limit=200,
                min_ts=self._mm_fill_min_ts_ms,
            )
        except KalshiRequestError as exc:
            self._log_kalshi_error("mm_fill_poll_error", exc, ticker=self.current_ticker)
            return self._mm_fills_cache_by_ticker.get(self.current_ticker, [])
        rows = payload.get("fills", [])
        if not isinstance(rows, list):
            rows = []
        typed_rows = [r for r in rows if isinstance(r, dict)]
        self._mm_fills_cache_by_ticker[self.current_ticker] = typed_rows
        self._mm_last_fills_fetch_epoch = now
        return typed_rows

    def _mm_place_order_raw(
        self,
        *,
        side: str,
        action: str,
        count: int,
        price_cents: int,
        post_only: bool,
        time_in_force: str,
        reduce_only: bool,
        client_order_id: str,
    ) -> Dict[str, Any]:
        tif = normalize_time_in_force(time_in_force)
        payload: Dict[str, Any] = {
            "ticker": self.current_ticker,
            "action": action,
            "type": "limit",
            "side": side,
            "count": int(count),
            "post_only": bool(post_only),
            "time_in_force": tif,
            "client_order_id": client_order_id,
        }
        if reduce_only:
            payload["reduce_only"] = True
        if side == "yes":
            payload["yes_price"] = int(price_cents)
        else:
            payload["no_price"] = int(price_cents)
        try:
            return self.client._request("POST", "/portfolio/orders", auth_required=True, payload=payload)
        except KalshiRequestError as exc:
            lower = str(exc).lower()
            if "timeinforce" in lower or "time_in_force" in lower:
                fallback_tif = fallback_time_in_force_alias(payload["time_in_force"])
                if fallback_tif and fallback_tif != payload["time_in_force"]:
                    payload["time_in_force"] = fallback_tif
                    return self.client._request("POST", "/portfolio/orders", auth_required=True, payload=payload)
            if reduce_only and ("reduce_only" in lower or "invalid_parameter" in lower):
                payload.pop("reduce_only", None)
                return self.client._request("POST", "/portfolio/orders", auth_required=True, payload=payload)
            raise

    def _mm_read_order_id(self, response: Dict[str, Any]) -> Optional[str]:
        if not isinstance(response, dict):
            return None
        order = response.get("order") or response.get("data") or response
        if isinstance(order, dict):
            oid = str(order.get("order_id") or order.get("id") or "").strip()
            return oid or None
        return None

    def _mm_order_rate_available(self) -> bool:
        now = time.time()
        self._prune_order_rate_window(now)
        return len(self._orders_last_minute) < int(self.cfg.execution.max_orders_per_minute)

    def _mm_queue_quote_value_cents(
        self,
        *,
        side: str,
        price_cents: int,
        ask_cents: int,
        micro: Dict[str, Any],
    ) -> float:
        if side not in {"yes", "no"}:
            return 0.0
        spread_key = "spread_yes" if side == "yes" else "spread_no"
        spread = float(extract_float(micro.get(spread_key)) or max(0.0, float(ask_cents - price_cents)))
        net_edge = float(ask_cents - price_cents) - (
            float(self.cfg.strategy.maker_fee_cents_per_contract)
            + float(self.cfg.strategy.maker_slippage_cents_per_contract)
        )
        if net_edge <= 0.0:
            return 0.0

        if side == "yes":
            best_bid = maybe_int(micro.get("best_yes_bid"))
            best_bid_qty = float(extract_float(micro.get("best_yes_bid_qty")) or 0.0)
            side_depth = float(extract_float(micro.get("yes_depth")) or 0.0)
        else:
            best_bid = maybe_int(micro.get("best_no_bid"))
            best_bid_qty = float(extract_float(micro.get("best_no_bid_qty")) or 0.0)
            side_depth = float(extract_float(micro.get("no_depth")) or 0.0)

        queue_ahead = 0.0
        inside_spread = False
        if best_bid is not None:
            if int(price_cents) > int(best_bid):
                inside_spread = True
            elif int(price_cents) == int(best_bid):
                queue_ahead = max(0.0, best_bid_qty)
            else:
                queue_ahead = max(0.0, best_bid_qty + side_depth)

        qty_scale = max(1.0, float(self.cfg.mm.queue_value_top_qty_scale))
        queue_factor = math.exp(-queue_ahead / qty_scale)
        spread_factor = clamp(
            spread / max(1.0, float(self.cfg.mm.min_spread_cents)),
            0.25,
            1.5,
        )
        fill_prob = queue_factor * spread_factor
        if inside_spread:
            fill_prob *= 1.0 + max(0.0, float(self.cfg.mm.queue_value_inside_spread_bonus))
        fill_prob = clamp(fill_prob, 0.02, 1.0)
        return float(net_edge * fill_prob)

    def _mm_queue_allows_new_order(
        self,
        *,
        side: str,
        count: int,
        price_cents: int,
        ask_cents: int,
        micro: Dict[str, Any],
    ) -> bool:
        if count <= 0:
            return False
        if not bool(self.cfg.mm.queue_value_enabled):
            return True
        value_cents = self._mm_queue_quote_value_cents(
            side=side,
            price_cents=price_cents,
            ask_cents=ask_cents,
            micro=micro,
        )
        threshold = max(0.0, float(self.cfg.mm.queue_value_new_min_cents))
        if value_cents + 1e-9 >= threshold:
            return True
        self._logger.info(
            "mm_queue_skip_low_value ticker=%s side=%s value=%.3f threshold=%.3f price=%s ask=%s",
            self.current_ticker,
            side,
            value_cents,
            threshold,
            price_cents,
            ask_cents,
        )
        return False

    def _mm_queue_should_requote(
        self,
        *,
        side: str,
        prev_price_cents: Optional[int],
        new_price_cents: int,
        ask_cents: int,
        micro: Dict[str, Any],
    ) -> bool:
        if not bool(self.cfg.mm.queue_value_enabled):
            return True
        if prev_price_cents is None:
            return True
        if int(prev_price_cents) == int(new_price_cents):
            return True
        prev_value = self._mm_queue_quote_value_cents(
            side=side,
            price_cents=int(prev_price_cents),
            ask_cents=ask_cents,
            micro=micro,
        )
        new_value = self._mm_queue_quote_value_cents(
            side=side,
            price_cents=int(new_price_cents),
            ask_cents=ask_cents,
            micro=micro,
        )
        hold_min = max(0.0, float(self.cfg.mm.queue_value_hold_min_cents))
        min_improve = max(0.0, float(self.cfg.mm.queue_value_min_improve_cents))
        improve = new_value - prev_value
        if prev_value + 1e-9 >= hold_min and improve + 1e-9 < min_improve:
            self._logger.info(
                "mm_queue_hold ticker=%s side=%s prev=%s new=%s prev_value=%.3f new_value=%.3f improve=%.3f min_improve=%.3f",
                self.current_ticker,
                side,
                int(prev_price_cents),
                int(new_price_cents),
                prev_value,
                new_value,
                improve,
                min_improve,
            )
            return False
        return True

    def _mm_upsert_buy_order(
        self,
        *,
        side: str,
        count: int,
        price_cents: int,
        ask_cents: int,
        micro: Dict[str, Any],
    ) -> None:
        if side not in {"yes", "no"}:
            return
        if count <= 0 or price_cents >= ask_cents:
            self._mm_cancel_slot(kind="buy", side=side)
            return
        now = time.time()
        if now - self._mm_last_cross_epoch[side] < max(0.1, float(self.cfg.mm.cross_cooldown_secs)):
            return
        prev_px = self._mm_last_px["buy"][side]
        if self._mm_order_ids["buy"][side] and not self._mm_should_requote(prev_px, price_cents):
            return
        if self._mm_order_ids["buy"][side] and not self._mm_queue_should_requote(
            side=side,
            prev_price_cents=prev_px,
            new_price_cents=price_cents,
            ask_cents=ask_cents,
            micro=micro,
        ):
            return
        if not self._mm_queue_allows_new_order(
            side=side,
            count=count,
            price_cents=price_cents,
            ask_cents=ask_cents,
            micro=micro,
        ):
            return
        if self._mm_order_ids["buy"][side]:
            self._mm_cancel_slot(kind="buy", side=side)
        if not self._mm_order_rate_available():
            return

        log_prefix = (
            f"mm_quote ticker={self.current_ticker} side={side} action=buy count={count} "
            f"price={price_cents} post_only=1"
        )
        if self.cfg.execution.dry_run:
            self._orders_last_minute.append(now)
            self._mm_last_px["buy"][side] = int(price_cents)
            self._logger.info("%s dry_run=1", log_prefix)
            return

        client_order_id = f"MM-{self.current_ticker}-B-{side[0].upper()}-{uuid.uuid4().hex[:10]}"
        self._order_attempts += 1
        try:
            response = self._mm_place_order_raw(
                side=side,
                action="buy",
                count=int(count),
                price_cents=int(price_cents),
                post_only=True,
                time_in_force=self.cfg.execution.time_in_force,
                reduce_only=False,
                client_order_id=client_order_id,
            )
            self._orders_last_minute.append(now)
            self._mm_order_ids["buy"][side] = self._mm_read_order_id(response)
            self._mm_last_px["buy"][side] = int(price_cents)
            self._logger.info("%s order_response=%s", log_prefix, compact_json(response))
        except KalshiRequestError as exc:
            self._order_failures += 1
            err_lower = str(exc).lower()
            if "post" in err_lower and "cross" in err_lower:
                self._mm_last_cross_epoch[side] = time.time()
            self._mm_order_ids["buy"][side] = None
            self._mm_last_px["buy"][side] = None
            self._log_kalshi_error(
                "mm_buy_order_error",
                exc,
                ticker=self.current_ticker,
                side=side,
                price=price_cents,
                count=count,
            )

    def _mm_upsert_resting_sell(self, *, side: str, count: int, price_cents: int, ask_cents: int) -> None:
        if side not in {"yes", "no"}:
            return
        if count <= 0 or price_cents >= ask_cents:
            self._mm_cancel_slot(kind="sell", side=side)
            return
        prev_px = self._mm_last_px["sell"][side]
        if self._mm_order_ids["sell"][side] and not self._mm_should_requote(prev_px, price_cents):
            return
        if self._mm_order_ids["sell"][side]:
            self._mm_cancel_slot(kind="sell", side=side)
        if not self._mm_order_rate_available():
            return

        now = time.time()
        log_prefix = (
            f"mm_tp_resting ticker={self.current_ticker} side={side} action=sell count={count} "
            f"price={price_cents} post_only=1 reduce_only=1"
        )
        if self.cfg.execution.dry_run:
            self._orders_last_minute.append(now)
            self._mm_last_px["sell"][side] = int(price_cents)
            self._logger.info("%s dry_run=1", log_prefix)
            return

        client_order_id = f"MM-{self.current_ticker}-S-{side[0].upper()}-{uuid.uuid4().hex[:10]}"
        self._order_attempts += 1
        try:
            response = self._mm_place_order_raw(
                side=side,
                action="sell",
                count=int(count),
                price_cents=int(price_cents),
                post_only=True,
                time_in_force=self.cfg.execution.time_in_force,
                reduce_only=True,
                client_order_id=client_order_id,
            )
            self._orders_last_minute.append(now)
            self._mm_order_ids["sell"][side] = self._mm_read_order_id(response)
            self._mm_last_px["sell"][side] = int(price_cents)
            self._logger.info("%s order_response=%s", log_prefix, compact_json(response))
        except KalshiRequestError as exc:
            self._order_failures += 1
            self._mm_order_ids["sell"][side] = None
            self._mm_last_px["sell"][side] = None
            self._log_kalshi_error(
                "mm_tp_resting_error",
                exc,
                ticker=self.current_ticker,
                side=side,
                price=price_cents,
                count=count,
            )

    def _mm_ioc_sell_reduce_only(self, *, side: str, count: int, price_cents: int) -> bool:
        if side not in {"yes", "no"} or count <= 0:
            return False
        if not self._mm_order_rate_available():
            return False
        now = time.time()
        log_prefix = (
            f"mm_tp_ioc ticker={self.current_ticker} side={side} action=sell count={count} "
            f"price={price_cents} reduce_only=1"
        )
        if self.cfg.execution.dry_run:
            self._orders_last_minute.append(now)
            self._logger.info("%s dry_run=1", log_prefix)
            return True

        client_order_id = f"MM-{self.current_ticker}-IOC-{side[0].upper()}-{uuid.uuid4().hex[:10]}"
        self._order_attempts += 1
        try:
            response = self._mm_place_order_raw(
                side=side,
                action="sell",
                count=int(count),
                price_cents=int(price_cents),
                post_only=False,
                time_in_force=self.cfg.execution.ioc_time_in_force,
                reduce_only=True,
                client_order_id=client_order_id,
            )
            self._orders_last_minute.append(now)
            self._logger.info("%s order_response=%s", log_prefix, compact_json(response))
            return True
        except KalshiRequestError as exc:
            self._order_failures += 1
            self._log_kalshi_error(
                "mm_tp_ioc_error",
                exc,
                ticker=self.current_ticker,
                side=side,
                price=price_cents,
                count=count,
            )
            return False

    def _mm_apply_straddle_cap(
        self,
        *,
        yes_buy_size: int,
        no_buy_size: int,
        buy_yes_px: int,
        buy_no_px: int,
        yes_pos: int,
        no_pos: int,
    ) -> tuple[int, int]:
        if yes_buy_size <= 0 or no_buy_size <= 0:
            return yes_buy_size, no_buy_size
        if (buy_yes_px + buy_no_px) <= int(self.cfg.mm.max_sum_buy_cents):
            return yes_buy_size, no_buy_size
        if yes_pos > no_pos:
            return 0, no_buy_size
        if no_pos > yes_pos:
            return yes_buy_size, 0
        if buy_yes_px <= buy_no_px:
            return yes_buy_size, 0
        return 0, no_buy_size

    def _mm_apply_size_mult(self, *, base_size: int, mult: float) -> int:
        size = max(0, int(base_size))
        scale = clamp(float(mult), 0.0, 1.0)
        return int(math.floor(size * scale))

    def _mm_buy_sizes(
        self,
        *,
        yes_pos: int,
        no_pos: int,
        buy_yes_px: int,
        buy_no_px: int,
        seconds_to_close: Optional[float],
        yes_size_mult: float = 1.0,
        no_size_mult: float = 1.0,
    ) -> tuple[int, int]:
        base_size = max(0, int(self.cfg.mm.base_size))
        inv_scale = max(1, int(self.cfg.mm.inv_scale))
        yes_buy_size = max(0, base_size - int(math.floor(yes_pos / inv_scale)))
        no_buy_size = max(0, base_size - int(math.floor(no_pos / inv_scale)))

        max_side = max(0, int(self.cfg.mm.max_pos_per_side))
        yes_buy_size = min(yes_buy_size, max(0, max_side - yes_pos))
        no_buy_size = min(no_buy_size, max(0, max_side - no_pos))

        max_gross = min(int(self.cfg.mm.max_gross_pos), int(self.cfg.risk.max_gross_position))
        max_net = min(int(self.cfg.mm.max_net_exposure), int(self.cfg.risk.max_net_exposure))
        gross = yes_pos + no_pos
        net = yes_pos - no_pos
        if abs(net) > max_net:
            if net > 0:
                yes_buy_size = 0
            elif net < 0:
                no_buy_size = 0
        if gross > max_gross:
            yes_buy_size = 0
            no_buy_size = 0

        if buy_yes_px >= int(self.cfg.mm.skip_if_buy_ge_cents):
            yes_buy_size = 0
        if buy_no_px >= int(self.cfg.mm.skip_if_buy_ge_cents):
            no_buy_size = 0
        if buy_yes_px <= int(self.cfg.mm.skip_if_buy_le_cents):
            yes_buy_size = 0
        if buy_no_px <= int(self.cfg.mm.skip_if_buy_le_cents):
            no_buy_size = 0

        if seconds_to_close is not None and seconds_to_close <= float(self.cfg.mm.reduce_risk_secs):
            yes_buy_size = 0
            no_buy_size = 0

        yes_buy_size = self._mm_apply_size_mult(base_size=yes_buy_size, mult=yes_size_mult)
        no_buy_size = self._mm_apply_size_mult(base_size=no_buy_size, mult=no_size_mult)

        return self._mm_apply_straddle_cap(
            yes_buy_size=yes_buy_size,
            no_buy_size=no_buy_size,
            buy_yes_px=buy_yes_px,
            buy_no_px=buy_no_px,
            yes_pos=yes_pos,
            no_pos=no_pos,
        )

    def _mm_mid_move_cents(self, now: float) -> float:
        window = max(0.5, float(self.cfg.mm.toxic_mid_move_window_secs))
        while self._mm_mid_history and self._mm_mid_history[0][0] < now - (window + 1.0):
            self._mm_mid_history.popleft()
        if len(self._mm_mid_history) < 2:
            return 0.0
        latest_mid = self._mm_mid_history[-1][1]
        baseline = self._mm_mid_history[0][1]
        for ts, mid in self._mm_mid_history:
            if ts >= now - window:
                baseline = mid
                break
        return abs(latest_mid - baseline)

    def _mm_toxic_pause_active(self, now: float) -> bool:
        return bool(self.cfg.mm.toxicity_enabled) and now < self._mm_toxic_pause_until_epoch

    def _mm_update_toxicity_pause(
        self,
        *,
        micro: Dict[str, Any],
        yb: int,
        ya: int,
        yes_spread: int,
        no_spread: int,
        now: float,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "regime": "normal",
            "yes_edge_add": 0,
            "no_edge_add": 0,
            "yes_size_mult": 1.0,
            "no_size_mult": 1.0,
            "paused": False,
            "imbalance": 0.0,
            "depth_ratio": 1.0,
            "mid_move": 0.0,
        }
        if not bool(self.cfg.mm.toxicity_enabled):
            self._mm_last_toxic_regime = "normal"
            return state
        mid_yes = (float(yb) + float(ya)) / 2.0
        self._mm_mid_history.append((now, mid_yes))
        while (
            self._mm_mid_history
            and self._mm_mid_history[0][0]
            < now - (max(0.5, float(self.cfg.mm.toxic_mid_move_window_secs)) + 1.0)
        ):
            self._mm_mid_history.popleft()

        signed_imbalance = float(extract_float(micro.get("imbalance")) or 0.0)
        imbalance = abs(signed_imbalance)
        yes_depth = max(1.0, float(extract_float(micro.get("yes_depth")) or 0.0))
        no_depth = max(1.0, float(extract_float(micro.get("no_depth")) or 0.0))
        depth_ratio = max(yes_depth, no_depth) / max(1.0, min(yes_depth, no_depth))
        mid_move = self._mm_mid_move_cents(now)
        state["imbalance"] = imbalance
        state["depth_ratio"] = depth_ratio
        state["mid_move"] = mid_move

        if bool(self.cfg.mm.toxicity_regime_enabled):
            widen_trigger = (
                imbalance >= float(self.cfg.mm.toxic_widen_imbalance_abs)
                or depth_ratio >= float(self.cfg.mm.toxic_widen_depth_ratio)
            )
            if widen_trigger:
                widen_edge = max(0, int(self.cfg.mm.toxic_widen_edge_cents))
                size_mult = clamp(float(self.cfg.mm.toxic_widen_size_mult), 0.0, 1.0)
                state["regime"] = "widen"
                state["yes_edge_add"] = int(widen_edge)
                state["no_edge_add"] = int(widen_edge)
                state["yes_size_mult"] = float(size_mult)
                state["no_size_mult"] = float(size_mult)

                if imbalance >= float(self.cfg.mm.toxic_skew_imbalance_abs):
                    opp_mult = clamp(float(self.cfg.mm.toxic_skew_opp_size_mult), 0.0, 1.0)
                    state["regime"] = "skew"
                    if signed_imbalance > 0:
                        state["no_size_mult"] = min(float(state["no_size_mult"]), float(opp_mult))
                        state["no_edge_add"] = max(int(state["no_edge_add"]), int(widen_edge) + 1)
                    elif signed_imbalance < 0:
                        state["yes_size_mult"] = min(float(state["yes_size_mult"]), float(opp_mult))
                        state["yes_edge_add"] = max(int(state["yes_edge_add"]), int(widen_edge) + 1)

        trigger_imbalance = imbalance >= float(self.cfg.mm.toxic_imbalance_abs)
        trigger_depth = depth_ratio >= float(self.cfg.mm.toxic_depth_ratio)
        trigger_move = mid_move >= float(self.cfg.mm.toxic_mid_move_cents)
        thin_spread = (
            yes_spread <= int(self.cfg.mm.min_spread_cents) + 1
            or no_spread <= int(self.cfg.mm.min_spread_cents) + 1
        )

        should_pause = trigger_move or (trigger_imbalance and trigger_depth and thin_spread)
        if should_pause:
            pause_secs = max(0.5, float(self.cfg.mm.toxic_pause_secs))
            self._mm_toxic_pause_until_epoch = max(self._mm_toxic_pause_until_epoch, now + pause_secs)
            state["paused"] = True
            state["regime"] = "pause"
            if now - self._mm_last_toxic_log_epoch >= 1.0:
                self._mm_last_toxic_log_epoch = now
                self._logger.warning(
                    "mm_toxic_pause ticker=%s pause=%.1fs imbalance=%.3f depth_ratio=%.2f mid_move=%.2f "
                    "spreads=%s/%s triggers=move:%s imbalance:%s depth:%s",
                    self.current_ticker,
                    pause_secs,
                    imbalance,
                    depth_ratio,
                    mid_move,
                    yes_spread,
                    no_spread,
                    int(trigger_move),
                    int(trigger_imbalance),
                    int(trigger_depth),
                )
        if state["regime"] != self._mm_last_toxic_regime:
            self._mm_last_toxic_regime = str(state["regime"])
            self._logger.info(
                "mm_toxic_regime ticker=%s regime=%s imbalance=%.3f depth_ratio=%.2f mid_move=%.2f",
                self.current_ticker,
                state["regime"],
                imbalance,
                depth_ratio,
                mid_move,
            )
        return state

    def _run_mm_cycle(self, *, micro: Dict[str, Any], seconds_to_close: Optional[float]) -> None:
        if not self.current_ticker:
            return
        yb, ya, nb, na = self._mm_topbook(micro)
        if yb is None or ya is None or nb is None or na is None:
            self._mm_cancel_tracked_orders(cancel_buy=True, cancel_sell=False)
            return
        if self.cfg.mm.skip_pinned_markets and self._mm_book_is_pinned_or_garbage(
            yb=yb, ya=ya, nb=nb, na=na, micro=micro
        ):
            self._mm_cancel_tracked_orders(cancel_buy=True, cancel_sell=True)
            self._logger.info(
                "mm_skip_pinned ticker=%s yes=%s/%s no=%s/%s",
                self.current_ticker,
                yb,
                ya,
                nb,
                na,
            )
            return
        yes_spread = int(ya - yb)
        no_spread = int(na - nb)
        if yes_spread > int(self.cfg.mm.max_spread_cents) or no_spread > int(self.cfg.mm.max_spread_cents):
            self._mm_cancel_tracked_orders(cancel_buy=True, cancel_sell=False)
            self._logger.info(
                "mm_skip_spread_wide ticker=%s yes=%s/%s no=%s/%s max_spread=%s",
                self.current_ticker,
                yb,
                ya,
                nb,
                na,
                int(self.cfg.mm.max_spread_cents),
            )
            return
        if (
            seconds_to_close is not None
            and seconds_to_close <= float(self.cfg.risk.no_quote_before_close_seconds)
        ):
            self._mm_cancel_tracked_orders(cancel_buy=True, cancel_sell=True)
            return

        self._mm_reconcile_order_ids()

        exposure = self._effective_position_for_ticker(self.current_ticker)
        yes_pos = int(exposure.yes)
        no_pos = int(exposure.no)

        fills = self._mm_fetch_fills_current_ticker()
        avg = self._mm_avg_cost_from_fills(fills)
        yes_avg = avg.get("yes")
        no_avg = avg.get("no")

        now = time.time()
        toxic_state = self._mm_update_toxicity_pause(
            micro=micro,
            yb=yb,
            ya=ya,
            yes_spread=yes_spread,
            no_spread=no_spread,
            now=now,
        )
        if now - self._mm_last_tp_epoch >= max(0.1, float(self.cfg.mm.tp_throttle_secs)):
            for side in ("yes", "no"):
                pos = yes_pos if side == "yes" else no_pos
                avg_cost = yes_avg if side == "yes" else no_avg
                bid = yb if side == "yes" else nb
                ask = ya if side == "yes" else na
                if pos > 0 and avg_cost is not None:
                    floor = int(
                        clamp(
                            int(avg_cost)
                            + int(self.cfg.mm.tp_min_profit_over_cost_cents)
                            + int(self.cfg.mm.fee_buffer_cents),
                            1,
                            99,
                        )
                    )
                    qty = min(int(self.cfg.mm.tp_clip), int(pos))
                    if bid >= floor:
                        ok = self._mm_ioc_sell_reduce_only(side=side, count=qty, price_cents=int(bid))
                        if ok:
                            self._mm_cancel_slot(kind="sell", side=side)
                    else:
                        px = int(clamp(max(floor, int(bid) + 1), 1, 99))
                        self._mm_upsert_resting_sell(side=side, count=qty, price_cents=px, ask_cents=int(ask))
                else:
                    self._mm_cancel_slot(kind="sell", side=side)
            self._mm_last_tp_epoch = now

        if self._mm_toxic_pause_active(now):
            self._mm_cancel_tracked_orders(cancel_buy=True, cancel_sell=False)
            if now - self._mm_last_toxic_log_epoch >= 1.0:
                self._mm_last_toxic_log_epoch = now
                self._logger.info(
                    "mm_toxic_hold ticker=%s pause_left=%.2fs",
                    self.current_ticker,
                    self._mm_toxic_pause_until_epoch - now,
                )
            return

        if yes_spread < int(self.cfg.mm.min_spread_cents) or no_spread < int(self.cfg.mm.min_spread_cents):
            self._mm_cancel_tracked_orders(cancel_buy=True, cancel_sell=False)
            self._logger.info(
                "mm_skip_spread_narrow ticker=%s yes=%s/%s no=%s/%s min_spread=%s",
                self.current_ticker,
                yb,
                ya,
                nb,
                na,
                int(self.cfg.mm.min_spread_cents),
            )
            return

        if now - self._mm_last_quote_epoch < max(0.1, float(self.cfg.mm.quote_refresh_secs)):
            return

        buy_yes_px, buy_no_px = self._mm_compute_quotes_from_mid(
            yb=yb,
            ya=ya,
            nb=nb,
            na=na,
            yes_edge_add=int(toxic_state.get("yes_edge_add") or 0),
            no_edge_add=int(toxic_state.get("no_edge_add") or 0),
        )
        yes_buy_size, no_buy_size = self._mm_buy_sizes(
            yes_pos=yes_pos,
            no_pos=no_pos,
            buy_yes_px=buy_yes_px,
            buy_no_px=buy_no_px,
            seconds_to_close=seconds_to_close,
            yes_size_mult=float(
                toxic_state["yes_size_mult"] if toxic_state.get("yes_size_mult") is not None else 1.0
            ),
            no_size_mult=float(
                toxic_state["no_size_mult"] if toxic_state.get("no_size_mult") is not None else 1.0
            ),
        )

        min_edge = max(0, int(self.cfg.mm.min_edge_after_fee_cents))
        if min_edge > 0:
            if int(ya) - int(buy_yes_px) < min_edge:
                yes_buy_size = 0
            if int(na) - int(buy_no_px) < min_edge:
                no_buy_size = 0

        self._mm_upsert_buy_order(
            side="yes",
            count=yes_buy_size,
            price_cents=buy_yes_px,
            ask_cents=ya,
            micro=micro,
        )
        self._mm_upsert_buy_order(
            side="no",
            count=no_buy_size,
            price_cents=buy_no_px,
            ask_cents=na,
            micro=micro,
        )
        self._mm_last_quote_epoch = time.time()

    def _safe_poll_spot(self) -> Optional[float]:
        try:
            return self.spot_monitor.poll_once()
        except Exception as exc:
            self._logger.warning("binance_poll_error=%s", exc)
            return None

    def _switch_to_current_market(self, force: bool) -> None:
        now = time.time()
        if not force and (now - self._last_market_resolve < 3.0):
            return
        self._last_market_resolve = now

        selection_source = "active"
        try:
            if self.cfg.strategy.market_ticker:
                target_market = unwrap_market_payload(self.client.get_market(self.cfg.strategy.market_ticker))
                selection_source = "explicit_ticker"
            else:
                target_market: Optional[Dict[str, Any]] = None
                if self._is_mm_mode() and bool(self.cfg.mm.market_selection_enabled):
                    refresh_secs = max(2.0, float(self.cfg.mm.market_select_refresh_secs))
                    if (
                        not force
                        and self.current_market
                        and (now - self._mm_last_market_scan_epoch) < refresh_secs
                    ):
                        target_market = dict(self.current_market)
                        selection_source = "cached"
                    else:
                        self._mm_last_market_scan_epoch = now
                        try:
                            target_market = unwrap_market_payload(
                                self.client.resolve_quoteable_mm_market(
                                    series_ticker=self.cfg.strategy.series_ticker,
                                    min_seconds_to_close=float(self.cfg.risk.min_seconds_to_close),
                                    min_spread_cents=int(self.cfg.mm.min_spread_cents),
                                    max_spread_cents=int(self.cfg.mm.max_spread_cents),
                                    target_mid_cents=float(self.cfg.mm.market_target_mid_cents),
                                    max_mid_distance_cents=float(self.cfg.mm.market_max_mid_distance_cents),
                                )
                            )
                            selection_source = "quoteable"
                        except KalshiRequestError as exc:
                            if "no_quoteable_mm_markets_for_series" in str(exc):
                                selection_source = "active_fallback"
                            else:
                                raise
                if target_market is None:
                    target_market = unwrap_market_payload(
                        self.client.resolve_active_market(self.cfg.strategy.series_ticker)
                    )
        except KalshiRequestError as exc:
            if "no_active_markets_for_series" in str(exc) and not self.cfg.strategy.market_ticker:
                if now - self._last_no_market_log >= 5.0:
                    self._last_no_market_log = now
                    self._logger.warning(
                        "market_roll_gap series=%s err=%s",
                        self.cfg.strategy.series_ticker,
                        exc,
                    )
                if self.current_ticker:
                    self.current_ticker = ""
                    self.current_market = {}
                    self._mm_reset_state()
                    if self.book_monitor:
                        self.book_monitor.stop()
                        self.book_monitor = None
                return
            raise
        target_ticker = str(target_market.get("ticker") or "").strip()
        if not target_ticker:
            raise KalshiRequestError("unable_to_resolve_market_ticker")

        seconds_to_close = self._seconds_to_close(target_market)
        if (
            seconds_to_close is not None
            and seconds_to_close <= self.cfg.risk.min_seconds_to_close
            and not self.cfg.strategy.market_ticker
        ):
            try:
                if self._is_mm_mode() and bool(self.cfg.mm.market_selection_enabled):
                    target_market = unwrap_market_payload(
                        self.client.resolve_quoteable_mm_market(
                            series_ticker=self.cfg.strategy.series_ticker,
                            min_seconds_to_close=float(self.cfg.risk.min_seconds_to_close),
                            min_spread_cents=int(self.cfg.mm.min_spread_cents),
                            max_spread_cents=int(self.cfg.mm.max_spread_cents),
                            target_mid_cents=float(self.cfg.mm.market_target_mid_cents),
                            max_mid_distance_cents=float(self.cfg.mm.market_max_mid_distance_cents),
                        )
                    )
                    selection_source = "quoteable"
                else:
                    target_market = unwrap_market_payload(
                        self.client.resolve_active_market(self.cfg.strategy.series_ticker)
                    )
                    selection_source = "active"
            except KalshiRequestError:
                target_market = unwrap_market_payload(
                    self.client.resolve_active_market(self.cfg.strategy.series_ticker)
                )
                selection_source = "active_fallback"
            target_ticker = str(target_market.get("ticker") or "").strip()

        if target_ticker != self.current_ticker:
            self._logger.info(
                "market_roll old=%s new=%s source=%s",
                self.current_ticker or "NONE",
                target_ticker,
                selection_source,
            )
            if self._is_mm_mode() and self.current_ticker:
                self._mm_cancel_tracked_orders(cancel_buy=True, cancel_sell=True)
            self._mm_reset_state()
            self.current_market = target_market
            self.current_ticker = target_ticker
            self._reset_orderbook_monitor(target_ticker)

    def _reset_orderbook_monitor(self, ticker: str) -> None:
        if self.book_monitor:
            self.book_monitor.stop()
        self.book_monitor = KalshiOrderbookMonitor(client=self.client, ticker=ticker)
        self.book_monitor.refresh_from_rest()
        self.book_monitor.start()
        self._last_market_fetch = 0.0

    def _refresh_market_metadata_if_needed(self) -> None:
        now = time.time()
        if not self.current_ticker:
            return
        if now - self._last_market_fetch < 5.0:
            return
        self._last_market_fetch = now
        try:
            self.current_market = unwrap_market_payload(self.client.get_market(self.current_ticker))
        except KalshiRequestError as exc:
            self._log_kalshi_error("market_refresh_error", exc, ticker=self.current_ticker)

    def _get_microstructure(self, *, allow_rest_fallback: bool = True) -> Dict[str, Any]:
        if self.book_monitor is None:
            return {
                "best_yes_bid": None,
                "best_yes_ask": None,
                "best_no_bid": None,
                "best_no_ask": None,
                "best_yes_bid_qty": 0,
                "best_no_bid_qty": 0,
                "spread_yes": None,
                "spread_no": None,
                "yes_depth": 0,
                "no_depth": 0,
                "imbalance": 0.0,
                "pinned_state": False,
                "zero_spread_state": False,
                "book_age_seconds": float("inf"),
                "book_stale_seconds": float("inf"),
            }
        micro = self.book_monitor.microstructure(levels=self.cfg.strategy.depth_levels)
        if allow_rest_fallback and micro["best_yes_bid"] is None and micro["best_no_bid"] is None:
            try:
                self.book_monitor.refresh_from_rest()
                micro = self.book_monitor.microstructure(levels=self.cfg.strategy.depth_levels)
            except Exception as exc:
                self._logger.warning("orderbook_refresh_error ticker=%s err=%s", self.current_ticker, exc)
        return micro

    def _is_data_stale(self, micro: Dict[str, Any], *, require_reference: bool = True) -> bool:
        if require_reference and self.spot_monitor.age_seconds() > self.cfg.strategy.reference_stale_seconds:
            return True
        if float(micro.get("book_age_seconds", float("inf"))) > self.cfg.strategy.book_stale_seconds:
            return True
        if float(micro.get("book_stale_seconds", float("inf"))) > self.cfg.strategy.book_unchanged_seconds:
            return True
        return False

    def _handle_data_stale_guard(self, micro: Dict[str, Any]) -> None:
        now = time.time()
        if now - self._last_guard_cancel_epoch < 4.0:
            return
        self._last_guard_cancel_epoch = now
        self._logger.warning(
            "data_stale_guard ref_age=%.3f book_age=%.3f book_stale=%.3f ticker=%s",
            self.spot_monitor.age_seconds(),
            float(micro.get("book_age_seconds", float("inf"))),
            float(micro.get("book_stale_seconds", float("inf"))),
            self.current_ticker,
        )
        if not self.cfg.execution.dry_run and self.current_ticker:
            try:
                self.client.cancel_all_open_orders(ticker=self.current_ticker)
            except KalshiRequestError as exc:
                self._log_kalshi_error("stale_guard_cancel_error", exc, ticker=self.current_ticker)
        if (
            self._is_mm_mode()
            and self.current_ticker
            and now - self._mm_last_stale_reconnect_epoch >= 15.0
        ):
            self._mm_last_stale_reconnect_epoch = now
            self._logger.warning("mm_stale_reconnect ticker=%s", self.current_ticker)
            self._reset_orderbook_monitor(self.current_ticker)

    def _compute_implied_yes(self, micro: Dict[str, Any]) -> float:
        yes_bid = micro.get("best_yes_bid")
        yes_ask = micro.get("best_yes_ask")
        if yes_bid is not None and yes_ask is not None:
            return clamp(((yes_bid + yes_ask) / 2.0) / 100.0, 0.01, 0.99)
        if yes_bid is not None:
            return clamp(yes_bid / 100.0, 0.01, 0.99)
        if yes_ask is not None:
            return clamp(yes_ask / 100.0, 0.01, 0.99)
        return 0.50

    def _extract_strike_price(self, market: Dict[str, Any], fallback_spot: float) -> float:
        direct_candidates = [
            market.get("strike_price"),
            market.get("strike"),
            market.get("target"),
            market.get("target_price"),
        ]
        for value in direct_candidates:
            strike = extract_float(value)
            if strike is not None and strike > 0:
                return strike

        for key in ["subtitle", "yes_sub_title", "title", "rules_primary", "rules"]:
            raw = market.get(key)
            if not raw:
                continue
            text = str(raw).replace(",", "")
            numbers = re.findall(r"(\d+(?:\.\d+)?)", text)
            if not numbers:
                continue
            candidates = [float(n) for n in numbers if float(n) >= 10.0]
            if not candidates:
                continue
            return min(candidates, key=lambda n: abs(n - fallback_spot))

        ticker = str(market.get("ticker") or "")
        ticker_nums = re.findall(r"(\d+(?:\.\d+)?)", ticker)
        if ticker_nums:
            candidates = [float(n) for n in ticker_nums if float(n) >= 10.0]
            if candidates:
                return min(candidates, key=lambda n: abs(n - fallback_spot))
        return fallback_spot

    def _infer_direction(self, market: Dict[str, Any]) -> str:
        text = " ".join(
            [
                str(market.get("title") or ""),
                str(market.get("subtitle") or ""),
                str(market.get("yes_sub_title") or ""),
                str(market.get("rules_primary") or ""),
                str(market.get("ticker") or ""),
            ]
        ).lower()
        down_words = ["below", "under", "down", "decrease", "lower", "at or below"]
        up_words = ["above", "over", "up", "increase", "higher", "at or above"]
        if any(word in text for word in down_words):
            return "down"
        if any(word in text for word in up_words):
            return "up"
        return "up"

    def _real_world_probability(
        self,
        *,
        spot: float,
        strike: float,
        direction: str,
        momentum_delta: float,
        seconds_to_close: Optional[float],
    ) -> float:
        strike_ref = strike if strike > 0 else spot
        base_dist_scale = max(25.0, abs(strike_ref) * self.cfg.strategy.distance_scale_pct)
        dist_scale = base_dist_scale
        if seconds_to_close is not None:
            ref_seconds = max(60.0, float(self.cfg.strategy.time_to_close_reference_seconds))
            min_mult = max(0.05, float(self.cfg.strategy.min_distance_scale_multiplier))
            max_mult = max(min_mult, float(self.cfg.strategy.max_distance_scale_multiplier))
            close_seconds = max(1.0, float(seconds_to_close))
            dist_scale *= clamp(math.sqrt(close_seconds / ref_seconds), min_mult, max_mult)
        z = (spot - strike_ref) / dist_scale
        if direction == "down":
            z = -z
        distance_prob = sigmoid(z)

        momentum_move = clamp(
            momentum_delta / max(1.0, self.cfg.strategy.momentum_full_scale_dollars),
            -0.5,
            0.5,
        )
        if direction == "down":
            momentum_move = -momentum_move
        dist_w = max(0.0, float(self.cfg.strategy.distance_weight))
        mom_w = max(0.0, float(self.cfg.strategy.momentum_weight))
        if dist_w + mom_w <= 1e-9:
            dist_w = 0.7
            mom_w = 0.3
        wsum = dist_w + mom_w
        momentum_ratio = mom_w / wsum
        pred = distance_prob + momentum_ratio * momentum_move
        return clamp(pred, 0.01, 0.99)

    def _blend_model_with_implied(self, *, model_yes: float, implied_yes: float) -> float:
        m = clamp(float(self.cfg.strategy.model_edge_multiplier), 0.0, 1.0)
        blended = implied_yes + (model_yes - implied_yes) * m
        return clamp(blended, 0.01, 0.99)

    def _classify_regime(self, spot: float) -> Dict[str, Any]:
        vol_short = self.spot_monitor.realized_volatility(self.cfg.strategy.vol_window_short_seconds)
        vol_long = self.spot_monitor.realized_volatility(self.cfg.strategy.vol_window_long_seconds)
        momentum = self.spot_monitor.delta_over_window(self.cfg.strategy.momentum_window_seconds)
        momentum_pct = abs(momentum) / max(1.0, spot)

        name = "calm"
        size_multiplier = 1.0
        edge_boost = 0.0
        price_cushion = 0
        drawdown_multiplier = 1.0

        if vol_short >= self.cfg.strategy.spiky_vol_threshold or momentum_pct >= 2.0 * self.cfg.strategy.trending_momentum_threshold:
            name = "spiky"
            size_multiplier = 0.45
            edge_boost = 0.02
            price_cushion = 2
            drawdown_multiplier = 0.75
        elif momentum_pct >= self.cfg.strategy.trending_momentum_threshold or vol_short >= self.cfg.strategy.calm_vol_threshold * 1.5:
            name = "trending"
            size_multiplier = 0.70
            edge_boost = 0.01
            price_cushion = 1
            drawdown_multiplier = 0.85

        return {
            "name": name,
            "vol_short": vol_short,
            "vol_long": vol_long,
            "momentum": momentum,
            "momentum_pct": momentum_pct,
            "size_multiplier": size_multiplier,
            "edge_boost": edge_boost,
            "price_cushion": price_cushion,
            "drawdown_multiplier": drawdown_multiplier,
        }

    def _cancel_for_volatility_guard(self, *, reason: str) -> None:
        if not self.cfg.strategy.spike_cancel_resting_orders:
            return
        if self.cfg.execution.dry_run:
            return
        if not self.current_ticker:
            return
        now = time.time()
        cooldown = max(0.5, float(self.cfg.strategy.spike_cancel_cooldown_seconds))
        if now - self._last_spike_cancel_epoch < cooldown:
            return
        self._last_spike_cancel_epoch = now
        try:
            result = self.client.cancel_all_open_orders(ticker=self.current_ticker)
            self._logger.warning(
                "volatility_cancel_guard reason=%s ticker=%s result=%s",
                reason,
                self.current_ticker,
                compact_json(result),
            )
        except KalshiRequestError as exc:
            self._logger.warning(
                "volatility_cancel_guard_error reason=%s ticker=%s err=%s",
                reason,
                self.current_ticker,
                exc,
            )

    def _maybe_place_cross_side_arb(
        self,
        *,
        micro: Dict[str, Any],
        seconds_to_close: Optional[float],
    ) -> None:
        if not self.cfg.strategy.cross_side_arb_enabled:
            return
        if not self.current_ticker:
            return
        if seconds_to_close is not None and seconds_to_close <= self.cfg.risk.force_flatten_seconds:
            return

        yes_bid = maybe_int(micro.get("best_yes_bid"))
        yes_ask = maybe_int(micro.get("best_yes_ask"))
        no_bid = maybe_int(micro.get("best_no_bid"))
        no_ask = maybe_int(micro.get("best_no_ask"))
        if yes_bid is None and yes_ask is None and no_bid is None and no_ask is None:
            return

        now = time.time()
        self._prune_order_rate_window(now)
        if len(self._orders_last_minute) >= self.cfg.execution.max_orders_per_minute:
            return
        open_orders = self._current_open_orders()
        if open_orders >= self.cfg.risk.max_open_orders:
            return

        max_pair_cfg = max(1, int(self.cfg.strategy.cross_side_arb_max_count))
        per_leg_cost = float(self.cfg.strategy.taker_fee_cents_per_contract) + float(
            self.cfg.strategy.taker_slippage_cents_per_contract
        )
        min_edge = float(self.cfg.strategy.cross_side_arb_min_edge_cents)

        buy_pair_count = 0
        buy_pair_edge = float("-inf")
        if yes_ask is not None and no_ask is not None and 1 <= yes_ask <= 99 and 1 <= no_ask <= 99:
            gross_room = max(0, int(self.cfg.risk.max_gross_position - self._effective_gross_position_for_limits()))
            max_pair_count_by_gross = gross_room // 2
            buy_pair_count = min(max_pair_count_by_gross, max_pair_cfg)
            if buy_pair_count > 0:
                buy_pair_edge = 100.0 - float(yes_ask + no_ask) - 2.0 * per_leg_cost

        sell_pair_count = 0
        sell_pair_edge = float("-inf")
        if (
            self.cfg.strategy.cross_side_arb_allow_sell_pair
            and yes_bid is not None
            and no_bid is not None
            and 1 <= yes_bid <= 99
            and 1 <= no_bid <= 99
        ):
            exposure = self._positions_by_ticker.get(self.current_ticker) or PositionExposure()
            sell_pair_count = min(max_pair_cfg, int(min(exposure.yes, exposure.no)))
            if sell_pair_count > 0:
                sell_pair_edge = float(yes_bid + no_bid) - 100.0 - 2.0 * per_leg_cost

        arb_mode = ""
        action = "buy"
        pair_count = 0
        yes_price = None
        no_price = None
        est_edge_cents = float("-inf")

        if buy_pair_count > 0 and buy_pair_edge >= min_edge:
            arb_mode = "buy_pair"
            action = "buy"
            pair_count = buy_pair_count
            yes_price = yes_ask
            no_price = no_ask
            est_edge_cents = buy_pair_edge

        if sell_pair_count > 0 and sell_pair_edge >= min_edge and sell_pair_edge > est_edge_cents:
            arb_mode = "sell_pair"
            action = "sell"
            pair_count = sell_pair_count
            yes_price = yes_bid
            no_price = no_bid
            est_edge_cents = sell_pair_edge

        if not arb_mode or pair_count <= 0 or yes_price is None or no_price is None:
            return

        tif = self.cfg.execution.ioc_time_in_force
        log_prefix = (
            f"cross_arb_signal ticker={self.current_ticker} mode={arb_mode} action={action} count={pair_count} "
            f"yes_price={yes_price} no_price={no_price} est_edge_cents={est_edge_cents:.3f}"
        )
        if self.cfg.execution.dry_run:
            self._orders_last_minute.append(now)
            self._logger.info("%s dry_run=1", log_prefix)
            return

        # Execute both legs as IOC limit buys. Partial execution risk remains if one leg fails.
        yes_client_id = uuid.uuid4().hex
        no_client_id = uuid.uuid4().hex
        yes_ok = False

        try:
            self._order_attempts += 1
            yes_response = self.client.place_limit_order(
                ticker=self.current_ticker,
                side="yes",
                count=pair_count,
                price_cents=yes_price,
                action=action,
                post_only=False,
                time_in_force=tif,
                client_order_id=yes_client_id,
            )
            yes_ok = True
            self._orders_last_minute.append(now)
            self._logger.info("%s leg=yes order_response=%s", log_prefix, compact_json(yes_response))
        except KalshiRequestError as exc:
            self._order_failures += 1
            self._log_kalshi_error(
                "cross_arb_order_error",
                exc,
                ticker=self.current_ticker,
                leg="yes",
                tif=tif,
            )
            return

        try:
            self._order_attempts += 1
            no_response = self.client.place_limit_order(
                ticker=self.current_ticker,
                side="no",
                count=pair_count,
                price_cents=no_price,
                action=action,
                post_only=False,
                time_in_force=tif,
                client_order_id=no_client_id,
            )
            self._orders_last_minute.append(time.time())
            self._logger.info("%s leg=no order_response=%s", log_prefix, compact_json(no_response))
        except KalshiRequestError as exc:
            self._order_failures += 1
            self._log_kalshi_error(
                "cross_arb_order_error",
                exc,
                ticker=self.current_ticker,
                leg="no",
                tif=tif,
                partial_yes=yes_ok,
            )

    def _maybe_reduce_inventory(
        self,
        *,
        real_yes: float,
        micro: Dict[str, Any],
        seconds_to_close: Optional[float],
    ) -> None:
        if not self.cfg.strategy.inventory_exit_enabled:
            return
        if not self.current_ticker:
            return
        if seconds_to_close is not None and seconds_to_close <= self.cfg.risk.force_flatten_seconds:
            return

        exposure = self._positions_by_ticker.get(self.current_ticker) or PositionExposure()
        if exposure.gross <= 0:
            return

        over_ratio = clamp(float(self.cfg.strategy.inventory_reduce_over_limit_ratio), 0.1, 1.0)
        over_limit = (
            self._effective_gross_position_for_limits() >= int(self.cfg.risk.max_gross_position * over_ratio)
            or abs(self._effective_net_exposure_for_limits()) >= int(self.cfg.risk.max_net_exposure * over_ratio)
        )
        min_exit_edge = float(self.cfg.strategy.inventory_exit_min_edge_cents)
        if over_limit:
            min_exit_edge = min(min_exit_edge, -1.0)

        now = time.time()
        self._prune_order_rate_window(now)
        if len(self._orders_last_minute) >= self.cfg.execution.max_orders_per_minute:
            return

        per_leg_cost = float(self.cfg.strategy.taker_fee_cents_per_contract) + float(
            self.cfg.strategy.taker_slippage_cents_per_contract
        )
        max_count = max(1, int(self.cfg.strategy.inventory_exit_max_count))
        tif = self.cfg.execution.ioc_time_in_force

        def maybe_exit(side: str, qty: int, bid_price: Optional[int], fair_hold_cents: float) -> None:
            if qty <= 0 or bid_price is None or bid_price < 1 or bid_price > 99:
                return
            exit_edge = float(bid_price) - per_leg_cost - fair_hold_cents
            if exit_edge < min_exit_edge:
                return
            count = min(qty, max_count)
            log_prefix = (
                f"inventory_exit ticker={self.current_ticker} side={side} count={count} "
                f"bid={bid_price} exit_edge_cents={exit_edge:.3f} fair_hold_cents={fair_hold_cents:.3f}"
            )
            if self.cfg.execution.dry_run:
                self._orders_last_minute.append(time.time())
                self._logger.info("%s dry_run=1", log_prefix)
                return
            client_order_id = uuid.uuid4().hex
            try:
                self._order_attempts += 1
                response = self.client.place_limit_order(
                    ticker=self.current_ticker,
                    side=side,
                    count=count,
                    price_cents=int(bid_price),
                    action="sell",
                    post_only=False,
                    time_in_force=tif,
                    client_order_id=client_order_id,
                )
                self._orders_last_minute.append(time.time())
                self._logger.info("%s order_response=%s", log_prefix, compact_json(response))
            except KalshiRequestError as exc:
                self._order_failures += 1
                self._log_kalshi_error(
                    "inventory_exit_order_error",
                    exc,
                    ticker=self.current_ticker,
                    side=side,
                    tif=tif,
                )

        maybe_exit(
            side="yes",
            qty=int(exposure.yes),
            bid_price=maybe_int(micro.get("best_yes_bid")),
            fair_hold_cents=float(real_yes) * 100.0,
        )
        maybe_exit(
            side="no",
            qty=int(exposure.no),
            bid_price=maybe_int(micro.get("best_no_bid")),
            fair_hold_cents=(1.0 - float(real_yes)) * 100.0,
        )

    def _maybe_place_limit_order(
        self,
        *,
        edge: float,
        real_yes: float,
        implied_yes: float,
        micro: Dict[str, Any],
        regime: Dict[str, Any],
        seconds_to_close: Optional[float],
    ) -> None:
        if not self.current_ticker:
            return

        now = time.time()
        self._prune_order_rate_window(now)
        if len(self._orders_last_minute) >= self.cfg.execution.max_orders_per_minute:
            return
        if self._current_open_orders() >= self.cfg.risk.max_open_orders:
            return

        side = self._select_trade_side(edge=edge, micro=micro)
        if self._side_blocked_by_inventory(side):
            self._logger.info("inventory_block side=%s net=%s gross=%s", side, self._net_exposure, self._gross_position)
            return

        is_garbage_state = bool(micro.get("pinned_state") or micro.get("zero_spread_state"))
        mode = "mm"
        if (
            not self.cfg.strategy.maker_only
            and not is_garbage_state
            and abs(edge) >= self.cfg.strategy.snipe_edge_threshold
        ):
            mode = "snipe"

        if not self._spread_gate_ok(mode=mode, micro=micro):
            return

        price_cents = self._derive_limit_price(
            side=side,
            mode=mode,
            real_yes=real_yes,
            micro=micro,
            regime_price_cushion=int(regime["price_cushion"]),
        )
        if price_cents is None:
            return

        ev_cents = self._expected_value_cents_per_contract(
            side=side,
            mode=mode,
            real_yes=real_yes,
            price_cents=price_cents,
        )
        if ev_cents < self.cfg.strategy.min_ev_cents_per_contract:
            self._logger.info(
                "ev_reject ticker=%s side=%s mode=%s ev=%.3f min_ev=%.3f",
                self.current_ticker,
                side,
                mode,
                ev_cents,
                self.cfg.strategy.min_ev_cents_per_contract,
            )
            return

        count = self._sized_order_count(abs(edge), float(regime["size_multiplier"]))
        if self.cfg.strategy.use_fractional_kelly:
            count = min(
                count,
                self._kelly_order_count(
                    side=side,
                    real_yes=real_yes,
                    price_cents=price_cents,
                    regime_size_multiplier=float(regime["size_multiplier"]),
                ),
            )
        count = self._cap_count_by_inventory(side, count)
        if count <= 0:
            return

        post_only = self.cfg.execution.post_only if mode == "mm" else False
        tif = self.cfg.execution.time_in_force if mode == "mm" else self.cfg.execution.ioc_time_in_force
        if post_only:
            price_cents = self._apply_post_only_price_guard(side=side, price_cents=price_cents, micro=micro)
            if price_cents is None:
                return

        client_order_id = uuid.uuid4().hex
        self._order_attempts += 1
        log_prefix = (
            f"trade_signal ticker={self.current_ticker} side={side} mode={mode} "
            f"count={count} price={price_cents} edge={edge:.4f} real_yes={real_yes:.4f} "
            f"implied_yes={implied_yes:.4f} ev_cents={ev_cents:.3f}"
        )

        if self.cfg.execution.dry_run:
            self._orders_last_minute.append(now)
            self._logger.info("%s dry_run=1", log_prefix)
            return

        try:
            response = self.client.place_limit_order(
                ticker=self.current_ticker,
                side=side,
                count=count,
                price_cents=price_cents,
                action="buy",
                post_only=post_only,
                time_in_force=tif,
                client_order_id=client_order_id,
            )
            self._orders_last_minute.append(now)
            self._order_targets[client_order_id] = {
                "ticker": self.current_ticker,
                "side": side,
                "mode": mode,
                "price_cents": price_cents,
                "count": count,
                "submitted_epoch": now,
                "spot_at_submit": self.spot_monitor.last_price(),
            }
            self._logger.info("%s order_response=%s", log_prefix, compact_json(response))
        except KalshiRequestError as exc:
            self._order_failures += 1
            self._log_kalshi_error(
                "order_error",
                exc,
                ticker=self.current_ticker,
                side=side,
                mode=mode,
                tif=tif,
                post_only=post_only,
            )

    def _expected_value_cents_per_contract(
        self,
        *,
        side: str,
        mode: str,
        real_yes: float,
        price_cents: int,
    ) -> float:
        if side == "yes":
            raw_ev = real_yes * 100.0 - float(price_cents)
        else:
            raw_ev = (1.0 - real_yes) * 100.0 - float(price_cents)

        if mode == "snipe":
            fee = self.cfg.strategy.taker_fee_cents_per_contract
            slip = self.cfg.strategy.taker_slippage_cents_per_contract
        else:
            fee = self.cfg.strategy.maker_fee_cents_per_contract
            slip = self.cfg.strategy.maker_slippage_cents_per_contract
        return raw_ev - fee - slip

    def _derive_limit_price(
        self,
        *,
        side: str,
        mode: str,
        real_yes: float,
        micro: Dict[str, Any],
        regime_price_cushion: int,
    ) -> Optional[int]:
        best_yes_bid = maybe_int(micro.get("best_yes_bid"))
        best_yes_ask = maybe_int(micro.get("best_yes_ask"))
        best_no_bid = maybe_int(micro.get("best_no_bid"))
        best_no_ask = maybe_int(micro.get("best_no_ask"))
        mm_quotes = self._mm_quotes(real_yes=real_yes, regime_price_cushion=regime_price_cushion)
        fair_yes = int(mm_quotes["fair_yes"])
        fair_no = int(mm_quotes["fair_no"])

        if side == "yes":
            if mode == "snipe":
                candidate = fair_yes + regime_price_cushion
                if best_yes_ask is not None:
                    candidate = max(candidate, best_yes_ask)
                else:
                    # For taker entries, require a live ask from orderbook; never infer from stale/last prints.
                    return None
            else:
                candidate = int(mm_quotes["yes_bid"])
                if best_yes_ask is not None:
                    candidate = min(candidate, best_yes_ask - 1)
        else:
            if mode == "snipe":
                candidate = fair_no + regime_price_cushion
                if best_no_ask is not None:
                    candidate = max(candidate, best_no_ask)
                else:
                    return None
            else:
                candidate = int(mm_quotes["no_bid"])
                if best_no_ask is not None:
                    candidate = min(candidate, best_no_ask - 1)

        return int(clamp(candidate, 1, 99))

    def _spread_gate_ok(self, *, mode: str, micro: Dict[str, Any]) -> bool:
        spread_yes = extract_float(micro.get("spread_yes"))
        if spread_yes is None:
            return True
        if mode == "snipe":
            max_spread = float(self.cfg.strategy.max_yes_spread_cents_for_snipe)
        else:
            max_spread = float(self.cfg.strategy.max_yes_spread_cents_for_mm)
        if max_spread <= 0:
            return True
        if spread_yes <= max_spread:
            return True
        self._logger.info(
            "spread_reject ticker=%s mode=%s spread_yes=%.3f max_allowed=%.3f",
            self.current_ticker,
            mode,
            spread_yes,
            max_spread,
        )
        return False

    def _mm_quotes(self, *, real_yes: float, regime_price_cushion: int) -> Dict[str, int]:
        fair_yes = clamp(real_yes * 100.0, 1.0, 99.0)
        fair_no = 100.0 - fair_yes
        half_spread = max(0.5, float(self.cfg.strategy.mm_half_spread_cents)) + float(regime_price_cushion)
        net_yes = float(self._net_exposure_for_current_market())
        skew = net_yes * float(self.cfg.strategy.inventory_skew_cents_per_contract)

        # Positive net_yes shifts YES quotes down and NO quotes up to rebalance.
        yes_bid = int(round(fair_yes - half_spread - skew))
        yes_ask = int(round(fair_yes + half_spread - skew))
        no_bid = int(round(fair_no - half_spread + skew))
        no_ask = int(round(fair_no + half_spread + skew))

        yes_bid = int(clamp(yes_bid, 1, 99))
        yes_ask = int(clamp(yes_ask, 1, 99))
        no_bid = int(clamp(no_bid, 1, 99))
        no_ask = int(clamp(no_ask, 1, 99))
        if yes_ask <= yes_bid:
            yes_ask = min(99, yes_bid + 1)
        if no_ask <= no_bid:
            no_ask = min(99, no_bid + 1)
        return {
            "fair_yes": int(round(fair_yes)),
            "fair_no": int(round(fair_no)),
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
        }

    def _select_trade_side(self, *, edge: float, micro: Dict[str, Any]) -> str:
        if edge > 1e-9:
            return "yes"
        if edge < -1e-9:
            return "no"
        net = self._net_exposure_for_current_market()
        if net > 0:
            return "no"
        if net < 0:
            return "yes"
        self._flat_side_toggle = not self._flat_side_toggle
        return "yes" if self._flat_side_toggle else "no"

    def _apply_post_only_price_guard(
        self,
        *,
        side: str,
        price_cents: int,
        micro: Dict[str, Any],
    ) -> Optional[int]:
        yes_ask = maybe_int(micro.get("best_yes_ask"))
        no_ask = maybe_int(micro.get("best_no_ask"))
        candidate = int(price_cents)
        if side == "yes" and yes_ask is not None:
            candidate = min(candidate, yes_ask - 1)
        if side == "no" and no_ask is not None:
            candidate = min(candidate, no_ask - 1)
        if candidate < 1 or candidate > 99:
            return None
        return candidate

    def _sized_order_count(self, abs_edge: float, regime_size_multiplier: float) -> int:
        span = max(0.0001, self.cfg.strategy.max_edge_for_sizing)
        ratio = clamp(abs_edge / span, 0.0, 1.0)
        base = int(self.cfg.strategy.base_order_count)
        max_size = int(self.cfg.strategy.max_order_count)
        count = int(round((base + (max_size - base) * ratio) * regime_size_multiplier))
        return max(1, count)

    def _kelly_order_count(
        self,
        *,
        side: str,
        real_yes: float,
        price_cents: int,
        regime_size_multiplier: float,
    ) -> int:
        prob_win = real_yes if side == "yes" else (1.0 - real_yes)
        prob_win = clamp(prob_win, 0.001, 0.999)
        entry = clamp(float(price_cents) / 100.0, 0.01, 0.99)
        b = (1.0 - entry) / entry
        if b <= 0:
            return 0
        full_kelly = ((b * prob_win) - (1.0 - prob_win)) / b
        if full_kelly <= 0:
            return 0
        fraction = clamp(float(self.cfg.strategy.kelly_fraction), 0.0, 1.0)
        bankroll_contracts = max(1.0, float(self.cfg.strategy.kelly_bankroll_contracts))
        sized = full_kelly * fraction * bankroll_contracts * max(0.25, regime_size_multiplier)
        return max(0, int(round(sized)))

    def _cap_count_by_inventory(self, side: str, count: int) -> int:
        gross_now = self._effective_gross_position_for_limits()
        net_now = self._effective_net_exposure_for_limits()
        gross_room = max(0, int(self.cfg.risk.max_gross_position - gross_now))
        if side == "yes":
            directional_room = max(0, int(self.cfg.risk.max_net_exposure - max(0, net_now)))
        else:
            directional_room = max(0, int(self.cfg.risk.max_net_exposure - max(0, -net_now)))
        cap = min(int(count), gross_room, directional_room)
        return max(0, cap)

    def _side_blocked_by_inventory(self, side: str) -> bool:
        gross_now = self._effective_gross_position_for_limits()
        net_now = self._effective_net_exposure_for_limits()
        if gross_now >= self.cfg.risk.max_gross_position:
            return True
        if side == "yes" and net_now >= self.cfg.risk.max_net_exposure:
            return True
        if side == "no" and -net_now >= self.cfg.risk.max_net_exposure:
            return True
        return False

    def _use_delta_position_limits(self) -> bool:
        mode = str(self.cfg.risk.position_limit_scope or "account").strip().lower()
        return mode in {"delta", "delta_from_start", "incremental"}

    def _effective_gross_position_for_limits(self) -> int:
        if not self._use_delta_position_limits():
            return int(self._gross_position)
        baseline = int(self._starting_gross_position or 0)
        return max(0, int(self._gross_position - baseline))

    def _effective_net_exposure_for_limits(self) -> int:
        if not self._use_delta_position_limits():
            return int(self._net_exposure)
        baseline = int(self._starting_net_exposure or 0)
        return int(self._net_exposure - baseline)

    def _effective_position_for_ticker(self, ticker: str) -> PositionExposure:
        pos = self._positions_by_ticker.get(ticker) or PositionExposure()
        if not self._use_delta_position_limits():
            return pos
        baseline = self._starting_positions_by_ticker.get(ticker) or PositionExposure()
        return PositionExposure(
            yes=max(0, int(pos.yes) - int(baseline.yes)),
            no=max(0, int(pos.no) - int(baseline.no)),
        )

    def _net_exposure_for_current_market(self) -> int:
        if not self.current_ticker:
            return 0
        pos = self._effective_position_for_ticker(self.current_ticker)
        return int(pos.net)

    def _prune_order_rate_window(self, now: float) -> None:
        cutoff = now - 60.0
        while self._orders_last_minute and self._orders_last_minute[0] < cutoff:
            self._orders_last_minute.popleft()

    def _current_open_orders(self) -> int:
        if self.cfg.execution.dry_run:
            return 0
        try:
            payload = self.client.get_orders(status="resting", limit=200)
            return len(payload.get("orders", []))
        except KalshiRequestError as exc:
            self._log_kalshi_error("open_orders_error", exc, ticker=self.current_ticker or "ALL")
            return 0

    def _poll_positions(self, force: bool) -> None:
        if self.cfg.execution.dry_run:
            return
        now = time.time()
        if not force and now - self._last_position_poll < max(2.0, self.cfg.risk.position_poll_seconds):
            return
        self._last_position_poll = now
        try:
            payload = self.client.get_positions(limit=1000)
        except KalshiRequestError as exc:
            self._log_kalshi_error("positions_poll_error", exc)
            return

        by_ticker: Dict[str, PositionExposure] = {}
        rows = payload.get("positions")
        if not isinstance(rows, list):
            rows = payload.get("market_positions")
        if not isinstance(rows, list):
            rows = payload.get("open_positions")
        if not isinstance(rows, list):
            rows = []

        for row in rows:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker") or row.get("market_ticker") or "").strip()
            if not ticker:
                continue
            yes_count = first_int(
                row,
                [
                    "yes_count",
                    "yes",
                    "yes_position",
                    "open_yes_count",
                    "position_yes",
                ],
            )
            no_count = first_int(
                row,
                [
                    "no_count",
                    "no",
                    "no_position",
                    "open_no_count",
                    "position_no",
                ],
            )
            if yes_count is None and no_count is None:
                net_guess = first_int(row, ["position", "net_position", "net_exposure", "count"])
                if net_guess is None:
                    continue
                if net_guess >= 0:
                    yes_count = net_guess
                    no_count = 0
                else:
                    yes_count = 0
                    no_count = -net_guess
            exposure = PositionExposure(yes=max(0, yes_count or 0), no=max(0, no_count or 0))
            by_ticker[ticker] = exposure

        self._positions_by_ticker = by_ticker
        self._gross_position = sum(pos.gross for pos in by_ticker.values())
        self._net_exposure = sum(pos.net for pos in by_ticker.values())
        if self._starting_gross_position is None:
            self._starting_gross_position = int(self._gross_position)
            self._starting_net_exposure = int(self._net_exposure)
            self._starting_positions_by_ticker = {
                ticker: PositionExposure(yes=int(pos.yes), no=int(pos.no))
                for ticker, pos in by_ticker.items()
            }
            self._logger.info(
                "position_baseline_set scope=%s gross=%s net=%s",
                self.cfg.risk.position_limit_scope,
                self._starting_gross_position,
                self._starting_net_exposure,
            )
        elif self._use_delta_position_limits():
            # New tickers can appear after startup; treat first-seen exposure as baseline.
            for ticker, pos in by_ticker.items():
                self._starting_positions_by_ticker.setdefault(
                    ticker,
                    PositionExposure(yes=int(pos.yes), no=int(pos.no)),
                )

    def _enforce_position_hard_limits(self) -> None:
        gross_now = self._effective_gross_position_for_limits()
        net_now = self._effective_net_exposure_for_limits()
        if gross_now > self.cfg.risk.max_gross_position:
            if self.cfg.risk.cancel_open_orders_on_kill:
                self._cancel_all_orders_safely()
            raise KillSwitchTriggered(
                "max_gross_position_exceeded "
                f"gross={gross_now} limit={self.cfg.risk.max_gross_position} "
                f"raw_gross={self._gross_position}"
            )
        if abs(net_now) > self.cfg.risk.max_net_exposure:
            if self.cfg.risk.cancel_open_orders_on_kill:
                self._cancel_all_orders_safely()
            raise KillSwitchTriggered(
                "max_net_exposure_exceeded "
                f"net={net_now} limit={self.cfg.risk.max_net_exposure} "
                f"raw_net={self._net_exposure}"
            )

    def _poll_balance(self, force: bool, regime_name: str) -> None:
        now = time.time()
        if not force and now - self._last_balance_poll < max(2.0, self.cfg.risk.balance_poll_seconds):
            return
        self._last_balance_poll = now
        try:
            payload = self.client.get_balance()
        except KalshiRequestError as exc:
            self._log_kalshi_error("balance_poll_error", exc)
            return

        balance = extract_balance_value(payload)
        if balance is None:
            self._logger.warning("balance_missing payload=%s", compact_json(payload))
            return

        self._balance_history.append((now, balance))
        cutoff = now - 3600.0
        while self._balance_history and self._balance_history[0][0] < cutoff:
            self._balance_history.popleft()

        day_key = dt.datetime.utcfromtimestamp(now).strftime("%Y-%m-%d")
        if day_key != self._daily_balance_day:
            self._daily_balance_day = day_key
            self._daily_start_balance = balance

        if self._daily_start_balance is None:
            self._daily_start_balance = balance

        regime_drawdown_mult = 1.0
        if regime_name == "trending":
            regime_drawdown_mult = 0.85
        if regime_name == "spiky":
            regime_drawdown_mult = 0.75

        peak = max(v for _, v in self._balance_history)
        if peak > 0:
            hourly_drawdown = (peak - balance) / peak
            max_hourly_allowed = self.cfg.risk.max_hourly_drawdown_pct * regime_drawdown_mult
            if hourly_drawdown >= max_hourly_allowed:
                if self.cfg.risk.cancel_open_orders_on_kill:
                    self._cancel_all_orders_safely()
                raise KillSwitchTriggered(
                    f"hourly_drawdown={hourly_drawdown:.3f} threshold={max_hourly_allowed:.3f} regime={regime_name}"
                )

        if self._daily_start_balance > 0:
            daily_loss = (self._daily_start_balance - balance) / self._daily_start_balance
            if daily_loss >= self.cfg.risk.daily_max_loss_pct:
                if self.cfg.risk.cancel_open_orders_on_kill:
                    self._cancel_all_orders_safely()
                raise KillSwitchTriggered(
                    f"daily_loss={daily_loss:.3f} threshold={self.cfg.risk.daily_max_loss_pct:.3f}"
                )

    def _force_flatten_current_market(self, micro: Dict[str, Any]) -> None:
        if not self.current_ticker:
            return
        now = time.time()
        cooldown = max(0.5, float(self.cfg.risk.force_flatten_cooldown_seconds))
        if now - self._last_force_flatten_epoch < cooldown:
            return
        self._last_force_flatten_epoch = now
        self._logger.warning("force_flatten ticker=%s", self.current_ticker)
        if self._is_mm_mode():
            self._mm_reset_state()
        if self.cfg.execution.dry_run:
            return

        try:
            self.client.cancel_all_open_orders(ticker=self.current_ticker)
        except KalshiRequestError as exc:
            self._log_kalshi_error("force_flatten_cancel_error", exc, ticker=self.current_ticker)

        self._poll_positions(force=True)
        exposure = self._positions_by_ticker.get(self.current_ticker)
        if not exposure or exposure.gross <= 0:
            return

        yes_bid = maybe_int(micro.get("best_yes_bid")) or 1
        no_bid = maybe_int(micro.get("best_no_bid")) or 1
        tif = self.cfg.execution.ioc_time_in_force

        try:
            if exposure.yes > 0:
                self.client.place_limit_order(
                    ticker=self.current_ticker,
                    side="yes",
                    count=exposure.yes,
                    price_cents=int(clamp(yes_bid, 1, 99)),
                    action="sell",
                    post_only=False,
                    time_in_force=tif,
                )
            if exposure.no > 0:
                self.client.place_limit_order(
                    ticker=self.current_ticker,
                    side="no",
                    count=exposure.no,
                    price_cents=int(clamp(no_bid, 1, 99)),
                    action="sell",
                    post_only=False,
                    time_in_force=tif,
                )
        except KalshiRequestError as exc:
            self._log_kalshi_error("force_flatten_order_error", exc, ticker=self.current_ticker)

    def _write_performance_if_due(
        self,
        *,
        real_yes: float,
        implied_yes: float,
        regime_name: str,
        micro: Dict[str, Any],
    ) -> None:
        now = time.time()
        if now - self._last_perf_write < 900.0:
            return
        self._last_perf_write = now

        wins = 0
        losses = 0
        total_slippage = 0.0
        fill_count = 0
        adverse_count = 0
        adverse_total = 0
        if not self.cfg.execution.dry_run:
            try:
                min_ts = int((now - 900.0) * 1000)
                fills_payload = self.client.get_fills(
                    ticker=self.current_ticker or None,
                    limit=1000,
                    min_ts=min_ts,
                )
                fills = fills_payload.get("fills", [])
                fill_count = len(fills)
                mark_yes_cents = int(round(implied_yes * 100.0))
                for fill in fills:
                    result = classify_fill_vs_mark(fill, mark_yes_cents)
                    if result is None:
                        continue
                    pnl_cents = result["mark_pnl_cents"]
                    if pnl_cents >= 0:
                        wins += 1
                    else:
                        losses += 1

                    client_order_id = str(fill.get("client_order_id") or "").strip()
                    if client_order_id and client_order_id in self._order_targets:
                        target_meta = self._order_targets[client_order_id]
                        fill_side_price = result["fill_side_cents"]
                        total_slippage += abs(fill_side_price - float(target_meta["price_cents"])) * float(
                            result["count"]
                        )
                adverse_count, adverse_total = self._estimate_adverse_selection(fills)
            except KalshiRequestError as exc:
                self._log_kalshi_error("performance_fill_error", exc, ticker=self.current_ticker or "ALL")

        payload = {
            "as_of": iso_now(),
            "ticker": self.current_ticker,
            "series_ticker": self.cfg.strategy.series_ticker,
            "window_minutes": 15,
            "wins": wins,
            "losses": losses,
            "fill_count": fill_count,
            "slippage_cents": round(total_slippage, 4),
            "adverse_selection_count": adverse_count,
            "adverse_selection_total": adverse_total,
            "adverse_selection_rate": (adverse_count / adverse_total) if adverse_total else 0.0,
            "last_real_yes_probability": round(real_yes, 6),
            "last_kalshi_implied_yes_probability": round(implied_yes, 6),
            "edge_threshold": self.cfg.strategy.edge_threshold,
            "order_attempts": self._order_attempts,
            "order_failures": self._order_failures,
            "last_401_epoch": self._last_401_epoch,
            "regime": regime_name,
            "microstructure": {
                "spread_yes": micro.get("spread_yes"),
                "spread_no": micro.get("spread_no"),
                "imbalance": micro.get("imbalance"),
                "yes_depth": micro.get("yes_depth"),
                "no_depth": micro.get("no_depth"),
                "pinned_state": micro.get("pinned_state"),
                "zero_spread_state": micro.get("zero_spread_state"),
                "book_age_seconds": micro.get("book_age_seconds"),
                "book_stale_seconds": micro.get("book_stale_seconds"),
            },
            "reference": {
                "source": self.spot_monitor.last_reference_source,
                "last_error": self.spot_monitor.last_reference_error,
                "spot_price": self.spot_monitor.last_spot_price,
                "perp_price": self.spot_monitor.last_perp_price,
                "coinbase_price": self.spot_monitor.last_coinbase_price,
                "kraken_price": self.spot_monitor.last_kraken_price,
                "cross_exchange_diff_pct": self.spot_monitor.last_cross_exchange_diff_pct,
                "flash_spike_pause_remaining_seconds": self.spot_monitor.pause_remaining_seconds(),
                "return_30s": self._latest_reference_returns["return_30s"],
                "return_1m": self._latest_reference_returns["return_1m"],
                "return_15m": self._latest_reference_returns["return_15m"],
            },
            "request_rate_limit": self.client.rate_limit_stats(),
            "inventory": {
                "gross_position": self._gross_position,
                "net_exposure": self._net_exposure,
                "effective_gross_for_limits": self._effective_gross_position_for_limits(),
                "effective_net_for_limits": self._effective_net_exposure_for_limits(),
                "position_limit_scope": self.cfg.risk.position_limit_scope,
            },
        }
        out_path = Path(self.cfg.paths.performance_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        self._logger.info("performance_write path=%s payload=%s", out_path.resolve(), compact_json(payload))

    def _estimate_adverse_selection(self, fills: list[dict[str, Any]]) -> tuple[int, int]:
        horizon = max(1.0, float(self.cfg.strategy.adverse_selection_horizon_seconds))
        move_threshold = float(self.cfg.strategy.adverse_selection_move_threshold)
        adverse = 0
        considered = 0
        for fill in fills:
            side = str(fill.get("side") or "").lower()
            action = str(fill.get("action") or "buy").lower()
            if action != "buy" or side not in {"yes", "no"}:
                continue
            ts_ms = extract_timestamp_ms(
                fill.get("created_time")
                or fill.get("created_ts")
                or fill.get("ts")
                or fill.get("fill_time")
            )
            if ts_ms is None:
                continue
            fill_epoch = ts_ms / 1000.0
            spot_at_fill = self._spot_price_at(fill_epoch, prefer_after=False)
            spot_after = self._spot_price_at(fill_epoch + horizon, prefer_after=True)
            if spot_at_fill is None or spot_after is None:
                continue
            considered += 1
            if side == "yes" and (spot_after <= spot_at_fill - move_threshold):
                adverse += 1
            if side == "no" and (spot_after >= spot_at_fill + move_threshold):
                adverse += 1
        return adverse, considered

    def _spot_price_at(self, target_epoch: float, prefer_after: bool) -> Optional[float]:
        samples = list(self.spot_monitor.samples)
        if not samples:
            return None
        if prefer_after:
            for ts, px in samples:
                if ts >= target_epoch:
                    return px
            return None
        nearest = min(samples, key=lambda sample: abs(sample[0] - target_epoch))
        return nearest[1]

    def _cancel_all_orders_safely(self) -> None:
        try:
            result = self.client.cancel_all_open_orders(ticker=None)
            self._logger.warning("cancel_all_orders result=%s", compact_json(result))
        except KalshiRequestError as exc:
            self._logger.error("cancel_all_orders_failed err=%s", exc)

    def _seconds_to_close(self, market: Dict[str, Any]) -> Optional[float]:
        close_ms = extract_timestamp_ms(
            market.get("close_time")
            or market.get("close_date")
            or market.get("expiration_time")
            or market.get("end_time")
            or market.get("close_ts")
        )
        if close_ms is None:
            close_ms = parse_ticker_close_epoch_ms(str(market.get("ticker") or ""))
        if close_ms is None:
            return None
        return (close_ms - int(time.time() * 1000)) / 1000.0

    def _shutdown(self) -> None:
        if self.book_monitor:
            self.book_monitor.stop()
        if self.cfg.risk.cancel_open_orders_on_shutdown and not self.cfg.execution.dry_run:
            self._cancel_all_orders_safely()

    def _log_kalshi_error(self, label: str, exc: Exception, **fields: Any) -> None:
        err_text = str(exc)
        err_lower = err_text.lower()
        if "status=401" in err_lower or "unauthorized" in err_lower:
            self._last_401_epoch = time.time()
        details = " ".join(f"{k}={v}" for k, v in fields.items()) if fields else ""
        if "status=429" in err_lower or "too many requests" in err_lower:
            self._logger.warning(
                "%s %s err=%s rate_limit=%s",
                label,
                details,
                err_text,
                compact_json(self.client.rate_limit_stats()),
            )
            return
        self._logger.warning("%s %s err=%s", label, details, err_text)


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("kalshi_hft")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_path = Path(log_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def extract_balance_value(payload: Dict[str, Any]) -> Optional[float]:
    key_order = [
        "balance",
        "available_balance",
        "portfolio_balance",
        "cash_balance",
        "balance_cents",
    ]
    for key in key_order:
        item = payload.get(key)
        value = extract_float(item)
        if value is None:
            continue
        if key.endswith("_cents"):
            return value / 100.0
        return value
    return None


def classify_fill_vs_mark(fill: Dict[str, Any], mark_yes_cents: int) -> Optional[Dict[str, float]]:
    side = str(fill.get("side") or "").lower()
    action = str(fill.get("action") or "buy").lower()
    count = extract_float(fill.get("count") or fill.get("quantity") or fill.get("size"))
    if count is None or count <= 0:
        return None

    yes_price = extract_float(fill.get("yes_price"))
    no_price = extract_float(fill.get("no_price"))
    if side == "yes":
        fill_side_cents = yes_price
        mark_side_cents = float(mark_yes_cents)
    elif side == "no":
        fill_side_cents = no_price
        mark_side_cents = float(100 - mark_yes_cents)
    else:
        return None

    if fill_side_cents is None:
        return None
    if action == "sell":
        pnl_per_contract = fill_side_cents - mark_side_cents
    else:
        pnl_per_contract = mark_side_cents - fill_side_cents
    return {
        "mark_pnl_cents": pnl_per_contract * count,
        "fill_side_cents": fill_side_cents,
        "count": count,
    }


def infer_binance_symbol(series_ticker: str) -> str:
    upper = series_ticker.upper()
    if "ETH" in upper:
        return "ETHUSDT"
    if "SOL" in upper:
        return "SOLUSDT"
    return "BTCUSDT"


_EASTERN_TZ = ZoneInfo("America/New_York")
_MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
_TICKER_CLOSE_RE = re.compile(
    r"-(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})(\d{2})(\d{2})-",
    re.IGNORECASE,
)


def parse_ticker_close_epoch_ms(ticker: str) -> Optional[int]:
    text = str(ticker or "").strip().upper()
    match = _TICKER_CLOSE_RE.search(text)
    if not match:
        return None
    yy, mon, day, hour, minute = match.groups()
    month = _MONTH_MAP.get(mon.upper())
    if month is None:
        return None
    year = 2000 + int(yy)
    try:
        close_local = dt.datetime(
            year,
            month,
            int(day),
            int(hour),
            int(minute),
            tzinfo=_EASTERN_TZ,
        )
    except ValueError:
        return None
    return int(close_local.timestamp() * 1000)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def compact_json(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), default=str)[:800]


def unwrap_market_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    market = payload.get("market")
    if isinstance(market, dict):
        return market
    return payload


def maybe_int(value: Any) -> Optional[int]:
    as_float = extract_float(value)
    if as_float is None:
        return None
    return int(round(as_float))


def first_int(payload: Dict[str, Any], keys: list[str]) -> Optional[int]:
    for key in keys:
        if key not in payload:
            continue
        parsed = maybe_int(payload.get(key))
        if parsed is not None:
            return parsed
    return None


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kalshi 15m crypto HFT execution engine.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.example.json"),
        help="Path to config JSON.",
    )
    parser.add_argument("--runtime-minutes", type=float, default=None, help="Optional runtime override.")
    parser.add_argument("--live", action="store_true", help="Force live mode (overrides dry_run=true).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = BotConfig.from_json(args.config)
    if args.runtime_minutes is not None:
        cfg.execution.runtime_minutes = args.runtime_minutes
    if args.live:
        cfg.execution.dry_run = False
    engine = KalshiHFTEngine(cfg)
    return engine.run()


if __name__ == "__main__":
    raise SystemExit(main())
