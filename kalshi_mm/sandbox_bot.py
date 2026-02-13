from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Literal
from zoneinfo import ZoneInfo

from .config import BotConfig
from .kalshi_api import KalshiApiError, KalshiClient, KalshiCredentials
from .models import MarketState
from .strategy import AdaptiveMarketMaker


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class MarketRuntimeState:
    ticker: str
    strategy: AdaptiveMarketMaker
    recent_mids: deque[float]
    cycles: int = 0
    orders_submitted: int = 0
    orders_canceled: int = 0
    skipped_cycles: int = 0
    max_abs_position: int = 0
    final_position: int = 0
    post_only_reprices: int = 0
    post_only_rejects: int = 0
    preclose_flatten_calls: int = 0
    last_submit_epoch: float = 0.0
    last_reason: str = ""


def run_sandbox(
    cfg: BotConfig,
    credentials: KalshiCredentials,
    markets: list[str],
    loop_seconds: float,
    runtime_minutes: float,
    submit_orders: bool,
    report_path: str | Path,
    base_url: str = "https://demo-api.kalshi.co",
    pss_salt_mode: Literal["digest", "max"] = "digest",
) -> dict[str, Any]:
    client = KalshiClient(
        credentials=credentials,
        base_url=base_url,
        pss_salt_mode=pss_salt_mode,
    )
    balance = client.get_balance()
    starting_balance_cents = _safe_int(balance.get("balance"))
    session_low_balance_cents = starting_balance_cents
    session_high_balance_cents = starting_balance_cents
    session_stop_loss_cents = max(0, _safe_int(cfg.risk.session_stop_loss_cents))
    session_take_profit_cents = max(0, _safe_int(cfg.risk.session_take_profit_cents))
    min_requote_seconds = max(loop_seconds, float(cfg.risk.min_requote_seconds))
    max_orders_per_market = max(1, _safe_int(cfg.risk.max_orders_per_market))
    no_trade_window_cycle_limit = max(2, _safe_int(cfg.risk.no_trade_window_cycle_limit))
    auto_roll_on_no_trade_window = bool(cfg.risk.auto_roll_on_no_trade_window)
    momentum_soft_guard = max(0.0005, float(cfg.risk.momentum_soft_guard))
    momentum_hard_guard = max(momentum_soft_guard + 0.0005, float(cfg.risk.momentum_hard_guard))
    adverse_side_size_cut = _clamp(float(cfg.risk.adverse_side_size_cut), 0.0, 0.95)
    toxicity_edge_boost = max(0.0, float(cfg.risk.toxicity_edge_boost))

    requested_markets = [m.strip().upper() for m in markets if m and m.strip()]
    preclose_guard_seconds = max(
        90.0,
        float(cfg.risk.preclose_guard_seconds),
        loop_seconds * 3.0,
    )
    entry_time_buffer_seconds = max(0.0, float(cfg.risk.entry_time_buffer_seconds))
    no_new_entries_before_close_seconds = preclose_guard_seconds + entry_time_buffer_seconds
    min_seconds_to_close_for_entry = (
        no_new_entries_before_close_seconds + max(60.0, loop_seconds * 2.0)
    )
    resolved_markets, resolution_notes = _resolve_markets_with_fallbacks(
        client=client,
        requested_markets=requested_markets,
        min_seconds_to_close_for_entry=min_seconds_to_close_for_entry,
        preclose_guard_seconds=preclose_guard_seconds,
    )
    if not resolved_markets:
        raise KalshiApiError(
            "no_markets_resolved from requested_markets="
            + ",".join(requested_markets)
        )

    runtime_by_market = _build_runtime_states(cfg, resolved_markets, previous=None)
    tracked_markets = list(runtime_by_market.keys())
    (
        live_size_scale,
        base_market_max_position,
        base_market_max_order_size,
        per_market_max_position,
        per_market_max_order_size,
    ) = _scaled_market_limits(cfg, len(resolved_markets))

    run_id = f"MMBOT-{int(time.time())}"
    run_tag = f"mm{_base36(int(time.time()))[-6:]}"
    started = time.time()
    deadline = started + max(1.0, runtime_minutes * 60.0)
    cycles = 0
    total_orders_submitted = 0
    total_orders_canceled = 0
    total_orders_canceled_on_exit = 0
    total_flatten_orders_submitted = 0
    total_preclose_flatten_warnings = 0
    total_flatten_errors = 0
    total_flatten_residual_markets = 0
    total_errors = 0
    stop_reason = "runtime_elapsed"
    order_seq = 0
    no_trade_window_cycles = 0
    market_rolls = 0

    while time.time() < deadline:
        cycles += 1
        cycle_submissions_before = total_orders_submitted
        if submit_orders and (session_stop_loss_cents > 0 or session_take_profit_cents > 0):
            try:
                session_balance = client.get_balance()
                current_balance_cents = _safe_int(session_balance.get("balance"))
                session_low_balance_cents = min(session_low_balance_cents, current_balance_cents)
                session_high_balance_cents = max(session_high_balance_cents, current_balance_cents)
                if _session_stop_loss_hit(
                    start_balance_cents=starting_balance_cents,
                    current_balance_cents=current_balance_cents,
                    stop_loss_cents=session_stop_loss_cents,
                ):
                    stop_reason = (
                        "session_stop_loss_hit_"
                        f"{starting_balance_cents - current_balance_cents}c"
                    )
                    break
                if _session_take_profit_hit(
                    start_balance_cents=starting_balance_cents,
                    current_balance_cents=current_balance_cents,
                    take_profit_cents=session_take_profit_cents,
                ):
                    stop_reason = (
                        "session_take_profit_hit_"
                        f"{current_balance_cents - starting_balance_cents}c"
                    )
                    break
            except KalshiApiError:
                total_errors += 1

        try:
            positions_resp = client.get_positions(limit=200)
        except KalshiApiError:
            positions_resp = {"market_positions": []}
            total_errors += 1

        pos_map = _extract_positions(positions_resp)

        for ticker in resolved_markets:
            state = runtime_by_market[ticker]
            state.cycles += 1
            position_qty = int(pos_map.get(ticker, 0))
            state.final_position = position_qty
            state.max_abs_position = max(state.max_abs_position, abs(position_qty))

            try:
                market_payload = client.get_market(ticker)
                market = market_payload.get("market", market_payload)
            except KalshiApiError as exc:
                state.skipped_cycles += 1
                state.last_reason = f"market_error:{exc}"
                total_errors += 1
                continue

            market_status = str(market.get("status", "")).lower()
            if market_status and market_status not in {"active", "open"}:
                state.skipped_cycles += 1
                state.last_reason = f"status_{market_status}"
                continue

            seconds_to_close = _market_seconds_to_close(market, now_epoch=time.time())
            if (
                seconds_to_close is not None
                and seconds_to_close <= no_new_entries_before_close_seconds
            ):
                if submit_orders:
                    state.preclose_flatten_calls += 1
                    canceled = _cancel_resting_bot_orders(
                        client,
                        ticker=ticker,
                        client_order_prefix=run_tag,
                    )
                    state.orders_canceled += canceled
                    total_orders_canceled += canceled
                    if position_qty != 0:
                        flatten_submitted, flatten_errors = _flatten_single_market_on_exit(
                            client=client,
                            ticker=ticker,
                            state=state,
                            run_tag=run_tag,
                            max_order_size=per_market_max_order_size,
                            order_seq_start=order_seq,
                        )
                        order_seq += max(1, flatten_submitted * 2)
                        total_flatten_orders_submitted += flatten_submitted
                        total_preclose_flatten_warnings += flatten_errors
                state.skipped_cycles += 1
                if (
                    seconds_to_close is not None
                    and seconds_to_close <= preclose_guard_seconds
                ):
                    state.last_reason = "preclose_guard"
                else:
                    state.last_reason = "entry_cutoff_guard"
                continue

            best_bid, best_ask = _extract_yes_bid_ask_prob(market)
            if best_ask <= best_bid:
                state.skipped_cycles += 1
                state.last_reason = "invalid_bid_ask"
                continue
            spread = best_ask - best_bid
            required_market_spread = max(
                float(cfg.risk.min_market_spread),
                float(cfg.risk.min_quote_edge) + 0.01,
            )
            if spread < required_market_spread:
                state.skipped_cycles += 1
                state.last_reason = "spread_too_tight"
                continue
            if spread > 0.25:
                state.skipped_cycles += 1
                state.last_reason = "spread_too_wide"
                continue

            mid = (best_bid + best_ask) * 0.5
            momentum = 0.0
            if state.recent_mids:
                momentum = mid - state.recent_mids[-1]
            state.recent_mids.append(mid)
            volatility = _mid_volatility(state.recent_mids)

            quote = state.strategy.make_quote(
                market=MarketState(
                    timestamp=time.time(),
                    mid=mid,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    volatility=volatility,
                    momentum=momentum,
                ),
                position_qty=position_qty,
                max_position=per_market_max_position,
            )
            if quote is None:
                state.skipped_cycles += 1
                state.last_reason = "no_quote"
                continue
            quote_edge = quote.ask - quote.bid
            toxicity_pressure = _toxicity_pressure(
                momentum=momentum,
                volatility=volatility,
                soft_guard=momentum_soft_guard,
                hard_guard=momentum_hard_guard,
            )
            required_quote_edge = (
                max(float(cfg.strategy.min_edge), float(cfg.risk.min_quote_edge))
                + (toxicity_edge_boost * toxicity_pressure)
            )
            if quote_edge < required_quote_edge:
                state.skipped_cycles += 1
                state.last_reason = "quote_edge_too_thin"
                continue

            yes_size = max(0, min(quote.buy_size, per_market_max_order_size, per_market_max_position - position_qty))
            no_size = max(0, min(quote.sell_size, per_market_max_order_size, per_market_max_position + position_qty))
            yes_size, no_size, toxicity_reason = _apply_toxicity_size_controls(
                yes_size=yes_size,
                no_size=no_size,
                position_qty=position_qty,
                momentum=momentum,
                volatility=volatility,
                soft_guard=momentum_soft_guard,
                hard_guard=momentum_hard_guard,
                adverse_side_size_cut=adverse_side_size_cut,
            )
            if toxicity_reason:
                state.skipped_cycles += 1
                state.last_reason = toxicity_reason
                continue

            yes_price_cents = _prob_to_cents(quote.bid)
            no_price_cents = _prob_to_cents(1.0 - quote.ask)
            # Keep prices one tick off touch so post-only orders are less likely to cross.
            yes_max_post_only = max(1, _prob_to_cents(best_ask) - 1)
            no_max_post_only = max(1, _prob_to_cents(1.0 - best_bid) - 1)
            if yes_max_post_only <= 1:
                yes_size = 0
            else:
                yes_price_cents = min(yes_price_cents, yes_max_post_only)
            if no_max_post_only <= 1:
                no_size = 0
            else:
                no_price_cents = min(no_price_cents, no_max_post_only)
            if yes_size == 0 and no_size == 0:
                state.skipped_cycles += 1
                state.last_reason = "no_safe_post_only_price"
                continue

            if submit_orders:
                if state.orders_submitted >= max_orders_per_market:
                    state.skipped_cycles += 1
                    state.last_reason = "max_orders_per_market_reached"
                    continue
                now_epoch = time.time()
                if (
                    state.last_submit_epoch > 0
                    and (now_epoch - state.last_submit_epoch) < min_requote_seconds
                ):
                    state.skipped_cycles += 1
                    state.last_reason = "requote_cooldown"
                    continue
                canceled = _cancel_resting_bot_orders(client, ticker=ticker, client_order_prefix=run_tag)
                state.orders_canceled += canceled
                total_orders_canceled += canceled

                submitted_this_cycle = 0
                if yes_size > 0:
                    yes_client_id, order_seq = _next_client_order_id(run_tag, ticker, "by", order_seq)
                    payload = _build_buy_yes_order_payload(
                        ticker=ticker,
                        count=yes_size,
                        yes_price=yes_price_cents,
                        client_order_id=yes_client_id,
                    )
                    ok, submit_error, reprices = _submit_post_only_with_reprice(
                        client=client,
                        payload=payload,
                        price_key="yes_price",
                    )
                    state.post_only_reprices += reprices
                    if ok:
                        state.orders_submitted += 1
                        submitted_this_cycle += 1
                        total_orders_submitted += 1
                    else:
                        state.last_reason = f"submit_yes_error:{submit_error}"
                        if "post only cross" in submit_error.lower():
                            state.post_only_rejects += 1
                        else:
                            total_errors += 1

                if no_size > 0:
                    no_client_id, order_seq = _next_client_order_id(run_tag, ticker, "bn", order_seq)
                    payload = _build_buy_no_order_payload(
                        ticker=ticker,
                        count=no_size,
                        no_price=no_price_cents,
                        client_order_id=no_client_id,
                    )
                    ok, submit_error, reprices = _submit_post_only_with_reprice(
                        client=client,
                        payload=payload,
                        price_key="no_price",
                    )
                    state.post_only_reprices += reprices
                    if ok:
                        state.orders_submitted += 1
                        submitted_this_cycle += 1
                        total_orders_submitted += 1
                    else:
                        state.last_reason = f"submit_no_error:{submit_error}"
                        if "post only cross" in submit_error.lower():
                            state.post_only_rejects += 1
                        else:
                            total_errors += 1
                if submitted_this_cycle > 0:
                    state.last_submit_epoch = now_epoch
            else:
                state.last_reason = (
                    f"dry_run yes={yes_size}@{yes_price_cents} no={no_size}@{no_price_cents} "
                    f"pos={position_qty}"
                )

        cycle_only_untradeable_reasons = all(
            _is_untradeable_reason(runtime_by_market[t].last_reason)
            for t in resolved_markets
        )
        if (
            submit_orders
            and total_orders_submitted == cycle_submissions_before
            and cycle_only_untradeable_reasons
        ):
            no_trade_window_cycles += 1
        else:
            no_trade_window_cycles = 0
        if no_trade_window_cycles >= no_trade_window_cycle_limit:
            if auto_roll_on_no_trade_window:
                rolled_markets, roll_notes = _resolve_markets_with_fallbacks(
                    client=client,
                    requested_markets=requested_markets,
                    min_seconds_to_close_for_entry=min_seconds_to_close_for_entry,
                    preclose_guard_seconds=preclose_guard_seconds,
                )
                if rolled_markets and rolled_markets != resolved_markets:
                    for note in roll_notes:
                        tagged = dict(note)
                        base_method = tagged.get("method", "")
                        tagged["method"] = (
                            f"roll_no_trade_window:{base_method}"
                            if base_method
                            else "roll_no_trade_window"
                        )
                        resolution_notes.append(tagged)
                    resolved_markets = rolled_markets
                    runtime_by_market = _build_runtime_states(
                        cfg=cfg,
                        markets=resolved_markets,
                        previous=runtime_by_market,
                    )
                    (
                        live_size_scale,
                        base_market_max_position,
                        base_market_max_order_size,
                        per_market_max_position,
                        per_market_max_order_size,
                    ) = _scaled_market_limits(cfg, len(resolved_markets))
                    tracked_markets = list(runtime_by_market.keys())
                    no_trade_window_cycles = 0
                    market_rolls += 1
                    continue
            stop_reason = "no_trade_window"
            break

        if total_errors >= 25:
            stop_reason = "error_limit_reached"
            break
        time.sleep(max(0.5, loop_seconds))

    ended = time.time()
    if submit_orders:
        for ticker in tracked_markets:
            canceled_on_exit = _cancel_resting_bot_orders(
                client,
                ticker=ticker,
                client_order_prefix=run_tag,
            )
            runtime_by_market[ticker].orders_canceled += canceled_on_exit
            total_orders_canceled += canceled_on_exit
            total_orders_canceled_on_exit += canceled_on_exit

        flatten_submitted, flatten_errors = _flatten_positions_on_exit(
            client=client,
            resolved_markets=tracked_markets,
            runtime_by_market=runtime_by_market,
            run_tag=run_tag,
            max_order_size=per_market_max_order_size,
            order_seq=order_seq,
        )
        total_flatten_orders_submitted += flatten_submitted
        total_flatten_errors += flatten_errors
        total_flatten_residual_markets = sum(
            1 for ticker in tracked_markets if runtime_by_market[ticker].final_position != 0
        )

        try:
            final_positions_payload = client.get_positions(limit=300)
            final_pos_map = _extract_positions(final_positions_payload)
            for ticker in tracked_markets:
                runtime_by_market[ticker].final_position = int(final_pos_map.get(ticker, 0))
        except KalshiApiError:
            total_errors += 1

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "sandbox",
        "base_url": base_url,
        "pss_salt_mode": pss_salt_mode,
        "submit_orders": submit_orders,
        "run_id": run_id,
        "run_tag": run_tag,
        "requested_markets": requested_markets,
        "resolved_markets": resolved_markets,
        "tracked_markets": tracked_markets,
        "market_resolution_notes": resolution_notes,
        "markets": resolved_markets,
        "starting_balance": balance,
        "risk_split": {
            "live_size_scale": live_size_scale,
            "base_per_market_max_position": base_market_max_position,
            "base_per_market_max_order_size": base_market_max_order_size,
            "per_market_max_position": per_market_max_position,
            "per_market_max_order_size": per_market_max_order_size,
        },
        "risk_controls": {
            "session_stop_loss_cents": session_stop_loss_cents,
            "session_take_profit_cents": session_take_profit_cents,
            "session_start_balance_cents": starting_balance_cents,
            "session_low_balance_cents": session_low_balance_cents,
            "session_high_balance_cents": session_high_balance_cents,
            "session_drawdown_cents": max(0, starting_balance_cents - session_low_balance_cents),
            "preclose_guard_seconds": preclose_guard_seconds,
            "no_new_entries_before_close_seconds": no_new_entries_before_close_seconds,
            "entry_min_seconds_to_close": min_seconds_to_close_for_entry,
            "no_trade_window_cycle_limit": no_trade_window_cycle_limit,
            "auto_roll_on_no_trade_window": auto_roll_on_no_trade_window,
            "momentum_soft_guard": momentum_soft_guard,
            "momentum_hard_guard": momentum_hard_guard,
            "adverse_side_size_cut": adverse_side_size_cut,
            "toxicity_edge_boost": toxicity_edge_boost,
            "min_requote_seconds": min_requote_seconds,
            "max_orders_per_market": max_orders_per_market,
            "min_market_spread": float(cfg.risk.min_market_spread),
            "min_quote_edge": float(cfg.risk.min_quote_edge),
        },
        "runtime": {
            "runtime_minutes_requested": runtime_minutes,
            "loop_seconds": loop_seconds,
            "started_epoch": started,
            "ended_epoch": ended,
            "elapsed_seconds": ended - started,
            "cycles": cycles,
            "market_rolls": market_rolls,
            "stop_reason": stop_reason,
        },
        "counts": {
            "orders_submitted": total_orders_submitted,
            "orders_canceled": total_orders_canceled,
            "orders_canceled_on_exit": total_orders_canceled_on_exit,
            "flatten_orders_submitted": total_flatten_orders_submitted,
            "preclose_flatten_warnings": total_preclose_flatten_warnings,
            "flatten_errors": total_flatten_errors,
            "flatten_residual_markets": total_flatten_residual_markets,
            "errors": total_errors,
        },
        "market_summaries": [_market_summary(runtime_by_market[m]) for m in tracked_markets],
    }
    _write_json(report_path, report)
    return report


def _resolve_requested_markets(
    client: KalshiClient,
    requested_markets: list[str],
    min_seconds_to_close: float = 0.0,
    allow_next_fallback: bool = True,
) -> tuple[list[str], list[dict[str, str]]]:
    notes: list[dict[str, str]] = []
    active_or_open_markets = _list_open_markets(client)
    all_markets = _list_all_markets(client)
    active_or_open_tickers = {
        str(m.get("ticker", "")).strip().upper()
        for m in active_or_open_markets
        if str(m.get("ticker", "")).strip()
    }

    resolved: list[str] = []
    for requested in requested_markets:
        req = requested.strip().upper()
        if not req:
            continue

        if req in active_or_open_tickers:
            resolved.append(req)
            notes.append({"requested": req, "resolved": req, "method": "exact_open"})
            continue

        if _market_exists(client, req):
            resolved.append(req)
            notes.append({"requested": req, "resolved": req, "method": "exact_lookup"})
            continue

        resolved_from_graph = _resolve_from_series_or_event(
            client,
            req,
            min_seconds_to_close=min_seconds_to_close,
            allow_non_open=False,
        )
        if resolved_from_graph:
            resolved.append(resolved_from_graph)
            notes.append(
                {
                    "requested": req,
                    "resolved": resolved_from_graph,
                    "method": "series_or_event",
                }
            )
            continue

        resolved_from_search = _resolve_by_symbol_search(
            req,
            active_or_open_markets,
            min_seconds_to_close=min_seconds_to_close,
            allow_non_open=False,
        )
        if resolved_from_search:
            resolved.append(resolved_from_search)
            notes.append(
                {
                    "requested": req,
                    "resolved": resolved_from_search,
                    "method": "symbol_search",
                }
            )
            continue

        if allow_next_fallback:
            resolved_next = _resolve_from_series_or_event(
                client,
                req,
                min_seconds_to_close=0.0,
                allow_non_open=True,
            )
            if resolved_next:
                resolved.append(resolved_next)
                notes.append(
                    {
                        "requested": req,
                        "resolved": resolved_next,
                        "method": "series_or_event_next",
                    }
                )
                continue

            resolved_search_next = _resolve_by_symbol_search(
                req,
                all_markets,
                min_seconds_to_close=0.0,
                allow_non_open=True,
            )
            if resolved_search_next:
                resolved.append(resolved_search_next)
                notes.append(
                    {
                        "requested": req,
                        "resolved": resolved_search_next,
                        "method": "symbol_search_next",
                    }
                )
                continue

        notes.append({"requested": req, "resolved": "", "method": "unresolved"})

    deduped: list[str] = []
    seen: set[str] = set()
    for ticker in resolved:
        if ticker in seen:
            continue
        seen.add(ticker)
        deduped.append(ticker)
    return deduped, notes


def _resolve_markets_with_fallbacks(
    client: KalshiClient,
    requested_markets: list[str],
    min_seconds_to_close_for_entry: float,
    preclose_guard_seconds: float,
) -> tuple[list[str], list[dict[str, str]]]:
    strict_min_seconds = max(0.0, float(min_seconds_to_close_for_entry))
    resolved, notes = _resolve_requested_markets(
        client=client,
        requested_markets=requested_markets,
        min_seconds_to_close=strict_min_seconds,
        allow_next_fallback=False,
    )
    if resolved:
        return resolved, notes

    relaxed_min_seconds = max(0.0, float(preclose_guard_seconds) + 30.0)
    if relaxed_min_seconds < strict_min_seconds:
        resolved, notes = _resolve_requested_markets(
            client=client,
            requested_markets=requested_markets,
            min_seconds_to_close=relaxed_min_seconds,
            allow_next_fallback=False,
        )
        notes.append(
            {
                "requested": "*",
                "resolved": ",".join(resolved),
                "method": f"fallback_relaxed_min_seconds_to_close={int(relaxed_min_seconds)}",
            }
        )
        if resolved:
            return resolved, notes

    resolved, notes = _resolve_requested_markets(
        client=client,
        requested_markets=requested_markets,
        min_seconds_to_close=0.0,
        allow_next_fallback=True,
    )
    notes.append(
        {
            "requested": "*",
            "resolved": ",".join(resolved),
            "method": "fallback_any_open_market",
        }
    )
    return resolved, notes


def _build_runtime_states(
    cfg: BotConfig,
    markets: list[str],
    previous: dict[str, MarketRuntimeState] | None = None,
) -> dict[str, MarketRuntimeState]:
    runtime_by_market: dict[str, MarketRuntimeState] = dict(previous or {})
    for market in markets:
        if market in runtime_by_market:
            continue
        runtime_by_market[market] = MarketRuntimeState(
            ticker=market,
            strategy=AdaptiveMarketMaker(cfg.strategy),
            recent_mids=deque(maxlen=20),
        )
    return runtime_by_market


def _scaled_market_limits(
    cfg: BotConfig,
    market_count: int,
) -> tuple[float, int, int, int, int]:
    count = max(1, int(market_count))
    live_size_scale = _clamp(float(cfg.risk.live_size_scale), 0.10, 1.0)
    base_market_max_position = max(1, cfg.risk.max_position // count)
    base_market_max_order_size = max(1, cfg.risk.max_order_size // count)
    per_market_max_position = max(1, int(round(base_market_max_position * live_size_scale)))
    per_market_max_order_size = max(1, int(round(base_market_max_order_size * live_size_scale)))
    return (
        live_size_scale,
        base_market_max_position,
        base_market_max_order_size,
        per_market_max_position,
        per_market_max_order_size,
    )


def _toxicity_pressure(
    momentum: float,
    volatility: float,
    soft_guard: float,
    hard_guard: float,
) -> float:
    abs_momentum = abs(momentum)
    adaptive_soft = max(soft_guard, volatility * 1.25)
    adaptive_hard = max(hard_guard, adaptive_soft * 1.8)
    if abs_momentum <= adaptive_soft:
        return 0.0
    if abs_momentum >= adaptive_hard:
        return 1.0
    return _clamp(
        (abs_momentum - adaptive_soft) / max(1e-6, adaptive_hard - adaptive_soft),
        0.0,
        1.0,
    )


def _apply_toxicity_size_controls(
    yes_size: int,
    no_size: int,
    position_qty: int,
    momentum: float,
    volatility: float,
    soft_guard: float,
    hard_guard: float,
    adverse_side_size_cut: float,
) -> tuple[int, int, str]:
    if yes_size <= 0 and no_size <= 0:
        return 0, 0, ""

    pressure = _toxicity_pressure(
        momentum=momentum,
        volatility=volatility,
        soft_guard=soft_guard,
        hard_guard=hard_guard,
    )
    if pressure <= 0:
        return yes_size, no_size, ""

    adaptive_soft = max(soft_guard, volatility * 1.25)
    adaptive_hard = max(hard_guard, adaptive_soft * 1.8)
    if abs(momentum) >= adaptive_hard and position_qty == 0:
        return 0, 0, "momentum_hard_guard"

    base_scale = 1.0 - (0.50 * pressure)
    yes_adj = max(0, int(round(yes_size * base_scale)))
    no_adj = max(0, int(round(no_size * base_scale)))

    toxic_side = "no" if momentum > 0 else "yes"
    de_risk_side = "no" if position_qty > 0 else ("yes" if position_qty < 0 else "")
    side_scale = 1.0 - (_clamp(adverse_side_size_cut, 0.0, 0.95) * pressure)

    if toxic_side == "yes" and yes_adj > 0:
        yes_adj = max(0, int(round(yes_adj * side_scale)))
        if de_risk_side == "yes" and yes_size > 0:
            yes_adj = max(1, yes_adj)
    if toxic_side == "no" and no_adj > 0:
        no_adj = max(0, int(round(no_adj * side_scale)))
        if de_risk_side == "no" and no_size > 0:
            no_adj = max(1, no_adj)

    if yes_adj <= 0 and no_adj <= 0:
        return 0, 0, "momentum_size_guard"
    return yes_adj, no_adj, ""


def _market_summary(state: MarketRuntimeState) -> dict[str, Any]:
    return {
        "ticker": state.ticker,
        "cycles": state.cycles,
        "orders_submitted": state.orders_submitted,
        "orders_canceled": state.orders_canceled,
        "skipped_cycles": state.skipped_cycles,
        "max_abs_position": state.max_abs_position,
        "final_position": state.final_position,
        "post_only_reprices": state.post_only_reprices,
        "post_only_rejects": state.post_only_rejects,
        "preclose_flatten_calls": state.preclose_flatten_calls,
        "last_reason": state.last_reason,
        "recent_mid_count": len(state.recent_mids),
    }


def _base36(value: int) -> str:
    if value <= 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    n = value
    while n > 0:
        n, rem = divmod(n, 36)
        out = digits[rem] + out
    return out


def _session_stop_loss_hit(
    start_balance_cents: int,
    current_balance_cents: int,
    stop_loss_cents: int,
) -> bool:
    if stop_loss_cents <= 0:
        return False
    return (start_balance_cents - current_balance_cents) >= stop_loss_cents


def _session_take_profit_hit(
    start_balance_cents: int,
    current_balance_cents: int,
    take_profit_cents: int,
) -> bool:
    if take_profit_cents <= 0:
        return False
    return (current_balance_cents - start_balance_cents) >= take_profit_cents


def _ticker_code(ticker: str) -> str:
    up = ticker.upper()
    if "BTC" in up:
        return "btc"
    if "ETH" in up:
        return "eth"
    if "SOL" in up:
        return "sol"
    alnum = "".join(ch for ch in up if ch.isalnum())
    return (alnum[:3] or "mkt").lower()


def _next_client_order_id(
    run_tag: str,
    ticker: str,
    side_code: str,
    seq: int,
) -> tuple[str, int]:
    next_seq = seq + 1
    oid = f"{run_tag}{_ticker_code(ticker)}{side_code}{next_seq:05d}"
    return oid[:30], next_seq


def _list_open_markets(client: KalshiClient) -> list[dict[str, Any]]:
    open_markets = _list_markets_paginated(client, status="open", limit=300, max_pages=20)
    active_markets = _list_markets_paginated(client, status="active", limit=300, max_pages=20)
    merged = _dedupe_markets(open_markets + active_markets)
    if merged:
        return merged

    all_markets = _list_markets_paginated(client, status=None, limit=300, max_pages=20)
    filtered = [
        m
        for m in all_markets
        if str(m.get("status", "")).strip().lower() in {"open", "active"}
    ]
    if filtered:
        return _dedupe_markets(filtered)

    # Last-resort fallback when status is missing in list payloads.
    return _dedupe_markets(
        [
            m
            for m in all_markets
            if not _is_terminal_status(str(m.get("status", "")).strip().lower())
        ]
    )


def _market_exists(client: KalshiClient, ticker: str) -> bool:
    try:
        payload = client.get_market(ticker)
        market = payload.get("market", payload)
        return isinstance(market, dict) and bool(str(market.get("ticker", "")).strip())
    except KalshiApiError:
        return False


def _resolve_from_series_or_event(
    client: KalshiClient,
    ticker: str,
    min_seconds_to_close: float = 0.0,
    allow_non_open: bool = False,
) -> str:
    now_epoch = time.time()
    event_markets: list[dict[str, Any]] = []
    try:
        event_payload = client.get_event(ticker)
        event = event_payload.get("event", event_payload)
        if isinstance(event, dict):
            event_markets.extend(_event_markets_from_event(event))
            if not event_markets:
                event_markets.extend(_markets_from_event_ticker(client, ticker))
    except KalshiApiError:
        pass
    if event_markets:
        event_markets = _hydrate_markets(client, event_markets, allow_non_open=allow_non_open)
        chosen = _choose_best_market(
            event_markets,
            now_epoch=now_epoch,
            min_seconds_to_close=min_seconds_to_close,
            allow_non_open=allow_non_open,
        )
        if chosen:
            return chosen

    series_markets: list[dict[str, Any]] = []
    try:
        series_payload = client.get_series(ticker)
        series = series_payload.get("series", series_payload)
        if isinstance(series, dict):
            for event in series.get("events", []) or []:
                if isinstance(event, dict):
                    series_markets.extend(_event_markets_from_event(event))
                    ev_ticker = str(event.get("ticker", "")).strip()
                    if ev_ticker:
                        series_markets.extend(_markets_from_event_ticker(client, ev_ticker))
    except KalshiApiError:
        pass

    series_markets.extend(_markets_from_series_ticker(client, ticker))
    if not series_markets:
        for event in _events_from_series_ticker(client, ticker):
            series_markets.extend(_event_markets_from_event(event))
            ev_ticker = str(event.get("ticker", "")).strip()
            if ev_ticker:
                series_markets.extend(_markets_from_event_ticker(client, ev_ticker))

    series_markets = _hydrate_markets(client, series_markets, allow_non_open=allow_non_open)
    chosen = _choose_best_market(
        series_markets,
        now_epoch=now_epoch,
        min_seconds_to_close=min_seconds_to_close,
        allow_non_open=allow_non_open,
    )
    return chosen or ""


def _events_from_series_ticker(client: KalshiClient, series_ticker: str) -> list[dict[str, Any]]:
    for status in ("open", "active", None):
        events = _list_events_paginated(
            client=client,
            series_ticker=series_ticker,
            status=status,
            limit=200,
            max_pages=20,
        )
        if events:
            return events
    return []


def _hydrate_markets(
    client: KalshiClient,
    markets: list[dict[str, Any]],
    allow_non_open: bool = False,
) -> list[dict[str, Any]]:
    hydrated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for market in markets:
        if not isinstance(market, dict):
            continue
        ticker = str(market.get("ticker", "")).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)

        status = str(market.get("status", "")).lower()
        has_touch = ("yes_bid" in market and "yes_ask" in market) or (
            "yes_bid_dollars" in market and "yes_ask_dollars" in market
        )
        if status in {"open", "active"} and has_touch:
            hydrated.append(market)
            continue
        if allow_non_open and status and not _is_terminal_status(status) and has_touch:
            hydrated.append(market)
            continue

        try:
            payload = client.get_market(ticker)
            full = payload.get("market", payload)
            if isinstance(full, dict):
                full_status = str(full.get("status", "")).lower()
                if full_status in {"open", "active"}:
                    hydrated.append(full)
                elif allow_non_open and full_status and not _is_terminal_status(full_status):
                    hydrated.append(full)
        except KalshiApiError:
            continue
    return hydrated


def _list_all_markets(client: KalshiClient) -> list[dict[str, Any]]:
    return _list_markets_paginated(client, status=None, limit=300, max_pages=24)


def _markets_from_series_ticker(client: KalshiClient, series_ticker: str) -> list[dict[str, Any]]:
    for status in ("open", "active", None):
        markets = _list_markets_paginated(
            client,
            status=status,
            series_ticker=series_ticker,
            limit=300,
            max_pages=20,
        )
        if markets:
            return markets
    return []


def _markets_from_event_ticker(client: KalshiClient, event_ticker: str) -> list[dict[str, Any]]:
    for status in ("open", "active", None):
        markets = _list_markets_paginated(
            client,
            status=status,
            event_ticker=event_ticker,
            limit=300,
            max_pages=20,
        )
        if markets:
            return markets
    return []


def _list_markets_paginated(
    client: KalshiClient,
    status: str | None = None,
    event_ticker: str | None = None,
    series_ticker: str | None = None,
    limit: int = 300,
    max_pages: int = 10,
) -> list[dict[str, Any]]:
    for auth in (False, True):
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        cursor: str | None = None
        for _ in range(max(1, max_pages)):
            try:
                payload = client.list_markets(
                    status=status,
                    event_ticker=event_ticker,
                    series_ticker=series_ticker,
                    limit=limit,
                    cursor=cursor,
                    auth=auth,
                )
            except KalshiApiError:
                break

            markets = payload.get("markets", [])
            if isinstance(markets, list):
                for market in markets:
                    if not isinstance(market, dict):
                        continue
                    ticker = str(market.get("ticker", "")).strip().upper()
                    if not ticker or ticker in seen:
                        continue
                    seen.add(ticker)
                    out.append(market)

            next_cursor = _next_page_cursor(payload)
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        if out:
            return out
    return []


def _list_events_paginated(
    client: KalshiClient,
    series_ticker: str,
    status: str | None = None,
    limit: int = 200,
    max_pages: int = 10,
) -> list[dict[str, Any]]:
    for auth in (False, True):
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        cursor: str | None = None
        for _ in range(max(1, max_pages)):
            try:
                payload = client.list_events(
                    series_ticker=series_ticker,
                    status=status,
                    limit=limit,
                    cursor=cursor,
                    auth=auth,
                )
            except KalshiApiError:
                break

            events = payload.get("events", [])
            if isinstance(events, list):
                for event in events:
                    if not isinstance(event, dict):
                        continue
                    ticker = str(event.get("ticker", "")).strip().upper()
                    if not ticker or ticker in seen:
                        continue
                    seen.add(ticker)
                    out.append(event)

            next_cursor = _next_page_cursor(payload)
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        if out:
            return out
    return []


def _next_page_cursor(payload: dict[str, Any]) -> str:
    raw = payload.get("cursor")
    if raw in (None, ""):
        raw = payload.get("next_cursor")
    if raw in (None, ""):
        raw = payload.get("next")
    return str(raw).strip() if raw not in (None, "") else ""


def _dedupe_markets(markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for market in markets:
        if not isinstance(market, dict):
            continue
        ticker = str(market.get("ticker", "")).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(market)
    return out


def _event_markets_from_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    markets = event.get("markets")
    if isinstance(markets, list) and markets:
        out: list[dict[str, Any]] = []
        for m in markets:
            if isinstance(m, dict):
                out.append(m)
            elif isinstance(m, str):
                out.append({"ticker": m})
        return out

    market_tickers = event.get("market_tickers")
    if isinstance(market_tickers, list):
        return [{"ticker": str(t)} for t in market_tickers if str(t).strip()]
    return []


def _resolve_by_symbol_search(
    requested: str,
    open_markets: list[dict[str, Any]],
    min_seconds_to_close: float = 0.0,
    allow_non_open: bool = False,
) -> str:
    if not open_markets:
        return ""
    requested_upper = requested.upper().strip()
    asset_tokens = _asset_tokens_for_requested(requested_upper)
    time_tokens = (
        "15M",
        "15 M",
        "15MIN",
        "15 MIN",
        "15-MIN",
        "15MINUTE",
        "15 MINUTE",
        "FIFTEEN MINUTE",
    )

    candidates: list[dict[str, Any]] = []
    for market in open_markets:
        text = _market_search_text(market)
        if asset_tokens and not any(token in text for token in asset_tokens):
            continue
        if any(token in text for token in time_tokens):
            candidates.append(market)

    if not candidates:
        for market in open_markets:
            text = _market_search_text(market)
            if asset_tokens and any(token in text for token in asset_tokens):
                candidates.append(market)

    chosen = _choose_best_market(
        candidates,
        now_epoch=time.time(),
        min_seconds_to_close=min_seconds_to_close,
        allow_non_open=allow_non_open,
    )
    return chosen or ""


def _asset_tokens_for_requested(requested: str) -> tuple[str, ...]:
    requested = requested.upper()
    if "BTC" in requested or "BITCOIN" in requested:
        return ("BTC", "BITCOIN")
    if "ETH" in requested or "ETHER" in requested or "ETHEREUM" in requested:
        return ("ETH", "ETHER", "ETHEREUM")
    if "SOL" in requested or "SOLANA" in requested:
        return ("SOL", "SOLANA")
    return (requested,)


def _market_search_text(market: dict[str, Any]) -> str:
    parts = [
        str(market.get("ticker", "")),
        str(market.get("title", "")),
        str(market.get("subtitle", "")),
        str(market.get("event_ticker", "")),
        str(market.get("series_ticker", "")),
    ]
    return " ".join(parts).upper()


def _choose_best_market(
    markets: list[dict[str, Any]],
    now_epoch: float | None = None,
    min_seconds_to_close: float = 0.0,
    allow_non_open: bool = False,
) -> str:
    if now_epoch is None:
        now_epoch = time.time()
    best_ticker = ""
    best_score = float("-inf")
    for market in markets:
        if not isinstance(market, dict):
            continue
        ticker = str(market.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        status = str(market.get("status", "")).lower()
        if allow_non_open:
            if _is_terminal_status(status):
                continue
        elif status not in {"open", "active"}:
            continue

        seconds_to_close = _market_seconds_to_close(market, now_epoch=now_epoch)
        if seconds_to_close is not None and seconds_to_close <= 0.0:
            continue
        if seconds_to_close is not None and seconds_to_close <= max(0.0, min_seconds_to_close):
            continue
        if allow_non_open and status not in {"open", "active"}:
            # For non-open fallback, require explicit status + close timestamp so
            # we don't pick stale artifacts with unknown state.
            if not status or seconds_to_close is None:
                continue

        mid, spread = _market_mid_and_spread(market)
        volume = _safe_float(
            market.get("volume")
            or market.get("volume_24h")
            or market.get("open_interest")
            or market.get("liquidity")
        )
        score = 0.0
        if mid is not None:
            score += max(0.0, 1.0 - abs(mid - 0.5) * 2.0) * 8.0
        if spread is not None:
            score += max(0.0, 0.20 - spread) * 12.0
        if volume > 0:
            score += min(4.0, math.log1p(volume))
        if seconds_to_close is not None:
            minutes_to_close = max(0.0, seconds_to_close / 60.0)
            required_minutes = max(0.0, min_seconds_to_close / 60.0)
            target_minutes = max(6.0, required_minutes + 3.0)
            runway_distance = abs(minutes_to_close - target_minutes)
            score += max(0.0, 9.0 - runway_distance)
            if minutes_to_close < 4.0:
                score -= 8.0
            elif minutes_to_close < 6.0:
                score -= 3.0
            elif minutes_to_close > 45.0:
                score -= min(2.0, (minutes_to_close - 45.0) / 20.0)
        if status in {"open", "active"}:
            score += 2.0
        elif allow_non_open:
            score -= 0.5

        if score > best_score:
            best_score = score
            best_ticker = ticker
    return best_ticker


def _is_terminal_status(status: str) -> bool:
    status = (status or "").strip().lower()
    if not status:
        return False
    return status in {
        "closed",
        "finalized",
        "determined",
        "settled",
        "resolved",
        "expired",
        "cancelled",
        "canceled",
        "voided",
    }


def _is_untradeable_reason(reason: str) -> bool:
    reason = (reason or "").strip().lower()
    if not reason:
        return False
    if reason in {"preclose_guard", "entry_cutoff_guard"}:
        return True
    return reason.startswith("status_")


def _market_mid_and_spread(market: dict[str, Any]) -> tuple[float | None, float | None]:
    bid, ask = _extract_yes_bid_ask_prob(market)
    if ask <= bid:
        return None, None
    return (bid + ask) * 0.5, ask - bid


def _market_seconds_to_close(market: dict[str, Any], now_epoch: float) -> float | None:
    close_epoch = _market_close_epoch(market)
    if close_epoch is None:
        return None
    return close_epoch - now_epoch


def _market_close_epoch(market: dict[str, Any]) -> float | None:
    # Prefer explicit timestamp fields if provided by the API.
    for key in (
        "close_time",
        "close_ts",
        "expiration_time",
        "expiration_ts",
        "settlement_time",
        "settlement_ts",
        "end_time",
    ):
        if key not in market:
            continue
        epoch = _to_epoch_seconds(market.get(key))
        if epoch is not None:
            return epoch

    ticker = str(market.get("ticker", "")).strip().upper()
    if not ticker:
        return None
    return _ticker_close_epoch_from_suffix(ticker)


def _to_epoch_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        if num > 1e12:
            return num / 1000.0
        if num > 1e9:
            return num
        return None

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        num = float(text)
        if num > 1e12:
            return num / 1000.0
        if num > 1e9:
            return num
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt_obj = datetime.fromisoformat(text)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        return dt_obj.timestamp()
    except ValueError:
        return None


def _ticker_close_epoch_from_suffix(ticker: str) -> float | None:
    match = re.search(r"-(\d{2}[A-Z]{3}\d{2}\d{4})-\d+$", ticker)
    if not match:
        return None
    stamp = match.group(1)
    try:
        dt_local = datetime.strptime(stamp, "%y%b%d%H%M")
        dt_local = dt_local.replace(tzinfo=ZoneInfo("America/New_York"))
        return dt_local.timestamp()
    except Exception:
        return None


def _cancel_resting_bot_orders(client: KalshiClient, ticker: str, client_order_prefix: str) -> int:
    canceled = 0
    try:
        resp = client.get_orders(ticker=ticker, status="resting", limit=200)
    except KalshiApiError:
        return 0

    orders = resp.get("orders", [])
    for order in orders:
        client_order_id = str(order.get("client_order_id", ""))
        if not client_order_id.startswith(client_order_prefix):
            continue
        order_id = str(order.get("order_id", ""))
        if not order_id:
            continue
        try:
            client.cancel_order(order_id)
            canceled += 1
        except KalshiApiError:
            continue
    return canceled


def _submit_post_only_with_reprice(
    client: KalshiClient,
    payload: dict[str, Any],
    price_key: str,
    max_retries: int = 3,
) -> tuple[bool, str, int]:
    """
    Submit post-only order. If the exchange returns "post only cross",
    move price one tick less aggressive and retry a few times.
    """
    working = dict(payload)
    reprices = 0
    last_error = ""

    for _ in range(max_retries):
        try:
            client.create_order(working)
            return True, "", reprices
        except KalshiApiError as exc:
            last_error = str(exc)
            if "order_already_exists" in last_error.lower():
                # Treat idempotent duplicate as success.
                return True, "", reprices
            if "post only cross" not in last_error.lower():
                return False, last_error, reprices

            current_price = _safe_int(working.get(price_key))
            if current_price <= 1:
                return False, last_error, reprices
            working[price_key] = current_price - 1
            reprices += 1

    return False, last_error or "post_only_retry_exhausted", reprices


def _flatten_positions_on_exit(
    client: KalshiClient,
    resolved_markets: list[str],
    runtime_by_market: dict[str, MarketRuntimeState],
    run_tag: str,
    max_order_size: int,
    order_seq: int,
) -> tuple[int, int]:
    flatten_orders_submitted = 0
    flatten_errors = 0

    for ticker in resolved_markets:
        state = runtime_by_market[ticker]
        submitted, errs = _flatten_single_market_on_exit(
            client=client,
            ticker=ticker,
            state=state,
            run_tag=run_tag,
            max_order_size=max_order_size,
            order_seq_start=order_seq,
        )
        flatten_orders_submitted += submitted
        flatten_errors += errs
        order_seq += max(1, submitted * 2)

    return flatten_orders_submitted, flatten_errors


def _flatten_single_market_on_exit(
    client: KalshiClient,
    ticker: str,
    state: MarketRuntimeState,
    run_tag: str,
    max_order_size: int,
    order_seq_start: int,
) -> tuple[int, int]:
    flatten_orders_submitted = 0
    flatten_errors = 0
    seq = order_seq_start
    attempts = 0
    max_attempts = 8

    while attempts < max_attempts:
        attempts += 1
        try:
            qty = _get_single_position_qty(client, ticker)
        except KalshiApiError:
            flatten_errors += 1
            break
        state.final_position = qty
        if qty == 0:
            break

        _cancel_resting_bot_orders(client, ticker=ticker, client_order_prefix=run_tag)

        count = min(abs(qty), max(1, max_order_size * 2))
        primary_payload, backup_payload, seq = _build_flatten_payloads(
            ticker=ticker,
            qty=qty,
            count=count,
            run_tag=run_tag,
            order_seq=seq,
        )

        submitted = False
        if _submit_flatten_order(client, primary_payload):
            flatten_orders_submitted += 1
            state.orders_submitted += 1
            submitted = True
        elif _submit_flatten_order(client, backup_payload):
            flatten_orders_submitted += 1
            state.orders_submitted += 1
            submitted = True

        if not submitted:
            flatten_errors += 1
            break

        time.sleep(0.35)

    try:
        state.final_position = _get_single_position_qty(client, ticker)
    except KalshiApiError:
        flatten_errors += 1
        return flatten_orders_submitted, flatten_errors
    if state.final_position != 0:
        flatten_errors += 1
    return flatten_orders_submitted, flatten_errors


def _submit_flatten_order(client: KalshiClient, payload: dict[str, Any]) -> bool:
    try:
        client.create_order(payload)
        return True
    except KalshiApiError as exc:
        return "order_already_exists" in str(exc).lower()


def _get_single_position_qty(client: KalshiClient, ticker: str) -> int:
    try:
        payload = client.get_positions(ticker=ticker, limit=20)
        pos_map = _extract_positions(payload)
        return int(pos_map.get(ticker, 0))
    except KalshiApiError:
        payload = client.get_positions(limit=300)
        pos_map = _extract_positions(payload)
        return int(pos_map.get(ticker, 0))


def _build_flatten_payloads(
    ticker: str,
    qty: int,
    count: int,
    run_tag: str,
    order_seq: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    if qty > 0:
        buy_client_id, order_seq = _next_client_order_id(run_tag, ticker, "bn", order_seq)
        primary = _build_buy_no_order_payload(
            ticker=ticker,
            count=count,
            no_price=99,
            client_order_id=buy_client_id,
            post_only=False,
        )
        sell_client_id, order_seq = _next_client_order_id(run_tag, ticker, "sy", order_seq)
        backup = _build_sell_yes_order_payload(
            ticker=ticker,
            count=count,
            yes_price=99,
            client_order_id=sell_client_id,
            post_only=False,
        )
        return primary, backup, order_seq

    buy_client_id, order_seq = _next_client_order_id(run_tag, ticker, "by", order_seq)
    primary = _build_buy_yes_order_payload(
        ticker=ticker,
        count=count,
        yes_price=99,
        client_order_id=buy_client_id,
        post_only=False,
    )
    sell_client_id, order_seq = _next_client_order_id(run_tag, ticker, "sn", order_seq)
    backup = _build_sell_no_order_payload(
        ticker=ticker,
        count=count,
        no_price=99,
        client_order_id=sell_client_id,
        post_only=False,
    )
    return primary, backup, order_seq


def _extract_positions(payload: dict[str, Any]) -> dict[str, int]:
    entries = payload.get("market_positions", [])
    if not isinstance(entries, list):
        return {}

    out: dict[str, int] = {}
    for entry in entries:
        ticker = str(entry.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        raw_position = entry.get("position")
        qty = _safe_int(raw_position)
        out[ticker] = qty
    return out


def _extract_yes_bid_ask_prob(market: dict[str, Any]) -> tuple[float, float]:
    bid_dollars = market.get("yes_bid_dollars")
    ask_dollars = market.get("yes_ask_dollars")
    if bid_dollars is not None and ask_dollars is not None:
        bid = _safe_float(bid_dollars)
        ask = _safe_float(ask_dollars)
        return _clamp(bid, 0.01, 0.98), _clamp(ask, 0.02, 0.99)

    bid_cents = _safe_float(market.get("yes_bid"))
    ask_cents = _safe_float(market.get("yes_ask"))
    if bid_cents > 1.0 or ask_cents > 1.0:
        bid = bid_cents / 100.0
        ask = ask_cents / 100.0
        return _clamp(bid, 0.01, 0.98), _clamp(ask, 0.02, 0.99)
    return _clamp(bid_cents, 0.01, 0.98), _clamp(ask_cents, 0.02, 0.99)


def _build_buy_yes_order_payload(
    ticker: str,
    count: int,
    yes_price: int,
    client_order_id: str,
    post_only: bool = True,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "client_order_id": client_order_id,
        "type": "limit",
        "action": "buy",
        "side": "yes",
        "count": int(count),
        "yes_price": int(_clamp(float(yes_price), 1, 99)),
        "post_only": bool(post_only),
    }


def _build_buy_no_order_payload(
    ticker: str,
    count: int,
    no_price: int,
    client_order_id: str,
    post_only: bool = True,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "client_order_id": client_order_id,
        "type": "limit",
        "action": "buy",
        "side": "no",
        "count": int(count),
        "no_price": int(_clamp(float(no_price), 1, 99)),
        "post_only": bool(post_only),
    }


def _build_sell_yes_order_payload(
    ticker: str,
    count: int,
    yes_price: int,
    client_order_id: str,
    post_only: bool = False,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "client_order_id": client_order_id,
        "type": "limit",
        "action": "sell",
        "side": "yes",
        "count": int(count),
        "yes_price": int(_clamp(float(yes_price), 1, 99)),
        "post_only": bool(post_only),
    }


def _build_sell_no_order_payload(
    ticker: str,
    count: int,
    no_price: int,
    client_order_id: str,
    post_only: bool = False,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "client_order_id": client_order_id,
        "type": "limit",
        "action": "sell",
        "side": "no",
        "count": int(count),
        "no_price": int(_clamp(float(no_price), 1, 99)),
        "post_only": bool(post_only),
    }


def _prob_to_cents(prob: float) -> int:
    return int(round(_clamp(prob, 0.01, 0.99) * 100.0))


def _mid_volatility(mids: deque[float]) -> float:
    if len(mids) < 3:
        return 0.004
    returns = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
    if not returns:
        return 0.004
    mean = sum(returns) / len(returns)
    var = sum((x - mean) ** 2 for x in returns) / len(returns)
    return max(0.001, min(0.08, math.sqrt(var)))


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
