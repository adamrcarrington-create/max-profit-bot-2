from __future__ import annotations

import argparse
import csv
from collections import deque
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from kalshi_hft.client import extract_float, extract_timestamp_ms
from kalshi_hft.engine import BotConfig, clamp, parse_ticker_close_epoch_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay backtest harness for Kalshi 15m crypto strategy."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to JSONL or CSV replay file.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.example.json"),
        help="Path to bot config JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/backtest_report.json",
        help="Output path for summary report JSON.",
    )
    return parser.parse_args()


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"replay file not found: {p}")
    rows: list[dict[str, Any]] = []
    if p.suffix.lower() in {".jsonl", ".ndjson"}:
        for line in p.read_text().splitlines():
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    elif p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    else:
        # fallback: treat as newline-delimited JSON
        for line in p.read_text().splitlines():
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def get_value(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def row_timestamp_ms(row: Dict[str, Any]) -> Optional[int]:
    return extract_timestamp_ms(
        get_value(
            row,
            [
                "timestamp_ms",
                "timestamp",
                "ts",
                "time",
                "event_ts",
            ],
        )
    )


def implied_yes_probability(row: Dict[str, Any]) -> Optional[float]:
    yes_bid = extract_float(get_value(row, ["best_yes_bid", "yes_bid"]))
    yes_ask = extract_float(get_value(row, ["best_yes_ask", "yes_ask"]))
    no_bid = extract_float(get_value(row, ["best_no_bid", "no_bid"]))
    no_ask = extract_float(get_value(row, ["best_no_ask", "no_ask"]))

    if yes_ask is None and no_bid is not None:
        yes_ask = 100.0 - no_bid
    if yes_bid is None and no_ask is not None:
        yes_bid = 100.0 - no_ask

    if yes_bid is not None and yes_ask is not None:
        return clamp(((yes_bid + yes_ask) / 2.0) / 100.0, 0.01, 0.99)
    if yes_bid is not None:
        return clamp(yes_bid / 100.0, 0.01, 0.99)
    if yes_ask is not None:
        return clamp(yes_ask / 100.0, 0.01, 0.99)
    return None


def spread_yes(row: Dict[str, Any]) -> Optional[float]:
    yes_bid = extract_float(get_value(row, ["best_yes_bid", "yes_bid"]))
    yes_ask = extract_float(get_value(row, ["best_yes_ask", "yes_ask"]))
    no_bid = extract_float(get_value(row, ["best_no_bid", "no_bid"]))
    if yes_ask is None and no_bid is not None:
        yes_ask = 100.0 - no_bid
    if yes_bid is None or yes_ask is None:
        return None
    return max(0.0, yes_ask - yes_bid)


def best_prices(row: Dict[str, Any]) -> Dict[str, Optional[int]]:
    yes_bid = extract_float(get_value(row, ["best_yes_bid", "yes_bid"]))
    yes_ask = extract_float(get_value(row, ["best_yes_ask", "yes_ask"]))
    no_bid = extract_float(get_value(row, ["best_no_bid", "no_bid"]))
    no_ask = extract_float(get_value(row, ["best_no_ask", "no_ask"]))
    if yes_ask is None and no_bid is not None:
        yes_ask = 100.0 - no_bid
    if no_ask is None and yes_bid is not None:
        no_ask = 100.0 - yes_bid
    return {
        "yes_bid": int(round(yes_bid)) if yes_bid is not None else None,
        "yes_ask": int(round(yes_ask)) if yes_ask is not None else None,
        "no_bid": int(round(no_bid)) if no_bid is not None else None,
        "no_ask": int(round(no_ask)) if no_ask is not None else None,
    }


def seconds_to_close_for_row(row: Dict[str, Any], ts_ms: int) -> Optional[float]:
    direct_seconds = extract_float(get_value(row, ["seconds_to_close"]))
    if direct_seconds is not None:
        return float(direct_seconds)
    ticker = str(get_value(row, ["ticker", "market_ticker"]) or "")
    if not ticker:
        return None
    close_ms = parse_ticker_close_epoch_ms(ticker)
    if close_ms is None:
        return None
    seconds = (close_ms - int(ts_ms)) / 1000.0
    # Ignore clearly invalid parses while still allowing modestly negative values near expiry.
    if seconds < -(6 * 3600) or seconds > (18 * 3600):
        return None
    return float(seconds)


def real_probability(
    *,
    spot: float,
    strike: float,
    direction: str,
    momentum_delta: float,
    seconds_to_close: Optional[float],
    distance_scale_pct: float,
    momentum_full_scale_dollars: float,
    distance_weight: float,
    momentum_weight: float,
    time_to_close_reference_seconds: float,
    min_distance_scale_multiplier: float,
    max_distance_scale_multiplier: float,
) -> float:
    strike_ref = strike if strike > 0 else spot
    base_dist_scale = max(25.0, abs(strike_ref) * distance_scale_pct)
    dist_scale = base_dist_scale
    if seconds_to_close is not None:
        ref_seconds = max(60.0, float(time_to_close_reference_seconds))
        min_mult = max(0.05, float(min_distance_scale_multiplier))
        max_mult = max(min_mult, float(max_distance_scale_multiplier))
        close_seconds = max(1.0, float(seconds_to_close))
        dist_scale *= clamp(math.sqrt(close_seconds / ref_seconds), min_mult, max_mult)
    z = (spot - strike_ref) / dist_scale
    if direction == "down":
        z = -z
    distance_prob = sigmoid(z)
    momentum_move = clamp(momentum_delta / max(1.0, momentum_full_scale_dollars), -0.5, 0.5)
    if direction == "down":
        momentum_move = -momentum_move
    dist_w = max(0.0, float(distance_weight))
    mom_w = max(0.0, float(momentum_weight))
    if dist_w + mom_w <= 1e-9:
        dist_w = 0.7
        mom_w = 0.3
    wsum = dist_w + mom_w
    momentum_ratio = mom_w / wsum
    pred = distance_prob + momentum_ratio * momentum_move
    return clamp(pred, 0.01, 0.99)


def classify_regime(
    spot_window: deque[tuple[float, float]],
    cfg: BotConfig,
) -> dict[str, Any]:
    if len(spot_window) < 4:
        return {"name": "calm", "edge_boost": 0.0, "size_mult": 1.0, "price_cushion": 0}
    prices = [p for _, p in spot_window]
    returns: list[float] = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        curr = prices[i]
        if prev <= 0:
            continue
        returns.append((curr - prev) / prev)
    if not returns:
        return {"name": "calm", "edge_boost": 0.0, "size_mult": 1.0, "price_cushion": 0}
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / max(1, len(returns) - 1)
    vol = var ** 0.5
    momentum = prices[-1] - prices[0]
    momentum_pct = abs(momentum) / max(1.0, prices[-1])
    if vol >= cfg.strategy.spiky_vol_threshold or momentum_pct >= 2.0 * cfg.strategy.trending_momentum_threshold:
        return {"name": "spiky", "edge_boost": 0.02, "size_mult": 0.45, "price_cushion": 2}
    if momentum_pct >= cfg.strategy.trending_momentum_threshold or vol >= cfg.strategy.calm_vol_threshold * 1.5:
        return {"name": "trending", "edge_boost": 0.01, "size_mult": 0.70, "price_cushion": 1}
    return {"name": "calm", "edge_boost": 0.0, "size_mult": 1.0, "price_cushion": 0}


def run_backtest(cfg: BotConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        ts = row_timestamp_ms(row)
        ref_price = extract_float(
            get_value(row, ["ref_price", "spot_price", "binance_price", "underlying_price"])
        )
        if ts is None or ref_price is None:
            continue
        clean_rows.append({"ts_ms": ts, "ref": ref_price, "raw": row})
    clean_rows.sort(key=lambda r: r["ts_ms"])
    if not clean_rows:
        raise RuntimeError("no usable replay rows: expected timestamp + reference price fields")

    strike = extract_float(
        get_value(clean_rows[0]["raw"], ["strike", "strike_price", "target", "target_price"])
    )
    if strike is None:
        strike = clean_rows[0]["ref"]
    direction = str(get_value(clean_rows[0]["raw"], ["direction"]) or "up").lower()

    position_yes = 0
    position_no = 0
    cash_cents = 0.0
    fees_cents = 0.0
    slippage_cents = 0.0
    signals = 0
    fills = 0
    signal_edge_sum = 0.0
    signal_abs_edge_sum = 0.0
    signal_positive_count = 0
    signal_negative_count = 0
    expected_raw_ev_signal_cents = 0.0
    expected_net_ev_signal_cents = 0.0
    expected_raw_ev_fill_cents = 0.0
    expected_net_ev_fill_cents = 0.0
    pair_arb_signals = 0
    pair_arb_fills = 0
    pair_arb_expected_ev_cents = 0.0
    trade_events: list[dict[str, Any]] = []
    regime_stats: dict[str, dict[str, float]] = {}
    spot_window: deque[tuple[float, float]] = deque(maxlen=1200)
    flat_side_toggle = False

    for idx, row in enumerate(clean_rows):
        ts_sec = row["ts_ms"] / 1000.0
        ref = float(row["ref"])
        raw = row["raw"]
        seconds_to_close = seconds_to_close_for_row(raw, row["ts_ms"])
        spot_window.append((ts_sec, ref))
        while spot_window and spot_window[0][0] < ts_sec - cfg.strategy.vol_window_long_seconds:
            spot_window.popleft()

        implied_yes = implied_yes_probability(raw)
        if implied_yes is None:
            continue
        prices = best_prices(raw)
        momentum_delta = ref - spot_window[0][1]
        regime = classify_regime(spot_window, cfg)
        model_yes = real_probability(
            spot=ref,
            strike=float(strike),
            direction=direction,
            momentum_delta=momentum_delta,
            seconds_to_close=seconds_to_close,
            distance_scale_pct=cfg.strategy.distance_scale_pct,
            momentum_full_scale_dollars=cfg.strategy.momentum_full_scale_dollars,
            distance_weight=cfg.strategy.distance_weight,
            momentum_weight=cfg.strategy.momentum_weight,
            time_to_close_reference_seconds=cfg.strategy.time_to_close_reference_seconds,
            min_distance_scale_multiplier=cfg.strategy.min_distance_scale_multiplier,
            max_distance_scale_multiplier=cfg.strategy.max_distance_scale_multiplier,
        )
        model_mult = clamp(float(cfg.strategy.model_edge_multiplier), 0.0, 1.0)
        real_yes = clamp(
            implied_yes + (model_yes - implied_yes) * model_mult,
            0.01,
            0.99,
        )
        edge = real_yes - implied_yes
        spread = spread_yes(raw)
        garbage = spread == 0 or prices["yes_bid"] in {1, 99} or prices["yes_ask"] in {1, 99}
        edge_threshold = cfg.strategy.edge_threshold + float(regime["edge_boost"])
        if garbage:
            edge_threshold += cfg.strategy.garbage_state_edge_boost

        if cfg.strategy.cross_side_arb_enabled:
            pair_mode = ""
            pair_count = 0
            pair_raw_ev = 0.0
            pair_ev = float("-inf")
            pair_yes_price: Optional[int] = None
            pair_no_price: Optional[int] = None

            max_pair_count = max(1, int(cfg.strategy.cross_side_arb_max_count))
            per_leg_cost = float(cfg.strategy.taker_fee_cents_per_contract) + float(
                cfg.strategy.taker_slippage_cents_per_contract
            )
            min_pair_edge = float(cfg.strategy.cross_side_arb_min_edge_cents)

            if prices["yes_ask"] is not None and prices["no_ask"] is not None:
                gross_room = max(0, int(cfg.risk.max_gross_position - (position_yes + position_no)))
                buy_count = min(max_pair_count, gross_room // 2)
                if buy_count > 0:
                    buy_raw_ev = 100.0 - float(prices["yes_ask"] + prices["no_ask"])
                    buy_ev = buy_raw_ev - 2.0 * per_leg_cost
                    if buy_ev >= min_pair_edge:
                        pair_mode = "pair_buy"
                        pair_count = buy_count
                        pair_raw_ev = buy_raw_ev
                        pair_ev = buy_ev
                        pair_yes_price = int(prices["yes_ask"])
                        pair_no_price = int(prices["no_ask"])

            if (
                cfg.strategy.cross_side_arb_allow_sell_pair
                and prices["yes_bid"] is not None
                and prices["no_bid"] is not None
            ):
                sell_count = min(max_pair_count, position_yes, position_no)
                if sell_count > 0:
                    sell_raw_ev = float(prices["yes_bid"] + prices["no_bid"]) - 100.0
                    sell_ev = sell_raw_ev - 2.0 * per_leg_cost
                    if sell_ev >= min_pair_edge and sell_ev > pair_ev:
                        pair_mode = "pair_sell"
                        pair_count = sell_count
                        pair_raw_ev = sell_raw_ev
                        pair_ev = sell_ev
                        pair_yes_price = int(prices["yes_bid"])
                        pair_no_price = int(prices["no_bid"])

            if pair_mode and pair_count > 0 and pair_yes_price is not None and pair_no_price is not None:
                pair_arb_signals += 1
                pair_arb_fills += 2
                pair_arb_expected_ev_cents += pair_ev * pair_count
                fills += 2

                pair_fee = 2.0 * float(cfg.strategy.taker_fee_cents_per_contract) * pair_count
                pair_slip = 2.0 * float(cfg.strategy.taker_slippage_cents_per_contract) * pair_count
                fees_cents += pair_fee
                slippage_cents += pair_slip

                if pair_mode == "pair_buy":
                    cash_cents -= (float(pair_yes_price + pair_no_price) * pair_count + pair_fee + pair_slip)
                    position_yes += pair_count
                    position_no += pair_count
                    action = "buy"
                else:
                    cash_cents += (float(pair_yes_price + pair_no_price) * pair_count - pair_fee - pair_slip)
                    position_yes -= pair_count
                    position_no -= pair_count
                    action = "sell"

                leg_fee = pair_fee / 2.0
                leg_slip = pair_slip / 2.0
                trade_events.append(
                    {
                        "idx": idx,
                        "ts_ms": row["ts_ms"],
                        "side": "yes",
                        "mode": "pair_arb",
                        "action": action,
                        "count": pair_count,
                        "ref": ref,
                        "edge": pair_raw_ev / 100.0,
                        "exec_price": pair_yes_price,
                        "target_price": pair_yes_price,
                        "fill_fee": leg_fee,
                        "fill_slippage": leg_slip,
                    }
                )
                trade_events.append(
                    {
                        "idx": idx,
                        "ts_ms": row["ts_ms"],
                        "side": "no",
                        "mode": "pair_arb",
                        "action": action,
                        "count": pair_count,
                        "ref": ref,
                        "edge": pair_raw_ev / 100.0,
                        "exec_price": pair_no_price,
                        "target_price": pair_no_price,
                        "fill_fee": leg_fee,
                        "fill_slippage": leg_slip,
                    }
                )
                continue

        if cfg.strategy.inventory_exit_enabled:
            over_ratio = clamp(float(cfg.strategy.inventory_reduce_over_limit_ratio), 0.1, 1.0)
            over_limit = (
                (position_yes + position_no) >= int(cfg.risk.max_gross_position * over_ratio)
                or abs(position_yes - position_no) >= int(cfg.risk.max_net_exposure * over_ratio)
            )
            min_exit_edge = float(cfg.strategy.inventory_exit_min_edge_cents)
            if over_limit:
                min_exit_edge = min(min_exit_edge, -1.0)
            per_leg_cost = float(cfg.strategy.taker_fee_cents_per_contract) + float(
                cfg.strategy.taker_slippage_cents_per_contract
            )
            max_exit_count = max(1, int(cfg.strategy.inventory_exit_max_count))

            yes_bid = prices.get("yes_bid")
            if position_yes > 0 and yes_bid is not None:
                fair_hold_yes = real_yes * 100.0
                exit_edge = float(yes_bid) - per_leg_cost - fair_hold_yes
                if exit_edge >= min_exit_edge:
                    exit_count = min(position_yes, max_exit_count)
                    fills += 1
                    fill_fee = float(cfg.strategy.taker_fee_cents_per_contract) * exit_count
                    fill_slip = float(cfg.strategy.taker_slippage_cents_per_contract) * exit_count
                    fees_cents += fill_fee
                    slippage_cents += fill_slip
                    cash_cents += float(yes_bid) * exit_count - fill_fee - fill_slip
                    position_yes -= exit_count
                    trade_events.append(
                        {
                            "idx": idx,
                            "ts_ms": row["ts_ms"],
                            "side": "yes",
                            "mode": "inventory_exit",
                            "action": "sell",
                            "count": exit_count,
                            "ref": ref,
                            "edge": exit_edge / 100.0,
                            "exec_price": int(yes_bid),
                            "target_price": int(yes_bid),
                            "fill_fee": fill_fee,
                            "fill_slippage": fill_slip,
                        }
                    )

            no_bid = prices.get("no_bid")
            if position_no > 0 and no_bid is not None:
                fair_hold_no = (1.0 - real_yes) * 100.0
                exit_edge = float(no_bid) - per_leg_cost - fair_hold_no
                if exit_edge >= min_exit_edge:
                    exit_count = min(position_no, max_exit_count)
                    fills += 1
                    fill_fee = float(cfg.strategy.taker_fee_cents_per_contract) * exit_count
                    fill_slip = float(cfg.strategy.taker_slippage_cents_per_contract) * exit_count
                    fees_cents += fill_fee
                    slippage_cents += fill_slip
                    cash_cents += float(no_bid) * exit_count - fill_fee - fill_slip
                    position_no -= exit_count
                    trade_events.append(
                        {
                            "idx": idx,
                            "ts_ms": row["ts_ms"],
                            "side": "no",
                            "mode": "inventory_exit",
                            "action": "sell",
                            "count": exit_count,
                            "ref": ref,
                            "edge": exit_edge / 100.0,
                            "exec_price": int(no_bid),
                            "target_price": int(no_bid),
                            "fill_fee": fill_fee,
                            "fill_slippage": fill_slip,
                        }
                    )

        if abs(edge) < edge_threshold:
            continue

        signals += 1
        signal_edge_sum += edge
        signal_abs_edge_sum += abs(edge)
        if edge >= 0:
            signal_positive_count += 1
        else:
            signal_negative_count += 1
        regime_bucket = regime_stats.setdefault(
            str(regime["name"]),
            {
                "signals": 0.0,
                "fills": 0.0,
                "expected_ev_cents": 0.0,
            },
        )
        regime_bucket["signals"] += 1.0

        if edge > 1e-9:
            side = "yes"
        elif edge < -1e-9:
            side = "no"
        else:
            net = position_yes - position_no
            if net > 0:
                side = "no"
            elif net < 0:
                side = "yes"
            else:
                flat_side_toggle = not flat_side_toggle
                side = "yes" if flat_side_toggle else "no"
        mode = "snipe" if (abs(edge) >= cfg.strategy.snipe_edge_threshold and not garbage) else "mm"
        spread_limit = (
            float(cfg.strategy.max_yes_spread_cents_for_snipe)
            if mode == "snipe"
            else float(cfg.strategy.max_yes_spread_cents_for_mm)
        )
        if spread_limit > 0 and spread is not None and float(spread) > spread_limit:
            continue
        if mode == "snipe":
            if side == "yes" and prices["yes_ask"] is None:
                continue
            if side == "no" and prices["no_ask"] is None:
                continue
        size_mult = float(regime["size_mult"])
        fair_yes = int(round(real_yes * 100.0))
        fair_no = 100 - fair_yes
        mm_quotes = mm_quotes_for_backtest(
            fair_yes=fair_yes,
            fair_no=fair_no,
            net_yes=position_yes - position_no,
            half_spread_cents=max(0.5, float(cfg.strategy.mm_half_spread_cents)) + float(regime["price_cushion"]),
            inventory_skew_cents_per_contract=float(cfg.strategy.inventory_skew_cents_per_contract),
        )
        target = int(mm_quotes["yes_bid"] if side == "yes" else mm_quotes["no_bid"])
        if mode == "snipe":
            if side == "yes" and prices["yes_ask"] is not None:
                target = max(target, int(prices["yes_ask"]))
            if side == "no" and prices["no_ask"] is not None:
                target = max(target, int(prices["no_ask"]))
        target = int(clamp(target, 1, 99))
        count = int(
            max(
                1,
                round(
                    (
                        cfg.strategy.base_order_count
                        + (cfg.strategy.max_order_count - cfg.strategy.base_order_count)
                        * clamp(abs(edge) / max(0.0001, cfg.strategy.max_edge_for_sizing), 0.0, 1.0)
                    )
                    * size_mult
                ),
            )
        )
        if cfg.strategy.use_fractional_kelly:
            count = min(
                count,
                kelly_contract_count(
                    side=side,
                    real_yes=real_yes,
                    price_cents=target,
                    regime_size_multiplier=size_mult,
                    kelly_fraction=cfg.strategy.kelly_fraction,
                    kelly_bankroll_contracts=cfg.strategy.kelly_bankroll_contracts,
                ),
            )
        if count <= 0:
            continue

        # Enforce replay-side inventory limits so simulated sizing matches live risk policy.
        gross_limit = max(0, int(cfg.risk.max_gross_position))
        net_limit = max(0, int(cfg.risk.max_net_exposure))
        gross_now = position_yes + position_no
        net_now = position_yes - position_no
        gross_room = max(0, gross_limit - gross_now)
        if side == "yes":
            net_room = max(0, net_limit - net_now)
        else:
            net_room = max(0, net_limit + net_now)
        count = min(count, gross_room, net_room)
        if count <= 0:
            continue

        if side == "yes":
            raw_ev = real_yes * 100.0 - target
        else:
            raw_ev = (1.0 - real_yes) * 100.0 - target
        fee = cfg.strategy.taker_fee_cents_per_contract if mode == "snipe" else cfg.strategy.maker_fee_cents_per_contract
        slip = cfg.strategy.taker_slippage_cents_per_contract if mode == "snipe" else cfg.strategy.maker_slippage_cents_per_contract
        ev = raw_ev - fee - slip
        expected_raw_ev_signal_cents += raw_ev * count
        expected_net_ev_signal_cents += ev * count
        if ev < cfg.strategy.min_ev_cents_per_contract:
            continue
        regime_bucket["expected_ev_cents"] += ev * count

        executed = False
        exec_price = target
        if side == "yes":
            if mode == "snipe" and prices["yes_ask"] is not None and target >= prices["yes_ask"]:
                executed = True
                exec_price = int(prices["yes_ask"])
            elif mode == "mm" and prices["yes_bid"] is not None and target >= prices["yes_bid"]:
                executed = deterministic_fill(idx, probability=0.35 if (spread or 0) >= 2.0 else 0.20)
                exec_price = int(target)
        else:
            if mode == "snipe" and prices["no_ask"] is not None and target >= prices["no_ask"]:
                executed = True
                exec_price = int(prices["no_ask"])
            elif mode == "mm" and prices["no_bid"] is not None and target >= prices["no_bid"]:
                executed = deterministic_fill(idx + 11, probability=0.35 if (spread or 0) >= 2.0 else 0.20)
                exec_price = int(target)

        if not executed:
            continue

        fills += 1
        regime_bucket["fills"] += 1.0
        fill_fee = fee * count
        fill_slip = abs(exec_price - target) * count + (slip * count)
        fees_cents += fill_fee
        slippage_cents += fill_slip
        expected_raw_ev_fill_cents += raw_ev * count
        expected_net_ev_fill_cents += ev * count

        if side == "yes":
            position_yes += count
            cash_cents -= (exec_price * count + fill_fee + fill_slip)
        else:
            position_no += count
            cash_cents -= (exec_price * count + fill_fee + fill_slip)

        trade_events.append(
            {
                "idx": idx,
                "ts_ms": row["ts_ms"],
                "side": side,
                "mode": mode,
                "action": "buy",
                "count": count,
                "ref": ref,
                "edge": edge,
                "exec_price": exec_price,
                "target_price": target,
                "fill_fee": fill_fee,
                "fill_slippage": fill_slip,
            }
        )

    final_implied = implied_yes_probability(clean_rows[-1]["raw"])
    if final_implied is None:
        final_implied = 0.5
    final_yes_mark = final_implied * 100.0
    final_no_mark = 100.0 - final_yes_mark
    mark_to_market = cash_cents + position_yes * final_yes_mark + position_no * final_no_mark

    mode_attribution: dict[str, dict[str, float]] = {
        "mm": {
            "fills": 0.0,
            "contracts": 0.0,
            "raw_pnl_before_costs_cents": 0.0,
            "fees_cents": 0.0,
            "slippage_cents": 0.0,
            "net_pnl_cents": 0.0,
        },
        "snipe": {
            "fills": 0.0,
            "contracts": 0.0,
            "raw_pnl_before_costs_cents": 0.0,
            "fees_cents": 0.0,
            "slippage_cents": 0.0,
            "net_pnl_cents": 0.0,
        },
    }
    raw_pnl_before_costs_cents = 0.0
    for evt in trade_events:
        side = str(evt.get("side") or "")
        action = str(evt.get("action") or "buy").lower()
        mode = str(evt.get("mode") or "mm")
        count = float(evt.get("count") or 0.0)
        exec_price = float(evt.get("exec_price") or 0.0)
        fill_fee = float(evt.get("fill_fee") or 0.0)
        fill_slip = float(evt.get("fill_slippage") or 0.0)
        if side == "yes":
            if action == "sell":
                raw_pnl = (exec_price - final_yes_mark) * count
            else:
                raw_pnl = (final_yes_mark - exec_price) * count
        elif side == "no":
            if action == "sell":
                raw_pnl = (exec_price - final_no_mark) * count
            else:
                raw_pnl = (final_no_mark - exec_price) * count
        else:
            continue
        net_pnl = raw_pnl - fill_fee - fill_slip
        raw_pnl_before_costs_cents += raw_pnl
        bucket = mode_attribution.setdefault(
            mode,
            {
                "fills": 0.0,
                "contracts": 0.0,
                "raw_pnl_before_costs_cents": 0.0,
                "fees_cents": 0.0,
                "slippage_cents": 0.0,
                "net_pnl_cents": 0.0,
            },
        )
        bucket["fills"] += 1.0
        bucket["contracts"] += count
        bucket["raw_pnl_before_costs_cents"] += raw_pnl
        bucket["fees_cents"] += fill_fee
        bucket["slippage_cents"] += fill_slip
        bucket["net_pnl_cents"] += net_pnl

    adverse_count = 0
    adverse_total = 0
    for evt in trade_events:
        if str(evt.get("action") or "buy").lower() != "buy":
            continue
        fill_sec = evt["ts_ms"] / 1000.0
        fill_ref = evt["ref"]
        future = spot_after(clean_rows, fill_sec, cfg.strategy.adverse_selection_horizon_seconds)
        if future is None:
            continue
        adverse_total += 1
        if evt["side"] == "yes" and future <= fill_ref - cfg.strategy.adverse_selection_move_threshold:
            adverse_count += 1
        if evt["side"] == "no" and future >= fill_ref + cfg.strategy.adverse_selection_move_threshold:
            adverse_count += 1

    per_regime = {}
    for name, stats in regime_stats.items():
        sig = stats["signals"]
        fil = stats["fills"]
        per_regime[name] = {
            "signals": int(sig),
            "fills": int(fil),
            "fill_rate": (fil / sig) if sig else 0.0,
            "expected_ev_cents": round(stats["expected_ev_cents"], 4),
        }

    rounded_mode_attribution: dict[str, dict[str, float]] = {}
    for mode, stats in mode_attribution.items():
        rounded_mode_attribution[mode] = {
            "fills": int(stats["fills"]),
            "contracts": round(stats["contracts"], 4),
            "raw_pnl_before_costs_cents": round(stats["raw_pnl_before_costs_cents"], 4),
            "fees_cents": round(stats["fees_cents"], 4),
            "slippage_cents": round(stats["slippage_cents"], 4),
            "net_pnl_cents": round(stats["net_pnl_cents"], 4),
        }

    signal_edge_mean = (signal_edge_sum / signals) if signals else 0.0
    signal_edge_abs_mean = (signal_abs_edge_sum / signals) if signals else 0.0

    return {
        "rows_total": len(clean_rows),
        "signals": signals,
        "fills": fills,
        "fill_rate": (fills / signals) if signals else 0.0,
        "pnl_cents_mark_to_market": round(mark_to_market, 4),
        "fees_cents": round(fees_cents, 4),
        "slippage_cents": round(slippage_cents, 4),
        "raw_pnl_before_costs_cents": round(raw_pnl_before_costs_cents, 4),
        "cost_drag_cents": round(fees_cents + slippage_cents, 4),
        "adverse_selection_count": adverse_count,
        "adverse_selection_total": adverse_total,
        "adverse_selection_rate": (adverse_count / adverse_total) if adverse_total else 0.0,
        "signal_quality": {
            "mean_edge": round(signal_edge_mean, 6),
            "mean_abs_edge": round(signal_edge_abs_mean, 6),
            "positive_edge_signals": signal_positive_count,
            "negative_edge_signals": signal_negative_count,
            "expected_raw_ev_signal_cents": round(expected_raw_ev_signal_cents, 4),
            "expected_net_ev_signal_cents": round(expected_net_ev_signal_cents, 4),
            "expected_raw_ev_fill_cents": round(expected_raw_ev_fill_cents, 4),
            "expected_net_ev_fill_cents": round(expected_net_ev_fill_cents, 4),
        },
        "pair_arb": {
            "signals": pair_arb_signals,
            "fills": pair_arb_fills,
            "expected_ev_cents": round(pair_arb_expected_ev_cents, 4),
        },
        "mode_attribution": rounded_mode_attribution,
        "ending_inventory": {
            "yes": position_yes,
            "no": position_no,
            "gross": position_yes + position_no,
            "net": position_yes - position_no,
        },
        "per_regime": per_regime,
    }


def deterministic_fill(seed: int, probability: float) -> bool:
    p = int(clamp(probability, 0.0, 1.0) * 1000)
    mixed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
    return (mixed % 1000) < p


def mm_quotes_for_backtest(
    *,
    fair_yes: int,
    fair_no: int,
    net_yes: int,
    half_spread_cents: float,
    inventory_skew_cents_per_contract: float,
) -> Dict[str, int]:
    skew = float(net_yes) * float(inventory_skew_cents_per_contract)
    half = max(0.5, float(half_spread_cents))
    yes_bid = int(round(float(fair_yes) - half - skew))
    yes_ask = int(round(float(fair_yes) + half - skew))
    no_bid = int(round(float(fair_no) - half + skew))
    no_ask = int(round(float(fair_no) + half + skew))
    yes_bid = int(clamp(yes_bid, 1, 99))
    yes_ask = int(clamp(yes_ask, 1, 99))
    no_bid = int(clamp(no_bid, 1, 99))
    no_ask = int(clamp(no_ask, 1, 99))
    if yes_ask <= yes_bid:
        yes_ask = min(99, yes_bid + 1)
    if no_ask <= no_bid:
        no_ask = min(99, no_bid + 1)
    return {
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
    }


def kelly_contract_count(
    *,
    side: str,
    real_yes: float,
    price_cents: int,
    regime_size_multiplier: float,
    kelly_fraction: float,
    kelly_bankroll_contracts: float,
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
    fraction = clamp(float(kelly_fraction), 0.0, 1.0)
    bankroll_contracts = max(1.0, float(kelly_bankroll_contracts))
    size = full_kelly * fraction * bankroll_contracts * max(0.25, float(regime_size_multiplier))
    return max(0, int(round(size)))


def spot_after(rows: list[dict[str, Any]], fill_sec: float, horizon_sec: float) -> Optional[float]:
    target = fill_sec + max(0.0, horizon_sec)
    for row in rows:
        ts = row["ts_ms"] / 1000.0
        if ts >= target:
            return float(row["ref"])
    return None


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def main() -> int:
    args = parse_args()
    cfg = BotConfig.from_json(args.config)
    rows = load_rows(args.input)
    report = run_backtest(cfg, rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"report={out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
