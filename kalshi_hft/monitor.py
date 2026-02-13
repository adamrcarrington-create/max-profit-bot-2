from __future__ import annotations

import asyncio
from collections import deque
import inspect
import json
import statistics
import threading
import time
from typing import Any, Dict, Optional

import requests
import websockets

from .client import KalshiClient, extract_float


class BinanceSpotMonitor:
    """Pulls live reference prices with Coinbase/Kraken primary and Binance fallback."""

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        window_seconds: float = 30.0,
        timeout_seconds: float = 2.0,
        include_perp_reference: bool = True,
        perp_weight: float = 0.5,
        cross_exchange_max_diff_pct: float = 0.001,
        flash_spike_pause_seconds: float = 5.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.symbol = symbol.upper().strip()
        self.window_seconds = max(2.0, float(window_seconds))
        self.timeout_seconds = timeout_seconds
        self.include_perp_reference = bool(include_perp_reference)
        self.perp_weight = max(0.0, min(1.0, float(perp_weight)))
        self.cross_exchange_max_diff_pct = max(0.0, float(cross_exchange_max_diff_pct))
        self.flash_spike_pause_seconds = max(0.0, float(flash_spike_pause_seconds))
        self.session = session or requests.Session()
        self.samples: deque[tuple[float, float]] = deque(maxlen=1200)
        self.last_update_epoch: float = 0.0
        self.last_spot_price: Optional[float] = None
        self.last_perp_price: Optional[float] = None
        self.last_coinbase_price: Optional[float] = None
        self.last_kraken_price: Optional[float] = None
        self.last_cross_exchange_diff_pct: float = 0.0
        self.flash_spike_pause_until_epoch: float = 0.0
        self.last_pause_reason: str = ""
        self.last_reference_source: str = ""
        self.last_reference_error: str = ""

    def poll_once(self) -> float:
        spot_price, source = self._fetch_spot_with_fallback()
        self.last_spot_price = spot_price
        self.last_reference_source = source
        if not self.last_reference_error:
            self.last_reference_error = ""
        ref_price = spot_price

        if self.include_perp_reference and source == "binance_spot":
            try:
                perp_price = self._fetch_binance_price("https://fapi.binance.com/fapi/v1/ticker/price")
                self.last_perp_price = perp_price
                ref_price = (
                    spot_price * (1.0 - self.perp_weight)
                    + perp_price * self.perp_weight
                )
            except Exception:
                # Keep the strategy alive on spot-only if perp endpoint is unavailable.
                self.last_perp_price = None
        else:
            self.last_perp_price = None

        now = time.time()
        self.samples.append((now, ref_price))
        self.last_update_epoch = now
        self._trim(now)
        return ref_price

    def _fetch_spot_with_fallback(self) -> tuple[float, str]:
        errors: list[str] = []
        coinbase_price: Optional[float] = None
        kraken_price: Optional[float] = None

        try:
            coinbase_price = self._fetch_coinbase_price()
            self.last_coinbase_price = coinbase_price
        except Exception as exc:
            self.last_coinbase_price = None
            errors.append(f"coinbase_spot={exc}")

        try:
            kraken_price = self._fetch_kraken_price()
            self.last_kraken_price = kraken_price
        except Exception as exc:
            self.last_kraken_price = None
            errors.append(f"kraken_spot={exc}")

        if coinbase_price is not None and kraken_price is not None:
            median_price = self._median_price([coinbase_price, kraken_price])
            self.last_cross_exchange_diff_pct = _relative_diff_pct(coinbase_price, kraken_price)
            if self.last_cross_exchange_diff_pct > self.cross_exchange_max_diff_pct:
                now = time.time()
                self.flash_spike_pause_until_epoch = max(
                    self.flash_spike_pause_until_epoch,
                    now + self.flash_spike_pause_seconds,
                )
                self.last_pause_reason = (
                    f"cross_exchange_divergence diff_pct={self.last_cross_exchange_diff_pct:.6f} "
                    f"threshold={self.cross_exchange_max_diff_pct:.6f}"
                )
                self.last_reference_error = self.last_pause_reason
            return median_price, "median_coinbase_kraken"

        if coinbase_price is not None:
            self.last_cross_exchange_diff_pct = 0.0
            return coinbase_price, "coinbase_spot"
        if kraken_price is not None:
            self.last_cross_exchange_diff_pct = 0.0
            return kraken_price, "kraken_spot"

        try:
            self.last_cross_exchange_diff_pct = 0.0
            return self._fetch_binance_price("https://api.binance.com/api/v3/ticker/price"), "binance_spot"
        except Exception as exc:
            errors.append(f"binance_spot={exc}")

        self.last_reference_error = " | ".join(errors)[:600]
        raise RuntimeError(f"reference_price_unavailable: {self.last_reference_error}")

    @staticmethod
    def _median_price(values: list[float]) -> float:
        return float(statistics.median(values))

    def _fetch_binance_price(self, url: str) -> float:
        response = self.session.get(
            url,
            params={"symbol": self.symbol},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return float(payload["price"])

    def _fetch_coinbase_price(self) -> float:
        base, _ = _split_symbol(self.symbol)
        product = f"{base}-USD"
        response = self.session.get(
            f"https://api.coinbase.com/v2/prices/{product}/spot",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        amount = payload.get("data", {}).get("amount")
        if amount is None:
            raise RuntimeError("coinbase_missing_amount")
        return float(amount)

    def _fetch_kraken_price(self) -> float:
        base, _ = _split_symbol(self.symbol)
        pair = _kraken_pair(base)
        response = self.session.get(
            "https://api.kraken.com/0/public/Ticker",
            params={"pair": pair},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        errs = payload.get("error", [])
        if errs:
            raise RuntimeError(f"kraken_error={errs}")
        result = payload.get("result", {})
        if not isinstance(result, dict) or not result:
            raise RuntimeError("kraken_missing_result")
        first = next(iter(result.values()))
        if not isinstance(first, dict):
            raise RuntimeError("kraken_invalid_result")
        close = first.get("c", [])
        if not isinstance(close, list) or not close:
            raise RuntimeError("kraken_missing_close")
        return float(close[0])

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds - 5.0
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def last_price(self) -> Optional[float]:
        if not self.samples:
            return None
        return self.samples[-1][1]

    def delta_over_window(self, seconds: Optional[float] = None) -> float:
        if len(self.samples) < 2:
            return 0.0
        lookback = max(1.0, float(seconds or self.window_seconds))
        now = self.samples[-1][0]
        baseline = self.samples[0]
        for sample in self.samples:
            if sample[0] >= now - lookback:
                baseline = sample
                break
        return self.samples[-1][1] - baseline[1]

    def age_seconds(self) -> float:
        if self.last_update_epoch <= 0:
            return float("inf")
        return max(0.0, time.time() - self.last_update_epoch)

    def realized_volatility(self, window_seconds: float = 60.0) -> float:
        if len(self.samples) < 3:
            return 0.0
        lookback = max(5.0, float(window_seconds))
        now = self.samples[-1][0]
        points = [p for t, p in self.samples if t >= now - lookback]
        if len(points) < 3:
            return 0.0
        returns: list[float] = []
        for i in range(1, len(points)):
            prev = points[i - 1]
            curr = points[i]
            if prev <= 0:
                continue
            returns.append((curr - prev) / prev)
        if not returns:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / max(1, len(returns) - 1)
        return var ** 0.5

    def return_over_window(self, window_seconds: float) -> float:
        if len(self.samples) < 2:
            return 0.0
        lookback = max(1.0, float(window_seconds))
        now = self.samples[-1][0]
        baseline = self.samples[0]
        for sample in self.samples:
            if sample[0] >= now - lookback:
                baseline = sample
                break
        if baseline[1] <= 0:
            return 0.0
        return (self.samples[-1][1] - baseline[1]) / baseline[1]

    def pause_remaining_seconds(self) -> float:
        if self.flash_spike_pause_until_epoch <= 0:
            return 0.0
        return max(0.0, self.flash_spike_pause_until_epoch - time.time())

    def flash_spike_pause_active(self) -> bool:
        return self.pause_remaining_seconds() > 0.0


class KalshiOrderbookMonitor:
    """Maintains an in-memory orderbook via Kalshi websocket deltas."""

    def __init__(
        self,
        client: KalshiClient,
        ticker: str,
        reconnect_seconds: float = 1.0,
    ) -> None:
        self.client = client
        self.ticker = ticker
        self.reconnect_seconds = max(0.2, float(reconnect_seconds))
        self._yes_levels: Dict[int, int] = {}
        self._no_levels: Dict[int, int] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.last_error: str = ""
        self.last_message_epoch: float = 0.0
        self.last_change_epoch: float = 0.0
        self._last_book_fingerprint: str = ""

    def set_ticker(self, ticker: str) -> None:
        with self._lock:
            self.ticker = ticker
            self._yes_levels.clear()
            self._no_levels.clear()

    def refresh_from_rest(self) -> None:
        payload = self.client.get_orderbook(self.ticker)
        self._apply_snapshot(payload)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _run_event_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._consume_forever())
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    def _connect_kwargs(self, headers: Dict[str, str]) -> Dict[str, Any]:
        base_kwargs: Dict[str, Any] = {
            "ping_interval": 15,
            "ping_timeout": 15,
            "close_timeout": 2,
            "max_size": 2_000_000,
        }
        signature = inspect.signature(websockets.connect)
        if "additional_headers" in signature.parameters:
            base_kwargs["additional_headers"] = headers
        else:
            base_kwargs["extra_headers"] = headers
        return base_kwargs

    async def _consume_forever(self) -> None:
        while not self._stop_event.is_set():
            candidates = self.client.websocket_header_candidates()
            connected = False
            saw_auth_failure = False
            for candidate in candidates:
                headers = dict(candidate.get("headers") or {})
                mode = str(candidate.get("pss_salt_mode") or "unknown")
                connect_kwargs = self._connect_kwargs(headers)
                try:
                    async with websockets.connect(self.client.ws_url, **connect_kwargs) as ws:
                        connected = True
                        if mode in {"digest", "max"} and mode != self.client.auth.pss_salt_mode:
                            # Persist winning mode for subsequent requests.
                            self.client.auth.pss_salt_mode = mode
                        sub_message = {
                            "id": int(time.time()),
                            "cmd": "subscribe",
                            "params": {
                                "channels": ["orderbook_delta"],
                                "market_ticker": self.ticker,
                            },
                        }
                        await ws.send(json.dumps(sub_message))
                        self.last_error = ""
                        while not self._stop_event.is_set():
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            except asyncio.TimeoutError:
                                continue
                            self.last_message_epoch = time.time()
                            self._handle_ws_message(raw)
                except Exception as exc:
                    err_text = str(exc)
                    self.last_error = f"{err_text} pss_mode={mode}"
                    if "401" in err_text:
                        saw_auth_failure = True
                        continue
                if connected:
                    break
            if not connected:
                if saw_auth_failure and self.client.auth.pss_salt_mode != "max":
                    self.client.auth.pss_salt_mode = "max"
                await asyncio.sleep(self.reconnect_seconds)

    def _handle_ws_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            return
        payload = message.get("msg")
        if not isinstance(payload, dict):
            payload = message.get("data")
        if not isinstance(payload, dict):
            payload = message

        msg_type = str(message.get("type") or payload.get("type") or "").lower()
        if "error" in msg_type:
            self.last_error = (
                str(payload.get("msg") or payload.get("message") or message.get("msg") or message)
            )[:500]
            return
        if "snapshot" in msg_type:
            self._apply_snapshot(payload)
            return
        if "delta" in msg_type:
            self._apply_delta(payload)
            return
        if "yes" in payload or "no" in payload or "orderbook" in payload:
            self._apply_snapshot(payload)

    def _apply_snapshot(self, payload: Dict[str, Any]) -> None:
        orderbook = payload.get("orderbook")
        if isinstance(orderbook, dict):
            yes_levels = orderbook.get("yes") or []
            no_levels = orderbook.get("no") or []
        else:
            yes_levels = payload.get("yes") or []
            no_levels = payload.get("no") or []

        if not isinstance(yes_levels, list):
            yes_levels = []
        if not isinstance(no_levels, list):
            no_levels = []

        yes_dict: Dict[int, int] = {}
        no_dict: Dict[int, int] = {}
        for level in yes_levels:
            parsed = _parse_level(level)
            if parsed is None:
                continue
            price, qty = parsed
            if 1 <= price <= 99 and qty > 0:
                yes_dict[price] = qty
        for level in no_levels:
            parsed = _parse_level(level)
            if parsed is None:
                continue
            price, qty = parsed
            if 1 <= price <= 99 and qty > 0:
                no_dict[price] = qty
        with self._lock:
            self._yes_levels = yes_dict
            self._no_levels = no_dict
            new_fingerprint = _fingerprint_book(self._yes_levels, self._no_levels)
            if new_fingerprint != self._last_book_fingerprint:
                self._last_book_fingerprint = new_fingerprint
                self.last_change_epoch = time.time()

    def _apply_delta(self, payload: Dict[str, Any]) -> None:
        side = str(payload.get("side") or "").lower()
        price = extract_float(payload.get("price"))
        delta = extract_float(payload.get("delta") or payload.get("quantity_delta"))
        if side not in {"yes", "no"} or price is None or delta is None:
            return
        p = int(round(price))
        d = int(round(delta))
        if p < 1 or p > 99:
            return
        with self._lock:
            book = self._yes_levels if side == "yes" else self._no_levels
            new_qty = int(book.get(p, 0) + d)
            if new_qty <= 0:
                book.pop(p, None)
            else:
                book[p] = new_qty
            new_fingerprint = _fingerprint_book(self._yes_levels, self._no_levels)
            if new_fingerprint != self._last_book_fingerprint:
                self._last_book_fingerprint = new_fingerprint
                self.last_change_epoch = time.time()

    def top_of_book(self) -> Dict[str, Optional[int]]:
        with self._lock:
            best_yes_bid = max(self._yes_levels.keys()) if self._yes_levels else None
            best_no_bid = max(self._no_levels.keys()) if self._no_levels else None
            best_yes_bid_qty = int(self._yes_levels.get(best_yes_bid, 0)) if best_yes_bid is not None else 0
            best_no_bid_qty = int(self._no_levels.get(best_no_bid, 0)) if best_no_bid is not None else 0
        best_yes_ask = (100 - best_no_bid) if best_no_bid is not None else None
        best_no_ask = (100 - best_yes_bid) if best_yes_bid is not None else None
        return {
            "best_yes_bid": best_yes_bid,
            "best_no_bid": best_no_bid,
            "best_yes_ask": best_yes_ask,
            "best_no_ask": best_no_ask,
            "best_yes_bid_qty": best_yes_bid_qty,
            "best_no_bid_qty": best_no_bid_qty,
        }

    def implied_yes_probability(self, fallback: float = 0.5) -> float:
        top = self.top_of_book()
        yes_bid = top["best_yes_bid"]
        yes_ask = top["best_yes_ask"]
        if yes_bid is not None and yes_ask is not None:
            mid = (yes_bid + yes_ask) / 2.0
            return max(0.01, min(0.99, mid / 100.0))
        if yes_bid is not None:
            return max(0.01, min(0.99, yes_bid / 100.0))
        if yes_ask is not None:
            return max(0.01, min(0.99, yes_ask / 100.0))
        return fallback

    def book_age_seconds(self) -> float:
        if self.last_message_epoch <= 0:
            return float("inf")
        return max(0.0, time.time() - self.last_message_epoch)

    def book_stale_seconds(self) -> float:
        if self.last_change_epoch <= 0:
            return float("inf")
        return max(0.0, time.time() - self.last_change_epoch)

    def depth(self, levels: int = 5) -> Dict[str, int]:
        lvl = max(1, int(levels))
        with self._lock:
            yes_sorted = sorted(self._yes_levels.items(), key=lambda kv: kv[0], reverse=True)[:lvl]
            no_sorted = sorted(self._no_levels.items(), key=lambda kv: kv[0], reverse=True)[:lvl]
        yes_depth = sum(qty for _, qty in yes_sorted)
        no_depth = sum(qty for _, qty in no_sorted)
        return {
            "yes_depth": yes_depth,
            "no_depth": no_depth,
        }

    def microstructure(self, levels: int = 5) -> Dict[str, Any]:
        top = self.top_of_book()
        yes_bid = top["best_yes_bid"]
        yes_ask = top["best_yes_ask"]
        no_bid = top["best_no_bid"]
        no_ask = top["best_no_ask"]
        best_yes_bid_qty = int(top.get("best_yes_bid_qty") or 0)
        best_no_bid_qty = int(top.get("best_no_bid_qty") or 0)
        spread_yes = None
        spread_no = None
        if yes_bid is not None and yes_ask is not None:
            spread_yes = yes_ask - yes_bid
        if no_bid is not None and no_ask is not None:
            spread_no = no_ask - no_bid

        depth = self.depth(levels=levels)
        yes_depth = depth["yes_depth"]
        no_depth = depth["no_depth"]
        denom = max(1, yes_depth + no_depth)
        imbalance = (yes_depth - no_depth) / denom
        pinned = (
            yes_bid in {1, 99}
            or yes_ask in {1, 99}
            or no_bid in {1, 99}
            or no_ask in {1, 99}
        )
        zero_spread = (spread_yes == 0) or (spread_no == 0)
        return {
            "best_yes_bid": yes_bid,
            "best_yes_ask": yes_ask,
            "best_no_bid": no_bid,
            "best_no_ask": no_ask,
            "best_yes_bid_qty": best_yes_bid_qty,
            "best_no_bid_qty": best_no_bid_qty,
            "spread_yes": spread_yes,
            "spread_no": spread_no,
            "yes_depth": yes_depth,
            "no_depth": no_depth,
            "imbalance": imbalance,
            "pinned_state": bool(pinned),
            "zero_spread_state": bool(zero_spread),
            "book_age_seconds": self.book_age_seconds(),
            "book_stale_seconds": self.book_stale_seconds(),
        }


def _parse_level(level: Any) -> Optional[tuple[int, int]]:
    if isinstance(level, dict):
        price = extract_float(level.get("price"))
        qty = extract_float(level.get("quantity") or level.get("qty") or level.get("count"))
        if price is None or qty is None:
            return None
        return int(round(price)), int(round(qty))
    if isinstance(level, list) and len(level) >= 2:
        price = extract_float(level[0])
        qty = extract_float(level[1])
        if price is None or qty is None:
            return None
        return int(round(price)), int(round(qty))
    return None


def _fingerprint_book(yes_levels: Dict[int, int], no_levels: Dict[int, int]) -> str:
    yes_sorted = sorted(yes_levels.items())
    no_sorted = sorted(no_levels.items())
    return f"yes={yes_sorted}|no={no_sorted}"


def _split_symbol(symbol: str) -> tuple[str, str]:
    upper = symbol.upper().strip()
    for quote in ("USDT", "USDC", "USD"):
        if upper.endswith(quote) and len(upper) > len(quote):
            return upper[: -len(quote)], quote
    return upper, "USD"


def _kraken_pair(base: str) -> str:
    normalized = base.upper().strip()
    if normalized == "BTC":
        return "XBTUSD"
    if normalized == "ETH":
        return "ETHUSD"
    if normalized == "SOL":
        return "SOLUSD"
    return f"{normalized}USD"


def _relative_diff_pct(a: float, b: float) -> float:
    denom = max(1e-12, (abs(a) + abs(b)) / 2.0)
    return abs(a - b) / denom
