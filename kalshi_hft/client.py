from __future__ import annotations

import base64
from collections import deque
import datetime as dt
from dataclasses import dataclass
import json
from pathlib import Path
import time
from threading import Lock
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlsplit
import uuid

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import requests


class KalshiAuthError(RuntimeError):
    """Raised when authentication fails."""


class KalshiRequestError(RuntimeError):
    """Raised when a Kalshi request fails."""


@dataclass
class KalshiAuthConfig:
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    api_key_id: str = ""
    private_key_path: str = ""
    email: str = ""
    password: str = ""
    token_refresh_seconds: int = 25 * 60
    timeout_seconds: float = 5.0
    pss_salt_mode: str = "digest"
    max_requests_per_second: float = 8.0
    rate_limit_max_retries: int = 4
    rate_limit_backoff_base_seconds: float = 0.35
    rate_limit_backoff_cap_seconds: float = 4.0


class KalshiClient:
    """
    Raw Kalshi API wrapper.

    Auth behavior:
    - Primary: RSA request signing with API key headers.
    - Optional: Legacy token login auto-refresh every ~30 minutes (or on 401).
    """

    def __init__(
        self,
        auth: KalshiAuthConfig,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.auth = auth
        self.base_url = self._normalize_base_url(auth.base_url)
        self.ws_url = auth.ws_url.strip() or "wss://api.elections.kalshi.com/trade-api/ws/v2"
        self.session = session or requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self.private_key: Optional[rsa.RSAPrivateKey] = None
        if auth.private_key_path:
            self.private_key = self._load_private_key(auth.private_key_path)
        if bool(auth.api_key_id) != bool(self.private_key):
            raise ValueError("Set both api_key_id and private_key_path together for RSA auth.")

        self._token_lock = Lock()
        self._token: Optional[str] = None
        self._member_id: Optional[str] = None
        self._token_expiry_epoch: float = 0.0
        self._token_enabled = bool(auth.email and auth.password)
        self._request_timestamps: deque[float] = deque()
        self._last_request_epoch: float = 0.0
        self._rate_limit_wait_seconds_total: float = 0.0
        self._rate_limit_wait_count: int = 0
        self._last_429_epoch: float = 0.0
        self._last_rate_limit_headers: Dict[str, Any] = {}

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        cleaned = base_url.strip().rstrip("/")
        if not cleaned:
            return "https://api.elections.kalshi.com/trade-api/v2"
        if cleaned.endswith("/trade-api/v2"):
            return cleaned
        return f"{cleaned}/trade-api/v2"

    @staticmethod
    def _load_private_key(file_path: str) -> rsa.RSAPrivateKey:
        key_bytes = Path(file_path).read_bytes()
        return serialization.load_pem_private_key(key_bytes, password=None)

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def _sign(
        self,
        timestamp_ms: str,
        method: str,
        path: str,
        pss_salt_mode: Optional[str] = None,
    ) -> str:
        if self.private_key is None:
            raise KalshiAuthError("Missing private key for RSA signing.")
        message = f"{timestamp_ms}{method.upper()}{path.split('?')[0]}".encode("utf-8")
        mode = (pss_salt_mode or self.auth.pss_salt_mode or "digest").strip().lower()
        if mode == "max":
            salt_length = padding.PSS.MAX_LENGTH
        else:
            salt_length = padding.PSS.DIGEST_LENGTH
        signature = self.private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=salt_length),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _signed_headers(
        self,
        method: str,
        path: str,
        pss_salt_mode: Optional[str] = None,
    ) -> Dict[str, str]:
        if not self.auth.api_key_id or self.private_key is None:
            return {}
        timestamp_ms = str(self._now_ms())
        signature = self._sign(
            timestamp_ms=timestamp_ms,
            method=method,
            path=path,
            pss_salt_mode=pss_salt_mode,
        )
        return {
            "KALSHI-ACCESS-KEY": self.auth.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }

    def _token_valid(self) -> bool:
        return bool(self._token) and time.time() < self._token_expiry_epoch

    def ensure_authenticated(self) -> None:
        if not self._token_enabled:
            return
        if self._token_valid():
            return
        self._refresh_legacy_token(force=True)

    def _refresh_legacy_token(self, force: bool = False) -> None:
        if not self._token_enabled:
            return
        with self._token_lock:
            if not force and self._token_valid():
                return

            login_payload = {
                "email": self.auth.email,
                "password": self.auth.password,
            }
            login_paths = ["/login", "/log_in"]
            errors: list[str] = []
            for login_path in login_paths:
                url = f"{self.base_url}{login_path}"
                try:
                    response = self.session.post(
                        url,
                        json=login_payload,
                        timeout=self.auth.timeout_seconds,
                    )
                except requests.RequestException as exc:
                    errors.append(f"{login_path}: network_error={exc}")
                    continue
                if response.status_code in (404, 405):
                    errors.append(f"{login_path}: not_supported status={response.status_code}")
                    continue
                if response.status_code >= 400:
                    body = response.text[:300]
                    errors.append(f"{login_path}: status={response.status_code} body={body}")
                    continue

                try:
                    payload = response.json()
                except ValueError:
                    errors.append(f"{login_path}: invalid_json")
                    continue

                token = str(
                    payload.get("token")
                    or payload.get("access_token")
                    or payload.get("auth_token")
                    or ""
                ).strip()
                if not token:
                    errors.append(f"{login_path}: token_missing")
                    continue
                self._token = token
                self._member_id = str(payload.get("member_id") or payload.get("memberId") or "").strip() or None
                self._token_expiry_epoch = time.time() + max(60, int(self.auth.token_refresh_seconds))
                return
            raise KalshiAuthError("legacy_token_login_failed: " + " | ".join(errors))

    def _request(
        self,
        method: str,
        path: str,
        *,
        auth_required: bool = True,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        retry_on_401: bool = True,
    ) -> Dict[str, Any]:
        method = method.upper().strip()
        rel_path = "/" + path.lstrip("/")
        params = params or {}
        query = urlencode(params, doseq=True)
        url = f"{self.base_url}{rel_path}"
        if query:
            url = f"{url}?{query}"

        if self._token_enabled:
            self.ensure_authenticated()

        request_path = urlsplit(url).path
        headers: Dict[str, str] = {}
        if auth_required:
            headers.update(self._signed_headers(method=method, path=request_path))
        if self._token_enabled and self._token:
            headers["Authorization"] = f"Bearer {self._token}"
            if self._member_id:
                headers["KALSHI-MEMBER-ID"] = self._member_id
        if payload is not None:
            headers["Content-Type"] = "application/json"

        max_429_retries = max(0, int(self.auth.rate_limit_max_retries))
        attempt_429 = 0
        while True:
            self._apply_request_rate_limit()
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=payload,
                    headers=headers,
                    timeout=self.auth.timeout_seconds,
                )
            except requests.RequestException as exc:
                raise KalshiRequestError(f"network_error method={method} path={rel_path}: {exc}") from exc

            self._update_rate_limit_headers(response.headers)

            if response.status_code != 429 or attempt_429 >= max_429_retries:
                break

            retry_after_seconds = self._retry_after_seconds(response.headers.get("Retry-After"))
            fallback_seconds = min(
                float(self.auth.rate_limit_backoff_cap_seconds),
                float(self.auth.rate_limit_backoff_base_seconds) * (2 ** attempt_429),
            )
            wait_seconds = max(retry_after_seconds, fallback_seconds)
            self._last_429_epoch = time.time()
            self._sleep_for_rate_limit(wait_seconds)
            attempt_429 += 1

        if response.status_code == 401 and self._token_enabled and retry_on_401:
            self._refresh_legacy_token(force=True)
            return self._request(
                method=method,
                path=rel_path,
                auth_required=auth_required,
                payload=payload,
                params=params,
                retry_on_401=False,
            )

        if response.status_code >= 400:
            body_preview = response.text[:500]
            if response.status_code == 429:
                raise KalshiRequestError(
                    "http_error status=429 "
                    f"method={method} path={rel_path} retries={attempt_429} "
                    f"rate_wait_count={self._rate_limit_wait_count} "
                    f"rate_wait_seconds_total={self._rate_limit_wait_seconds_total:.3f} "
                    f"body={body_preview}"
                )
            raise KalshiRequestError(
                f"http_error status={response.status_code} method={method} path={rel_path} body={body_preview}"
            )

        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise KalshiRequestError(f"invalid_json method={method} path={rel_path}") from exc

    def _apply_request_rate_limit(self) -> None:
        max_rps = max(1, int(round(float(self.auth.max_requests_per_second))))
        now = time.time()
        one_second_ago = now - 1.0
        while self._request_timestamps and self._request_timestamps[0] < one_second_ago:
            self._request_timestamps.popleft()

        if len(self._request_timestamps) >= max_rps:
            wait_seconds = max(0.0, self._request_timestamps[0] + 1.0 - now)
            self._sleep_for_rate_limit(wait_seconds)
            now = time.time()
            one_second_ago = now - 1.0
            while self._request_timestamps and self._request_timestamps[0] < one_second_ago:
                self._request_timestamps.popleft()

        min_interval = 1.0 / float(max_rps)
        delta = now - self._last_request_epoch
        if self._last_request_epoch > 0 and delta < min_interval:
            self._sleep_for_rate_limit(min_interval - delta)
            now = time.time()
        self._last_request_epoch = now
        self._request_timestamps.append(now)

    def _sleep_for_rate_limit(self, seconds: float) -> None:
        sleep_seconds = max(0.0, float(seconds))
        if sleep_seconds <= 0:
            return
        self._rate_limit_wait_count += 1
        self._rate_limit_wait_seconds_total += sleep_seconds
        time.sleep(sleep_seconds)

    @staticmethod
    def _retry_after_seconds(raw: Optional[str]) -> float:
        if raw is None:
            return 0.0
        text = str(raw).strip()
        if not text:
            return 0.0
        try:
            return max(0.0, float(text))
        except ValueError:
            return 0.0

    def rate_limit_stats(self) -> Dict[str, Any]:
        return {
            "max_requests_per_second": float(self.auth.max_requests_per_second),
            "rate_limit_wait_count": int(self._rate_limit_wait_count),
            "rate_limit_wait_seconds_total": round(self._rate_limit_wait_seconds_total, 6),
            "last_429_epoch": float(self._last_429_epoch),
            "last_rate_limit_headers": dict(self._last_rate_limit_headers),
        }

    def _update_rate_limit_headers(self, headers: Dict[str, Any]) -> None:
        limit = headers.get("X-RateLimit-Limit")
        remaining = headers.get("X-RateLimit-Remaining")
        reset = headers.get("X-RateLimit-Reset")
        retry_after = headers.get("Retry-After")
        self._last_rate_limit_headers = {
            "x_rate_limit_limit": str(limit) if limit is not None else "",
            "x_rate_limit_remaining": str(remaining) if remaining is not None else "",
            "x_rate_limit_reset": str(reset) if reset is not None else "",
            "retry_after": str(retry_after) if retry_after is not None else "",
        }

    def websocket_headers(self) -> Dict[str, str]:
        ws_path = urlsplit(self.ws_url).path or "/trade-api/ws/v2"
        headers = self._signed_headers(method="GET", path=ws_path)
        if self._token_enabled and self._token_valid():
            headers["Authorization"] = f"Bearer {self._token}"
            if self._member_id:
                headers["KALSHI-MEMBER-ID"] = self._member_id
        return headers

    def websocket_header_candidates(self) -> list[dict[str, Any]]:
        ws_path = urlsplit(self.ws_url).path or "/trade-api/ws/v2"
        modes: list[str] = []
        if self.auth.api_key_id and self.private_key is not None:
            primary = (self.auth.pss_salt_mode or "digest").strip().lower()
            modes.append(primary)
            alt = "max" if primary != "max" else "digest"
            if alt not in modes:
                modes.append(alt)
        if not modes:
            return [{"headers": {}, "pss_salt_mode": "none"}]

        candidates: list[dict[str, Any]] = []
        for mode in modes:
            headers = self._signed_headers(method="GET", path=ws_path, pss_salt_mode=mode)
            if self._token_enabled and self._token_valid():
                headers["Authorization"] = f"Bearer {self._token}"
                if self._member_id:
                    headers["KALSHI-MEMBER-ID"] = self._member_id
            candidates.append({"headers": headers, "pss_salt_mode": mode})
        return candidates

    def get_market(self, ticker: str) -> Dict[str, Any]:
        return self._request("GET", f"/markets/{ticker}", auth_required=False)

    def list_markets(
        self,
        *,
        status: Optional[str] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": min(1000, max(1, limit))}
        if status:
            params["status"] = status
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/markets", auth_required=False, params=params)

    def get_orderbook(self, ticker: str, depth: int = 100) -> Dict[str, Any]:
        # Kalshi API requires depth in range [0, 100]; depth=0 means all levels.
        depth_norm = int(depth)
        if depth_norm < 0:
            depth_norm = 0
        if depth_norm > 100:
            depth_norm = 100
        params = {"depth": depth_norm}
        return self._request("GET", f"/markets/{ticker}/orderbook", auth_required=False, params=params)

    def get_balance(self) -> Dict[str, Any]:
        return self._request("GET", "/portfolio/balance", auth_required=True)

    def get_positions(
        self,
        *,
        ticker: Optional[str] = None,
        settlement_status: Optional[str] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": min(1000, max(1, limit))}
        if ticker:
            params["ticker"] = ticker
        if settlement_status:
            params["settlement_status"] = settlement_status
        return self._request("GET", "/portfolio/positions", auth_required=True, params=params)

    def get_orders(
        self,
        *,
        ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": min(200, max(1, limit))}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        return self._request("GET", "/portfolio/orders", auth_required=True, params=params)

    def get_fills(
        self,
        *,
        ticker: Optional[str] = None,
        limit: int = 200,
        min_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": min(1000, max(1, limit))}
        if ticker:
            params["ticker"] = ticker
        if min_ts:
            params["min_ts"] = int(min_ts)
        return self._request("GET", "/portfolio/fills", auth_required=True, params=params)

    def place_limit_order(
        self,
        *,
        ticker: str,
        side: str,
        count: int,
        price_cents: int,
        action: str = "buy",
        post_only: bool = True,
        time_in_force: str = "good_til_cancelled",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        side_norm = side.lower().strip()
        if side_norm not in {"yes", "no"}:
            raise ValueError("side must be 'yes' or 'no'.")
        cents = int(price_cents)
        if cents < 1 or cents > 99:
            raise ValueError("price_cents must be in [1, 99].")
        normalized_tif = normalize_time_in_force(time_in_force)
        payload: Dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side_norm,
            "count": int(count),
            "type": "limit",
            "time_in_force": normalized_tif,
            "client_order_id": client_order_id or uuid.uuid4().hex,
            "post_only": bool(post_only),
        }
        if side_norm == "yes":
            payload["yes_price"] = cents
        else:
            payload["no_price"] = cents
        try:
            return self._request("POST", "/portfolio/orders", auth_required=True, payload=payload)
        except KalshiRequestError as exc:
            err = str(exc)
            lower = err.lower()
            if "timeinforce" in lower or "time_in_force" in lower:
                fallback_tif = fallback_time_in_force_alias(payload["time_in_force"])
                if fallback_tif and fallback_tif != payload["time_in_force"]:
                    payload["time_in_force"] = fallback_tif
                    return self._request("POST", "/portfolio/orders", auth_required=True, payload=payload)
            raise

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/portfolio/orders/{order_id}", auth_required=True)

    def cancel_all_open_orders(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        orders_payload = self.get_orders(status="resting", limit=200)
        orders = orders_payload.get("orders", [])
        canceled = 0
        errors: list[str] = []
        for order in orders:
            order_ticker = str(order.get("ticker") or "").strip()
            if ticker and ticker != order_ticker:
                continue
            order_id = str(order.get("order_id") or order.get("id") or "").strip()
            if not order_id:
                continue
            try:
                self.cancel_order(order_id)
                canceled += 1
            except KalshiRequestError as exc:
                errors.append(str(exc))
        return {"canceled": canceled, "errors": errors}

    def resolve_active_market(self, series_ticker: str) -> Dict[str, Any]:
        """
        Returns the next tradable market for a rolling 15m series ticker.
        """
        now_ms = self._now_ms()
        cursor: Optional[str] = None
        candidates: list[tuple[int, Dict[str, Any]]] = []
        while True:
            payload = self.list_markets(
                status="open",
                series_ticker=series_ticker,
                cursor=cursor,
                limit=200,
            )
            for market in payload.get("markets", []):
                status = str(market.get("status") or "").lower()
                if status and status not in {"open", "active", "trading", "initialized"}:
                    continue
                close_ms = extract_timestamp_ms(
                    market.get("close_time")
                    or market.get("close_date")
                    or market.get("expiration_time")
                    or market.get("end_time")
                    or market.get("close_ts")
                )
                if close_ms is not None and close_ms <= now_ms + 5_000:
                    continue
                candidates.append((close_ms or 9_999_999_999_999, market))
            cursor = payload.get("cursor")
            if not cursor:
                break

        if not candidates:
            raise KalshiRequestError(f"no_active_markets_for_series={series_ticker}")
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    @staticmethod
    def _market_top_of_book_from_payload(market: Dict[str, Any]) -> Dict[str, Optional[int]]:
        yes_bid = extract_float(market.get("yes_bid"))
        yes_ask = extract_float(market.get("yes_ask"))
        no_bid = extract_float(market.get("no_bid"))
        no_ask = extract_float(market.get("no_ask"))

        if yes_ask is None and no_bid is not None:
            yes_ask = 100.0 - no_bid
        if yes_bid is None and no_ask is not None:
            yes_bid = 100.0 - no_ask
        if no_ask is None and yes_bid is not None:
            no_ask = 100.0 - yes_bid
        if no_bid is None and yes_ask is not None:
            no_bid = 100.0 - yes_ask

        return {
            "yes_bid": int(round(yes_bid)) if yes_bid is not None else None,
            "yes_ask": int(round(yes_ask)) if yes_ask is not None else None,
            "no_bid": int(round(no_bid)) if no_bid is not None else None,
            "no_ask": int(round(no_ask)) if no_ask is not None else None,
        }

    def resolve_quoteable_mm_market(
        self,
        *,
        series_ticker: str,
        min_seconds_to_close: float = 60.0,
        min_spread_cents: int = 1,
        max_spread_cents: int = 20,
        target_mid_cents: float = 50.0,
        max_mid_distance_cents: float = 35.0,
    ) -> Dict[str, Any]:
        """
        Resolve an open market in the series that is most quoteable for MM:
        - non-pinned top-of-book
        - spread within configured bounds
        - yes-mid close to target mid (default 50c)
        """
        now_ms = self._now_ms()
        min_close_ms = now_ms + max(5_000, int(max(0.0, min_seconds_to_close) * 1000.0))
        cursor: Optional[str] = None
        candidates: list[tuple[tuple[float, float, float], Dict[str, Any]]] = []

        while True:
            payload = self.list_markets(
                status="open",
                series_ticker=series_ticker,
                cursor=cursor,
                limit=200,
            )
            for market in payload.get("markets", []):
                status = str(market.get("status") or "").lower()
                if status and status not in {"open", "active", "trading", "initialized"}:
                    continue

                close_ms = extract_timestamp_ms(
                    market.get("close_time")
                    or market.get("close_date")
                    or market.get("expiration_time")
                    or market.get("end_time")
                    or market.get("close_ts")
                )
                if close_ms is not None and close_ms <= min_close_ms:
                    continue

                top = self._market_top_of_book_from_payload(market)
                yes_bid = top["yes_bid"]
                yes_ask = top["yes_ask"]
                no_bid = top["no_bid"]
                no_ask = top["no_ask"]
                if yes_bid is None or yes_ask is None:
                    continue

                spread_yes = yes_ask - yes_bid
                if spread_yes < int(min_spread_cents) or spread_yes > int(max_spread_cents):
                    continue

                pinned = bool(
                    yes_bid <= 1
                    or yes_ask >= 99
                    or (no_bid is not None and no_bid <= 1)
                    or (no_ask is not None and no_ask >= 99)
                )
                if pinned:
                    continue

                yes_mid = (yes_bid + yes_ask) / 2.0
                mid_distance = abs(float(yes_mid) - float(target_mid_cents))
                if mid_distance > float(max_mid_distance_cents):
                    continue

                # Score: mid-distance first, then tighter spread, then earlier close.
                close_score = float(close_ms or 9_999_999_999_999)
                score = (mid_distance, float(spread_yes), close_score)
                candidates.append((score, market))

            cursor = payload.get("cursor")
            if not cursor:
                break

        if not candidates:
            raise KalshiRequestError(f"no_quoteable_mm_markets_for_series={series_ticker}")
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]


def extract_timestamp_ms(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        raw = int(value)
        if raw <= 0:
            return None
        if raw < 1_000_000_000_000:
            return raw * 1000
        return raw
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            raw = int(text)
            if raw < 1_000_000_000_000:
                return raw * 1000
            return raw
        try:
            parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return int(parsed.timestamp() * 1000)
    return None


def extract_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def parse_json_file(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def normalize_time_in_force(value: str) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "gtc": "good_till_canceled",
        "good_till_canceled": "good_till_canceled",
        "good_til_canceled": "good_till_canceled",
        "good_till_cancelled": "good_till_canceled",
        "good_til_cancelled": "good_till_canceled",
        "ioc": "immediate_or_cancel",
        "immediate_or_cancel": "immediate_or_cancel",
        "fok": "fill_or_kill",
        "fill_or_kill": "fill_or_kill",
    }
    return aliases.get(text, text or "good_till_canceled")


def fallback_time_in_force_alias(value: str) -> Optional[str]:
    text = str(value or "").strip().lower()
    if text == "immediate_or_cancel":
        return "fill_or_kill"
    if text == "fill_or_kill":
        return "immediate_or_cancel"
    if text == "good_till_canceled":
        return "good_till_cancelled"
    if text == "good_till_cancelled":
        return "good_til_cancelled"
    if text == "good_til_cancelled":
        return "good_til_canceled"
    if text == "good_til_canceled":
        return "good_till_canceled"
    return None
