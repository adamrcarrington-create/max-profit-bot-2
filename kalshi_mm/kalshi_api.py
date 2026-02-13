from __future__ import annotations

import base64
import datetime as dt
from dataclasses import dataclass
import os
from typing import Any, Dict, Literal, Optional
from urllib.parse import urlencode

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import requests


class KalshiApiError(RuntimeError):
    pass


@dataclass
class KalshiCredentials:
    api_key_id: str
    private_key_path: str


class KalshiClient:
    def __init__(
        self,
        credentials: KalshiCredentials,
        base_url: str = "https://demo-api.kalshi.co",
        timeout_seconds: float = 10.0,
        pss_salt_mode: Literal["digest", "max"] = "digest",
        session: Optional[requests.Session] = None,
    ) -> None:
        self.credentials = credentials
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.pss_salt_mode = pss_salt_mode
        self.session = session or requests.Session()
        self.private_key = _load_private_key_from_file(credentials.private_key_path)

    def get(self, path: str, params: Optional[Dict[str, Any]] = None, auth: bool = True) -> Dict[str, Any]:
        return self._request("GET", path, params=params, auth=auth)

    def post(self, path: str, payload: Dict[str, Any], auth: bool = True) -> Dict[str, Any]:
        return self._request("POST", path, payload=payload, auth=auth)

    def delete(
        self,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        auth: bool = True,
    ) -> Dict[str, Any]:
        return self._request("DELETE", path, payload=payload, params=params, auth=auth)

    def get_balance(self) -> Dict[str, Any]:
        return self.get("/trade-api/v2/portfolio/balance")

    def get_market(self, ticker: str) -> Dict[str, Any]:
        return self.get(f"/trade-api/v2/markets/{ticker}", auth=False)

    def list_markets(
        self,
        status: Optional[str] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        limit: int = 200,
        cursor: Optional[str] = None,
        auth: bool = False,
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
        return self.get("/trade-api/v2/markets", params=params, auth=auth)

    def get_event(self, event_ticker: str) -> Dict[str, Any]:
        return self.get(f"/trade-api/v2/events/{event_ticker}", auth=False)

    def list_events(
        self,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
        cursor: Optional[str] = None,
        auth: bool = False,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": min(1000, max(1, limit))}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        return self.get("/trade-api/v2/events", params=params, auth=auth)

    def get_series(self, series_ticker: str) -> Dict[str, Any]:
        return self.get(f"/trade-api/v2/series/{series_ticker}", auth=False)

    def get_positions(self, ticker: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": min(1000, max(1, limit))}
        if ticker:
            params["ticker"] = ticker
        return self.get("/trade-api/v2/portfolio/positions", params=params, auth=True)

    def get_orders(
        self,
        ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": min(200, max(1, limit))}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        return self.get("/trade-api/v2/portfolio/orders", params=params, auth=True)

    def create_order(self, order_payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.post("/trade-api/v2/portfolio/orders", payload=order_payload, auth=True)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return self.delete(f"/trade-api/v2/portfolio/orders/{order_id}", auth=True)

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        auth: bool = True,
    ) -> Dict[str, Any]:
        method = method.upper().strip()
        params = params or {}
        url = self.base_url + path
        if params:
            url = f"{url}?{urlencode(params)}"

        headers: Dict[str, str] = {}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if auth:
            headers.update(self._signed_headers(method=method, path=path))

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise KalshiApiError(f"network_error method={method} path={path}: {exc}") from exc

        if response.status_code >= 400:
            body_preview = response.text[:500]
            raise KalshiApiError(
                f"http_error status={response.status_code} method={method} path={path} body={body_preview}"
            )

        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise KalshiApiError(f"invalid_json method={method} path={path}") from exc

    def _signed_headers(self, method: str, path: str) -> Dict[str, str]:
        timestamp = str(int(dt.datetime.now().timestamp() * 1000))
        signature = _sign_request(self.private_key, timestamp, method, path, self.pss_salt_mode)
        return {
            "KALSHI-ACCESS-KEY": self.credentials.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }


def resolve_credentials(
    api_key_id: Optional[str],
    private_key_path: Optional[str],
) -> KalshiCredentials:
    resolved_key_id = (api_key_id or os.getenv("KALSHI_API_KEY_ID") or "").strip()
    resolved_key_path = (private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH") or "").strip()
    if not resolved_key_id:
        raise ValueError("Missing API key id. Pass --api-key-id or set KALSHI_API_KEY_ID.")
    if not resolved_key_path:
        raise ValueError("Missing private key path. Pass --private-key-path or set KALSHI_PRIVATE_KEY_PATH.")
    return KalshiCredentials(api_key_id=resolved_key_id, private_key_path=resolved_key_path)


def _load_private_key_from_file(file_path: str) -> rsa.RSAPrivateKey:
    with open(file_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _sign_request(
    private_key: rsa.RSAPrivateKey,
    timestamp: str,
    method: str,
    path: str,
    pss_salt_mode: Literal["digest", "max"] = "digest",
) -> str:
    path_without_query = path.split("?")[0]
    message = f"{timestamp}{method}{path_without_query}".encode("utf-8")
    if pss_salt_mode == "max":
        salt_length = padding.PSS.MAX_LENGTH
    else:
        salt_length = padding.PSS.DIGEST_LENGTH
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=salt_length),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def auth_diagnostics(
    credentials: KalshiCredentials,
    timeout_seconds: float = 8.0,
) -> list[dict[str, str]]:
    targets = [
        "https://demo-api.kalshi.co",
        "https://api.kalshi.com",
        "https://api.elections.kalshi.com",
    ]
    modes: list[Literal["digest", "max"]] = ["digest", "max"]
    results: list[dict[str, str]] = []
    for base_url in targets:
        for mode in modes:
            client = KalshiClient(
                credentials=credentials,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                pss_salt_mode=mode,
            )
            outcome = {
                "base_url": base_url,
                "pss_salt_mode": mode,
                "status": "unknown",
                "detail": "",
            }
            try:
                payload = client.get_balance()
                outcome["status"] = "ok"
                balance = payload.get("balance")
                outcome["detail"] = f"balance={balance}"
            except KalshiApiError as exc:
                outcome["status"] = "error"
                outcome["detail"] = str(exc)
            results.append(outcome)
    return results
