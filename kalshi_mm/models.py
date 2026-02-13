from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarketState:
    timestamp: float
    mid: float
    best_bid: float
    best_ask: float
    volatility: float
    momentum: float


@dataclass
class Quote:
    bid: float
    ask: float
    buy_size: int
    sell_size: int
    reason: str = ""


@dataclass
class Fill:
    side: str
    price: float
    size: int


@dataclass
class PositionState:
    qty: int = 0
    avg_price: float = 0.0
    cash: float = 0.0
    realized_pnl: float = 0.0

    def unrealized_pnl(self, mark: float) -> float:
        if self.qty == 0:
            return 0.0
        return self.qty * (mark - self.avg_price)

    def equity(self, mark: float) -> float:
        return self.cash + (self.qty * mark)

    def apply_fill(self, side: str, price: float, size: int) -> None:
        if size <= 0:
            return
        if side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported side: {side}")

        if side == "buy":
            self.cash -= price * size
            self._apply_buy(price, size)
            return

        self.cash += price * size
        self._apply_sell(price, size)

    def _apply_buy(self, price: float, size: int) -> None:
        if self.qty >= 0:
            new_qty = self.qty + size
            if new_qty > 0:
                self.avg_price = (
                    (self.avg_price * self.qty) + (price * size)
                ) / new_qty
            self.qty = new_qty
            return

        close_size = min(size, -self.qty)
        self.realized_pnl += (self.avg_price - price) * close_size
        self.qty += close_size
        remaining = size - close_size

        if self.qty == 0:
            self.avg_price = 0.0
        if remaining > 0:
            self.qty = remaining
            self.avg_price = price

    def _apply_sell(self, price: float, size: int) -> None:
        if self.qty <= 0:
            short_qty = -self.qty
            new_short_qty = short_qty + size
            if new_short_qty > 0:
                self.avg_price = (
                    (self.avg_price * short_qty) + (price * size)
                ) / new_short_qty
            self.qty = -new_short_qty
            return

        close_size = min(size, self.qty)
        self.realized_pnl += (price - self.avg_price) * close_size
        self.qty -= close_size
        remaining = size - close_size

        if self.qty == 0:
            self.avg_price = 0.0
        if remaining > 0:
            self.qty = -remaining
            self.avg_price = price

