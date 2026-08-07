"""Versioned, file-based signal contract for the vn.py integration."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from tradingagents.agents.schemas import PortfolioRating


class VnpyAction(str, Enum):
    """The only execution intents a consumer may receive."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class VnpySignal(BaseModel):
    """A validated, reviewable decision signal for an external vn.py strategy.

    The signal is an intent, not an order.  Consumers must apply their own
    whitelists, portfolio limits, and risk controls before submitting orders.
    """

    schema_version: str = "1.0"
    signal_id: str
    generated_at: datetime
    valid_until: datetime
    ticker: str = Field(description="Original TradingAgents ticker, for auditability.")
    symbol: str = Field(description="vn.py symbol without an exchange suffix.")
    exchange: str = Field(description="vn.py exchange name, e.g. SSE or SZSE.")
    action: VnpyAction
    rating: PortfolioRating
    target_position_pct: float = Field(
        ge=0.0,
        le=1.0,
        description="Conservative long-only target weight; the consumer owns shorting policy.",
    )
    price_target: float | None = None
    time_horizon: str | None = None
    thesis: str
