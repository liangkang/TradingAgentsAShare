"""Export TradingAgents' final decision as a vn.py-safe JSON signal."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from tradingagents.agents.schemas import PortfolioRating
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.integrations.vnpy.signal import VnpyAction, VnpySignal

_ACTION_BY_RATING = {
    PortfolioRating.BUY: VnpyAction.BUY,
    PortfolioRating.OVERWEIGHT: VnpyAction.BUY,
    PortfolioRating.HOLD: VnpyAction.HOLD,
    PortfolioRating.UNDERWEIGHT: VnpyAction.SELL,
    PortfolioRating.SELL: VnpyAction.SELL,
}
_TARGET_POSITION_BY_RATING = {
    PortfolioRating.BUY: 0.10,
    PortfolioRating.OVERWEIGHT: 0.05,
    PortfolioRating.HOLD: 0.0,
    # The contract is deliberately long-only.  An LLM rating must never imply
    # a new short position in a market where shorting may be unavailable or
    # heavily constrained; the consumer decides whether a SELL reduces or
    # closes an existing position.
    PortfolioRating.UNDERWEIGHT: 0.0,
    PortfolioRating.SELL: 0.0,
}
_TICKER_EXCHANGES = {
    ".SS": "SSE",
    ".SH": "SSE",
    ".SZ": "SZSE",
}


def _extract_markdown_field(text: str, name: str) -> str | None:
    match = re.search(rf"^\*\*{re.escape(name)}\*\*:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def resolve_vnpy_instrument(ticker: str, exchange: str | None = None) -> tuple[str, str]:
    """Resolve a ticker to the symbol/exchange pair vn.py expects.

    Only unambiguous Mainland China suffixes are inferred.  Other markets must
    pass an explicit exchange; guessing a venue could route a valid symbol to
    the wrong market.
    """
    normalized = ticker.strip().upper()
    if not normalized:
        raise ValueError("ticker must not be empty")

    symbol = normalized
    inferred_exchange = None
    for suffix, mapped_exchange in _TICKER_EXCHANGES.items():
        if normalized.endswith(suffix):
            symbol = normalized[: -len(suffix)]
            inferred_exchange = mapped_exchange
            break

    resolved_exchange = exchange.strip().upper() if exchange else inferred_exchange
    if not resolved_exchange:
        raise ValueError(
            f"cannot infer a vn.py exchange for {ticker!r}; pass --vnpy-exchange explicitly"
        )
    return symbol, resolved_exchange


def build_vnpy_signal(
    final_state: dict,
    ticker: str,
    *,
    exchange: str | None = None,
    ttl_hours: int = 24,
    generated_at: datetime | None = None,
) -> VnpySignal:
    """Create a signal from a completed graph state without writing to disk."""
    if ttl_hours <= 0:
        raise ValueError("ttl_hours must be greater than zero")

    decision = final_state.get("final_trade_decision")
    if not isinstance(decision, str) or not decision.strip():
        raise ValueError("final_state must contain a non-empty final_trade_decision")

    rating = PortfolioRating(parse_rating(decision))
    symbol, resolved_exchange = resolve_vnpy_instrument(ticker, exchange)
    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    price_target_text = _extract_markdown_field(decision, "Price Target")
    try:
        price_target = float(price_target_text) if price_target_text is not None else None
    except ValueError:
        price_target = None

    thesis = _extract_markdown_field(decision, "Investment Thesis") or decision.strip()
    return VnpySignal(
        signal_id=str(uuid4()),
        generated_at=timestamp,
        valid_until=timestamp + timedelta(hours=ttl_hours),
        ticker=ticker,
        symbol=symbol,
        exchange=resolved_exchange,
        action=_ACTION_BY_RATING[rating],
        rating=rating,
        target_position_pct=_TARGET_POSITION_BY_RATING[rating],
        price_target=price_target,
        time_horizon=_extract_markdown_field(decision, "Time Horizon"),
        thesis=thesis,
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write a complete JSON file before making it visible to a consumer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary_path = Path(handle.name)
    try:
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def export_vnpy_signal(
    final_state: dict,
    ticker: str,
    output_dir: Path | str,
    *,
    exchange: str | None = None,
    ttl_hours: int = 24,
) -> Path:
    """Build and atomically export the latest and immutable copies of a signal.

    ``<output_dir>/<ticker>.json`` is the file a live strategy polls.  The
    matching ``history/`` file is immutable audit evidence for the same signal.
    """
    signal = build_vnpy_signal(
        final_state, ticker, exchange=exchange, ttl_hours=ttl_hours
    )
    directory = Path(output_dir)
    safe_ticker = safe_ticker_component(ticker)
    payload = signal.model_dump(mode="json")
    latest_path = directory / f"{safe_ticker}.json"
    history_path = directory / "history" / f"{safe_ticker}.{signal.signal_id}.json"
    _atomic_write_json(history_path, payload)
    _atomic_write_json(latest_path, payload)
    return latest_path
