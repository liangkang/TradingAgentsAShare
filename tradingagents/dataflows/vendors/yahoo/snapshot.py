"""Deterministic market-data verification snapshot.

The market analyst is an LLM that can confabulate exact numbers — citing a
Bollinger band or a "historically validated bounce" that the underlying data
doesn't support (#830). This module computes a ground-truth snapshot (latest
OHLCV row on or before the analysis date, common indicators, recent closes)
the analyst is told to treat as the source of truth for any exact numeric
claim. Deterministic, no LLM involved.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.router import get_vendor
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.akshare import _load_ohlcv_akshare
from tradingagents.dataflows.vendors.yahoo.common import price_text
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

logger = logging.getLogger(__name__)

# A fixed, common indicator set so the snapshot is the same shape every run.
DEFAULT_SNAPSHOT_INDICATORS: tuple[str, ...] = (
    "close_10_ema", "close_50_sma", "close_200_sma",
    "rsi", "boll", "boll_ub", "boll_lb",
    "macd", "macds", "macdh", "atr",
)


def _load_rows(symbol: str, as_of_date: str) -> pd.DataFrame:
    """OHLCV, as the vendor reported it, from the chain that serves get_stock_data.

    The snapshot is the analyst's ground truth for the prices get_stock_data
    shows it, so it reads from the same configured ``core_stock_apis`` chain
    (e.g. ``akshare,yfinance``) rather than always from Yahoo: two sources can
    disagree on price adjustment and session coverage. A vendor with no data for
    the symbol, or one that cannot be reached, hands over to the next. A chain
    naming no vendor the snapshot can read falls back to Yahoo.
    """
    loaders = {
        "akshare": lambda: _load_ohlcv_akshare(symbol, as_of_date),
        "yfinance": lambda: load_ohlcv(symbol, as_of_date, fill_gaps=False),
    }
    configured = [v.strip() for v in get_vendor("core_stock_apis", "get_stock_data").split(",")]
    chain = [v for v in configured if v in loaders] or ["yfinance"]

    error: Exception | None = None
    for vendor in chain:
        try:
            return loaders[vendor]()
        except (NoMarketDataError, VendorUnavailableError, OSError) as exc:
            logger.warning("Snapshot vendor %r could not serve %s: %s", vendor, symbol, exc)
            error = exc
    if isinstance(error, OSError):
        # A tool that raises ends the run; an unreachable vendor must not.
        raise VendorUnavailableError(f"no snapshot vendor reachable for {symbol}: {error}") from error
    raise error


def _verified_rows(symbol: str, as_of_date: str) -> pd.DataFrame:
    """OHLCV on or before as_of_date, date-sorted. Raises NoMarketDataError if nothing usable.

    ``load_ohlcv`` already normalizes the Date column and filters out
    look-ahead rows, but we re-apply the cutoff defensively — this is a
    verification path, so it must not trust its input to be pre-filtered.
    """
    # As reported: this snapshot is quoted by the agents as exact prices, so a
    # gap-filled cell would put the previous session's number under this date.
    data = _load_rows(symbol, as_of_date)
    if data is None or data.empty:
        raise NoMarketDataError(symbol, normalize_symbol(symbol), "no price rows")

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] <= pd.to_datetime(as_of_date)].sort_values("Date")
    if df.empty:
        raise NoMarketDataError(symbol, normalize_symbol(symbol), f"no price rows on or before {as_of_date}")
    return df


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return price_text(value)
    return str(value)


def build_verified_market_snapshot(
    symbol: str,
    as_of_date: str,
    look_back_days: int = 30,
    indicators: Iterable[str] | None = None,
) -> str:
    """Render a ground-truth snapshot: latest OHLCV row, indicators, recent closes."""
    # `df` keeps the original capitalized OHLCV columns (Open/High/Low/Close/
    # Volume); stockstats `wrap()` lowercases columns and adds indicator
    # columns, so read raw prices from `df` and indicators from `stock_df`.
    df = _verified_rows(symbol, as_of_date)
    stock_df = wrap(df.copy())

    selected = tuple(indicators or DEFAULT_SNAPSHOT_INDICATORS)
    indicator_values: dict[str, str] = {}
    for name in selected:
        try:
            stock_df[name]  # triggers stockstats calculation
            indicator_values[name] = _fmt(stock_df.iloc[-1][name])
        except Exception as exc:  # noqa: BLE001 — one bad indicator shouldn't sink the snapshot
            indicator_values[name] = f"N/A ({type(exc).__name__})"

    latest = df.iloc[-1]
    latest_date = _fmt(latest["Date"])
    window = max(1, min(int(look_back_days), 30))
    recent = df.tail(window)

    lines = [
        f"## Verified market data snapshot for {symbol.upper()}",
        "",
        f"- Requested analysis date: {as_of_date}",
        f"- Latest trading row used: {latest_date}",
        "- Rows after the requested analysis date are excluded before verification.",
        "",
        "### Latest verified OHLCV row",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    for field in ("Open", "High", "Low", "Close", "Volume"):
        lines.append(f"| {field} | {_fmt(latest.get(field))} |")

    lines += ["", "### Verified technical indicators (latest row)", "",
              "| Indicator | Value |", "|---|---:|"]
    for name, value in indicator_values.items():
        lines.append(f"| {name} | {value} |")

    lines += ["", f"### Recent verified closes (last {len(recent)} rows)", "",
              "| Date | Close |", "|---|---:|"]
    for _, row in recent.iterrows():
        lines.append(f"| {_fmt(row['Date'])} | {_fmt(row.get('Close'))} |")

    lines += [
        "",
        "Use this snapshot as the source of truth for exact OHLCV, price-level, "
        "and indicator-value claims. If another tool output conflicts with it, "
        "flag the discrepancy rather than inventing a reconciled number. Do not "
        "claim historical validation, support/resistance bounces, or exact "
        "percentage moves unless directly supported by tool output with concrete "
        "dates and prices.",
    ]
    return "\n".join(lines)
