"""akshare-based data vendor module.

Provides the same function signatures as the yfinance and alpha_vantage
vendors so the router in ``interface.py`` can dispatch to akshare as the
primary data source. All functions return ``str`` (CSV with a header, or
formatted prose for news/insider).

Coverage:
- OHLCV stock data:  US (stock_us_daily), HK (stock_hk_hist), CN (stock_zh_a_hist)
- Technical indicators:  via stockstats (same pipeline, different data source)
- Fundamentals:          CN (stock_profile_cninfo), HK (stock_hk_company_profile_em)
                         — US raises NoMarketDataError (falls back to yfinance)
- Financial statements:  US (stock_financial_us_report_em, long→wide pivot),
                         CN (stock_balance_sheet_by_report_em, etc.)
- News:                  CN only (East Money company news + CNInfo disclosures).
                         US/HK raise NoMarketDataError.
- Global news:           China-focused CLS flashes, CCTV news, and a look-ahead-
                         safe current Caixin headline snapshot.
- Insider transactions:  Always raises NoMarketDataError (not supported).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from dateutil.relativedelta import relativedelta
from stockstats import wrap

from .config import get_config
from .symbol_utils import NoMarketDataError
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)

# Network-related exception classes that should propagate so the
# router can fall back to the next vendor instead of returning a
# "no data" sentinel.
_NETWORK_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_ak_date(yyyy_mm_dd: str) -> str:
    """Convert ``YYYY-MM-DD`` → ``YYYYMMDD`` (akshare param format)."""
    return yyyy_mm_dd.replace("-", "")


def _fetch_cn_daily(ak_sym: str) -> "pd.DataFrame":
    """Fetch A-share daily OHLCV — Sina first, Tencent fallback.

    Sina (``stock_zh_a_daily``) has full columns (open/high/low/close/volume).
    Tencent (``stock_zh_a_hist_tx``) has fewer columns (date/open/close/high/low/amount
    but no volume). Both are normalised to lowercase columns before return.

    Raises directly if both sources fail — callers decide whether to wrap
    in *NoMarketDataError*.
    """
    import akshare as ak

    sina_sym = ak_sym.replace("SH", "sh").replace("SZ", "sz")
    errors: list[str] = []

    # 1. Sina Finance (primary)
    try:
        df = ak.stock_zh_a_daily(symbol=sina_sym, adjust="qfq")
        if df is not None and not df.empty:
            return df
        errors.append("Sina returned empty DataFrame")
    except Exception as exc:
        errors.append(f"Sina: {type(exc).__name__}")

    # 2. Tencent (fallback)
    try:
        df = ak.stock_zh_a_hist_tx(symbol=sina_sym, adjust="qfq")
        if df is not None and not df.empty:
            # Tencent returns: date, open, close, high, low, amount
            # Rename to standard lowercase columns; no volume column.
            return df.rename(columns={"amount": "volume"}) if "amount" in df.columns and "volume" not in df.columns else df
        errors.append("Tencent returned empty DataFrame")
    except Exception as exc:
        errors.append(f"Tencent: {type(exc).__name__}")

    raise RuntimeError(f"All CN daily sources failed: {'; '.join(errors)}")


def _extract_market(ticker: str) -> tuple[str, str, str]:
    """Dispatch ticker suffix → (market, ak_symbol, display_name).

    Returns
    -------
    market : str
        One of ``"us"``, ``"hk"``, ``"cn"``, or ``"unknown"``.
    ak_symbol : str
        The symbol ready to pass to akshare functions.
    display_name : str
        Human-readable label for headers / error messages.
    """
    upper = ticker.strip().upper()
    # Remove OTC suffixes that akshare does not use
    if upper.endswith(".OQ") or upper.endswith(".O"):
        upper = upper.rsplit(".", 1)[0]
    # Remove .PK suffix
    if upper.endswith(".PK"):
        upper = upper.rsplit(".", 1)[0]

    if upper.endswith(".HK"):
        code = upper[:-3].lstrip("0") or "0"
        return ("hk", code.zfill(5), upper)
    if upper.endswith((".SS", ".SH")):
        return ("cn", "SH" + upper[:-3], upper)
    if upper.endswith(".SZ"):
        return ("cn", "SZ" + upper[:-3], upper)
    # Suffixes akshare does not cover → raise downstream
    if upper.endswith(".T"):
        return ("unknown", upper, upper)
    if upper.endswith(".L"):
        return ("unknown", upper, upper)
    if upper.endswith(".TO"):
        return ("unknown", upper, upper)
    if upper.endswith(".AX"):
        return ("unknown", upper, upper)
    if upper.endswith(".NS"):
        return ("unknown", upper, upper)
    if upper.endswith(".BO"):
        return ("unknown", upper, upper)
    # Crypto (e.g. BTC-USD) and other suffixes → unknown
    if "-" in upper and any(
        upper.endswith(s) for s in ("-USD", "USD=X", "=X")
    ):
        return ("unknown", upper, upper)
    # US tickers: bare symbol, possibly with dots (BRK.B)
    return ("us", upper, upper)


def _ak_to_csv(df: pd.DataFrame, header_text: str) -> str:
    """Normalise akshare DataFrame columns and return CSV string with header.

    akshare uses lowercase column names (``date``, ``open``, ``high``,
    ``low``, ``close``, ``volume``).  Rename to the capitalised convention
    the rest of the system expects, round OHLCV columns to 2 dp, and prepend
    the caller-supplied header.
    """
    rename_map = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj close": "Adj Close",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Round numeric price columns
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: round(x, 2) if pd.notna(x) else x
            )

    return header_text + df.to_csv(index=False)


def _ak_date_filter(
    df: pd.DataFrame, start_date: str, end_date: str, date_col: str = "Date"
) -> pd.DataFrame:
    """Filter DataFrame to keep rows between *start_date* and *end_date*."""
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    return df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]


# ---------------------------------------------------------------------------
# OHLCV stock data
# ---------------------------------------------------------------------------


def _load_ohlcv_akshare(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch & cache OHLCV via akshare, returning a cleaned DataFrame.

    Columns: ``Date, Open, High, Low, Close, Volume``.
    Rows after *curr_date* are excluded (look-ahead prevention).
    """
    import akshare as ak

    market, ak_sym, display = _extract_market(symbol)
    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)
    today = pd.Timestamp.today()
    start_str = (today - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    safe_sym = safe_ticker_component(ak_sym)
    cache_file = os.path.join(
        config["data_cache_dir"],
        f"{safe_sym}-AK-data-{start_str}-{end_str}.csv",
    )

    # --- cache hit ---
    data = None
    if os.path.exists(cache_file):
        try:
            cached = pd.read_csv(cache_file, on_bad_lines="skip", encoding="utf-8")
            if not cached.empty and "Close" in cached.columns:
                data = cached
        except Exception:
            pass

    # --- cache miss: fetch ---
    if data is None:
        try:
            if market == "us":
                df = ak.stock_us_daily(symbol=ak_sym, adjust="qfq")
            elif market == "hk":
                df = ak.stock_hk_daily(symbol=ak_sym, adjust="qfq")
            elif market == "cn":
                df = _fetch_cn_daily(ak_sym)
            else:
                raise NoMarketDataError(
                    symbol,
                    ak_sym,
                    f"market {market} not supported by akshare OHLCV loader",
                )
        except NoMarketDataError:
            raise
        except _NETWORK_ERRORS:
            raise  # let router fall back to next vendor
        except Exception as exc:
            raise NoMarketDataError(
                symbol, ak_sym, f"akshare download error: {exc}"
            ) from exc

        if df is None or df.empty:
            raise NoMarketDataError(
                symbol, ak_sym, "akshare returned no rows"
            )

        # Normalise columns
        df = df.rename(
            columns={
                "日期": "Date",
                "开盘": "Open",
                "最高": "High",
                "最低": "Low",
                "收盘": "Close",
                "成交量": "Volume",
                "成交额": "Amount",
            }
        )
        # Rename typical akshare lowercase columns
        lower_map = {
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "adj close": "Adj Close",
        }
        df = df.rename(
            columns={k: v for k, v in lower_map.items() if k in df.columns}
        )

        if "Date" not in df.columns:
            raise NoMarketDataError(
                symbol,
                ak_sym,
                "akshare DataFrame missing Date column",
            )

        # Only cache real data
        if "Close" in df.columns and not df.empty:
            df.to_csv(cache_file, index=False, encoding="utf-8")
        data = df

    # --- clean and filter ---
    if data is None or data.empty:
        raise NoMarketDataError(
            symbol, ak_sym, "no OHLCV data after cache resolution"
        )

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])
    price_cols = [
        c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns
    ]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()
    data = data[data["Date"] <= curr_date_dt]

    if data.empty:
        raise NoMarketDataError(
            symbol, ak_sym, "no OHLCV rows on or before analysis date"
        )

    return data


def get_stock_data_akshare(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """OHLCV stock price data via akshare → CSV string with header."""
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    market, ak_sym, display = _extract_market(symbol)

    import akshare as ak

    try:
        if market == "us":
            df = ak.stock_us_daily(symbol=ak_sym, adjust="qfq")
        elif market == "hk":
            # Sina Finance source — more reliable than East Money (push2his)
            df = ak.stock_hk_daily(symbol=ak_sym, adjust="qfq")
        elif market == "cn":
            df = _fetch_cn_daily(ak_sym)
        else:
            raise NoMarketDataError(
                symbol, ak_sym, f"akshare does not support market '{market}'"
            )
    except NoMarketDataError:
        raise
    except _NETWORK_ERRORS:
        raise  # let router fall back to next vendor
    except Exception as exc:
        raise NoMarketDataError(
            symbol, ak_sym, f"akshare error: {exc}"
        ) from exc

    if df is None or df.empty:
        raise NoMarketDataError(
            symbol, ak_sym, "no rows returned by akshare"
        )

    # Filter to requested date range — all akshare daily functions return
    # full history regardless of market.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[
        (df["date"] >= pd.to_datetime(start_date))
        & (df["date"] <= pd.to_datetime(end_date))
    ]

    if df.empty:
        raise NoMarketDataError(
            symbol, ak_sym, f"no rows in date range {start_date} → {end_date}"
        )

    label = ak_sym if ak_sym == symbol.upper() else f"{ak_sym} (from {symbol})"
    header = (
        f"# Stock data for {label} from {start_date} to {end_date}\n"
        f"# Total records: {len(df)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Source: akshare\n\n"
    )
    return _ak_to_csv(df, header)


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

# Indicator descriptions — identical to y_finance.py so agent prompts
# stay consistent regardless of vendor.
_BEST_IND_PARAMS: dict[str, str] = {
    "close_50_sma": (
        "50 SMA: A medium-term trend indicator. "
        "Usage: Identify trend direction and serve as dynamic support/resistance. "
        "Tips: It lags price; combine with faster indicators for timely signals."
    ),
    "close_200_sma": (
        "200 SMA: A long-term trend benchmark. "
        "Usage: Confirm overall market trend and identify golden/death cross setups. "
        "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
    ),
    "close_10_ema": (
        "10 EMA: A responsive short-term average. "
        "Usage: Capture quick shifts in momentum and potential entry points. "
        "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
    ),
    "macd": (
        "MACD: Computes momentum via differences of EMAs. "
        "Usage: Look for crossovers and divergence as signals of trend changes. "
        "Tips: Confirm with other indicators in low-volatility or sideways markets."
    ),
    "macds": (
        "MACD Signal: An EMA smoothing of the MACD line. "
        "Usage: Use crossovers with the MACD line to trigger trades. "
        "Tips: Should be part of a broader strategy to avoid false positives."
    ),
    "macdh": (
        "MACD Histogram: Shows the gap between the MACD line and its signal. "
        "Usage: Visualize momentum strength and spot divergence early. "
        "Tips: Can be volatile; complement with additional filters in fast-moving markets."
    ),
    "rsi": (
        "RSI: Measures momentum to flag overbought/oversold conditions. "
        "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
        "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
    ),
    "boll": (
        "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
        "Usage: Acts as a dynamic benchmark for price movement. "
        "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
    ),
    "boll_ub": (
        "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
        "Usage: Signals potential overbought conditions and breakout zones. "
        "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
    ),
    "boll_lb": (
        "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
        "Usage: Indicates potential oversold conditions. "
        "Tips: Use additional analysis to avoid false reversal signals."
    ),
    "atr": (
        "ATR: Averages true range to measure volatility. "
        "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
        "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
    ),
    "vwma": (
        "VWMA: A moving average weighted by volume. "
        "Usage: Confirm trends by integrating price action with volume data. "
        "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
    ),
    "mfi": (
        "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
        "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
        "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
    ),
}


def _get_stock_stats_bulk_akshare(symbol: str, indicator: str, curr_date: str) -> dict:
    """Bulk indicator calculation → dict mapping date strings to values."""
    data = _load_ohlcv_akshare(symbol, curr_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    df[indicator]  # trigger stockstats calculation

    result: dict[str, str] = {}
    for _, row in df.iterrows():
        val = row[indicator]
        result[row["Date"]] = "N/A" if pd.isna(val) else str(val)
    return result


def get_stock_stats_indicators_akshare(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
) -> str:
    """Technical indicator values via akshare OHLCV + stockstats."""
    if indicator not in _BEST_IND_PARAMS:
        raise ValueError(
            f"Indicator {indicator} is not supported. "
            f"Please choose from: {list(_BEST_IND_PARAMS.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    try:
        indicator_data = _get_stock_stats_bulk_akshare(
            symbol, indicator, curr_date
        )

        current_dt = curr_date_dt
        ind_string = ""
        while current_dt >= before:
            date_str = current_dt.strftime("%Y-%m-%d")
            value = indicator_data.get(date_str, "N/A: Not a trading day (weekend or holiday)")
            ind_string += f"{date_str}: {value}\n"
            current_dt = current_dt - relativedelta(days=1)
    except NoMarketDataError:
        raise
    except Exception as exc:
        logger.warning("Error getting bulk stockstats via akshare: %s", exc)
        # Fallback: one-by-one via StockstatsUtils pattern
        ind_string = ""
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        while curr_dt >= before:
            date_str = curr_dt.strftime("%Y-%m-%d")
            try:
                data = _load_ohlcv_akshare(symbol, date_str)
                sdf = wrap(data)
                sdf["Date"] = sdf["Date"].dt.strftime("%Y-%m-%d")
                sdf[indicator]
                matching = sdf[sdf["Date"].str.startswith(date_str)]
                if not matching.empty:
                    val = matching[indicator].values[0]
                    ind_string += f"{date_str}: {val}\n"
                else:
                    ind_string += f"{date_str}: N/A: Not a trading day\n"
            except Exception:
                ind_string += f"{date_str}: N/A: Error\n"
            curr_dt = curr_dt - relativedelta(days=1)

    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + _BEST_IND_PARAMS.get(indicator, "No description available.")
    )


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------


def get_fundamentals_akshare(ticker: str, curr_date: str = None) -> str:
    """Company fundamentals via akshare.

    US stocks are NOT supported (raises *NoMarketDataError* so the router
    falls back to yfinance).
    """
    market, ak_sym, display = _extract_market(ticker)

    import akshare as ak

    try:
        if market == "cn":
            # Use stock_info_a_code_name (cached table of all A-share codes →
            # names) as the reliable minimum. Falls back to cninfo profile for
            # richer metadata when available.
            bare_code = ak_sym.replace("SH", "").replace("SZ", "")
            lines = []
            try:
                name_df = ak.stock_info_a_code_name()
                row = name_df[name_df["code"] == bare_code]
                if not row.empty:
                    lines.append(f"Name: {row.iloc[0]['name']}")
            except Exception:
                pass

            # Also try cninfo for sector/industry (best-effort)
            try:
                df = ak.stock_profile_cninfo(symbol=ak_sym)
                if df is not None and not df.empty:
                    profile = df.iloc[0]
                    industry = profile.get("所属行业")
                    if industry and str(industry).strip():
                        lines.append(f"Industry: {industry}")
                    established = profile.get("公司成立日期")
                    if established and str(established).strip():
                        lines.append(f"Established: {established}")
            except Exception:
                pass

            if not lines:
                raise NoMarketDataError(
                    ticker, ak_sym, "no A-share company info from akshare"
                )

        elif market == "hk":
            df = ak.stock_hk_company_profile_em(symbol=ak_sym)
            if df is None or df.empty:
                raise NoMarketDataError(
                    ticker, ak_sym, "no HK profile from akshare"
                )
            row = df.iloc[0]
            lines = []
            field_map = {
                "公司名称": "Name",
                "英文名称": "English Name",
                "所属行业": "Industry",
                "公司成立日期": "Established",
                "注册地": "Registered In",
                "董事长": "Chairman",
                "员工人数": "Employees",
                "办公地址": "Address",
                "公司网址": "Website",
                "公司介绍": "Description",
            }
            for cn_key, en_label in field_map.items():
                val = row.get(cn_key)
                if val is not None and str(val).strip():
                    lines.append(f"{en_label}: {val}")

        else:
            # US / unknown — raise so yfinance picks it up
            raise NoMarketDataError(
                ticker,
                ak_sym,
                f"akshare fundamentals not available for market '{market}'",
            )

    except NoMarketDataError:
        raise
    except _NETWORK_ERRORS:
        raise
    except Exception as exc:
        raise NoMarketDataError(
            ticker, ak_sym, f"akshare fundamentals error: {exc}"
        ) from exc

    header = (
        f"# Company Fundamentals for {display}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Source: akshare\n\n"
    )
    return header + "\n".join(lines)


# ---------------------------------------------------------------------------
# Financial statements
# ---------------------------------------------------------------------------


def _pivot_financial_long_to_wide(
    df: pd.DataFrame,
    item_col: str = "ITEM_NAME",
    date_col: str = "REPORT_DATE",
    value_col: str = "AMOUNT",
) -> pd.DataFrame:
    """Pivot akshare long-format financials to wide (rows=items, cols=dates)."""
    if df is None or df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index=item_col,
        columns=date_col,
        values=value_col,
        aggfunc="first",
    )
    pivot = pivot.apply(pd.to_numeric, errors="coerce")
    return pivot


def _filter_financials_by_date(
    data: pd.DataFrame, curr_date: Optional[str]
) -> pd.DataFrame:
    """Drop financial statement columns (fiscal-period timestamps) after curr_date."""
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]


def get_balance_sheet_akshare(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str = None,
) -> str:
    """Balance sheet via akshare."""
    market, ak_sym, display = _extract_market(ticker)

    import akshare as ak

    try:
        if market == "us":
            indicator = "年报" if freq.lower() == "annual" else "累计季报"
            df = ak.stock_financial_us_report_em(
                stock=ak_sym, symbol="资产负债表", indicator=indicator
            )
            if df is None or df.empty:
                raise NoMarketDataError(
                    ticker, ak_sym, "no US balance sheet from akshare"
                )
            data = _pivot_financial_long_to_wide(df)
            data = _filter_financials_by_date(data, curr_date)

        elif market == "cn":
            df = ak.stock_balance_sheet_by_report_em(symbol=ak_sym)
            if df is None or df.empty:
                raise NoMarketDataError(
                    ticker, ak_sym, "no CN balance sheet from akshare"
                )
            # A-share BS comes wide-format with 'REPORT_DATE' column
            # Transpose: items as rows, dates as columns
            if "REPORT_DATE" in df.columns:
                data = df.set_index("REPORT_DATE").T
            else:
                data = df
            data = _filter_financials_by_date(data, curr_date)

        else:
            raise NoMarketDataError(
                ticker,
                ak_sym,
                f"akshare balance sheet not available for market '{market}'",
            )
    except NoMarketDataError:
        raise
    except _NETWORK_ERRORS:
        raise
    except Exception as exc:
        raise NoMarketDataError(
            ticker, ak_sym, f"akshare balance sheet error: {exc}"
        ) from exc

    if data.empty:
        raise NoMarketDataError(
            ticker, ak_sym, "balance sheet empty after filtering"
        )

    header = (
        f"# Balance Sheet data for {display} ({freq})\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Source: akshare\n\n"
    )
    return header + data.to_csv()


def get_cashflow_akshare(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str = None,
) -> str:
    """Cash flow statement via akshare."""
    market, ak_sym, display = _extract_market(ticker)

    import akshare as ak

    try:
        if market == "us":
            indicator = "年报" if freq.lower() == "annual" else "累计季报"
            df = ak.stock_financial_us_report_em(
                stock=ak_sym, symbol="现金流量表", indicator=indicator
            )
            if df is None or df.empty:
                raise NoMarketDataError(
                    ticker, ak_sym, "no US cash flow from akshare"
                )
            data = _pivot_financial_long_to_wide(df)
            data = _filter_financials_by_date(data, curr_date)

        elif market == "cn":
            df = ak.stock_cash_flow_sheet_by_report_em(symbol=ak_sym)
            if df is None or df.empty:
                raise NoMarketDataError(
                    ticker, ak_sym, "no CN cash flow from akshare"
                )
            if "REPORT_DATE" in df.columns:
                data = df.set_index("REPORT_DATE").T
            else:
                data = df
            data = _filter_financials_by_date(data, curr_date)

        else:
            raise NoMarketDataError(
                ticker,
                ak_sym,
                f"akshare cash flow not available for market '{market}'",
            )
    except NoMarketDataError:
        raise
    except _NETWORK_ERRORS:
        raise
    except Exception as exc:
        raise NoMarketDataError(
            ticker, ak_sym, f"akshare cash flow error: {exc}"
        ) from exc

    if data.empty:
        raise NoMarketDataError(
            ticker, ak_sym, "cash flow empty after filtering"
        )

    header = (
        f"# Cash Flow data for {display} ({freq})\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Source: akshare\n\n"
    )
    return header + data.to_csv()


def get_income_statement_akshare(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str = None,
) -> str:
    """Income statement via akshare."""
    market, ak_sym, display = _extract_market(ticker)

    import akshare as ak

    try:
        if market == "us":
            indicator = "年报" if freq.lower() == "annual" else "累计季报"
            df = ak.stock_financial_us_report_em(
                stock=ak_sym, symbol="综合损益表", indicator=indicator
            )
            if df is None or df.empty:
                raise NoMarketDataError(
                    ticker, ak_sym, "no US income statement from akshare"
                )
            data = _pivot_financial_long_to_wide(df)
            data = _filter_financials_by_date(data, curr_date)

        elif market == "cn":
            df = ak.stock_profit_sheet_by_report_em(symbol=ak_sym)
            if df is None or df.empty:
                raise NoMarketDataError(
                    ticker, ak_sym, "no CN income statement from akshare"
                )
            if "REPORT_DATE" in df.columns:
                data = df.set_index("REPORT_DATE").T
            else:
                data = df
            data = _filter_financials_by_date(data, curr_date)

        else:
            raise NoMarketDataError(
                ticker,
                ak_sym,
                f"akshare income statement not available for market '{market}'",
            )
    except NoMarketDataError:
        raise
    except _NETWORK_ERRORS:
        raise
    except Exception as exc:
        raise NoMarketDataError(
            ticker, ak_sym, f"akshare income statement error: {exc}"
        ) from exc

    if data.empty:
        raise NoMarketDataError(
            ticker, ak_sym, "income statement empty after filtering"
        )

    header = (
        f"# Income Statement data for {display} ({freq})\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Source: akshare\n\n"
    )
    return header + data.to_csv()


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------


def _news_timestamp(value: Any) -> pd.Timestamp | None:
    """Parse a source timestamp as a timezone-naive value."""
    if value is None or not str(value).strip() or str(value).lower() == "nan":
        return None
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)
        return timestamp
    except (TypeError, ValueError):
        return None


def _news_in_window(value: Any, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> bool:
    """Return whether a verifiable publication time is in the inclusive window."""
    timestamp = _news_timestamp(value)
    return timestamp is not None and start_dt <= timestamp < end_dt + pd.Timedelta(days=1)


def _first_text(row: pd.Series, *columns: str) -> str:
    for column in columns:
        value = row.get(column)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value).strip()
    return ""


def _render_news_item(
    *,
    title: str,
    content: str = "",
    published: str = "",
    link: str = "",
) -> str:
    lines = [f"### {title or '(untitled)'}"]
    if content:
        lines.append(content)
    if published:
        lines.append(f"Published: {published}")
    if link:
        lines.append(f"Link: {link}")
    return "\n".join(lines)


def get_news_akshare(ticker: str, start_date: str, end_date: str) -> str:
    """A-share company news and official disclosures via AKShare.

    The public signature and string return type match the existing yfinance
    vendor. Every included item has a publication time inside the requested
    range; undated items are excluded because they cannot be made look-ahead
    safe.
    """
    market, ak_sym, display = _extract_market(ticker)

    if market != "cn":
        raise NoMarketDataError(
            ticker,
            ak_sym,
            f"akshare news only supports A-share tickers (market='{market}')",
        )

    import akshare as ak

    start_dt = pd.Timestamp(datetime.strptime(start_date, "%Y-%m-%d"))
    end_dt = pd.Timestamp(datetime.strptime(end_date, "%Y-%m-%d"))
    if start_dt > end_dt:
        raise ValueError("start_date must be on or before end_date")

    bare_code = ak_sym.replace("SH", "").replace("SZ", "")
    article_limit = max(1, int(get_config().get("news_article_limit", 20)))
    eastmoney_items: list[str] = []
    cninfo_items: list[str] = []
    errors: list[str] = []

    # Company news from East Money.
    try:
        frame = ak.stock_news_em(symbol=bare_code)
        if frame is not None and not frame.empty:
            for _, row in frame.iterrows():
                published = _first_text(row, "发布时间", "时间", "日期")
                if not _news_in_window(published, start_dt, end_dt):
                    continue
                eastmoney_items.append(
                    _render_news_item(
                        title=_first_text(row, "新闻标题", "标题"),
                        content=_first_text(row, "新闻内容", "内容", "摘要"),
                        published=published,
                        link=_first_text(row, "新闻链接", "链接", "网址"),
                    )
                )
                if len(eastmoney_items) >= article_limit:
                    break
    except Exception as exc:
        errors.append(f"East Money: {type(exc).__name__}")

    # Exchange-regulated disclosures from CNInfo.
    try:
        frame = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=bare_code,
            market="沪深京",
            keyword="",
            category="",
            start_date=_to_ak_date(start_date),
            end_date=_to_ak_date(end_date),
        )
        if frame is not None and not frame.empty:
            for _, row in frame.iterrows():
                published = _first_text(row, "公告时间", "发布时间", "日期")
                if not _news_in_window(published, start_dt, end_dt):
                    continue
                cninfo_items.append(
                    _render_news_item(
                        title=_first_text(row, "公告标题", "标题"),
                        published=published,
                        link=_first_text(row, "公告链接", "链接"),
                    )
                )
                if len(cninfo_items) >= article_limit:
                    break
    except Exception as exc:
        errors.append(f"CNInfo: {type(exc).__name__}")

    if not eastmoney_items and not cninfo_items:
        detail = "; ".join(errors) if errors else "no dated items in requested range"
        raise NoMarketDataError(
            ticker,
            ak_sym,
            f"akshare returned no usable company news/disclosures: {detail}",
        )

    # Share the configured article budget between both sources so official
    # disclosures cannot be crowded out by a long media feed (and vice versa).
    selected_eastmoney: list[str] = []
    selected_cninfo: list[str] = []
    source_queues = (
        (cninfo_items, selected_cninfo),
        (eastmoney_items, selected_eastmoney),
    )
    while len(selected_eastmoney) + len(selected_cninfo) < article_limit:
        added = False
        for source_items, selected_items in source_queues:
            index = len(selected_items)
            if index < len(source_items):
                selected_items.append(source_items[index])
                added = True
                if len(selected_eastmoney) + len(selected_cninfo) >= article_limit:
                    break
        if not added:
            break

    sections: list[str] = []
    if selected_eastmoney:
        sections.append(
            "## East Money company news\n\n" + "\n\n".join(selected_eastmoney)
        )
    if selected_cninfo:
        sections.append(
            "## CNInfo official disclosures\n\n" + "\n\n".join(selected_cninfo)
        )

    return f"# {display} News and Disclosures, from {start_date} to {end_date}\n\n" + "\n\n".join(
        sections
    )


def get_global_news_akshare(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """China-focused financial and macro news via AKShare.

    CLS flashes are filtered by their publication timestamp. CCTV is queried
    for the exact analysis date. Caixin's endpoint is a current, undated
    snapshot, so it is included only for a current-day analysis.
    """
    import akshare as ak

    config = get_config()
    if look_back_days is None:
        look_back_days = int(config["global_news_lookback_days"])
    if limit is None:
        limit = int(config["global_news_article_limit"])
    look_back_days = max(0, int(look_back_days))
    limit = max(1, int(limit))

    end_dt = pd.Timestamp(datetime.strptime(curr_date, "%Y-%m-%d"))
    start_dt = end_dt - pd.Timedelta(days=look_back_days)
    candidates: list[dict[str, Any]] = []

    # Financial flashes from CLS with explicit timestamps.
    try:
        frame = ak.stock_info_global_cls(symbol="全部")
        if frame is not None and not frame.empty:
            for _, row in frame.iterrows():
                published_date = _first_text(row, "发布日期", "日期")
                published_time = _first_text(row, "发布时间", "时间")
                published = (
                    f"{published_date} {published_time}".strip()
                    if published_date
                    else published_time
                )
                if not _news_in_window(published, start_dt, end_dt):
                    continue
                candidates.append(
                    {
                        "source": "CLS",
                        "title": _first_text(row, "标题", "新闻标题"),
                        "content": _first_text(row, "内容", "摘要"),
                        "published": published,
                        "timestamp": _news_timestamp(published),
                        "link": _first_text(row, "链接", "网址"),
                    }
                )
    except Exception:
        pass

    # CCTV data is keyed by the exact requested date, so its otherwise-undated
    # rows are safe to associate with that date.
    try:
        frame = ak.news_cctv(date=_to_ak_date(curr_date))
        if frame is not None and not frame.empty:
            for _, row in frame.iterrows():
                title = _first_text(row, "title", "标题", "新闻标题")
                if not title and len(row) > 1:
                    title = str(row.iloc[1]).strip()
                candidates.append(
                    {
                        "source": "CCTV",
                        "title": title,
                        "content": _first_text(row, "content", "内容", "摘要"),
                        "published": curr_date,
                        "timestamp": end_dt,
                        "link": _first_text(row, "link", "链接", "网址"),
                    }
                )
    except Exception:
        pass

    # This source has no reliable point-in-time parameter. Restrict it to today
    # instead of leaking the latest headlines into a historical analysis.
    if end_dt.date() == datetime.now().date():
        try:
            frame = ak.stock_news_main_cx()
            if frame is not None and not frame.empty:
                for _, row in frame.iterrows():
                    title = _first_text(row, "标题", "新闻标题", "summary")
                    if not title and len(row) > 1:
                        title = str(row.iloc[1]).strip()
                    candidates.append(
                        {
                            "source": "Caixin current snapshot",
                            "title": title,
                            "content": _first_text(row, "内容", "摘要"),
                            "published": curr_date,
                            "timestamp": end_dt,
                            "link": _first_text(row, "链接", "网址", "url"),
                        }
                    )
        except Exception:
            pass

    # Deduplicate across syndicated feeds, newest first.
    candidates.sort(
        key=lambda item: item.get("timestamp") or pd.Timestamp.min,
        reverse=True,
    )
    seen_titles: set[str] = set()
    rendered: list[str] = []
    for item in candidates:
        title = str(item.get("title", "")).strip()
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        rendered.append(
            _render_news_item(
                title=f"{title} (source: {item['source']})",
                content=str(item.get("content", "")).strip(),
                published=str(item.get("published", "")).strip(),
                link=str(item.get("link", "")).strip(),
            )
        )
        if len(rendered) >= limit:
            break

    if not rendered:
        raise NoMarketDataError(
            "GLOBAL",
            "GLOBAL",
            f"akshare returned no dated China news between "
            f"{start_dt.strftime('%Y-%m-%d')} and {curr_date}",
        )

    return (
        f"# China Market News, from {start_dt.strftime('%Y-%m-%d')} "
        f"to {curr_date}\n\n" + "\n\n".join(rendered)
    )


# ---------------------------------------------------------------------------
# Insider transactions
# ---------------------------------------------------------------------------


def get_insider_transactions_akshare(ticker: str) -> str:
    """akshare does not provide insider transaction data.

    Always raises *NoMarketDataError* so the router falls back to yfinance.
    """
    raise NoMarketDataError(
        ticker,
        ticker,
        "akshare does not support insider transactions",
    )


# ---------------------------------------------------------------------------
# Extended tools: fund flow, LHB, institute holdings
# ---------------------------------------------------------------------------


def get_fund_flow_akshare(ticker: str, curr_date: str = None) -> str:
    """个股资金流向 via akshare (同花顺数据中心).

    Returns recent capital flow data: inflow, outflow, net flow, turnover rate.
    Only covers A-shares (CN market).
    """
    market, ak_sym, display = _extract_market(ticker)
    if market != "cn":
        raise NoMarketDataError(
            ticker, ak_sym,
            "akshare fund flow only available for A-share stocks",
        )

    import akshare as ak
    bare_code = ak_sym.replace("SH", "").replace("SZ", "")

    try:
        df = ak.stock_fund_flow_individual(symbol="即时")
    except _NETWORK_ERRORS:
        raise
    except Exception as exc:
        raise NoMarketDataError(
            ticker, ak_sym, f"akshare fund flow error: {exc}"
        ) from exc

    if df is None or df.empty:
        raise NoMarketDataError(ticker, ak_sym, "fund flow data empty")

    # The column is '股票代码' in the DataFrame
    code_col = "股票代码" if "股票代码" in df.columns else df.columns[1]
    row = df[df[code_col].astype(str).str.strip() == bare_code]
    if row.empty:
        raise NoMarketDataError(
            ticker, ak_sym,
            f"no fund flow data for {display} (may not be in top flow list today)",
        )

    r = row.iloc[0]
    lines = [
        f"# 资金流向 for {display}",
        f"# Source: akshare (同花顺数据中心)",
        f"# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    col_map = {
        "最新价": "Latest Price",
        "涨跌幅": "Change %",
        "换手率": "Turnover Rate",
        "流入资金": "Inflow",
        "流出资金": "Outflow",
        "净额": "Net Flow",
        "成交额": "Total Volume",
    }
    for cn_label, en_label in col_map.items():
        if cn_label in r.index:
            val = r[cn_label]
            lines.append(f"- {en_label}: {val}")
    return "\n".join(lines)


def get_lhb_detail_akshare(ticker: str, curr_date: str = None) -> str:
    """龙虎榜详情 via akshare (东方财富).

    Returns recent dragon-tiger board appearance details for a stock:
    buy/sell amounts by seats, net buy amount,上榜 reason, post-listing returns.
    Only covers A-shares.
    """
    market, ak_sym, display = _extract_market(ticker)
    if market != "cn":
        raise NoMarketDataError(
            ticker, ak_sym,
            "akshare LHB detail only available for A-share stocks",
        )

    import akshare as ak
    from datetime import timedelta
    bare_code = ak_sym.replace("SH", "").replace("SZ", "")

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=60)
    try:
        df = ak.stock_lhb_detail_em(
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=end_dt.strftime("%Y%m%d"),
        )
    except _NETWORK_ERRORS:
        raise
    except Exception as exc:
        raise NoMarketDataError(
            ticker, ak_sym, f"akshare LHB error: {exc}"
        ) from exc

    if df is None or df.empty:
        raise NoMarketDataError(ticker, ak_sym, "LHB data empty")

    code_col = "代码" if "代码" in df.columns else df.columns[1]
    rows = df[df[code_col].astype(str).str.strip() == bare_code]
    if rows.empty:
        raise NoMarketDataError(
            ticker, ak_sym,
            f"{display} 近60日未上龙虎榜",
        )

    lines = [
        f"# 龙虎榜详情 for {display}",
        f"# Source: akshare (东方财富数据中心)",
        f"# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for _, r in rows.head(10).iterrows():
        lines.append(
            f"- {r.get('上榜日', '?')} | "
            f"涨跌幅: {r.get('涨跌幅', '?')} | "
            f"净买额: {r.get('龙虎榜净买额', '?')} | "
            f"买入: {r.get('龙虎榜买入额', '?')} | "
            f"卖出: {r.get('龙虎榜卖出额', '?')} | "
            f"原因: {r.get('上榜原因', '?')}"
        )
        # Post-LHB returns if available
        post = []
        for d in ["1日", "2日", "5日", "10日"]:
            col = f"上榜后{d}"
            if col in r.index and pd.notna(r[col]):
                post.append(f"后{d}: {r[col]}")
        if post:
            lines.append(f"  上榜后表现: {' | '.join(post)}")

    if len(rows) == 0:
        lines.append(f"(无近期龙虎榜记录)")
    return "\n".join(lines)


def get_institute_hold_akshare(ticker: str, curr_date: str = None) -> str:
    """机构持仓 via akshare (新浪财经).

    Returns institutional holdings data: number of institutions, holding
    ratio, circulating share ratio for the latest reporting period.
    Only covers A-shares.
    """
    market, ak_sym, display = _extract_market(ticker)
    if market != "cn":
        raise NoMarketDataError(
            ticker, ak_sym,
            "akshare institute hold only available for A-share stocks",
        )

    import akshare as ak
    bare_code = ak_sym.replace("SH", "").replace("SZ", "")

    # Try recent quarters. The naming convention is YYYYQ where Q is
    # 1/2/3/4 for each reporting period. Start from the most recent
    # and work backwards until we find data.
    current_year = datetime.now().year
    current_month = datetime.now().month
    # Determine which quarter's data is most likely available
    if current_month >= 5:
        latest = f"{current_year}1"  # Q1 report available after April
    else:
        latest = f"{current_year-1}4"
    year = int(latest[:4])
    qtr = int(latest[4])
    quarters = []
    for _ in range(8):  # try up to 8 quarters back
        quarters.append(f"{year}{qtr}")
        qtr -= 1
        if qtr == 0:
            qtr = 4
            year -= 1

    all_rows = []
    for q in quarters:
        try:
            df = ak.stock_institute_hold(symbol=q)
        except _NETWORK_ERRORS:
            raise
        except Exception:
            continue

        if df is None or df.empty:
            continue

        code_col = "证券代码" if "证券代码" in df.columns else df.columns[1]
        row = df[df[code_col].astype(str).str.strip() == bare_code]
        if not row.empty:
            all_rows.append((q, row.iloc[0]))

    if not all_rows:
        raise NoMarketDataError(
            ticker, ak_sym,
            f"no institute holding data for {display}",
        )

    lines = [
        f"# 机构持仓 for {display}",
        f"# Source: akshare (新浪财经)",
        f"# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for period, r in all_rows:
        name = r.get("证券简称", "?")
        inst_count = r.get("机构数", "?")
        inst_change = r.get("机构数变化", "?")
        hold_ratio = r.get("持股比例", "?")
        hold_change = r.get("持股比例增幅", "?")
        circ_ratio = r.get("占流通股比例", "?")
        circ_change = r.get("占流通股比例增幅", "?")

        period_label = f"{period[:4]}年{'Q1' if period[4]=='1' else 'Q2' if period[4]=='2' else 'Q3' if period[4]=='3' else 'Q4'}"
        lines.append(
            f"## {period_label}\n"
            f"- 证券简称: {name}\n"
            f"- 机构数: {inst_count} (变化: {inst_change})\n"
            f"- 持股比例: {hold_ratio} (变化: {hold_change})\n"
            f"- 占流通股比例: {circ_ratio} (变化: {circ_change})\n"
        )

    return "\n".join(lines)
