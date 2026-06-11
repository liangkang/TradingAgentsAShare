import functools
import logging
from typing import Any, Mapping, Optional

import yfinance as yf
from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)
from tradingagents.agents.utils.market_data_validation_tools import (
    get_verified_market_snapshot
)

logger = logging.getLogger(__name__)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def _clean_identity_value(value: Any) -> Optional[str]:
    """Return a trimmed string, or None for empty / placeholder-ish values."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "nan", "null"}:
        return None
    return cleaned


@functools.lru_cache(maxsize=256)
def resolve_instrument_identity(ticker: str) -> dict:
    """Resolve deterministic identity metadata (company name, sector, …) for a ticker.

    This exists to stop the pipeline from hallucinating a *different* company
    when a chart pattern suggests a different industry than the real one
    (#814): without a ground-truth name, the market analyst would pattern-match
    the price action to a narrative and invent an identity that then cascaded
    through every downstream agent.

    Tries yfinance first (best global coverage), then falls back to akshare
    for Chinese and HK markets. Best-effort by design: if all sources fail,
    returns ``{}`` and the caller falls back to ticker-only context rather
    than failing before analysis starts. Cached so the lookup happens at most
    once per ticker per process.
    """
    info: dict[str, str] = {}
    try:
        yf_info = yf.Ticker(ticker.upper()).info or {}
    except Exception as exc:  # noqa: BLE001 — fail open, never block the run
        logger.debug("yfinance identity unavailable for %s: %s", ticker, exc)
        yf_info = {}

    identity: dict[str, str] = {}
    company_name = _clean_identity_value(yf_info.get("longName")) or _clean_identity_value(
        yf_info.get("shortName")
    )
    if company_name:
        identity["company_name"] = company_name
    for source_key, target_key in (
        ("sector", "sector"),
        ("industry", "industry"),
        ("exchange", "exchange"),
        ("quoteType", "quote_type"),
    ):
        value = _clean_identity_value(yf_info.get(source_key))
        if value:
            identity[target_key] = value

    # If yfinance returned usable identity, return it immediately.
    if identity:
        return identity

    # --- akshare fallback for CN and HK tickers ---
    upper = ticker.upper()
    try:
        import akshare as ak

        if upper.endswith(".SS"):
            bare_code = upper[:-3]
            identity["exchange"] = "Shanghai Stock Exchange"
            identity["quote_type"] = "EQUITY"
            # Use stock_info_a_code_name for reliable name lookup
            try:
                name_df = ak.stock_info_a_code_name()
                row = name_df[name_df["code"] == bare_code]
                if not row.empty:
                    cn_name = _clean_identity_value(row.iloc[0].get("name"))
                    if cn_name:
                        identity["company_name"] = cn_name
            except Exception:
                pass
            # Best-effort industry from profile
            try:
                df = ak.stock_profile_cninfo(symbol="SH" + bare_code)
                if df is not None and not df.empty:
                    industry = _clean_identity_value(df.iloc[0].get("所属行业"))
                    if industry:
                        identity["industry"] = industry
            except Exception:
                pass
        elif upper.endswith(".SZ"):
            bare_code = upper[:-3]
            identity["exchange"] = "Shenzhen Stock Exchange"
            identity["quote_type"] = "EQUITY"
            try:
                name_df = ak.stock_info_a_code_name()
                row = name_df[name_df["code"] == bare_code]
                if not row.empty:
                    cn_name = _clean_identity_value(row.iloc[0].get("name"))
                    if cn_name:
                        identity["company_name"] = cn_name
            except Exception:
                pass
            try:
                df = ak.stock_profile_cninfo(symbol="SZ" + bare_code)
                if df is not None and not df.empty:
                    industry = _clean_identity_value(df.iloc[0].get("所属行业"))
                    if industry:
                        identity["industry"] = industry
            except Exception:
                pass
        elif upper.endswith(".HK"):
            code = upper[:-3].lstrip("0").zfill(5)
            df = ak.stock_hk_company_profile_em(symbol=code)
            if df is not None and not df.empty:
                row = df.iloc[0]
                cn_name = _clean_identity_value(row.get("公司名称"))
                if cn_name:
                    identity["company_name"] = cn_name
                industry = _clean_identity_value(row.get("所属行业"))
                if industry:
                    identity["industry"] = industry
                identity["exchange"] = "Hong Kong Stock Exchange"
                identity["quote_type"] = "EQUITY"
    except Exception as exc:
        logger.debug("akshare identity fallback failed for %s: %s", ticker, exc)

    return identity


def build_instrument_context(
    ticker: str,
    asset_type: str = "stock",
    identity: Optional[Mapping[str, str]] = None,
) -> str:
    """Describe the exact instrument so agents preserve identity and ticker.

    When ``identity`` is provided (resolved deterministically via
    :func:`resolve_instrument_identity`), the company name and business
    classification are injected so agents anchor to the real company rather
    than pattern-matching the price chart to a wrong one (#814).
    """
    is_crypto = asset_type == "crypto"
    instrument_label = "asset" if is_crypto else "instrument"
    context = (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
    )

    details = []
    if identity:
        name = identity.get("company_name") or identity.get("name")
        if name:
            details.append(f"{'Name' if is_crypto else 'Company'}: {name}")
        sector, industry = identity.get("sector"), identity.get("industry")
        if sector and industry:
            details.append(f"Business classification: {sector} / {industry}")
        elif sector:
            details.append(f"Sector: {sector}")
        elif industry:
            details.append(f"Industry: {industry}")
        if identity.get("exchange"):
            details.append(f"Exchange: {identity['exchange']}")

    if details:
        context += (
            f" Resolved identity: {'; '.join(details)}. "
            "Do not substitute a different company or ticker unless a tool "
            "result explicitly disproves this resolved identity."
        )

    if is_crypto:
        context += (
            " Treat it as a crypto asset rather than a company, and do not "
            "assume company fundamentals are available."
        )
    return context


def get_instrument_context_from_state(state: Mapping[str, Any]) -> str:
    """Return the instrument context for the current run.

    Prefers the identity-resolved context computed once at run start and
    stored on the state (see ``TradingAgentsGraph.resolve_instrument_context``).
    Falls back to a ticker-only context — with no network lookup — when the
    state was constructed without it (bare programmatic states, tests), so a
    consumer is never forced to make a yfinance call mid-graph.
    """
    context = state.get("instrument_context")
    if isinstance(context, str) and context.strip():
        return context
    return build_instrument_context(
        str(state["company_of_interest"]),
        state.get("asset_type", "stock"),
    )


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add a context-anchored placeholder.

        The placeholder must not be a bare ``"Continue"``: some
        OpenAI-compatible providers interpret that literally as the user task
        and produce output about the word "continue" instead of analysing the
        instrument (#888). Anchoring it to the resolved instrument context and
        date keeps the next analyst on-task even if the provider treats the
        placeholder as a standalone request.
        """
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        instrument_context = get_instrument_context_from_state(state)
        trade_date = state.get("trade_date", "the requested date")
        placeholder = HumanMessage(
            content=(
                f"Proceed with your assigned analysis for this workflow. "
                f"{instrument_context} The analysis date is {trade_date}."
            )
        )
        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
