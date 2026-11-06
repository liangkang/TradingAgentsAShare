"""Extended analysis tools leveraging akshare-specific data sources.

These tools are only available for A-share stocks (CN market).  For other
markets the vendor router will raise NoMarketDataError.

Like the core tools in ``tradingagents.agents.tools``, each takes the
instrument and the run's ``trade_date`` from graph state (``InjectedState``)
and never serves data past it, whatever date the model asks for.
"""

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from tradingagents.agents.context import is_cn_ticker
from tradingagents.dataflows.date_window import as_of
from tradingagents.dataflows.router import route_to_vendor

# The tools below serve China A-shares only; an analyst offers them only on one.
A_SHARE_TOOL_NAMES = frozenset({"get_fund_flow", "get_lhb_detail", "get_institute_hold"})


def offered_tools(tools, ticker: str) -> tuple:
    """The analyst's tools for ``ticker``: the A-share-only ones only for an A-share.

    Offered elsewhere they would only spend tool rounds on a NO_DATA answer. The
    analyst's tool node still holds all of them, so every offered tool can run.
    """
    if is_cn_ticker(ticker):
        return tuple(tools)
    return tuple(t for t in tools if t.name not in A_SHARE_TOOL_NAMES)


@tool
def get_fund_flow(
    ticker: Annotated[str, InjectedState("company_of_interest")],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    trade_date: Annotated[str, InjectedState("trade_date")] = "",
) -> str:
    """Retrieve capital flow data (inflow, outflow, net flow) for an A-share stock.

    Uses akshare's 同花顺 fund flow endpoint.  Shows real-time or daily
    capital movement — a leading signal of institutional / retail sentiment.
    Only available for CN-market tickers (.SS / .SZ).

    Args:
        curr_date: Current analysis date, yyyy-mm-dd
    Returns:
        Formatted report with latest price, change%, turnover, inflow,
        outflow, and net capital flow.
    """
    return route_to_vendor("get_fund_flow", ticker, as_of(curr_date, trade_date))


@tool
def get_lhb_detail(
    ticker: Annotated[str, InjectedState("company_of_interest")],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    trade_date: Annotated[str, InjectedState("trade_date")] = "",
) -> str:
    """Retrieve dragon-tiger board (龙虎榜) detail for an A-share stock.

    The dragon-tiger board lists the top brokerage seats buying/selling a
    stock that hit daily price-move thresholds.  It reveals which major
    players (institutional / retail / proprietary) are active in the stock
    and is a key short-term signal in the A-share market.  Only available
    for CN-market tickers (.SS / .SZ).

    Args:
        curr_date: Current analysis date
    Returns:
        Formatted report with最近60日上榜记录: seat-level buy/sell amounts,
        net buy,上榜 reason, and post-listing returns (1d/2d/5d/10d).
    """
    return route_to_vendor("get_lhb_detail", ticker, as_of(curr_date, trade_date))


@tool
def get_institute_hold(
    ticker: Annotated[str, InjectedState("company_of_interest")],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    trade_date: Annotated[str, InjectedState("trade_date")] = "",
) -> str:
    """Retrieve institutional holdings data for an A-share stock.

    Shows the number of institutions holding the stock, their aggregate
    holding ratio, and the circulating-share ratio across recent reporting
    quarters.  Increasing institutional ownership is generally a bullish
    signal (professional investors increasing exposure).  Only available
    for CN-market tickers (.SS / .SZ).

    Args:
        curr_date: Current analysis date
    Returns:
        Formatted report with institution count, holding ratio, and
        circulating-share ratio per reporting period.
    """
    return route_to_vendor("get_institute_hold", ticker, as_of(curr_date, trade_date))
