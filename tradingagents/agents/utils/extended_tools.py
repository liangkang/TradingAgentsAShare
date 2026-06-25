"""Extended analysis tools leveraging akshare-specific data sources.

These tools are only available for A-share stocks (CN market).  For other
markets the vendor router will raise NoMarketDataError.
"""

from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_fund_flow(
    ticker: Annotated[str, "ticker symbol, e.g. 600028.SS"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """Retrieve capital flow data (inflow, outflow, net flow) for an A-share stock.

    Uses akshare's 同花顺 fund flow endpoint.  Shows real-time or daily
    capital movement — a leading signal of institutional / retail sentiment.
    Only available for CN-market tickers (.SS / .SZ).

    Args:
        ticker: Ticker symbol with exchange suffix (e.g. 600028.SS)
        curr_date: Current analysis date, yyyy-mm-dd
    Returns:
        Formatted report with latest price, change%, turnover, inflow,
        outflow, and net capital flow.
    """
    return route_to_vendor("get_fund_flow", ticker, curr_date)


@tool
def get_lhb_detail(
    ticker: Annotated[str, "ticker symbol, e.g. 600028.SS"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """Retrieve dragon-tiger board (龙虎榜) detail for an A-share stock.

    The dragon-tiger board lists the top brokerage seats buying/selling a
    stock that hit daily price-move thresholds.  It reveals which major
    players (institutional / retail / proprietary) are active in the stock
    and is a key short-term signal in the A-share market.  Only available
    for CN-market tickers (.SS / .SZ).

    Args:
        ticker: Ticker symbol with exchange suffix
        curr_date: Current analysis date
    Returns:
        Formatted report with最近60日上榜记录: seat-level buy/sell amounts,
        net buy,上榜 reason, and post-listing returns (1d/2d/5d/10d).
    """
    return route_to_vendor("get_lhb_detail", ticker, curr_date)


@tool
def get_institute_hold(
    ticker: Annotated[str, "ticker symbol, e.g. 600028.SS"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """Retrieve institutional holdings data for an A-share stock.

    Shows the number of institutions holding the stock, their aggregate
    holding ratio, and the circulating-share ratio across recent reporting
    quarters.  Increasing institutional ownership is generally a bullish
    signal (professional investors increasing exposure).  Only available
    for CN-market tickers (.SS / .SZ).

    Args:
        ticker: Ticker symbol with exchange suffix
        curr_date: Current analysis date
    Returns:
        Formatted report with institution count, holding ratio, and
        circulating-share ratio per reporting period.
    """
    return route_to_vendor("get_institute_hold", ticker, curr_date)
