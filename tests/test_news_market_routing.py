from __future__ import annotations

import inspect

import pytest

import tradingagents.agents.analysts.news_analyst as news_analyst
import tradingagents.agents.utils.news_data_tools as news_tools


@pytest.mark.unit
def test_global_news_excludes_akshare_for_us_ticker(monkeypatch):
    captured = {}

    def route(method, *args, **kwargs):
        captured.update(method=method, args=args, kwargs=kwargs)
        return "news"

    monkeypatch.setattr(news_tools, "route_to_vendor", route)

    result = news_tools.get_global_news.func(
        "2025-05-09",
        7,
        10,
        "AAPL",
    )

    assert result == "news"
    assert captured["method"] == "get_global_news"
    assert captured["kwargs"]["_exclude_vendors"] == {"akshare"}


@pytest.mark.unit
@pytest.mark.parametrize("ticker", ["300750.SZ", "600519.SH", "600519.SS"])
def test_global_news_keeps_akshare_for_a_share_ticker(monkeypatch, ticker):
    captured = {}

    def route(method, *args, **kwargs):
        captured.update(method=method, args=args, kwargs=kwargs)
        return "news"

    monkeypatch.setattr(news_tools, "route_to_vendor", route)

    news_tools.get_global_news.func("2025-05-09", ticker=ticker)

    assert captured["kwargs"]["_exclude_vendors"] == set()


@pytest.mark.unit
def test_global_news_legacy_call_without_ticker_remains_supported(monkeypatch):
    captured = {}

    def route(method, *args, **kwargs):
        captured.update(method=method, args=args, kwargs=kwargs)
        return "news"

    monkeypatch.setattr(news_tools, "route_to_vendor", route)

    news_tools.get_global_news.func("2025-05-09", 7, 10)

    assert captured["args"] == ("2025-05-09", 7, 10)
    assert captured["kwargs"]["_exclude_vendors"] == set()


@pytest.mark.unit
def test_news_analyst_prompt_requests_market_aware_global_news():
    source = inspect.getsource(news_analyst)

    assert "get_global_news(curr_date, look_back_days, limit, ticker)" in source
    assert "always pass the current ticker" in source
