from __future__ import annotations

import pytest

import tradingagents.agents.tools as news_tools


def _capture_route(monkeypatch) -> dict:
    captured = {}

    def route(method, *args, **kwargs):
        captured.update(method=method, args=args, kwargs=kwargs)
        return "news"

    monkeypatch.setattr(news_tools, "route_to_vendor", route)
    return captured


@pytest.mark.unit
def test_global_news_excludes_akshare_for_us_ticker(monkeypatch):
    captured = _capture_route(monkeypatch)

    result = news_tools.get_global_news.func("2025-05-09", 7, 10, ticker="AAPL")

    assert result == "news"
    assert captured["method"] == "get_global_news"
    assert captured["kwargs"]["_exclude_vendors"] == {"akshare"}


@pytest.mark.unit
@pytest.mark.parametrize("ticker", ["300750.SZ", "600519.SH", "600519.SS"])
def test_global_news_keeps_akshare_for_a_share_ticker(monkeypatch, ticker):
    captured = _capture_route(monkeypatch)

    news_tools.get_global_news.func("2025-05-09", ticker=ticker)

    assert captured["kwargs"]["_exclude_vendors"] == set()


@pytest.mark.unit
def test_global_news_call_without_ticker_keeps_every_vendor(monkeypatch):
    captured = _capture_route(monkeypatch)

    news_tools.get_global_news.func("2025-05-09", 7, 10)

    assert captured["args"] == ("2025-05-09", 7, 10)
    assert captured["kwargs"]["_exclude_vendors"] == set()


@pytest.mark.unit
def test_the_model_never_passes_the_ticker_for_global_news():
    """The instrument comes from graph state, so the model can neither omit nor mistype it."""
    model_args = news_tools.get_global_news.tool_call_schema.model_json_schema()["properties"]

    assert "ticker" not in model_args
    assert "trade_date" not in model_args
    assert "curr_date" in model_args
