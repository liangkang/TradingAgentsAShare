"""Agents on the schema-only structured-output path must not invite tool calls (#1130).

`with_structured_output` binds exactly one tool (the schema). A prompt that
primes tool use makes models emit an unknown `web_search` call, which discards
the structured attempt and forces a free-text retry — an extra LLM round trip
and the loss of typed output.

These assert the constraint reaches the *rendered* prompt each agent actually
sends, not merely that the constant is referenced in the module.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

import tradingagents.agents.analysts.sentiment_analyst as sentiment
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.structured import NO_EXTERNAL_TOOLS


def _capturing_llm(captured: dict, result):
    """LLM whose structured binding records the prompt it was handed."""
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: captured.__setitem__("prompt", prompt) or result
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def _prompt_text(prompt) -> str:
    """Flatten a captured prompt (str, message list, or objects) to text."""
    if isinstance(prompt, str):
        return prompt
    parts = []
    for m in prompt:
        parts.append(m.get("content", "") if isinstance(m, dict) else getattr(m, "content", ""))
    return "\n".join(str(p) for p in parts)


@pytest.mark.unit
def test_trader_prompt_states_constraint():
    from tradingagents.agents.schemas import TraderAction, TraderProposal

    captured = {}
    llm = _capturing_llm(captured, TraderProposal(action=TraderAction.BUY, reasoning="x"))
    create_trader(llm)(
        {
            "company_of_interest": "NVDA",
            "investment_plan": "**Recommendation**: Buy",
        }
    )
    assert NO_EXTERNAL_TOOLS in _prompt_text(captured["prompt"])


@pytest.mark.unit
def test_research_manager_prompt_states_constraint():
    from tradingagents.agents.schemas import PortfolioRating, ResearchPlan

    captured = {}
    llm = _capturing_llm(
        captured,
        ResearchPlan(recommendation=PortfolioRating.BUY, rationale="x", strategic_actions="y"),
    )
    create_research_manager(llm)(
        {
            "company_of_interest": "NVDA",
            "investment_debate_state": {
                "history": "h",
                "bull_history": "b",
                "bear_history": "r",
                "current_response": "",
                "judge_decision": "",
                "count": 1,
            },
        }
    )
    assert NO_EXTERNAL_TOOLS in _prompt_text(captured["prompt"])


@pytest.mark.unit
def test_portfolio_manager_prompt_states_constraint():
    from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating

    captured = {}
    llm = _capturing_llm(
        captured,
        PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="x",
            investment_thesis="y",
        ),
    )
    risk = {
        "history": "h",
        "aggressive_history": "a",
        "conservative_history": "c",
        "neutral_history": "n",
        "current_aggressive_response": "",
        "current_conservative_response": "",
        "current_neutral_response": "",
        "latest_speaker": "Neutral",
        "count": 1,
    }
    create_portfolio_manager(llm)(
        {
            "company_of_interest": "NVDA",
            "risk_debate_state": risk,
            "investment_plan": "plan",
            "trader_investment_plan": "trader plan",
        }
    )
    assert NO_EXTERNAL_TOOLS in _prompt_text(captured["prompt"])


@pytest.mark.unit
def test_sentiment_prompt_states_constraint(monkeypatch):
    from tradingagents.agents.schemas import SentimentBand, SentimentReport

    # Pre-fetched sources are stubbed so the prompt builds without network I/O.
    monkeypatch.setattr(sentiment, "fetch_stocktwits_messages", lambda *a, **k: "st")
    monkeypatch.setattr(sentiment, "fetch_reddit_posts", lambda *a, **k: "rd")
    monkeypatch.setattr(sentiment.get_news, "func", lambda *a, **k: "news", raising=False)

    captured = {}
    llm = _capturing_llm(
        captured,
        SentimentReport(
            overall_band=SentimentBand.BULLISH,
            overall_score=7.5,
            confidence="high",
            narrative="n",
        ),
    )
    sentiment.create_sentiment_analyst(llm)(
        {
            "company_of_interest": "NVDA",
            "trade_date": "2026-01-15",
            "asset_type": "stock",
            "messages": [],
        }
    )
    text = _prompt_text(captured["prompt"])
    assert NO_EXTERNAL_TOOLS in text
    # This agent binds no tools, so tool-range wording must not reappear.
    assert "tool-call date ranges" not in text


@pytest.mark.unit
def test_tool_using_analysts_keep_their_date_guidance():
    # The analysts that really do call tools keep the wording that anchors their
    # tool date ranges (#836) — this fix is scoped to no-tool agents.
    import tradingagents.agents.analysts.market_analyst as market
    import tradingagents.agents.analysts.news_analyst as news

    for module in (market, news):
        assert "tool-call date ranges" in inspect.getsource(module)


@pytest.mark.unit
def test_constraint_text_is_unambiguous():
    assert "do not call external tools" in NO_EXTERNAL_TOOLS.lower()
    # No template braces: it is embedded in ChatPromptTemplate strings, where
    # braces would be parsed as input variables.
    assert "{" not in NO_EXTERNAL_TOOLS and "}" not in NO_EXTERNAL_TOOLS


def _bind_tools_capturing_llm(captured: dict):
    """LLM that records the tools passed to bind_tools and returns an empty AI response."""
    bound = MagicMock()
    bound.invoke.return_value = MagicMock(content="", tool_calls=[])
    llm = MagicMock()
    llm.bind_tools.side_effect = lambda tools: captured.__setitem__("tools", tools) or bound
    return llm


@pytest.mark.unit
def test_news_analyst_offers_fred_for_default_market():
    import tradingagents.agents.analysts.news_analyst as news

    captured = {}
    llm = _bind_tools_capturing_llm(captured)
    news.create_news_analyst(llm)(
        {
            "company_of_interest": "AAPL",
            "trade_date": "2026-07-24",
            "asset_type": "stock",
            "instrument_context": "AAPL is Apple Inc.",
            "messages": [],
        }
    )
    tool_names = {t.name for t in captured["tools"]}
    assert "get_macro_indicators" in tool_names
    assert "get_news" in tool_names


@pytest.mark.unit
def test_news_analyst_offers_fred_for_hk_ticker():
    # Non-China A-share markets fall back to the upstream default behavior,
    # so FRED macro indicators remain available.
    import tradingagents.agents.analysts.news_analyst as news

    captured = {}
    llm = _bind_tools_capturing_llm(captured)
    news.create_news_analyst(llm)(
        {
            "company_of_interest": "0700.HK",
            "trade_date": "2026-07-24",
            "asset_type": "stock",
            "instrument_context": "0700.HK is Tencent.",
            "messages": [],
        }
    )
    tool_names = {t.name for t in captured["tools"]}
    assert "get_macro_indicators" in tool_names
    assert "get_news" in tool_names


@pytest.mark.unit
def test_news_analyst_skips_fred_for_shenzhen_ticker():
    import tradingagents.agents.analysts.news_analyst as news

    captured = {}
    llm = _bind_tools_capturing_llm(captured)
    news.create_news_analyst(llm)(
        {
            "company_of_interest": "300750.SZ",
            "trade_date": "2026-07-24",
            "asset_type": "stock",
            "instrument_context": "300750.SZ is 宁德时代.",
            "messages": [],
        }
    )
    tool_names = {t.name for t in captured["tools"]}
    assert "get_macro_indicators" not in tool_names
    assert "get_news" in tool_names


@pytest.mark.unit
def test_news_analyst_skips_fred_for_shanghai_ticker():
    import tradingagents.agents.analysts.news_analyst as news

    captured = {}
    llm = _bind_tools_capturing_llm(captured)
    news.create_news_analyst(llm)(
        {
            "company_of_interest": "601857.SH",
            "trade_date": "2026-07-24",
            "asset_type": "stock",
            "instrument_context": "601857.SH is 中国石油.",
            "messages": [],
        }
    )
    tool_names = {t.name for t in captured["tools"]}
    assert "get_macro_indicators" not in tool_names
    assert "get_news" in tool_names


@pytest.mark.unit
def test_sentiment_prompt_uses_akshare_source_for_shenzhen_ticker(monkeypatch):
    from tradingagents.agents.schemas import SentimentBand, SentimentReport

    monkeypatch.setattr(sentiment, "fetch_stocktwits_messages", lambda *a, **k: "st")
    monkeypatch.setattr(sentiment, "fetch_reddit_posts", lambda *a, **k: "rd")
    monkeypatch.setattr(
        sentiment,
        "fetch_china_social_sentiment",
        lambda *a, **k: "mock China social signals",
    )
    monkeypatch.setattr(sentiment.get_news, "func", lambda *a, **k: "news", raising=False)

    captured = {}
    llm = _capturing_llm(
        captured,
        SentimentReport(
            overall_band=SentimentBand.BULLISH,
            overall_score=7.5,
            confidence="high",
            narrative="n",
        ),
    )
    sentiment.create_sentiment_analyst(llm)(
        {
            "company_of_interest": "300750.SZ",
            "trade_date": "2026-07-24",
            "asset_type": "stock",
            "messages": [],
        }
    )
    text = _prompt_text(captured["prompt"])
    assert "akshare (East Money)" in text
    assert "Yahoo Finance" not in text
    assert "StockTwits and Reddit are US-centric" in text
    assert "China A-share" in text
    assert "mock China social signals" in text
    assert "<start_of_stocktwits>" not in text
    assert "Separate attention from direction" in text


@pytest.mark.unit
def test_sentiment_prompt_uses_akshare_source_for_shanghai_ticker(monkeypatch):
    from tradingagents.agents.schemas import SentimentBand, SentimentReport

    monkeypatch.setattr(sentiment, "fetch_stocktwits_messages", lambda *a, **k: "st")
    monkeypatch.setattr(sentiment, "fetch_reddit_posts", lambda *a, **k: "rd")
    monkeypatch.setattr(
        sentiment,
        "fetch_china_social_sentiment",
        lambda *a, **k: "mock China social signals",
    )
    monkeypatch.setattr(sentiment.get_news, "func", lambda *a, **k: "news", raising=False)

    captured = {}
    llm = _capturing_llm(
        captured,
        SentimentReport(
            overall_band=SentimentBand.BULLISH,
            overall_score=7.5,
            confidence="high",
            narrative="n",
        ),
    )
    sentiment.create_sentiment_analyst(llm)(
        {
            "company_of_interest": "601857.SH",
            "trade_date": "2026-07-24",
            "asset_type": "stock",
            "messages": [],
        }
    )
    text = _prompt_text(captured["prompt"])
    assert "akshare (East Money)" in text
    assert "Yahoo Finance" not in text


@pytest.mark.unit
def test_sentiment_prompt_keeps_yahoo_for_default_market(monkeypatch):
    from tradingagents.agents.schemas import SentimentBand, SentimentReport

    monkeypatch.setattr(sentiment, "fetch_stocktwits_messages", lambda *a, **k: "st")
    monkeypatch.setattr(sentiment, "fetch_reddit_posts", lambda *a, **k: "rd")
    monkeypatch.setattr(sentiment.get_news, "func", lambda *a, **k: "news", raising=False)

    captured = {}
    llm = _capturing_llm(
        captured,
        SentimentReport(
            overall_band=SentimentBand.BULLISH,
            overall_score=7.5,
            confidence="high",
            narrative="n",
        ),
    )
    sentiment.create_sentiment_analyst(llm)(
        {
            "company_of_interest": "NVDA",
            "trade_date": "2026-07-24",
            "asset_type": "stock",
            "messages": [],
        }
    )
    text = _prompt_text(captured["prompt"])
    assert "Yahoo Finance" in text
    assert "akshare (East Money)" not in text
    assert "three complementary data sources" in text
