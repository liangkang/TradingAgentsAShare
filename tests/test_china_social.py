from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tradingagents.dataflows.china_social import fetch_china_social_sentiment


@pytest.mark.unit
def test_historical_analysis_does_not_query_current_social_snapshots(monkeypatch):
    import akshare as ak

    calls = []

    def unexpected_call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("current snapshot source must not be queried")

    for name in (
        "stock_individual_info_em",
        "stock_hot_rank_em",
        "stock_hot_rank_detail_em",
        "stock_hot_keyword_em",
        "stock_hot_tweet_xq",
        "stock_zh_vote_baidu",
        "stock_js_weibo_report",
    ):
        monkeypatch.setattr(ak, name, unexpected_call)

    result = fetch_china_social_sentiment("300750.SZ", "2000-01-01")

    assert calls == []
    assert "historical analysis" in result
    assert "look-ahead bias" in result


@pytest.mark.unit
def test_current_a_share_social_sources_are_aggregated(monkeypatch):
    import akshare as ak

    monkeypatch.setattr(
        ak,
        "stock_individual_info_em",
        lambda **kwargs: pd.DataFrame({"item": ["股票简称"], "value": ["宁德时代"]}),
    )
    monkeypatch.setattr(
        ak,
        "stock_hot_rank_em",
        lambda: pd.DataFrame(
            {
                "当前排名": [3],
                "代码": ["300750"],
                "股票名称": ["宁德时代"],
                "最新价": [250.0],
                "涨跌幅": [1.5],
            }
        ),
    )
    monkeypatch.setattr(
        ak,
        "stock_hot_rank_detail_em",
        lambda **kwargs: pd.DataFrame({"时间": ["2026-07-25"], "排名": [3], "粉丝增长率": ["8%"]}),
    )
    monkeypatch.setattr(
        ak,
        "stock_hot_keyword_em",
        lambda **kwargs: pd.DataFrame({"概念名称": ["动力电池", "储能"]}),
    )
    monkeypatch.setattr(
        ak,
        "stock_hot_tweet_xq",
        lambda **kwargs: pd.DataFrame(
            {
                "股票代码": ["SZ300750"],
                "股票简称": ["宁德时代"],
                "关注": [12345],
                "最新价": [250.0],
            }
        ),
    )
    monkeypatch.setattr(
        ak,
        "stock_zh_vote_baidu",
        lambda **kwargs: pd.DataFrame(
            {
                "周期": ["今日"],
                "看涨": [700],
                "看跌": [300],
                "看涨比例": ["70%"],
                "看跌比例": ["30%"],
            }
        ),
    )
    monkeypatch.setattr(
        ak,
        "stock_js_weibo_report",
        lambda **kwargs: pd.DataFrame({"name": ["宁德时代"], "rate": [88.8]}),
    )

    result = fetch_china_social_sentiment(
        "300750.SZ",
        date.today().isoformat(),
    )

    assert "East Money popularity" in result
    assert "动力电池" in result
    assert "Xueqiu discussion ranking" in result
    assert "Baidu Finance bullish/bearish vote" in result
    assert "看涨比例: 70%" in result
    assert "Weibo 7-day attention" in result
    assert "popularity/attention measures participation, not bullishness" in result


@pytest.mark.unit
def test_one_social_source_failure_does_not_hide_other_sources(monkeypatch):
    import akshare as ak

    monkeypatch.setattr(
        ak,
        "stock_individual_info_em",
        lambda **kwargs: pd.DataFrame({"item": ["股票简称"], "value": ["贵州茅台"]}),
    )
    monkeypatch.setattr(
        ak,
        "stock_hot_rank_em",
        lambda: (_ for _ in ()).throw(RuntimeError("source down")),
    )
    monkeypatch.setattr(
        ak,
        "stock_hot_tweet_xq",
        lambda **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        ak,
        "stock_zh_vote_baidu",
        lambda **kwargs: pd.DataFrame({"周期": ["今日"], "看涨比例": ["60%"], "看跌比例": ["40%"]}),
    )
    monkeypatch.setattr(
        ak,
        "stock_js_weibo_report",
        lambda **kwargs: pd.DataFrame(),
    )

    result = fetch_china_social_sentiment(
        "600519.SH",
        date.today().isoformat(),
    )

    assert "East Money popularity: unavailable (RuntimeError)" in result
    assert "看涨比例: 60%" in result
