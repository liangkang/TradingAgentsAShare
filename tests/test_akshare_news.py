from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import tradingagents.dataflows.akshare_vendor as ak_vendor
from tradingagents.dataflows.akshare_vendor import (
    get_global_news_akshare,
    get_news_akshare,
)
from tradingagents.dataflows.symbol_utils import NoMarketDataError


@pytest.mark.unit
def test_company_news_combines_eastmoney_and_cninfo_and_filters_future(monkeypatch):
    import akshare as ak

    monkeypatch.setattr(
        ak,
        "stock_news_em",
        lambda **kwargs: pd.DataFrame(
            {
                "新闻标题": ["范围内新闻", "未来新闻", "无日期新闻"],
                "新闻内容": ["公司签订合同", "不应出现", "无法验证"],
                "发布时间": [
                    "2025-05-08 10:00:00",
                    "2025-05-10 00:00:00",
                    "",
                ],
                "新闻链接": ["https://example.com/news", "", ""],
            }
        ),
    )
    disclosure_calls = []

    def disclosures(**kwargs):
        disclosure_calls.append(kwargs)
        return pd.DataFrame(
            {
                "公告标题": ["年度股东大会公告", "未来公告"],
                "公告时间": ["2025-05-09", "2025-05-10"],
                "公告链接": [
                    "https://example.com/disclosure",
                    "https://example.com/future",
                ],
            }
        )

    monkeypatch.setattr(ak, "stock_zh_a_disclosure_report_cninfo", disclosures)

    result = get_news_akshare("300750.SZ", "2025-05-01", "2025-05-09")

    assert "East Money company news" in result
    assert "范围内新闻" in result
    assert "CNInfo official disclosures" in result
    assert "年度股东大会公告" in result
    assert "未来新闻" not in result
    assert "未来公告" not in result
    assert "无日期新闻" not in result
    assert disclosure_calls == [
        {
            "symbol": "300750",
            "market": "沪深京",
            "keyword": "",
            "category": "",
            "start_date": "20250501",
            "end_date": "20250509",
        }
    ]


@pytest.mark.unit
def test_company_news_shares_configured_limit_between_sources(monkeypatch):
    import akshare as ak

    monkeypatch.setattr(
        ak_vendor,
        "get_config",
        lambda: {"news_article_limit": 2},
    )
    monkeypatch.setattr(
        ak,
        "stock_news_em",
        lambda **kwargs: pd.DataFrame(
            {
                "新闻标题": ["媒体一", "媒体二", "媒体三"],
                "发布时间": ["2025-05-09 12:00:00"] * 3,
            }
        ),
    )
    monkeypatch.setattr(
        ak,
        "stock_zh_a_disclosure_report_cninfo",
        lambda **kwargs: pd.DataFrame(
            {
                "公告标题": ["公告一", "公告二", "公告三"],
                "公告时间": ["2025-05-09"] * 3,
            }
        ),
    )

    result = get_news_akshare("300750.SZ", "2025-05-01", "2025-05-09")

    assert result.count("### ") == 2
    assert "媒体一" in result
    assert "公告一" in result
    assert "媒体二" not in result
    assert "公告二" not in result


@pytest.mark.unit
def test_company_news_keeps_partial_result_when_one_source_fails(monkeypatch):
    import akshare as ak

    monkeypatch.setattr(
        ak,
        "stock_news_em",
        lambda **kwargs: (_ for _ in ()).throw(ConnectionError("down")),
    )
    monkeypatch.setattr(
        ak,
        "stock_zh_a_disclosure_report_cninfo",
        lambda **kwargs: pd.DataFrame(
            {
                "公告标题": ["回购进展公告"],
                "公告时间": ["2025-05-09"],
                "公告链接": ["https://example.com/buyback"],
            }
        ),
    )

    result = get_news_akshare("600519.SH", "2025-05-01", "2025-05-09")

    assert "回购进展公告" in result
    assert "CNInfo official disclosures" in result


@pytest.mark.unit
def test_company_news_raises_for_router_fallback_when_all_sources_empty(monkeypatch):
    import akshare as ak

    monkeypatch.setattr(ak, "stock_news_em", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        ak,
        "stock_zh_a_disclosure_report_cninfo",
        lambda **kwargs: pd.DataFrame(),
    )

    with pytest.raises(NoMarketDataError):
        get_news_akshare("600519.SH", "2025-05-01", "2025-05-09")


@pytest.mark.unit
def test_global_news_filters_cls_and_skips_current_caixin_for_history(monkeypatch):
    import akshare as ak

    monkeypatch.setattr(
        ak,
        "stock_info_global_cls",
        lambda **kwargs: pd.DataFrame(
            {
                "标题": ["范围内快讯", "未来快讯", "过旧快讯"],
                "内容": ["政策落地", "不应出现", "不应出现"],
                "发布日期": ["2025-05-08", "2025-05-10", "2025-04-30"],
                "发布时间": ["12:00:00", "00:00:00", "23:59:59"],
            }
        ),
    )
    monkeypatch.setattr(
        ak,
        "news_cctv",
        lambda **kwargs: pd.DataFrame({"标题": ["分析日央视新闻"], "内容": ["宏观内容"]}),
    )
    caixin_calls = []

    def caixin():
        caixin_calls.append(True)
        return pd.DataFrame({"summary": ["当前头条"]})

    monkeypatch.setattr(ak, "stock_news_main_cx", caixin)

    result = get_global_news_akshare(
        "2025-05-09",
        look_back_days=7,
        limit=10,
    )

    assert "范围内快讯" in result
    assert "分析日央视新闻" in result
    assert "未来快讯" not in result
    assert "过旧快讯" not in result
    assert "当前头条" not in result
    assert caixin_calls == []


@pytest.mark.unit
def test_global_news_allows_current_caixin_snapshot_only_today(monkeypatch):
    import akshare as ak

    monkeypatch.setattr(ak, "stock_info_global_cls", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(ak, "news_cctv", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        ak,
        "stock_news_main_cx",
        lambda: pd.DataFrame({"summary": ["当天财经头条"]}),
    )

    result = get_global_news_akshare(date.today().isoformat(), limit=5)

    assert "当天财经头条" in result
    assert "Caixin current snapshot" in result
