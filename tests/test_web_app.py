"""FastAPI request validation and shared-runtime integration."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import web.app as web_app


def _payload(**overrides):
    values = {
        "ticker": "AAPL",
        "analysis_date": "2026-07-20",
        "asset_type": "stock",
        "analysts": ["market"],
        "research_depth": 1,
        "llm_provider": "openai",
        "quick_think_llm": "gpt-5.4-mini",
        "deep_think_llm": "gpt-5.5",
        "output_language": "English",
    }
    values.update(overrides)
    return values


@pytest.mark.unit
def test_defaults_use_shared_provider_catalog():
    response = TestClient(web_app.app).get("/api/config/defaults")
    assert response.status_code == 200
    config = response.json()
    providers = {item["key"] for item in config["providers"]}
    assert {"openai", "mistral", "kimi", "groq", "nvidia", "bedrock"}.issubset(
        providers
    )
    assert {"qwen-cn", "glm-cn", "minimax-cn", "openai_compatible"}.issubset(
        providers
    )
    assert config["defaults"]["research_depth"] == 3


@pytest.mark.unit
def test_web_form_selects_configured_default_research_depth():
    template = web_app.TEMPLATES_DIR.joinpath("index.html").read_text(encoding="utf-8")
    assert "d.value === configData.defaults.research_depth" in template


@pytest.mark.unit
def test_analyze_rejects_future_date_before_starting_runner(monkeypatch):
    monkeypatch.setattr(
        web_app,
        "AnalysisRunner",
        lambda *args, **kwargs: pytest.fail("runner must not be constructed"),
    )
    future = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    response = TestClient(web_app.app).post(
        "/api/analyze",
        json=_payload(analysis_date=future),
    )
    assert response.status_code == 422
    assert "future" in response.json()["detail"]


@pytest.mark.unit
def test_analyze_streams_events_from_shared_runner(monkeypatch, tmp_path):
    class FakeRunner:
        def __init__(self, spec, config, callbacks):
            self.spec = spec
            self.final_state = {}

        def stream(self, cancel_event=None):
            chunks = [
                {"market_report": "Market report", "messages": []},
                {
                    "market_report": "Market report",
                    "investment_debate_state": {
                        "bull_history": "Bull",
                        "bear_history": "Bear",
                        "judge_decision": "Hold",
                    },
                    "trader_investment_plan": "Hold",
                    "risk_debate_state": {
                        "aggressive_history": "Risk on",
                        "conservative_history": "Risk off",
                        "neutral_history": "Balanced",
                        "judge_decision": "Rating: Hold",
                    },
                    "final_trade_decision": "Rating: Hold",
                    "messages": [],
                },
            ]
            for chunk in chunks:
                self.final_state.update(chunk)
                yield chunk

    saved = {}

    def fake_save(**kwargs):
        saved.update(kwargs)
        return tmp_path / "saved.json"

    monkeypatch.setattr(web_app, "AnalysisRunner", FakeRunner)
    monkeypatch.setattr(web_app, "_save_analysis_to_disk", fake_save)
    stock_list = pd.DataFrame({"code": ["601869"], "name": ["长飞光纤"]})
    monkeypatch.setattr(
        "akshare.stock_info_a_code_name",
        lambda: stock_list,
    )

    with TestClient(web_app.app).stream(
        "POST",
        "/api/analyze",
        json=_payload(ticker="长飞光纤"),
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: report_update" in body
    assert "event: complete" in body
    assert saved["ticker"] == "601869.SH"
    assert saved["reports"]["market_report"] == "Market report"
    assert "final_trade_decision" in saved["reports"]


@pytest.mark.unit
@pytest.mark.parametrize("analysis_id", ["../secrets", "not-hex", "a" * 13])
def test_history_rejects_invalid_analysis_ids(analysis_id):
    response = TestClient(web_app.app).get(f"/api/history/{analysis_id}")
    assert response.status_code == 404


@pytest.mark.unit
def test_save_report_only_accepts_markdown():
    response = TestClient(web_app.app).post(
        "/api/save-report",
        json={
            "ticker": "AAPL",
            "report_content": "report",
            "format": "../txt",
        },
    )
    assert response.status_code == 422
