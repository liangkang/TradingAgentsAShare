"""Shared runtime validation and per-run config isolation."""

from __future__ import annotations

import threading
from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.dataflows.config import config_context, get_config
from tradingagents.runtime import AnalysisSpec, build_run_config


def _spec(**overrides) -> AnalysisSpec:
    values = {
        "ticker": "aapl",
        "analysis_date": "2026-07-20",
        "analysts": ["news", "market"],
        "research_depth": 3,
        "llm_provider": "openai",
        "quick_think_llm": "gpt-5.4-mini",
        "deep_think_llm": "gpt-5.5",
    }
    values.update(overrides)
    return AnalysisSpec.create(**values)


@pytest.mark.unit
def test_spec_normalizes_ticker_and_analyst_order():
    spec = _spec()
    assert spec.ticker == "AAPL"
    assert spec.analysts == ("market", "news")
    assert spec.asset_type == "stock"


@pytest.mark.unit
def test_spec_resolves_chinese_a_share_name_before_ticker_validation():
    stock_list = pd.DataFrame({"code": ["601869"], "name": ["长飞光纤"]})
    with patch("akshare.stock_info_a_code_name", return_value=stock_list):
        spec = _spec(ticker="长飞光纤")

    assert spec.ticker == "601869.SH"
    assert spec.asset_type == "stock"


@pytest.mark.unit
def test_spec_infers_crypto_and_rejects_fundamentals():
    with pytest.raises(ValueError, match="fundamentals"):
        _spec(
            ticker="btcusdt",
            asset_type=None,
            analysts=["market", "fundamentals"],
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("analysis_date", "2999-01-01", "future"),
        ("analysts", ["market", "bogus"], "unknown analyst"),
        ("llm_provider", "bogus", "unknown LLM provider"),
        ("ticker", "../../etc/passwd", "invalid ticker"),
    ],
)
def test_spec_rejects_invalid_inputs(field, value, message):
    with pytest.raises(ValueError, match=message):
        _spec(**{field: value})


@pytest.mark.unit
def test_build_run_config_uses_validated_values():
    spec = _spec(output_language="Chinese", research_depth=5)
    config = build_run_config(spec, checkpoint=True)
    assert config["max_debate_rounds"] == 5
    assert config["max_risk_discuss_rounds"] == 5
    assert config["output_language"] == "Chinese"
    assert config["checkpoint_enabled"] is True


@pytest.mark.unit
def test_config_context_isolates_concurrent_threads():
    barrier = threading.Barrier(2)
    observed: dict[str, tuple[str, str]] = {}

    def run(name: str, language: str, vendor: str) -> None:
        with config_context(
            {
                "output_language": language,
                "data_vendors": {"core_stock_apis": vendor},
            }
        ):
            barrier.wait(timeout=2)
            config = get_config()
            observed[name] = (
                config["output_language"],
                config["data_vendors"]["core_stock_apis"],
            )

    first = threading.Thread(target=run, args=("first", "Chinese", "akshare"))
    second = threading.Thread(target=run, args=("second", "English", "yfinance"))
    first.start()
    second.start()
    first.join(timeout=2)
    second.join(timeout=2)

    assert observed == {
        "first": ("Chinese", "akshare"),
        "second": ("English", "yfinance"),
    }
