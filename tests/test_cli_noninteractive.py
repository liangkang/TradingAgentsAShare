from unittest import mock

import pandas as pd
import pytest
from typer.testing import CliRunner

from cli.main import app


def _non_interactive_args(**overrides):
    args = {
        "ticker": "0700.HK",
        "date": "2026-07-22",
        "analysts": "market,fundamentals,news,social",
        "provider": "deepseek",
        "deep_model": "deepseek-v4-flash",
        "quick_model": "deepseek-v4-flash",
    }
    args.update(overrides)
    cli_args = ["analyze", "--no-interactive"]
    for key, value in args.items():
        flag = key.replace("_", "-")
        cli_args.extend([f"--{flag}", str(value)])
    return cli_args


@pytest.mark.unit
def test_analyze_accepts_non_interactive_options():
    runner = CliRunner()
    with mock.patch("cli.main.run_analysis") as run:
        result = runner.invoke(
            app,
            _non_interactive_args(
                research_depth="4",
                output_language="Chinese",
            ),
        )

    assert result.exit_code == 0, result.output
    selections = run.call_args.kwargs["selections"]
    assert selections["ticker"] == "0700.HK"
    assert selections["llm_provider"] == "deepseek"
    assert selections["deep_thinker"] == "deepseek-v4-flash"
    assert selections["shallow_thinker"] == "deepseek-v4-flash"
    assert selections["research_depth"] == 4
    assert selections["research_depth_explicit"] is True
    assert selections["output_language"] == "Chinese"
    assert run.call_args.kwargs["interactive"] is False


@pytest.mark.unit
def test_analyze_passes_save_report_flags():
    runner = CliRunner()
    with mock.patch("cli.main.run_analysis") as run:
        result = runner.invoke(
            app,
            [
                *_non_interactive_args(),
                "--save-report",
                "--save-path",
                "/tmp/ta-report",
            ],
        )

    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs["save_report"] is True
    assert run.call_args.kwargs["save_path"] == "/tmp/ta-report"


@pytest.mark.unit
def test_analyze_passes_vnpy_signal_flags():
    runner = CliRunner()
    with mock.patch("cli.main.run_analysis") as run:
        result = runner.invoke(
            app,
            [
                *_non_interactive_args(),
                "--vnpy-signal-dir",
                "/tmp/ta-signals",
                "--vnpy-exchange",
                "SSE",
                "--vnpy-signal-ttl-hours",
                "12",
            ],
        )

    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs["vnpy_signal_dir"] == "/tmp/ta-signals"
    assert run.call_args.kwargs["vnpy_exchange"] == "SSE"
    assert run.call_args.kwargs["vnpy_signal_ttl_hours"] == 12


@pytest.mark.unit
def test_save_complete_report_writes_complete_report(tmp_path):
    from cli.main import _save_complete_report

    final_state = {
        "market_report": "Market outlook is stable.",
        "risk_debate_state": {
            "judge_decision": "**Rating**: Hold\n\n**Executive Summary**: Wait.",
        },
    }
    save_dir = tmp_path / "0700.HK_test"

    _save_complete_report(final_state, "0700.HK", save_dir)

    complete_report = save_dir / "complete_report.md"
    assert complete_report.exists()
    text = complete_report.read_text(encoding="utf-8")
    assert "Trading Analysis Report: 0700.HK" in text
    assert "**Rating**: Hold" in text


@pytest.mark.unit
def test_analyze_resolves_chinese_a_share_name():
    stock_list = pd.DataFrame({"code": ["601869"], "name": ["长飞光纤"]})
    with (
        mock.patch("akshare.stock_info_a_code_name", return_value=stock_list),
        mock.patch("cli.main.run_analysis") as run,
    ):
        result = CliRunner().invoke(
            app,
            [
                "analyze",
                "--ticker", "长飞光纤",
                "--date", "2026-07-22",
                "--analysts", "market,news",
                "--provider", "deepseek",
                "--deep-model", "deepseek-v4-flash",
                "--quick-model", "deepseek-v4-flash",
                "--no-interactive",
            ],
        )

    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs["selections"]["ticker"] == "601869.SH"


@pytest.mark.unit
def test_non_interactive_requires_all_selection_options():
    result = CliRunner().invoke(app, ["analyze", "--ticker", "0700.HK", "--no-interactive"])

    assert result.exit_code == 2
    assert "non-interactive mode requires" in result.output


@pytest.mark.unit
@pytest.mark.parametrize("depth", ["0", "6"])
def test_non_interactive_rejects_research_depth_outside_web_range(depth):
    result = CliRunner().invoke(
        app,
        ["analyze", "--research-depth", depth, "--no-interactive"],
    )

    assert result.exit_code == 2
    assert "--research-depth" in result.output
