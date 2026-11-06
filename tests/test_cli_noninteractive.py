"""`tradingagents analyze`: the fork's flag interface over the upstream CLI run."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

import cli.run as cli_run
from cli.main import _cwd_report_dir, app
from cli.prompts import parse_ticker
from cli.selections import unattended_gaps


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


def _invoke(args):
    with mock.patch("cli.main.run_analysis") as run:
        result = CliRunner().invoke(app, args)
    return result, run


@pytest.mark.unit
def test_analyze_accepts_non_interactive_options():
    result, run = _invoke(_non_interactive_args(research_depth="4", output_language="Chinese"))

    assert result.exit_code == 0, result.output
    flags = run.call_args.kwargs["flags"]
    assert flags["ticker"] == "0700.HK"
    assert flags["date"] == "2026-07-22"
    assert flags["analysts"] == "market,fundamentals,news,social"
    assert flags["provider"] == "deepseek"
    assert flags["deep_model"] == "deepseek-v4-flash"
    assert flags["quick_model"] == "deepseek-v4-flash"
    assert flags["research_depth"] == 4
    assert flags["output_language"] == "Chinese"
    assert flags["no_interactive"] is True
    # Unattended: the report is neither saved (no --save-report) nor displayed.
    assert flags["save"] is False
    assert flags["show"] is False
    assert run.call_args.kwargs["vnpy"] is None


@pytest.mark.unit
def test_non_interactive_defaults_depth_and_language(monkeypatch):
    for var in ("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "TRADINGAGENTS_MAX_RISK_ROUNDS"):
        monkeypatch.delenv(var, raising=False)
    result, run = _invoke(_non_interactive_args())

    assert result.exit_code == 0, result.output
    flags = run.call_args.kwargs["flags"]
    assert flags["research_depth"] == cli_run.DEFAULT_CONFIG["max_debate_rounds"]
    assert flags["output_language"] == cli_run.DEFAULT_CONFIG["output_language"]


@pytest.mark.unit
def test_analyze_passes_save_report_flags():
    result, run = _invoke([*_non_interactive_args(), "--save-report", "--save-path", "/tmp/ta-report"])

    assert result.exit_code == 0, result.output
    flags = run.call_args.kwargs["flags"]
    assert flags["save"] is True
    assert flags["save_path"] == "/tmp/ta-report"


@pytest.mark.unit
def test_save_report_without_a_path_saves_under_the_working_directory():
    result, run = _invoke([*_non_interactive_args(), "--save-report"])

    assert result.exit_code == 0, result.output
    rule = run.call_args.kwargs["flags"]["save_path"]
    assert rule is _cwd_report_dir
    path = rule("0700.HK")
    assert path.parent == Path.cwd() / "reports"
    assert path.name.startswith("0700.HK_")


@pytest.mark.unit
def test_analyze_passes_vnpy_signal_flags():
    result, run = _invoke([
        *_non_interactive_args(),
        "--vnpy-signal-dir", "/tmp/ta-signals",
        "--vnpy-exchange", "SSE",
        "--vnpy-signal-ttl-hours", "12",
    ])

    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs["vnpy"] == {
        "signal_dir": "/tmp/ta-signals", "exchange": "SSE", "ttl_hours": 12,
    }


@pytest.mark.unit
def test_offer_reports_saves_where_the_rule_says_without_asking(tmp_path):
    graph = SimpleNamespace(save_reports=mock.MagicMock(return_value=tmp_path / "complete_report.md"))
    final_state = {"final_trade_decision": "**Rating**: Hold"}

    cli_run._offer_reports(
        final_state, graph, "0700.HK", save=True, show=False, html=False,
        save_path=lambda ticker: tmp_path / f"{ticker}_test",
    )

    graph.save_reports.assert_called_once_with(
        final_state, "0700.HK", tmp_path / "0700.HK_test", html=False
    )


@pytest.mark.unit
def test_a_chinese_a_share_name_given_as_a_flag_resolves_to_its_code():
    stock_list = pd.DataFrame({"code": ["601869"], "name": ["长飞光纤"]})
    with mock.patch("akshare.stock_info_a_code_name", return_value=stock_list):
        assert parse_ticker("长飞光纤") == "601869.SS"


@pytest.mark.unit
def test_the_flags_answer_every_unattended_question(monkeypatch):
    for var in ("TRADINGAGENTS_OUTPUT_LANGUAGE", "TRADINGAGENTS_MAX_DEBATE_ROUNDS",
                "TRADINGAGENTS_MAX_RISK_ROUNDS", "TRADINGAGENTS_LLM_PROVIDER",
                "TRADINGAGENTS_QUICK_THINK_LLM", "TRADINGAGENTS_DEEP_THINK_LLM"):
        monkeypatch.delenv(var, raising=False)
    flags = {
        "ticker": "0700.HK", "date": "2026-07-22", "analysts": "market",
        "provider": "deepseek", "deep_model": "m", "quick_model": "m",
        "research_depth": 1, "output_language": "English", "save": False, "show": False,
    }

    assert unattended_gaps(flags) == []


@pytest.mark.unit
def test_no_interactive_never_prompts_even_at_a_terminal():
    """conftest puts every test at a terminal; --no-interactive must still refuse to ask."""
    with pytest.raises(typer.Exit):
        cli_run.run_analysis(flags={"no_interactive": True, "ticker": "0700.HK"})


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
