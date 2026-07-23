from unittest import mock

import pytest
from typer.testing import CliRunner

from cli.main import app


@pytest.mark.unit
def test_analyze_accepts_non_interactive_options():
    runner = CliRunner()
    with mock.patch("cli.main.run_analysis") as run:
        result = runner.invoke(
            app,
            [
                "analyze",
                "--ticker", "0700.HK",
                "--date", "2026-07-22",
                "--analysts", "market,fundamentals,news,social",
                "--provider", "deepseek",
                "--deep-model", "deepseek-v4-flash",
                "--quick-model", "deepseek-v4-flash",
                "--output-language", "Chinese",
                "--no-interactive",
            ],
        )

    assert result.exit_code == 0, result.output
    selections = run.call_args.kwargs["selections"]
    assert selections["ticker"] == "0700.HK"
    assert selections["llm_provider"] == "deepseek"
    assert selections["deep_thinker"] == "deepseek-v4-flash"
    assert selections["shallow_thinker"] == "deepseek-v4-flash"
    assert selections["output_language"] == "Chinese"
    assert run.call_args.kwargs["interactive"] is False


@pytest.mark.unit
def test_non_interactive_requires_all_selection_options():
    result = CliRunner().invoke(app, ["analyze", "--ticker", "0700.HK", "--no-interactive"])

    assert result.exit_code == 2
    assert "non-interactive mode requires" in result.output

