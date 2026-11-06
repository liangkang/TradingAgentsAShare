import sys
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.panel import Panel

from cli.display import console
from cli.models import AnalystType, AssetType
from cli.prompts import filter_analysts_for_asset_type, parse_analysts
from cli.run import run_analysis
from cli.selections import depth_from_env
from tradingagents.backtest import iter_grid, run_backtest, summarize
from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.openai_client import SIGNED_IN_PROVIDERS
from tradingagents.portfolio import load_portfolio

# prompt_toolkit's win32 output module is importable only on Windows (it asserts
# the platform at import time), so gate on the platform rather than catching the
# failure — that way a genuinely broken prompt_toolkit on Windows still surfaces
# instead of silently disabling the handler below. Off Windows this stays an
# empty tuple, which `except` accepts and never matches (#1138).
if sys.platform == "win32":  # pragma: no cover - platform dependent
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError

    _NO_CONSOLE_ERRORS: tuple[type[BaseException], ...] = (NoConsoleScreenBufferError,)
else:
    _NO_CONSOLE_ERRORS = ()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


@app.callback(invoke_without_command=True)
def analyze(
    ctx: typer.Context,
    checkpoint: bool | None = typer.Option(
        None,
        "--checkpoint/--no-checkpoint",
        help="Enable/disable checkpoint-resume (save state after each node so a "
        "crashed run can resume). Omit to honor TRADINGAGENTS_CHECKPOINT_ENABLED.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
    portfolio: str = typer.Option(
        None,
        "--portfolio",
        help="JSON file with current holdings and cash, so the trader, risk and "
        "portfolio agents size against your actual position.",
    ),
    ticker: str = typer.Option(None, "--ticker", help="Ticker to analyze, e.g. NVDA or 0700.HK; skips the prompt"),
    date: str = typer.Option(None, "--date", help="Analysis date, YYYY-MM-DD; skips the prompt"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts, e.g. market,news; skips the prompt"
    ),
    save: bool | None = typer.Option(
        None, "--save/--no-save", help="Save the report under results_dir without asking"
    ),
    show: bool | None = typer.Option(
        None, "--show/--no-show", help="Show the full report at the end without asking"
    ),
    html: bool | None = typer.Option(
        None, "--html/--no-html",
        help="Save the report as one HTML page too, complete_report.html, without asking (default: yes)",
    ),
):
    """Run an analysis. This is what a bare `tradingagents` does.

    Flags answer their questions; with provider, models, depth and language also
    set through TRADINGAGENTS_* variables, the run asks nothing.
    """
    if ctx.invoked_subcommand is not None:
        return
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    portfolio_context = None
    if portfolio:
        try:
            portfolio_context = load_portfolio(portfolio)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None

    try:
        flags = {"ticker": ticker, "date": date, "analysts": analysts, "save": save, "show": show, "html": html}
        run_analysis(checkpoint=checkpoint, portfolio=portfolio_context, flags=flags)
    except _NO_CONSOLE_ERRORS:
        # A terminal with no console buffer cannot host the interactive prompts.
        # Emit one actionable line on stderr instead of a prompt_toolkit
        # traceback; plain text, since rich may not render here either (#1138).
        typer.echo(
            "Error: no Windows console available. The interactive CLI needs a real "
            "console buffer — run it from Windows Terminal, PowerShell, or cmd.exe "
            "rather than a piped or embedded terminal.",
            err=True,
        )
        raise typer.Exit(code=1) from None


@app.command()
def backtest(
    tickers: str = typer.Argument(..., help="Comma-separated tickers, e.g. NVDA,AAPL"),
    start: str = typer.Option(..., "--start", help="First analysis date, YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="Last analysis date, YYYY-MM-DD"),
    every: int = typer.Option(7, "--every", help="Days between analysis dates"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts to run: market, sentiment, news, fundamentals; omit for all the asset type allows"
    ),
    asset_type: str = typer.Option("stock", "--asset-type", help="stock or crypto"),
    portfolio: str = typer.Option(
        None, "--portfolio", help="JSON file with holdings and cash, held constant across the grid"
    ),
    run_id: str = typer.Option(
        None, "--run-id", help="Continue an earlier sweep: its cells are skipped and its log reused"
    ),
):
    """Score past decisions over a grid of tickers and dates."""

    try:
        dates = iter_grid(start, end, every)
        book = load_portfolio(portfolio) if portfolio else None
        kind = AssetType(asset_type.strip().lower())
        # The analysts are named and checked as for an analysis; without a
        # choice, every analyst the asset type allows runs.
        chosen = (parse_analysts(analysts, kind) if analysts
                  else filter_analysts_for_asset_type(list(AnalystType), kind))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    names = [t.strip() for t in tickers.split(",") if t.strip()]
    if not names:
        console.print("[red]No ticker to analyze; pass them comma-separated, e.g. NVDA,AAPL[/red]")
        raise typer.Exit(code=1)

    def show_progress(done, total, ticker, date):
        console.print(f"[dim][{done}/{total}] {ticker} {date}[/dim]")

    kwargs = {"asset_type": kind.value, "portfolio": book, "run_id": run_id, "progress": show_progress,
              "selected_analysts": [a.value for a in chosen]}

    try:
        result = run_backtest(names, dates, DEFAULT_CONFIG, **kwargs)
    except Exception as exc:  # a missing key or an unknown analyst is a setup error
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(summarize(result).render())
    console.print(f"\nRan {result.cells_run} cells, skipped {result.skipped}. Log: {result.log_path}")
    console.print(f"Continue or settle this sweep: --run-id {result.run_id}")
    for ticker, date, reason in result.failures:
        console.print(f"[yellow]failed:[/yellow] {ticker} {date}: {reason}")
    for ticker, reason in result.settlement_failures:
        console.print(f"[yellow]unsettled:[/yellow] {ticker}: {reason}")


# The plans `login` and `logout` take: the providers signed in to, not keyed.
Plan = StrEnum("Plan", {name: name for name in SIGNED_IN_PROVIDERS})


@app.command()
def login(plan: Annotated[Plan, typer.Argument(help="The plan to sign in with")]):
    """Sign in with your ChatGPT Plus or Pro plan, for the chatgpt provider."""
    from tradingagents.llm_clients import chatgpt_auth

    try:
        # One unwrapped line, so the sign-in address can be copied whole.
        email = chatgpt_auth.sign_in(
            say=lambda line: console.print(line, soft_wrap=True, markup=False, highlight=False))
    except chatgpt_auth.ChatGPTAuthError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"[green]Signed in to ChatGPT as {escape(email)}.[/green]")


@app.command()
def logout(plan: Annotated[Plan, typer.Argument(help="The plan to sign out of")]):
    """Sign out of your ChatGPT plan: revoke this machine's access and forget it."""
    from tradingagents.llm_clients import chatgpt_auth

    outcome = chatgpt_auth.sign_out()
    if outcome == "revoked":
        console.print("Signed out of ChatGPT.")
    elif outcome == "forgotten":
        console.print("[yellow]Signed out here, but OpenAI could not be reached to revoke the sign-in; "
                      "it stays valid at OpenAI until it expires.[/yellow]")
    else:
        console.print("Not signed in to ChatGPT.")


# ---------------------------------------------------------------------------
# A-share fork: the `analyze` and `web` commands
# ---------------------------------------------------------------------------
# `analyze` keeps the interface the tradingagents-analyze skill, the README and
# AGENTS.md/CLAUDE.md document; it runs the same analysis as a bare
# `tradingagents`, with every setup question answerable by a flag.

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _cwd_report_dir(ticker: str) -> Path:
    """Where `analyze --save-report` saves without --save-path: ./reports/{TICKER}_{timestamp}."""
    return Path.cwd() / "reports" / f"{safe_ticker_component(ticker)}_{datetime.now():%Y%m%d_%H%M%S}"


@app.command("analyze")
def analyze_command(
    ticker: str | None = typer.Option(
        None,
        "--ticker",
        help="Ticker symbol or exact Chinese A-share name, e.g. 0700.HK or 长飞光纤.",
    ),
    analysis_date: str | None = typer.Option(None, "--date", help="Analysis date (YYYY-MM-DD)."),
    analysts: str | None = typer.Option(
        None, "--analysts", help="Comma-separated: market,fundamentals,news,social (or sentiment)."
    ),
    provider: str | None = typer.Option(None, "--provider", help="LLM provider."),
    deep_model: str | None = typer.Option(None, "--deep-model", help="Deep-thinking model ID."),
    quick_model: str | None = typer.Option(None, "--quick-model", help="Quick-thinking model ID."),
    research_depth: int | None = typer.Option(
        None,
        "--research-depth",
        min=1,
        max=5,
        help="Research/debate depth from 1 to 5.",
    ),
    output_language: str | None = typer.Option(
        None, "--output-language", help="Report language (default with --no-interactive: English)."
    ),
    no_interactive: bool = typer.Option(
        False, "--no-interactive", help="Run without selection or post-run prompts."
    ),
    checkpoint: bool | None = typer.Option(
        None,
        "--checkpoint/--no-checkpoint",
        help="Enable/disable checkpoint-resume (save state after each node so a "
        "crashed run can resume). Omit to honor TRADINGAGENTS_CHECKPOINT_ENABLED.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
    portfolio: str | None = typer.Option(
        None,
        "--portfolio",
        help="JSON file with current holdings and cash, so the trader, risk and "
        "portfolio agents size against your actual position.",
    ),
    save_report: bool = typer.Option(
        False,
        "--save-report",
        help="Save complete_report.md (and the report tree) without asking.",
    ),
    save_path: str | None = typer.Option(
        None,
        "--save-path",
        help="Directory for --save-report output (default: ./reports/{TICKER}_{timestamp}).",
    ),
    vnpy_signal_dir: str | None = typer.Option(
        None,
        "--vnpy-signal-dir",
        help="Write a validated vn.py signal JSON file to this directory after analysis.",
    ),
    vnpy_exchange: str | None = typer.Option(
        None,
        "--vnpy-exchange",
        help="vn.py exchange override, required when it cannot be inferred from the ticker.",
    ),
    vnpy_signal_ttl_hours: int = typer.Option(
        24,
        "--vnpy-signal-ttl-hours",
        min=1,
        help="Hours before an exported vn.py signal expires.",
    ),
):
    """Run an analysis with every setup question answerable by a flag.

    With --no-interactive nothing is asked: --ticker, --date, --analysts,
    --provider, --deep-model and --quick-model are required, the research depth
    and language fall back to their configured defaults, and the report is saved
    only with --save-report and never displayed.
    """
    if no_interactive:
        required = (
            ("--ticker", ticker), ("--date", analysis_date), ("--analysts", analysts),
            ("--provider", provider), ("--deep-model", deep_model), ("--quick-model", quick_model),
        )
        missing = [name for name, value in required if not value]
        if missing:
            raise typer.BadParameter(f"non-interactive mode requires: {', '.join(missing)}")
        if research_depth is None and not depth_from_env():
            research_depth = DEFAULT_CONFIG["max_debate_rounds"]
        output_language = output_language or DEFAULT_CONFIG["output_language"]

    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    portfolio_context = None
    if portfolio:
        try:
            portfolio_context = load_portfolio(portfolio)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None

    flags = {
        "ticker": ticker,
        "date": analysis_date,
        "analysts": analysts,
        "provider": provider,
        "deep_model": deep_model,
        "quick_model": quick_model,
        "research_depth": research_depth,
        "output_language": output_language,
        "no_interactive": no_interactive,
        # --save-report saves without asking; an unattended run without it does not save.
        "save": True if save_report else (False if no_interactive else None),
        "show": False if no_interactive else None,
        "save_path": save_path or (_cwd_report_dir if save_report else None),
    }
    vnpy = (
        {"signal_dir": vnpy_signal_dir, "exchange": vnpy_exchange, "ttl_hours": vnpy_signal_ttl_hours}
        if vnpy_signal_dir is not None else None
    )
    try:
        run_analysis(checkpoint=checkpoint, portfolio=portfolio_context, flags=flags, vnpy=vnpy)
    except _NO_CONSOLE_ERRORS:
        typer.echo(
            "Error: no Windows console available. The interactive CLI needs a real "
            "console buffer — run it from Windows Terminal, PowerShell, or cmd.exe "
            "rather than a piped or embedded terminal.",
            err=True,
        )
        raise typer.Exit(code=1) from None


@app.command()
def web(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host to bind the web server to.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        help="Port to bind the web server to.",
    ),
    allow_remote: bool = typer.Option(
        False,
        "--allow-remote",
        help="Allow binding to a non-loopback host. The Web UI has no authentication.",
    ),
):
    """Launch the TradingAgents web interface.

    Opens a browser-based UI that mirrors the CLI workflow:
    configuration form → real-time dashboard → final report.

    API keys are read from your .env file (same as the CLI mode).
    Only accessible from localhost by default — single-user local use.
    """
    if host not in _LOCAL_HOSTS and not allow_remote:
        console.print(
            "[red]Refusing to expose the unauthenticated Web UI on a remote "
            "interface.[/red]\n"
            "[dim]Use --allow-remote only on a trusted network.[/dim]"
        )
        raise typer.Exit(code=2)

    try:
        import uvicorn

        from web.app import app as web_app
    except ImportError as e:
        console.print(
            f"[red]Missing dependency: {e}[/red]\n"
            f"[dim]Install with: pip install fastapi uvicorn[/dim]"
        )
        raise typer.Exit(code=1) from e

    console.print()
    console.print(
        Panel(
            "[bold green]TradingAgents Web 界面[/bold green]\n\n"
            f"[bold]服务器启动地址：[/bold] [cyan]http://{host}:{port}[/cyan]\n\n"
            "[dim]在浏览器中打开上述地址即可使用。[/dim]\n"
            "[dim]按 Ctrl+C 停止服务器。[/dim]",
            border_style="green",
            padding=(1, 2),
            title="Web 模式",
        )
    )
    console.print()
    if allow_remote and host not in _LOCAL_HOSTS:
        console.print(
            "[bold yellow]Warning:[/bold yellow] remote access is enabled and "
            "the Web UI has no authentication."
        )
    uvicorn.run(web_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
