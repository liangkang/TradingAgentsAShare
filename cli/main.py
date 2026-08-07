import datetime
import os
import sys
import time
from collections import deque
from functools import wraps
from pathlib import Path

import typer
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from cli.announcements import display_announcements, fetch_announcements
from cli.models import AnalystType
from cli.stats_handler import StatsCallbackHandler
from cli.utils import (
    ask_anthropic_effort,
    ask_gemini_thinking_config,
    ask_glm_region,
    ask_minimax_region,
    ask_openai_reasoning_effort,
    ask_output_language,
    ask_qwen_region,
    confirm_ollama_endpoint,
    detect_asset_type,
    ensure_api_key,
    get_ticker,
    is_valid_ticker_input,
    normalize_ticker_symbol,
    prompt_openai_compatible_url,
    resolve_backend_url,
    select_analysts,
    select_deep_thinking_agent,
    select_llm_provider,
    select_research_depth,
    select_shallow_thinking_agent,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
    get_initial_analyst_node,
)
from tradingagents.reporting import write_report_tree
from tradingagents.runtime import (
    AnalysisEventProjector,
    AnalysisRunner,
    AnalysisSpec,
    build_run_config,
    build_run_config_values,
)

console = Console()

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


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    # Fixed teams that always run (not user-selectable)
    FIXED_AGENTS = {
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Analyst name mapping
    ANALYST_MAPPING = {
        "market": "Market Analyst",
        "social": "Sentiment Analyst",
        "news": "News Analyst",
        "fundamentals": "Fundamentals Analyst",
    }

    # Report section mapping: section -> (analyst_key for filtering, finalizing_agent)
    # analyst_key: which analyst selection controls this section (None = always included)
    # finalizing_agent: which agent must be "completed" for this report to count as done
    REPORT_SECTIONS = {
        "market_report": ("market", "Market Analyst"),
        "sentiment_report": ("social", "Sentiment Analyst"),
        "news_report": ("news", "News Analyst"),
        "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
        "investment_plan": (None, "Research Manager"),
        "trader_investment_plan": (None, "Trader"),
        "final_trade_decision": (None, "Portfolio Manager"),
    }

    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {}
        self.current_agent = None
        self.report_sections = {}
        self.selected_analysts = []
        self._processed_message_ids = set()

    def init_for_analysis(self, selected_analysts):
        """Initialize agent status and report sections based on selected analysts.

        Args:
            selected_analysts: List of analyst type strings (e.g., ["market", "news"])
        """
        self.selected_analysts = [a.lower() for a in selected_analysts]

        # Build agent_status dynamically
        self.agent_status = {}

        # Add selected analysts
        for analyst_key in self.selected_analysts:
            if analyst_key in self.ANALYST_MAPPING:
                self.agent_status[self.ANALYST_MAPPING[analyst_key]] = "pending"

        # Add fixed teams
        for team_agents in self.FIXED_AGENTS.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

        # Build report_sections dynamically
        self.report_sections = {}
        for section, (analyst_key, _) in self.REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in self.selected_analysts:
                self.report_sections[section] = None

        # Reset other state
        self.current_report = None
        self.final_report = None
        self.current_agent = None
        self.messages.clear()
        self.tool_calls.clear()
        self._processed_message_ids.clear()

    def get_completed_reports_count(self):
        """Count reports that are finalized (their finalizing agent is completed).

        A report is considered complete when:
        1. The report section has content (not None), AND
        2. The agent responsible for finalizing that report has status "completed"

        This prevents interim updates (like debate rounds) from counting as completed.
        """
        count = 0
        for section in self.report_sections:
            if section not in self.REPORT_SECTIONS:
                continue
            _, finalizing_agent = self.REPORT_SECTIONS[section]
            # Report is complete if it has content AND its finalizing agent is done
            has_content = self.report_sections.get(section) is not None
            agent_done = self.agent_status.get(finalizing_agent) == "completed"
            if has_content and agent_done:
                count += 1
        return count

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content

        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports - use .get() to handle missing sections
        analyst_sections = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]
        if any(self.report_sections.get(section) for section in analyst_sections):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections.get("market_report"):
                report_parts.append(
                    f"### Market Analysis\n{self.report_sections['market_report']}"
                )
            if self.report_sections.get("sentiment_report"):
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['sentiment_report']}"
                )
            if self.report_sections.get("news_report"):
                report_parts.append(
                    f"### News Analysis\n{self.report_sections['news_report']}"
                )
            if self.report_sections.get("fundamentals_report"):
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}"
                )

        # Research Team Reports
        if self.report_sections.get("investment_plan"):
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        # Trading Team Reports
        if self.report_sections.get("trader_investment_plan"):
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        # Portfolio Management Decision
        if self.report_sections.get("final_trade_decision"):
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def format_tokens(n):
    """Format token count for display."""
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


def update_display(layout, spinner_text=None, stats_handler=None, start_time=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents by team - filter to only include agents in agent_status
    all_teams = {
        "Analyst Team": [
            "Market Analyst",
            "Sentiment Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Filter teams to only include agents that are in agent_status
    teams = {}
    for team, agents in all_teams.items():
        active_agents = [a for a in agents if a in message_buffer.agent_status]
        if active_agents:
            teams[team] = active_agents

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status.get(first_agent, "pending")
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status.get(agent, "pending")
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        formatted_args = format_tool_args(args)
        all_messages.append((timestamp, "Tool", f"{tool_name}: {formatted_args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        content_str = str(content) if content else ""
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp descending (newest first)
    all_messages.sort(key=lambda x: x[0], reverse=True)

    # Calculate how many messages we can show based on available space
    max_messages = 12

    # Get the first N messages (newest ones)
    recent_messages = all_messages[:max_messages]

    # Add messages to table (already in newest-first order)
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    # Agent progress - derived from agent_status dict
    agents_completed = sum(
        1 for status in message_buffer.agent_status.values() if status == "completed"
    )
    agents_total = len(message_buffer.agent_status)

    # Report progress - based on agent completion (not just content existence)
    reports_completed = message_buffer.get_completed_reports_count()
    reports_total = len(message_buffer.report_sections)

    # Build stats parts
    stats_parts = [f"Agents: {agents_completed}/{agents_total}"]

    # LLM and tool stats from callback handler
    if stats_handler:
        stats = stats_handler.get_stats()
        stats_parts.append(f"LLM: {stats['llm_calls']}")
        stats_parts.append(f"Tools: {stats['tool_calls']}")

        # Token display with graceful fallback
        if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
            tokens_str = f"Tokens: {format_tokens(stats['tokens_in'])}\u2191 {format_tokens(stats['tokens_out'])}\u2193"
        else:
            tokens_str = "Tokens: --"
        stats_parts.append(tokens_str)

    stats_parts.append(f"Reports: {reports_completed}/{reports_total}")

    # Elapsed time
    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = f"\u23f1 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        stats_parts.append(elapsed_str)

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(stats_parts))

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections():
    """Get all user selections before starting the analysis display."""
    # Display ASCII art welcome message
    with open(Path(__file__).parent / "static" / "welcome.txt", encoding="utf-8") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()
    console.print()  # Add vertical space before announcements

    # Fetch and display announcements (silent on failure)
    announcements = fetch_announcements()
    display_announcements(console, announcements)

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    def thinking_value_or_prompt(env_var, config_key, label, box_title, box_body, prompt_fn):
        """Return the env-configured reasoning/thinking value, or prompt for it.

        When ``env_var`` is set the interactive choice is skipped and the value
        the env overlay placed on DEFAULT_CONFIG is used — mirroring the
        env-precedence rule applied to the other selection steps.
        """
        if os.environ.get(env_var):
            value = DEFAULT_CONFIG[config_key]
            console.print(f"[green]✓ {label} from environment:[/green] {value}")
            return value
        console.print(create_question_box(box_title, box_body))
        return prompt_fn()

    # Step 1: Ticker symbol
    console.print(
        create_question_box(
            "Step 1: Ticker Symbol",
            "Enter the ticker, with exchange suffix when needed (e.g. SPY, 0700.HK, BTC-USD)",
            "SPY",
        )
    )
    selected_ticker = get_ticker()
    asset_type = detect_asset_type(selected_ticker)
    # Only announce when it's not the default stock path, to avoid printing
    # "stock" on every run.
    if asset_type.value != "stock":
        console.print(
            f"[green]Detected asset type:[/green] {asset_type.value}"
        )

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # Step 3: Output language (skipped when set via TRADINGAGENTS_OUTPUT_LANGUAGE)
    if os.environ.get("TRADINGAGENTS_OUTPUT_LANGUAGE"):
        output_language = DEFAULT_CONFIG["output_language"]
        console.print(
            f"[green]✓ Output language from environment:[/green] {output_language}"
        )
    else:
        console.print(
            create_question_box(
                "Step 3: Output Language",
                "Select the language for analyst reports and final decision"
            )
        )
        output_language = ask_output_language()

    # Step 4: Select analysts
    console.print(
        create_question_box(
            "Step 4: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    selected_analysts = select_analysts(asset_type)
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 5: Research depth (skipped when both round counts are set via env).
    # Research depth maps to the debate + risk round counts; when both are
    # supplied through TRADINGAGENTS_MAX_DEBATE_ROUNDS / _MAX_RISK_ROUNDS we keep
    # the run non-interactive and honor the env values (#977).
    depth_from_env = bool(os.environ.get("TRADINGAGENTS_MAX_DEBATE_ROUNDS")) and bool(
        os.environ.get("TRADINGAGENTS_MAX_RISK_ROUNDS")
    )
    if depth_from_env:
        selected_research_depth = DEFAULT_CONFIG["max_debate_rounds"]
        console.print(
            f"[green]✓ Research depth from environment:[/green] "
            f"{DEFAULT_CONFIG['max_debate_rounds']} debate / "
            f"{DEFAULT_CONFIG['max_risk_discuss_rounds']} risk rounds"
        )
    else:
        console.print(
            create_question_box(
                "Step 5: Research Depth", "Select your research depth level"
            )
        )
        selected_research_depth = select_research_depth()

    # Step 6: LLM Provider (skipped when set via TRADINGAGENTS_LLM_PROVIDER).
    # The backend URL comes from TRADINGAGENTS_LLM_BACKEND_URL when set,
    # otherwise the provider's default endpoint — the same value the menu
    # would have picked.
    provider_from_env = bool(os.environ.get("TRADINGAGENTS_LLM_PROVIDER"))
    if provider_from_env:
        selected_llm_provider = DEFAULT_CONFIG["llm_provider"].lower()
        backend_url = resolve_backend_url(
            selected_llm_provider, env_url=DEFAULT_CONFIG["backend_url"]
        )
        console.print(f"[green]✓ LLM provider from environment:[/green] {selected_llm_provider}")
        console.print(f"[green]✓ Backend URL:[/green] {backend_url}")
        # Still confirm/persist the API key so the run doesn't fail later.
        ensure_api_key(selected_llm_provider)
    else:
        console.print(
            create_question_box(
                "Step 6: LLM Provider", "Select your LLM provider"
            )
        )
        selected_llm_provider, backend_url = select_llm_provider()

        # Providers with regional endpoints prompt for the region as a secondary
        # step so the main dropdown stays clean (mainland China and international
        # accounts cannot share API keys).
        if selected_llm_provider == "qwen":
            selected_llm_provider, backend_url = ask_qwen_region()
        elif selected_llm_provider == "minimax":
            selected_llm_provider, backend_url = ask_minimax_region()
        elif selected_llm_provider == "glm":
            selected_llm_provider, backend_url = ask_glm_region()

        # Honor an explicit env backend URL even when the provider was chosen
        # interactively, so it isn't overwritten by the menu default (#978).
        backend_url = resolve_backend_url(
            selected_llm_provider, backend_url, env_url=DEFAULT_CONFIG["backend_url"]
        )

        # The generic OpenAI-compatible endpoint has no default; ask for it if
        # neither the menu nor the environment supplied one.
        if selected_llm_provider == "openai_compatible" and not backend_url:
            backend_url = prompt_openai_compatible_url()

        # For Ollama, surface the resolved endpoint (OLLAMA_BASE_URL vs default)
        # before model selection so it's obvious where we're connecting.
        if selected_llm_provider == "ollama":
            confirm_ollama_endpoint(backend_url)

        # Confirm the provider's API key is present; prompt the user to paste
        # one and persist it to .env if it's missing, so the analysis run
        # doesn't fail later at the first API call.
        ensure_api_key(selected_llm_provider)

    # Step 7: Thinking agents (skipped when either model is set via environment)
    if os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM") or os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM"):
        selected_shallow_thinker = DEFAULT_CONFIG["quick_think_llm"]
        selected_deep_thinker = DEFAULT_CONFIG["deep_think_llm"]
        console.print(
            f"[green]✓ Thinking agents from environment:[/green] "
            f"quick={selected_shallow_thinker}, deep={selected_deep_thinker}"
        )
    else:
        console.print(
            create_question_box(
                "Step 7: Thinking Agents", "Select your thinking agents for analysis"
            )
        )
        selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
        selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    # Step 8: Provider-specific reasoning/thinking configuration. Each knob is
    # settable via its TRADINGAGENTS_* env var; when that var is set (or the
    # provider itself came from env) the prompt is skipped and the configured
    # value is used — same env-precedence rule as the steps above. None = each
    # provider's own default.
    thinking_level = None
    reasoning_effort = None
    anthropic_effort = None

    provider_lower = selected_llm_provider.lower()
    if provider_from_env:
        thinking_level = DEFAULT_CONFIG["google_thinking_level"]
        reasoning_effort = DEFAULT_CONFIG["openai_reasoning_effort"]
        anthropic_effort = DEFAULT_CONFIG["anthropic_effort"]
    elif provider_lower == "google":
        thinking_level = thinking_value_or_prompt(
            "TRADINGAGENTS_GOOGLE_THINKING_LEVEL", "google_thinking_level",
            "Gemini thinking mode", "Step 8: Thinking Mode",
            "Configure Gemini thinking mode", ask_gemini_thinking_config,
        )
    elif provider_lower == "openai":
        reasoning_effort = thinking_value_or_prompt(
            "TRADINGAGENTS_OPENAI_REASONING_EFFORT", "openai_reasoning_effort",
            "Reasoning effort", "Step 8: Reasoning Effort",
            "Configure OpenAI reasoning effort level", ask_openai_reasoning_effort,
        )
    elif provider_lower in ("anthropic", "evolink"):
        anthropic_effort = thinking_value_or_prompt(
            "TRADINGAGENTS_ANTHROPIC_EFFORT", "anthropic_effort",
            "Claude effort", "Step 8: Effort Level",
            "Configure Claude effort level", ask_anthropic_effort,
        )

    return {
        "ticker": selected_ticker,
        "asset_type": asset_type.value,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort": anthropic_effort,
        "output_language": output_language,
    }


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def save_report_to_disk(final_state, ticker: str, save_path: Path):
    """Save the complete analysis report to disk (shared CLI/API writer)."""
    return write_report_tree(final_state, ticker, save_path)


def _save_complete_report(final_state, ticker: str, save_path: Path | str | None = None):
    """Write the report tree and print the destination path."""
    if save_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path.cwd() / "reports" / f"{ticker}_{timestamp}"
    else:
        save_path = Path(save_path)
    try:
        report_file = save_report_to_disk(final_state, ticker, save_path)
        console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
        console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
    except Exception as e:
        console.print(f"[red]Error saving report: {e}[/red]")


def display_complete_report(final_state):
    """Display the complete analysis report sequentially (avoids truncation)."""
    console.print()
    console.print(Rule("Complete Analysis Report", style="bold green"))

    # I. Analyst Team Reports
    analysts = []
    if final_state.get("market_report"):
        analysts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analysts:
        console.print(Panel("[bold]I. Analyst Team Reports[/bold]", border_style="cyan"))
        for title, content in analysts:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # II. Research Team Reports
    if final_state.get("investment_debate_state"):
        debate = final_state["investment_debate_state"]
        research = []
        if debate.get("bull_history"):
            research.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research.append(("Research Manager", debate["judge_decision"]))
        if research:
            console.print(Panel("[bold]II. Research Team Decision[/bold]", border_style="magenta"))
            for title, content in research:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # III. Trading Team
    if final_state.get("trader_investment_plan"):
        console.print(Panel("[bold]III. Trading Team Plan[/bold]", border_style="yellow"))
        console.print(Panel(Markdown(final_state["trader_investment_plan"]), title="Trader", border_style="blue", padding=(1, 2)))

    # IV. Risk Management Team
    if final_state.get("risk_debate_state"):
        risk = final_state["risk_debate_state"]
        risk_reports = []
        if risk.get("aggressive_history"):
            risk_reports.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_reports.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_reports.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_reports:
            console.print(Panel("[bold]IV. Risk Management Team Decision[/bold]", border_style="red"))
            for title, content in risk_reports:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

        # V. Portfolio Manager Decision
        if risk.get("judge_decision"):
            console.print(Panel("[bold]V. Portfolio Manager Decision[/bold]", border_style="green"))
            console.print(Panel(Markdown(risk["judge_decision"]), title="Portfolio Manager", border_style="blue", padding=(1, 2)))


def format_tool_args(args, max_length=80) -> str:
    """Format tool arguments for terminal display."""
    result = str(args)
    if len(result) > max_length:
        return result[:max_length - 3] + "..."
    return result

def _build_run_config(selections: dict, checkpoint: bool | None) -> dict:
    """Assemble the run config from interactive selections, honoring env precedence.

    Interactive selections retain the historical environment-variable
    precedence. An explicit non-interactive ``--research-depth`` is a CLI flag
    and therefore overrides the environment for that run.
    """
    return build_run_config_values(
        research_depth=selections["research_depth"],
        quick_think_llm=selections["shallow_thinker"],
        deep_think_llm=selections["deep_thinker"],
        backend_url=selections["backend_url"],
        llm_provider=selections["llm_provider"].lower(),
        google_thinking_level=selections.get("google_thinking_level"),
        openai_reasoning_effort=selections.get("openai_reasoning_effort"),
        anthropic_effort=selections.get("anthropic_effort"),
        output_language=selections.get("output_language", "English"),
        checkpoint=checkpoint,
        preserve_env_rounds=not selections.get("research_depth_explicit", False),
        base_config=DEFAULT_CONFIG,
    )


def run_analysis(
    checkpoint: bool | None = None,
    selections: dict | None = None,
    interactive: bool = True,
    save_report: bool = False,
    save_path: Path | str | None = None,
    vnpy_signal_dir: Path | str | None = None,
    vnpy_exchange: str | None = None,
    vnpy_signal_ttl_hours: int = 24,
):
    # First get all user selections
    if selections is None:
        selections = get_user_selections()

    spec = AnalysisSpec.create(
        ticker=selections["ticker"],
        analysis_date=selections["analysis_date"],
        analysts=[analyst.value for analyst in selections["analysts"]],
        research_depth=selections["research_depth"],
        llm_provider=selections["llm_provider"],
        quick_think_llm=selections["shallow_thinker"],
        deep_think_llm=selections["deep_thinker"],
        output_language=selections.get("output_language", "English"),
        asset_type=selections["asset_type"],
        backend_url=selections["backend_url"],
        google_thinking_level=selections.get("google_thinking_level"),
        openai_reasoning_effort=selections.get("openai_reasoning_effort"),
        anthropic_effort=selections.get("anthropic_effort"),
    )
    config = build_run_config(
        spec,
        checkpoint=checkpoint,
        preserve_env_rounds=not selections.get("research_depth_explicit", False),
        base_config=DEFAULT_CONFIG,
    )

    # Create stats callback handler for tracking LLM/tool calls
    stats_handler = StatsCallbackHandler()

    # Normalize analyst selection to predefined order (selection is a 'set', order is fixed)
    selected_analyst_keys = list(spec.analysts)
    analyst_execution_plan = build_analyst_execution_plan(selected_analyst_keys)
    analyst_wall_time_tracker = AnalystWallTimeTracker(analyst_execution_plan)

    runner = AnalysisRunner(
        spec,
        config,
        callbacks=[stats_handler],
    )
    event_projector = AnalysisEventProjector(
        spec.analysts,
        analyst_wall_time_tracker,
    )

    # Initialize message buffer with selected analysts
    message_buffer.init_for_analysis(selected_analyst_keys)

    # Track start time for elapsed display
    start_time = time.time()

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper

    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    text = "\n".join(str(item) for item in content) if isinstance(content, list) else content
                    with open(report_dir / file_name, "w", encoding="utf-8") as f:
                        f.write(text)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # Now start the display layout
    layout = create_layout()

    with Live(layout, refresh_per_second=4):
        # Initial display
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        if selections["asset_type"] != "stock":
            message_buffer.add_message("System", f"Detected asset type: {selections['asset_type']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Update agent status to in_progress for the first analyst
        first_analyst = get_initial_analyst_node(analyst_execution_plan)
        message_buffer.update_agent_status(first_analyst, "in_progress")
        analyst_wall_time_tracker.mark_started(selected_analyst_keys[0])
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Create spinner text
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text, stats_handler=stats_handler, start_time=start_time)

        for chunk in runner.stream():
            for event in event_projector.process_chunk(chunk):
                if event.event == "message":
                    message_buffer.add_message(
                        event.data["type"],
                        event.data["content"],
                    )
                elif event.event == "tool_call":
                    message_buffer.add_tool_call(
                        event.data["name"],
                        event.data["args"],
                    )
                elif event.event == "agent_status":
                    message_buffer.update_agent_status(
                        event.data["agent"],
                        event.data["status"],
                    )
                elif event.event == "report_update":
                    message_buffer.update_report_section(
                        event.data["section"],
                        event.data["content"],
                    )

            # Update the display
            update_display(layout, stats_handler=stats_handler, start_time=start_time)

        final_state = runner.final_state

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )
        message_buffer.add_message("System", analyst_wall_time_tracker.format_summary())

        # Update final report sections
        for section in message_buffer.report_sections:
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    # Post-analysis prompts (outside Live context for clean interaction)
    console.print("\n[bold cyan]Analysis Complete![/bold cyan]\n")
    console.print(f"[dim]{analyst_wall_time_tracker.format_summary()}[/dim]")

    if vnpy_signal_dir is not None:
        from tradingagents.integrations.vnpy import export_vnpy_signal

        signal_path = export_vnpy_signal(
            final_state,
            selections["ticker"],
            vnpy_signal_dir,
            exchange=vnpy_exchange,
            ttl_hours=vnpy_signal_ttl_hours,
        )
        console.print(f"[green]Exported vn.py signal:[/green] {signal_path}")

    if not interactive:
        if save_report:
            _save_complete_report(final_state, selections["ticker"], save_path)
        return final_state

    # Prompt to save report
    save_choice = typer.prompt("Save report?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
        save_path_str = typer.prompt(
            "Save path (press Enter for default)",
            default=str(default_path)
        ).strip()
        _save_complete_report(final_state, selections["ticker"], save_path_str)

    # Prompt to display full report
    display_choice = typer.prompt("\nDisplay full report on screen?", default="Y").strip().upper()
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)


def _non_interactive_selections(
    ticker: str | None,
    analysis_date: str | None,
    analysts: str | None,
    provider: str | None,
    deep_model: str | None,
    quick_model: str | None,
    research_depth: int | None,
    output_language: str,
) -> dict:
    """Validate CLI values and translate them to the existing run selections."""
    missing = [
        name
        for name, value in {
            "--ticker": ticker,
            "--date": analysis_date,
            "--analysts": analysts,
            "--provider": provider,
            "--deep-model": deep_model,
            "--quick-model": quick_model,
        }.items()
        if not value
    ]
    if missing:
        raise typer.BadParameter(
            f"non-interactive mode requires: {', '.join(missing)}"
        )

    assert ticker and analysis_date and analysts and provider and deep_model and quick_model
    if not is_valid_ticker_input(ticker):
        raise typer.BadParameter("invalid ticker", param_hint="--ticker")
    try:
        parsed_date = datetime.datetime.strptime(analysis_date, "%Y-%m-%d").date()
    except ValueError:
        raise typer.BadParameter("must use YYYY-MM-DD", param_hint="--date") from None
    if parsed_date > datetime.datetime.now().date():
        raise typer.BadParameter("cannot be in the future", param_hint="--date")

    try:
        ticker = normalize_ticker_symbol(ticker)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--ticker") from None
    asset_type = detect_asset_type(ticker)
    analyst_names = [value.strip().lower() for value in analysts.split(",") if value.strip()]
    valid_names = {item.value for item in AnalystType}
    invalid = sorted(set(analyst_names) - valid_names)
    if invalid:
        raise typer.BadParameter(
            f"unknown analyst(s): {', '.join(invalid)}; choose from {', '.join(sorted(valid_names))}",
            param_hint="--analysts",
        )
    if not analyst_names:
        raise typer.BadParameter("select at least one analyst", param_hint="--analysts")
    if asset_type.value == "crypto" and "fundamentals" in analyst_names:
        raise typer.BadParameter(
            "fundamentals is not available for crypto assets", param_hint="--analysts"
        )

    provider = provider.lower()
    return {
        "ticker": ticker,
        "asset_type": asset_type.value,
        "analysis_date": analysis_date,
        "analysts": [AnalystType(value) for value in analyst_names],
        "research_depth": (
            research_depth
            if research_depth is not None
            else DEFAULT_CONFIG["max_debate_rounds"]
        ),
        "research_depth_explicit": research_depth is not None,
        "llm_provider": provider,
        "backend_url": resolve_backend_url(provider, env_url=DEFAULT_CONFIG["backend_url"]),
        "shallow_thinker": quick_model,
        "deep_thinker": deep_model,
        "google_thinking_level": DEFAULT_CONFIG["google_thinking_level"],
        "openai_reasoning_effort": DEFAULT_CONFIG["openai_reasoning_effort"],
        "anthropic_effort": DEFAULT_CONFIG["anthropic_effort"],
        "output_language": output_language,
    }


@app.callback(invoke_without_command=True)
def root(ctx: typer.Context):
    """Keep the historical bare `tradingagents` command interactive."""
    if ctx.invoked_subcommand is None:
        try:
            run_analysis()
        except _NO_CONSOLE_ERRORS:
            typer.echo(
                "Error: no Windows console available. The interactive CLI needs a real "
                "console buffer — run it from Windows Terminal, PowerShell, or cmd.exe "
                "rather than a piped or embedded terminal.",
                err=True,
            )
            raise typer.Exit(code=1) from None


@app.command()
def analyze(
    ticker: str | None = typer.Option(
        None,
        "--ticker",
        help="Ticker symbol or exact Chinese A-share name, e.g. 0700.HK or 长飞光纤.",
    ),
    analysis_date: str | None = typer.Option(None, "--date", help="Analysis date (YYYY-MM-DD)."),
    analysts: str | None = typer.Option(
        None, "--analysts", help="Comma-separated: market,fundamentals,news,social."
    ),
    provider: str | None = typer.Option(None, "--provider", help="LLM provider."),
    deep_model: str | None = typer.Option(None, "--deep-model", help="Deep-thinking model ID."),
    quick_model: str | None = typer.Option(None, "--quick-model", help="Quick-thinking model ID."),
    research_depth: int | None = typer.Option(
        None,
        "--research-depth",
        min=1,
        max=5,
        help="Research/debate depth from 1 to 5 (non-interactive mode).",
    ),
    output_language: str = typer.Option("English", "--output-language"),
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
    save_report: bool = typer.Option(
        False,
        "--save-report",
        help="Write complete_report.md after a non-interactive run.",
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
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    try:
        selections = None
        if no_interactive:
            selections = _non_interactive_selections(
                ticker,
                analysis_date,
                analysts,
                provider,
                deep_model,
                quick_model,
                research_depth,
                output_language,
            )
        run_analysis(
            checkpoint=checkpoint,
            selections=selections,
            interactive=not no_interactive,
            save_report=save_report,
            save_path=save_path,
            vnpy_signal_dir=vnpy_signal_dir,
            vnpy_exchange=vnpy_exchange,
            vnpy_signal_ttl_hours=vnpy_signal_ttl_hours,
        )
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
    if host not in {"127.0.0.1", "localhost", "::1"} and not allow_remote:
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
    if allow_remote and host not in {"127.0.0.1", "localhost", "::1"}:
        console.print(
            "[bold yellow]Warning:[/bold yellow] remote access is enabled and "
            "the Web UI has no authentication."
        )
    uvicorn.run(web_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
