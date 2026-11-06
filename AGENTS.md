# AGENTS.md

This file provides guidance to coding agents (such as Codex) when working with code in this repository.

## Project Overview

TradingAgents is a multi-agent LLM-powered financial trading framework built with LangGraph. It simulates a real-world trading firm with specialized agents (analysts, researchers, traders, risk managers, portfolio managers) that collaboratively evaluate market conditions and make trading decisions.

This repository is a fork of TauricResearch/TradingAgents (currently based on v0.7.1). On top of upstream it adds:
- **akshare** as the first data vendor, so A-shares (`.SS` / `.SZ`) and Hong Kong stocks (`.HK`) work without Yahoo
- A-share-only extended tools (fund flow, dragon-tiger list, institutional holdings) and China social sentiment
- A **Web frontend** (`web/`, FastAPI) sharing the CLI's run pipeline (`tradingagents/runtime/`)
- A **non-interactive CLI** (`tradingagents analyze --no-interactive ...`) used by the agent skills in `.agents/skills/`
- A **vn.py signal export** (`tradingagents/integrations/vnpy/`)

**Key Technologies:**
- LangGraph for agent orchestration and state management
- Multi-provider LLM support (OpenAI, ChatGPT plan, Anthropic, Evolink, Google, xAI, DeepSeek, Qwen, GLM, MiniMax, OpenRouter, Mistral, Kimi, Groq, NVIDIA NIM, Azure, Bedrock, Ollama)
- Structured output via Pydantic schemas (Sentiment Analyst, Research Manager, Trader, Portfolio Manager)
- akshare / yfinance / SEC EDGAR / Alpha Vantage / FRED / Polymarket for data
- FastAPI + uvicorn for the Web frontend

## Common Commands

### Installation & Setup
```bash
# Install package and dependencies
pip install .

# Copy environment template and add API keys
cp .env.example .env

# Docker alternative
docker compose run --rm tradingagents
```

### Running the System
```bash
# Interactive CLI (primary interface)
tradingagents
# or
python -m cli.main

# Non-interactive run (every selection given as a flag; used by the skills)
tradingagents analyze --no-interactive \
  --ticker 600519.SS --date 2026-07-22 \
  --analysts market,fundamentals,news,social \
  --provider deepseek --deep-model deepseek-v4-flash --quick-model deepseek-v4-flash \
  --save-report                      # -> ./reports/<TICKER>_<timestamp>/

# Web frontend
tradingagents web

# Backtest a grid of tickers and dates
tradingagents backtest --help

# Programmatic usage (via Python)
python main.py  # example implementation
```

`tradingagents analyze` without `--no-interactive` prompts only for the selections not given as flags. `--vnpy-signal-dir` (with `--vnpy-exchange`, `--vnpy-signal-ttl-hours`) writes a vn.py signal after the run.

### Testing
```bash
# Run all tests
pytest

# Run specific test markers
pytest -m unit          # fast isolated unit tests
pytest -m integration   # tests requiring external services
pytest -m smoke         # quick sanity checks

# Run a single test file
pytest tests/test_structured_agents.py

# Run a specific test
pytest tests/test_checkpoint_resume.py::test_checkpoint_resume_after_crash

# Lint (the rule set lives in pyproject.toml)
ruff check .
```

## Architecture

### Multi-Agent Workflow

The system orchestrates agents via a LangGraph-based state machine (`TradingAgentsGraph`):

```
Analyst Team (parallel execution)
├── Fundamentals Analyst (balance sheet, income, cashflow; A-shares: institutional holdings)
├── Market/Technical Analyst (MACD, RSI, price patterns; A-shares: fund flow)
├── Sentiment Analyst (social media scoring; A-shares: China social platforms)
└── News Analyst (company and global news, macro events; A-shares: dragon-tiger list)
        ↓
Researcher Team (debate rounds)
├── Bull Researcher (optimistic case)
├── Bear Researcher (pessimistic case)
└── Research Manager (synthesizes → ResearchPlan with a PortfolioRating)
        ↓
Trading Team
└── Trader (translates plan → TraderProposal: Buy/Hold/Sell, entry, stop, sizing)
        ↓
Risk Management (debate rounds)
├── Aggressive Debator
├── Neutral Debator
├── Conservative Debator (produce risk assessment)
        ↓
Portfolio Management
└── Portfolio Manager (final PortfolioDecision: Buy/Overweight/Hold/Underweight/Sell)
```

### Key Components

**`tradingagents/graph/`**
- `trading_graph.py` - Main orchestrator (`TradingAgentsGraph`); also the run lifecycle API (`create_run_state`, `begin_checkpoint`, `stream_run`, `record_decision`, ...) used by the CLI and Web
- `setup.py` - Graph node construction
- `analyst_execution.py` - Analyst specs (node name, report key, `TOOLS`) and the parallel analyst subgraphs
- `propagation.py` - Initial state and graph invocation arguments
- `conditional_logic.py` - State transition routing
- `checkpointer.py` - LangGraph SQLite checkpoint management

**`tradingagents/agents/`**
- `analysts/` - 4 analysts (fundamentals, market, sentiment, news); `turn.py` runs one tool-using turn
- `researchers/` - Bull/Bear researchers
- `managers/` - Research Manager + Portfolio Manager (structured output)
- `trader/` - Trader agent (structured output)
- `risk_mgmt/` - 3 risk debators (aggressive, neutral, conservative)
- `tools.py` - The data tools the analysts call (routed through `dataflows/router.py`)
- `extended_tools.py` - A-share-only tools (`get_fund_flow`, `get_lhb_detail`, `get_institute_hold`) and `offered_tools()`, which drops them for non-A-share tickers
- `context.py` - Prompt context: instrument identity (akshare first for `.SS/.SZ/.HK`), output language, portfolio
- `rating.py` - 5-tier rating vocabulary and the decision → rating parser
- `schemas.py` - Pydantic schemas for structured agents
- `structured.py` - Structured-output invocation with free-text fallback
- `state.py` - Agent state definitions

**`tradingagents/memory/`**
- `log.py` - Append-only markdown decision log and past-context retrieval
- `settlement.py` - Scores pending decisions against their benchmark once the holding window has traded (Yahoo prices, akshare fallback)
- `reflection.py` - Turns a settled outcome into a lesson

**`tradingagents/dataflows/`**
- `router.py` - Vendor registry (`TOOLS_CATEGORIES`, `VENDOR_METHODS`) and `route_to_vendor()`
- `vendors/` - Vendor implementations: `akshare.py`, `china_social.py`, `yahoo/`, `alpha_vantage/`, `sec_edgar.py`, `fred.py`, `polymarket.py`, `reddit.py`, `stocktwits.py`
- `config.py` - Process config plus the per-run config (`run_config`)
- `symbols.py` - Symbol normalization and `safe_ticker_component()`
- `errors.py` - `NoMarketDataError`, `VendorUnavailableError`, `VendorNotConfiguredError`
- `date_window.py` - Point-in-time (look-ahead) rules for dated calls

**`tradingagents/runtime/`** (shared by CLI and Web)
- `runner.py` - Streaming runner over the graph's lifecycle API
- `spec.py` - Validation and config for one analysis run
- `events.py` / `tracking.py` - UI events and per-analyst wall time
- `stats.py` - LLM/tool usage callback

**`tradingagents/llm_clients/`**
- `factory.py` - Multi-provider LLM client factory
- `{provider}_client.py` - Provider-specific clients
- `provider_catalog.py` - Providers offered by the CLI/Web and their endpoints
- `model_catalog.py` - Unified model registry across providers
- `validators.py` - Model name validation per provider

**`cli/`**
- `main.py` - Typer app: interactive entry, `analyze` (flag interface), `backtest`, `login`/`logout`, `web`
- `run.py` - Runs one analysis: build the graph, stream into the live view, save the report, export vn.py
- `selections.py` - Interactive choices (ticker, date, analysts, depth, provider, models); flags answer their steps
- `prompts.py` - Individual prompts and ticker parsing (accepts Chinese A-share names)
- `display.py` - Live Rich view; `prefs.py` - last run's choices; `updates.py` - release notices

**`web/`** - FastAPI app (`web/app.py`) and templates; runs through `tradingagents/runtime/`

**`tradingagents/integrations/vnpy/`** - Final decision → versioned vn.py JSON signal (`signal.py`, `exporter.py`)

### Configuration System

All behavior is controlled via `DEFAULT_CONFIG` in `tradingagents/default_config.py` (most keys can also be set via `TRADINGAGENTS_*` environment variables, see `.env.example`):

```python
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"        # see provider_catalog.py for every provider
config["deep_think_llm"] = "gpt-6-sol"   # Research Manager, Portfolio Manager
config["quick_think_llm"] = "gpt-6-luna" # analysts, researchers, trader, risk debators
config["backend_url"] = None             # Provider-specific override (default: None → native endpoints)
config["max_debate_rounds"] = 1          # Researcher debate iterations
config["max_risk_discuss_rounds"] = 1    # Risk management debate iterations
config["output_language"] = "English"    # Report language (debate stays English for quality)
config["checkpoint_enabled"] = False     # LangGraph checkpoint resume (opt-in)
config["data_vendors"] = {               # Category-level vendor chains, tried in order
    "core_stock_apis": "akshare,yfinance",
    "technical_indicators": "akshare,yfinance",
    "fundamental_data": "akshare,sec_edgar,yfinance",
    "news_data": "akshare,yfinance",
    "macro_data": "fred",
    "prediction_markets": "polymarket",
}
```

**Provider-specific thinking controls:**
- `google_thinking_level` - "high", "minimal" (Gemini thinking mode)
- `openai_reasoning_effort` - "low", "medium", "high" (OpenAI reasoning effort)
- `anthropic_effort` - "low", "medium", "high" (Claude extended thinking)

A graph binds its config to each run (`run_config`), so concurrent runs with different configs do not see each other's vendors.

### Persistence Features

**Decision Log (`~/.tradingagents/memory/trading_memory.md`)**
- Automatically appends each completed decision
- Later runs settle pending decisions: realized return and alpha against a regional benchmark (`benchmark_map`, overridable with `benchmark_ticker`), then a reflection
- Injects recent same-ticker decisions + cross-ticker lessons into the Portfolio Manager prompt; dated runs only see outcomes known by their date
- Override path: `TRADINGAGENTS_MEMORY_LOG_PATH`
- Optional rotation: `config["memory_log_max_entries"]` (prunes oldest resolved entries)

**Checkpoint Resume (opt-in via `--checkpoint`)**
- LangGraph saves state after each node to per-ticker SQLite DB
- Crashed/interrupted runs resume from last successful step
- Databases: `~/.tradingagents/cache/checkpoints/<TICKER>.db`
- Override base: `TRADINGAGENTS_CACHE_DIR`
- Clear all: `tradingagents --clear-checkpoints` (or `tradingagents analyze --clear-checkpoints`)

### Structured Output Agents

Four agents use structured output (`agents/structured.py`), falling back to free text when a provider cannot produce it (the Portfolio Manager then parses its rating from the text with `agents/rating.py`):

1. **Sentiment Analyst** → `SentimentReport` (overall_band, overall_score, confidence, narrative)
2. **Research Manager** → `ResearchPlan` (recommendation: PortfolioRating, rationale, strategic_actions)
3. **Trader** → `TraderProposal` (action: TraderAction, reasoning, entry_price, stop_loss, position_sizing)
4. **Portfolio Manager** → `PortfolioDecision` (rating: PortfolioRating, executive_summary, investment_thesis, price_target, time_horizon)

Each provider's native mode is used:
- OpenAI/xAI: `json_schema`
- Gemini: `response_schema`
- Anthropic: tool-use
- DeepSeek/Qwen/GLM/Azure: function-calling

Render helpers in `schemas.py` (`render_research_plan`, `render_trader_proposal`, `render_pm_decision`, `render_sentiment_report`) convert Pydantic instances back to markdown for display/memory/reports.

## Data Vendor Abstraction

**Design:** Analysts call the tools in `tradingagents/agents/tools.py` (`get_stock_data`, `get_indicators`, `get_fundamentals`, `get_news`, ...). Each tool calls `route_to_vendor()` in `tradingagents/dataflows/router.py`, which walks the configured vendor chain.

**Vendor Selection Hierarchy:**
1. Tool-level override: `config["tool_vendors"]["get_stock_data"] = "alpha_vantage"`
2. Category-level chain: `config["data_vendors"]["core_stock_apis"] = "akshare,yfinance"`
3. No configuration (`"default"`): every vendor that implements the method

**Fallback:** the configured list *is* the chain — vendors not listed are never tried. A vendor that raises `NoMarketDataError`, `VendorUnavailableError` or any other error passes the call to the next vendor; when the chain is exhausted the tool returns `NO_DATA_AVAILABLE` / `DATA_UNAVAILABLE` text for the agent. A vendor that cannot serve a call must raise, not return an error table, or the router treats it as an answer. Tools may pass `_exclude_vendors` (e.g. global news drops akshare for non-A-share tickers).

**Supported Vendors:**
- `akshare` - Default first vendor, no API key; A-shares, HK and US (coverage varies by data type)
- `yfinance` - No API key required
- `sec_edgar` - US filers' statements (set `SEC_EDGAR_USER_AGENT`)
- `alpha_vantage` - Requires `ALPHA_VANTAGE_API_KEY`
- `fred` - Macro data, requires `FRED_API_KEY`
- `polymarket` - Prediction markets, keyless

## Environment Variables

**LLM Providers (set the key for your chosen provider):**
```bash
OPENAI_API_KEY          # OpenAI (GPT)
GOOGLE_API_KEY          # Google (Gemini)
ANTHROPIC_API_KEY       # Anthropic (Claude)
EVOLINK_API_KEY         # Evolink (Claude API)
XAI_API_KEY             # xAI (Grok)
DEEPSEEK_API_KEY        # DeepSeek
DASHSCOPE_API_KEY       # Qwen (International; DASHSCOPE_CN_API_KEY for China)
ZHIPU_API_KEY           # GLM (International; ZHIPU_CN_API_KEY for China)
MINIMAX_API_KEY         # MiniMax (International; MINIMAX_CN_API_KEY for China)
OPENROUTER_API_KEY      # OpenRouter
MISTRAL_API_KEY         # Mistral
MOONSHOT_API_KEY        # Kimi
GROQ_API_KEY            # Groq
NVIDIA_API_KEY          # NVIDIA NIM
```

**Data Sources:**
```bash
ALPHA_VANTAGE_API_KEY   # Alpha Vantage (optional)
SEC_EDGAR_USER_AGENT    # SEC EDGAR contact string
FRED_API_KEY            # FRED macro data (optional)
TYPESAFE_API_KEY        # Social-post screening (optional)
```

**Enterprise (copy `.env.enterprise.example` to `.env.enterprise`):**
```bash
AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_DEPLOYMENT_NAME
```

**Persistence Overrides:**
```bash
TRADINGAGENTS_RESULTS_DIR         # Default: ~/.tradingagents/logs
TRADINGAGENTS_CACHE_DIR           # Default: ~/.tradingagents/cache
TRADINGAGENTS_MEMORY_LOG_PATH     # Default: ~/.tradingagents/memory/trading_memory.md
```

## Important Patterns

### Ticker Validation
**Always use `safe_ticker_component()` before using a ticker in file paths:**
```python
from tradingagents.dataflows.symbols import safe_ticker_component

ticker = "NVDA"  # user input
safe_name = safe_ticker_component(ticker)  # validates and sanitizes
path = os.path.join(cache_dir, f"{safe_name}_data.json")  # safe for filesystem
```

This prevents path traversal attacks (e.g., `../../../etc/passwd` as ticker).

A-share tickers normalize to `.SS` (Shanghai) and `.SZ` (Shenzhen); `.SH` input becomes `.SS`. Use `is_cn_ticker()` from `tradingagents/agents/context.py` for A-share-only behavior.

### Agent Tools
Agents receive tool instances via `ToolNode` in LangGraph. When modifying agent prompts or tools:
- Analysts use data fetching tools; each analyst module declares its `TOOLS` tuple, which builds its `ToolNode`
- The ticker and trade date come from graph state (`InjectedState("company_of_interest")`, `InjectedState("trade_date")`), never from the model; dated tools clamp the model's date with `as_of(curr_date, trade_date)` so they cannot read past the run's date
- A-share-only tools go in `extended_tools.py` and are filtered per ticker by `offered_tools()`
- Researchers/Risk use no tools (pure reasoning over analyst reports)
- Trader/Portfolio Manager use no tools (decision-making based on upstream synthesis)

### Model Selection
The framework uses two-tier LLM approach:
- **Deep thinking** (`deep_think_llm`) - Research Manager and Portfolio Manager
- **Quick thinking** (`quick_think_llm`) - Analysts, Bull/Bear researchers, Trader, risk debators

When adding new agent nodes, choose the appropriate LLM tier based on task complexity.

### Testing Notes
- Tests use lazy LLM imports + placeholder API keys (via pytest fixtures in `conftest.py`)
- Integration tests (marked `@pytest.mark.integration`) may require real API keys
- `test_safe_ticker_component.py` covers the security-critical ticker validation
- `test_structured_agents.py` validates Pydantic schema compliance across providers
- `test_checkpoint_resume.py` tests LangGraph state persistence
- `test_graph_end_to_end.py` runs the whole graph offline (akshare and yfinance patched)
- Tests that assume a vendor should pin it in config rather than rely on the default chain

## Development Workflow

When modifying the trading graph:
1. **Graph structure changes** → `tradingagents/graph/setup.py` (node definitions)
2. **Routing logic** → `tradingagents/graph/conditional_logic.py` (state transitions)
3. **New agents** → Add to `tradingagents/agents/{category}/` + register in `setup.py`
4. **New analyst tools** → `agents/tools.py` (or `extended_tools.py` if A-share only), add to the analyst's `TOOLS`
5. **New data vendors** → Implement in `dataflows/vendors/`, register in `dataflows/router.py`
6. **New LLM providers** → Add client in `llm_clients/`, update `factory.py`, `provider_catalog.py`, extend `model_catalog.py`

Always test changes with:
```bash
# Unit tests and lint first
pytest -m unit
ruff check .

# Then a full offline run
pytest tests/test_graph_end_to_end.py

# Finally a real run
tradingagents analyze --no-interactive ...   # or: python main.py
```

## License & Disclaimer

This framework is for research purposes. Trading performance varies based on models, temperature, data quality, and non-deterministic factors. See LICENSE and https://tauric.ai/disclaimer/ for full disclaimer.
