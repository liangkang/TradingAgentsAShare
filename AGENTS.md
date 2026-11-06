# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

TradingAgents is a multi-agent LLM-powered financial trading framework built with LangGraph. It simulates a real-world trading firm with specialized agents (analysts, researchers, traders, risk managers, portfolio managers) that collaboratively evaluate market conditions and make trading decisions.

**Key Technologies:**
- LangGraph for agent orchestration and state management
- Multi-provider LLM support (OpenAI, Anthropic, Google, xAI, DeepSeek, Qwen, GLM, Azure, OpenRouter, Ollama)
- Structured output via Pydantic schemas (Research Manager, Trader, Portfolio Manager)
- Backtrader for backtesting
- yfinance/Alpha Vantage for market data

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

# Programmatic usage (via Python)
python main.py  # example implementation
```

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

# Diagnostic for structured output agents (all providers)
python scripts/smoke_structured_output.py
```

## Architecture

### Multi-Agent Workflow

The system orchestrates agents via a LangGraph-based state machine (`TradingAgentsGraph`):

```
Analyst Team (parallel execution)
├── Fundamentals Analyst (balance sheet, income, cashflow)
├── Market/Technical Analyst (MACD, RSI, price patterns)
├── Sentiment Analyst (social media scoring)
└── News Analyst (global news, macro events)
        ↓
Researcher Team (debate rounds)
├── Bull Researcher (optimistic case)
├── Bear Researcher (pessimistic case)
└── Research Manager (synthesizes → PortfolioRating: Buy/Overweight/Hold/Underweight/Sell)
        ↓
Trading Team
└── Trader (translates plan → TraderAction: Buy/Hold/Sell with position size)
        ↓
Risk Management (debate rounds)
├── Aggressive Debator
├── Neutral Debator
├── Conservative Debator (produce risk assessment)
        ↓
Portfolio Management
└── Portfolio Manager (final approval/rejection → transaction execution)
```

### Key Components

**`tradingagents/graph/`**
- `trading_graph.py` - Main orchestrator (`TradingAgentsGraph` class)
- `setup.py` - Graph node construction
- `propagation.py` - Execution logic (`propagate()` method)
- `conditional_logic.py` - State transition routing
- `checkpointer.py` - LangGraph SQLite checkpoint management
- `reflection.py` - Post-decision reflection and memory update
- `signal_processing.py` - Decision → 5-tier rating conversion

**`tradingagents/agents/`**
- `analysts/` - 4 analyst types (fundamentals, market, social_media, news)
- `researchers/` - Bull/Bear researchers + Research Manager (structured output)
- `trader/` - Trader agent (structured output)
- `risk_mgmt/` - 3 risk debators (aggressive, neutral, conservative)
- `managers/` - Research Manager + Portfolio Manager (structured output)
- `schemas.py` - Pydantic schemas for structured agents
- `utils/` - Agent states, memory log, BM25 retrieval

**`tradingagents/llm_clients/`**
- `factory.py` - Multi-provider LLM client factory
- `{provider}_client.py` - Provider-specific clients (OpenAI, Anthropic, Google, xAI, DeepSeek, Qwen, GLM, Azure, Ollama)
- `model_catalog.py` - Unified model registry across providers
- `validators.py` - Model name validation per provider

**`tradingagents/dataflows/`**
- `interface.py` - Abstract tool interface (vendor-agnostic)
- `y_finance.py` / `alpha_vantage*.py` - Vendor implementations
- `stockstats_utils.py` - Technical indicator calculations
- `config.py` - Vendor routing configuration

**`cli/`**
- `main.py` - Typer-based interactive CLI with Rich UI
- `config.py` - User input prompts (ticker, date, provider, depth)
- `stats_handler.py` - LangChain callback for LLM/tool usage tracking

### Configuration System

All behavior is controlled via `DEFAULT_CONFIG` in `tradingagents/default_config.py`:

```python
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"        # openai, google, anthropic, xai, deepseek, qwen, glm, azure, openrouter, ollama
config["deep_think_llm"] = "gpt-5.4"     # Complex reasoning (analysts, researchers, portfolio manager)
config["quick_think_llm"] = "gpt-5.4-mini" # Quick tasks (signal processing, utilities)
config["backend_url"] = None             # Provider-specific override (default: None → native endpoints)
config["max_debate_rounds"] = 1          # Researcher debate iterations
config["max_risk_discuss_rounds"] = 1    # Risk management debate iterations
config["output_language"] = "English"    # Report language (debate stays English for quality)
config["checkpoint_enabled"] = False     # LangGraph checkpoint resume (opt-in)
config["data_vendors"] = {               # Category-level vendor selection
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}
```

**Provider-specific thinking controls:**
- `google_thinking_level` - "high", "minimal" (Gemini thinking mode)
- `openai_reasoning_effort` - "low", "medium", "high" (OpenAI reasoning effort)
- `anthropic_effort` - "low", "medium", "high" (Codex extended thinking)

### Persistence Features

**Decision Log (`~/.tradingagents/memory/trading_memory.md`)**
- Automatically appends each completed decision
- Next same-ticker run fetches realized return (raw + alpha vs SPY), generates reflection
- Injects recent same-ticker decisions + cross-ticker lessons into Portfolio Manager prompt
- Override path: `TRADINGAGENTS_MEMORY_LOG_PATH`
- Optional rotation: `config["memory_log_max_entries"]` (prunes oldest resolved entries)

**Checkpoint Resume (opt-in via `--checkpoint`)**
- LangGraph saves state after each node to per-ticker SQLite DB
- Crashed/interrupted runs resume from last successful step
- Databases: `~/.tradingagents/cache/checkpoints/<TICKER>.db`
- Override base: `TRADINGAGENTS_CACHE_DIR`
- Clear all: `tradingagents analyze --clear-checkpoints`

### Structured Output Agents

Three decision-making agents use `llm.with_structured_output(Schema)`:

1. **Research Manager** → `ResearchPlan` (recommendation: PortfolioRating, rationale: str, strategic_actions: list)
2. **Trader** → `TraderProposal` (action: TraderAction, position_size: int, reasoning: str, risk_considerations: str)
3. **Portfolio Manager** → `PortfolioDecision` (decision: Literal["Approve", "Reject"], rating: PortfolioRating, final_reasoning: str)

Each provider's native mode is used:
- OpenAI/xAI: `json_schema`
- Gemini: `response_schema`
- Anthropic: tool-use
- DeepSeek/Qwen/GLM/Azure: function-calling

Render helpers (`{schema}.render_as_markdown()`) convert Pydantic instances back to markdown for display/memory/reports.

## Data Vendor Abstraction

**Design:** All agents call abstract tool methods (`get_stock_data`, `get_fundamentals`, `get_news`, etc.) via `tradingagents/agents/utils/agent_utils.py`. The `dataflows/interface.py` routes to vendor implementations based on config.

**Vendor Selection Hierarchy:**
1. Tool-level override: `config["tool_vendors"]["get_stock_data"] = "alpha_vantage"`
2. Category-level default: `config["data_vendors"]["core_stock_apis"] = "yfinance"`
3. Fallback: yfinance

**Supported Vendors:**
- `yfinance` - Default, no API key required
- `alpha_vantage` - Requires `ALPHA_VANTAGE_API_KEY`

## Environment Variables

**LLM Providers (set the key for your chosen provider):**
```bash
OPENAI_API_KEY          # OpenAI (GPT)
GOOGLE_API_KEY          # Google (Gemini)
ANTHROPIC_API_KEY       # Anthropic (Codex)
XAI_API_KEY             # xAI (Grok)
DEEPSEEK_API_KEY        # DeepSeek
DASHSCOPE_API_KEY       # Qwen (Alibaba)
ZHIPU_API_KEY           # GLM (Zhipu)
OPENROUTER_API_KEY      # OpenRouter
```

**Data Sources:**
```bash
ALPHA_VANTAGE_API_KEY   # Alpha Vantage (optional, yfinance is default)
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
from tradingagents.dataflows.utils import safe_ticker_component

ticker = "NVDA"  # user input
safe_name = safe_ticker_component(ticker)  # validates and sanitizes
path = os.path.join(cache_dir, f"{safe_name}_data.json")  # safe for filesystem
```

This prevents path traversal attacks (e.g., `../../../etc/passwd` as ticker).

### Agent Tools
Agents receive tool instances via `ToolNode` in LangGraph. When modifying agent prompts or tools:
- Analysts use data fetching tools (stock data, indicators, fundamentals, news)
- Researchers/Risk use no tools (pure reasoning over analyst reports)
- Trader/Portfolio Manager use no tools (decision-making based on upstream synthesis)

### Model Selection
The framework uses two-tier LLM approach:
- **Deep thinking** (`deep_think_llm`) - Complex reasoning tasks: analysts, researchers, portfolio decisions
- **Quick thinking** (`quick_think_llm`) - Fast tasks: signal processing, basic transforms

When adding new agent nodes, choose the appropriate LLM tier based on task complexity.

### Testing Notes
- Tests use lazy LLM imports + placeholder API keys (via pytest fixtures in `conftest.py`)
- Integration tests (marked `@pytest.mark.integration`) may require real API keys
- `test_safe_ticker_component.py` covers the security-critical ticker validation
- `test_structured_agents.py` validates Pydantic schema compliance across providers
- `test_checkpoint_resume.py` tests LangGraph state persistence

## Development Workflow

When modifying the trading graph:
1. **Graph structure changes** → `tradingagents/graph/setup.py` (node definitions)
2. **Routing logic** → `tradingagents/graph/conditional_logic.py` (state transitions)
3. **New agents** → Add to `tradingagents/agents/{category}/` + register in `setup.py`
4. **New data vendors** → Implement in `dataflows/`, register in `interface.py`
5. **New LLM providers** → Add client in `llm_clients/`, update `factory.py`, extend `model_catalog.py`

Always test changes with:
```bash
# Unit tests first
pytest -m unit

# Then smoke test the full pipeline
python scripts/smoke_structured_output.py

# Finally integration test with real API
python main.py
```

## License & Disclaimer

This framework is for research purposes. Trading performance varies based on models, temperature, data quality, and non-deterministic factors. See LICENSE and https://tauric.ai/disclaimer/ for full disclaimer.
