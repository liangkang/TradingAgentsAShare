---
name: tradingagents-analyze
description: >-
  Map user intent to tradingagents analyze CLI flags and run a non-interactive
  stock analysis. Use when user types /analyze, asks to analyze a ticker,
  run TradingAgents, or start a trading analysis.
---

# TradingAgents Analyze

Run a full multi-agent trading analysis via `tradingagents analyze --no-interactive`.

## When to Use

- User types `/analyze`
- User asks to analyze a stock, ticker, or Chinese A-share name
- User wants to run TradingAgents headlessly from Cursor

## Workflow

Copy this checklist and track progress:

```
Task Progress:
- [ ] Step 1: Parse user intent
- [ ] Step 2: Fill missing params (AskQuestion if needed)
- [ ] Step 3: Show command summary for confirmation
- [ ] Step 4: Execute analyze command
- [ ] Step 5: Report output paths and suggest /report-qa
```

### Step 1: Parse user intent

Extract from the user message:

| Intent | CLI flag |
|--------|----------|
| Ticker / stock name | `--ticker` |
| Analysis date | `--date YYYY-MM-DD` |
| Analyst team | `--analysts` (comma-separated) |
| Research depth | `--research-depth` (1–5) |
| Report language | `--output-language` |
| LLM provider | `--provider` |
| Deep / quick models | `--deep-model`, `--quick-model` |
| Checkpoint resume | `--checkpoint` |
| Save full report | `--save-report` (always include) |

### Step 2: Fill defaults

**Priority:** user message → `.env` / `TRADINGAGENTS_*` → `DEFAULT_CONFIG` → skill fallback.

| Param | Default when missing |
|-------|---------------------|
| `--date` | Today (`YYYY-MM-DD`); must not be in the future |
| `--analysts` | `market,fundamentals,news,social` for stocks; drop `fundamentals` for crypto |
| `--research-depth` | `TRADINGAGENTS_MAX_DEBATE_ROUNDS` or `1` |
| `--output-language` | `Chinese` if user writes in Chinese; else `TRADINGAGENTS_OUTPUT_LANGUAGE` or `English` |
| `--provider` | `TRADINGAGENTS_LLM_PROVIDER` or **`deepseek`** (skill fallback; not `evolink`) |
| `--deep-model` | `TRADINGAGENTS_DEEP_THINK_LLM` or `deepseek-v4-pro` when provider is `deepseek` |
| `--quick-model` | `TRADINGAGENTS_QUICK_THINK_LLM` or `deepseek-v4-flash` when provider is `deepseek` |
| `--save-path` | `./reports/{TICKER}_{YYYYMMDD_HHMMSS}` |

**Required in non-interactive mode:** `--ticker`, `--date`, `--analysts`, `--provider`, `--deep-model`, `--quick-model`.

If any required param is missing, use `AskQuestion` to collect it before proceeding.

Read `.env` and `tradingagents/default_config.py` for env-based defaults. For model IDs, see `tradingagents/llm_clients/model_catalog.py`.

### Step 3: Confirm before running

Show the user a summary:

```
Ticker:       NVDA
Date:         2026-08-02
Analysts:     market,fundamentals,news,social
Provider:     deepseek
Deep model:   deepseek-v4-pro
Quick model:  deepseek-v4-flash
Depth:        1
Language:     Chinese
Save path:    ./reports/NVDA_20260802_214500
```

Proceed unless the user objects.

### Step 4: Execute

Run from the repo root. **Always** pass `--no-interactive` and `--save-report`.

```bash
cd /Users/liangkang/Works/code/source/TradingAgents
tradingagents analyze \
  --ticker "<TICKER>" \
  --date "<YYYY-MM-DD>" \
  --analysts "<comma-separated>" \
  --provider "<provider>" \
  --deep-model "<model>" \
  --quick-model "<model>" \
  --research-depth <1-5> \
  --output-language "<language>" \
  --save-report \
  --save-path "<path>" \
  --no-interactive
```

**Runtime notes:**

- Analysis takes several minutes to tens of minutes. Set a large `block_until_ms` or run in background and poll terminal output.
- Rich Live UI output in the terminal is expected.
- Ensure the chosen provider's API key is set in `.env`.

### Step 5: Hand off

After success, tell the user:

```
Report saved: <save-path>/complete_report.md
Section reports: ~/.tradingagents/logs/<TICKER>/<DATE>/reports/
Follow-up: use /report-qa or ask questions about the report.
```

## Additional Resources

- CLI parameter details: [reference.md](reference.md)
