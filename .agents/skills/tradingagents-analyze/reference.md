# TradingAgents Analyze — Reference

## CLI Flags

| Flag | Required (non-interactive) | Description |
|------|---------------------------|-------------|
| `--ticker` | Yes | Symbol or exact Chinese A-share name (e.g. `0700.HK`, `长飞光纤`) |
| `--date` | Yes | Analysis date `YYYY-MM-DD`; cannot be in the future |
| `--analysts` | Yes | Comma-separated: `market`, `fundamentals`, `news`, `social` |
| `--provider` | Yes | LLM provider (e.g. `deepseek`, `openai`, `google`, `anthropic`); skill default: `deepseek` |
| `--deep-model` | Yes | Deep-thinking model ID (skill default with `deepseek`: `deepseek-v4-pro`) |
| `--quick-model` | Yes | Quick-thinking model ID (skill default with `deepseek`: `deepseek-v4-flash`) |
| `--research-depth` | No | Debate depth 1–5 (default: env or `1`) |
| `--output-language` | No | Report language (default: `English`) |
| `--no-interactive` | — | Skip prompts; **required for Cursor** |
| `--save-report` | — | Write `complete_report.md` after run |
| `--save-path` | No | Output directory (default: `./reports/{TICKER}_{timestamp}`) |
| `--checkpoint` / `--no-checkpoint` | No | Enable LangGraph checkpoint resume |
| `--clear-checkpoints` | No | Delete saved checkpoints before run |

## Environment Variables

From `.env.example` / `TRADINGAGENTS_*`:

| Variable | Config key |
|----------|-----------|
| `TRADINGAGENTS_LLM_PROVIDER` | `llm_provider` |
| `TRADINGAGENTS_DEEP_THINK_LLM` | `deep_think_llm` |
| `TRADINGAGENTS_QUICK_THINK_LLM` | `quick_think_llm` |
| `TRADINGAGENTS_OUTPUT_LANGUAGE` | `output_language` |
| `TRADINGAGENTS_MAX_DEBATE_ROUNDS` | `max_debate_rounds` |
| `TRADINGAGENTS_CHECKPOINT_ENABLED` | `checkpoint_enabled` |

## Output Paths

| Artifact | Path |
|----------|------|
| Complete report | `{--save-path}/complete_report.md` |
| Per-section tree | `{--save-path}/1_analysts/`, `2_research/`, … `5_portfolio/decision.md` |
| Live section dumps | `~/.tradingagents/logs/{TICKER}/{DATE}/reports/*.md` |
| Message log | `~/.tradingagents/logs/{TICKER}/{DATE}/message_tool.log` |

Override base log dir with `TRADINGAGENTS_RESULTS_DIR`.

## Analyst Selection Rules

- Valid analyst keys: `market`, `fundamentals`, `news`, `social`
- Crypto assets: `fundamentals` is rejected
- At least one analyst required

## Example Commands

```bash
# US stock, Chinese report, save full report
tradingagents analyze \
  --ticker NVDA \
  --date 2026-08-02 \
  --analysts market,fundamentals,news,social \
  --provider openai \
  --deep-model gpt-5.5 \
  --quick-model gpt-5.4-mini \
  --output-language Chinese \
  --save-report \
  --no-interactive

# Hong Kong stock
tradingagents analyze \
  --ticker 0700.HK \
  --date 2026-08-02 \
  --analysts market,news,social \
  --provider deepseek \
  --deep-model deepseek-v4-flash \
  --quick-model deepseek-v4-flash \
  --save-report \
  --no-interactive
```
