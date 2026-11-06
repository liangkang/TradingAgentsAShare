---
name: tradingagents-report-qa
description: >-
  Read a TradingAgents complete_report.md and answer follow-up questions
  grounded in the report. Use when user types /report-qa, asks about a
  previous analysis, portfolio decision, rating, or saved report.
---

# TradingAgents Report QA

Answer user questions about a completed TradingAgents analysis by reading `complete_report.md`.

## When to Use

- User types `/report-qa`
- User asks about a previous analysis result, rating, or recommendation
- User wants to understand bull/bear debate, risk views, or portfolio decision
- Follow-up after `/analyze` (tradingagents-analyze skill)

## Workflow

```
Task Progress:
- [ ] Step 1: Locate complete_report.md
- [ ] Step 2: Read the full report
- [ ] Step 3: Answer grounded in report content
```

### Step 1: Locate the report

Search in this order:

1. **User-provided path** — absolute or relative path to `complete_report.md` or its parent directory
2. **Session context** — path printed by a prior `/analyze` run in this conversation
3. **Latest by ticker** — `{cwd}/reports/{TICKER}_*/complete_report.md` (newest timestamp)
4. **Saved reports dir** — scan `~/.tradingagents/logs/{TICKER}/{DATE}/` if a complete report was copied there

If multiple candidates exist, pick the newest or ask the user to confirm.

### Step 2: Read the report

Read `complete_report.md` in full before answering.

The report has five sections (see [reference.md](reference.md)):

- **I.** Analyst Team Reports
- **II.** Research Team Decision (bull / bear / research manager)
- **III.** Trading Team Plan
- **IV.** Risk Management Team Decision
- **V.** Portfolio Manager Decision — **primary source for final rating**

If a question needs detail beyond the consolidated file, read sibling files under the same save directory (e.g. `2_research/bull.md`, `5_portfolio/decision.md`).

### Step 3: Answer

**Rules:**

1. Ground every claim in the report; cite the section (e.g. "Section V — Portfolio Manager").
2. If the report does not cover the question, say **「报告中未提及」** and do not speculate.
3. Do not invent price targets, ratings, or trade actions not present in the report.

**Answer structure:**

1. **Direct conclusion** — final Rating and recommended action
2. **Evidence** — bullet points from relevant sections
3. **Cross-reference** (when useful) — e.g. how PM decision relates to bull/bear debate or trader proposal

## Common Question Types

| User asks | Where to look |
|-----------|---------------|
| 最终评级 / 建议 | Section V — `**Rating**` |
| 为什么 Hold 而不是 Buy | Sections II (debate), IV (risk), V (thesis) |
| 主要风险 | Section IV + Section V `Investment Thesis` |
| 目标价 / 持有周期 | Section V — `Price Target`, `Time Horizon` |
| 交易员提案 | Section III |
| 基本面 / 技术面观点 | Section I (fundamentals / market analysts) |
| 多空分歧 | Section II (bull vs bear) |

## Handoff from Analyze Skill

When continuing from `/analyze`, the report path is typically:

```
./reports/{TICKER}_{YYYYMMDD_HHMMSS}/complete_report.md
```

Use that path directly unless the user specifies another.

## Additional Resources

- Report structure and path rules: [reference.md](reference.md)
