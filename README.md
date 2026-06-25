# akshare 数据源迁移说明

本文档说明将 TradingAgents 默认数据源从 yfinance 切换到 akshare 的改动。

## 背景

yfinance 在国内网络环境下连接频繁中断（限流、超时），导致分析流程无法正常完成。akshare 是一个免费的 Python 金融数据接口库，通过多源架构接入新浪财经、东方财富、财新等国内数据源，在国内网络下稳定可用。

## 改动概览

### 数据源切换

| 类别 | 原数据源 | 现数据源 | 上游 |
|---|---|---|---|
| 行情 OHLCV（美股） | yfinance | akshare | Yahoo Finance US API |
| 行情 OHLCV（A 股） | yfinance | akshare | 新浪财经（主）→ 腾讯（备） |
| 行情 OHLCV（港股） | yfinance | akshare | 新浪财经（主）→ 腾讯（备） |
| 技术指标 | yfinance + stockstats | akshare + stockstats | 新浪财经（主）→ 腾讯（备） |
| 公司名称（A 股） | yfinance | akshare | `stock_info_a_code_name` |
| 公司名称（港股） | yfinance | akshare | 东方财富 |
| 基本面（A 股） | yfinance | akshare | 东方财富 |
| 基本面（美股） | akshare 不覆盖 | yfinance（自动回退） | — |
| 个股新闻（A 股） | yfinance | akshare | 东方财富 |
| 个股新闻（美股/港股） | akshare 不覆盖 | yfinance（自动回退） | — |
| 宏观新闻 | yfinance Search | akshare | 财新 + 央视新闻 |
| 内幕交易 | akshare 不覆盖 | yfinance（自动回退） | — |

### 回退机制

所有数据类别配置为 `"akshare,yfinance"`，路由器按优先级尝试：
1. 先走 akshare
2. A 股行情在 akshare 内部也有兜底：新浪财经（主）→ 腾讯财经（备）
3. akshare 抛出 `NoMarketDataError`（表示该类别不支持此市场）→ 自动回退到 yfinance
4. akshare 抛出网络错误（`ConnectionError`/`TimeoutError`/`OSError`）→ 透传，同样回退到 yfinance

### 配置变更

`default_config.py` 中 `data_vendors` 的默认值：

```python
# 原来
"data_vendors": {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}

# 现在
"data_vendors": {
    "core_stock_apis": "akshare,yfinance",
    "technical_indicators": "akshare,yfinance",
    "fundamental_data": "akshare,yfinance",
    "news_data": "akshare,yfinance",
}
```

`global_news_queries` 同步更新为中国市场相关关键词。

### 新增文件

- `tradingagents/dataflows/akshare_vendor.py` — akshare 供应商模块，实现了 9 个标准接口函数

### 修改文件

| 文件 | 变更 |
|---|---|
| `pyproject.toml` | 添加 `akshare>=1.18.0` 依赖 |
| `tradingagents/dataflows/akshare_vendor.py` | 新建，akshare 数据供应商 |
| `tradingagents/dataflows/interface.py` | 注册 akshare 到 `VENDOR_LIST` 和 `VENDOR_METHODS` |
| `tradingagents/dataflows/stockstats_utils.py` | `load_ohlcv` 拆分为 yfinance/akshare 两个加载器，添加供应商感知分发函数 |
| `tradingagents/dataflows/y_finance.py` | 内部调用改为 `load_ohlcv_yfinance`（自包含） |
| `tradingagents/default_config.py` | 默认供应商改为 `akshare,yfinance`；宏观新闻查询词更新 |
| `tradingagents/agents/utils/agent_utils.py` | `resolve_instrument_identity()` 新增 akshare 回退（A 股/港股） |
| `tradingagents/graph/trading_graph.py` | `_fetch_returns()` 新增 akshare 回退；新增模块级 `_history()` 函数 |

## 网络适配

akshare 不同市场走不同的上游数据源。在国内网络环境下，某些上游可能被阻断：

| 上游 | 状态 | 用途 |
|---|---|---|
| 新浪财经 | ⚠️ 偶有超时 | A 股 / 港股行情（主源，超时时自动切腾讯） |
| 腾讯财经 | ✅ 可用 | A 股行情备选源 |
| Yahoo Finance US | ✅ 可用 | 美股行情 |
| 东方财富（datacenter） | ✅ 可用 | 基本面、财务报表 |
| 财新 / 央视新闻 | ✅ 可用 | 宏观新闻 |
| 东方财富（push2his） | ❌ 被阻断 | K线数据 — 已切换至新浪 |
| Reddit / StockTwits | ❌ 被墙 | 情绪分析社交数据 — 自动降级 |

## 使用方式

无需任何额外配置。安装后直接使用，数据自动走 akshare：

```bash
pip install -e .
tradingagents
```

如需切回 yfinance 或使用 Alpha Vantage，修改配置中的 `data_vendors` 即可。详见 `default_config.py` 和原始 [README.md](README.md) 的 Data Architecture 章节。

---

## Web 前端界面

除了命令行模式，项目还提供了 **Web 前端界面**，可在浏览器中完成分析配置、实时监控分析进度、查看历史报告。

### 启动

```bash
pip install -e .
tradingagents web                    # 默认 http://127.0.0.1:8000
tradingagents web --port 3000        # 自定义端口
```

浏览器打开 `http://127.0.0.1:8000` 即可使用。

### 功能

| 功能 | 说明 |
|---|---|
| **配置表单** | 8 步骤配置（股票代码、日期、语言、分析师、研究深度、LLM 提供商、模型选择） |
| **实时仪表盘** | 12 个 Agent 状态面板 + 12 个独立 Tab 实时展示各阶段报告 |
| **阶段提示** | 蓝色提示条标识当前分析阶段（分析师 → 研究辩论 → 交易员 → 风险管理 → 最终决策） |
| **消息日志** | 底部可折叠面板，实时展示 LLM 调用和工具调用记录 |
| **历史记录** | 分析完成后自动保存，支持随时查看和删除历史报告 |

### 架构

```
Web 前端（HTML/CSS/JS，单文件）
        │  SSE (Server-Sent Events)
FastAPI 后端（web/app.py）
        │  调用 graph.stream()
TradingAgentsGraph 核心引擎（零修改）
```

Web 模式与 CLI 模式共享同一核心引擎（`TradingAgentsGraph`），API Key 同样从 `.env` 文件读取。

### 扩展工具

Web 模式新增了 3 个 A 股专属分析工具（基于 akshare）：

| 工具 | 分配 Analyst | 说明 |
|---|---|---|
| `get_fund_flow` | Market Analyst | 同花顺资金流向：实时流入/流出/净额 |
| `get_lhb_detail` | News Analyst | 东方财富龙虎榜：席位买卖明细、上榜原因 |
| `get_institute_hold` | Fundamentals Analyst | 新浪财经机构持仓：机构数量、持股比例变化 |
