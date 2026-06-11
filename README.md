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
