"""
FastAPI application for TradingAgents Web Interface.

Provides a REST API and SSE streaming endpoint that wraps the existing
TradingAgentsGraph engine — no core code changes needed.
"""

import asyncio
import json
import os
import shutil
import threading
import time
import uuid
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
    get_initial_analyst_node,
    sync_analyst_tracker_from_chunk,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS, get_model_options
from tradingagents.llm_clients.api_key_env import get_api_key_env

# ---- FastAPI app -----------------------------------------------------------

app = FastAPI(
    title="TradingAgents Web",
    description="Multi-Agents LLM Financial Trading Framework — Web Interface",
    version="0.2.5",
)

TEMPLATES_DIR = Path(__file__).parent / "templates"


# ---- Pydantic models -------------------------------------------------------

class AnalysisRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g. NVDA, 0700.HK")
    analysis_date: str = Field(..., description="Analysis date YYYY-MM-DD")
    asset_type: str = Field(default="stock", description="Asset type: stock or crypto")
    analysts: List[str] = Field(
        default=["market", "social", "news", "fundamentals"],
        description="Selected analyst types",
    )
    research_depth: int = Field(default=1, ge=1, le=5, description="Research depth 1-5")
    llm_provider: str = Field(default="openai", description="LLM provider key")
    backend_url: Optional[str] = Field(default=None, description="LLM backend URL override")
    quick_think_llm: str = Field(default="gpt-5.4-mini", description="Quick-thinking model")
    deep_think_llm: str = Field(default="gpt-5.5", description="Deep-thinking model")
    output_language: str = Field(default="Simplified Chinese (简体中文)", description="Report output language")
    google_thinking_level: Optional[str] = Field(default=None)
    openai_reasoning_effort: Optional[str] = Field(default=None)
    anthropic_effort: Optional[str] = Field(default=None)


# ---- HTML page --------------------------------------------------------------

_INDEX_HTML: Optional[str] = None


def _load_index_html() -> str:
    """Lazy-load the index.html template from disk."""
    global _INDEX_HTML
    if _INDEX_HTML is None:
        path = TEMPLATES_DIR / "index.html"
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        _INDEX_HTML = path.read_text(encoding="utf-8")
    return _INDEX_HTML


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page web frontend."""
    return HTMLResponse(
        content=_load_index_html(),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ---- Config endpoint -------------------------------------------------------

@app.get("/api/config/defaults")
async def get_defaults():
    """Return available configuration options for the frontend form."""
    # LLM providers (from the same catalog the CLI uses)
    providers = [
        {"display": "OpenAI", "key": "openai", "default_url": "https://api.openai.com/v1"},
        {"display": "Google", "key": "google", "default_url": None},
        {"display": "Anthropic", "key": "anthropic", "default_url": "https://api.anthropic.com/"},
        {"display": "xAI", "key": "xai", "default_url": "https://api.x.ai/v1"},
        {"display": "DeepSeek", "key": "deepseek", "default_url": "https://api.deepseek.com"},
        {"display": "Qwen", "key": "qwen", "default_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"},
        {"display": "GLM", "key": "glm", "default_url": "https://open.bigmodel.cn/api/paas/v4/"},
        {"display": "MiniMax", "key": "minimax", "default_url": "https://api.minimax.io/v1"},
        {"display": "OpenRouter", "key": "openrouter", "default_url": "https://openrouter.ai/api/v1"},
        {"display": "Azure OpenAI", "key": "azure", "default_url": None},
        {"display": "Ollama", "key": "ollama", "default_url": os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"},
    ]

    # Model options per provider (quick / deep)
    model_options: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    for provider_key, modes in MODEL_OPTIONS.items():
        model_options[provider_key] = {}
        for mode, options in modes.items():
            model_options[provider_key][mode] = [
                {"label": label, "value": value} for label, value in options
            ]

    # Language options
    languages = [
        "English", "Simplified Chinese (简体中文)", "Traditional Chinese (繁體中文)",
        "Japanese (日本語)", "Korean (한국어)", "Spanish (Español)",
        "French (Français)", "German (Deutsch)", "Italian (Italiano)",
        "Portuguese (Português)", "Arabic (العربية)", "Other (Custom)",
    ]

    return {
        "providers": providers,
        "model_options": model_options,
        "languages": languages,
        "research_depths": [
            {"label": "浅度（1轮，快速）", "value": 1},
            {"label": "中度（3轮）", "value": 3},
            {"label": "深度（5轮，详尽）", "value": 5},
        ],
        "analysts": [
            {"key": "market", "label": "市场分析师", "description": "技术与市场数据分析"},
            {"key": "social", "label": "情绪分析师", "description": "社交媒体与情绪分析"},
            {"key": "news", "label": "新闻分析师", "description": "新闻与宏观事件分析"},
            {"key": "fundamentals", "label": "基本面分析师", "description": "财务报表与基本面分析"},
        ],
        "defaults": {
            "ticker": "SPY",
            "analysis_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "llm_provider": DEFAULT_CONFIG.get("llm_provider", "openai"),
            "quick_think_llm": DEFAULT_CONFIG.get("quick_think_llm", "gpt-5.4-mini"),
            "deep_think_llm": DEFAULT_CONFIG.get("deep_think_llm", "gpt-5.5"),
            "output_language": "Simplified Chinese (简体中文)",
        },
    }


# ---- SSE helpers -----------------------------------------------------------

def _sse_event(event: str, data: Any) -> str:
    """Format a Server-Sent Event line."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


# Agent / team constants (mirrors CLI's MessageBuffer)
FIXED_AGENTS = {
    "Analyst Team": [],
    "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
    "Trading Team": ["Trader"],
    "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
    "Portfolio Management": ["Portfolio Manager"],
}

ANALYST_AGENT_MAP = {
    "market": "Market Analyst",
    "social": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}

ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}

REPORT_TITLES = {
    "market_report": "Market Analysis",
    "sentiment_report": "Social Sentiment",
    "news_report": "News Analysis",
    "fundamentals_report": "Fundamentals Analysis",
    "investment_plan": "Research Team Decision",
    "trader_investment_plan": "Trading Team Plan",
    "final_trade_decision": "Portfolio Management Decision",
}


def _classify_message(message) -> tuple:
    """Classify LangChain message into (type, content)."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = getattr(message, "content", None)
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text", "").strip()
                if t:
                    text_parts.append(t)
        text = " ".join(text_parts)
    elif content is not None:
        text = str(content).strip()
    else:
        text = ""

    if not text:
        return None, None

    if isinstance(message, HumanMessage):
        if text == "Continue":
            return "Control", text
        return "User", text
    if isinstance(message, ToolMessage):
        return "Data", text
    if isinstance(message, AIMessage):
        return "Agent", text
    return "System", text


# ---- API key validation ----------------------------------------------------

_PROVIDERS_WITHOUT_KEY = {"ollama"}


def _check_api_key(provider: str) -> None:
    """Validate that the API key is available for *provider*.

    Raises HTTPException(400) with a clear message when the key is missing,
    so the frontend can show the error before streaming begins.
    """
    if provider in _PROVIDERS_WITHOUT_KEY:
        return  # no key needed

    env_var = get_api_key_env(provider)
    if not env_var:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider '{provider}' — cannot determine API key environment variable.",
        )

    key_value = os.environ.get(env_var, "").strip()
    if not key_value:
        raise HTTPException(
            status_code=400,
            detail=(
                f"API Key not found for provider '{provider}'.\n"
                f"Please set the environment variable {env_var} in your .env file, "
                f"or run 'tradingagents analyze' once to configure it interactively."
            ),
        )


# ---- Analysis SSE endpoint -------------------------------------------------

@app.post("/api/analyze")
async def analyze(req: AnalysisRequest, request: Request):
    """Run analysis and stream results via Server-Sent Events.

    Uses a background thread + asyncio.Queue pattern so the blocking
    LangGraph stream() call runs in its own OS thread while the async
    generator can properly integrate with FastAPI's event loop.
    """

    # --- Pre-flight validation (before streaming) ---
    provider = req.llm_provider.lower()
    _check_api_key(provider)

    analyst_order = ["market", "social", "news", "fundamentals"]
    selected_keys = [a for a in analyst_order if a in req.analysts]
    if not selected_keys:
        selected_keys = ["market"]

    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = req.research_depth
    config["max_risk_discuss_rounds"] = req.research_depth
    config["quick_think_llm"] = req.quick_think_llm
    config["deep_think_llm"] = req.deep_think_llm
    config["backend_url"] = req.backend_url
    config["llm_provider"] = provider
    config["google_thinking_level"] = req.google_thinking_level
    config["openai_reasoning_effort"] = req.openai_reasoning_effort
    config["anthropic_effort"] = req.anthropic_effort
    config["output_language"] = req.output_language

    analyst_plan = build_analyst_execution_plan(
        selected_keys, concurrency_limit=config.get("analyst_concurrency_limit", 1)
    )

    # Build initial agent status
    initial_agent_status: Dict[str, str] = {}
    for key in selected_keys:
        initial_agent_status[ANALYST_AGENT_MAP[key]] = "pending"
    for team_agents in FIXED_AGENTS.values():
        for agent in team_agents:
            initial_agent_status[agent] = "pending"
    first_agent = get_initial_analyst_node(analyst_plan)
    initial_agent_status[first_agent] = "in_progress"

    # Queue for communication between background thread and async generator
    queue: asyncio.Queue = asyncio.Queue()

    def run_analysis_in_thread():
        """Runs the blocking graph.stream() in a background thread.

        Puts SSE event strings into the async queue.  A sentinel None
        marks completion; an Exception instance signals an error.
        """
        analysis_id = uuid.uuid4().hex[:12]
        collected_reports: Dict[str, str] = {}  # section_key → latest content
        final_stats: Dict[str, Any] = {}
        try:
            # --- Send initial events ---
            queue.put_nowait(_sse_event("connected", {
                "message": f"Analysis started for {req.ticker} on {req.analysis_date}",
                "ticker": req.ticker,
                "analysts": selected_keys,
            }))
            for agent, status in initial_agent_status.items():
                queue.put_nowait(_sse_event("agent_status", {"agent": agent, "status": status}))
            queue.put_nowait(_sse_event("stats", {
                "llm_calls": 0, "tool_calls": 0, "tokens_in": 0, "tokens_out": 0,
            }))

            # --- Initialize graph ---
            queue.put_nowait(_sse_event("log", {
                "level": "info",
                "message": f"正在初始化 LLM 客户端 (provider={config['llm_provider']}, deep={config['deep_think_llm']}, quick={config['quick_think_llm']})..."
            }))
            from cli.stats_handler import StatsCallbackHandler
            stats_handler = StatsCallbackHandler()
            try:
                graph = TradingAgentsGraph(
                    selected_keys,
                    config=config,
                    debug=True,
                    callbacks=[stats_handler],
                )
            except Exception as init_err:
                queue.put_nowait(_sse_event("log", {
                    "level": "error",
                    "message": f"LLM 客户端初始化失败：{init_err}"
                }))
                raise
            queue.put_nowait(_sse_event("log", {
                "level": "info",
                "message": "LLM 客户端初始化完成，正在构建分析图..."
            }))

            wall_tracker = AnalystWallTimeTracker(analyst_plan)
            wall_tracker.mark_started(selected_keys[0])
            agent_status = dict(initial_agent_status)

            # resolve_instrument_context calls yfinance internally, which can
            # hang for minutes on networks that block Yahoo Finance (common in
            # mainland China).  For A-share / HK tickers we skip the call
            # entirely since akshare provides better identity data anyway.
            # For US tickers we keep it but with a 15 s timeout.
            ticker_upper = req.ticker.upper()
            is_cn_ticker = any(
                ticker_upper.endswith(suffix) for suffix in (".SS", ".SZ", ".HK")
            )
            if is_cn_ticker:
                queue.put_nowait(_sse_event("log", {
                    "level": "info",
                    "message": "A股/港股标的，跳过 yfinance 身份解析，直接使用 akshare 数据..."
                }))
                instrument_context = ""
            else:
                from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
                queue.put_nowait(_sse_event("log", {
                    "level": "info",
                    "message": "正在解析股票身份信息（yfinance 查询，最多等待 15 秒）..."
                }))
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            graph.resolve_instrument_context, req.ticker, req.asset_type
                        )
                        instrument_context = future.result(timeout=15)
                    queue.put_nowait(_sse_event("log", {
                        "level": "info",
                        "message": "股票身份信息解析完成。"
                    }))
                except FuturesTimeoutError:
                    queue.put_nowait(_sse_event("log", {
                        "level": "warn",
                        "message": "yfinance 查询超时（15 秒），跳过身份解析，继续分析..."
                    }))
                    instrument_context = ""
                except Exception as ctx_err:
                    queue.put_nowait(_sse_event("log", {
                        "level": "warn",
                        "message": f"股票身份解析失败：{ctx_err}，继续分析..."
                    }))
                    instrument_context = ""

            init_state = graph.propagator.create_initial_state(
                req.ticker, req.analysis_date,
                asset_type=req.asset_type,
                instrument_context=instrument_context,
            )
            args = graph.propagator.get_graph_args(callbacks=[stats_handler])

            queue.put_nowait(_sse_event("log", {
                "level": "info",
                "message": f"开始流式分析 {req.ticker}... 等待第一个 LLM 响应（可能需要 30-60 秒）..."
            }))

            start_time = time.time()
            processed_msg_ids: set = set()
            chunk_count = 0
            current_phase = "initializing"

            # --- Stream chunks from LangGraph ---
            for chunk in graph.graph.stream(init_state, **args):
                chunk_count += 1

                # --- Phase detection ---
                # Detect when all selected analyst reports are complete
                all_analysts_done = (
                    current_phase == "initializing"
                    and all(
                        chunk.get(ANALYST_REPORT_MAP.get(k, ""))
                        for k in selected_keys
                    )
                )
                if all_analysts_done:
                    current_phase = "analysts_done"
                    queue.put_nowait(_sse_event("phase", {
                        "phase": "research",
                        "label": "研究团队辩论中",
                        "message": "4 位分析师已完成报告，Bull Researcher 正在阅读并撰写看多论证（深度思考，需要 60-120 秒）..."
                    }))
                    queue.put_nowait(_sse_event("log", {
                        "level": "info",
                        "message": "分析师阶段完成，进入研究团队辩论。Bull Researcher 正在处理全部报告..."
                    }))

                if chunk.get("investment_debate_state") and current_phase in ("analysts_done", "research"):
                    debate = chunk["investment_debate_state"]
                    if debate.get("bull_history") or debate.get("bear_history"):
                        if current_phase != "research":
                            current_phase = "research"
                            queue.put_nowait(_sse_event("phase", {
                                "phase": "research",
                                "label": "研究团队辩论中",
                                "message": "Bull 与 Bear 研究员正在辩论..."
                            }))
                if chunk.get("trader_investment_plan") and current_phase != "trader":
                    current_phase = "trader"
                    queue.put_nowait(_sse_event("phase", {
                        "phase": "trader",
                        "label": "交易员制定计划中",
                        "message": "Trader 正在根据研究裁决制定交易方案..."
                    }))
                if chunk.get("risk_debate_state") and current_phase not in ("risk", "portfolio"):
                    risk = chunk["risk_debate_state"]
                    if risk.get("aggressive_history") or risk.get("conservative_history") or risk.get("neutral_history"):
                        current_phase = "risk"
                        queue.put_nowait(_sse_event("phase", {
                            "phase": "risk",
                            "label": "风险管理辩论中",
                            "message": "激进/保守/中立三方正在就交易方案展开辩论..."
                        }))
                if chunk.get("risk_debate_state") and current_phase == "risk":
                    risk = chunk["risk_debate_state"]
                    if risk.get("judge_decision") and current_phase != "portfolio":
                        current_phase = "portfolio"
                        queue.put_nowait(_sse_event("phase", {
                            "phase": "portfolio",
                            "label": "最终决策中",
                            "message": "Portfolio Manager 正在综合三方观点做出最终决策..."
                        }))

                queue.put_nowait(_sse_event("log", {
                    "level": "debug",
                    "message": f"收到第 {chunk_count} 个 chunk (keys: {list(chunk.keys())})"
                }))

                # --- Messages ---
                for message in chunk.get("messages", []):
                    msg_id = getattr(message, "id", None)
                    if msg_id is not None:
                        if msg_id in processed_msg_ids:
                            continue
                        processed_msg_ids.add(msg_id)

                    msg_type, content = _classify_message(message)
                    if content:
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        queue.put_nowait(_sse_event("message", {
                            "type": msg_type,
                            "content": content[:500],
                            "timestamp": ts,
                        }))

                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tc in message.tool_calls:
                            tc_data = tc if isinstance(tc, dict) else {"name": tc.name, "args": tc.args}
                            ts = datetime.datetime.now().strftime("%H:%M:%S")
                            queue.put_nowait(_sse_event("tool_call", {
                                "name": tc_data.get("name", "unknown"),
                                "args": str(tc_data.get("args", {}))[:200],
                                "timestamp": ts,
                            }))

                # --- Analyst statuses ---
                sync_analyst_tracker_from_chunk(wall_tracker, chunk)
                found_active = False
                for key in analyst_order:
                    if key not in selected_keys:
                        continue
                    agent_name = ANALYST_AGENT_MAP[key]
                    report_key = ANALYST_REPORT_MAP[key]

                    if chunk.get(report_key):
                        if agent_status.get(agent_name) != "completed":
                            agent_status[agent_name] = "completed"
                            queue.put_nowait(_sse_event("agent_status", {
                                "agent": agent_name, "status": "completed",
                            }))
                        collected_reports[report_key] = chunk[report_key]
                        queue.put_nowait(_sse_event("report_update", {
                            "section": report_key,
                            "title": REPORT_TITLES.get(report_key, report_key),
                            "content": chunk[report_key],
                        }))
                    elif not found_active and agent_status.get(agent_name) != "completed":
                        if agent_status.get(agent_name) != "in_progress":
                            agent_status[agent_name] = "in_progress"
                            queue.put_nowait(_sse_event("agent_status", {
                                "agent": agent_name, "status": "in_progress",
                            }))
                        found_active = True

                if not found_active and selected_keys:
                    if agent_status.get("Bull Researcher") == "pending":
                        agent_status["Bull Researcher"] = "in_progress"
                        queue.put_nowait(_sse_event("agent_status", {
                            "agent": "Bull Researcher", "status": "in_progress",
                        }))

                # --- Research Team ---
                debate = chunk.get("investment_debate_state")
                if debate:
                    bull = debate.get("bull_history", "").strip()
                    bear = debate.get("bear_history", "").strip()
                    judge = debate.get("judge_decision", "").strip()

                    if bull or bear:
                        for agent in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
                            if agent_status.get(agent) == "pending":
                                agent_status[agent] = "in_progress"
                                queue.put_nowait(_sse_event("agent_status", {
                                    "agent": agent, "status": "in_progress",
                                }))
                    if bull:
                        collected_reports["research_bull"] = bull
                        queue.put_nowait(_sse_event("report_update", {
                            "section": "research_bull",
                            "title": "看多研究员",
                            "content": bull,
                        }))
                    if bear:
                        collected_reports["research_bear"] = bear
                        queue.put_nowait(_sse_event("report_update", {
                            "section": "research_bear",
                            "title": "看空研究员",
                            "content": bear,
                        }))
                    if judge:
                        collected_reports["research_manager"] = judge
                        queue.put_nowait(_sse_event("report_update", {
                            "section": "research_manager",
                            "title": "研究经理裁决",
                            "content": judge,
                        }))
                        for agent in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
                            agent_status[agent] = "completed"
                            queue.put_nowait(_sse_event("agent_status", {"agent": agent, "status": "completed"}))
                        if agent_status.get("Trader") == "pending":
                            agent_status["Trader"] = "in_progress"
                            queue.put_nowait(_sse_event("agent_status", {"agent": "Trader", "status": "in_progress"}))

                    # Combined backward-compat section
                    combined_parts = []
                    if bull: combined_parts.append(f"### 看多研究员\n{bull}")
                    if bear: combined_parts.append(f"### 看空研究员\n{bear}")
                    if judge: combined_parts.append(f"### 研究经理裁决\n{judge}")
                    if combined_parts:
                        collected_reports["investment_plan"] = "\n\n".join(combined_parts)
                        queue.put_nowait(_sse_event("report_update", {
                            "section": "investment_plan",
                            "title": "研究团队决策",
                            "content": "\n\n".join(combined_parts),
                        }))

                # --- Trader ---
                if chunk.get("trader_investment_plan"):
                    collected_reports["trader_investment_plan"] = chunk["trader_investment_plan"]
                    queue.put_nowait(_sse_event("report_update", {
                        "section": "trader_investment_plan",
                        "title": "Trading Team Plan",
                        "content": chunk["trader_investment_plan"],
                    }))
                    if agent_status.get("Trader") != "completed":
                        agent_status["Trader"] = "completed"
                        agent_status["Aggressive Analyst"] = "in_progress"
                        queue.put_nowait(_sse_event("agent_status", {"agent": "Trader", "status": "completed"}))
                        queue.put_nowait(_sse_event("agent_status", {"agent": "Aggressive Analyst", "status": "in_progress"}))

                # --- Risk Management ---
                risk = chunk.get("risk_debate_state")
                if risk:
                    agg = risk.get("aggressive_history", "").strip()
                    con = risk.get("conservative_history", "").strip()
                    neu = risk.get("neutral_history", "").strip()
                    rjudge = risk.get("judge_decision", "").strip()

                    if agg:
                        if agent_status.get("Aggressive Analyst") != "completed":
                            agent_status["Aggressive Analyst"] = "in_progress"
                            queue.put_nowait(_sse_event("agent_status", {"agent": "Aggressive Analyst", "status": "in_progress"}))
                        collected_reports["risk_aggressive"] = agg
                        queue.put_nowait(_sse_event("report_update", {
                            "section": "risk_aggressive",
                            "title": "激进风险分析师",
                            "content": agg,
                        }))
                    if con:
                        if agent_status.get("Conservative Analyst") != "completed":
                            agent_status["Conservative Analyst"] = "in_progress"
                            queue.put_nowait(_sse_event("agent_status", {"agent": "Conservative Analyst", "status": "in_progress"}))
                        collected_reports["risk_conservative"] = con
                        queue.put_nowait(_sse_event("report_update", {
                            "section": "risk_conservative",
                            "title": "保守风险分析师",
                            "content": con,
                        }))
                    if neu:
                        if agent_status.get("Neutral Analyst") != "completed":
                            agent_status["Neutral Analyst"] = "in_progress"
                            queue.put_nowait(_sse_event("agent_status", {"agent": "Neutral Analyst", "status": "in_progress"}))
                        collected_reports["risk_neutral"] = neu
                        queue.put_nowait(_sse_event("report_update", {
                            "section": "risk_neutral",
                            "title": "中立风险分析师",
                            "content": neu,
                        }))
                    if rjudge:
                        agent_status["Portfolio Manager"] = "in_progress"
                        queue.put_nowait(_sse_event("agent_status", {"agent": "Portfolio Manager", "status": "in_progress"}))
                        collected_reports["risk_portfolio"] = rjudge
                        queue.put_nowait(_sse_event("report_update", {
                            "section": "risk_portfolio",
                            "title": "投资组合经理最终决策",
                            "content": rjudge,
                        }))
                        for a in ["Aggressive Analyst", "Conservative Analyst", "Neutral Analyst", "Portfolio Manager"]:
                            agent_status[a] = "completed"
                            queue.put_nowait(_sse_event("agent_status", {"agent": a, "status": "completed"}))

                    # Also keep the combined final_trade_decision for backward compat
                    combined_parts = []
                    if agg: combined_parts.append(f"### 激进风险分析师\n{agg}")
                    if con: combined_parts.append(f"### 保守风险分析师\n{con}")
                    if neu: combined_parts.append(f"### 中立风险分析师\n{neu}")
                    if rjudge: combined_parts.append(f"### 投资组合经理最终决策\n{rjudge}")
                    if combined_parts:
                        collected_reports["final_trade_decision"] = "\n\n".join(combined_parts)
                        queue.put_nowait(_sse_event("report_update", {
                            "section": "final_trade_decision",
                            "title": "风险管理与投资组合决策",
                            "content": "\n\n".join(combined_parts),
                        }))

                # --- Stats ---
                final_stats = stats_handler.get_stats()
                queue.put_nowait(_sse_event("stats", {
                    "llm_calls": final_stats["llm_calls"],
                    "tool_calls": final_stats["tool_calls"],
                    "tokens_in": final_stats["tokens_in"],
                    "tokens_out": final_stats["tokens_out"],
                }))

            # --- Complete & save ---
            elapsed = time.time() - start_time
            # Persist to disk so the user can review later
            try:
                _save_analysis_to_disk(
                    analysis_id=analysis_id,
                    ticker=req.ticker,
                    analysis_date=req.analysis_date,
                    reports=collected_reports,
                    stats=final_stats,
                    elapsed=elapsed,
                )
            except Exception as save_err:
                queue.put_nowait(_sse_event("log", {
                    "level": "warn",
                    "message": f"保存分析结果失败：{save_err}"
                }))

            queue.put_nowait(_sse_event("log", {
                "level": "info",
                "message": f"分析完成！共收到 {chunk_count} 个 chunks，耗时 {elapsed:.1f}s"
            }))
            queue.put_nowait(_sse_event("complete", {
                "analysis_id": analysis_id,
                "message": f"Analysis complete in {int(elapsed // 60):02d}:{int(elapsed % 60):02d}",
                "elapsed_seconds": round(elapsed, 1),
                "analyst_wall_times": wall_tracker.get_wall_times(),
                "wall_time_summary": wall_tracker.format_summary(),
            }))

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            queue.put_nowait(_sse_event("log", {
                "level": "error",
                "message": f"分析出错：{e}"
            }))
            queue.put_nowait(_sse_event("error", {
                "message": str(e),
                "traceback": tb,
            }))
        finally:
            # Signal the async generator that we're done
            queue.put_nowait(None)

    # Start the analysis in a background thread
    thread = threading.Thread(target=run_analysis_in_thread, daemon=True)
    thread.start()

    async def event_generator():
        """Async generator that reads SSE events from the queue."""
        while True:
            # Wait for the next event with a short timeout so we can check
            # for client disconnection periodically
            try:
                event_str = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Send a heartbeat comment to keep the connection alive
                yield ": heartbeat\n\n"
                # Check if client disconnected
                if hasattr(request, "is_disconnected"):
                    try:
                        if await request.is_disconnected():
                            break
                    except Exception:
                        pass
                continue

            if event_str is None:
                # Sentinel: analysis thread finished
                break

            yield event_str

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---- Saved reports directory ------------------------------------------------

_SAVED_DIR = Path(DEFAULT_CONFIG["results_dir"]) / "_saved_reports"
_SAVED_DIR.mkdir(parents=True, exist_ok=True)


def _save_analysis_to_disk(
    analysis_id: str,
    ticker: str,
    analysis_date: str,
    reports: Dict[str, str],
    stats: Dict[str, Any],
    elapsed: float,
) -> Path:
    """Persist a completed analysis as JSON so it can be reviewed later."""
    data = {
        "id": analysis_id,
        "ticker": ticker,
        "analysis_date": analysis_date,
        "saved_at": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "reports": reports,
        "stats": stats,
    }
    path = _SAVED_DIR / f"{analysis_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---- History API ------------------------------------------------------------

@app.get("/api/history")
async def list_saved():
    """List all saved analysis reports, newest first."""
    items = []
    for f in sorted(_SAVED_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            items.append({
                "id": data["id"],
                "ticker": data["ticker"],
                "analysis_date": data.get("analysis_date", ""),
                "saved_at": data.get("saved_at", ""),
                "elapsed_seconds": data.get("elapsed_seconds", 0),
                "report_count": len(data.get("reports", {})),
            })
        except Exception:
            pass
    return {"items": items}


@app.get("/api/history/{analysis_id}")
async def get_saved(analysis_id: str):
    """Retrieve a single saved analysis with full report data."""
    path = _SAVED_DIR / f"{analysis_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.delete("/api/history/{analysis_id}")
async def delete_saved(analysis_id: str):
    """Delete a saved analysis."""
    path = _SAVED_DIR / f"{analysis_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")
    path.unlink()
    return {"status": "ok"}


# ---- Save report endpoint (legacy) ------------------------------------------

class SaveReportRequest(BaseModel):
    ticker: str
    report_content: str
    format: str = "md"


@app.post("/api/save-report")
async def save_report(req: SaveReportRequest):
    """Save the final report to disk as Markdown."""
    from tradingagents.dataflows.utils import safe_ticker_component
    safe_ticker = safe_ticker_component(req.ticker)
    results_dir = Path(DEFAULT_CONFIG["results_dir"]) / safe_ticker
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.{req.format}"
    filepath = results_dir / filename
    filepath.write_text(req.report_content, encoding="utf-8")
    return {"status": "ok", "path": str(filepath.resolve()), "filename": filename}
