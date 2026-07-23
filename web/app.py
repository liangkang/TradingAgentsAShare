"""
FastAPI application for TradingAgents Web Interface.

Provides a REST API and SSE streaming endpoint over the shared AnalysisRunner
used by both the CLI and Web transports.
"""

import asyncio
import datetime
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
)
from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV, get_api_key_env
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS
from tradingagents.llm_clients.provider_catalog import get_provider_options
from tradingagents.runtime import (
    AnalysisCancelled,
    AnalysisEventProjector,
    AnalysisRunner,
    AnalysisSpec,
    StatsCallbackHandler,
    build_run_config,
)

# ---- FastAPI app -----------------------------------------------------------

app = FastAPI(
    title="TradingAgents Web",
    description="Multi-Agents LLM Financial Trading Framework — Web Interface",
    version="0.3.1",
)

TEMPLATES_DIR = Path(__file__).parent / "templates"


# ---- Pydantic models -------------------------------------------------------

class AnalysisRequest(BaseModel):
    ticker: str = Field(
        ...,
        description="Ticker symbol or exact Chinese A-share name, e.g. NVDA, 0700.HK, 长飞光纤",
    )
    analysis_date: str = Field(..., description="Analysis date YYYY-MM-DD")
    asset_type: str | None = Field(default=None, description="Asset type: stock or crypto")
    analysts: list[str] = Field(
        default=["market", "social", "news", "fundamentals"],
        description="Selected analyst types",
    )
    research_depth: int = Field(default=3, ge=1, le=5, description="Research depth 1-5")
    llm_provider: str = Field(default="openai", description="LLM provider key")
    backend_url: str | None = Field(default=None, description="LLM backend URL override")
    quick_think_llm: str = Field(default="gpt-5.4-mini", description="Quick-thinking model")
    deep_think_llm: str = Field(default="gpt-5.5", description="Deep-thinking model")
    output_language: str = Field(default="Simplified Chinese (简体中文)", description="Report output language")
    google_thinking_level: str | None = Field(default=None)
    openai_reasoning_effort: str | None = Field(default=None)
    anthropic_effort: str | None = Field(default=None)


# ---- HTML page --------------------------------------------------------------

_INDEX_HTML: str | None = None


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
        {
            "display": option.display,
            "key": option.key,
            "default_url": option.default_url,
        }
        for option in get_provider_options(include_regions=True)
    ]

    # Model options per provider (quick / deep)
    model_options: dict[str, dict[str, list[dict[str, str]]]] = {}
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
            "research_depth": 3,
            # Web preselects DeepSeek; TRADINGAGENTS_* env vars still win so a
            # server-side .env can pin a different provider/model pair.
            "llm_provider": os.getenv("TRADINGAGENTS_LLM_PROVIDER") or "deepseek",
            "quick_think_llm": (
                os.getenv("TRADINGAGENTS_QUICK_THINK_LLM") or "deepseek-v4-flash"
            ),
            "deep_think_llm": (
                os.getenv("TRADINGAGENTS_DEEP_THINK_LLM") or "deepseek-v4-pro"
            ),
            "output_language": "Simplified Chinese (简体中文)",
        },
    }


# ---- SSE helpers -----------------------------------------------------------

def _sse_event(event: str, data: Any) -> str:
    """Format a Server-Sent Event line."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


class _ThreadsafeAsyncQueue:
    """Allow a worker thread to publish into an event-loop-owned queue."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self._loop = loop
        self._queue = queue

    def put_nowait(self, item: Any) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)


# ---- API key validation ----------------------------------------------------

_PROVIDERS_WITHOUT_KEY = {"ollama", "bedrock", "openai_compatible"}


def _check_api_key(provider: str) -> None:
    """Validate that the API key is available for *provider*.

    Raises HTTPException(400) with a clear message when the key is missing,
    so the frontend can show the error before streaming begins.
    """
    if provider not in PROVIDER_API_KEY_ENV:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{provider}'.")
    if provider in _PROVIDERS_WITHOUT_KEY:
        return

    env_var = get_api_key_env(provider)
    if not env_var:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{provider}' has no configured credential source.",
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
    try:
        spec = AnalysisSpec.create(
            ticker=req.ticker,
            analysis_date=req.analysis_date,
            asset_type=req.asset_type,
            analysts=req.analysts,
            research_depth=req.research_depth,
            llm_provider=req.llm_provider,
            backend_url=req.backend_url,
            quick_think_llm=req.quick_think_llm,
            deep_think_llm=req.deep_think_llm,
            output_language=req.output_language,
            google_thinking_level=req.google_thinking_level,
            openai_reasoning_effort=req.openai_reasoning_effort,
            anthropic_effort=req.anthropic_effort,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _check_api_key(spec.llm_provider)
    config = build_run_config(spec)
    analyst_plan = build_analyst_execution_plan(spec.analysts)

    # The asyncio queue belongs to this request's event loop. Worker threads
    # publish through a thread-safe proxy rather than touching it directly.
    event_queue: asyncio.Queue = asyncio.Queue()
    queue = _ThreadsafeAsyncQueue(asyncio.get_running_loop(), event_queue)
    cancel_event = threading.Event()

    def run_analysis_in_thread():
        """Runs the blocking graph.stream() in a background thread.

        Puts SSE event strings into the async queue.  A sentinel None
        marks completion; an Exception instance signals an error.
        """
        analysis_id = uuid.uuid4().hex[:12]
        final_stats: dict[str, Any] = {}
        try:
            # --- Send initial events ---
            queue.put_nowait(_sse_event("connected", {
                "message": f"Analysis started for {spec.ticker} on {spec.analysis_date}",
                "ticker": spec.ticker,
                "analysts": spec.analysts,
            }))
            queue.put_nowait(_sse_event("stats", {
                "llm_calls": 0, "tool_calls": 0, "tokens_in": 0, "tokens_out": 0,
            }))

            # --- Initialize graph ---
            queue.put_nowait(_sse_event("log", {
                "level": "info",
                "message": f"正在初始化 LLM 客户端 (provider={config['llm_provider']}, deep={config['deep_think_llm']}, quick={config['quick_think_llm']})..."
            }))
            stats_handler = StatsCallbackHandler()
            try:
                runner = AnalysisRunner(
                    spec,
                    config,
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
            wall_tracker.mark_started(spec.analysts[0])
            projector = AnalysisEventProjector(spec.analysts, wall_tracker)
            for event in projector.initial_events():
                queue.put_nowait(_sse_event(event.event, event.data))

            queue.put_nowait(_sse_event("log", {
                "level": "info",
                "message": f"开始流式分析 {spec.ticker}... 等待第一个 LLM 响应（可能需要 30-60 秒）..."
            }))

            start_time = time.time()
            chunk_count = 0

            # --- Stream chunks from LangGraph ---
            for chunk in runner.stream(cancel_event=cancel_event):
                chunk_count += 1
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                for event in projector.process_chunk(chunk):
                    data = dict(event.data)
                    if event.event in {"message", "tool_call"}:
                        data["timestamp"] = timestamp
                    if event.event == "message":
                        data["content"] = data["content"][:500]
                    elif event.event == "tool_call":
                        data["args"] = str(data["args"])[:200]
                    queue.put_nowait(_sse_event(event.event, data))

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
                    ticker=spec.ticker,
                    analysis_date=spec.analysis_date,
                    reports=projector.reports,
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

        except AnalysisCancelled:
            queue.put_nowait(_sse_event("cancelled", {
                "message": f"Analysis cancelled for {spec.ticker}",
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
        try:
            while True:
                # Wait for the next event with a short timeout so we can check
                # for client disconnection periodically.
                try:
                    event_str = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    try:
                        if await request.is_disconnected():
                            break
                    except Exception:
                        pass
                    continue

                if event_str is None:
                    break
                yield event_str
        finally:
            cancel_event.set()

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
_ANALYSIS_ID = re.compile(r"^[0-9a-f]{12}$")


def _save_analysis_to_disk(
    analysis_id: str,
    ticker: str,
    analysis_date: str,
    reports: dict[str, str],
    stats: dict[str, Any],
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


def _saved_report_path(analysis_id: str) -> Path:
    if not _ANALYSIS_ID.fullmatch(analysis_id):
        raise HTTPException(status_code=404, detail="Analysis not found")
    return _SAVED_DIR / f"{analysis_id}.json"


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
    path = _saved_report_path(analysis_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.delete("/api/history/{analysis_id}")
async def delete_saved(analysis_id: str):
    """Delete a saved analysis."""
    path = _saved_report_path(analysis_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")
    path.unlink()
    return {"status": "ok"}


# ---- Save report endpoint (legacy) ------------------------------------------

class SaveReportRequest(BaseModel):
    ticker: str
    report_content: str
    format: Literal["md"] = "md"


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
