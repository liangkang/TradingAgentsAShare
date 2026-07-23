"""Shared streaming runner for CLI and Web transports."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator

from tradingagents.dataflows.config import config_context
from tradingagents.graph.checkpointer import (
    checkpoint_step,
    clear_checkpoint,
    get_checkpointer,
    thread_id,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph

from .spec import AnalysisSpec

logger = logging.getLogger(__name__)


class AnalysisCancelled(RuntimeError):
    """Raised after a client requests cancellation of a streaming analysis."""


class AnalysisRunner:
    """Own one complete analysis lifecycle and expose its LangGraph chunks.

    The runner is transport-neutral: CLI renders chunks with Rich while Web
    projects them into SSE events. Checkpoints, memory context, final logging,
    and decision persistence happen here so both entry points behave the same.
    """

    def __init__(
        self,
        spec: AnalysisSpec,
        config: dict,
        *,
        callbacks: list | None = None,
        graph_factory: Callable[..., TradingAgentsGraph] = TradingAgentsGraph,
    ):
        self.spec = spec
        self.config = config
        self.callbacks = callbacks or []
        self.graph = graph_factory(
            spec.analysts,
            config=config,
            debug=True,
            callbacks=self.callbacks,
        )
        self.final_state: dict = {}
        self.signal: str | None = None

    def stream(
        self,
        *,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[dict]:
        graph = self.graph
        spec = self.spec
        graph.ticker = spec.ticker
        checkpointer_ctx = None
        completed = False

        with config_context(self.config):
            graph._resolve_pending_entries(spec.ticker)

            if self.config.get("checkpoint_enabled"):
                checkpointer_ctx = get_checkpointer(
                    self.config["data_cache_dir"],
                    spec.ticker,
                )
                saver = checkpointer_ctx.__enter__()
                graph.graph = graph.workflow.compile(checkpointer=saver)
                step = checkpoint_step(
                    self.config["data_cache_dir"],
                    spec.ticker,
                    spec.analysis_date,
                    graph._run_signature(spec.asset_type),
                )
                logger.info(
                    "Resuming from step %d for %s on %s"
                    if step is not None
                    else "Starting fresh for %s on %s",
                    *(
                        (step, spec.ticker, spec.analysis_date)
                        if step is not None
                        else (spec.ticker, spec.analysis_date)
                    ),
                )

            try:
                past_context = graph.memory_log.get_past_context(spec.ticker)
                instrument_context = graph.resolve_instrument_context(
                    spec.ticker,
                    spec.asset_type,
                )
                initial_state = graph.propagator.create_initial_state(
                    spec.ticker,
                    spec.analysis_date,
                    asset_type=spec.asset_type,
                    past_context=past_context,
                    instrument_context=instrument_context,
                )
                args = graph.propagator.get_graph_args(callbacks=self.callbacks)
                if self.config.get("checkpoint_enabled"):
                    args.setdefault("config", {}).setdefault("configurable", {})[
                        "thread_id"
                    ] = thread_id(
                        spec.ticker,
                        spec.analysis_date,
                        graph._run_signature(spec.asset_type),
                    )

                for chunk in graph.graph.stream(initial_state, **args):
                    if cancel_event is not None and cancel_event.is_set():
                        raise AnalysisCancelled(
                            f"analysis cancelled for {spec.ticker}"
                        )
                    self.final_state.update(chunk)
                    yield chunk

                if cancel_event is not None and cancel_event.is_set():
                    raise AnalysisCancelled(f"analysis cancelled for {spec.ticker}")
                if not self.final_state:
                    raise RuntimeError("analysis completed without producing state")

                graph.curr_state = self.final_state
                graph._log_state(spec.analysis_date, self.final_state)
                graph.memory_log.store_decision(
                    ticker=spec.ticker,
                    trade_date=spec.analysis_date,
                    final_trade_decision=self.final_state["final_trade_decision"],
                )
                self.signal = graph.process_signal(
                    self.final_state["final_trade_decision"]
                )
                if self.config.get("checkpoint_enabled"):
                    clear_checkpoint(
                        self.config["data_cache_dir"],
                        spec.ticker,
                        spec.analysis_date,
                        graph._run_signature(spec.asset_type),
                    )
                completed = True
            finally:
                if checkpointer_ctx is not None:
                    checkpointer_ctx.__exit__(None, None, None)
                    graph.graph = graph.workflow.compile()
                if not completed:
                    graph.curr_state = self.final_state or None
