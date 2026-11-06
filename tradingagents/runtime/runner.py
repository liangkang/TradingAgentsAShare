"""Shared streaming runner for CLI and Web transports."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator

from tradingagents.agents.rating import run_rating
from tradingagents.dataflows.config import run_config
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
        """Run the analysis, yielding each step as a state chunk.

        Follows the graph's own run lifecycle (``create_run_state``,
        ``begin_checkpoint``, ``stream_run``, ``record_decision``), the one
        ``propagate()`` and the upstream CLI use, so checkpoint resume, memory
        settlement and the decision log behave the same from every entry point.

        A chunk is the step's state (for a step inside an analyst's graph, that
        analyst's report) with the step's messages under ``"messages"``.
        """
        graph = self.graph
        spec = self.spec
        graph.ticker = spec.ticker
        completed = False

        # Instrument identity reads vendors through the run's config.
        with run_config(self.config):
            initial_state = graph.create_run_state(
                spec.ticker, spec.analysis_date, spec.asset_type
            )
        args = graph.propagator.get_graph_args(callbacks=self.callbacks)
        checkpoint_tid = graph.begin_checkpoint(
            spec.ticker, spec.analysis_date, spec.asset_type
        )
        if checkpoint_tid is not None:
            logger.info("Checkpointing %s on %s", spec.ticker, spec.analysis_date)
            args.setdefault("config", {}).setdefault("configurable", {})[
                "thread_id"
            ] = checkpoint_tid

        try:
            for messages, state in graph.stream_run(
                graph.checkpoint_input(initial_state), **args
            ):
                if cancel_event is not None and cancel_event.is_set():
                    raise AnalysisCancelled(f"analysis cancelled for {spec.ticker}")
                if state is not None:
                    self.final_state.update(state)
                yield {**(state or {}), "messages": messages}

            if cancel_event is not None and cancel_event.is_set():
                raise AnalysisCancelled(f"analysis cancelled for {spec.ticker}")
            if not self.final_state:
                raise RuntimeError("analysis completed without producing state")

            graph.curr_state = self.final_state
            graph.record_decision(spec.ticker, spec.analysis_date, self.final_state)
            self.signal = run_rating(self.final_state)
            graph.clear_checkpoint_on_success(
                spec.ticker, spec.analysis_date, spec.asset_type
            )
            completed = True
        finally:
            # Always restore the plain, uncheckpointed graph.
            graph.end_checkpoint()
            if not completed:
                graph.curr_state = self.final_state or None
