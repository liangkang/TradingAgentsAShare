"""The shared runner owns the full graph lifecycle for every UI."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.runtime import AnalysisCancelled, AnalysisRunner, AnalysisSpec


def _spec() -> AnalysisSpec:
    return AnalysisSpec.create(
        ticker="AAPL",
        analysis_date="2026-07-20",
        analysts=["market"],
        research_depth=1,
        llm_provider="openai",
        quick_think_llm="gpt-5.4-mini",
        deep_think_llm="gpt-5.5",
    )


def _fake_graph(chunks):
    graph = SimpleNamespace()
    graph.ticker = None
    graph.config = {}
    graph.curr_state = None
    graph._resolve_pending_entries = MagicMock()
    graph.resolve_instrument_context = MagicMock(return_value="IDENTITY")
    graph._log_state = MagicMock()
    graph.process_signal = MagicMock(return_value="Buy")
    graph._run_signature = MagicMock(return_value="signature")
    graph.memory_log = SimpleNamespace(
        get_past_context=MagicMock(return_value="PAST"),
        store_decision=MagicMock(),
    )
    graph.propagator = SimpleNamespace(
        create_initial_state=MagicMock(return_value={"initial": True}),
        get_graph_args=MagicMock(return_value={"stream_mode": "values", "config": {}}),
    )
    graph.graph = SimpleNamespace(stream=MagicMock(return_value=iter(chunks)))
    graph.workflow = SimpleNamespace(compile=MagicMock())
    return graph


@pytest.mark.unit
def test_runner_injects_context_and_finalizes_memory():
    final = {
        "market_report": "market",
        "final_trade_decision": "Rating: Buy",
    }
    graph = _fake_graph([{"market_report": "market"}, final])
    runner = AnalysisRunner(
        _spec(),
        {"checkpoint_enabled": False},
        graph_factory=lambda *args, **kwargs: graph,
    )

    assert list(runner.stream()) == [{"market_report": "market"}, final]
    assert runner.final_state == final
    graph._resolve_pending_entries.assert_called_once_with("AAPL")
    graph.propagator.create_initial_state.assert_called_once_with(
        "AAPL",
        "2026-07-20",
        asset_type="stock",
        past_context="PAST",
        instrument_context="IDENTITY",
    )
    graph._log_state.assert_called_once_with("2026-07-20", final)
    graph.memory_log.store_decision.assert_called_once_with(
        ticker="AAPL",
        trade_date="2026-07-20",
        final_trade_decision="Rating: Buy",
    )
    assert runner.signal == "Buy"


@pytest.mark.unit
def test_runner_cancellation_does_not_persist_decision():
    graph = _fake_graph([{"final_trade_decision": "Rating: Buy"}])
    runner = AnalysisRunner(
        _spec(),
        {"checkpoint_enabled": False},
        graph_factory=lambda *args, **kwargs: graph,
    )
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(AnalysisCancelled):
        list(runner.stream(cancel_event=cancelled))

    graph._log_state.assert_not_called()
    graph.memory_log.store_decision.assert_not_called()


@pytest.mark.unit
def test_runner_wires_and_clears_checkpoint(tmp_path):
    final = {"final_trade_decision": "Rating: Hold"}
    graph = _fake_graph([])
    checkpoint_graph = SimpleNamespace(stream=MagicMock(return_value=iter([final])))
    fresh_graph = SimpleNamespace()
    graph.workflow.compile = MagicMock(side_effect=[checkpoint_graph, fresh_graph])
    context = MagicMock()
    context.__enter__.return_value = object()
    runner = AnalysisRunner(
        _spec(),
        {
            "checkpoint_enabled": True,
            "data_cache_dir": str(tmp_path),
        },
        graph_factory=lambda *args, **kwargs: graph,
    )

    with (
        patch("tradingagents.runtime.runner.get_checkpointer", return_value=context),
        patch("tradingagents.runtime.runner.checkpoint_step", return_value=None),
        patch("tradingagents.runtime.runner.thread_id", return_value="thread-1"),
        patch("tradingagents.runtime.runner.clear_checkpoint") as clear,
    ):
        list(runner.stream())

    args = graph.propagator.get_graph_args.return_value
    assert args["config"]["configurable"]["thread_id"] == "thread-1"
    clear.assert_called_once_with(
        str(tmp_path),
        "AAPL",
        "2026-07-20",
        "signature",
    )
    context.__exit__.assert_called_once_with(None, None, None)
    assert graph.graph is fresh_graph
