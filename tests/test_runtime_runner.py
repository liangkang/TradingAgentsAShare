"""The shared runner drives the graph's own run lifecycle for every UI."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def _fake_graph(steps, checkpoint_tid=None):
    """A graph exposing the public run lifecycle; ``steps`` are stream_run's (messages, state) pairs."""
    graph = SimpleNamespace()
    graph.ticker = None
    graph.curr_state = None
    graph.create_run_state = MagicMock(return_value={"initial": True})
    graph.begin_checkpoint = MagicMock(return_value=checkpoint_tid)
    graph.checkpoint_input = MagicMock(side_effect=lambda state: state)
    graph.stream_run = MagicMock(return_value=iter(steps))
    graph.record_decision = MagicMock()
    graph.clear_checkpoint_on_success = MagicMock()
    graph.end_checkpoint = MagicMock()
    graph.propagator = SimpleNamespace(
        get_graph_args=MagicMock(return_value={"config": {}}),
    )
    return graph


def _runner(graph, **config):
    return AnalysisRunner(
        _spec(),
        {"checkpoint_enabled": False, **config},
        graph_factory=lambda *args, **kwargs: graph,
    )


@pytest.mark.unit
def test_runner_streams_steps_and_records_the_decision():
    final = {"market_report": "market", "final_trade_decision": "Rating: Buy", "final_rating": "Buy"}
    message = SimpleNamespace(id="m1", content="hi")
    graph = _fake_graph([
        ([message], None),                       # an analyst step with no report yet
        ([message], {"market_report": "market"}),  # that analyst's report, filed
        ([], final),                             # the run's final state
    ])
    runner = _runner(graph)

    chunks = list(runner.stream())

    assert chunks == [
        {"messages": [message]},
        {"market_report": "market", "messages": [message]},
        {**final, "messages": []},
    ]
    assert runner.final_state == final
    graph.create_run_state.assert_called_once_with("AAPL", "2026-07-20", "stock")
    graph.stream_run.assert_called_once_with({"initial": True}, config={})
    graph.record_decision.assert_called_once_with("AAPL", "2026-07-20", final)
    graph.clear_checkpoint_on_success.assert_called_once_with("AAPL", "2026-07-20", "stock")
    graph.end_checkpoint.assert_called_once_with()
    assert graph.curr_state == final
    assert runner.signal == "Buy"


@pytest.mark.unit
def test_runner_cancellation_does_not_persist_decision():
    graph = _fake_graph([([], {"final_trade_decision": "Rating: Buy"})])
    runner = _runner(graph)
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(AnalysisCancelled):
        list(runner.stream(cancel_event=cancelled))

    graph.record_decision.assert_not_called()
    graph.clear_checkpoint_on_success.assert_not_called()
    graph.end_checkpoint.assert_called_once_with()  # the plain graph is restored


@pytest.mark.unit
def test_runner_injects_the_checkpoint_thread_and_restores_the_graph(tmp_path):
    graph = _fake_graph([([], {"final_trade_decision": "Rating: Hold"})], checkpoint_tid="thread-1")
    runner = _runner(graph, checkpoint_enabled=True, data_cache_dir=str(tmp_path))

    list(runner.stream())

    graph.begin_checkpoint.assert_called_once_with("AAPL", "2026-07-20", "stock")
    args = graph.propagator.get_graph_args.return_value
    assert args["config"]["configurable"]["thread_id"] == "thread-1"
    graph.clear_checkpoint_on_success.assert_called_once()
    graph.end_checkpoint.assert_called_once_with()


@pytest.mark.unit
def test_a_failed_run_restores_the_graph_and_keeps_partial_state():
    def boom():
        yield [], {"market_report": "market"}
        raise RuntimeError("model unavailable")

    graph = _fake_graph([])
    graph.stream_run = MagicMock(return_value=boom())
    runner = _runner(graph)

    with pytest.raises(RuntimeError, match="model unavailable"):
        list(runner.stream())

    graph.record_decision.assert_not_called()
    graph.end_checkpoint.assert_called_once_with()
    assert graph.curr_state == {"market_report": "market"}
