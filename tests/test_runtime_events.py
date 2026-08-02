"""Shared chunk projection produces the same UI state for every transport."""

from __future__ import annotations

import pytest

from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
)
from tradingagents.runtime import AnalysisEventProjector


@pytest.mark.unit
def test_projector_builds_complete_reports_and_statuses():
    analysts = ("market",)
    tracker = AnalystWallTimeTracker(build_analyst_execution_plan(analysts))
    tracker.mark_started("market")
    projector = AnalysisEventProjector(analysts, tracker)

    events = projector.process_chunk(
        {
            "market_report": "Market",
            "investment_debate_state": {
                "bull_history": "Bull",
                "bear_history": "Bear",
                "judge_decision": "Research decision",
            },
            "trader_investment_plan": "Trader plan",
            "risk_debate_state": {
                "aggressive_history": "Aggressive",
                "conservative_history": "Conservative",
                "neutral_history": "Neutral",
                "judge_decision": "Rating: Hold",
            },
        }
    )

    assert projector.reports["market_report"] == "Market"
    assert "Bull" in projector.reports["investment_plan"]
    assert projector.reports["final_trade_decision"] == "Rating: Hold"
    assert "Aggressive" not in projector.reports["final_trade_decision"]
    assert projector.reports["risk_aggressive"] == "Aggressive"
    assert all(status == "completed" for status in projector.agent_status.values())
    assert {event.event for event in events} >= {
        "agent_status",
        "phase",
        "report_update",
    }


@pytest.mark.unit
def test_projector_deduplicates_unchanged_reports():
    analysts = ("market",)
    tracker = AnalystWallTimeTracker(build_analyst_execution_plan(analysts))
    projector = AnalysisEventProjector(analysts, tracker)

    first = projector.process_chunk({"market_report": "Market"})
    second = projector.process_chunk({"market_report": "Market"})

    assert any(event.event == "report_update" for event in first)
    assert not any(event.event == "report_update" for event in second)
