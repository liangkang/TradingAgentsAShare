"""Project LangGraph state chunks into transport-neutral UI events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tradingagents.graph.analyst_execution import (
    ANALYST_NODE_SPECS,
    AnalystWallTimeTracker,
    sync_analyst_tracker_from_chunk,
)


@dataclass(frozen=True)
class RuntimeEvent:
    event: str
    data: dict[str, Any]


REPORT_TITLES = {
    "market_report": "Market Analysis",
    "sentiment_report": "Social Sentiment",
    "news_report": "News Analysis",
    "fundamentals_report": "Fundamentals Analysis",
    "investment_plan": "Research Team Decision",
    "trader_investment_plan": "Trading Team Plan",
    "final_trade_decision": "Portfolio Management Decision",
}

FIXED_AGENTS = (
    "Bull Researcher",
    "Bear Researcher",
    "Research Manager",
    "Trader",
    "Aggressive Analyst",
    "Neutral Analyst",
    "Conservative Analyst",
    "Portfolio Manager",
)


def _message_content(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    parts.append(text)
        return " ".join(parts)
    return "" if content is None else str(content).strip()


def _classify_message(message: Any) -> tuple[str | None, str | None]:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    text = _message_content(message)
    if not text:
        return None, None
    if isinstance(message, HumanMessage):
        return ("Control" if text == "Continue" else "User"), text
    if isinstance(message, ToolMessage):
        return "Data", text
    if isinstance(message, AIMessage):
        return "Agent", text
    return "System", text


class AnalysisEventProjector:
    """Stateful chunk-to-event projection shared by CLI and Web."""

    def __init__(
        self,
        selected_analysts: tuple[str, ...] | list[str],
        wall_tracker: AnalystWallTimeTracker,
    ):
        self.selected_analysts = tuple(selected_analysts)
        self.wall_tracker = wall_tracker
        self.processed_message_ids: set[str] = set()
        self.current_phase = "analysts"
        self.reports: dict[str, str] = {}
        self.agent_status: dict[str, str] = {
            ANALYST_NODE_SPECS[key].agent_node: "pending"
            for key in self.selected_analysts
        }
        self.agent_status.update(dict.fromkeys(FIXED_AGENTS, "pending"))
        first = ANALYST_NODE_SPECS[self.selected_analysts[0]].agent_node
        self.agent_status[first] = "in_progress"

    def initial_events(self) -> list[RuntimeEvent]:
        return [
            RuntimeEvent("agent_status", {"agent": agent, "status": status})
            for agent, status in self.agent_status.items()
        ]

    def _set_status(
        self,
        events: list[RuntimeEvent],
        agent: str,
        status: str,
    ) -> None:
        if self.agent_status.get(agent) == status:
            return
        self.agent_status[agent] = status
        events.append(RuntimeEvent("agent_status", {"agent": agent, "status": status}))

    def _report(
        self,
        events: list[RuntimeEvent],
        section: str,
        title: str,
        content: str,
    ) -> None:
        if not content or self.reports.get(section) == content:
            return
        self.reports[section] = content
        events.append(
            RuntimeEvent(
                "report_update",
                {"section": section, "title": title, "content": content},
            )
        )

    def _phase(
        self,
        events: list[RuntimeEvent],
        phase: str,
        label: str,
        message: str,
    ) -> None:
        if self.current_phase == phase:
            return
        self.current_phase = phase
        events.append(
            RuntimeEvent(
                "phase",
                {"phase": phase, "label": label, "message": message},
            )
        )

    def process_chunk(self, chunk: dict[str, Any]) -> list[RuntimeEvent]:
        events: list[RuntimeEvent] = []

        for message in chunk.get("messages", []):
            message_id = getattr(message, "id", None)
            if message_id is not None:
                if message_id in self.processed_message_ids:
                    continue
                self.processed_message_ids.add(message_id)
            message_type, content = _classify_message(message)
            if content:
                events.append(
                    RuntimeEvent(
                        "message",
                        {"type": message_type, "content": content},
                    )
                )
            for tool_call in getattr(message, "tool_calls", None) or []:
                data = (
                    tool_call
                    if isinstance(tool_call, dict)
                    else {"name": tool_call.name, "args": tool_call.args}
                )
                events.append(
                    RuntimeEvent(
                        "tool_call",
                        {
                            "name": data.get("name", "unknown"),
                            "args": data.get("args", {}),
                        },
                    )
                )

        sync_analyst_tracker_from_chunk(self.wall_tracker, chunk)
        active_found = False
        for key in self.selected_analysts:
            spec = ANALYST_NODE_SPECS[key]
            content = chunk.get(spec.report_key, "")
            if content:
                self._set_status(events, spec.agent_node, "completed")
                self._report(
                    events,
                    spec.report_key,
                    REPORT_TITLES[spec.report_key],
                    content,
                )
            elif not active_found:
                self._set_status(events, spec.agent_node, "in_progress")
                active_found = True

        analysts_complete = all(
            self.agent_status.get(ANALYST_NODE_SPECS[key].agent_node) == "completed"
            for key in self.selected_analysts
        )
        if analysts_complete and self.agent_status.get("Bull Researcher") == "pending":
            self._phase(
                events,
                "research",
                "研究团队辩论中",
                f"{len(self.selected_analysts)} 位分析师已完成报告，研究团队开始辩论。",
            )
            self._set_status(events, "Bull Researcher", "in_progress")

        debate = chunk.get("investment_debate_state") or {}
        bull = str(debate.get("bull_history", "")).strip()
        bear = str(debate.get("bear_history", "")).strip()
        research_judge = str(debate.get("judge_decision", "")).strip()
        if bull or bear or research_judge:
            self._phase(
                events,
                "research",
                "研究团队辩论中",
                "Bull 与 Bear 研究员正在辩论并由研究经理裁决。",
            )
            for agent in ("Bull Researcher", "Bear Researcher", "Research Manager"):
                if self.agent_status.get(agent) == "pending":
                    self._set_status(events, agent, "in_progress")
        if bull:
            self._report(events, "research_bull", "看多研究员", bull)
        if bear:
            self._report(events, "research_bear", "看空研究员", bear)
        if research_judge:
            self._report(events, "research_manager", "研究经理裁决", research_judge)
            for agent in ("Bull Researcher", "Bear Researcher", "Research Manager"):
                self._set_status(events, agent, "completed")
            self._set_status(events, "Trader", "in_progress")
        research_parts = []
        if bull:
            research_parts.append(f"### Bull Researcher Analysis\n{bull}")
        if bear:
            research_parts.append(f"### Bear Researcher Analysis\n{bear}")
        if research_judge:
            research_parts.append(f"### Research Manager Decision\n{research_judge}")
        if research_parts:
            self._report(
                events,
                "investment_plan",
                REPORT_TITLES["investment_plan"],
                "\n\n".join(research_parts),
            )

        trader_plan = str(chunk.get("trader_investment_plan", "")).strip()
        if trader_plan:
            self._phase(
                events,
                "trader",
                "交易员制定计划中",
                "Trader 正在根据研究裁决制定交易方案。",
            )
            self._report(
                events,
                "trader_investment_plan",
                REPORT_TITLES["trader_investment_plan"],
                trader_plan,
            )
            self._set_status(events, "Trader", "completed")
            self._set_status(events, "Aggressive Analyst", "in_progress")

        risk = chunk.get("risk_debate_state") or {}
        aggressive = str(risk.get("aggressive_history", "")).strip()
        conservative = str(risk.get("conservative_history", "")).strip()
        neutral = str(risk.get("neutral_history", "")).strip()
        risk_judge = str(risk.get("judge_decision", "")).strip()
        if aggressive or conservative or neutral:
            self._phase(
                events,
                "risk",
                "风险管理辩论中",
                "激进、保守和中立三方正在评估交易风险。",
            )
        risk_reports = (
            ("Aggressive Analyst", "risk_aggressive", "激进风险分析师", aggressive),
            ("Conservative Analyst", "risk_conservative", "保守风险分析师", conservative),
            ("Neutral Analyst", "risk_neutral", "中立风险分析师", neutral),
        )
        for agent, section, title, content in risk_reports:
            if content:
                self._set_status(events, agent, "in_progress")
                self._report(events, section, title, content)
        if risk_judge:
            self._phase(
                events,
                "portfolio",
                "最终决策中",
                "Portfolio Manager 正在综合风险观点做出最终决策。",
            )
            self._report(events, "risk_portfolio", "投资组合经理最终决策", risk_judge)
            for agent in (
                "Aggressive Analyst",
                "Conservative Analyst",
                "Neutral Analyst",
                "Portfolio Manager",
            ):
                self._set_status(events, agent, "completed")

        # Previously final_trade_decision concatenated all risk-debate sections.
        # Web "最终决策" now shows only the Portfolio Manager decision; the
        # individual risk analyst tabs still receive their own report_update events.
        # risk_parts = []
        # if aggressive:
        #     risk_parts.append(f"### Aggressive Analyst Analysis\n{aggressive}")
        # if conservative:
        #     risk_parts.append(f"### Conservative Analyst Analysis\n{conservative}")
        # if neutral:
        #     risk_parts.append(f"### Neutral Analyst Analysis\n{neutral}")
        # if risk_judge:
        #     risk_parts.append(f"### Portfolio Manager Decision\n{risk_judge}")
        # if risk_parts:
        #     self._report(
        #         events,
        #         "final_trade_decision",
        #         REPORT_TITLES["final_trade_decision"],
        #         "\n\n".join(risk_parts),
        #     )
        if risk_judge:
            self._report(
                events,
                "final_trade_decision",
                REPORT_TITLES["final_trade_decision"],
                risk_judge,
            )

        return events
