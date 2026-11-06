"""Per-analyst wall time for the runtime's UIs (Web, and the shared runner's callers).

Mirrors the tracker the CLI keeps in ``cli.display``, which the library cannot
import, plus ``get_wall_times`` for the Web's completion event. The analysts run
together, so each starts with the run and stops its clock when its report lands.
"""

from __future__ import annotations

from time import monotonic

from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS, AnalystExecutionPlan


class AnalystWallTimeTracker:
    def __init__(self, plan: AnalystExecutionPlan):
        self.plan = plan
        self._started_at: dict[str, float] = {}
        self._wall_times: dict[str, float] = {}

    def mark_started(self, analyst_key: str, started_at: float | None = None) -> None:
        if analyst_key not in ANALYST_NODE_SPECS:
            raise ValueError(f"unknown analyst key: {analyst_key}")
        self._started_at.setdefault(analyst_key, monotonic() if started_at is None else started_at)

    def mark_completed(self, analyst_key: str, completed_at: float | None = None) -> None:
        if analyst_key not in ANALYST_NODE_SPECS:
            raise ValueError(f"unknown analyst key: {analyst_key}")
        if analyst_key in self._wall_times:
            return
        started_at = self._started_at.get(analyst_key)
        if started_at is None:
            return
        finished_at = monotonic() if completed_at is None else completed_at
        self._wall_times[analyst_key] = max(0.0, finished_at - started_at)

    def get_wall_times(self) -> dict[str, float]:
        return dict(self._wall_times)

    def format_summary(self) -> str:
        parts = []
        for spec in self.plan.specs:
            duration = self._wall_times.get(spec.key)
            if duration is not None:
                label = spec.agent_node.removesuffix(" Analyst")
                parts.append(f"{label} {duration:.2f}s")
        if not parts:
            return "Analyst wall time: pending"
        return "Analyst wall time: " + " | ".join(parts)


def sync_analyst_tracker_from_chunk(
    tracker: AnalystWallTimeTracker,
    chunk: dict,
    now: float | None = None,
) -> None:
    """The analysts start together; each stops its clock when its report lands."""
    current_time = monotonic() if now is None else now
    for spec in tracker.plan.specs:
        tracker.mark_started(spec.key, started_at=current_time)
        if chunk.get(spec.report_key):
            tracker.mark_completed(spec.key, completed_at=current_time)
