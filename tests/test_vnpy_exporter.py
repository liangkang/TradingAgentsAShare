import json
from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.integrations.vnpy.exporter import (
    build_vnpy_signal,
    export_vnpy_signal,
    resolve_vnpy_instrument,
)
from tradingagents.integrations.vnpy.signal import VnpyAction, VnpySignal


def _final_state(rating: str = "Buy") -> dict:
    return {
        "final_trade_decision": "\n".join(
            [
                f"**Rating**: {rating}",
                "",
                "**Executive Summary**: Enter gradually.",
                "",
                "**Investment Thesis**: Earnings and momentum support the position.",
                "",
                "**Price Target**: 210.5",
                "",
                "**Time Horizon**: 3-6 months",
            ]
        )
    }


@pytest.mark.unit
def test_build_vnpy_signal_maps_a_share_rating_and_fields():
    generated_at = datetime(2026, 8, 6, 9, 0, tzinfo=timezone.utc)

    signal = build_vnpy_signal(
        _final_state(), "600519.SS", ttl_hours=12, generated_at=generated_at
    )

    assert signal.symbol == "600519"
    assert signal.exchange == "SSE"
    assert signal.action is VnpyAction.BUY
    assert signal.target_position_pct == 0.10
    assert signal.price_target == 210.5
    assert signal.time_horizon == "3-6 months"
    assert signal.thesis == "Earnings and momentum support the position."
    assert signal.valid_until == generated_at + timedelta(hours=12)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rating", "action", "position"),
    [
        ("Overweight", VnpyAction.BUY, 0.05),
        ("Hold", VnpyAction.HOLD, 0.0),
        ("Underweight", VnpyAction.SELL, 0.0),
        ("Sell", VnpyAction.SELL, 0.0),
    ],
)
def test_build_vnpy_signal_maps_all_remaining_ratings(rating, action, position):
    signal = build_vnpy_signal(_final_state(rating), "000001.SZ")

    assert signal.action is action
    assert signal.target_position_pct == position
    assert signal.exchange == "SZSE"


@pytest.mark.unit
def test_export_vnpy_signal_writes_latest_and_history_atomically(tmp_path):
    latest_path = export_vnpy_signal(_final_state(), "600519.SS", tmp_path)

    assert latest_path == tmp_path / "600519.SS.json"
    latest = VnpySignal.model_validate_json(latest_path.read_text(encoding="utf-8"))
    history_files = list((tmp_path / "history").glob("600519.SS.*.json"))
    assert len(history_files) == 1
    assert json.loads(history_files[0].read_text(encoding="utf-8"))["signal_id"] == latest.signal_id


@pytest.mark.unit
def test_unknown_exchange_requires_explicit_override():
    with pytest.raises(ValueError, match="cannot infer"):
        resolve_vnpy_instrument("AAPL")

    assert resolve_vnpy_instrument("AAPL", "NASDAQ") == ("AAPL", "NASDAQ")
