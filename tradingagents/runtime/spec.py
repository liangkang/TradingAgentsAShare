"""Canonical validation and configuration for an analysis run."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass

from tradingagents.dataflows.symbol_utils import (
    contains_chinese,
    crypto_base,
    is_yahoo_safe,
    normalize_symbol,
    resolve_a_share_name,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV
from tradingagents.llm_clients.provider_catalog import provider_default_url

ANALYST_ORDER = ("market", "social", "news", "fundamentals")
ASSET_TYPES = {"stock", "crypto"}


def _detect_asset_type(ticker: str) -> str:
    return "crypto" if crypto_base(ticker) is not None else "stock"


@dataclass(frozen=True)
class AnalysisSpec:
    """Validated, normalized inputs shared by every user interface."""

    ticker: str
    analysis_date: str
    analysts: tuple[str, ...]
    research_depth: int
    llm_provider: str
    quick_think_llm: str
    deep_think_llm: str
    output_language: str = "English"
    asset_type: str = "stock"
    backend_url: str | None = None
    google_thinking_level: str | None = None
    openai_reasoning_effort: str | None = None
    anthropic_effort: str | None = None

    @classmethod
    def create(
        cls,
        *,
        ticker: str,
        analysis_date: str,
        analysts: Iterable[str],
        research_depth: int,
        llm_provider: str,
        quick_think_llm: str,
        deep_think_llm: str,
        output_language: str = "English",
        asset_type: str | None = None,
        backend_url: str | None = None,
        google_thinking_level: str | None = None,
        openai_reasoning_effort: str | None = None,
        anthropic_effort: str | None = None,
    ) -> AnalysisSpec:
        raw_ticker = (ticker or "SPY").strip()
        if len(raw_ticker) > 32:
            raise ValueError(
                "invalid ticker; use a symbol or exact Chinese A-share name"
            )
        if contains_chinese(raw_ticker):
            canonical_ticker = resolve_a_share_name(raw_ticker)
        else:
            if not is_yahoo_safe(raw_ticker):
                raise ValueError(
                    "invalid ticker; use a symbol such as AAPL, 0700.HK, "
                    "BTC-USD, or an exact Chinese A-share name"
                )
            canonical_ticker = normalize_symbol(raw_ticker)

        try:
            parsed_date = dt.datetime.strptime(analysis_date, "%Y-%m-%d").date()
        except (TypeError, ValueError) as exc:
            raise ValueError("analysis_date must use YYYY-MM-DD") from exc
        if parsed_date > dt.datetime.now().date():
            raise ValueError("analysis_date cannot be in the future")

        requested = {str(value).strip().lower() for value in analysts if str(value).strip()}
        invalid = sorted(requested - set(ANALYST_ORDER))
        if invalid:
            raise ValueError(
                f"unknown analyst(s): {', '.join(invalid)}; "
                f"choose from {', '.join(ANALYST_ORDER)}"
            )
        ordered_analysts = tuple(key for key in ANALYST_ORDER if key in requested)
        if not ordered_analysts:
            raise ValueError("at least one analyst must be selected")

        inferred_asset_type = _detect_asset_type(canonical_ticker)
        normalized_asset_type = (asset_type or inferred_asset_type).strip().lower()
        if normalized_asset_type not in ASSET_TYPES:
            raise ValueError("asset_type must be 'stock' or 'crypto'")
        if normalized_asset_type != inferred_asset_type:
            raise ValueError(
                f"asset_type '{normalized_asset_type}' does not match ticker "
                f"'{canonical_ticker}' ({inferred_asset_type})"
            )
        if normalized_asset_type == "crypto" and "fundamentals" in ordered_analysts:
            raise ValueError("fundamentals is not available for crypto assets")

        try:
            depth = int(research_depth)
        except (TypeError, ValueError) as exc:
            raise ValueError("research_depth must be an integer from 1 to 5") from exc
        if not 1 <= depth <= 5:
            raise ValueError("research_depth must be between 1 and 5")

        provider = llm_provider.strip().lower()
        if provider not in PROVIDER_API_KEY_ENV:
            raise ValueError(f"unknown LLM provider: {provider}")
        if not quick_think_llm or not deep_think_llm:
            raise ValueError("quick_think_llm and deep_think_llm are required")
        resolved_backend_url = (
            backend_url.strip()
            if backend_url and backend_url.strip()
            else provider_default_url(provider)
        )
        if provider == "openai_compatible" and not resolved_backend_url:
            raise ValueError("backend_url is required for openai_compatible")

        return cls(
            ticker=canonical_ticker,
            analysis_date=parsed_date.isoformat(),
            analysts=ordered_analysts,
            research_depth=depth,
            llm_provider=provider,
            quick_think_llm=quick_think_llm.strip(),
            deep_think_llm=deep_think_llm.strip(),
            output_language=(output_language or "English").strip(),
            asset_type=normalized_asset_type,
            backend_url=resolved_backend_url,
            google_thinking_level=google_thinking_level,
            openai_reasoning_effort=openai_reasoning_effort,
            anthropic_effort=anthropic_effort,
        )


def build_run_config(
    spec: AnalysisSpec,
    *,
    checkpoint: bool | None = None,
    preserve_env_rounds: bool = False,
    base_config: dict | None = None,
) -> dict:
    """Build the graph config from one validated spec.

    ``DEFAULT_CONFIG`` already contains environment overrides. Interactive CLI
    runs preserve explicit debate/risk environment variables for backwards
    compatibility; Web requests treat the submitted depth as explicit.
    """
    return build_run_config_values(
        research_depth=spec.research_depth,
        quick_think_llm=spec.quick_think_llm,
        deep_think_llm=spec.deep_think_llm,
        backend_url=spec.backend_url,
        llm_provider=spec.llm_provider,
        output_language=spec.output_language,
        google_thinking_level=spec.google_thinking_level,
        openai_reasoning_effort=spec.openai_reasoning_effort,
        anthropic_effort=spec.anthropic_effort,
        checkpoint=checkpoint,
        preserve_env_rounds=preserve_env_rounds,
        base_config=base_config,
    )


def build_run_config_values(
    *,
    research_depth: int,
    quick_think_llm: str,
    deep_think_llm: str,
    backend_url: str | None,
    llm_provider: str,
    output_language: str,
    google_thinking_level: str | None = None,
    openai_reasoning_effort: str | None = None,
    anthropic_effort: str | None = None,
    checkpoint: bool | None = None,
    preserve_env_rounds: bool = False,
    base_config: dict | None = None,
) -> dict:
    """Value-based adapter used before a full ``AnalysisSpec`` is available."""
    config = deepcopy(DEFAULT_CONFIG if base_config is None else base_config)
    if not preserve_env_rounds or not os.environ.get("TRADINGAGENTS_MAX_DEBATE_ROUNDS"):
        config["max_debate_rounds"] = research_depth
    if not preserve_env_rounds or not os.environ.get("TRADINGAGENTS_MAX_RISK_ROUNDS"):
        config["max_risk_discuss_rounds"] = research_depth
    config.update(
        {
            "quick_think_llm": quick_think_llm,
            "deep_think_llm": deep_think_llm,
            "backend_url": backend_url,
            "llm_provider": llm_provider,
            "google_thinking_level": google_thinking_level,
            "openai_reasoning_effort": openai_reasoning_effort,
            "anthropic_effort": anthropic_effort,
            "output_language": output_language,
        }
    )
    if checkpoint is not None:
        config["checkpoint_enabled"] = checkpoint
    return config
