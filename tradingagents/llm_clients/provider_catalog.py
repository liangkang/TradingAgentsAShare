"""Canonical LLM provider presentation metadata for CLI and Web."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderOption:
    display: str
    key: str
    default_url: str | None


_PRIMARY_OPTIONS = (
    ProviderOption("OpenAI", "openai", "https://api.openai.com/v1"),
    ProviderOption("Google", "google", None),
    ProviderOption("Anthropic", "anthropic", "https://api.anthropic.com/"),
    ProviderOption("Evolink (Claude API)", "evolink", "https://direct.evolink.ai"),
    ProviderOption("xAI", "xai", "https://api.x.ai/v1"),
    ProviderOption("DeepSeek", "deepseek", "https://api.deepseek.com"),
    ProviderOption("Qwen (International)", "qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    ProviderOption("GLM (International)", "glm", "https://api.z.ai/api/paas/v4/"),
    ProviderOption("MiniMax (International)", "minimax", "https://api.minimax.io/v1"),
    ProviderOption("OpenRouter", "openrouter", "https://openrouter.ai/api/v1"),
    ProviderOption("Mistral", "mistral", "https://api.mistral.ai/v1"),
    ProviderOption("Kimi (Moonshot)", "kimi", "https://api.moonshot.ai/v1"),
    ProviderOption("Groq", "groq", "https://api.groq.com/openai/v1"),
    ProviderOption("NVIDIA NIM", "nvidia", "https://integrate.api.nvidia.com/v1"),
    ProviderOption("Azure OpenAI", "azure", None),
    ProviderOption("Amazon Bedrock", "bedrock", None),
    ProviderOption("Ollama", "ollama", "http://localhost:11434/v1"),
    ProviderOption(
        "OpenAI-compatible (vLLM, LM Studio, llama.cpp, custom relay)",
        "openai_compatible",
        None,
    ),
)

_REGIONAL_OPTIONS = (
    ProviderOption("Qwen (China)", "qwen-cn", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ProviderOption("GLM (China)", "glm-cn", "https://open.bigmodel.cn/api/paas/v4/"),
    ProviderOption("MiniMax (China)", "minimax-cn", "https://api.minimaxi.com/v1"),
)


def get_provider_options(*, include_regions: bool = False) -> list[ProviderOption]:
    options = list(_PRIMARY_OPTIONS)
    if include_regions:
        options.extend(_REGIONAL_OPTIONS)
    ollama_url = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
    return [
        ProviderOption(item.display, item.key, ollama_url)
        if item.key == "ollama"
        else item
        for item in options
    ]


def provider_default_url(provider: str) -> str | None:
    key = provider.lower()
    for option in get_provider_options(include_regions=True):
        if option.key == key:
            return option.default_url
    return None
