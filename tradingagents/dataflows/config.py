from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

import tradingagents.default_config as default_config

# Use default config but allow it to be overridden
_config: dict | None = None
_run_config: ContextVar[dict | None] = ContextVar(
    "tradingagents_run_config",
    default=None,
)


def initialize_config():
    """Initialize the configuration with default values."""
    global _config
    if _config is None:
        _config = deepcopy(default_config.DEFAULT_CONFIG)


def set_config(config: dict):
    """Update the configuration with custom values.

    Dict-valued keys (e.g. ``data_vendors``) are merged one level deep so a
    partial update like ``{"data_vendors": {"core_stock_apis": "alpha_vantage"}}``
    keeps the other nested keys from the default; scalar keys are replaced.
    """
    global _config
    initialize_config()
    incoming = deepcopy(config)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(_config.get(key), dict):
            _config[key].update(value)
        else:
            _config[key] = value


@contextmanager
def config_context(config: dict) -> Iterator[None]:
    """Temporarily bind a complete dataflow config to the current run.

    Web analyses execute concurrently in separate threads/tasks. Mutating the
    legacy module-level config from ``TradingAgentsGraph.__init__`` allowed one
    request to change another request's vendor and output-language settings.
    A ContextVar keeps the active run isolated while retaining ``set_config``
    as the process-wide configuration API used by existing callers and tests.
    """
    initialize_config()
    scoped = deepcopy(_config)
    incoming = deepcopy(config)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(scoped.get(key), dict):
            scoped[key].update(value)
        else:
            scoped[key] = value

    token = _run_config.set(scoped)
    try:
        yield
    finally:
        _run_config.reset(token)


def get_config() -> dict:
    """Get the current configuration."""
    scoped = _run_config.get()
    if scoped is not None:
        return deepcopy(scoped)
    if _config is None:
        initialize_config()
    return deepcopy(_config)


# Initialize with default config
initialize_config()
