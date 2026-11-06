"""Portable signal exports consumable by vn.py strategies.

This package deliberately has no vn.py dependency.  It defines the file
contract on the TradingAgents side; a vn.py strategy can consume the JSON in
its own environment.
"""

from .exporter import export_vnpy_signal
from .signal import VnpySignal

__all__ = ["VnpySignal", "export_vnpy_signal"]
