"""CLI symbol validation/classification must agree with the data path.

Regressions for #980 (validation rejected GC=F), #981 (BTCUSD misclassified as
stock), #982 (BTC-USDT accepted but unpriceable on Yahoo).
"""
from unittest.mock import patch

import pandas as pd
import pytest

from cli.models import AssetType
from cli.utils import detect_asset_type, is_valid_ticker_input, normalize_ticker_symbol
from tradingagents.dataflows.symbol_utils import (
    AShareNameResolutionError,
    normalize_symbol,
    resolve_a_share_name,
)


# --- #982: stablecoin-quoted crypto normalizes to Yahoo's -USD pair ---
@pytest.mark.parametrize("raw,expected", [
    ("BTCUSD", "BTC-USD"),
    ("BTCUSDT", "BTC-USD"),
    ("BTC-USDT", "BTC-USD"),
    ("BTC-USDC", "BTC-USD"),
    ("ethusdt", "ETH-USD"),
    # non-crypto must be untouched
    ("AAPL", "AAPL"),
    ("GC=F", "GC=F"),
    ("600519.SS", "600519.SS"),
    ("EURUSD", "EURUSD=X"),
])
def test_normalize_symbol_crypto_and_passthrough(raw, expected):
    assert normalize_symbol(raw) == expected


# --- #980: validation accepts Yahoo futures/forex symbols ---
@pytest.mark.parametrize("value,ok", [
    ("GC=F", True),
    ("EURUSD=X", True),
    ("AAPL", True),
    ("0700.HK", True),
    ("长飞光纤", True),
    ("*ST 亚振", True),
    ("长飞/光纤", False),
    ("^GSPC", True),
    ("", True),                 # empty -> defaults to SPY downstream
    ("bad symbol!", False),     # space + '!' rejected
    ("A" * 40, False),          # too long
])
def test_ticker_input_validation(value, ok):
    assert is_valid_ticker_input(value) is ok


# --- #981/#982: asset-type classified on the canonical symbol ---
@pytest.mark.parametrize("raw,expected", [
    ("BTCUSD", AssetType.CRYPTO),
    ("BTC-USDT", AssetType.CRYPTO),
    ("BTC-USD", AssetType.CRYPTO),
    ("ETHUSD", AssetType.CRYPTO),
    ("AAPL", AssetType.STOCK),
    ("GC=F", AssetType.STOCK),
    ("600519.SS", AssetType.STOCK),
])
def test_detect_asset_type(raw, expected):
    assert detect_asset_type(raw) == expected


def test_cli_normalize_delegates_to_data_layer():
    # CLI must produce the same canonical symbol the data path will price.
    for raw in ("XAUUSD", "BTCUSD", "btc-usdt", "AAPL"):
        assert normalize_ticker_symbol(raw) == normalize_symbol(raw)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "code", "expected"),
    [
        ("长飞光纤", "601869", "601869.SH"),
        ("宁德时代", "300750", "300750.SZ"),
        ("贝特瑞", "835185", "835185.BJ"),
    ],
)
def test_resolve_chinese_a_share_name(name, code, expected):
    stock_list = pd.DataFrame({"code": [code], "name": [name]})
    with patch("akshare.stock_info_a_code_name", return_value=stock_list):
        assert resolve_a_share_name(name) == expected
        assert normalize_ticker_symbol(name) == expected


@pytest.mark.unit
def test_resolve_chinese_a_share_name_normalizes_width_and_spaces():
    stock_list = pd.DataFrame({"code": ["000002"], "name": ["万  科Ａ"]})
    with patch("akshare.stock_info_a_code_name", return_value=stock_list):
        assert resolve_a_share_name("万科A") == "000002.SZ"


@pytest.mark.unit
def test_unknown_chinese_a_share_name_is_rejected():
    stock_list = pd.DataFrame({"code": ["601869"], "name": ["长飞光纤"]})
    with (
        patch("akshare.stock_info_a_code_name", return_value=stock_list),
        pytest.raises(AShareNameResolutionError, match="no exact A-share match"),
    ):
        normalize_ticker_symbol("不存在的股票")
