"""Tests for src/data/crypto_data.py — all tests use mocked HTTP calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.data.crypto_data import (
    SYMBOL_MAP,
    _normalize_symbol,
    get_crypto_fear_greed,
    get_crypto_fundamentals,
    get_crypto_history,
    get_crypto_on_chain,
    get_crypto_price,
)

# ---------------------------------------------------------------------------
# _normalize_symbol
# ---------------------------------------------------------------------------


def test_normalize_symbol_btc_usdt():
    """BTC/USDT should resolve to 'bitcoin'."""
    assert _normalize_symbol("BTC/USDT") == "bitcoin"


def test_normalize_symbol_eth():
    """ETH should resolve to 'ethereum'."""
    assert _normalize_symbol("ETH") == "ethereum"


def test_normalize_symbol_unknown():
    """Unknown symbol 'PEPE' should fall back to lowercase 'pepe'."""
    assert _normalize_symbol("PEPE") == "pepe"


def test_normalize_symbol_btcusdt_no_slash():
    """BTCUSDT (no slash) should resolve to 'bitcoin'."""
    assert _normalize_symbol("BTCUSDT") == "bitcoin"


def test_normalize_symbol_map_completeness():
    """All SYMBOL_MAP keys should map to non-empty strings."""
    for sym, coin_id in SYMBOL_MAP.items():
        assert coin_id, f"SYMBOL_MAP[{sym!r}] is empty"


# ---------------------------------------------------------------------------
# get_crypto_price
# ---------------------------------------------------------------------------


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_price_success(mock_get):
    """Successful price fetch returns formatted string without TOOL_ERROR."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "bitcoin": {
            "usd": 95000,
            "usd_24h_change": 2.5,
            "usd_market_cap": 1.87e12,
            "usd_24h_vol": 4.5e10,
        }
    }
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_crypto_price.invoke({"ticker": "BTC"})

    assert isinstance(result, str)
    assert "TOOL_ERROR" not in result
    assert "95,000.00" in result
    assert "+2.50%" in result


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_price_btc_usdt_format(mock_get):
    """BTC/USDT ticker format is handled correctly."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "bitcoin": {
            "usd": 95000,
            "usd_24h_change": -1.0,
            "usd_market_cap": 1.87e12,
            "usd_24h_vol": 4.5e10,
        }
    }
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_crypto_price.invoke({"ticker": "BTC/USDT"})

    assert isinstance(result, str)
    assert "TOOL_ERROR" not in result


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_price_missing_data(mock_get):
    """Empty response body returns TOOL_ERROR."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {}
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_crypto_price.invoke({"ticker": "BTC"})

    assert result.startswith("TOOL_ERROR")


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_price_tool_error(mock_get):
    """ConnectionError from requests results in TOOL_ERROR string."""
    mock_get.side_effect = ConnectionError("Network unreachable")

    result = get_crypto_price.invoke({"ticker": "BTC"})

    assert result.startswith("TOOL_ERROR")
    assert "ConnectionError" in result


# ---------------------------------------------------------------------------
# get_crypto_history
# ---------------------------------------------------------------------------


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_history_success(mock_get):
    """Successful history fetch returns formatted string without TOOL_ERROR."""
    prices = [[1700000000000 + i * 86400000, 95000 + i * 100] for i in range(30)]
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"prices": prices}
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_crypto_history.invoke({"ticker": "BTC", "days": 30})

    assert isinstance(result, str)
    assert "TOOL_ERROR" not in result
    assert "Period Return" in result
    assert "Ann. Volatility" in result


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_history_contains_recent_prices(mock_get):
    """History output includes the 'Recent 5 Prices' section."""
    prices = [[1700000000000 + i * 86400000, 95000 + i * 100] for i in range(30)]
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"prices": prices}
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_crypto_history.invoke({"ticker": "ETH"})

    assert "Recent 5 Prices" in result


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_history_insufficient_data(mock_get):
    """Single data point triggers TOOL_ERROR."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"prices": [[1700000000000, 95000]]}
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_crypto_history.invoke({"ticker": "BTC"})

    assert result.startswith("TOOL_ERROR")


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_history_tool_error(mock_get):
    """Network error on history fetch returns TOOL_ERROR."""
    mock_get.side_effect = OSError("timeout")

    result = get_crypto_history.invoke({"ticker": "BTC"})

    assert result.startswith("TOOL_ERROR")


# ---------------------------------------------------------------------------
# get_crypto_fundamentals
# ---------------------------------------------------------------------------


def _make_fundamentals_response() -> dict:
    """Build a realistic minimal CoinGecko /coins/{id} response."""
    return {
        "name": "Bitcoin",
        "symbol": "btc",
        "market_cap_rank": 1,
        "market_data": {
            "current_price": {"usd": 95000.0},
            "market_cap": {"usd": 1.87e12},
            "circulating_supply": 19_600_000.0,
            "total_supply": 21_000_000.0,
            "max_supply": 21_000_000.0,
            "ath": {"usd": 108_000.0},
            "ath_change_percentage": {"usd": -12.0},
        },
        "developer_data": {
            "commit_count_4_weeks": 42,
            "stars": 70_000,
            "forks": 35_000,
            "code_additions_deletions_4_weeks": {"additions": 1200, "deletions": 300},
        },
        "community_data": {
            "reddit_subscribers": 5_000_000,
            "twitter_followers": 6_000_000,
            "reddit_accounts_active_48h": 12_000,
        },
        "links": {
            "blockchain_site": [
                "https://blockchair.com/bitcoin",
                "https://blockchain.com",
                "",
            ]
        },
    }


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_fundamentals_returns_string(mock_get):
    """Successful fundamentals fetch returns a non-empty string without TOOL_ERROR."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_fundamentals_response()
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_crypto_fundamentals.invoke({"ticker": "BTC"})

    assert isinstance(result, str)
    assert "TOOL_ERROR" not in result
    assert "Bitcoin" in result


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_fundamentals_contains_key_sections(mock_get):
    """Fundamentals output contains supply, developer, and community sections."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_fundamentals_response()
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_crypto_fundamentals.invoke({"ticker": "BTC"})

    assert "Supply" in result
    assert "Developer" in result
    assert "Community" in result


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_fundamentals_length_limit(mock_get):
    """Output is at most 2500 characters."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_fundamentals_response()
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_crypto_fundamentals.invoke({"ticker": "BTC"})

    assert len(result) <= 2500


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_fundamentals_tool_error(mock_get):
    """Network error on fundamentals fetch returns TOOL_ERROR."""
    mock_get.side_effect = RuntimeError("api down")

    result = get_crypto_fundamentals.invoke({"ticker": "BTC"})

    assert result.startswith("TOOL_ERROR")


# ---------------------------------------------------------------------------
# get_crypto_fear_greed
# ---------------------------------------------------------------------------


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_fear_greed_success(mock_get):
    """Successful fear/greed fetch returns formatted string without TOOL_ERROR."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {
                "value": "45",
                "value_classification": "Fear",
                "timestamp": "1700000000",
            }
        ]
    }
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_crypto_fear_greed.invoke({"days": 1})

    assert isinstance(result, str)
    assert "TOOL_ERROR" not in result
    assert "45" in result
    assert "Fear" in result


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_fear_greed_interpretation_extreme_greed(mock_get):
    """Value of 85 should produce 'Extreme Greed' interpretation."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {
                "value": "85",
                "value_classification": "Extreme Greed",
                "timestamp": "1700000000",
            }
        ]
    }
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_crypto_fear_greed.invoke({"days": 1})

    assert "Extreme Greed" in result


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_fear_greed_empty_data(mock_get):
    """Empty data list returns TOOL_ERROR."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": []}
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_crypto_fear_greed.invoke({"days": 1})

    assert result.startswith("TOOL_ERROR")


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_fear_greed_tool_error(mock_get):
    """Network error returns TOOL_ERROR."""
    mock_get.side_effect = ConnectionError("no connection")

    result = get_crypto_fear_greed.invoke({"days": 1})

    assert result.startswith("TOOL_ERROR")


# ---------------------------------------------------------------------------
# get_crypto_on_chain
# ---------------------------------------------------------------------------


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_on_chain_returns_string(mock_get):
    """Successful on-chain fetch returns a non-empty string without TOOL_ERROR."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_fundamentals_response()
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_crypto_on_chain.invoke({"ticker": "BTC"})

    assert isinstance(result, str)
    assert "TOOL_ERROR" not in result
    assert "Bitcoin" in result


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_on_chain_contains_disclaimer(mock_get):
    """On-chain output includes the premium provider disclaimer."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_fundamentals_response()
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_crypto_on_chain.invoke({"ticker": "BTC"})

    assert "premium" in result.lower() or "NOTE" in result


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_on_chain_length_limit(mock_get):
    """Output is at most 2500 characters."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_fundamentals_response()
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_crypto_on_chain.invoke({"ticker": "BTC"})

    assert len(result) <= 2500


@patch("src.data.crypto_data.requests.get")
def test_get_crypto_on_chain_tool_error(mock_get):
    """Network error returns TOOL_ERROR."""
    mock_get.side_effect = ConnectionError("timeout")

    result = get_crypto_on_chain.invoke({"ticker": "BTC"})

    assert result.startswith("TOOL_ERROR")


# ---------------------------------------------------------------------------
# LangChain tool attribute checks
# ---------------------------------------------------------------------------


def test_all_tools_are_langchain_tools():
    """All 5 exported functions must have a .name attribute (LangChain @tool marker)."""
    tools = [
        get_crypto_price,
        get_crypto_history,
        get_crypto_fundamentals,
        get_crypto_fear_greed,
        get_crypto_on_chain,
    ]
    for t in tools:
        assert hasattr(t, "name"), f"{t} is missing .name — not a LangChain tool"


def test_tool_names_are_correct():
    """Tool names match expected snake_case function names."""
    expected = [
        (get_crypto_price, "get_crypto_price"),
        (get_crypto_history, "get_crypto_history"),
        (get_crypto_fundamentals, "get_crypto_fundamentals"),
        (get_crypto_fear_greed, "get_crypto_fear_greed"),
        (get_crypto_on_chain, "get_crypto_on_chain"),
    ]
    for tool_fn, name in expected:
        assert tool_fn.name == name
