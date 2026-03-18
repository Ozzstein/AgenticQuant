"""Tests for new market_data and fundamental_data @tool functions added in WU-2 of TASK-04."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with *n* rows."""
    np.random.seed(42)
    prices = 150 * np.cumprod(1 + np.random.normal(0.001, 0.02, n))
    idx = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.02,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        },
        index=idx,
    )


def _make_small_ohlcv(n: int = 5) -> pd.DataFrame:
    return _make_ohlcv(n)


def _make_mock_ticker(ohlcv: pd.DataFrame, info: dict | None = None) -> MagicMock:
    mock = MagicMock()
    mock.history.return_value = ohlcv
    mock.info = info or {}
    mock.fast_info = MagicMock()
    return mock


def _make_download_response(tickers: list[str], n: int = 22) -> pd.DataFrame:
    """Return a MultiIndex DataFrame as yf.download would for multiple tickers."""
    np.random.seed(0)
    idx = pd.bdate_range("2023-01-01", periods=n)
    arrays = [
        ["Close"] * len(tickers),
        tickers,
    ]
    cols = pd.MultiIndex.from_arrays(arrays)
    data = np.random.uniform(100, 200, (n, len(tickers)))
    return pd.DataFrame(data, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# Tests: get_technical_indicators
# ---------------------------------------------------------------------------

class TestGetTechnicalIndicators:
    def test_returns_string(self):
        from src.data.market_data import get_technical_indicators

        ohlcv = _make_ohlcv(200)
        mock_ticker = _make_mock_ticker(ohlcv)

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_technical_indicators.invoke({"ticker": "AAPL"})

        assert isinstance(result, str)
        assert "RSI" in result
        assert "MACD" in result
        assert "AAPL" in result.upper()

    def test_contains_key_indicators(self):
        from src.data.market_data import get_technical_indicators

        ohlcv = _make_ohlcv(200)
        mock_ticker = _make_mock_ticker(ohlcv)

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_technical_indicators.invoke({"ticker": "MSFT"})

        assert "ATR" in result
        assert "BB" in result or "Bollinger" in result or "Upper" in result
        assert "OBV" in result

    def test_insufficient_data_returns_tool_error(self):
        from src.data.market_data import get_technical_indicators

        small_df = _make_small_ohlcv(5)
        mock_ticker = _make_mock_ticker(small_df)

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_technical_indicators.invoke({"ticker": "TINY"})

        assert result.startswith("TOOL_ERROR")
        assert "TINY" in result

    def test_empty_dataframe_returns_tool_error(self):
        from src.data.market_data import get_technical_indicators

        mock_ticker = _make_mock_ticker(pd.DataFrame())

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_technical_indicators.invoke({"ticker": "EMPTY"})

        assert result.startswith("TOOL_ERROR")

    def test_output_within_char_limit(self):
        from src.data.market_data import get_technical_indicators

        ohlcv = _make_ohlcv(200)
        mock_ticker = _make_mock_ticker(ohlcv)

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_technical_indicators.invoke({"ticker": "AAPL"})

        assert len(result) <= 2500

    def test_exception_returns_tool_error(self):
        from src.data.market_data import get_technical_indicators

        with patch("yfinance.Ticker", side_effect=RuntimeError("network failure")):
            result = get_technical_indicators.invoke({"ticker": "FAIL"})

        assert result.startswith("TOOL_ERROR")
        assert "network failure" in result


# ---------------------------------------------------------------------------
# Tests: get_support_resistance
# ---------------------------------------------------------------------------

class TestGetSupportResistance:
    def test_returns_string(self):
        from src.data.market_data import get_support_resistance

        ohlcv = _make_ohlcv(200)
        mock_ticker = _make_mock_ticker(ohlcv)

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_support_resistance.invoke({"ticker": "AAPL"})

        assert isinstance(result, str)
        assert "Pivot" in result
        assert "AAPL" in result.upper()

    def test_contains_pivot_levels(self):
        from src.data.market_data import get_support_resistance

        ohlcv = _make_ohlcv(100)
        mock_ticker = _make_mock_ticker(ohlcv)

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_support_resistance.invoke({"ticker": "SPY"})

        assert "R1" in result
        assert "R2" in result
        assert "S1" in result
        assert "S2" in result

    def test_contains_volume_levels(self):
        from src.data.market_data import get_support_resistance

        ohlcv = _make_ohlcv(100)
        mock_ticker = _make_mock_ticker(ohlcv)

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_support_resistance.invoke({"ticker": "SPY"})

        assert "Volume" in result or "volume" in result

    def test_output_within_char_limit(self):
        from src.data.market_data import get_support_resistance

        ohlcv = _make_ohlcv(200)
        mock_ticker = _make_mock_ticker(ohlcv)

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_support_resistance.invoke({"ticker": "AAPL"})

        assert len(result) <= 2500

    def test_exception_returns_tool_error(self):
        from src.data.market_data import get_support_resistance

        with patch("yfinance.Ticker", side_effect=ConnectionError("timeout")):
            result = get_support_resistance.invoke({"ticker": "FAIL"})

        assert result.startswith("TOOL_ERROR")


# ---------------------------------------------------------------------------
# Tests: get_sector_comparison
# ---------------------------------------------------------------------------

class TestGetSectorComparison:
    def test_returns_string(self):
        from src.data.market_data import get_sector_comparison

        ohlcv = _make_ohlcv(22)
        info = {"sector": "Technology"}
        mock_ticker = _make_mock_ticker(ohlcv, info)
        dl_df = _make_download_response(["XLK", "SPY"], n=22)

        with patch("yfinance.Ticker", return_value=mock_ticker), patch(
            "yfinance.download", return_value=dl_df
        ):
            result = get_sector_comparison.invoke({"ticker": "AAPL"})

        assert isinstance(result, str)
        assert "SPY" in result

    def test_contains_return_data(self):
        from src.data.market_data import get_sector_comparison

        ohlcv = _make_ohlcv(22)
        info = {"sector": "Technology"}
        mock_ticker = _make_mock_ticker(ohlcv, info)
        dl_df = _make_download_response(["XLK", "SPY"], n=22)

        with patch("yfinance.Ticker", return_value=mock_ticker), patch(
            "yfinance.download", return_value=dl_df
        ):
            result = get_sector_comparison.invoke({"ticker": "AAPL"})

        assert "return" in result.lower() or "%" in result

    def test_falls_back_to_spy_on_missing_sector(self):
        from src.data.market_data import get_sector_comparison

        ohlcv = _make_ohlcv(22)
        info = {}  # no sector key
        mock_ticker = _make_mock_ticker(ohlcv, info)
        dl_df = _make_download_response(["SPY"], n=22)

        with patch("yfinance.Ticker", return_value=mock_ticker), patch(
            "yfinance.download", return_value=dl_df
        ):
            result = get_sector_comparison.invoke({"ticker": "XYZ"})

        assert isinstance(result, str)
        assert not result.startswith("TOOL_ERROR")

    def test_output_within_char_limit(self):
        from src.data.market_data import get_sector_comparison

        ohlcv = _make_ohlcv(22)
        mock_ticker = _make_mock_ticker(ohlcv, {"sector": "Energy"})
        dl_df = _make_download_response(["XLE", "SPY"], n=22)

        with patch("yfinance.Ticker", return_value=mock_ticker), patch(
            "yfinance.download", return_value=dl_df
        ):
            result = get_sector_comparison.invoke({"ticker": "XOM"})

        assert len(result) <= 2500

    def test_exception_returns_tool_error(self):
        from src.data.market_data import get_sector_comparison

        with patch("yfinance.Ticker", side_effect=ValueError("bad ticker")):
            result = get_sector_comparison.invoke({"ticker": "FAIL"})

        assert result.startswith("TOOL_ERROR")


# ---------------------------------------------------------------------------
# Tests: get_financial_statements
# ---------------------------------------------------------------------------

def _make_financials_df() -> pd.DataFrame:
    """Synthetic income statement DataFrame (rows=metrics, cols=years)."""
    years = pd.to_datetime(["2023-12-31", "2022-12-31", "2021-12-31"])
    return pd.DataFrame(
        {
            years[0]: [300e9, 120e9, 80e9, 60e9],
            years[1]: [280e9, 110e9, 75e9, 55e9],
            years[2]: [250e9, 100e9, 70e9, 50e9],
        },
        index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income"],
    )


def _make_balance_sheet_df() -> pd.DataFrame:
    years = pd.to_datetime(["2023-12-31", "2022-12-31", "2021-12-31"])
    return pd.DataFrame(
        {
            years[0]: [350e9, 120e9],
            years[1]: [320e9, 110e9],
            years[2]: [300e9, 100e9],
        },
        index=["Total Assets", "Total Debt"],
    )


def _make_cashflow_df() -> pd.DataFrame:
    years = pd.to_datetime(["2023-12-31", "2022-12-31", "2021-12-31"])
    return pd.DataFrame(
        {years[0]: [90e9], years[1]: [85e9], years[2]: [78e9]},
        index=["Operating Cash Flow"],
    )


class TestGetFinancialStatements:
    def _make_mock(self):
        mock = MagicMock()
        mock.financials = _make_financials_df()
        mock.balance_sheet = _make_balance_sheet_df()
        mock.cashflow = _make_cashflow_df()
        return mock

    def test_returns_string(self):
        from src.data.fundamental_data import get_financial_statements

        with patch("yfinance.Ticker", return_value=self._make_mock()):
            result = get_financial_statements.invoke({"ticker": "AAPL"})

        assert isinstance(result, str)
        assert "AAPL" in result.upper()

    def test_contains_key_metrics(self):
        from src.data.fundamental_data import get_financial_statements

        with patch("yfinance.Ticker", return_value=self._make_mock()):
            result = get_financial_statements.invoke({"ticker": "AAPL"})

        assert "Revenue" in result
        assert "Net Income" in result

    def test_output_within_char_limit(self):
        from src.data.fundamental_data import get_financial_statements

        with patch("yfinance.Ticker", return_value=self._make_mock()):
            result = get_financial_statements.invoke({"ticker": "AAPL"})

        assert len(result) <= 2500

    def test_handles_missing_data_gracefully(self):
        from src.data.fundamental_data import get_financial_statements

        mock = MagicMock()
        mock.financials = pd.DataFrame()
        mock.balance_sheet = pd.DataFrame()
        mock.cashflow = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock):
            result = get_financial_statements.invoke({"ticker": "EMPTY"})

        assert isinstance(result, str)
        assert not result.startswith("TOOL_ERROR")

    def test_exception_returns_tool_error(self):
        from src.data.fundamental_data import get_financial_statements

        with patch("yfinance.Ticker", side_effect=Exception("API down")):
            result = get_financial_statements.invoke({"ticker": "FAIL"})

        assert result.startswith("TOOL_ERROR")


# ---------------------------------------------------------------------------
# Tests: get_financial_ratios
# ---------------------------------------------------------------------------

_SAMPLE_RATIOS_INFO = {
    "trailingPE": 22.5,
    "priceToBook": 3.5,
    "priceToSalesTrailing12Months": 5.2,
    "enterpriseToEbitda": 15.0,
    "pegRatio": 1.3,
    "currentRatio": 1.8,
    "quickRatio": 1.4,
    "debtToEquity": 80.0,
}


class TestGetFinancialRatios:
    def test_returns_string(self):
        from src.data.fundamental_data import get_financial_ratios

        mock = MagicMock()
        mock.info = _SAMPLE_RATIOS_INFO

        with patch("yfinance.Ticker", return_value=mock):
            result = get_financial_ratios.invoke({"ticker": "AAPL"})

        assert isinstance(result, str)
        assert "AAPL" in result.upper()

    def test_contains_key_ratios(self):
        from src.data.fundamental_data import get_financial_ratios

        mock = MagicMock()
        mock.info = _SAMPLE_RATIOS_INFO

        with patch("yfinance.Ticker", return_value=mock):
            result = get_financial_ratios.invoke({"ticker": "AAPL"})

        assert "P/E" in result
        assert "P/B" in result
        assert "EV/EBITDA" in result
        assert "PEG" in result

    def test_contains_interpretations(self):
        from src.data.fundamental_data import get_financial_ratios

        mock = MagicMock()
        mock.info = _SAMPLE_RATIOS_INFO

        with patch("yfinance.Ticker", return_value=mock):
            result = get_financial_ratios.invoke({"ticker": "AAPL"})

        # PE 22.5 → Fair Value
        assert "Fair Value" in result

    def test_handles_missing_info(self):
        from src.data.fundamental_data import get_financial_ratios

        mock = MagicMock()
        mock.info = {}

        with patch("yfinance.Ticker", return_value=mock):
            result = get_financial_ratios.invoke({"ticker": "EMPTY"})

        assert isinstance(result, str)
        assert "N/A" in result

    def test_output_within_char_limit(self):
        from src.data.fundamental_data import get_financial_ratios

        mock = MagicMock()
        mock.info = _SAMPLE_RATIOS_INFO

        with patch("yfinance.Ticker", return_value=mock):
            result = get_financial_ratios.invoke({"ticker": "AAPL"})

        assert len(result) <= 2500

    def test_exception_returns_tool_error(self):
        from src.data.fundamental_data import get_financial_ratios

        with patch("yfinance.Ticker", side_effect=RuntimeError("quota exceeded")):
            result = get_financial_ratios.invoke({"ticker": "FAIL"})

        assert result.startswith("TOOL_ERROR")


# ---------------------------------------------------------------------------
# Tests: get_macro_context
# ---------------------------------------------------------------------------

def _make_macro_download() -> pd.DataFrame:
    """Synthetic multi-ticker download for macro symbols."""
    symbols = ["^GSPC", "^VIX", "^TNX", "DX-Y.NYB"]
    n = 22
    idx = pd.bdate_range("2023-01-01", periods=n)
    np.random.seed(7)
    arrays = [["Close"] * len(symbols), symbols]
    cols = pd.MultiIndex.from_arrays(arrays)
    data = np.column_stack(
        [
            np.linspace(4000, 4200, n),   # GSPC
            np.linspace(20, 18, n),        # VIX
            np.linspace(4.0, 4.2, n),      # TNX
            np.linspace(103, 104, n),      # DXY
        ]
    )
    return pd.DataFrame(data, index=idx, columns=cols)


class TestGetMacroContext:
    def test_returns_string(self):
        from src.data.fundamental_data import get_macro_context

        with patch("yfinance.download", return_value=_make_macro_download()):
            result = get_macro_context.invoke({})

        assert isinstance(result, str)
        assert "Macro" in result or "S&P" in result

    def test_contains_key_indicators(self):
        from src.data.fundamental_data import get_macro_context

        with patch("yfinance.download", return_value=_make_macro_download()):
            result = get_macro_context.invoke({})

        assert "VIX" in result
        assert "Treasury" in result or "TNX" in result

    def test_output_within_char_limit(self):
        from src.data.fundamental_data import get_macro_context

        with patch("yfinance.download", return_value=_make_macro_download()):
            result = get_macro_context.invoke({})

        assert len(result) <= 2500

    def test_empty_download_handled(self):
        from src.data.fundamental_data import get_macro_context

        with patch("yfinance.download", return_value=pd.DataFrame()):
            result = get_macro_context.invoke({})

        assert isinstance(result, str)

    def test_exception_returns_tool_error(self):
        from src.data.fundamental_data import get_macro_context

        with patch("yfinance.download", side_effect=Exception("server error")):
            result = get_macro_context.invoke({})

        assert result.startswith("TOOL_ERROR")


# ---------------------------------------------------------------------------
# Tests: all tools return TOOL_ERROR on unexpected exceptions
# ---------------------------------------------------------------------------

class TestToolsReturnToolErrorOnException:
    """Verify every new tool catches all exceptions and returns TOOL_ERROR."""

    @pytest.mark.parametrize(
        "import_path,func_name,args",
        [
            ("src.data.market_data", "get_technical_indicators", {"ticker": "X"}),
            ("src.data.market_data", "get_support_resistance", {"ticker": "X"}),
            ("src.data.market_data", "get_sector_comparison", {"ticker": "X"}),
            ("src.data.fundamental_data", "get_financial_statements", {"ticker": "X"}),
            ("src.data.fundamental_data", "get_financial_ratios", {"ticker": "X"}),
        ],
    )
    def test_ticker_tool_error(self, import_path: str, func_name: str, args: dict):
        import importlib

        mod = importlib.import_module(import_path)
        func = getattr(mod, func_name)

        with patch("yfinance.Ticker", side_effect=Exception("boom")):
            result = func.invoke(args)

        assert isinstance(result, str)
        assert result.startswith("TOOL_ERROR"), f"{func_name} did not return TOOL_ERROR"

    def test_get_macro_context_error(self):
        from src.data.fundamental_data import get_macro_context

        with patch("yfinance.download", side_effect=Exception("boom")):
            result = get_macro_context.invoke({})

        assert result.startswith("TOOL_ERROR")
