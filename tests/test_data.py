"""Tests for all equity data tools: market_data, news_data, fundamental_data,
earnings_calendar, alternative_data."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 30, start_price: float = 150.0) -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame."""
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = np.linspace(start_price, start_price * 1.05, n)
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.02,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 50_000_000, n),
        },
        index=dates,
    )
    return df


def _make_fast_info(**kwargs) -> MagicMock:
    fi = MagicMock()
    fi.last_price = kwargs.get("last_price", 175.23)
    fi.last_volume = kwargs.get("last_volume", 52_300_000)
    fi.three_month_average_volume = kwargs.get("three_month_average_volume", 52_300_000)
    fi.day_high = kwargs.get("day_high", 176.80)
    fi.day_low = kwargs.get("day_low", 172.10)
    fi.previous_close = kwargs.get("previous_close", 174.00)
    return fi


# ===========================================================================
# market_data tests
# ===========================================================================

class TestGetStockPrice:
    def test_success(self):
        from src.data.market_data import get_stock_price

        mock_ticker = MagicMock()
        mock_ticker.fast_info = _make_fast_info()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_stock_price.invoke({"ticker": "AAPL"})

        assert "AAPL" in result
        assert "$175.23" in result
        assert "Volume" in result
        assert "Day" in result
        assert "Change" in result

    def test_percent_change_positive(self):
        from src.data.market_data import get_stock_price

        fi = _make_fast_info(last_price=180.0, previous_close=174.0)
        mock_ticker = MagicMock()
        mock_ticker.fast_info = fi

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_stock_price.invoke({"ticker": "AAPL"})

        assert "+" in result

    def test_percent_change_negative(self):
        from src.data.market_data import get_stock_price

        fi = _make_fast_info(last_price=170.0, previous_close=174.0)
        mock_ticker = MagicMock()
        mock_ticker.fast_info = fi

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_stock_price.invoke({"ticker": "AAPL"})

        assert "-" in result

    def test_no_price_returns_tool_error(self):
        from src.data.market_data import get_stock_price

        fi = _make_fast_info(last_price=None)
        mock_ticker = MagicMock()
        mock_ticker.fast_info = fi

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_stock_price.invoke({"ticker": "INVALID"})

        assert "TOOL_ERROR" in result

    def test_exception_returns_tool_error(self):
        from src.data.market_data import get_stock_price

        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            result = get_stock_price.invoke({"ticker": "AAPL"})

        assert "TOOL_ERROR" in result
        assert "network error" in result

    def test_volume_millions_format(self):
        from src.data.market_data import get_stock_price

        fi = _make_fast_info(last_volume=52_300_000)
        mock_ticker = MagicMock()
        mock_ticker.fast_info = fi

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_stock_price.invoke({"ticker": "AAPL"})

        assert "M" in result


class TestGetStockHistory:
    def test_success(self):
        from src.data.market_data import get_stock_history

        hist_df = _make_ohlcv(30)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_stock_history.invoke({"ticker": "AAPL", "period": "1mo"})

        assert "AAPL" in result
        assert "Period Return" in result
        assert "Volatility" in result
        assert "Avg Daily Volume" in result
        assert "Recent 5 Closes" in result

    def test_empty_history_returns_tool_error(self):
        from src.data.market_data import get_stock_history

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_stock_history.invoke({"ticker": "INVALID"})

        assert "TOOL_ERROR" in result

    def test_result_under_2500_chars(self):
        from src.data.market_data import get_stock_history

        hist_df = _make_ohlcv(252)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_stock_history.invoke({"ticker": "AAPL", "period": "1y"})

        assert len(result) <= 2500

    def test_exception_returns_tool_error(self):
        from src.data.market_data import get_stock_history

        with patch("yfinance.Ticker", side_effect=RuntimeError("timeout")):
            result = get_stock_history.invoke({"ticker": "AAPL"})

        assert "TOOL_ERROR" in result

    def test_default_period(self):
        from src.data.market_data import get_stock_history

        hist_df = _make_ohlcv(126)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_stock_history.invoke({"ticker": "MSFT"})

        assert "MSFT" in result


class TestGetOptionsData:
    def _make_options_chain(self):
        calls = pd.DataFrame({
            "strike": [170.0, 175.0, 180.0, 185.0, 190.0],
            "lastPrice": [8.0, 5.0, 3.0, 1.5, 0.5],
            "impliedVolatility": [0.25, 0.24, 0.23, 0.22, 0.21],
            "volume": [1000, 2000, 3000, 500, 200],
            "openInterest": [5000, 8000, 10000, 3000, 1000],
        })
        puts = pd.DataFrame({
            "strike": [160.0, 165.0, 170.0, 175.0, 180.0],
            "lastPrice": [0.5, 1.0, 2.5, 4.5, 7.0],
            "impliedVolatility": [0.28, 0.27, 0.26, 0.25, 0.24],
            "volume": [800, 1500, 2500, 4000, 6000],
            "openInterest": [2000, 4000, 6000, 9000, 12000],
        })
        chain = MagicMock()
        chain.calls = calls
        chain.puts = puts
        return chain

    def test_success(self):
        from src.data.market_data import get_options_data

        mock_ticker = MagicMock()
        mock_ticker.options = ["2024-02-16", "2024-03-15"]
        mock_ticker.option_chain.return_value = self._make_options_chain()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_options_data.invoke({"ticker": "AAPL"})

        assert "AAPL" in result
        assert "Put/Call" in result
        assert "IV" in result or "Avg" in result

    def test_no_options_returns_tool_error(self):
        from src.data.market_data import get_options_data

        mock_ticker = MagicMock()
        mock_ticker.options = []

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_options_data.invoke({"ticker": "INVALID"})

        assert "TOOL_ERROR" in result

    def test_exception_returns_tool_error(self):
        from src.data.market_data import get_options_data

        with patch("yfinance.Ticker", side_effect=Exception("API error")):
            result = get_options_data.invoke({"ticker": "AAPL"})

        assert "TOOL_ERROR" in result

    def test_result_under_2500_chars(self):
        from src.data.market_data import get_options_data

        mock_ticker = MagicMock()
        mock_ticker.options = ["2024-02-16"]
        mock_ticker.option_chain.return_value = self._make_options_chain()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_options_data.invoke({"ticker": "AAPL"})

        assert len(result) <= 2500


# ===========================================================================
# news_data tests
# ===========================================================================

class TestGetCompanyNews:
    def _make_feed(self, n: int = 5):
        feed = MagicMock()
        entries = []
        for i in range(n):
            e = MagicMock()
            e.get = lambda k, d="", i=i: {
                "title": f"AAPL headline {i}: stock surges",
                "published": "2024-01-15",
                "summary": f"Apple reports strong earnings for Q{i}",
            }.get(k, d)
            entries.append(e)
        feed.entries = entries
        return feed

    def test_rss_fallback_success(self):
        from src.data.news_data import get_company_news
        from src.utils.config import reset_config

        reset_config()
        with patch("src.utils.config.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(finnhub_api_key="")
            with patch("feedparser.parse", return_value=self._make_feed(5)):
                result = get_company_news.invoke({"ticker": "AAPL"})

        assert "AAPL" in result
        assert "headline" in result.lower() or "surges" in result.lower()

    def test_finnhub_success(self):
        from src.data.news_data import get_company_news

        mock_news = [
            {"datetime": 1705363200, "headline": "Apple beats earnings estimates", "summary": "Strong Q1 results"},
            {"datetime": 1705276800, "headline": "Apple launches new product", "summary": "New iPhone announced"},
        ]

        with patch("src.utils.config.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(finnhub_api_key="test-key")
            mock_client = MagicMock()
            mock_client.company_news.return_value = mock_news
            with patch("finnhub.Client", return_value=mock_client):
                result = get_company_news.invoke({"ticker": "AAPL"})

        assert "AAPL" in result
        assert "FinnHub" in result

    def test_finnhub_fallback_to_rss_on_error(self):
        from src.data.news_data import get_company_news

        with patch("src.utils.config.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(finnhub_api_key="bad-key")
            with patch("finnhub.Client", side_effect=Exception("auth error")):
                with patch("feedparser.parse", return_value=self._make_feed(3)):
                    result = get_company_news.invoke({"ticker": "AAPL"})

        assert "AAPL" in result

    def test_result_under_2500_chars(self):
        from src.data.news_data import get_company_news

        with patch("src.utils.config.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(finnhub_api_key="")
            with patch("feedparser.parse", return_value=self._make_feed(5)):
                result = get_company_news.invoke({"ticker": "AAPL"})

        assert len(result) <= 2500

    def test_empty_feed_returns_no_news(self):
        from src.data.news_data import get_company_news

        with patch("src.utils.config.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(finnhub_api_key="")
            empty_feed = MagicMock()
            empty_feed.entries = []
            with patch("feedparser.parse", return_value=empty_feed):
                result = get_company_news.invoke({"ticker": "AAPL"})

        assert "No news found" in result

    def test_exception_returns_tool_error(self):
        from src.data.news_data import get_company_news

        with patch("src.utils.config.get_config", side_effect=Exception("config error")):
            result = get_company_news.invoke({"ticker": "AAPL"})

        assert "TOOL_ERROR" in result

    def test_sentiment_hints(self):
        from src.data.news_data import get_company_news

        with patch("src.utils.config.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(finnhub_api_key="")
            feed = MagicMock()
            entry = MagicMock()
            entry.get = lambda k, d="": {
                "title": "Apple stock surges on strong earnings beat",
                "published": "2024-01-15",
                "summary": "",
            }.get(k, d)
            feed.entries = [entry]
            with patch("feedparser.parse", return_value=feed):
                result = get_company_news.invoke({"ticker": "AAPL"})

        # Should have a sentiment hint
        assert "[+]" in result or "[-]" in result or "[~]" in result


class TestGetMarketNews:
    def _make_feed(self, n: int = 5):
        feed = MagicMock()
        entries = []
        for i in range(n):
            e = MagicMock()
            e.get = lambda k, d="", i=i: {
                "title": f"Market update {i}: S&P 500 rises",
                "published": "2024-01-15",
                "summary": f"Markets rally on Fed news {i}",
            }.get(k, d)
            entries.append(e)
        feed.entries = entries
        return feed

    def test_rss_fallback_success(self):
        from src.data.news_data import get_market_news

        with patch("src.utils.config.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(finnhub_api_key="")
            with patch("feedparser.parse", return_value=self._make_feed(8)):
                result = get_market_news.invoke({"category": "general"})

        assert "Market News" in result

    def test_finnhub_market_news_success(self):
        from src.data.news_data import get_market_news

        mock_news = [
            {"datetime": 1705363200, "headline": "Fed holds rates steady", "summary": "Federal Reserve decision"},
            {"datetime": 1705276800, "headline": "S&P 500 hits new high", "summary": "Bull market continues"},
        ]

        with patch("src.utils.config.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(finnhub_api_key="test-key")
            mock_client = MagicMock()
            mock_client.general_news.return_value = mock_news
            with patch("finnhub.Client", return_value=mock_client):
                result = get_market_news.invoke({"category": "general"})

        assert "FinnHub" in result
        assert "Fed" in result or "S&P" in result

    def test_default_category(self):
        from src.data.news_data import get_market_news

        with patch("src.utils.config.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(finnhub_api_key="")
            with patch("feedparser.parse", return_value=self._make_feed(5)):
                result = get_market_news.invoke({})

        assert "general" in result.lower() or "Market" in result

    def test_result_under_2500_chars(self):
        from src.data.news_data import get_market_news

        with patch("src.utils.config.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(finnhub_api_key="")
            with patch("feedparser.parse", return_value=self._make_feed(10)):
                result = get_market_news.invoke({"category": "general"})

        assert len(result) <= 2500

    def test_exception_returns_tool_error(self):
        from src.data.news_data import get_market_news

        with patch("src.utils.config.get_config", side_effect=Exception("fail")):
            result = get_market_news.invoke({"category": "general"})

        assert "TOOL_ERROR" in result


# ===========================================================================
# fundamental_data tests
# ===========================================================================

class TestGetFinancials:
    def _make_ticker_mock(self):
        t = MagicMock()
        t.info = {
            "totalRevenue": 385_000_000_000,
            "netIncomeToCommon": 95_000_000_000,
            "ebitda": 125_000_000_000,
            "trailingEps": 6.08,
            "trailingPE": 28.5,
            "priceToSalesTrailing12Months": 7.2,
            "debtToEquity": 1.8,
            "returnOnEquity": 0.147,
            "returnOnAssets": 0.285,
            "freeCashflow": 90_000_000_000,
            "marketCap": 2_750_000_000_000,
        }
        # Mock financials DataFrame
        fins = pd.DataFrame(
            {
                "2023-09-30": [385e9, 95e9, 125e9],
            },
            index=["Total Revenue", "Net Income", "EBITDA"],
        )
        t.financials = fins
        return t

    def test_success(self):
        from src.data.fundamental_data import get_financials

        mock_ticker = self._make_ticker_mock()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_financials.invoke({"ticker": "AAPL"})

        assert "AAPL" in result
        assert "Revenue" in result
        assert "Net Income" in result
        assert "P/E" in result

    def test_result_under_2500_chars(self):
        from src.data.fundamental_data import get_financials

        mock_ticker = self._make_ticker_mock()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_financials.invoke({"ticker": "AAPL"})

        assert len(result) <= 2500

    def test_exception_returns_tool_error(self):
        from src.data.fundamental_data import get_financials

        with patch("yfinance.Ticker", side_effect=Exception("API error")):
            result = get_financials.invoke({"ticker": "AAPL"})

        assert "TOOL_ERROR" in result

    def test_missing_fields_show_na(self):
        from src.data.fundamental_data import get_financials

        mock_ticker = MagicMock()
        mock_ticker.info = {}  # all missing
        mock_ticker.financials = pd.DataFrame()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_financials.invoke({"ticker": "AAPL"})

        assert "N/A" in result


class TestGetInsiderTrades:
    def _make_txns(self):
        return pd.DataFrame({
            "Start Date": ["2024-01-10", "2024-01-05", "2023-12-28"],
            "Insider": ["Tim Cook", "Luca Maestri", "Jeff Williams"],
            "Shares": [100000, 50000, 75000],
            "Transaction": ["Sale", "Sale", "Purchase"],
            "Value": [17_500_000, 8_750_000, 13_125_000],
        })

    def test_success(self):
        from src.data.fundamental_data import get_insider_trades

        mock_ticker = MagicMock()
        mock_ticker.insider_transactions = self._make_txns()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_insider_trades.invoke({"ticker": "AAPL"})

        assert "AAPL" in result
        assert "Tim Cook" in result
        assert "Sale" in result or "Purchase" in result

    def test_no_data_returns_message(self):
        from src.data.fundamental_data import get_insider_trades

        mock_ticker = MagicMock()
        mock_ticker.insider_transactions = None

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_insider_trades.invoke({"ticker": "AAPL"})

        assert "No insider" in result or "TOOL_ERROR" in result

    def test_exception_returns_tool_error(self):
        from src.data.fundamental_data import get_insider_trades

        with patch("yfinance.Ticker", side_effect=Exception("fail")):
            result = get_insider_trades.invoke({"ticker": "AAPL"})

        assert "TOOL_ERROR" in result

    def test_result_under_2500_chars(self):
        from src.data.fundamental_data import get_insider_trades

        mock_ticker = MagicMock()
        # Make large dataset
        big_txns = self._make_txns()
        big_txns = pd.concat([big_txns] * 5, ignore_index=True)
        mock_ticker.insider_transactions = big_txns

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_insider_trades.invoke({"ticker": "AAPL"})

        assert len(result) <= 2500


class TestGetAnalystRatings:
    def _make_ticker_mock(self):
        t = MagicMock()
        t.info = {
            "targetMeanPrice": 200.0,
            "targetHighPrice": 220.0,
            "targetLowPrice": 170.0,
            "currentPrice": 175.0,
            "recommendationKey": "buy",
            "numberOfAnalystOpinions": 32,
        }
        t.recommendations_summary = pd.DataFrame({
            "period": ["0q", "-1q"],
            "strongBuy": [10, 8],
            "buy": [15, 12],
            "hold": [5, 7],
            "sell": [1, 2],
            "strongSell": [1, 3],
        })
        t.recommendations = pd.DataFrame(
            {
                "Firm": ["Goldman Sachs", "Morgan Stanley"],
                "To Grade": ["Buy", "Overweight"],
            },
            index=pd.to_datetime(["2024-01-10", "2024-01-08"]),
        )
        return t

    def test_success(self):
        from src.data.fundamental_data import get_analyst_ratings

        with patch("yfinance.Ticker", return_value=self._make_ticker_mock()):
            result = get_analyst_ratings.invoke({"ticker": "AAPL"})

        assert "AAPL" in result
        assert "BUY" in result
        assert "$200.00" in result or "200" in result

    def test_result_under_2500_chars(self):
        from src.data.fundamental_data import get_analyst_ratings

        with patch("yfinance.Ticker", return_value=self._make_ticker_mock()):
            result = get_analyst_ratings.invoke({"ticker": "AAPL"})

        assert len(result) <= 2500

    def test_exception_returns_tool_error(self):
        from src.data.fundamental_data import get_analyst_ratings

        with patch("yfinance.Ticker", side_effect=Exception("network")):
            result = get_analyst_ratings.invoke({"ticker": "AAPL"})

        assert "TOOL_ERROR" in result

    def test_missing_info_shows_na(self):
        from src.data.fundamental_data import get_analyst_ratings

        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.recommendations_summary = pd.DataFrame()
        mock_ticker.recommendations = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_analyst_ratings.invoke({"ticker": "AAPL"})

        assert "N/A" in result


class TestGetSecFilings:
    def _make_feed(self, n: int = 5):
        feed = MagicMock()
        entries = []
        for i in range(n):
            e = MagicMock()
            e.get = lambda k, d="", i=i: {
                "title": f"Apple Inc 10-Q {i}",
                "published": "2024-01-10",
                "summary": f"Quarterly report for Q{i}",
                "category": "10-Q",
            }.get(k, d)
            entries.append(e)
        feed.entries = entries
        return feed

    def test_success(self):
        from src.data.fundamental_data import get_sec_filings

        with patch("feedparser.parse", return_value=self._make_feed(5)):
            result = get_sec_filings.invoke({"ticker": "AAPL"})

        assert "AAPL" in result
        assert "SEC" in result or "Filing" in result or "10-" in result

    def test_no_entries_message(self):
        from src.data.fundamental_data import get_sec_filings

        empty_feed = MagicMock()
        empty_feed.entries = []

        with patch("feedparser.parse", return_value=empty_feed):
            result = get_sec_filings.invoke({"ticker": "AAPL"})

        assert "No recent SEC filings" in result or "AAPL" in result

    def test_exception_returns_tool_error(self):
        from src.data.fundamental_data import get_sec_filings

        with patch("feedparser.parse", side_effect=Exception("network")):
            result = get_sec_filings.invoke({"ticker": "AAPL"})

        assert "TOOL_ERROR" in result

    def test_result_under_2500_chars(self):
        from src.data.fundamental_data import get_sec_filings

        with patch("feedparser.parse", return_value=self._make_feed(5)):
            result = get_sec_filings.invoke({"ticker": "AAPL"})

        assert len(result) <= 2500


# ===========================================================================
# earnings_calendar tests
# ===========================================================================

class TestGetEarningsCalendar:
    def _make_calendar_dict(self):
        return {
            "Earnings Date": "2024-02-01",
            "EPS Estimate": 2.10,
            "Revenue Estimate": 118_000_000_000,
            "Earnings Call Time": "After Market Close",
        }

    def test_success_dict_calendar(self):
        from src.data.earnings_calendar import get_earnings_calendar

        mock_ticker = MagicMock()
        mock_ticker.calendar = self._make_calendar_dict()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_earnings_calendar.invoke({"ticker": "AAPL"})

        assert "AAPL" in result
        assert "2024-02-01" in result
        assert "2.10" in result or "EPS" in result
        assert "AMC" in result

    def test_success_dataframe_calendar(self):
        from src.data.earnings_calendar import get_earnings_calendar

        cal_df = pd.DataFrame(
            {"Earnings Date": ["2024-02-01"], "EPS Estimate": [2.10], "Revenue Estimate": [118e9]},
        )
        mock_ticker = MagicMock()
        mock_ticker.calendar = cal_df

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_earnings_calendar.invoke({"ticker": "AAPL"})

        assert "AAPL" in result

    def test_none_calendar_message(self):
        from src.data.earnings_calendar import get_earnings_calendar

        mock_ticker = MagicMock()
        mock_ticker.calendar = None

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_earnings_calendar.invoke({"ticker": "AAPL"})

        assert "No earnings calendar" in result

    def test_exception_returns_tool_error(self):
        from src.data.earnings_calendar import get_earnings_calendar

        with patch("yfinance.Ticker", side_effect=Exception("fail")):
            result = get_earnings_calendar.invoke({"ticker": "AAPL"})

        assert "TOOL_ERROR" in result

    def test_bmo_timing(self):
        from src.data.earnings_calendar import get_earnings_calendar

        cal = {
            "Earnings Date": "2024-02-01",
            "EPS Estimate": 2.10,
            "Revenue Estimate": 118e9,
            "Earnings Call Time": "Before Market Open",
        }
        mock_ticker = MagicMock()
        mock_ticker.calendar = cal

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_earnings_calendar.invoke({"ticker": "AAPL"})

        assert "BMO" in result

    def test_result_under_2500_chars(self):
        from src.data.earnings_calendar import get_earnings_calendar

        mock_ticker = MagicMock()
        mock_ticker.calendar = self._make_calendar_dict()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_earnings_calendar.invoke({"ticker": "AAPL"})

        assert len(result) <= 2500


class TestGetEarningsEstimate:
    def test_success_with_earnings_estimate(self):
        from src.data.earnings_calendar import get_earnings_estimate

        ee_df = pd.DataFrame(
            {
                "avg": [2.10, 2.30, 8.50, 9.20],
                "low": [1.90, 2.10, 8.00, 8.80],
                "high": [2.30, 2.50, 9.00, 9.60],
                "numberOfAnalysts": [28, 25, 30, 27],
                "growth": [0.08, 0.095, 0.085, 0.09],
            },
            index=["0q", "+1q", "0y", "+1y"],
        )
        mock_ticker = MagicMock()
        mock_ticker.earnings_estimate = ee_df
        mock_ticker.eps_trend = pd.DataFrame()
        mock_ticker.earnings_history = pd.DataFrame()
        mock_ticker.info = {"forwardEps": 9.20, "trailingEps": 6.08, "forwardPE": 19.0}

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_earnings_estimate.invoke({"ticker": "AAPL"})

        assert "AAPL" in result
        assert "Estimates" in result or "EPS" in result

    def test_fallback_to_info(self):
        from src.data.earnings_calendar import get_earnings_estimate

        mock_ticker = MagicMock()
        mock_ticker.earnings_estimate = None
        mock_ticker.eps_trend = None
        mock_ticker.earnings_history = None
        mock_ticker.info = {"forwardEps": 9.20, "trailingEps": 6.08, "forwardPE": 19.0}

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_earnings_estimate.invoke({"ticker": "AAPL"})

        assert "9.2" in result or "6.08" in result or "EPS" in result

    def test_no_data_message(self):
        from src.data.earnings_calendar import get_earnings_estimate

        mock_ticker = MagicMock()
        mock_ticker.earnings_estimate = None
        mock_ticker.eps_trend = None
        mock_ticker.earnings_history = None
        mock_ticker.info = {}

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_earnings_estimate.invoke({"ticker": "AAPL"})

        assert "No earnings estimate data" in result or "AAPL" in result

    def test_exception_returns_tool_error(self):
        from src.data.earnings_calendar import get_earnings_estimate

        with patch("yfinance.Ticker", side_effect=Exception("fail")):
            result = get_earnings_estimate.invoke({"ticker": "AAPL"})

        assert "TOOL_ERROR" in result

    def test_result_under_2500_chars(self):
        from src.data.earnings_calendar import get_earnings_estimate

        mock_ticker = MagicMock()
        mock_ticker.earnings_estimate = pd.DataFrame({"avg": [2.1, 2.3]})
        mock_ticker.eps_trend = pd.DataFrame()
        mock_ticker.earnings_history = pd.DataFrame()
        mock_ticker.info = {"forwardEps": 9.20}

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = get_earnings_estimate.invoke({"ticker": "AAPL"})

        assert len(result) <= 2500


# ===========================================================================
# alternative_data tests
# ===========================================================================

class TestGetFearGreedIndex:
    def _make_response(self, value: int = 72, label: str = "Greed") -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {
                    "value": str(value),
                    "value_classification": label,
                    "timestamp": "1705363200",
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_success(self):
        from src.data.alternative_data import get_fear_greed_index

        with patch("requests.get", return_value=self._make_response(72, "Greed")):
            result = get_fear_greed_index.invoke({})

        assert "Fear" in result or "Greed" in result
        assert "72" in result

    def test_extreme_fear(self):
        from src.data.alternative_data import get_fear_greed_index

        with patch("requests.get", return_value=self._make_response(15, "Extreme Fear")):
            result = get_fear_greed_index.invoke({})

        assert "15" in result
        assert "Extreme Fear" in result or "buying opportunity" in result

    def test_extreme_greed(self):
        from src.data.alternative_data import get_fear_greed_index

        with patch("requests.get", return_value=self._make_response(85, "Extreme Greed")):
            result = get_fear_greed_index.invoke({})

        assert "85" in result
        assert "Extreme Greed" in result or "market top" in result.lower()

    def test_exception_returns_tool_error(self):
        from src.data.alternative_data import get_fear_greed_index

        with patch("requests.get", side_effect=Exception("connection refused")):
            result = get_fear_greed_index.invoke({})

        assert "TOOL_ERROR" in result

    def test_result_under_2500_chars(self):
        from src.data.alternative_data import get_fear_greed_index

        with patch("requests.get", return_value=self._make_response(50, "Neutral")):
            result = get_fear_greed_index.invoke({})

        assert len(result) <= 2500

    def test_neutral_range(self):
        from src.data.alternative_data import get_fear_greed_index

        with patch("requests.get", return_value=self._make_response(50, "Neutral")):
            result = get_fear_greed_index.invoke({})

        assert "Neutral" in result or "balanced" in result.lower()


class TestGetVix:
    def _make_vix_df(self, current: float = 18.5) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-08", periods=5)
        closes = [17.0, 18.0, 19.0, 18.5, current]
        highs = [h + 1 for h in closes]
        lows = [l - 1 for l in closes]
        return pd.DataFrame(
            {"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": [0] * 5},
            index=dates,
        )

    def test_success_normal_regime(self):
        from src.data.alternative_data import get_vix

        with patch("yfinance.download", return_value=self._make_vix_df(18.5)):
            result = get_vix.invoke({})

        assert "VIX" in result
        assert "18.5" in result or "18.50" in result
        assert "Normal" in result

    def test_low_volatility_regime(self):
        from src.data.alternative_data import get_vix

        with patch("yfinance.download", return_value=self._make_vix_df(12.0)):
            result = get_vix.invoke({})

        assert "Low Volatility" in result

    def test_high_volatility_regime(self):
        from src.data.alternative_data import get_vix

        with patch("yfinance.download", return_value=self._make_vix_df(35.0)):
            result = get_vix.invoke({})

        assert "High Volatility" in result

    def test_empty_df_returns_tool_error(self):
        from src.data.alternative_data import get_vix

        with patch("yfinance.download", return_value=pd.DataFrame()):
            result = get_vix.invoke({})

        assert "TOOL_ERROR" in result

    def test_exception_returns_tool_error(self):
        from src.data.alternative_data import get_vix

        with patch("yfinance.download", side_effect=Exception("API error")):
            result = get_vix.invoke({})

        assert "TOOL_ERROR" in result

    def test_result_under_2500_chars(self):
        from src.data.alternative_data import get_vix

        with patch("yfinance.download", return_value=self._make_vix_df(20.0)):
            result = get_vix.invoke({})

        assert len(result) <= 2500


class TestGetSectorPerformance:
    def _make_sector_df(self) -> pd.DataFrame:
        etfs = ["XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
        dates = pd.bdate_range("2024-01-02", periods=22)
        close_data = {}
        np.random.seed(42)
        for etf in etfs:
            base = np.random.uniform(40, 100)
            ret = np.random.normal(0.001, 0.01, len(dates))
            prices = base * np.cumprod(1 + ret)
            close_data[etf] = prices

        close_df = pd.DataFrame(close_data, index=dates)
        # Simulate MultiIndex columns like yfinance returns
        arrays = [["Close"] * len(etfs), etfs]
        tuples = list(zip(*arrays))
        mi = pd.MultiIndex.from_tuples(tuples, names=["Price", "Ticker"])
        result = pd.DataFrame(close_df.values, index=dates, columns=mi)
        return result

    def test_success(self):
        from src.data.alternative_data import get_sector_performance

        with patch("yfinance.download", return_value=self._make_sector_df()):
            result = get_sector_performance.invoke({})

        assert "Sector Performance" in result
        assert "XLK" in result
        assert "%" in result

    def test_all_sectors_present(self):
        from src.data.alternative_data import get_sector_performance

        with patch("yfinance.download", return_value=self._make_sector_df()):
            result = get_sector_performance.invoke({})

        for etf in ["XLK", "XLF", "XLV", "XLE", "XLY"]:
            assert etf in result

    def test_empty_df_returns_tool_error(self):
        from src.data.alternative_data import get_sector_performance

        with patch("yfinance.download", return_value=pd.DataFrame()):
            result = get_sector_performance.invoke({})

        assert "TOOL_ERROR" in result

    def test_exception_returns_tool_error(self):
        from src.data.alternative_data import get_sector_performance

        with patch("yfinance.download", side_effect=Exception("network")):
            result = get_sector_performance.invoke({})

        assert "TOOL_ERROR" in result

    def test_result_under_2500_chars(self):
        from src.data.alternative_data import get_sector_performance

        with patch("yfinance.download", return_value=self._make_sector_df()):
            result = get_sector_performance.invoke({})

        assert len(result) <= 2500

    def test_sorted_by_return(self):
        from src.data.alternative_data import get_sector_performance

        with patch("yfinance.download", return_value=self._make_sector_df()):
            result = get_sector_performance.invoke({})

        # Find all return values in output and verify descending order
        lines = [l for l in result.split("\n") if "%" in l and any(e in l for e in ["XLK", "XLF", "XLV", "XLE"])]
        returns = []
        for line in lines:
            for part in line.split():
                part_clean = part.replace("%", "").replace("+", "")
                try:
                    returns.append(float(part_clean))
                    break
                except ValueError:
                    continue
        if len(returns) >= 2:
            assert returns == sorted(returns, reverse=True)
