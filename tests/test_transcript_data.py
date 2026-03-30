"""Tests for earnings call transcript data tools (FMP API)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.utils.config import AppConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(data: list) -> MagicMock:
    """Build a mock requests.Response with .json() and .raise_for_status()."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = data
    mock_resp.raise_for_status.return_value = None
    return mock_resp


def _mock_config(api_key: str = "test-fmp-key") -> AppConfig:
    """Return an AppConfig with the given FMP API key."""
    return AppConfig(fmp_api_key=api_key)


SAMPLE_TRANSCRIPT = [
    {
        "symbol": "AAPL",
        "quarter": 4,
        "year": 2024,
        "date": "2025-01-28",
        "content": "Good quarter everyone. Revenue grew significantly this year.",
    }
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetEarningsTranscript:
    def test_success_returns_formatted_transcript(self):
        """Mock API returns valid transcript; verify header in output."""
        from src.data.transcript_data import get_earnings_transcript

        with (
            patch("src.data.transcript_data.get_config", return_value=_mock_config()),
            patch("src.data.transcript_data._most_recent_quarter", return_value=(2024, 4)),
            patch(
                "src.data.transcript_data.requests.get",
                return_value=_make_mock_response(SAMPLE_TRANSCRIPT),
            ),
        ):
            result = get_earnings_transcript.invoke({"ticker": "AAPL"})

        assert "=== AAPL Earnings Call Transcript" in result
        assert "Q4 2024" in result
        assert "2025-01-28" in result
        assert "Good quarter" in result

    def test_output_truncated_to_2500_chars(self):
        """Long transcript content must be truncated to 2500 chars."""
        from src.data.transcript_data import get_earnings_transcript

        long_content = "A" * 5000
        data = [{"symbol": "AAPL", "quarter": 4, "year": 2024, "date": "2025-01-28", "content": long_content}]

        with (
            patch("src.data.transcript_data.get_config", return_value=_mock_config()),
            patch(
                "src.data.transcript_data.requests.get",
                return_value=_make_mock_response(data),
            ),
        ):
            result = get_earnings_transcript.invoke({"ticker": "AAPL"})

        assert len(result) <= 2500

    def test_no_api_key_returns_tool_error(self):
        """Missing FMP API key must return TOOL_ERROR immediately."""
        from src.data.transcript_data import get_earnings_transcript

        with patch("src.data.transcript_data.get_config", return_value=_mock_config(api_key="")):
            result = get_earnings_transcript.invoke({"ticker": "AAPL"})

        assert result.startswith("TOOL_ERROR")
        assert "FMP API key" in result

    def test_api_failure_returns_tool_error(self):
        """Network failure must return TOOL_ERROR."""
        from src.data.transcript_data import get_earnings_transcript

        with (
            patch("src.data.transcript_data.get_config", return_value=_mock_config()),
            patch(
                "src.data.transcript_data.requests.get",
                side_effect=ConnectionError("Connection refused"),
            ),
        ):
            result = get_earnings_transcript.invoke({"ticker": "AAPL"})

        assert result.startswith("TOOL_ERROR")

    def test_empty_response_returns_tool_error(self):
        """API returning empty list for both quarters must return TOOL_ERROR."""
        from src.data.transcript_data import get_earnings_transcript

        with (
            patch("src.data.transcript_data.get_config", return_value=_mock_config()),
            patch(
                "src.data.transcript_data.requests.get",
                return_value=_make_mock_response([]),
            ),
        ):
            result = get_earnings_transcript.invoke({"ticker": "AAPL"})

        assert result.startswith("TOOL_ERROR")
        assert "No transcript found" in result

    def test_fallback_to_previous_quarter(self):
        """First call returns []; second call returns valid data — verify fallback succeeds."""
        from src.data.transcript_data import get_earnings_transcript

        fallback_data = [
            {
                "symbol": "AAPL",
                "quarter": 3,
                "year": 2024,
                "date": "2024-10-31",
                "content": "Great third quarter results.",
            }
        ]

        empty_resp = _make_mock_response([])
        fallback_resp = _make_mock_response(fallback_data)

        with (
            patch("src.data.transcript_data.get_config", return_value=_mock_config()),
            patch(
                "src.data.transcript_data.requests.get",
                side_effect=[empty_resp, fallback_resp],
            ),
        ):
            result = get_earnings_transcript.invoke({"ticker": "AAPL"})

        assert not result.startswith("TOOL_ERROR")
        assert "=== AAPL Earnings Call Transcript" in result
        assert "Great third quarter" in result

    def test_ticker_uppercased(self):
        """Lowercase ticker input must be normalised to uppercase in API URL."""
        from src.data.transcript_data import get_earnings_transcript

        with (
            patch("src.data.transcript_data.get_config", return_value=_mock_config()),
            patch(
                "src.data.transcript_data.requests.get",
                return_value=_make_mock_response(SAMPLE_TRANSCRIPT),
            ) as mock_get,
        ):
            get_earnings_transcript.invoke({"ticker": "aapl"})

        called_url: str = mock_get.call_args[0][0]
        assert "/AAPL?" in called_url
        assert "/aapl?" not in called_url


class TestGetEarningsTranscriptForQuarter:
    def test_specific_quarter_success(self):
        """Invoking with explicit year/quarter returns a formatted transcript."""
        from src.data.transcript_data import get_earnings_transcript_for_quarter

        msft_data = [
            {
                "symbol": "MSFT",
                "quarter": 3,
                "year": 2024,
                "date": "2024-04-25",
                "content": "Microsoft had a strong third quarter.",
            }
        ]

        with (
            patch("src.data.transcript_data.get_config", return_value=_mock_config()),
            patch(
                "src.data.transcript_data.requests.get",
                return_value=_make_mock_response(msft_data),
            ),
        ):
            result = get_earnings_transcript_for_quarter.invoke(
                {"ticker": "MSFT", "year": 2024, "quarter": 3}
            )

        assert "=== MSFT Earnings Call Transcript (Q3 2024) ===" in result
        assert "Microsoft had a strong third quarter." in result

    def test_specific_quarter_not_found(self):
        """API returning [] for explicit quarter must return TOOL_ERROR."""
        from src.data.transcript_data import get_earnings_transcript_for_quarter

        with (
            patch("src.data.transcript_data.get_config", return_value=_mock_config()),
            patch(
                "src.data.transcript_data.requests.get",
                return_value=_make_mock_response([]),
            ),
        ):
            result = get_earnings_transcript_for_quarter.invoke(
                {"ticker": "MSFT", "year": 2024, "quarter": 3}
            )

        assert result.startswith("TOOL_ERROR")
        assert "No transcript found" in result
