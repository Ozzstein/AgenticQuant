"""Earnings call transcript data tools using Financial Modeling Prep (FMP) API."""

from __future__ import annotations

from datetime import datetime

import requests
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _most_recent_quarter() -> tuple[int, int]:
    """Return (year, quarter) for the most recently completed fiscal quarter.

    Uses datetime.now(). Quarter 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec.
    If today is in Q1 (Jan-Mar), the most recently completed quarter is Q4 of the
    previous year.

    Returns:
        Tuple of (year, quarter) for the most recently completed fiscal quarter.
    """
    now = datetime.now()
    current_quarter = (now.month - 1) // 3 + 1

    if current_quarter == 1:
        return now.year - 1, 4
    return now.year, current_quarter - 1


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _fetch_transcript(ticker: str, year: int, quarter: int, api_key: str) -> str | None:
    """Fetch transcript from FMP API. Returns formatted string or None if not found.

    Args:
        ticker: Stock ticker symbol (already uppercased).
        year: Fiscal year (e.g., 2025).
        quarter: Quarter number (1-4).
        api_key: FMP API key.

    Returns:
        Formatted transcript string or None if not available.
    """
    url = (
        f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}"
        f"?quarter={quarter}&year={year}&apikey={api_key}"
    )

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return None

    item = data[0]
    content = item.get("content", "")
    if not content:
        return None

    date = item.get("date", "N/A")
    header = f"=== {ticker} Earnings Call Transcript (Q{quarter} {year}) ==="
    body = content[:2200]
    result = f"{header}\nDate: {date}\n\n{body}"

    if len(content) > 2200:
        result += f"\n\n[Transcript truncated — {len(content)} chars total]"

    return result


@tool
def get_earnings_transcript(ticker: str) -> str:
    """Get the most recent earnings call transcript for a ticker from Financial Modeling Prep.

    Fetches the transcript for the most recently completed quarter. Falls back to the prior
    quarter if the latest is not yet available (transcripts often lag 2-3 weeks).

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").

    Returns:
        Formatted transcript excerpt or TOOL_ERROR on failure.
    """
    try:
        config = get_config()
        api_key = config.fmp_api_key
        if not api_key:
            return "TOOL_ERROR: FMP API key not configured. Set AIQUANT_FMP_API_KEY."

        symbol = ticker.upper()
        year, quarter = _most_recent_quarter()

        result = _fetch_transcript(symbol, year, quarter, api_key)

        if result is None:
            prior_year = year - 1 if quarter == 1 else year
            prior_quarter = 4 if quarter == 1 else quarter - 1
            result = _fetch_transcript(symbol, prior_year, prior_quarter, api_key)

            if result is None:
                return (
                    f"TOOL_ERROR: No transcript found for {ticker} "
                    f"(tried Q{quarter} {year} and prior quarter)"
                )

        return result[:2500]

    except Exception as e:
        logger.warning(f"get_earnings_transcript failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)[:200]}"


@tool
def get_earnings_transcript_for_quarter(ticker: str, year: int, quarter: int) -> str:
    """Get the earnings call transcript for a specific quarter from Financial Modeling Prep.

    Args:
        ticker: Stock ticker symbol.
        year: Fiscal year (e.g., 2025).
        quarter: Quarter number (1-4).

    Returns:
        Formatted transcript excerpt or TOOL_ERROR on failure.
    """
    try:
        config = get_config()
        api_key = config.fmp_api_key
        if not api_key:
            return "TOOL_ERROR: FMP API key not configured. Set AIQUANT_FMP_API_KEY."

        symbol = ticker.upper()
        result = _fetch_transcript(symbol, year, quarter, api_key)

        if result is None:
            return f"TOOL_ERROR: No transcript found for {ticker} Q{quarter} {year}"

        return result[:2500]

    except Exception as e:
        logger.warning(f"get_earnings_transcript_for_quarter failed for {ticker} Q{quarter} {year}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)[:200]}"
