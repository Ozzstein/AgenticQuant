"""Earnings call transcript data tools using Financial Modeling Prep (FMP) API."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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
    if quarter not in (1, 2, 3, 4):
        raise ValueError("quarter must be 1..4")
    url = "https://financialmodelingprep.com/stable/earning-call-transcript"
    resp = requests.get(url, params={"symbol": ticker, "quarter": quarter, "year": year, "apikey": api_key}, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return None

    item = data[0]
    content = item.get("content", "")
    if not content:
        return None

    # Capture the complete response before constructing a bounded legacy tool
    # excerpt. This endpoint remains current-only; the typed adapter accepts an
    # explicit RunContext and canonical Instrument for historical workflows.
    from src.data.adapters.base import store_observation
    from src.data.evidence.store import EvidenceStore
    from src.domain.common import RunContext
    captured = datetime.now(timezone.utc)
    context = RunContext(run_id=f"transcript-capture:{ticker}:{captured.isoformat()}", mode="research",
                         as_of=captured, config_hash="0"*64, code_version="legacy-transcript-capture-v1")
    store = EvidenceStore(Path(get_config().output_dir) / "evidence" / "evidence.sqlite")
    raw = resp.content
    if not isinstance(raw, bytes):
        raise ValueError("provider response must supply raw bytes")
    envelope = store_observation(store, raw=raw, normalized=item, document=content, provider="fmp",
                                 version="stable-transcript-capture-v1", source_uri=url, kind="transcript",
                                 logical_key=f"unresolved-symbol:{ticker}:{year}:Q{quarter}", context=context,
                                 ingested_at=captured)
    # No stable identity is invented from the display ticker; typed consumers
    # require the canonical instrument adapter, not this compatibility tool.
    date = item.get("date", "N/A")
    header = f"=== {ticker} Earnings Call Transcript (Q{quarter} {year}) ==="
    body = content[:2200]
    result = f"{header}\nDate: {date}\nSource: evidence:{envelope.record.evidence_id}#chars=0:{len(body)}\n\n{body}"

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
        logger.warning("get_earnings_transcript failed: {}", type(e).__name__)
        return f"TOOL_ERROR: {type(e).__name__}: provider request failed"


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
        logger.warning("get_earnings_transcript_for_quarter failed: {}", type(e).__name__)
        return f"TOOL_ERROR: {type(e).__name__}: provider request failed"
