"""News data tools: company news and market news with FinnHub + RSS fallback."""

from __future__ import annotations

from datetime import datetime, timedelta

from langchain_core.tools import tool

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _sentiment_hint(text: str) -> str:
    """Simple keyword-based sentiment hint."""
    text_lower = text.lower()
    positive = ["beat", "surge", "rise", "gain", "record", "strong", "growth", "profit", "up", "rally", "buy", "upgrade"]
    negative = ["miss", "fall", "drop", "loss", "weak", "cut", "decline", "down", "sell", "downgrade", "risk", "concern"]
    pos_count = sum(1 for w in positive if w in text_lower)
    neg_count = sum(1 for w in negative if w in text_lower)
    if pos_count > neg_count:
        return "[+]"
    elif neg_count > pos_count:
        return "[-]"
    return "[~]"


@tool
def get_company_news(ticker: str) -> str:
    """Get recent company news headlines with sentiment for a ticker. Uses FinnHub if configured, else Yahoo Finance RSS."""
    try:
        config = get_config()
        today = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        articles = []

        if config.finnhub_api_key:
            try:
                import finnhub
                client = finnhub.Client(api_key=config.finnhub_api_key)
                news = client.company_news(ticker.upper(), _from=from_date, to=today)
                articles = news[:5] if news else []
                lines = [f"=== {ticker.upper()} News (FinnHub, last 30d) ==="]
                for item in articles:
                    dt = datetime.fromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d")
                    headline = item.get("headline", "")
                    sentiment = _sentiment_hint(headline + " " + item.get("summary", ""))
                    lines.append(f"{sentiment} [{dt}] {headline}")
                if not articles:
                    lines.append("No news found.")
                return "\n".join(lines)[:2500]
            except Exception as fh_err:
                logger.warning(f"FinnHub company news failed for {ticker}: {fh_err}, falling back to RSS")

        # RSS fallback
        import feedparser
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker.upper()}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        entries = feed.entries[:5]
        lines = [f"=== {ticker.upper()} News (Yahoo RSS, last 30d) ==="]
        for entry in entries:
            published = entry.get("published", "")[:10]
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            sentiment = _sentiment_hint(title + " " + summary)
            lines.append(f"{sentiment} [{published}] {title}")
        if not entries:
            lines.append("No news found.")
        return "\n".join(lines)[:2500]

    except Exception as e:
        logger.warning(f"get_company_news failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_market_news(category: str = "general") -> str:
    """Get recent market news headlines. Uses FinnHub if configured, else Yahoo Finance RSS fallback."""
    try:
        config = get_config()
        articles = []

        if config.finnhub_api_key:
            try:
                import finnhub
                client = finnhub.Client(api_key=config.finnhub_api_key)
                news = client.general_news(category, min_id=0)
                articles = news[:10] if news else []
                lines = [f"=== Market News ({category}, FinnHub) ==="]
                for item in articles:
                    dt = datetime.fromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d")
                    headline = item.get("headline", "")
                    sentiment = _sentiment_hint(headline + " " + item.get("summary", ""))
                    lines.append(f"{sentiment} [{dt}] {headline}")
                if not articles:
                    lines.append("No news found.")
                return "\n".join(lines)[:2500]
            except Exception as fh_err:
                logger.warning(f"FinnHub market news failed: {fh_err}, falling back to RSS")

        # RSS fallback
        import feedparser
        rss_map = {
            "general": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
            "forex": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=EURUSD=X&region=US&lang=en-US",
            "crypto": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US",
            "merger": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^DJI&region=US&lang=en-US",
        }
        url = rss_map.get(category, rss_map["general"])
        feed = feedparser.parse(url)
        entries = feed.entries[:10]
        lines = [f"=== Market News ({category}, Yahoo RSS) ==="]
        for entry in entries:
            published = entry.get("published", "")[:10]
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            sentiment = _sentiment_hint(title + " " + summary)
            lines.append(f"{sentiment} [{published}] {title}")
        if not entries:
            lines.append("No news found.")
        return "\n".join(lines)[:2500]

    except Exception as e:
        logger.warning(f"get_market_news failed: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"
