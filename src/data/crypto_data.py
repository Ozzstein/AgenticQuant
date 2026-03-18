"""Crypto data tools: price, history, fundamentals, fear/greed, and on-chain metrics."""

from __future__ import annotations

import requests
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import get_logger

logger = get_logger(__name__)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Map common symbols to CoinGecko IDs
SYMBOL_MAP: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
    "LINK": "chainlink",
    "UNI": "uniswap",
}


def _normalize_symbol(ticker: str) -> str:
    """Normalize a crypto ticker to CoinGecko ID.

    Handles formats: "BTC", "BTC/USDT", "BTCUSDT" -> "bitcoin"

    Args:
        ticker: Raw ticker string in any common format.

    Returns:
        CoinGecko coin ID (e.g. "bitcoin", "ethereum").
    """
    # Strip /USDT, USDT suffix, etc.
    clean = ticker.upper().replace("/USDT", "").replace("USDT", "").replace("/USD", "").strip()
    return SYMBOL_MAP.get(clean, clean.lower())


def _fmt_large(value: float | None) -> str:
    """Format a large number with T/B/M suffix.

    Args:
        value: Numeric value to format, or None.

    Returns:
        Formatted string like "1.87T", "45.2B", "N/A".
    """
    if value is None:
        return "N/A"
    if value >= 1e12:
        return f"{value / 1e12:.2f}T"
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    return f"{value:,.0f}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _fetch_simple_price(coin_id: str) -> dict:
    """Fetch simple price data from CoinGecko with retry.

    Args:
        coin_id: CoinGecko coin ID.

    Returns:
        Parsed JSON response dict.
    """
    url = (
        f"{COINGECKO_BASE}/simple/price"
        f"?ids={coin_id}&vs_currencies=usd"
        "&include_market_cap=true&include_24hr_vol=true&include_24hr_change=true"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _fetch_market_chart(coin_id: str, days: int) -> dict:
    """Fetch market chart history from CoinGecko with retry.

    Args:
        coin_id: CoinGecko coin ID.
        days: Number of days of history to retrieve.

    Returns:
        Parsed JSON response dict containing a "prices" list.
    """
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _fetch_coin_detail(coin_id: str, params: dict | None = None) -> dict:
    """Fetch full coin detail from CoinGecko with retry.

    Args:
        coin_id: CoinGecko coin ID.
        params: Optional query parameters to include.

    Returns:
        Parsed JSON response dict.
    """
    url = f"{COINGECKO_BASE}/coins/{coin_id}"
    resp = requests.get(url, params=params or {}, timeout=10)
    resp.raise_for_status()
    return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _fetch_fear_greed(limit: int) -> dict:
    """Fetch Fear & Greed Index data from alternative.me with retry.

    Args:
        limit: Number of data points to retrieve.

    Returns:
        Parsed JSON response dict.
    """
    url = f"https://api.alternative.me/fng/?limit={limit}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


@tool
def get_crypto_price(ticker: str) -> str:
    """Get current price, 24h change, market cap, and volume for a cryptocurrency.

    Args:
        ticker: Crypto ticker in any common format (e.g. "BTC", "BTC/USDT", "ETH").

    Returns:
        Formatted string with price, 24h change, market cap, and volume,
        or "TOOL_ERROR: ..." on failure.
    """
    try:
        coin_id = _normalize_symbol(ticker)
        data = _fetch_simple_price(coin_id)

        if coin_id not in data:
            return f"TOOL_ERROR: No data returned for {ticker} (id={coin_id})"

        entry = data[coin_id]
        price = entry.get("usd")
        change_24h = entry.get("usd_24h_change")
        market_cap = entry.get("usd_market_cap")
        volume_24h = entry.get("usd_24h_vol")

        if price is None:
            return f"TOOL_ERROR: No price data for {ticker}"

        symbol = ticker.upper().replace("/USDT", "").replace("USDT", "").replace("/USD", "")
        change_str = f"{change_24h:+.2f}%" if change_24h is not None else "N/A"
        mcap_str = _fmt_large(market_cap)
        vol_str = _fmt_large(volume_24h)

        return (
            f"{symbol}: ${price:,.2f} | "
            f"24h: {change_str} | "
            f"MCap: ${mcap_str} | "
            f"Vol: ${vol_str}"
        )
    except Exception as e:
        logger.warning(f"get_crypto_price failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_crypto_history(ticker: str, days: int = 30) -> str:
    """Get price history summary for a cryptocurrency including period return, volatility, and recent prices.

    Args:
        ticker: Crypto ticker in any common format (e.g. "BTC", "ETH/USDT").
        days: Number of days of history to retrieve (default 30).

    Returns:
        Formatted string with period return, volatility, high/low, and recent closes,
        or "TOOL_ERROR: ..." on failure.
    """
    try:
        coin_id = _normalize_symbol(ticker)
        data = _fetch_market_chart(coin_id, days)

        prices_raw = data.get("prices", [])
        if not prices_raw or len(prices_raw) < 2:
            return f"TOOL_ERROR: Insufficient price history for {ticker}"

        # Extract closing prices (last value per day when multiple per day)
        prices = [p[1] for p in prices_raw]

        period_return = (prices[-1] / prices[0] - 1) * 100
        high = max(prices)
        low = min(prices)

        # Daily returns for volatility
        daily_returns = [
            (prices[i] / prices[i - 1] - 1) for i in range(1, len(prices))
        ]
        n = len(daily_returns)
        mean_r = sum(daily_returns) / n
        variance = sum((r - mean_r) ** 2 for r in daily_returns) / max(n - 1, 1)
        daily_vol = variance ** 0.5
        ann_vol = daily_vol * (365 ** 0.5) * 100  # annualize with 365 for crypto

        # Last 5 data points
        recent_5 = prices_raw[-5:]

        symbol = ticker.upper().replace("/USDT", "").replace("USDT", "").replace("/USD", "")
        lines = [
            f"=== {symbol} History ({days}d) ===",
            f"Latest Price:   ${prices[-1]:,.2f}",
            f"Period Return:  {period_return:+.2f}%",
            f"Ann. Volatility: {ann_vol:.1f}%",
            f"Period High:    ${high:,.2f}",
            f"Period Low:     ${low:,.2f}",
            "Recent 5 Prices:",
        ]
        for ts_ms, price in recent_5:
            # Convert ms timestamp to date
            from datetime import datetime, timezone

            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            lines.append(f"  {dt.strftime('%Y-%m-%d')}: ${price:,.2f}")

        return "\n".join(lines)[:2500]
    except Exception as e:
        logger.warning(f"get_crypto_history failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_crypto_fundamentals(ticker: str) -> str:
    """Get fundamental data for a cryptocurrency including market cap rank, circulating supply, max supply, developer activity, and community metrics.

    Args:
        ticker: Crypto ticker in any common format (e.g. "BTC", "ETH").

    Returns:
        Formatted string with fundamental metrics,
        or "TOOL_ERROR: ..." on failure.
    """
    try:
        coin_id = _normalize_symbol(ticker)
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
        }
        data = _fetch_coin_detail(coin_id, params)

        name = data.get("name", coin_id)
        symbol_out = data.get("symbol", "").upper()
        rank = data.get("market_cap_rank", "N/A")

        mkt = data.get("market_data", {})
        circulating = mkt.get("circulating_supply")
        max_supply = mkt.get("max_supply")
        total_supply = mkt.get("total_supply")
        market_cap = (mkt.get("market_cap") or {}).get("usd")
        current_price = (mkt.get("current_price") or {}).get("usd")
        ath = (mkt.get("ath") or {}).get("usd")
        ath_change_pct = (mkt.get("ath_change_percentage") or {}).get("usd")

        dev = data.get("developer_data", {})
        commits_4w = dev.get("commit_count_4_weeks", 0)
        stars = dev.get("stars", 0)
        forks = dev.get("forks", 0)

        community = data.get("community_data", {})
        reddit_subs = community.get("reddit_subscribers", 0)
        twitter_followers = community.get("twitter_followers", 0)

        supply_pct = (
            f"{(circulating / max_supply * 100):.1f}%"
            if circulating and max_supply and max_supply > 0
            else "N/A"
        )

        lines = [
            f"=== {name} ({symbol_out}) Fundamentals ===",
            f"Market Cap Rank:      #{rank}",
            f"Current Price:        ${current_price:,.2f}" if current_price else "Current Price:        N/A",
            f"Market Cap:           ${_fmt_large(market_cap)}",
            "",
            "--- Supply ---",
            f"Circulating Supply:   {_fmt_large(circulating)}",
            f"Max Supply:           {_fmt_large(max_supply)}",
            f"Total Supply:         {_fmt_large(total_supply)}",
            f"% of Max Circulating: {supply_pct}",
            "",
            "--- Price History ---",
            f"All-Time High:        ${ath:,.2f}" if ath else "All-Time High:        N/A",
            f"% from ATH:           {ath_change_pct:.1f}%" if ath_change_pct is not None else "% from ATH:           N/A",
            "",
            "--- Developer Activity ---",
            f"Commits (4 weeks):    {commits_4w}",
            f"GitHub Stars:         {stars:,}" if stars else "GitHub Stars:         N/A",
            f"GitHub Forks:         {forks:,}" if forks else "GitHub Forks:         N/A",
            "",
            "--- Community ---",
            f"Reddit Subscribers:   {reddit_subs:,}" if reddit_subs else "Reddit Subscribers:   N/A",
            f"Twitter Followers:    {twitter_followers:,}" if twitter_followers else "Twitter Followers:    N/A",
        ]

        return "\n".join(lines)[:2500]
    except Exception as e:
        logger.warning(f"get_crypto_fundamentals failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_crypto_fear_greed(days: int = 1) -> str:
    """Get the Crypto Fear & Greed Index from alternative.me.

    Args:
        days: Number of days of historical index values to retrieve (default 1).

    Returns:
        Formatted string with Fear & Greed Index values and interpretations,
        or "TOOL_ERROR: ..." on failure.
    """
    try:
        data = _fetch_fear_greed(days)
        entries = data.get("data", [])

        if not entries:
            return "TOOL_ERROR: No Fear & Greed data returned"

        def _interpret(value: int) -> str:
            if value <= 25:
                return "Extreme Fear — potential buying opportunity"
            if value <= 45:
                return "Fear — market sentiment is negative"
            if value <= 55:
                return "Neutral — balanced sentiment"
            if value <= 75:
                return "Greed — bullish sentiment"
            return "Extreme Greed — potential market top risk"

        lines = ["=== Crypto Fear & Greed Index ==="]

        if days == 1 or len(entries) == 1:
            entry = entries[0]
            value = int(entry["value"])
            label = entry.get("value_classification", "")
            timestamp = entry.get("timestamp", "")
            lines += [
                f"Value:          {value} / 100",
                f"Classification: {label}",
                f"Interpretation: {_interpret(value)}",
            ]
            if timestamp:
                lines.append(f"Updated:        {timestamp}")
        else:
            lines.append(f"{'Date':<12} {'Value':>6}  Classification")
            lines.append("-" * 40)
            for entry in entries[:days]:
                value = int(entry["value"])
                label = entry.get("value_classification", "")
                timestamp = entry.get("timestamp", "")
                lines.append(f"{timestamp:<12} {value:>6}  {label}")

        return "\n".join(lines)[:2500]
    except Exception as e:
        logger.warning(f"get_crypto_fear_greed failed: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_crypto_on_chain(ticker: str) -> str:
    """Get on-chain metrics for a cryptocurrency using available CoinGecko data as proxy.

    Note: Full on-chain data (UTXO set, active addresses, mempool depth) requires a
    premium provider such as Glassnode or CryptoQuant. This tool returns the best
    available proxies via CoinGecko's free API.

    Args:
        ticker: Crypto ticker in any common format (e.g. "BTC", "ETH").

    Returns:
        Formatted string with available on-chain proxy metrics,
        or "TOOL_ERROR: ..." on failure.
    """
    try:
        coin_id = _normalize_symbol(ticker)
        # Use a minimal set of fields to reduce payload size
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false",
        }
        data = _fetch_coin_detail(coin_id, params)

        name = data.get("name", coin_id)
        symbol_out = data.get("symbol", "").upper()

        mkt = data.get("market_data", {})
        volume_usd = (mkt.get("total_volume") or {}).get("usd")
        market_cap = (mkt.get("market_cap") or {}).get("usd")
        circulating = mkt.get("circulating_supply")
        total_supply = mkt.get("total_supply")
        price_change_7d = mkt.get("price_change_percentage_7d")
        price_change_30d = mkt.get("price_change_percentage_30d")

        dev = data.get("developer_data", {})
        commits_4w = dev.get("commit_count_4_weeks", 0)
        code_additions_4w = dev.get("code_additions_deletions_4_weeks", {})
        additions = (code_additions_4w or {}).get("additions", 0)
        deletions = (code_additions_4w or {}).get("deletions", 0)

        community = data.get("community_data", {})
        reddit_active = community.get("reddit_accounts_active_48h", 0)

        # Blockchain explorer links
        blockchain_sites = data.get("links", {}).get("blockchain_site", [])
        explorer_links = [s for s in blockchain_sites if s][:2]

        # Volume/MCap ratio as liquidity proxy
        vol_mcap = (volume_usd / market_cap * 100) if volume_usd and market_cap else None
        supply_inflation = (
            (circulating / total_supply * 100) if circulating and total_supply and total_supply > 0 else None
        )

        lines = [
            f"=== {name} ({symbol_out}) On-Chain Metrics (CoinGecko Proxy) ===",
            "",
            "--- Network Activity Proxies ---",
            f"24h Trading Volume:   ${_fmt_large(volume_usd)}",
            f"Vol / Market Cap:     {vol_mcap:.2f}% (liquidity proxy)" if vol_mcap is not None else "Vol / Market Cap:     N/A",
            f"Reddit Active (48h):  {reddit_active:,}" if reddit_active else "Reddit Active (48h):  N/A",
            "",
            "--- Supply Metrics ---",
            f"Circulating Supply:   {_fmt_large(circulating)}",
            f"Total Supply:         {_fmt_large(total_supply)}",
            f"Emission Progress:    {supply_inflation:.1f}%" if supply_inflation is not None else "Emission Progress:    N/A",
            "",
            "--- Developer Activity (On-Chain Health Proxy) ---",
            f"Commits (4 weeks):    {commits_4w}",
            f"Code Additions (4w):  {additions}",
            f"Code Deletions (4w):  {deletions}",
            "",
            "--- Price Trend ---",
            f"7D Change:            {price_change_7d:+.2f}%" if price_change_7d is not None else "7D Change:            N/A",
            f"30D Change:           {price_change_30d:+.2f}%" if price_change_30d is not None else "30D Change:           N/A",
        ]

        if explorer_links:
            lines += ["", "--- Block Explorers ---"]
            lines += [f"  {link}" for link in explorer_links]

        lines += [
            "",
            "NOTE: Full on-chain metrics (active addresses, UTXO set, mempool) require",
            "a premium provider such as Glassnode or CryptoQuant.",
        ]

        return "\n".join(lines)[:2500]
    except Exception as e:
        logger.warning(f"get_crypto_on_chain failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"
