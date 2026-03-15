"""Alternative data tools: Fear & Greed index, VIX, and sector performance."""

from __future__ import annotations

from langchain_core.tools import tool

from src.utils.logger import get_logger

logger = get_logger(__name__)


@tool
def get_fear_greed_index() -> str:
    """Get the current CNN Fear & Greed Index value (0-100) and market sentiment label."""
    try:
        import requests

        url = "https://api.alternative.me/fng/?limit=1"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        entry = data["data"][0]
        value = int(entry["value"])
        label = entry["value_classification"]
        timestamp = entry.get("timestamp", "")

        if value <= 25:
            interpretation = "Extreme Fear — potential buying opportunity"
        elif value <= 45:
            interpretation = "Fear — market sentiment is negative"
        elif value <= 55:
            interpretation = "Neutral — balanced sentiment"
        elif value <= 75:
            interpretation = "Greed — bullish sentiment"
        else:
            interpretation = "Extreme Greed — potential market top risk"

        lines = [
            "=== Fear & Greed Index ===",
            f"Value:          {value} / 100",
            f"Classification: {label}",
            f"Interpretation: {interpretation}",
        ]
        if timestamp:
            lines.append(f"Updated:        {timestamp}")

        return "\n".join(lines)[:2500]
    except Exception as e:
        logger.warning(f"get_fear_greed_index failed: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_vix() -> str:
    """Get the current VIX level, 5-day range, and market volatility interpretation."""
    try:
        import yfinance as yf

        vix = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)

        if vix.empty:
            return "TOOL_ERROR: No VIX data available"

        # Handle MultiIndex columns (yfinance may return them)
        if hasattr(vix.columns, "levels"):
            vix.columns = vix.columns.get_level_values(0)

        current = float(vix["Close"].iloc[-1])
        high_5d = float(vix["High"].max())
        low_5d = float(vix["Low"].min())

        if current < 15:
            regime = "Low Volatility — calm market conditions"
        elif current <= 25:
            regime = "Normal Volatility — typical market conditions"
        else:
            regime = "High Volatility — elevated fear and uncertainty"

        lines = [
            "=== VIX (Volatility Index) ===",
            f"Current VIX:  {current:.2f}",
            f"5D Range:     {low_5d:.2f} - {high_5d:.2f}",
            f"Regime:       {regime}",
            "",
            "Reference: <15 Low | 15-25 Normal | >25 High",
        ]

        return "\n".join(lines)[:2500]
    except Exception as e:
        logger.warning(f"get_vix failed: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Svcs",
}


@tool
def get_sector_performance() -> str:
    """Get 1-month sector ETF performance for the 11 SPDR sector ETFs."""
    try:
        import yfinance as yf

        symbols = list(SECTOR_ETFS.keys())
        raw = yf.download(symbols, period="1mo", progress=False, auto_adjust=True)

        if raw.empty:
            return "TOOL_ERROR: No sector data available"

        # Handle MultiIndex columns
        if hasattr(raw.columns, "levels"):
            close = raw["Close"]
        else:
            close = raw

        results = {}
        for sym in symbols:
            try:
                if sym in close.columns:
                    col = close[sym].dropna()
                    if len(col) >= 2:
                        ret = (col.iloc[-1] / col.iloc[0] - 1) * 100
                        results[sym] = ret
            except Exception:
                pass

        # Sort by return descending
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        lines = ["=== Sector Performance (1 Month) ==="]
        lines.append(f"{'ETF':<6} {'Sector':<26} {'1M Return':>10}")
        lines.append("-" * 45)
        for sym, ret in sorted_results:
            name = SECTOR_ETFS.get(sym, sym)
            sign = "+" if ret >= 0 else ""
            lines.append(f"{sym:<6} {name:<26} {sign}{ret:.2f}%")

        return "\n".join(lines)[:2500]
    except Exception as e:
        logger.warning(f"get_sector_performance failed: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"
