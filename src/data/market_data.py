"""Equity market data tools: price, history, options."""

from __future__ import annotations

from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import get_logger

logger = get_logger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _yf_ticker_info(ticker: str):
    import yfinance as yf
    t = yf.Ticker(ticker)
    return t.fast_info, t.info


@tool
def get_stock_price(ticker: str) -> str:
    """Get current stock price, volume, day high/low, and percent change for a ticker."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker.upper())
        fi = t.fast_info

        price = fi.last_price
        volume = fi.three_month_average_volume or fi.last_volume
        day_high = fi.day_high
        day_low = fi.day_low
        prev_close = fi.previous_close

        if price is None:
            return f"TOOL_ERROR: No price data available for {ticker}"

        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
        change_sign = "+" if change_pct >= 0 else ""
        vol_str = f"{volume / 1_000_000:.1f}M" if volume and volume >= 1_000_000 else str(volume)

        return (
            f"{ticker.upper()}: ${price:.2f} | "
            f"Volume: {vol_str} | "
            f"Day: ${day_low:.2f} - ${day_high:.2f} | "
            f"Change: {change_sign}{change_pct:.2f}%"
        )
    except Exception as e:
        logger.warning(f"get_stock_price failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_stock_history(ticker: str, period: str = "6mo") -> str:
    """Get OHLCV history summary including recent prices, returns, volatility, and average volume for a ticker."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker.upper())
        hist = t.history(period=period)

        if hist.empty:
            return f"TOOL_ERROR: No historical data available for {ticker}"

        hist = hist.dropna(subset=["Close"])
        close = hist["Close"]
        volume = hist["Volume"]

        total_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
        daily_returns = close.pct_change().dropna()
        volatility_ann = daily_returns.std() * (252 ** 0.5) * 100
        avg_vol = volume.mean()
        recent_5 = close.tail(5)

        lines = [
            f"=== {ticker.upper()} History ({period}) ===",
            f"Period: {hist.index[0].date()} to {hist.index[-1].date()}",
            f"Latest Close: ${close.iloc[-1]:.2f}",
            f"Period Return: {total_return:+.2f}%",
            f"Annualized Volatility: {volatility_ann:.1f}%",
            f"Avg Daily Volume: {avg_vol / 1_000_000:.1f}M",
            f"52W High: ${close.max():.2f} | 52W Low: ${close.min():.2f}",
            "Recent 5 Closes:",
        ]
        for date, price in recent_5.items():
            lines.append(f"  {date.date()}: ${price:.2f}")

        result = "\n".join(lines)
        return result[:2500]
    except Exception as e:
        logger.warning(f"get_stock_history failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_options_data(ticker: str) -> str:
    """Get near-dated options chain summary including put/call ratio and implied volatility for a ticker."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker.upper())
        expirations = t.options

        if not expirations:
            return f"TOOL_ERROR: No options data available for {ticker}"

        nearest_exp = expirations[0]
        chain = t.option_chain(nearest_exp)
        calls = chain.calls
        puts = chain.puts

        total_call_volume = calls["volume"].sum() if "volume" in calls.columns else 0
        total_put_volume = puts["volume"].sum() if "volume" in puts.columns else 0
        pc_ratio = (total_put_volume / total_call_volume) if total_call_volume > 0 else float("nan")

        avg_call_iv = calls["impliedVolatility"].mean() * 100 if "impliedVolatility" in calls.columns else None
        avg_put_iv = puts["impliedVolatility"].mean() * 100 if "impliedVolatility" in puts.columns else None

        # Top 5 calls by open interest
        top_calls = calls.nlargest(5, "openInterest")[["strike", "lastPrice", "impliedVolatility", "openInterest"]] if "openInterest" in calls.columns else calls.head(5)
        top_puts = puts.nlargest(5, "openInterest")[["strike", "lastPrice", "impliedVolatility", "openInterest"]] if "openInterest" in puts.columns else puts.head(5)

        lines = [
            f"=== {ticker.upper()} Options (Exp: {nearest_exp}) ===",
            f"Put/Call Volume Ratio: {pc_ratio:.2f}" if not (pc_ratio != pc_ratio) else "Put/Call Volume Ratio: N/A",
            f"Avg Call IV: {avg_call_iv:.1f}%" if avg_call_iv is not None else "Avg Call IV: N/A",
            f"Avg Put IV:  {avg_put_iv:.1f}%" if avg_put_iv is not None else "Avg Put IV: N/A",
            "",
            "Top 5 Calls (by OI):",
            top_calls.to_string(index=False),
            "",
            "Top 5 Puts (by OI):",
            top_puts.to_string(index=False),
        ]

        result = "\n".join(lines)
        return result[:2500]
    except Exception as e:
        logger.warning(f"get_options_data failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"
