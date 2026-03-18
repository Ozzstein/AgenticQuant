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


_SECTOR_TO_ETF: dict[str, str] = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


@tool
def get_technical_indicators(ticker: str, period: str = "6mo") -> str:
    """Get technical indicators for a ticker including RSI(14), MACD(12/26/9), Bollinger Bands(20), SMA20/50/200, ATR(14), Stochastic(14/3), and OBV computed from OHLCV data."""
    try:
        import numpy as np
        import yfinance as yf

        hist = yf.Ticker(ticker.upper()).history(period=period)

        if hist.empty or len(hist) < 30:
            return f"TOOL_ERROR: Insufficient data for {ticker}"

        close = hist["Close"]
        high = hist["High"]
        low = hist["Low"]
        volume = hist["Volume"]
        price = float(close.iloc[-1])

        # RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = float((100 - 100 / (1 + rs)).iloc[-1])
        if rsi >= 70:
            rsi_interp = "Overbought"
        elif rsi <= 30:
            rsi_interp = "Oversold"
        else:
            rsi_interp = "Neutral"

        # MACD(12/26/9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_val = float(macd_line.iloc[-1])
        signal_val = float(signal_line.iloc[-1])
        macd_interp = "Bullish" if macd_val > signal_val else "Bearish"

        # Bollinger Bands(20)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = float((sma20 + 2 * std20).iloc[-1])
        bb_lower = float((sma20 - 2 * std20).iloc[-1])
        bb_mid = float(sma20.iloc[-1])
        bb_range = bb_upper - bb_lower
        bb_pos = (price - bb_lower) / bb_range if bb_range > 0 else float("nan")

        # SMAs
        sma20_val = float(sma20.iloc[-1]) if len(close) >= 20 else None
        sma50_val = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        sma200_val = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        # ATR(14)
        tr = np.maximum(
            high - low,
            np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))),
        )
        atr = float(tr.rolling(14).mean().iloc[-1])

        # Stochastic(14,3)
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        h_l = high14 - low14
        stoch_k = ((close - low14) / h_l.replace(0, float("nan")) * 100)
        stoch_d = float(stoch_k.rolling(3).mean().iloc[-1])
        stoch_k_val = float(stoch_k.iloc[-1])

        # OBV
        price_change = close.diff()
        direction = price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = float((direction * volume).cumsum().iloc[-1])
        obv_str = f"{obv / 1_000_000:.1f}M" if abs(obv) >= 1_000_000 else str(int(obv))

        lines = [
            f"=== {ticker.upper()} Technical Indicators ({period}) ===",
            f"Current Price: ${price:.2f}",
            "",
            f"RSI(14):          {rsi:.1f} → {rsi_interp}",
            f"MACD(12/26/9):    {macd_val:.3f} | Signal: {signal_val:.3f} → {macd_interp}",
            f"BB(20):           Upper: ${bb_upper:.2f} | Mid: ${bb_mid:.2f} | Lower: ${bb_lower:.2f}",
            f"BB Position:      {bb_pos:.1%}" if not (bb_pos != bb_pos) else "BB Position:      N/A",
            "",
            f"SMA20:  ${sma20_val:.2f}" if sma20_val else "SMA20:  N/A",
            f"SMA50:  ${sma50_val:.2f}" if sma50_val else "SMA50:  N/A",
            f"SMA200: ${sma200_val:.2f}" if sma200_val else "SMA200: N/A",
            "",
            f"ATR(14):          ${atr:.2f}",
            f"Stochastic %K:    {stoch_k_val:.1f}",
            f"Stochastic %D:    {stoch_d:.1f}",
            f"OBV:              {obv_str}",
        ]

        result = "\n".join(lines)
        return result[:2500]
    except Exception as e:
        logger.warning(f"get_technical_indicators failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_support_resistance(ticker: str, period: str = "6mo") -> str:
    """Get key support and resistance levels for a ticker using pivot points and volume-weighted price levels."""
    try:
        import yfinance as yf

        hist = yf.Ticker(ticker.upper()).history(period=period)

        if hist.empty or len(hist) < 5:
            return f"TOOL_ERROR: Insufficient data for {ticker}"

        close = hist["Close"]
        high = hist["High"]
        low = hist["Low"]
        volume = hist["Volume"]

        # Pivot points from last trading day
        last_high = float(high.iloc[-1])
        last_low = float(low.iloc[-1])
        last_close = float(close.iloc[-1])
        current_price = last_close

        pivot = (last_high + last_low + last_close) / 3
        r1 = 2 * pivot - last_low
        r2 = pivot + (last_high - last_low)
        s1 = 2 * pivot - last_high
        s2 = pivot - (last_high - last_low)

        # Volume-weighted price levels: bin prices into 5% buckets
        price_min = float(close.min())
        price_max = float(close.max())
        bin_size = (price_max - price_min) * 0.05
        if bin_size <= 0:
            bin_size = 1.0
        vol_by_bin: dict[float, float] = {}
        for price_val, vol in zip(close, volume):
            bin_idx = int((float(price_val) - price_min) / bin_size)
            bin_price = round(price_min + bin_idx * bin_size, 2)
            vol_by_bin[bin_price] = vol_by_bin.get(bin_price, 0.0) + float(vol)

        top3_levels = sorted(vol_by_bin.items(), key=lambda x: x[1], reverse=True)[:3]

        lines = [
            f"=== {ticker.upper()} Support & Resistance ({period}) ===",
            f"Current Price: ${current_price:.2f}",
            "",
            "--- Pivot Points (Daily) ---",
            f"Pivot:  ${pivot:.2f}",
            f"R1:     ${r1:.2f}" + (" ← above price" if r1 > current_price else " ← below price"),
            f"R2:     ${r2:.2f}" + (" ← above price" if r2 > current_price else " ← below price"),
            f"S1:     ${s1:.2f}" + (" ← below price" if s1 < current_price else " ← above price"),
            f"S2:     ${s2:.2f}" + (" ← below price" if s2 < current_price else " ← above price"),
            "",
            "--- High-Volume Price Levels ---",
        ]
        for level, vol in top3_levels:
            vol_m = vol / 1_000_000
            position = "above" if level > current_price else "below"
            lines.append(f"  ${level:.2f} ({position} price) — {vol_m:.1f}M cumulative volume")

        result = "\n".join(lines)
        return result[:2500]
    except Exception as e:
        logger.warning(f"get_support_resistance failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_sector_comparison(ticker: str) -> str:
    """Get 1-month return comparison for a ticker vs its GICS sector ETF and SPY."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker.upper())
        hist = t.history(period="1mo")

        if hist.empty or len(hist) < 2:
            return f"TOOL_ERROR: Insufficient data for {ticker}"

        ticker_return = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100

        # Determine sector ETF
        sector_etf = "SPY"
        sector_name = "Unknown"
        try:
            info = t.info or {}
            sector = info.get("sector", "")
            sector_etf = _SECTOR_TO_ETF.get(sector, "SPY")
            sector_name = sector or "N/A"
        except Exception:
            pass

        symbols_to_dl = list({sector_etf, "SPY"})
        raw = yf.download(symbols_to_dl, period="1mo", progress=False, auto_adjust=True)

        returns: dict[str, float] = {}
        if not raw.empty:
            if hasattr(raw.columns, "levels"):
                close = raw["Close"]
            else:
                close = raw
            for sym in symbols_to_dl:
                try:
                    if sym in close.columns:
                        col = close[sym].dropna()
                    else:
                        col = close.dropna()
                    if len(col) >= 2:
                        returns[sym] = (col.iloc[-1] / col.iloc[0] - 1) * 100
                except Exception:
                    pass

        spy_return = returns.get("SPY", float("nan"))
        sector_return = returns.get(sector_etf, float("nan"))

        def fmt_ret(r: float) -> str:
            if r != r:  # nan check
                return "N/A"
            sign = "+" if r >= 0 else ""
            return f"{sign}{r:.2f}%"

        rel_vs_spy = ticker_return - spy_return if spy_return == spy_return else float("nan")
        rel_vs_sector = ticker_return - sector_return if sector_return == sector_return else float("nan")

        lines = [
            f"=== {ticker.upper()} Sector Comparison (1 Month) ===",
            f"Sector: {sector_name} (ETF: {sector_etf})",
            "",
            f"{ticker.upper():<10} return: {fmt_ret(ticker_return)}",
            f"{sector_etf:<10} return: {fmt_ret(sector_return)}",
            f"{'SPY':<10} return: {fmt_ret(spy_return)}",
            "",
            f"vs SPY:    {fmt_ret(rel_vs_spy)} relative",
            f"vs Sector: {fmt_ret(rel_vs_sector)} relative",
        ]

        result = "\n".join(lines)
        return result[:2500]
    except Exception as e:
        logger.warning(f"get_sector_comparison failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"
