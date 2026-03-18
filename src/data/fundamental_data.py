"""Fundamental data tools: financials, insider trades, analyst ratings, SEC filings."""

from __future__ import annotations

from langchain_core.tools import tool

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _fmt_num(val, prefix="$", suffix="", scale=1, decimals=2) -> str:
    """Format a numeric value with optional scaling."""
    if val is None:
        return "N/A"
    try:
        v = float(val) * scale
        if abs(v) >= 1e9:
            return f"{prefix}{v / 1e9:.{decimals}f}B{suffix}"
        elif abs(v) >= 1e6:
            return f"{prefix}{v / 1e6:.{decimals}f}M{suffix}"
        elif abs(v) >= 1e3:
            return f"{prefix}{v / 1e3:.{decimals}f}K{suffix}"
        else:
            return f"{prefix}{v:.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


@tool
def get_financials(ticker: str) -> str:
    """Get key financial metrics for a ticker including revenue, net income, EBITDA, EPS, P/E, P/S, debt/equity, ROE, ROIC, and FCF yield."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker.upper())
        info = t.info or {}

        def g(key, default=None):
            return info.get(key, default)

        # Income statement
        try:
            fins = t.financials
            if fins is not None and not fins.empty:
                revenue = fins.loc["Total Revenue"].iloc[0] if "Total Revenue" in fins.index else g("totalRevenue")
                net_income = fins.loc["Net Income"].iloc[0] if "Net Income" in fins.index else g("netIncomeToCommon")
                ebitda_fs = fins.loc["EBITDA"].iloc[0] if "EBITDA" in fins.index else g("ebitda")
            else:
                revenue = g("totalRevenue")
                net_income = g("netIncomeToCommon")
                ebitda_fs = g("ebitda")
        except Exception:
            revenue = g("totalRevenue")
            net_income = g("netIncomeToCommon")
            ebitda_fs = g("ebitda")

        pe = g("trailingPE")
        ps = g("priceToSalesTrailing12Months")
        de = g("debtToEquity")
        roe = g("returnOnEquity")
        roic = g("returnOnAssets")  # approximate; ROIC not directly in yf.info
        eps = g("trailingEps")
        fcf = g("freeCashflow")
        market_cap = g("marketCap")
        fcf_yield = (fcf / market_cap * 100) if fcf and market_cap else None

        lines = [
            f"=== {ticker.upper()} Financials ===",
            f"Revenue (TTM):     {_fmt_num(revenue)}",
            f"Net Income (TTM):  {_fmt_num(net_income)}",
            f"EBITDA (TTM):      {_fmt_num(ebitda_fs)}",
            f"EPS (TTM):         {_fmt_num(eps, prefix='$', decimals=2)}",
            f"P/E Ratio:         {pe:.1f}" if pe else "P/E Ratio:         N/A",
            f"P/S Ratio:         {ps:.2f}x" if ps else "P/S Ratio:         N/A",
            f"Debt/Equity:       {de:.2f}" if de else "Debt/Equity:       N/A",
            f"ROE:               {roe * 100:.1f}%" if roe else "ROE:               N/A",
            f"ROA (proxy ROIC):  {roic * 100:.1f}%" if roic else "ROA (proxy ROIC):  N/A",
            f"FCF Yield:         {fcf_yield:.2f}%" if fcf_yield else "FCF Yield:         N/A",
            f"Market Cap:        {_fmt_num(market_cap)}",
        ]

        result = "\n".join(lines)
        return result[:2500]
    except Exception as e:
        logger.warning(f"get_financials failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_insider_trades(ticker: str) -> str:
    """Get last 10 insider transactions for a ticker including date, name, shares, type, and value."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker.upper())
        txns = t.insider_transactions

        if txns is None or txns.empty:
            return f"No insider transaction data available for {ticker.upper()}"

        txns = txns.head(10)
        lines = [f"=== {ticker.upper()} Insider Transactions (Last 10) ==="]
        for _, row in txns.iterrows():
            date = str(row.get("Start Date", row.get("Date", "N/A")))[:10]
            name = row.get("Insider", row.get("Name", "N/A"))
            shares = row.get("Shares", "N/A")
            tx_type = row.get("Transaction", row.get("Type", "N/A"))
            value = row.get("Value", None)
            val_str = _fmt_num(value) if value else "N/A"
            lines.append(f"  [{date}] {name} | {tx_type} | Shares: {shares} | Value: {val_str}")

        result = "\n".join(lines)
        return result[:2500]
    except Exception as e:
        logger.warning(f"get_insider_trades failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_analyst_ratings(ticker: str) -> str:
    """Get analyst ratings summary for a ticker including buy/sell/hold counts, consensus, and price targets."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker.upper())
        info = t.info or {}

        target_mean = info.get("targetMeanPrice")
        target_high = info.get("targetHighPrice")
        target_low = info.get("targetLowPrice")
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        recommendation = info.get("recommendationKey", "N/A").upper()
        num_analysts = info.get("numberOfAnalystOpinions", "N/A")

        lines = [f"=== {ticker.upper()} Analyst Ratings ==="]
        lines.append(f"Consensus: {recommendation}")
        lines.append(f"Number of Analysts: {num_analysts}")
        if target_mean:
            upside = ((target_mean - current_price) / current_price * 100) if current_price else None
            lines.append(f"Price Target (Mean): ${target_mean:.2f}" + (f" ({upside:+.1f}% upside)" if upside is not None else ""))
        else:
            lines.append("Price Target (Mean): N/A")
        lines.append(f"Price Target (High): ${target_high:.2f}" if target_high else "Price Target (High): N/A")
        lines.append(f"Price Target (Low):  ${target_low:.2f}" if target_low else "Price Target (Low): N/A")
        lines.append(f"Current Price:       ${current_price:.2f}" if current_price else "Current Price: N/A")

        # Try recommendations_summary
        try:
            rec_sum = t.recommendations_summary
            if rec_sum is not None and not rec_sum.empty:
                lines.append("\nRecent Recommendations Summary:")
                lines.append(rec_sum.head(4).to_string(index=False))
        except Exception:
            pass

        # Try recommendations history
        try:
            recs = t.recommendations
            if recs is not None and not recs.empty:
                recent = recs.tail(10)
                lines.append("\nRecent Analyst Actions (last 10):")
                for idx, row in recent.iterrows():
                    date = str(idx)[:10] if hasattr(idx, '__str__') else "N/A"
                    firm = row.get("Firm", "N/A")
                    action = row.get("To Grade", row.get("Action", "N/A"))
                    lines.append(f"  [{date}] {firm}: {action}")
        except Exception:
            pass

        result = "\n".join(lines)
        return result[:2500]
    except Exception as e:
        logger.warning(f"get_analyst_ratings failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_sec_filings(ticker: str) -> str:
    """Get last 5 SEC filings for a ticker (10-K, 10-Q, 8-K) from EDGAR full-text search."""
    try:
        from datetime import datetime, timedelta

        import feedparser

        today = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker.upper()}%22"
            f"&dateRange=custom&startdt={start}&enddt={today}"
            f"&forms=10-K,10-Q,8-K"
        )

        feed = feedparser.parse(url)
        entries = feed.entries[:5]

        lines = [f"=== {ticker.upper()} SEC Filings (last 90 days) ==="]
        if not entries:
            # Try alternative EDGAR RSS
            alt_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker.upper()}&type=10-K,10-Q,8-K&dateb=&owner=include&count=5&search_text=&output=atom"
            feed2 = feedparser.parse(alt_url)
            entries = feed2.entries[:5]

        if entries:
            for entry in entries:
                title = entry.get("title", "N/A")
                published = entry.get("published", entry.get("updated", "N/A"))[:10]
                summary = entry.get("summary", "")[:100]
                form_type = entry.get("category", "N/A")
                if hasattr(form_type, '__iter__') and not isinstance(form_type, str):
                    form_type = form_type[0].get("term", "N/A") if form_type else "N/A"
                lines.append(f"  [{published}] {form_type} - {title}")
                if summary:
                    lines.append(f"    {summary}")
        else:
            lines.append("No recent SEC filings found.")

        result = "\n".join(lines)
        return result[:2500]
    except Exception as e:
        logger.warning(f"get_sec_filings failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_financial_statements(ticker: str) -> str:
    """Get 3-year financial statement highlights for a ticker including revenue, gross profit, operating income, net income, total assets, total debt, and operating cash flow."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker.upper())

        def _get_row(df, keys: list[str]):
            """Try multiple row keys and return first match or None."""
            if df is None or df.empty:
                return None
            for key in keys:
                if key in df.index:
                    return df.loc[key]
            return None

        fins = t.financials  # income statement (annual), cols = dates
        bal = t.balance_sheet
        cf = t.cashflow

        # We want at most 3 years (columns)
        def _cols(df):
            if df is None or df.empty:
                return []
            return list(df.columns[:3])

        year_cols = _cols(fins) or _cols(bal) or _cols(cf)
        year_labels = [str(c)[:10] for c in year_cols] if year_cols else []

        def _row_vals(df, keys: list[str]) -> list[str]:
            row = _get_row(df, keys)
            if row is None:
                return ["N/A"] * max(len(year_labels), 1)
            return [_fmt_num(row.get(c)) if hasattr(row, "get") else _fmt_num(row.iloc[i] if i < len(row) else None) for i, c in enumerate(year_cols[:3])]

        revenue_vals = _row_vals(fins, ["Total Revenue", "Revenue"])
        gross_vals = _row_vals(fins, ["Gross Profit"])
        opinc_vals = _row_vals(fins, ["Operating Income", "EBIT"])
        netinc_vals = _row_vals(fins, ["Net Income", "Net Income Common Stockholders"])
        assets_vals = _row_vals(bal, ["Total Assets"])
        debt_vals = _row_vals(bal, ["Total Debt", "Long Term Debt"])
        ocf_vals = _row_vals(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])

        col_w = 14
        header = "Metric" + " " * (22 - 6)
        for lbl in year_labels:
            header += f"{lbl:>{col_w}}"

        def _row_line(name: str, vals: list[str]) -> str:
            line = f"{name:<22}"
            for v in vals[:3]:
                line += f"{v:>{col_w}}"
            return line

        lines = [
            f"=== {ticker.upper()} Financial Statements (Annual, up to 3Y) ===",
            header,
            "-" * (22 + col_w * len(year_labels)),
            _row_line("Revenue", revenue_vals),
            _row_line("Gross Profit", gross_vals),
            _row_line("Operating Income", opinc_vals),
            _row_line("Net Income", netinc_vals),
            _row_line("Total Assets", assets_vals),
            _row_line("Total Debt", debt_vals),
            _row_line("Operating Cash Flow", ocf_vals),
        ]

        result = "\n".join(lines)
        return result[:2500]
    except Exception as e:
        logger.warning(f"get_financial_statements failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_financial_ratios(ticker: str) -> str:
    """Get key financial ratios for a ticker including P/E, P/B, P/S, EV/EBITDA, PEG, current ratio, quick ratio, and debt/equity."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker.upper()).info or {}

        def g(key):
            return info.get(key)

        pe = g("trailingPE")
        pb = g("priceToBook")
        ps = g("priceToSalesTrailing12Months")
        ev_ebitda = g("enterpriseToEbitda")
        peg = g("pegRatio")
        current_ratio = g("currentRatio")
        quick_ratio = g("quickRatio")
        de = g("debtToEquity")

        def _pe_interp(val) -> str:
            if val is None:
                return "N/A"
            if val < 0:
                return "Negative earnings"
            if val < 15:
                return "Value"
            if val <= 25:
                return "Fair Value"
            return "Growth/Expensive"

        def _pb_interp(val) -> str:
            if val is None:
                return "N/A"
            if val < 1:
                return "Below book (potential value)"
            if val <= 3:
                return "Reasonable"
            return "Premium to book"

        def _ev_ebitda_interp(val) -> str:
            if val is None:
                return "N/A"
            if val < 10:
                return "Cheap"
            if val <= 20:
                return "Fair"
            return "Expensive"

        def _peg_interp(val) -> str:
            if val is None:
                return "N/A"
            if val < 1:
                return "Undervalued relative to growth"
            if val <= 2:
                return "Fairly valued"
            return "Overvalued relative to growth"

        def _cr_interp(val) -> str:
            if val is None:
                return "N/A"
            if val >= 2:
                return "Strong liquidity"
            if val >= 1:
                return "Adequate liquidity"
            return "Potential liquidity concern"

        def _de_interp(val) -> str:
            if val is None:
                return "N/A"
            if val < 50:
                return "Low leverage"
            if val <= 150:
                return "Moderate leverage"
            return "High leverage"

        def fmt(val, decimals=2, suffix="x") -> str:
            if val is None:
                return "N/A"
            return f"{val:.{decimals}f}{suffix}"

        lines = [
            f"=== {ticker.upper()} Financial Ratios ===",
            f"P/E (Trailing):   {fmt(pe, 1, 'x')} — {_pe_interp(pe)}",
            f"P/B:              {fmt(pb, 2, 'x')} — {_pb_interp(pb)}",
            f"P/S (TTM):        {fmt(ps, 2, 'x')}",
            f"EV/EBITDA:        {fmt(ev_ebitda, 1, 'x')} — {_ev_ebitda_interp(ev_ebitda)}",
            f"PEG Ratio:        {fmt(peg, 2, 'x')} — {_peg_interp(peg)}",
            f"Current Ratio:    {fmt(current_ratio, 2, '')} — {_cr_interp(current_ratio)}",
            f"Quick Ratio:      {fmt(quick_ratio, 2, '')}",
            f"Debt/Equity:      {fmt(de, 1, '')} — {_de_interp(de)}",
        ]

        result = "\n".join(lines)
        return result[:2500]
    except Exception as e:
        logger.warning(f"get_financial_ratios failed for {ticker}: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"


@tool
def get_macro_context() -> str:
    """Get current macro environment snapshot including S&P 500 level and return, VIX, 10Y Treasury yield, DXY, and NYSE advance/decline breadth."""
    try:
        import yfinance as yf

        symbols = ["^GSPC", "^VIX", "^TNX", "DX-Y.NYB"]
        raw = yf.download(symbols, period="1mo", progress=False, auto_adjust=True)

        def _get_close(df, sym: str):
            if df.empty:
                return None
            if hasattr(df.columns, "levels"):
                if sym in df["Close"].columns:
                    return df["Close"][sym].dropna()
            else:
                if sym in df.columns:
                    return df[sym].dropna()
            return None

        def _current(series) -> float | None:
            if series is None or len(series) == 0:
                return None
            return float(series.iloc[-1])

        def _1m_return(series) -> float | None:
            if series is None or len(series) < 2:
                return None
            return (float(series.iloc[-1]) / float(series.iloc[0]) - 1) * 100

        sp500 = _get_close(raw, "^GSPC")
        vix = _get_close(raw, "^VIX")
        tnx = _get_close(raw, "^TNX")
        dxy = _get_close(raw, "DX-Y.NYB")

        sp500_lvl = _current(sp500)
        sp500_ret = _1m_return(sp500)
        vix_lvl = _current(vix)
        tnx_lvl = _current(tnx)
        dxy_lvl = _current(dxy)

        def fmt_lvl(val, fmt_str=".2f") -> str:
            return f"{val:{fmt_str}}" if val is not None else "N/A"

        def fmt_ret(val) -> str:
            if val is None:
                return "N/A"
            sign = "+" if val >= 0 else ""
            return f"{sign}{val:.2f}%"

        # Regime interpretation
        if vix_lvl is not None:
            if vix_lvl < 15:
                vix_regime = "Low Volatility (calm)"
            elif vix_lvl <= 25:
                vix_regime = "Normal Volatility"
            else:
                vix_regime = "High Volatility (fear elevated)"
        else:
            vix_regime = "N/A"

        if tnx_lvl is not None:
            if tnx_lvl < 2:
                tnx_interp = "Historically low — accommodative"
            elif tnx_lvl <= 4:
                tnx_interp = "Moderate — neutral to tight"
            else:
                tnx_interp = "Elevated — restrictive for equities"
        else:
            tnx_interp = "N/A"

        lines = [
            "=== Macro Context Snapshot ===",
            "",
            f"S&P 500 (^GSPC):    {fmt_lvl(sp500_lvl, ',.0f')} | 1M Return: {fmt_ret(sp500_ret)}",
            f"VIX:                {fmt_lvl(vix_lvl)} — {vix_regime}",
            f"10Y Treasury (^TNX):{fmt_lvl(tnx_lvl)} % — {tnx_interp}",
            f"DXY (USD Index):    {fmt_lvl(dxy_lvl)}",
            "",
            "Note: NYSE A/D breadth not directly available via yfinance.",
            "Use sector ETF breadth as a proxy (see get_sector_performance).",
        ]

        result = "\n".join(lines)
        return result[:2500]
    except Exception as e:
        logger.warning(f"get_macro_context failed: {e}")
        return f"TOOL_ERROR: {type(e).__name__}: {str(e)}"
